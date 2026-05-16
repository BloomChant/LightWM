"""
ReAct baseline agent.

Env-agnostic: the ReAct loop (system prompt + chat history + parse Thought/Action
lines) lives here; everything environment-specific (command vocabulary, action
normalization, verb detection, fallback action, env description) is delegated
to a pluggable BaseEnvObserver.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import BadRequestError, OpenAI  # type: ignore

from LightWM.env.base import BaseEnvObserver
from LightWM.prompt.ReactAgent_prompt import get_react_prompt


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


class ReactAgent:
    """
    Standard ReAct Agent:
    - Maintains chat history: [System, (User, Assistant)*, User]
    - Prunes history to keep last K turns (history_k) if set.
    - All env-specific logic (verb detection, action normalization, fallback
      action, env description in the system prompt) routes through env_observer.
    """

    def __init__(
        self,
        model_name: str = "qwen3-8b",
        history_k: int = -1,  # -1 means unlimited
        enable_geh: bool = False,
        similarity_match: bool = True,
        align_env_rules: bool = True,
        env_observer: Optional[BaseEnvObserver] = None,
    ):
        self.model_name = model_name
        self.history_k = history_k
        self.enable_geh = enable_geh
        self.similarity_match = similarity_match
        self.align_env_rules = align_env_rules
        self.env_observer: BaseEnvObserver = env_observer or BaseEnvObserver()
        self.client = self._initialize_client()
        self.chat_history: List[Dict[str, str]] = []

    def _initialize_client(self) -> OpenAI:
        api_key = os.getenv("DASH_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("DASH_BASE_URL")

        try:
            from LightWM.config import get_settings  # type: ignore

            settings = get_settings()
            api_key = api_key or getattr(settings.api, "dash_api_key", None)
            base_url = base_url or getattr(settings.api, "dash_base_url", None)
        except Exception:
            pass

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise RuntimeError(
                "Neither DASH_API_KEY nor OPENAI_API_KEY is configured."
            )

        return OpenAI(api_key=api_key, base_url=base_url)

    def reset(self, task_description: str) -> None:
        """Start a new episode: clear history, set system prompt."""
        p = get_react_prompt(
            task_description,
            env_description=self.env_observer.get_env_description(),
            align_env_rules=self.align_env_rules,
            enable_geh=self.enable_geh,
        )
        self.chat_history = [{"role": "system", "content": p.render_system()}]

    def _prune_history(self) -> List[Dict[str, str]]:
        """
        Return pruned messages list suitable for the API call.
        Preserves:
          - System message (index 0)
          - Last user message (current observation)
          - Last K pairs of (User, Assistant) interactions if history_k > 0
        """
        if self.history_k < 0:
            return self.chat_history

        if len(self.chat_history) <= 2:
            return self.chat_history

        # history structure: [System, U1, A1, U2, A2, ..., CurrentUser]
        system_msg = self.chat_history[0]
        interaction_pool = self.chat_history[1:-1]

        keep_count = self.history_k * 2
        kept_interactions = interaction_pool[-keep_count:] if keep_count > 0 else []

        current_msg = self.chat_history[-1]
        return [system_msg] + kept_interactions + [current_msg]

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Raw LLM call with stats."""
        det_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "top_p": float(os.getenv("LLM_TOP_P", "1") or 1.0),
            "presence_penalty": float(os.getenv("LLM_PRESENCE_PENALTY", "0") or 0.0),
            "frequency_penalty": float(os.getenv("LLM_FREQUENCY_PENALTY", "0") or 0.0),
            "n": 1,
        }
        seed_env = (os.getenv("LLM_SEED") or "").strip()
        if seed_env:
            try:
                det_kwargs["seed"] = int(seed_env)
            except Exception:
                pass

        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            start = time.perf_counter()
            resp = None
            last_exc: Optional[BaseException] = None
            attempts: List[Dict[str, Any]] = [
                {**det_kwargs, "extra_body": extra_body, "enable_thinking": False},
                {**det_kwargs, "extra_body": extra_body},
                {**det_kwargs},
                {"extra_body": extra_body},
                {},
            ]
            for kwargs in attempts:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **kwargs,
                    )
                    last_exc = None
                    break
                except TypeError as exc:
                    last_exc = exc
                    continue
                except BadRequestError as exc:
                    msg = str(exc)
                    if ("Unknown parameter" in msg or "unknown_parameter" in msg or "invalid_parameter" in msg):
                        last_exc = exc
                        continue
                    raise
            if resp is None:
                raise last_exc or RuntimeError("LLM call failed with unknown error")
            latency_ms = (time.perf_counter() - start) * 1000
        except Exception as e:
            fallback = self.env_observer.default_fallback_action() or ""
            print(f"Error during LLM call: {e}")
            return {
                "answer": f"Thought: Error occurred.\nAction: {fallback}",
                "input_token": 0,
                "output_token": 0,
                "latency_ms": 0.0,
                "error": str(e),
            }

        content = resp.choices[0].message.content
        usage = resp.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return {
            "answer": content,
            "input_token": input_tokens,
            "output_token": output_tokens,
            "latency_ms": latency_ms,
        }

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        Extract Thought and Action. Action must be a single line starting with
        "Action:". On parse failure, fall back to the last line that the env
        recognizes as a command, then to the observer's default_fallback_action.
        """
        thought_match = re.search(r"(?im)^Thought\s*:\s*(.+)$", response)
        thought = thought_match.group(1).strip().strip("\"'") if thought_match else ""

        action_match = re.search(r"(?im)^Action\s*:\s*(.+)$", response)
        if action_match:
            action = action_match.group(1).strip().strip("\"'")
        else:
            # Fallback: last line the env observer recognizes as a command.
            action = self.env_observer.default_fallback_action() or ""
            for line in reversed(response.splitlines()):
                candidate = line.strip().strip("\"'")
                if candidate and self.env_observer.detect_action_verb(candidate):
                    action = candidate
                    break

        # Truncate at first newline and strip in case the model smuggled extras.
        action = action.splitlines()[0].strip() if action else action
        return thought, action

    def select_legal_action(
        self,
        *,
        executor_action: str,
        admissible_commands: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        if not admissible_commands:
            return executor_action, {"reason": "no_admissible_commands", "corrected": False}

        action = (executor_action or "").strip()
        if action in admissible_commands:
            return action, {"reason": "exact_match", "corrected": False}

        lower_map = {cmd.lower(): cmd for cmd in admissible_commands if isinstance(cmd, str)}
        if action.lower() in lower_map:
            return lower_map[action.lower()], {"reason": "case_insensitive_match", "corrected": True}

        normalized = self.env_observer.normalize_action(action)
        if normalized in admissible_commands:
            return normalized, {"reason": "normalized_match", "corrected": True}

        verb = self.env_observer.detect_action_verb(action)
        try:
            from difflib import SequenceMatcher

            candidates = admissible_commands
            if verb:
                candidates = [c for c in admissible_commands if c.lower().startswith(verb)]
                if not candidates:
                    candidates = admissible_commands
            best = max(candidates, key=lambda c: SequenceMatcher(None, c.lower(), normalized.lower()).ratio())
            return best, {"reason": "similarity_match", "corrected": True, "verb": verb}
        except Exception:
            return action, {"reason": "no_match_fallback", "corrected": False, "verb": verb}

    def react_step(self, observation: str, admissible_commands: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        One ReAct step:
        1. Append observation as user message
        2. Prune history
        3. Call LLM
        4. Append response as assistant message
        5. Parse Thought/Action and legalize the action
        """
        p = get_react_prompt(
            "",
            env_description=self.env_observer.get_env_description(),
            align_env_rules=self.align_env_rules,
        )
        user_content = p.render_user_observation(observation)
        self.chat_history.append({"role": "user", "content": user_content})

        messages_to_send = self._prune_history()
        result = self.chat(messages_to_send)
        raw_response = result["answer"]

        thought, action = self._parse_response(raw_response)

        corrected_action = action
        correction_meta: Dict[str, Any] = {"corrected": False}
        if self.similarity_match and admissible_commands:
            corrected_action, correction_meta = self.select_legal_action(
                executor_action=action,
                admissible_commands=admissible_commands,
            )

        # Record the assistant's full response (Thought + Action) for history context.
        self.chat_history.append({"role": "assistant", "content": raw_response})

        return {
            "thought": thought,
            "action": corrected_action,
            "raw_action": action,
            "raw_response": raw_response,
            "action_correction": correction_meta,
            "stats": {
                "input_tokens": result["input_token"],
                "output_tokens": result["output_token"],
                "total_tokens": result["input_token"] + result["output_token"],
                "latency_ms": result["latency_ms"],
            },
            "prompt": {"messages": messages_to_send},
        }
