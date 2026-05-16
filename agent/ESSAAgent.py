"""
ESSA (Explicit State SLM-based Agent) — env-agnostic core.

Architecture:
- MacroStateInitializer  (one LLM call per episode): initialize macro task_state from schema + observation.
- StateUpdater           (one LLM call per step):    update subtask_state via patch operations.
- Executor               (one LLM call per step):    propose next action.
- SubtaskCaller          (control layer):             inject inputs into subtask_state at subtask start.
- ReturnApplier          (control layer):             sync subtask outputs back to macro task_state on done.

Environment-specific behavior (observation parsing, action normalization, command vocabulary,
evidence gating, derived macro updates) is delegated to a pluggable BaseEnvObserver. The
agent core has no direct dependency on any particular environment module.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import BadRequestError, OpenAI  # type: ignore

from LightWM.env.base import BaseEnvObserver
from LightWM.prompt.ESSA_prompts import get_executor_prompt, get_macro_init_prompt, get_state_updater_prompt


class _DefaultFormatDict(dict):
    """Dict that returns '' for missing keys — used for safe goal_template formatting."""

    def __missing__(self, key: str) -> str:
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _lower(text: str) -> str:
    return (text or "").strip().lower()


def _extract_json_block(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        return match.group(1)
    return None


def _safe_json_load(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _load_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


@dataclass
class ModelStats:
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# ESSAAgent
# ---------------------------------------------------------------------------

class ESSAAgent:
    def __init__(
        self,
        model_name: str = "qwen3-4b-instruct-2507",
        executor_mode: str = "auto",
        action_space_mode: str = "base",
        state_update_mode: str = "patch",
        env_observer: Optional[BaseEnvObserver] = None,
    ):
        self.model_name = model_name
        self.executor_mode = (executor_mode or "auto").strip().lower()
        self.action_space_mode = self._normalize_action_space_mode(action_space_mode)
        self.state_update_mode = self._normalize_state_update_mode(state_update_mode)
        # A neutral BaseEnvObserver means no env-specific hooks fire. Callers should
        # pass a concrete observer (e.g. AlfworldObserver) for useful behavior.
        self.env_observer: BaseEnvObserver = env_observer or BaseEnvObserver()
        self.client = self._initialize_client()

        self.task_specs: Dict[str, Any] = self._load_task_specs() or {}
        self.subtask_specs: Dict[str, Any] = self._load_subtask_specs() or {}
        self.task_type: str = ""
        self.task_family_spec: Dict[str, Any] = {}

        # Episode-level container
        self.task_spec: Dict[str, Any] = {}
        self.subtask_flow: List[Dict[str, Any]] = []
        self.subtask_states: Dict[int, Dict[str, Any]] = {}

        self.last_macro_init_call: Optional[Dict[str, Any]] = None
        self.last_state_update_prompt_variant: str = "patch_ops"

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _normalize_action_space_mode(self, mode: str) -> str:
        return "full" if str(mode or "base").strip().lower() == "full" else "base"

    def _normalize_state_update_mode(self, mode: str) -> str:
        return "full_state" if str(mode or "patch").strip().lower() == "full_state" else "patch"

    def _resolve_executor_base_actions(self, spec_base_actions: Any) -> List[object]:
        if self.action_space_mode == "full":
            return self.env_observer.get_full_action_space()
        if isinstance(spec_base_actions, list):
            return spec_base_actions
        return []

    def _build_neutralized_full_state_hints(self, spec: Dict[str, Any]) -> Optional[str]:
        if not isinstance(spec, dict):
            return None
        policy = spec.get("patch_ops_policy")
        allowed = policy.get("allowed") if isinstance(policy, dict) else None
        lines: List[str] = [
            "Use observation-grounded state transitions for full_state output.",
            "Do NOT output patch_ops examples in this mode; reflect effects directly in returned subtask_state.",
        ]
        if isinstance(allowed, list) and allowed:
            lines.append("Allowed update targets in this subtask:")
            for item in allowed:
                if not isinstance(item, dict):
                    continue
                op = str(item.get("op") or "").strip()
                path = str(item.get("path") or "").strip()
                if op and path:
                    lines.append(f"- {path} (effect type aligned with {op})")
        lines.append("If no precondition evidence is present, keep subtask_state unchanged.")
        return "\n".join(lines)

    def _resolve_state_updater_operation_space(self, spec: Dict[str, Any]) -> Tuple[Optional[str], str]:
        if self.state_update_mode != "full_state":
            op_space = spec.get("operation_space") if isinstance(spec, dict) else None
            if isinstance(op_space, str) and op_space.strip():
                return op_space, "patch_ops"
            return None, "patch_ops"

        full_space = spec.get("operation_space_full_state") if isinstance(spec, dict) else None
        if isinstance(full_space, str) and full_space.strip():
            return full_space, "full_state_ops"

        fallback = self._build_neutralized_full_state_hints(spec)
        if isinstance(fallback, str) and fallback.strip():
            return fallback, "fallback_neutralized"
        return None, "fallback_neutralized"

    # ------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------

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
            raise RuntimeError("No API key configured. Set DASH_API_KEY or OPENAI_API_KEY.")

        return OpenAI(api_key=api_key, base_url=base_url)

    def _call_chat(
        self, messages: List[Dict[str, str]], *, extra_body: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, ModelStats, Dict[str, Any]]:
        det_kwargs: Dict[str, Any] = {
            "temperature": 0.7,
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

        base_extra = {"chat_template_kwargs": {"enable_thinking": False}}
        if extra_body:
            base_extra.update(extra_body)

        start = _now_ms()
        resp = None
        last_exc: Optional[BaseException] = None
        attempts: List[Dict[str, Any]] = [
            {**det_kwargs, "extra_body": base_extra, "enable_thinking": False},
            {**det_kwargs, "extra_body": base_extra},
            {**det_kwargs},
            {"extra_body": base_extra},
            {},
        ]
        for kwargs in attempts:
            try:
                resp = self.client.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
                last_exc = None
                break
            except TypeError as exc:
                last_exc = exc
                continue
            except BadRequestError as exc:
                msg = str(exc)
                if "Unknown parameter" in msg or "unknown_parameter" in msg or "invalid_parameter" in msg:
                    last_exc = exc
                    continue
                raise
        if resp is None:
            raise last_exc or RuntimeError("LLM call failed with unknown error")
        latency_ms = _now_ms() - start

        content = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total = int(getattr(usage, "total_tokens", in_tok + out_tok) or (in_tok + out_tok)) if usage else (in_tok + out_tok)
        stats = ModelStats(model=self.model_name, input_tokens=in_tok, output_tokens=out_tok, total_tokens=total, latency_ms=latency_ms)
        return content, stats, {"messages": messages}

    # ------------------------------------------------------------------
    # Spec loaders
    # ------------------------------------------------------------------

    def _load_specs_with_subdirs(self, filename: str) -> Dict[str, Any]:
        """
        Load memory/ESSA/<filename> and merge any memory/ESSA/<env>/<filename>
        files on top. Per-env subdir files (e.g. memory/ESSA/scienceworld/) win
        on key collision. Keeps env specs decoupled from the shared file.
        """
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_dir = os.path.join(root, "memory", "ESSA")
        merged: Dict[str, Any] = dict(_load_json_file(os.path.join(base_dir, filename)) or {})
        if os.path.isdir(base_dir):
            for entry in sorted(os.listdir(base_dir)):
                sub = os.path.join(base_dir, entry)
                if not os.path.isdir(sub):
                    continue
                extra = _load_json_file(os.path.join(sub, filename))
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        if isinstance(k, str) and k and not k.startswith("__"):
                            merged[k] = v
        return merged

    def _load_task_specs(self) -> Optional[Dict[str, Any]]:
        return self._load_specs_with_subdirs("task_specs.json")

    def _load_subtask_specs(self) -> Optional[Dict[str, Any]]:
        return self._load_specs_with_subdirs("subtask_specs.json")

    def _load_subtask_spec(self, subtask_type: str) -> Optional[Dict[str, Any]]:
        if isinstance(self.subtask_specs, dict):
            spec = self.subtask_specs.get(subtask_type)
            if isinstance(spec, dict):
                return spec
        return None

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset(self, goal_text: str, initial_observation: str, task_type: str = "") -> None:
        self.task_type = (task_type or "").strip()
        self.task_family_spec = self.task_specs.get(self.task_type, {}) if isinstance(self.task_specs, dict) else {}

        task_state_schema = (
            (self.task_family_spec.get("task_state_schema") or self.task_family_spec.get("task_status_schema") or {})
            if isinstance(self.task_family_spec, dict) else {}
        )
        base_seq = (self.task_family_spec.get("base_subtask_sequence") or []) if isinstance(self.task_family_spec, dict) else []

        task_state = self._macro_init_task_status(
            task_type=self.task_type,
            task_state_schema=task_state_schema,
            goal_text=goal_text,
            initial_observation=initial_observation,
        )
        task_state = self.env_observer.finalize_task_state(
            task_state, initial_observation=initial_observation
        )

        self.task_spec = {
            "task_type": self.task_type,
            "task_state_schema": task_state_schema,
            "task_state": task_state,
            "goal_text": goal_text,
            "initial_observation": initial_observation,
            "base_subtask_sequence": base_seq,
        }

        self.subtask_flow = self._build_subtask_flow_from_spec(task_state=task_state, base_seq=base_seq)
        self.subtask_states = {}
        for idx, spec in enumerate(self.subtask_flow):
            self.subtask_states[idx] = self._init_subtask_state(spec)

    def _macro_init_task_status(
        self,
        *,
        task_type: str,
        task_state_schema: Dict[str, Any],
        goal_text: str,
        initial_observation: str,
    ) -> Dict[str, Any]:
        init_rules = None
        if isinstance(self.task_family_spec, dict):
            init_rules = self.task_family_spec.get("init_rules")
        prompt = get_macro_init_prompt(task_type=task_type, init_rules=init_rules, task_state_schema=task_state_schema)
        system = prompt.render_system()
        user = prompt.render_user(goal_text=goal_text, initial_observation=initial_observation)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        try:
            response, stats, prompt_snapshot = self._call_chat(messages)
            block = _extract_json_block(response or "") or (response or "")
            parsed = _safe_json_load(block) if isinstance(block, str) else None
            if isinstance(parsed, dict) and parsed:
                self.last_macro_init_call = {
                    "prompt": prompt_snapshot,
                    "raw_response": response,
                    "stats": stats.to_dict(),
                    "task_state": parsed,
                }
                return parsed
        except Exception:
            pass

        # Minimal fallback — env-specific; observer knows which fields matter.
        return self.env_observer.macro_init_fallback(
            task_type=task_type,
            task_state_schema=task_state_schema,
            goal_text=goal_text,
            initial_observation=initial_observation,
        )

    # ------------------------------------------------------------------
    # Subtask compilation
    # ------------------------------------------------------------------

    def _build_subtask_flow_from_spec(self, *, task_state: Dict[str, Any], base_seq: List[Any]) -> List[Dict[str, Any]]:
        """
        Compile the flat base_subtask_sequence into per-subtask dicts. Each
        subtask's human-readable goal string is rendered from its spec's
        goal_template against the current task_state; missing task_state fields
        fall back first to goal_template_defaults in the spec, then to "".
        """
        ts = task_state if isinstance(task_state, dict) else {}

        flow: List[Dict[str, Any]] = []
        for step in base_seq if isinstance(base_seq, list) else []:
            if not isinstance(step, dict):
                continue
            stype = str(step.get("subtask_type") or "").strip()
            if not stype:
                continue
            spec_meta = self._load_subtask_spec(stype) or {}
            goal = self._render_subtask_goal(step=step, spec_meta=spec_meta, task_state=ts)

            flow.append({
                "type": stype,
                "goal": goal,
                "constraints": {},
                "status": "pending",
                "input_para": spec_meta.get("input_para", []),
                "output_para": spec_meta.get("output_para", []),
                "base_actions": spec_meta.get("base_actions", []),
                "subtask_status_schema": spec_meta.get("subtask_status_schema", {}),
            })
        return flow

    @staticmethod
    def _render_subtask_goal(*, step: Dict[str, Any], spec_meta: Dict[str, Any], task_state: Dict[str, Any]) -> str:
        template = spec_meta.get("goal_template") if isinstance(spec_meta, dict) else None
        if isinstance(template, str) and template:
            merged: Dict[str, Any] = {}
            defaults = spec_meta.get("goal_template_defaults") if isinstance(spec_meta, dict) else None
            if isinstance(defaults, dict):
                for k, v in defaults.items():
                    merged[str(k)] = v
            for k, v in task_state.items():
                if v is None or v == "" or v == []:
                    continue
                merged[str(k)] = v
            try:
                return template.format_map(_DefaultFormatDict(merged))
            except Exception:
                pass
        return str(step.get("signature") or spec_meta.get("signature") or step.get("subtask_type") or "")

    def _init_subtask_state(self, subtask_spec: Dict[str, Any]) -> Dict[str, Any]:
        stype = str(subtask_spec.get("type", "") or "").strip()
        spec = self._load_subtask_spec(stype) or {}
        schema = spec.get("subtask_status_schema", {}) if isinstance(spec, dict) else {}
        core = dict((schema.get("core") or {}) if isinstance(schema, dict) else {})
        context = dict((schema.get("context") or {}) if isinstance(schema, dict) else {})
        memory = dict((schema.get("memory") or {}) if isinstance(schema, dict) else {})
        ret = dict((schema.get("return") or {}) if isinstance(schema, dict) else {})
        core.setdefault("subtask_type", stype)
        core.setdefault("status", "running")
        core.setdefault("step_count", 0)
        return {"core": core, "context": context, "memory": memory, "return": ret}

    # ------------------------------------------------------------------
    # SubtaskCaller + ReturnApplier (control layer, no LLM)
    # ------------------------------------------------------------------

    def subtask_caller_prepare(self, *, subtask_spec: Dict[str, Any], subtask_state: Dict[str, Any]) -> Dict[str, Any]:
        task_state = self.task_spec.get("task_state") if isinstance(self.task_spec, dict) else {}
        if not isinstance(task_state, dict):
            task_state = {}

        stype = str(subtask_spec.get("type") or "").strip()
        core = dict(subtask_state.get("core") or {})
        context = dict(subtask_state.get("context") or {})
        memory = dict(subtask_state.get("memory") or {})
        ret = dict(subtask_state.get("return") or {})

        for key, value in list(ret.items()):
            if not isinstance(value, str):
                continue
            lower = value.strip().lower()
            if not lower:
                continue
            if "bool" in lower:
                ret[key] = False
                continue
            if "string" in lower or "null" in lower or "list" in lower or "int" in lower or "float" in lower:
                ret[key] = None

        # Env-declared macro fields (e.g. agent_position, inventory for ALFWorld)
        # are auto-copied into core so the Executor LLM can see them every step.
        for field in self.env_observer.macro_fields_to_subtask_core():
            if isinstance(field, str) and field:
                core[field] = task_state.get(field)

        goal = str(subtask_spec.get("goal") or "").strip()
        if goal:
            core["subtask_goal"] = goal

        for k in list(context.keys()):
            cur = context.get(k)
            if k in task_state and (cur is None or cur == "" or cur == []):
                context[k] = task_state.get(k)

        spec = self._load_subtask_spec(stype) or {}
        caller = spec.get("caller_mapping") if isinstance(spec, dict) else None
        if isinstance(caller, dict):
            inject_ctx = caller.get("inject_context_from_task_state") or caller.get("inject_context_from_task_status")
            if isinstance(inject_ctx, list):
                for item in inject_ctx:
                    if not isinstance(item, dict):
                        continue
                    src = str(item.get("from") or "").strip()
                    dst = str(item.get("to") or "").strip()
                    if not src or not dst:
                        continue
                    cur = context.get(dst)
                    if dst not in context or cur is None or cur == "" or cur == []:
                        context[dst] = task_state.get(src)

            map_ctx_to_mem = caller.get("map_context_to_memory")
            if isinstance(map_ctx_to_mem, list):
                for item in map_ctx_to_mem:
                    if not isinstance(item, dict):
                        continue
                    src = str(item.get("from") or "").strip()
                    dst = str(item.get("to") or "").strip()
                    if not src or not dst:
                        continue
                    cur = memory.get(dst)
                    if dst not in memory or cur is None or cur == "" or cur == []:
                        value = context.get(src)
                        if (value is None or value == "" or value == []) and src in task_state:
                            value = task_state.get(src)
                        memory[dst] = value

        # Env-declared memory prefills (e.g. inventory_snapshot <- task_state.inventory).
        mem_prefill = self.env_observer.macro_to_memory_prefill(memory, task_state)
        if isinstance(mem_prefill, dict):
            for k, v in mem_prefill.items():
                if isinstance(k, str) and k:
                    memory[k] = v

        core.setdefault("subtask_type", stype)
        core.setdefault("status", "running")
        return {"core": core, "context": context, "memory": memory, "return": ret}

    def macro_status_update(self, *, subtask_spec: Dict[str, Any], subtask_state: Dict[str, Any]) -> Dict[str, Any]:
        patch: Dict[str, Any] = {"updated": False, "changes": {}}
        if not isinstance(self.task_spec, dict):
            return patch
        task_state = self.task_spec.get("task_state")
        if not isinstance(task_state, dict):
            return patch

        stype = str(subtask_spec.get("type") or "").strip()
        ret = subtask_state.get("return") if isinstance(subtask_state, dict) else None
        if not isinstance(ret, dict):
            return patch

        def _set_if_present(k: str) -> None:
            v = ret.get(k)
            if v is None or v == "" or v == []:
                return
            # When the macro field is already a list, use append-unique semantics
            # so multiple subtask completions accumulate rather than overwrite.
            if isinstance(task_state.get(k), list):
                cur_list = list(task_state.get(k) or [])
                new_items = v if isinstance(v, list) else [v]
                changed = False
                for item in new_items:
                    if isinstance(item, str) and item.strip() and item not in cur_list:
                        cur_list.append(item)
                        changed = True
                if changed:
                    task_state[k] = cur_list
                    patch["updated"] = True
                    patch["changes"][k] = list(cur_list)
                return

            if task_state.get(k) != v:
                task_state[k] = v
                patch["updated"] = True
                patch["changes"][k] = v

        output_para = subtask_spec.get("output_para")
        if not isinstance(output_para, list) or not output_para:
            spec = self._load_subtask_spec(stype) or {}
            output_para = spec.get("output_para", []) if isinstance(spec, dict) else []
        if isinstance(output_para, list):
            for key in output_para:
                if isinstance(key, str) and key.strip():
                    _set_if_present(key.strip())

        derived = self.env_observer.on_subtask_return(
            subtask_type=stype,
            subtask_state=subtask_state,
            task_state=task_state,
        )
        if isinstance(derived, dict):
            for k, v in derived.items():
                if not isinstance(k, str) or not k:
                    continue
                if task_state.get(k) != v:
                    task_state[k] = v
                    patch["updated"] = True
                    patch["changes"][k] = list(v) if isinstance(v, list) else v

        self.task_spec["task_state"] = task_state
        return patch

    # ------------------------------------------------------------------
    # observe() — control-layer state extraction (delegates to env_observer)
    # ------------------------------------------------------------------

    def observe(self, *, last_action: str, observation: str) -> Dict[str, Any]:
        """
        Parse the environment observation and update macro task_state in-place.

        The observer owns field semantics: it receives the current task_state and
        returns a {field: new_value} dict. ESSAAgent merges every key verbatim
        (replace-if-different). Observers wanting append/remove semantics should
        compute the final value using task_state and return it directly.
        """
        result: Dict[str, Any] = {"updated": False, "changes": {}}
        if not isinstance(self.task_spec, dict):
            return result
        task_state = self.task_spec.get("task_state")
        if not isinstance(task_state, dict):
            return result

        patch = self.env_observer.extract_state_patch(
            last_action, observation, task_state=task_state
        )
        if not isinstance(patch, dict):
            self.task_spec["task_state"] = task_state
            return result

        for k, v in patch.items():
            if not isinstance(k, str) or not k:
                continue
            if task_state.get(k) != v:
                task_state[k] = v
                result["updated"] = True
                result["changes"][k] = list(v) if isinstance(v, list) else v

        self.task_spec["task_state"] = task_state
        return result

    # ------------------------------------------------------------------
    # Call-1: StateUpdater
    # ------------------------------------------------------------------

    def state_update(
        self,
        *,
        subtask_spec: Dict[str, Any],
        subtask_state: Dict[str, Any],
        last_action: str,
        observation: str,
    ) -> Dict[str, Any]:
        stype = str(subtask_spec.get("type") or "").strip()
        spec = self._load_subtask_spec(stype) or {}
        schema = spec.get("subtask_status_schema", {}) if isinstance(spec, dict) else {}
        field_guide = spec.get("field_guide", {}) if isinstance(spec, dict) else {}
        sys_output_format = spec.get("sys_output_format") if isinstance(spec, dict) else None
        operation_space, prompt_variant = self._resolve_state_updater_operation_space(spec)
        if not isinstance(sys_output_format, str):
            sys_output_format = None
        self.last_state_update_prompt_variant = prompt_variant

        prompt = get_state_updater_prompt(
            subtask_type=stype,
            subtask_status_schema=schema,
            field_guide=field_guide,
            sys_output_format=sys_output_format,
            operation_space=operation_space,
            update_mode=self.state_update_mode,
        )
        system = prompt.render_system()
        user = prompt.render_user(subtask_state=subtask_state, last_action=last_action, observation=observation)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        response, stats, prompt_snapshot = self._call_chat(messages)
        parsed = self._parse_state_update(
            response,
            fallback_subtask_state=subtask_state,
            last_action=last_action,
            observation=observation,
            state_update_mode=self.state_update_mode,
        )
        parsed["raw_response"] = response
        parsed["stats"] = stats.to_dict()
        parsed["prompt"] = prompt_snapshot
        parsed["prompt_variant"] = prompt_variant
        return parsed

    def _parse_state_update(
        self,
        response: str,
        *,
        fallback_subtask_state: Dict[str, Any],
        last_action: str,
        observation: str,
        state_update_mode: str = "patch",
    ) -> Dict[str, Any]:
        fallback = {"done": False, "subtask_state": fallback_subtask_state, "meta": {"evidence": [], "warnings": []}}
        use_patch_ops = (state_update_mode or "patch") != "full_state"
        block = _extract_json_block(response or "")
        if not block:
            return fallback
        parsed = _safe_json_load(block)
        if not parsed:
            return fallback

        patch_ops = None
        if use_patch_ops:
            if "patch_ops" not in parsed:
                if all(k in parsed for k in ("op", "path", "value")):
                    parsed = {"patch_ops": [parsed]}
            else:
                po = parsed.get("patch_ops")
                if isinstance(po, dict) and all(k in po for k in ("op", "path", "value")):
                    parsed["patch_ops"] = [po]

            patch_ops = parsed.get("patch_ops")
            if not isinstance(patch_ops, list):
                patch_ops = None

        stype = ""
        try:
            fb_core = fallback_subtask_state.get("core") if isinstance(fallback_subtask_state, dict) else None
            if isinstance(fb_core, dict):
                stype = str(fb_core.get("subtask_type") or "").strip()
        except Exception:
            stype = ""

        if use_patch_ops and patch_ops is not None:
            patch_ops = self._sanitize_patch_ops(
                subtask_type=stype,
                base_state=fallback_subtask_state,
                patch_ops=patch_ops,
                last_action=last_action,
                observation=observation,
            )

        subtask_state = parsed.get("subtask_state")
        if not isinstance(subtask_state, dict) and state_update_mode == "full_state":
            if any(k in parsed for k in ("core", "context", "memory", "return")):
                candidate = {
                    "core": parsed.get("core", {}),
                    "context": parsed.get("context", {}),
                    "memory": parsed.get("memory", {}),
                    "return": parsed.get("return", {}),
                }
                if isinstance(candidate.get("core"), dict):
                    subtask_state = candidate
        if not isinstance(subtask_state, dict):
            if use_patch_ops and patch_ops is not None and isinstance(fallback_subtask_state, dict):
                try:
                    subtask_state = copy.deepcopy(fallback_subtask_state)
                    subtask_state = self._apply_patch_ops(subtask_state, patch_ops)
                except Exception:
                    subtask_state = fallback_subtask_state
            else:
                subtask_state = fallback_subtask_state
        else:
            if use_patch_ops and patch_ops is not None and isinstance(subtask_state, dict):
                try:
                    subtask_state = self._apply_patch_ops(subtask_state, patch_ops)
                except Exception:
                    pass

        try:
            fb = fallback_subtask_state if isinstance(fallback_subtask_state, dict) else {}
            fb_core = fb.get("core") if isinstance(fb.get("core"), dict) else {}
            fb_ctx = fb.get("context") if isinstance(fb.get("context"), dict) else {}
            fb_mem = fb.get("memory") if isinstance(fb.get("memory"), dict) else {}
            fb_ret = fb.get("return") if isinstance(fb.get("return"), dict) else {}

            st_ctx = subtask_state.get("context") if isinstance(subtask_state.get("context"), dict) else None
            st_mem = subtask_state.get("memory") if isinstance(subtask_state.get("memory"), dict) else None
            st_ret = subtask_state.get("return") if isinstance(subtask_state.get("return"), dict) else None
            st_core = subtask_state.get("core") if isinstance(subtask_state.get("core"), dict) else {}

            if st_ctx is None:
                subtask_state["context"] = dict(fb_ctx)
            if st_mem is None:
                subtask_state["memory"] = dict(fb_mem)
            if st_ret is None:
                subtask_state["return"] = dict(fb_ret)
            if not st_core:
                subtask_state["core"] = dict(fb_core)
        except Exception:
            pass

        explicit_done = parsed.get("done")
        done = bool(explicit_done) if explicit_done is True else False
        ret = subtask_state.get("return") if isinstance(subtask_state, dict) else None
        has_done_when_all = False

        if not done:
            if isinstance(ret, dict) and "subtask_accomplished" in ret:
                done = bool(ret.get("subtask_accomplished"))

            if not done:
                spec = self._load_subtask_spec(stype) or {}
                done_when_all = spec.get("done_when_all") if isinstance(spec, dict) else None
                if isinstance(done_when_all, list):
                    paths = [p.strip() for p in done_when_all if isinstance(p, str) and p.strip()]
                    has_done_when_all = bool(paths)
                else:
                    paths = []

                def _get_path_value(state: Dict[str, Any], path: str) -> Any:
                    cur: Any = state
                    for key in path.split("."):
                        if not isinstance(cur, dict) or key not in cur:
                            return None
                        cur = cur.get(key)
                    return cur

                def _is_present(val: Any) -> bool:
                    if val is None:
                        return False
                    # bool MUST be checked before the catch-all `return True` —
                    # in Python `isinstance(True, int)` is True, but more importantly
                    # `False` should mean "not present" for done-check semantics.
                    # Without this branch, `return.flag = false` is treated as "done",
                    # which causes subtasks with bool progress markers to terminate
                    # immediately on step 1 (see ScienceWorld tool_focused regression).
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, str):
                        return bool(val.strip())
                    if isinstance(val, list):
                        return len(val) > 0
                    return True

                if paths and isinstance(subtask_state, dict):
                    done = all(_is_present(_get_path_value(subtask_state, p)) for p in paths)

            if not done and not has_done_when_all and isinstance(ret, dict):
                for k, v in ret.items():
                    if k == "subtask_accomplished":
                        continue
                    if v is None:
                        continue
                    if isinstance(v, str) and not v.strip():
                        continue
                    if isinstance(v, list) and len(v) == 0:
                        continue
                    done = True
                    break

        meta = parsed.get("meta")
        if not isinstance(meta, dict):
            meta = {"evidence": [], "warnings": []}
        meta.setdefault("evidence", [])
        meta.setdefault("warnings", [])

        if done:
            ok, warnings = self.env_observer.validate_subtask_done(
                subtask_type=stype,
                subtask_state=subtask_state,
                last_action=last_action,
                observation=observation,
            )
            if not ok:
                done = False
                if isinstance(warnings, list):
                    for w in warnings:
                        if isinstance(w, str) and w:
                            meta["warnings"].append(w)

        out = {"done": done, "subtask_state": subtask_state, "meta": meta}
        if use_patch_ops and patch_ops is not None:
            out["patch_ops"] = patch_ops
        return out

    def _sanitize_patch_ops(
        self,
        *,
        subtask_type: str,
        base_state: Dict[str, Any],
        patch_ops: List[Any],
        last_action: str,
        observation: str,
    ) -> List[Any]:
        """
        Two-stage patch op cleanup:
          1. Spec-driven allow-list filter (reads patch_ops_policy.allowed from the
             subtask spec JSON). Env-agnostic.
          2. Per-op evidence gating + cross-op post-processing delegated to the
             observer. That is where env-specific text-pattern checks live.
        """
        ops = patch_ops if isinstance(patch_ops, list) else []
        stype = (subtask_type or "").strip()
        spec = self._load_subtask_spec(stype) or {}
        policy = spec.get("patch_ops_policy") if isinstance(spec, dict) else None
        allowed_pairs: set[tuple[str, str]] = set()
        if isinstance(policy, dict):
            allow = policy.get("allowed")
            if isinstance(allow, list):
                for it in allow:
                    if not isinstance(it, dict):
                        continue
                    op = str(it.get("op") or "").strip()
                    path = str(it.get("path") or "").strip()
                    if op and path:
                        allowed_pairs.add((op, path))

        filtered: List[Any] = []
        for it in ops:
            if not isinstance(it, dict):
                continue
            op = str(it.get("op") or "").strip()
            path = str(it.get("path") or "").strip()
            if allowed_pairs and (op, path) not in allowed_pairs:
                continue
            if not self.env_observer.evidence_gate(
                subtask_type=stype,
                op=op,
                path=path,
                value=it.get("value"),
                last_action=last_action,
                observation=observation,
                base_state=base_state,
            ):
                continue
            filtered.append(it)

        processed = self.env_observer.post_process_patch_ops(
            subtask_type=stype,
            base_state=base_state,
            filtered_ops=filtered,
            last_action=last_action,
            observation=observation,
        )
        return processed if isinstance(processed, list) else filtered

    def _apply_patch_ops(self, base_state: Dict[str, Any], patch_ops: List[Any]) -> Dict[str, Any]:
        """
        Apply a list of patch ops to subtask_state in-place.

        Supported ops:
          set              {"op":"set","path":"return.done","value":true}
          list_remove      {"op":"list_remove","path":"memory.unsearched","value":"desk 1"}
          list_append_unique  {"op":"list_append_unique","path":"return.inventory","value":"mug 1"}
        """
        if not isinstance(base_state, dict) or not isinstance(patch_ops, list):
            return base_state

        def _resolve_parent(path: str) -> Tuple[Optional[Dict[str, Any]], str]:
            parts = [p for p in (path or "").split(".") if p]
            if not parts:
                return None, ""
            cur: Any = base_state
            for key in parts[:-1]:
                if not isinstance(cur, dict):
                    return None, ""
                if key not in cur or not isinstance(cur.get(key), dict):
                    return None, ""
                cur = cur.get(key)
            if not isinstance(cur, dict):
                return None, ""
            return cur, parts[-1]

        for item in patch_ops:
            if not isinstance(item, dict):
                continue
            op = str(item.get("op") or "").strip()
            path = str(item.get("path") or "").strip()
            parent, leaf = _resolve_parent(path)
            if not parent or not leaf:
                continue

            if op == "set":
                if "value" not in item:
                    continue
                parent[leaf] = item.get("value")
                continue

            if op == "list_remove":
                val = item.get("value")
                cur = parent.get(leaf)
                if isinstance(cur, list):
                    try:
                        idx = cur.index(val)
                        cur.pop(idx)
                    except ValueError:
                        pass
                continue

            if op == "list_append_unique":
                val = item.get("value")
                cur = parent.get(leaf)
                if isinstance(cur, list) and val not in cur:
                    cur.append(val)
                continue

        return base_state

    # ------------------------------------------------------------------
    # Call-2: Executor
    # ------------------------------------------------------------------

    def executor(
        self,
        *,
        subtask_state: Dict[str, Any],
        goal_text: str,
        admissible_commands: List[str],
        last_action: str,
        observation: str,
    ) -> Dict[str, Any]:
        stype = ""
        if isinstance(subtask_state, dict):
            core = subtask_state.get("core")
            if isinstance(core, dict):
                stype = str(core.get("subtask_type", "") or "").strip()
        spec = self._load_subtask_spec(stype) or {}
        base_actions = spec.get("base_actions", []) if isinstance(spec, dict) else []
        base_actions = self._resolve_executor_base_actions(base_actions)
        executor_sys_rules = spec.get("executor_sys_rules", None) if isinstance(spec, dict) else None

        sub_goal = goal_text
        try:
            if isinstance(subtask_state, dict):
                core = subtask_state.get("core")
                if isinstance(core, dict):
                    sg = core.get("subtask_goal")
                    if isinstance(sg, str) and sg.strip():
                        sub_goal = sg.strip()
        except Exception:
            pass

        use_guided_choice = self.executor_mode == "guided_choice"
        if use_guided_choice and admissible_commands:
            guided = self._executor_guided_choice(
                subtask_state=subtask_state,
                goal_text=goal_text,
                admissible_commands=admissible_commands,
                subtask_type=stype,
                base_actions=base_actions,
                last_action=last_action,
                observation=observation,
            )
            if guided:
                action_only = guided.get("action", "")
                return {
                    "action_intent": {"verb": self.env_observer.detect_action_verb(action_only), "args": {}},
                    "action": action_only,
                    "meta": {"warnings": [], "mode": "guided_choice"},
                    "raw_response": guided.get("raw_response", action_only),
                    "stats": guided.get("stats", {}),
                    "prompt": guided.get("prompt", {}),
                }

        prompt = get_executor_prompt(output_mode="react", subtask_type=stype, base_actions=base_actions, executor_sys_rules=executor_sys_rules)
        system = prompt.render_system()
        user = prompt.render_user(subtask_state=subtask_state, goal_text=sub_goal, last_action=last_action, observation=observation)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response, stats, prompt_snapshot = self._call_chat(messages)
        action_only = self._parse_action_only(response)
        return {
            "action_intent": {"verb": self.env_observer.detect_action_verb(action_only), "args": {}},
            "action": action_only,
            "meta": {"warnings": [], "mode": "action"},
            "raw_response": response,
            "stats": stats.to_dict(),
            "prompt": prompt_snapshot,
        }

    def _executor_guided_choice(
        self,
        *,
        subtask_state: Dict[str, Any],
        goal_text: str,
        admissible_commands: List[str],
        subtask_type: str,
        base_actions: List[object],
        last_action: str,
        observation: str,
    ) -> Optional[Dict[str, Any]]:
        spec = self._load_subtask_spec(subtask_type) or {}
        executor_sys_rules = spec.get("executor_sys_rules", None) if isinstance(spec, dict) else None
        sub_goal = goal_text
        try:
            if isinstance(subtask_state, dict):
                core = subtask_state.get("core")
                if isinstance(core, dict):
                    sg = core.get("subtask_goal")
                    if isinstance(sg, str) and sg.strip():
                        sub_goal = sg.strip()
        except Exception:
            pass
        prompt = get_executor_prompt(output_mode="action", subtask_type=subtask_type, base_actions=base_actions, executor_sys_rules=executor_sys_rules)
        system = prompt.render_system()
        user = prompt.render_user(subtask_state=subtask_state, goal_text=sub_goal, last_action=last_action, observation=observation)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            response, stats, prompt_snapshot = self._call_chat(messages, extra_body={"guided_choice": admissible_commands})
        except Exception:
            return None
        action = (response or "").strip().splitlines()[0]
        if action not in admissible_commands:
            return None
        return {"action": action, "raw_response": response, "stats": stats.to_dict(), "prompt": prompt_snapshot}

    def _parse_action_only(self, response: str) -> str:
        fallback = self.env_observer.default_fallback_action() or ""
        text = (response or "").strip()
        if not text:
            return fallback

        def _clean(s: str) -> str:
            ss = (s or "").strip()
            if ss.startswith(("`", '"', "'")) and ss.endswith(("`", '"', "'")) and len(ss) >= 2:
                ss = ss[1:-1].strip()
            return ss

        def _is_env_command(s: str) -> bool:
            return bool(self.env_observer.detect_action_verb(_normalize_ws(s)))

        m = re.search(r"(?im)^\s*action\s*:\s*(.+?)\s*$", text)
        if m:
            act_line = _clean(m.group(1) or "")
            if act_line:
                act_line = _normalize_ws(act_line)
                return act_line if _is_env_command(act_line) else fallback
        if text.startswith("{"):
            block = _extract_json_block(text) or text
            parsed = _safe_json_load(block) if isinstance(block, str) else None
            if isinstance(parsed, dict):
                act = parsed.get("action")
                if isinstance(act, str) and act.strip():
                    act_line = _normalize_ws(_clean(act))
                    return act_line if _is_env_command(act_line) else fallback
        first = _normalize_ws(text.splitlines()[0].strip() or "")
        return first if _is_env_command(first) else fallback

    # ------------------------------------------------------------------
    # Action legalizer
    # ------------------------------------------------------------------

    def select_legal_action(
        self,
        *,
        executor_action: str,
        action_intent: Optional[Dict[str, Any]],
        admissible_commands: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        if not admissible_commands:
            # Grammar-based envs (e.g. ScienceWorld) do not enumerate admissible
            # commands. Trust the executor's raw action — the env will reject
            # invalid grammar itself. Do NOT normalize: ScienceWorld's normalize
            # lowercases everything, which destroys case-sensitive object IDs
            # like "unknown substance B". Apply only minimal whitespace cleanup.
            import re as _re
            raw = _re.sub(r"\s+", " ", (executor_action or "")).strip().rstrip(".").rstrip(",").strip()
            if not raw:
                return self.env_observer.default_fallback_action() or "", {
                    "reason": "empty_action_no_admissible"
                }
            return raw, {"reason": "passthrough_no_admissible"}

        action = (executor_action or "").strip()
        if action in admissible_commands:
            return action, {"reason": "exact_match"}

        lower_map = {cmd.lower(): cmd for cmd in admissible_commands if isinstance(cmd, str)}
        if action.lower() in lower_map:
            return lower_map[action.lower()], {"reason": "case_insensitive_match"}

        normalized = self.env_observer.normalize_action(action)
        if normalized in admissible_commands:
            return normalized, {"reason": "normalized_match"}

        verb = ""
        if isinstance(action_intent, dict):
            verb = str(action_intent.get("verb", "") or "").strip().lower()
        if not verb:
            verb = self.env_observer.detect_action_verb(action)

        disable_sim = str(os.getenv("ESSA_DISABLE_SIMILARITY_MATCH", "0") or "0").strip().lower() in {"1", "true", "yes"}
        if not disable_sim:
            try:
                from difflib import SequenceMatcher
                candidates = admissible_commands
                if verb:
                    candidates = [c for c in admissible_commands if c.lower().startswith(verb)]
                    if not candidates:
                        candidates = admissible_commands
                best = max(candidates, key=lambda c: SequenceMatcher(None, c.lower(), normalized.lower()).ratio())
                return best, {"reason": "similarity_match", "verb": verb}
            except Exception:
                pass

        return self.env_observer.default_fallback_action() or "", {"reason": "no_match_fallback", "verb": verb}
