"""
System / user prompt templates for ReactAgent.

Env-agnostic: the ReAct format (Thought / Action) is fixed here, but every
piece of environment-specific text (command vocabulary, interaction rules)
comes in via `env_description` from the active BaseEnvObserver. Nothing in
this module references a particular env.
"""

from __future__ import annotations

from dataclasses import dataclass


REACT_OUTPUT_INSTRUCTIONS = """[OUTPUT FORMAT]
At every step output exactly two lines and nothing else:
Thought: <one-line reasoning grounded in the latest observation>
Action: <one command accepted by the environment>

Rules:
- Action MUST be a single environment command on one line. No JSON, no chaining, no commentary.
- Use only commands from the environment's command vocabulary.
- If the previous action had no visible effect, change strategy rather than repeating it.
"""


@dataclass
class ReactPrompt:
    """Builder for ReactAgent's system and user messages."""

    task_description: str
    env_description: str = ""
    align_env_rules: bool = True
    enable_geh: bool = False  # reserved for future use; currently unused

    def render_system(self) -> str:
        parts = [
            "You are a ReAct agent solving a single text-game task.",
            "",
            f"[TASK]\n{self.task_description.strip()}" if self.task_description else "[TASK]\n(unspecified)",
        ]
        if self.align_env_rules and self.env_description.strip():
            parts.append("")
            parts.append("[ENVIRONMENT]")
            parts.append(self.env_description.strip())
        parts.append("")
        parts.append(REACT_OUTPUT_INSTRUCTIONS.strip())
        return "\n".join(parts)

    def render_user_observation(self, observation: str) -> str:
        obs = (observation or "").strip()
        return f"[OBSERVATION]\n{obs}" if obs else "[OBSERVATION]\n(empty)"


def get_react_prompt(
    task_description: str,
    *,
    env_description: str = "",
    align_env_rules: bool = True,
    enable_geh: bool = False,
) -> ReactPrompt:
    """
    Build a ReactPrompt.

    Args:
        task_description: the goal string shown in the system prompt.
        env_description: human-readable environment overview — should come from
            BaseEnvObserver.get_env_description(). Omit (or pass "") together
            with align_env_rules=False for a fully env-neutral prompt.
        align_env_rules: when True (and env_description is non-empty), inject
            the env description into the system prompt.
        enable_geh: reserved flag, currently ignored — kept so callers that
            pass it today don't break.
    """
    return ReactPrompt(
        task_description=task_description or "",
        env_description=env_description or "",
        align_env_rules=bool(align_env_rules),
        enable_geh=bool(enable_geh),
    )
