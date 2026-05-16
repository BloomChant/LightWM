"""
ESSA prompts for ALFWorld.
Two single-turn calls per environment step:
- StateUpdater: update explicit states, decide subtask done.
- Executor: propose an action intent or action string.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict


COMMAND_LIST_BLOCK = """[COMMAND LIST]
You must ONLY use the following commands. Do NOT invent new actions.
- inventory:                        check your current inventory
- go to <receptacle>:               move to a receptacle
- open <receptacle>:                open a receptacle (fridge, drawer, cabinet, microwave, safe)
- close <receptacle>:               close a receptacle
- take <object> from <receptacle>:  take an object from a receptacle
- move <object> to <receptacle>:    place an object in or on a receptacle
- use <object>:                     use an object
- heat <object> with <receptacle>:  heat an object using a receptacle (microwave)
- clean <object> with <receptacle>: clean an object using a receptacle (sinkbasin)
- cool <object> with <receptacle>:  cool an object using a receptacle (fridge)
- slice <object> with <object>:     slice an object using a sharp object (knife)
- examine <object>:                 examine an object in place
- look:                             refresh / observe the current location
- help:                             show help / command hints
"""

STATE_UPDATER_RULES = """[STATE UPDATER RULES]
- Your job: decide whether the current subtask is complete, and update key dynamic fields in subtask_state.
- Use ONLY [LAST_ACTION] + [OBSERVATION] as evidence. If no evidence, do NOT change anything.
- Output JSON only following [OUTPUT FORMAT] EXACTLY. Do NOT plan actions.
- If multiple preconditions are satisfied in the same observation, you SHOULD update all relevant fields in one response.
- You MUST NOT guess locations. Only set return.*_location if OBSERVATION contains the exact object id (e.g., "<object> <n>") AND a concrete location from "You arrive at <X>.".
"""

MACRO_INIT_RULES = """[MACRO STATE INIT RULES]
1. You are given a task_state_schema + initial observation + goal text.
2. Output JSON ONLY: a concrete initialized task_state instance that follows the schema.
3. You MUST include ALL keys required by the schema (use null / [] when unknown).
4. You MUST NOT add extra keys not in the schema/output format.
5. Extract all_receptacles from the initial observation as a list of concrete ids (e.g., 'drawer 1', 'desk 2').
6. Best-effort infer target_object and target_receptacle from the goal text.
"""

EXECUTOR_RULES = """[EXECUTOR RULES]
1. Follow [OUTPUT FORMAT] EXACTLY. Action MUST be a SINGLE environment command (no JSON, no chaining). NEVER output fake actions like 'complete subtask'.
2. Do NOT output or mention any subtask name/type (e.g., SEARCH_OBJECT/SEARCH_ITEM/TAKE_OBJECT/MOVE_OBJECT_TO_RECEP) in Thought or Action.
3. Ground everything in [OBSERVATION] + [SUBTASK_STATE]. Prefer [BASE_ACTIONS]. If last action had no effect, change strategy (e.g., open if closed, go elsewhere).
"""

DEFAULT_STATE_UPDATER_OUTPUT_JSON = """{
  "subtask_state": { "core": {}, "payload": {} },
  "meta": { "evidence": [], "warnings": [] }
}"""

DEFAULT_EXECUTOR_OUTPUT_JSON = """{
  "__comment__": "Executor must output JSON only with these top-level keys.",
  "action_intent": {
    "verb": "go to",
    "args": {
      "target": "cabinet 1"
    }
  },
  "action": "go to cabinet 1",
  "meta": {
    "warnings": []
  }
}"""

DEFAULT_TASK_STATUS_INIT_OUTPUT = """NOTE: This is an OUTPUT FORMAT specification (not the output itself).

[OUTPUT FORMAT JSON]
{
  "task_type": "pick_and_place_simple",
  "goal": "take apple and move it to desk",
  "target_object": "apple",
  "target_receptacle": "desk",
  "target_receptacle_id": "desk 1",
  "all_receptacles": [
    "desk 1",
    "drawer 1",
    "cabinet 1"
  ],
  "agent_position": null,
  "inventory": [],
  "target_object_location": null,
  "moved_target_object_location": []
}

[FIELD RULES]
- goal: MUST be a concise normalized string (e.g., "take <obj> and move it to <recep>").
- target_object / target_receptacle: best-effort infer from GOAL.
- target_receptacle_id: best-effort infer from target_receptacle + all_receptacles; deterministic: choose the smallest number.
- all_receptacles:
  - MUST include ALL receptacle/surface/container ids mentioned in INITIAL_OBSERVATION "you see ..." list.
  - MUST NOT filter to only the target receptacle type (e.g., NOT only "cabinet *").
  - Keep concrete ids (e.g., "drawer 2", "countertop 1"). Preserve order if possible.
  - If too many, keep up to 80.
"""

def _drop_guide_keys(obj: Any) -> Any:
    """
    Drop any keys starting with '__' recursively.
    We use these keys only for inline guidance in sys_output_format.
    They should not be fed back into subsequent prompts.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if ks.startswith("__"):
                continue
            out[ks] = _drop_guide_keys(v)
        return out
    if isinstance(obj, list):
        return [_drop_guide_keys(x) for x in obj]
    return obj


def _dump_json_for_prompt(obj: Any) -> str:
    """
    Dump JSON for SLM prompts.
    Default: pretty/indented (structure-friendly).
    Optional: compact (token-efficient).
    """
    pretty = (os.getenv("ESSA_PRETTY_SUBTASK_STATE", "1") or "").strip() == "1"
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _infer_placeholder_from_type(type_desc: str) -> Any:
    low = (type_desc or "").strip().lower()
    if "list" in low:
        return []
    if "null" in low:
        return None
    if "string" in low:
        return ""
    if "int" in low:
        return 0
    if "float" in low:
        return 0.0
    if "bool" in low:
        return False
    return None


def _build_macro_output_format(task_state_schema: Dict[str, Any] | None) -> str:
    if not isinstance(task_state_schema, dict) or not task_state_schema:
        return ""
    fields = task_state_schema.get("fields")
    if not isinstance(fields, dict) or not fields:
        return ""
    output: Dict[str, Any] = {}
    task_type = task_state_schema.get("task_type")
    if isinstance(task_type, str) and task_type.strip():
        output["task_type"] = task_type.strip()
    for key, type_desc in fields.items():
        k = str(key or "").strip()
        if not k:
            continue
        if isinstance(type_desc, str):
            output[k] = _infer_placeholder_from_type(type_desc)
        else:
            output[k] = None
    return json.dumps(output, ensure_ascii=False, indent=2)


@dataclass
class StateUpdaterPrompt:
    subtask_type: str
    subtask_status_schema: Dict[str, Any]
    field_guide: Dict[str, Any]
    sys_output_format: str | None = None
    operation_space: str | None = None
    update_mode: str = "patch"

    def render_system(self) -> str:
        if self.update_mode == "full_state":
            output_block = "\n\n".join(
                [
                    "[OUTPUT PROTOCOL]\n- Return JSON ONLY.\n- Top-level MUST be a single JSON object.\n- You MUST return a full new subtask_state derived from [SUBTASK_STATE].\n- Keep ALL keys that appear in [SUBTASK_STATE], even when values are empty (null, [], \"\").\n- You MUST NOT drop fields, rename fields, or change container types.\n- If a field has no new evidence, copy it unchanged from [SUBTASK_STATE].\n- If [STATE TRANSITION HINTS] are provided, apply them as state effects in subtask_state.",
                    "[OUTPUT FORMAT]\n{\n  \"subtask_state\": {\"...\": \"full object with the same keys as [SUBTASK_STATE]\"},\n  \"meta\": {\"evidence\": [], \"warnings\": []}\n}",
                ]
            )
        elif isinstance(self.sys_output_format, str) and self.sys_output_format.strip():
            output_block = "[OUTPUT FORMAT]\n" + self.sys_output_format.strip()
        else:
            # Backward-compatible fallback: generic format + field_guide block.
            format_text = DEFAULT_STATE_UPDATER_OUTPUT_JSON
            guide_lines = []
            if isinstance(self.field_guide, dict) and self.field_guide:
                def _key_rank(x: object) -> tuple[int, str]:
                    xs = str(x or "").strip()
                    return (1 if xs.startswith("__") else 0, xs)

                for k in sorted(self.field_guide.keys(), key=_key_rank):
                    v = self.field_guide.get(k)
                    kk = str(k or "").strip()
                    vv = str(v or "").strip()
                    if kk and vv:
                        guide_lines.append(f"- {kk}: {vv}")
            field_guide_block = "[FIELD_GUIDE]\n" + ("\n".join(guide_lines) if guide_lines else "{}")
            output_block = "[OUTPUT FORMAT]\n" + format_text + "\n\n" + field_guide_block
        if isinstance(self.operation_space, str) and self.operation_space.strip():
            section_title = "[STATE TRANSITION HINTS]" if self.update_mode == "full_state" else "[OPERATION SPACE]"
            output_block = output_block + "\n\n" + section_title + "\n" + self.operation_space.strip()
        return "\n\n".join(
            [
                f"[ROLE]\nYou are StateUpdater for ESSA subtask_type={self.subtask_type}.",
                STATE_UPDATER_RULES.strip(),
                output_block,
            ]
        )

    def render_user(
        self,
        *,
        subtask_state: Dict[str, Any],
        last_action: str,
        observation: str,
    ) -> str:
        # Reduce noise for SLM: remove program-only / redundant fields.
        st = subtask_state if isinstance(subtask_state, dict) else {}
        core = dict(st.get("core") or {}) if isinstance(st.get("core"), dict) else {}
        memory = dict(st.get("memory") or {}) if isinstance(st.get("memory"), dict) else {}
        ret = dict(st.get("return") or {}) if isinstance(st.get("return"), dict) else {}
        core.pop("step_count", None)
        core.pop("latest_observation", None)
        # status is program-level; avoid distracting SLM
        core.pop("status", None)
        slim_state = {"core": core, "memory": memory, "return": ret}
        slim_state = _drop_guide_keys(slim_state)

        return "\n".join(
            [
                "[SUBTASK_STATE]",
                _dump_json_for_prompt(slim_state),
                "",
                "[LAST_ACTION]",
                (last_action or "").strip(),
                "",
                "[OBSERVATION]",
                observation,
                "",
                "Return JSON only.",
            ]
        )

@dataclass
class MacroStateInitPrompt:
    task_type: str
    init_rules: str | list[str] | None = None
    task_state_schema: Dict[str, Any] | None = None

    def render_system(self) -> str:
        format_text = _build_macro_output_format(self.task_state_schema)
        if not format_text:
            format_text = DEFAULT_TASK_STATUS_INIT_OUTPUT
        rules_block = ""
        if isinstance(self.init_rules, str) and self.init_rules.strip():
            rules_block = "[INIT RULES]\n" + self.init_rules.strip()
        elif isinstance(self.init_rules, list):
            lines = []
            for item in self.init_rules:
                if isinstance(item, str) and item.strip():
                    lines.append(f"- {item.strip()}")
            if lines:
                rules_block = "[INIT RULES]\n" + "\n".join(lines)
        return "\n\n".join(
            [
                f"[ROLE]\nYou are MacroStateInitializer for ESSA task_type={self.task_type}.",
                MACRO_INIT_RULES.strip(),
                "[OUTPUT PROTOCOL]\n- Return JSON ONLY.\n- Top-level MUST be a single JSON object.\n- The object MUST follow [OUTPUT FORMAT] keys and types.\n- Do NOT wrap the output (no patch_ops, no markdown, no explanations).",
                "[OUTPUT FORMAT]\n" + format_text,
                rules_block,
            ]
        )

    def render_user(
        self,
        *,
        goal_text: str,
        initial_observation: str,
    ) -> str:
        return "\n".join(
            [
                "[GOAL]",
                goal_text,
                "",
                "[INITIAL_OBSERVATION]",
                initial_observation,
                "",
                "Return JSON only.",
            ]
        )


@dataclass
class ExecutorPrompt:
    output_mode: str = "react"  # "react" | "action" | "json"
    subtask_type: str = ""
    base_actions: list[object] | None = None
    executor_sys_rules: str | list[str] | None = None

    def render_system(self) -> str:
        format_text = DEFAULT_EXECUTOR_OUTPUT_JSON
        format_block = "[OUTPUT FORMAT]\n" + format_text
        if self.output_mode == "action":
            format_block = (
                "[OUTPUT FORMAT]\nOutput ONLY a single action string from COMMAND LIST. No JSON."
            )
        if self.output_mode == "react":
            format_block = (
                "[OUTPUT FORMAT]\n"
                "Output EXACTLY two lines:\n"
                "Thought: <short>\n"
                "Action: <single command>\n"
                "No JSON. No extra lines."
            )
        ba = self.base_actions or []
        lines = []
        for item in ba:
            if isinstance(item, str):
                lines.append(f"- {item}")
                continue
            if isinstance(item, dict):
                act = str(item.get("action", "") or "").strip()
                when = str(item.get("when", "") or "").strip()
                if act and when:
                    lines.append(f"- {act} — {when}")
                elif act:
                    lines.append(f"- {act}")
                continue
        base_actions_block = "[BASE_ACTIONS]\n" + ("\n".join(lines) if lines else "(none)")

        # Per-subtask system rules (keep short; loaded from subtask spec JSON).
        sub_rules = self.executor_sys_rules
        subtask_rules_block = ""
        if isinstance(sub_rules, list):
            cleaned = [str(x).strip() for x in sub_rules if isinstance(x, (str, int, float)) and str(x).strip()]
            if cleaned:
                subtask_rules_block = "[SUBTASK RULES]\n" + "\n".join(f"- {x}" for x in cleaned[:3])
        elif isinstance(sub_rules, str) and sub_rules.strip():
            subtask_rules_block = "[SUBTASK RULES]\n" + sub_rules.strip()

        return "\n\n".join(
            [
                "[ROLE]\nYou are Executor for ESSA. Propose the next action for the current subtask.",
                f"[CURRENT_SUBTASK]\n{self.subtask_type}\n(Do NOT output this string.)",
                # COMMAND_LIST_BLOCK.strip(),
                EXECUTOR_RULES.strip(),
                subtask_rules_block.strip() if subtask_rules_block.strip() else "",
                base_actions_block,
                format_block,
            ]
        )

    def render_user(
        self,
        *,
        subtask_state: Dict[str, Any],
        goal_text: str | None = None,
        last_action: str | None = None,
        observation: str | None = None,
    ) -> str:
        # Reduce noise for SLM: remove program-only / redundant fields.
        st = subtask_state if isinstance(subtask_state, dict) else {}
        core = dict(st.get("core") or {}) if isinstance(st.get("core"), dict) else {}
        memory = dict(st.get("memory") or {}) if isinstance(st.get("memory"), dict) else {}
        ret = dict(st.get("return") or {}) if isinstance(st.get("return"), dict) else {}
        core.pop("step_count", None)
        core.pop("latest_observation", None)
        core.pop("status", None)
        slim_state = {"core": core, "memory": memory, "return": ret}
        slim_state = _drop_guide_keys(slim_state)

        parts = []
        if goal_text:
            parts.extend(["[GOAL]", goal_text, ""])
        parts.extend(
            [
                "[SUBTASK_STATE]",
                _dump_json_for_prompt(slim_state),
                "",
                "[LAST_ACTION]",
                (last_action or "").strip(),
                "",
                "[OBSERVATION]",
                observation or "",
                "",
                "Return only the required output format.",
            ]
        )
        return "\n".join(parts)


def get_state_updater_prompt(
    *,
    subtask_type: str,
    subtask_status_schema: Dict[str, Any],
    field_guide: Dict[str, Any],
    sys_output_format: str | None = None,
    operation_space: str | None = None,
    update_mode: str = "patch",
) -> StateUpdaterPrompt:
    return StateUpdaterPrompt(
        subtask_type=subtask_type,
        subtask_status_schema=subtask_status_schema,
        field_guide=field_guide,
        sys_output_format=sys_output_format,
        operation_space=operation_space,
        update_mode=update_mode,
    )


def get_macro_init_prompt(
    *, task_type: str, init_rules: str | list[str] | None = None, task_state_schema: Dict[str, Any] | None = None
) -> MacroStateInitPrompt:
    return MacroStateInitPrompt(task_type=task_type, init_rules=init_rules, task_state_schema=task_state_schema)


def get_executor_prompt(
    *,
    output_mode: str = "json",
    subtask_type: str = "",
    base_actions: list[object] | None = None,
    executor_sys_rules: str | list[str] | None = None,
) -> ExecutorPrompt:
    return ExecutorPrompt(
        output_mode=output_mode,
        subtask_type=subtask_type,
        base_actions=base_actions,
        executor_sys_rules=executor_sys_rules,
    )
