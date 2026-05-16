"""
AlfworldObserver — BaseEnvObserver implementation for the ALFWorld text-game env.

Owns every piece of ALFWorld-specific knowledge:
- Observation parsing (agent arrival / inventory / pickup / move)
- Action normalization and verb vocabulary
- Entity id extraction ("<name> <N>")
- Macro task_state init fallback + default filling
- Per-op evidence gating on patch ops
- Cross-op post-processing (SEARCH_TWO_OBJECT redirect + auto-fill)
- Derived macro updates on subtask return (MOVE_OBJECT_TO_RECEP inventory cleanup)
- Subtask done validation (TAKE_OBJECT inventory sanity)

ESSAAgent calls into here via the BaseEnvObserver protocol and contains no
ALFWorld-specific text or field names.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from LightWM.env.base import BaseEnvObserver


# ---------------------------------------------------------------------------
# ALFWorld command vocabulary
# ---------------------------------------------------------------------------

COMMAND_PREFIXES: List[str] = [
    "inventory",
    "look",
    "help",
    "go to ",
    "open ",
    "close ",
    "take ",
    "move ",
    "use ",
    "heat ",
    "clean ",
    "cool ",
    "slice ",
    "examine ",
]

FULL_ACTION_SPACE: List[Dict[str, str]] = [
    {"action": "inventory",                         "when": "Check your current inventory when needed."},
    {"action": "go to <receptacle>",                "when": "Navigate to a concrete receptacle id."},
    {"action": "open <receptacle>",                 "when": "Open a closed receptacle before interacting with contents."},
    {"action": "close <receptacle>",                "when": "Close receptacles only when useful for follow-up actions."},
    {"action": "take <object> from <receptacle>",   "when": "Take a visible concrete object id from a concrete receptacle id."},
    {"action": "move <object> to <receptacle>",     "when": "Place a carried object to a concrete destination receptacle id."},
    {"action": "use <object>",                      "when": "Use a concrete object id when the task requires it."},
    {"action": "heat <object> with <receptacle>",   "when": "Heat an object with a microwave receptacle id."},
    {"action": "clean <object> with <receptacle>",  "when": "Clean an object with a sinkbasin receptacle id."},
    {"action": "cool <object> with <receptacle>",   "when": "Cool an object with a fridge receptacle id."},
    {"action": "slice <object> with <object>",      "when": "Slice with concrete object ids only when needed."},
    {"action": "examine <object>",                  "when": "Examine a concrete object id in place."},
    {"action": "look",                              "when": "Refresh the current observation."},
    {"action": "help",                              "when": "Request command hints if needed."},
]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _lower(text: str) -> str:
    return (text or "").strip().lower()


class AlfworldObserver(BaseEnvObserver):
    """
    Parses ALFWorld natural-language observations and normalizes actions.

    Observation patterns handled:
    - "You arrive at <location>."     → agent_position
    - "You are carrying: <items>"     → full inventory replacement
    - "You pick up the <object>."     → append item to inventory
    - "You move the <object> to the " → remove item from inventory
    """

    # ------------------------------------------------------------------
    # Core BaseEnvObserver methods
    # ------------------------------------------------------------------

    def extract_state_patch(
        self,
        last_action: str,
        observation: str,
        *,
        task_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return a flat {field: value} patch for ALFWorld:
          - "agent_position" from "You arrive at ..."
          - "inventory" — always the fully-resolved new list:
              * "You are carrying: ..." -> full replacement
              * "You pick up the X"     -> current inventory + [X]
              * "You move the X to Y"   -> current inventory - [X]
        """
        patch: Dict[str, Any] = {}

        pos = self._extract_agent_position(observation)
        if pos is not None:
            patch["agent_position"] = pos

        cur_inv = task_state.get("inventory") if isinstance(task_state, dict) else None
        if not isinstance(cur_inv, list):
            cur_inv = []

        inv = self._extract_inventory(observation)
        if inv is not None:
            patch["inventory"] = inv
        else:
            picked = self._extract_picked_up_object_id(observation)
            moved = self._extract_moved_object_id(observation)
            if picked or moved:
                new_inv = list(cur_inv)
                if picked and picked not in new_inv:
                    new_inv.append(picked)
                if moved and moved in new_inv:
                    new_inv = [it for it in new_inv if it != moved]
                if new_inv != cur_inv:
                    patch["inventory"] = new_inv

        return patch

    def normalize_action(self, action: str) -> str:
        text = _normalize_ws(action)
        if not text:
            return "look"
        text = text.strip().rstrip(".").rstrip(",").strip()
        low = text.lower()
        low = re.sub(r"^(go to|open|close|examine)\s+the\s+", r"\1 ", low)
        low = re.sub(r"^(take)\s+the\s+", r"\1 ", low)
        low = re.sub(r"^(move)\s+the\s+", r"\1 ", low)
        put_match = re.match(r"(put|place)\s+(.+?)\s+(in|into|on|onto|to)\s+(.+)", low)
        if put_match:
            obj = put_match.group(2).strip()
            target = put_match.group(4).strip()
            return f"move {obj} to {target}"
        return low

    def detect_action_verb(self, action: str) -> str:
        low = _lower(action)
        for prefix in COMMAND_PREFIXES:
            if low.startswith(prefix):
                return prefix.strip()
        return ""

    def get_full_action_space(self) -> List[Dict[str, str]]:
        return list(FULL_ACTION_SPACE)

    # ------------------------------------------------------------------
    # Agent safety fallbacks and human-readable env description
    # ------------------------------------------------------------------

    def default_fallback_action(self) -> str:
        """ALFWorld always accepts `look` as a read-only no-op."""
        return "look"

    def get_env_description(self) -> str:
        """
        Compact command reference for the ReactAgent system prompt. Lists every
        ALFWorld verb with a short description. Kept short so it doesn't bloat
        the prompt for small models.
        """
        lines = [
            "You are in an ALFWorld text environment. Commands accept concrete",
            "entity ids of the form '<name> <N>' (e.g. 'cabinet 1', 'apple 2').",
            "",
            "Available commands:",
        ]
        for entry in FULL_ACTION_SPACE:
            action = entry.get("action", "")
            when = entry.get("when", "")
            if action:
                lines.append(f"- {action}: {when}" if when else f"- {action}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Macro-to-subtask injection hooks
    # ------------------------------------------------------------------

    def macro_fields_to_subtask_core(self) -> List[str]:
        """ALFWorld executors always need the current room and carried items."""
        return ["agent_position", "inventory"]

    def macro_to_memory_prefill(
        self,
        memory: Dict[str, Any],
        task_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prefill ALFWorld-specific memory aliases at subtask start.
        - memory.inventory_snapshot <- task_state.inventory  (different names)
        - memory.agent_position    <- task_state.agent_position (same name, only
          overwrite when task_state has a concrete string value)
        """
        out: Dict[str, Any] = {}
        if not isinstance(memory, dict) or not isinstance(task_state, dict):
            return out
        if "inventory_snapshot" in memory and isinstance(task_state.get("inventory"), list):
            out["inventory_snapshot"] = task_state.get("inventory")
        if "agent_position" in memory and isinstance(task_state.get("agent_position"), str):
            out["agent_position"] = task_state.get("agent_position")
        return out

    # ------------------------------------------------------------------
    # Entity id parsing
    # ------------------------------------------------------------------

    def extract_entity_ids(self, text: str) -> List[str]:
        return _extract_object_ids(text)

    # ------------------------------------------------------------------
    # Macro init
    # ------------------------------------------------------------------

    def macro_init_fallback(
        self,
        *,
        task_type: str,
        task_state_schema: Dict[str, Any],
        goal_text: str,
        initial_observation: str,
    ) -> Dict[str, Any]:
        inferred = self._infer_goal_payload(goal_text)
        return {
            "task_type": task_type,
            "goal": goal_text,
            "target_object": inferred.get("target_object") or "",
            "target_receptacle": inferred.get("target_receptacle") or "",
            "target_receptacle_id": None,
            "all_receptacles": _extract_object_ids(initial_observation),
            "agent_position": None,
            "inventory": [],
            "target_object_location": None,
            "moved_target_object_location": [],
        }

    def finalize_task_state(
        self,
        task_state: Dict[str, Any],
        *,
        initial_observation: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(task_state, dict):
            return {}
        ts = dict(task_state)
        ts.setdefault("target_receptacle_id", None)
        ts.setdefault("inventory", [])
        ts.setdefault("all_receptacles", [])
        ts.setdefault("moved_target_object_location", [])

        if not isinstance(ts.get("inventory"), list):
            ts["inventory"] = []
        if not isinstance(ts.get("all_receptacles"), list):
            ts["all_receptacles"] = []
        if not isinstance(ts.get("moved_target_object_location"), list):
            ts["moved_target_object_location"] = []

        try:
            obs = initial_observation if isinstance(initial_observation, str) else ""
            extracted = _extract_object_ids(obs) if obs else []
            cur_all = ts.get("all_receptacles")
            if not isinstance(cur_all, list):
                cur_all = []
            if extracted and len(extracted) >= 8 and len(cur_all) < max(3, len(extracted) // 2):
                ts["all_receptacles"] = extracted
        except Exception:
            pass

        tri = ts.get("target_receptacle_id")
        if not (isinstance(tri, str) and tri.strip()):
            inferred = self._infer_target_receptacle_id(
                target_receptacle=str(ts.get("target_receptacle") or ""),
                all_receptacles=ts.get("all_receptacles") or [],
            )
            if inferred:
                ts["target_receptacle_id"] = inferred
        return ts

    # ------------------------------------------------------------------
    # Patch-op evidence gating (per op)
    # ------------------------------------------------------------------

    def evidence_gate(
        self,
        *,
        subtask_type: str,
        op: str,
        path: str,
        value: Any,
        last_action: str,
        observation: str,
        base_state: Dict[str, Any],
    ) -> bool:
        stype = (subtask_type or "").strip()
        last = _lower(_normalize_ws(last_action))
        obs = observation if isinstance(observation, str) else ""

        # Only allow frontier removal with concrete evidence
        if op == "list_remove" and path == "memory.unsearched_receptacle_ids":
            has_arrive = bool(re.search(r"\bYou arrive at\b", obs, flags=re.IGNORECASE))
            has_open = last.startswith("open ") or (isinstance(obs, str) and obs.strip().lower().startswith("you open the "))
            if not (has_arrive or has_open):
                return False

        # TAKE_OBJECT: only update inventory on explicit pickup evidence
        if stype in {"TAKE_OBJECT", "TAKE_OBJECT_SECONDARY"} and op == "set" and path == "return.inventory":
            allowed = False
            if last == "inventory" and isinstance(obs, str) and obs.strip().startswith("You are carrying:"):
                allowed = True
            if not allowed and isinstance(obs, str) and re.search(r"\bYou pick up\b", obs, flags=re.IGNORECASE):
                allowed = True
            if not allowed and last.startswith("take "):
                if isinstance(obs, str) and re.search(r"\bYou pick up\b", obs, flags=re.IGNORECASE):
                    allowed = True
            if not allowed:
                return False

        # SEARCH_OBJECT: location must include '<target> <N>' and "You arrive at"
        if stype == "SEARCH_OBJECT" and op == "set" and path == "return.target_object_location":
            if not isinstance(value, str) or not value.strip():
                return False
            mem = base_state.get("memory") if isinstance(base_state, dict) else None
            target_obj = ""
            if isinstance(mem, dict):
                target_obj = str(mem.get("target_object") or "").strip().lower()
            if target_obj:
                if not re.search(rf"\b{re.escape(target_obj)}\s+\d+\b", value.lower()):
                    return False
            if not re.search(r"\bYou arrive at\b", obs, flags=re.IGNORECASE):
                return False

        # SEARCH_OBJ_AND_LIGHT: both target and lamp locations require named id + arrival
        if stype == "SEARCH_OBJ_AND_LIGHT" and op == "set" and path in {
            "return.target_object_location",
            "return.lamp_object_location",
        }:
            if not isinstance(value, str) or not value.strip():
                return False
            mem = base_state.get("memory") if isinstance(base_state, dict) else None
            ctx = base_state.get("context") if isinstance(base_state, dict) else None
            target_obj = ""
            lamp_obj = ""
            if isinstance(mem, dict):
                target_obj = str(mem.get("target_object") or "").strip().lower()
                lamp_obj = str(mem.get("lamp_object") or "").strip().lower()
            if isinstance(ctx, dict):
                if not target_obj:
                    target_obj = str(ctx.get("target_object") or "").strip().lower()
                if not lamp_obj:
                    lamp_obj = str(ctx.get("lamp_object") or "").strip().lower()
            if path == "return.target_object_location" and target_obj:
                if not re.search(rf"\b{re.escape(target_obj)}\s+\d+\b", value.lower()):
                    return False
            if path == "return.lamp_object_location" and lamp_obj:
                if not re.search(rf"\b{re.escape(lamp_obj)}\s+\d+\b", value.lower()):
                    return False
            if not re.search(r"\bYou arrive at\b", obs, flags=re.IGNORECASE):
                return False

        return True

    # ------------------------------------------------------------------
    # Cross-op post-processing
    # ------------------------------------------------------------------

    def post_process_patch_ops(
        self,
        *,
        subtask_type: str,
        base_state: Dict[str, Any],
        filtered_ops: List[Dict[str, Any]],
        last_action: str,
        observation: str,
    ) -> List[Dict[str, Any]]:
        """
        SEARCH_TWO_OBJECT: redirect duplicate target writes → secondary, auto-prune
        the frontier when a location is announced, and auto-fill the second
        location when two target instance ids appear at the same receptacle.
        """
        stype = (subtask_type or "").strip()
        if stype != "SEARCH_TWO_OBJECT":
            return list(filtered_ops)

        ops = [dict(it) if isinstance(it, dict) else it for it in filtered_ops]
        obs = observation if isinstance(observation, str) else ""
        base_ret = base_state.get("return") if isinstance(base_state, dict) else None
        base_mem = base_state.get("memory") if isinstance(base_state, dict) else None
        target_set = bool(isinstance(base_ret, dict) and base_ret.get("target_object_location") not in (None, ""))
        secondary_set = bool(isinstance(base_ret, dict) and base_ret.get("secondary_object_location") not in (None, ""))

        redirected: List[Dict[str, Any]] = []
        extra_ops: List[Dict[str, Any]] = []

        for it in ops:
            if not isinstance(it, dict):
                continue
            op = str(it.get("op") or "").strip()
            path = str(it.get("path") or "").strip()

            if op == "set" and path in {"return.target_object_location", "return.secondary_object_location"}:
                if path == "return.target_object_location":
                    if target_set and not secondary_set:
                        it["path"] = "return.secondary_object_location"
                        path = it["path"]
                    elif target_set and secondary_set:
                        continue
                if path == "return.secondary_object_location" and secondary_set:
                    continue

                v = it.get("value")
                if isinstance(v, str):
                    m = re.search(r"\bat\s+(.+)$", v.strip(), flags=re.IGNORECASE)
                    if m:
                        recep = m.group(1).strip()
                        unsearched = base_mem.get("unsearched_receptacle_ids") if isinstance(base_mem, dict) else None
                        if isinstance(unsearched, list) and recep in unsearched:
                            has_remove = any(
                                isinstance(x, dict)
                                and x.get("op") == "list_remove"
                                and x.get("path") == "memory.unsearched_receptacle_ids"
                                and x.get("value") == recep
                                for x in ops
                            )
                            if not has_remove:
                                extra_ops.append({
                                    "op": "list_remove",
                                    "path": "memory.unsearched_receptacle_ids",
                                    "value": recep,
                                })

            redirected.append(it)

        # Auto-fill secondary_object_location when two target ids are visible at the current receptacle
        try:
            target_obj = str(base_mem.get("target_object") or "").strip().lower() if isinstance(base_mem, dict) else ""
            if target_obj and isinstance(obs, str):
                m = re.search(r"\bYou arrive at (?:the )?([^\.]+)\.", obs, flags=re.IGNORECASE)
                recep = _normalize_ws(m.group(1)) if m else ""
                ids = [x for x in _extract_object_ids(obs) if x.startswith(f"{target_obj} ")]
                if recep and len(ids) >= 2:
                    target_loc = base_ret.get("target_object_location") if isinstance(base_ret, dict) else None
                    secondary_loc = base_ret.get("secondary_object_location") if isinstance(base_ret, dict) else None
                    for op_it in reversed(redirected):
                        if not isinstance(op_it, dict):
                            continue
                        if op_it.get("op") == "set" and op_it.get("path") == "return.target_object_location":
                            target_loc = op_it.get("value")
                            break
                    for op_it in reversed(redirected):
                        if not isinstance(op_it, dict):
                            continue
                        if op_it.get("op") == "set" and op_it.get("path") == "return.secondary_object_location":
                            secondary_loc = op_it.get("value")
                            break
                    target_ids = _extract_object_ids(str(target_loc or ""))
                    secondary_ids = _extract_object_ids(str(secondary_loc or ""))
                    target_id = target_ids[0] if target_ids else ""
                    secondary_id = secondary_ids[0] if secondary_ids else ""
                    used = {target_id, secondary_id}
                    if (target_loc or target_id) and not (secondary_loc or secondary_id):
                        for cid in ids:
                            if cid and cid not in used:
                                extra_ops.append({
                                    "op": "set",
                                    "path": "return.secondary_object_location",
                                    "value": f"{cid} at {recep}",
                                })
                                break
        except Exception:
            pass

        if extra_ops:
            redirected.extend(extra_ops)
        return redirected

    # ------------------------------------------------------------------
    # Derived macro updates on subtask return
    # ------------------------------------------------------------------

    def on_subtask_return(
        self,
        *,
        subtask_type: str,
        subtask_state: Dict[str, Any],
        task_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        MOVE_OBJECT_TO_RECEP: after a successful move, strip the carried target
        object out of the macro inventory list.
        """
        if (subtask_type or "").strip() != "MOVE_OBJECT_TO_RECEP":
            return {}
        mem = subtask_state.get("memory") if isinstance(subtask_state, dict) else None
        if not isinstance(mem, dict):
            return {}
        target_object = str(mem.get("target_object") or "").strip().lower()
        if not target_object:
            return {}
        inv = task_state.get("inventory") if isinstance(task_state, dict) else None
        if not isinstance(inv, list):
            return {}
        filtered = [
            item for item in inv
            if not (isinstance(item, str) and item.lower().startswith(f"{target_object} "))
        ]
        if filtered == inv:
            return {}
        return {"inventory": filtered}

    # ------------------------------------------------------------------
    # Subtask done validation
    # ------------------------------------------------------------------

    def validate_subtask_done(
        self,
        *,
        subtask_type: str,
        subtask_state: Dict[str, Any],
        last_action: str,
        observation: str,
    ) -> Tuple[bool, List[str]]:
        """
        TAKE_OBJECT: only confirm done when the subtask return.inventory actually
        contains the target object id pattern.
        """
        if (subtask_type or "").strip() != "TAKE_OBJECT":
            return True, []
        try:
            ret = subtask_state.get("return") if isinstance(subtask_state, dict) else None
            mem = subtask_state.get("memory") if isinstance(subtask_state, dict) else None
            inv = ret.get("inventory") if isinstance(ret, dict) else None
            target_obj = str(mem.get("target_object") or "").strip().lower() if isinstance(mem, dict) else ""
            if not target_obj or not isinstance(inv, list):
                return True, []
            for it in inv:
                if not isinstance(it, str):
                    continue
                if re.search(rf"^{re.escape(target_obj)}\s+\d+$", it.strip().lower()):
                    return True, []
            return False, ["take_object_inventory_missing_target"]
        except Exception:
            return True, []

    # ------------------------------------------------------------------
    # Internal parsing helpers
    # ------------------------------------------------------------------

    def _extract_agent_position(self, observation: str) -> Optional[str]:
        """Parse "You arrive at [the] <location>." → lowercased location string."""
        if not isinstance(observation, str):
            return None
        m = re.search(r"\bYou arrive at (?:the )?([^\.]+)\.", observation)
        if not m:
            return None
        pos = _normalize_ws(m.group(1) or "")
        return pos.lower() if pos else None

    def _extract_inventory(self, observation: str) -> Optional[List[str]]:
        """
        Parse "You are carrying: <items>" → list of object ids.
        Returns None if the pattern is not present (do NOT update inventory in that case).
        Returns [] if carrying nothing.
        """
        if not isinstance(observation, str):
            return None
        m = re.search(r"\bYou are carrying:\s*(.+)$", observation.strip(), flags=re.IGNORECASE)
        if not m:
            return None
        tail = (m.group(1) or "").strip()
        if not tail or tail.lower().startswith("nothing"):
            return []
        return _extract_object_ids(tail)

    def _extract_picked_up_object_id(self, observation: str) -> Optional[str]:
        if not isinstance(observation, str):
            return None
        m = re.search(r"\bYou pick up the (.+?)(?: from |\.)", observation, flags=re.IGNORECASE)
        if not m:
            return None
        seg = (m.group(1) or "").strip()
        ids = _extract_object_ids(seg)
        return ids[0] if ids else None

    def _extract_moved_object_id(self, observation: str) -> Optional[str]:
        if not isinstance(observation, str):
            return None
        m = re.search(r"\bYou move the (.+?) to the ", observation, flags=re.IGNORECASE)
        if not m:
            return None
        seg = (m.group(1) or "").strip()
        ids = _extract_object_ids(seg)
        return ids[0] if ids else None

    # ------------------------------------------------------------------
    # ALFWorld goal-text + id inference (used by macro_init_fallback and finalize)
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_goal_payload(goal_text: str) -> Dict[str, Any]:
        low = _lower(_normalize_ws(goal_text))
        payload = {"action_type": "", "target_object": "", "target_receptacle": "", "target_tool": ""}

        def _strip_articles(s: str) -> str:
            return re.sub(r"^(the|a|an)\s+", "", (s or "").strip(), flags=re.IGNORECASE)

        m = re.search(r"(?:put|place|move)\s+(?:the\s+)?(.+?)\s+(?:in|into|inside|on|onto|to)\s+(?:the\s+)?(.+)", low)
        if m:
            payload["action_type"] = "move"
            payload["target_object"] = _strip_articles(m.group(1))
            payload["target_receptacle"] = _strip_articles(m.group(2))
            return payload

        payload["action_type"] = "unknown"
        payload["target_object"] = _strip_articles(low)
        return payload

    @staticmethod
    def _infer_target_receptacle_id(*, target_receptacle: str, all_receptacles: List[Any]) -> Optional[str]:
        tr = str(target_receptacle or "").strip().lower()
        if not tr or not isinstance(all_receptacles, list):
            return None
        candidates: List[Tuple[int, str]] = []
        for it in all_receptacles:
            if not isinstance(it, str):
                continue
            s = it.strip()
            low = s.lower()
            if not low.startswith(tr + " "):
                continue
            m = re.search(r"\b(\d+)\b\s*$", low)
            if not m:
                continue
            try:
                num = int(m.group(1))
            except Exception:
                continue
            candidates.append((num, s))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]


# ---------------------------------------------------------------------------
# Module-level utility (deprecated: prefer AlfworldObserver.extract_entity_ids)
# ---------------------------------------------------------------------------

def extract_object_ids(text: str) -> List[str]:
    """
    Extract concrete object/receptacle ids matching "<name> <N>".

    Retained as a module-level function for backward compatibility with any
    downstream scripts that imported it directly. ESSAAgent no longer calls this.
    """
    return _extract_object_ids(text)


def _extract_object_ids(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    pairs = re.findall(r"\b([a-zA-Z]+)\s+(\d+)\b", text)
    out: List[str] = []
    for name, num in pairs:
        cand = f"{name.lower()} {num}"
        if cand not in out:
            out.append(cand)
    return out[:80]
