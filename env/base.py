"""
BaseEnvObserver — the boundary between ESSAAgent and a specific text environment.

The agent core is environment-agnostic: anything that depends on how a particular
environment phrases observations, names entities, or expresses commands must live
inside a BaseEnvObserver subclass.

Implementations should either inherit from BaseEnvObserver and override the hooks
they need, or duck-type all methods. The base class provides safe no-op defaults
so a minimal observer only has to implement:
  - extract_state_patch
  - normalize_action
  - detect_action_verb
  - get_full_action_space

The remaining hooks (extract_entity_ids, macro_init_fallback, finalize_task_state,
evidence_gate, post_process_patch_ops, on_subtask_return, validate_subtask_done)
exist so environments with richer schemas (ALFWorld today; ScienceWorld / WebShop
next) can plug domain knowledge in without touching ESSAAgent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class BaseEnvObserver:
    """
    Environment adapter used by ESSAAgent.

    Required overrides: extract_state_patch, normalize_action, detect_action_verb,
    get_full_action_space. The other methods are optional hooks with neutral
    defaults — override them to inject environment-specific behavior.
    """

    # ------------------------------------------------------------------
    # Core (required) — parsing and action vocabulary
    # ------------------------------------------------------------------

    def extract_state_patch(
        self,
        last_action: str,
        observation: str,
        *,
        task_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Parse the raw environment observation and return a {field: new_value} dict.
        ESSAAgent merges every returned key into the macro task_state verbatim.

        Observers own the choice of field names and are responsible for computing
        final values (e.g. a full inventory list after a pickup), using task_state
        if the new value depends on the current state.

        Return {} if nothing can be inferred. Unknown keys are merged as-is; there
        are no agent-side "standard" keys.
        """
        return {}

    def normalize_action(self, action: str) -> str:
        """Canonicalize an LLM-generated action into the env's accepted form."""
        return (action or "").strip()

    def detect_action_verb(self, action: str) -> str:
        """Return the leading verb for similarity-match candidate filtering."""
        return ""

    def get_full_action_space(self) -> List[Dict[str, str]]:
        """Return the full action space; only called in action_space_mode='full'."""
        return []

    # ------------------------------------------------------------------
    # Optional hooks — override to add domain knowledge
    # ------------------------------------------------------------------

    def extract_entity_ids(self, text: str) -> List[str]:
        """
        Extract concrete entity ids (objects, receptacles, pages, products …) from
        a blob of text. ESSAAgent uses this when populating receptacle lists or
        cross-checking inventory contents. Default: [].
        """
        return []

    def macro_init_fallback(
        self,
        *,
        task_type: str,
        task_state_schema: Dict[str, Any],
        goal_text: str,
        initial_observation: str,
    ) -> Dict[str, Any]:
        """
        Produce a minimal macro task_state when the LLM MacroStateInitializer call
        fails or returns something unusable. Default: {task_type, goal}.
        """
        return {"task_type": task_type, "goal": goal_text}

    def finalize_task_state(
        self,
        task_state: Dict[str, Any],
        *,
        initial_observation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Post-process macro task_state after LLM init — fill list defaults, derive
        ids from observation, normalize types, etc. Default: return as-is.
        """
        return task_state if isinstance(task_state, dict) else {}

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
        """
        Decide whether a single patch op is supported by the current evidence.
        Return False to drop it. Default: accept everything that made it past
        the spec allow-list.
        """
        return True

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
        Return the final op list (may append derived ops). Called after evidence
        gating. Default: return filtered_ops unchanged.
        """
        return filtered_ops

    def on_subtask_return(
        self,
        *,
        subtask_type: str,
        subtask_state: Dict[str, Any],
        task_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute macro task_state updates derived on subtask completion, beyond the
        spec's declared output_para copies. Return a {field: new_value} dict;
        ESSAAgent merges it into task_state. Default: {}.
        """
        return {}

    def validate_subtask_done(
        self,
        *,
        subtask_type: str,
        subtask_state: Dict[str, Any],
        last_action: str,
        observation: str,
    ) -> Tuple[bool, List[str]]:
        """
        Final sanity check before marking a subtask done. Return (ok, warnings).
        When ok=False, ESSAAgent reverts done=False and appends warnings to meta.
        Default: (True, []).
        """
        return True, []

    def macro_fields_to_subtask_core(self) -> List[str]:
        """
        Macro task_state field names that ESSAAgent should auto-copy into every
        subtask's core at subtask start (shown to the Executor LLM). Examples:
        ALFWorld uses ["agent_position", "inventory"]; WebShop would use [].
        Default: [].
        """
        return []

    def macro_to_memory_prefill(
        self,
        memory: Dict[str, Any],
        task_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return {memory_key: new_value} patches to prefill subtask memory from the
        macro task_state. Useful when memory contains alias fields whose name
        differs from the macro field (e.g. memory.inventory_snapshot <-
        task_state.inventory). Default: {}.
        """
        return {}

    def default_fallback_action(self) -> str:
        """
        A safe no-op action the env always accepts — used by the agent when an
        LLM response is unparseable or when no admissible command matches.
        Returning "" signals "no safe default"; callers should handle that case.
        Default: "".
        """
        return ""

    def get_env_description(self) -> str:
        """
        Human-readable environment overview (command vocabulary, key rules,
        interaction style). Injected into ReactAgent's system prompt when the
        agent is configured to align with env rules. Default: "".
        """
        return ""
