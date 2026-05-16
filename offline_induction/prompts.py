"""
Prompt templates for the two-stage offline spec distillation pipeline.

These templates are ALFWorld-focused: they assume ALFWorld-style text
observations, action grammar, receptacles, and household object state changes.

Stage 1 — Cognitive Alignment (3 rounds)
  Builds a structured task understanding from traces before touching any ESSA schema.

Stage 2 — ESSA Format Alignment (5 rounds, single shared conversation)
  Translates Stage 1 understanding into ESSA-compatible task/subtask specs.

Change from original design: the former Stage 2 Round 1 ("DSL recap") has been
removed. Its constraints are now part of the Stage 2 system prompt directly.
All 5 Stage 2 rounds run in one continuous multi-turn conversation so each round
has access to all prior reasoning.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def get_stage1_system_prompt(task_background: str, action_list: str) -> str:
    return (
        "# ROLE\n"
        "You are SpecDistiller.Stage1, a task planning and POMDP expert.\n"
        "\n"
        "# TASK BACKGROUND\n"
        f"{task_background}\n"
        "\n"
        f"{action_list}\n"
        "\n"
        "# GLOBAL RULES\n"
        "- Output JSON ONLY for every assistant response. No markdown. No extra keys.\n"
        "- If task_specific_rules are provided, treat them as hard constraints.\n"
    )


def get_stage1_round1_prompt() -> str:
    return (
        "# STAGE1 ROUND1: Task/Environment Understanding\n"
        "Given only the task_type and optional task_specific_rules, summarize the environment\n"
        "physics and action constraints that are always true for this task family.\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "{\n"
        '  "env_rules": ["<always-true constraint about the environment>", ...],\n'
        '  "action_constraints": ["<precondition or sequencing constraint>", ...],\n'
        '  "failure_patterns": ["<common failure and its cause>", ...]\n'
        "}\n"
    )


def get_stage1_round2_prompt() -> str:
    return (
        "# STAGE1 ROUND2: Planner Rollout Success Modeling\n"
        "Given multiple planner rollout traces, extract success-critical state conditions\n"
        "and action preconditions that are explicitly evidenced in the traces.\n"
        "(If no planner traces were provided, output best-effort inferences from the prior round.)\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "{\n"
        '  "success_conditions": ["<what must be true when the task succeeds>", ...],\n'
        '  "success_state": {\n'
        '    "inventory": ["<held objects at completion, if known>"],\n'
        '    "agent_position": "<final position, or null>",\n'
        '    "notes": "<any other relevant end-state observations>"\n'
        "  },\n"
        '  "evidence_patterns": ["<observation substring that signals a key state change>", ...],\n'
        '  "action_preconditions": ["<action X requires Y, evidenced by Z>", ...]\n'
        "}\n"
    )


def get_stage1_round3_prompt() -> str:
    return (
        "# STAGE1 ROUND3: Teacher Trace + POMDP State Modeling\n"
        "Given multiple teacher traces, identify the hidden state fields the agent must track,\n"
        "the observation triggers that update them, and the minimal action chain to success.\n"
        "Action preconditions MUST be grounded in specific trace evidence.\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "{\n"
        '  "required_state_fields": ["<field name and why it is needed>", ...],\n'
        '  "state_update_triggers": [\n'
        '    {"field": "<field>", "trigger": "<observation substring or action>", "effect": "<new value or update rule>"}\n'
        "  ],\n"
        '  "minimal_action_chain": ["<step description in order>", ...],\n'
        '  "action_preconditions": ["<action X requires Y, evidenced by trace>", ...]\n'
        "}\n"
    )


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def get_stage2_system_prompt() -> str:
    """
    Stage 2 system prompt.

    Incorporates the DSL alignment content that was previously elicited as a
    separate Round 1 ("recap round"). All five rounds run in one continuous
    conversation under this system prompt.
    """
    return (
        "# ROLE\n"
        "You are SpecDistiller.Stage2, an ESSA spec synthesizer.\n"
        "\n"
        "# ESSA FRAMEWORK SUMMARY\n"
        "ESSA structures a task as a sequence of subtasks, each with:\n"
        "  subtask_status_schema: {core, context, memory, return}  — four compartments\n"
        "  patch_ops_policy.allowed: [{op, path}]                  — safe update paths\n"
        "  operation_space: string                                  — preconditions + example patch_ops\n"
        "  caller_mapping: {inject_context_from_task_state: [],\n"
        "                   map_context_to_memory: [{from, to}]}\n"
        "  base_actions: [{action, when}]                          — full grammar, no bare verbs\n"
        "  executor_sys_rules: [string, ...]                       — max 3 short rules\n"
        "  done_when_all: [\"return.<field>\", ...]                  — auto-complete trigger\n"
        "\n"
        "Macro task_spec contains:\n"
        "  task_state_schema.fields: {<field>: <type>}  — shared/runtime state\n"
        "  base_subtask_sequence: [{name, subtask_type, signature, expects, produces}]\n"
        "  init_rules: [string, ...]                    — how to populate fields from INITIAL_OBSERVATION\n"
        "\n"
        "[EXAMPLE: PICK_AND_PLACE_SIMPLE]\n"
        "Goal: put a pencil on desk.\n"
        "  SEARCH_OBJECT   — inputs: target_object, all_receptacles\n"
        "                    memory: unsearched_receptacle_ids\n"
        "                    return: target_object_location\n"
        "  TAKE_OBJECT     — inputs: target_object, target_object_location\n"
        "                    return: inventory\n"
        "  MOVE_OBJECT_TO_RECEP — inputs: target_object, target_receptacle_id\n"
        "                         return: moved_target_object_location\n"
        "\n"
        "[PATCH OPS CONSTRAINTS]\n"
        "- Allowed ops: set, list_remove, list_append_unique ONLY.\n"
        "- Every patch op MUST be gated on explicit observation evidence (exact substring).\n"
        "- patch_ops_policy.allowed is the allow-list; all other paths are blocked.\n"
        "- operation_space MUST be a string. MUST contain at least one concrete patch_ops output example.\n"
        "- sys_output_format MUST instruct patch_ops JSON output (not raw field dumps).\n"
        "\n"
        "[FIELD MINIMALITY]\n"
        "- task_state: fields shared across >=2 subtasks, or fields required by the\n"
        "  runtime observer/control layer (for example goal, agent_position, inventory).\n"
        "- Every field MUST have a clear producer: task_state input, prior subtask return, or an\n"
        "  explicit init_rules derivation.\n"
        "- Do NOT introduce fields with no producer.\n"
        "\n"
        "[MERGE PRINCIPLE]\n"
        "- Do NOT create SEARCH_* subtasks whose goal is to locate fixed appliances\n"
        "  (fridge, microwave, sinkbasin). Use init_rules to set their ids from all_receptacles.\n"
        "- SEARCH_OBJECT may still navigate to or open a fixed appliance/container when\n"
        "  searching for the target object inside it.\n"
        "- For look_at_obj_in_light, use lamp_object / lamp_object_location; NEVER secondary_object.\n"
        "- Prefer combined-search subtasks over two separate searches when evidence supports it.\n"
        "\n"
        "[CALLER MAPPING]\n"
        "- inject_context_from_task_state MUST always be [] (runtime auto-injects same-name fields).\n"
        "- map_context_to_memory MUST be explicit for every subtask.\n"
        "- context MUST declare all task_state fields referenced in map_context_to_memory.\n"
        "\n"
        "# GLOBAL RULES\n"
        "- Output JSON ONLY for every assistant response. No markdown. No extra keys.\n"
        "- Follow the ESSA schema strictly throughout this conversation.\n"
    )


def get_stage2_round1_prompt() -> str:
    """Coverage analysis — was Round 2 in original design."""
    return (
        "# STAGE2 ROUND1: Coverage Analysis\n"
        "Given the Stage 1 outputs and the current subtask library summary, determine whether\n"
        "the existing subtasks are sufficient for this task type.\n"
        "Work from evidence — do NOT assume capabilities from the task_type name alone.\n"
        "\n"
        "Checklist:\n"
        "- Is each capability in stage1 minimal_action_chain covered by an existing subtask?\n"
        "- Do NOT propose SEARCH subtasks whose goal is to locate fixed appliances\n"
        "  (fridge/microwave/sinkbasin). Searching for an object inside one is allowed.\n"
        "- Do NOT propose OPEN_RECEPTACLE if SEARCH_OBJECT already handles it.\n"
        "- For lamp tasks: verify lamp_object/lamp_object_location fields are present.\n"
        "- For multi-instance goals: assess whether combined-search covers the need.\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "{\n"
        '  "supported": "yes|partial|no",\n'
        '  "missing_capabilities": ["<description of gap and which trace step implies it>", ...],\n'
        '  "minimal_delta": ["<proposed new subtask or extension, with justification>", ...]\n'
        "}\n"
    )


def get_stage2_round2_prompt() -> str:
    """Subtask spec synthesis and patch planning — was Round 3."""
    return (
        "# STAGE2 ROUND2: Subtask Spec Synthesis\n"
        "Using the coverage_report from the prior round, synthesize specs for new subtasks\n"
        "and patches for existing subtasks when an extension is sufficient.\n"
        "If supported=yes and minimal_delta is empty, output {\"new_subtasks\": [], \"subtask_patches\": {}} and stop.\n"
        "\n"
        "Requirements:\n"
        "- Prefer subtask_patches over new_subtasks when an existing subtask can be safely extended.\n"
        "- Only synthesize or patch subtasks directly implied by coverage gaps and trace evidence.\n"
        "- New subtasks MUST include goal_template.\n"
        "- operation_space MUST name explicit evidence strings from OBSERVATION/LAST_ACTION.\n"
        "- operation_space_full_state is recommended when the full-state effect is easy to state,\n"
        "  but patch_ops operation_space remains the source of truth.\n"
        "- done_when_all MUST be included when a subtask returns multiple fields.\n"
        "- base_actions: [{action, when}] with full command grammar.\n"
        "- Never create SEARCH_* whose goal is to locate fixed appliances. SEARCH_OBJECT may\n"
        "  still navigate to/open fixed appliances while searching for a target object.\n"
        "- For multi-instance search: define deterministic first/second-match rules.\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "{\n"
        '  "new_subtasks": [\n'
        "    {\n"
        '      "subtask_type": "EXAMPLE_SUBTASK",\n'
        '      "goal_template": "human-readable goal with {field} placeholders",\n'
        '      "input_para": [],\n'
        '      "output_para": [],\n'
        '      "caller_mapping": {\n'
        '        "inject_context_from_task_state": [],\n'
        '        "map_context_to_memory": [{"from": "task_state_field", "to": "memory_field"}]\n'
        "      },\n"
        '      "subtask_status_schema": {\n'
        '        "core": {}, "context": {}, "memory": {}, "return": {}\n'
        "      },\n"
        '      "patch_ops_policy": {"allowed": [{"op": "set", "path": "return.field"}]},\n'
        '      "sys_output_format": "<patch_ops output format string>",\n'
        '      "operation_space": "<preconditions + evidence patterns + example patch_ops>",\n'
        '      "operation_space_full_state": "<optional full-state equivalent, or omit if unnecessary>",\n'
        '      "done_when_all": ["return.field"],\n'
        '      "executor_sys_rules": [],\n'
        '      "base_actions": [{"action": "go to <receptacle>", "when": "..."}]\n'
        "    }\n"
        "  ],\n"
        '  "subtask_patches": {\n'
        '    "EXISTING_SUBTASK": {\n'
        '      "rationale": "<why a patch is enough>",\n'
        '      "patch": {\n'
        '        "base_actions": [{"action": "open <receptacle>", "when": "..."}],\n'
        '        "executor_sys_rules": ["..."],\n'
        '        "operation_space": "<corrected or extended operation_space if needed>"\n'
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def get_stage2_round3_prompt() -> str:
    """Task spec synthesis — was Round 4."""
    return (
        "# STAGE2 ROUND3: Task Spec Synthesis\n"
        "Using all Stage 1 outputs and the synthesized subtasks, output the task_spec.\n"
        "\n"
        "Requirements:\n"
        "- base_subtask_sequence MUST be derived from stage1 minimal_action_chain and\n"
        "  available subtask capabilities, NOT copied from a task-name template.\n"
        "- task_state_schema.fields: fields shared across >=2 subtasks, or fields required by\n"
        "  the runtime observer/control layer (for example goal, agent_position, inventory).\n"
        "- base_subtask_sequence entries SHOULD match existing specs: include name,\n"
        "  subtask_type, signature, expects, and produces.\n"
        "- If the goal implies multiple placements, sequence must include all required\n"
        "  pickup-and-place cycles with valid field dependencies between steps.\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "{\n"
        '  "task_type": "<task_type string>",\n'
        '  "task_state_schema": {\n'
        '    "fields": {\n'
        '      "<field_name>": "<type: string|list|null|bool>"\n'
        "    }\n"
        "  },\n"
        '  "base_subtask_sequence": [\n'
        "    {\n"
        '      "name": "<camelCaseStepName>",\n'
        '      "subtask_type": "<SUBTASK_TYPE>",\n'
        '      "signature": "<short human-readable description>",\n'
        '      "expects": ["<task_state_field>", ...],\n'
        '      "produces": ["<task_state_field>", ...]\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def get_stage2_round4_prompt() -> str:
    """caller_mapping + init_rules — was Round 5."""
    return (
        "# STAGE2 ROUND4: Mapping + Init Rules\n"
        "Complete caller_mapping for every subtask and generate init_rules.\n"
        "\n"
        "Requirements:\n"
        "- inject_context_from_task_state MUST be [] for every subtask.\n"
        "- map_context_to_memory MUST be present and non-empty for every subtask that uses memory.\n"
        "- context MUST declare all task_state fields referenced by map_context_to_memory.\n"
        "- init_rules: rules for populating task_state fields from INITIAL_OBSERVATION.\n"
        "  Use the init_rules_template style, but support ALFWorld variants such as\n"
        "  'you see ...' lists, 'You arrive at <receptacle>.', and closed-receptacle text.\n"
        "- No new fields may be introduced at this stage.\n"
        "- For lamp tasks: use lamp_object/lamp_object_location only.\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "{\n"
        '  "subtask_mappings": {\n'
        '    "<SUBTASK_TYPE>": {\n'
        '      "caller_mapping": {\n'
        '        "inject_context_from_task_state": [],\n'
        '        "map_context_to_memory": [{"from": "ctx_field", "to": "mem_field"}]\n'
        "      }\n"
        "    }\n"
        "  },\n"
        '  "task_spec_patch": {\n'
        '    "init_rules": ["<rule string derived from INITIAL_OBSERVATION pattern>", ...]\n'
        "  }\n"
        "}\n"
    )


def get_stage2_round5_prompt() -> str:
    """Schema repair — was Round 6."""
    return (
        "# STAGE2 ROUND5: Spec Repair\n"
        "Fix any remaining structural issues in the subtask specs and task spec.\n"
        "\n"
        "Checklist (address ALL that apply):\n"
        "- inject_context_from_task_state MUST be [] everywhere.\n"
        "- subtask_status_schema shape MUST be {core, context, memory, return} for every subtask.\n"
        "- patch_ops_policy.allowed MUST be a list of {op, path} dicts (NOT tuples, NOT strings).\n"
        "- All patch ops MUST be in {set, list_remove, list_append_unique}; rewrite any others.\n"
        "- sys_output_format MUST produce patch_ops output; rewrite raw field dumps.\n"
        "- operation_space MUST be a string with explicit OBSERVATION evidence and example patch_ops.\n"
        "- base_actions MUST use {action, when} with full grammar (no bare verbs).\n"
        "- init_rules MUST describe how to parse all_receptacles from ALFWorld initial\n"
        "  observations, including 'you see ...', 'You arrive at <receptacle>.', and\n"
        "  closed-receptacle variants when present.\n"
        "- Remove any SEARCH_* whose goal is to locate fridge/microwave/sinkbasin; add\n"
        "  their ids to init_rules. Do not remove SEARCH_OBJECT just because it may open\n"
        "  one of those containers while searching for a target object.\n"
        "- For lamp tasks: replace any secondary_object with lamp_object.\n"
        "\n"
        "Output ONLY the patches that require fixing. If a subtask needs no changes, omit it.\n"
        "\n"
        "# OUTPUT FORMAT\n"
        "{\n"
        '  "subtask_spec_patches": {\n'
        '    "<SUBTASK_TYPE>": {\n'
        '      "caller_mapping": {"inject_context_from_task_state": [], "map_context_to_memory": []},\n'
        '      "subtask_status_schema": {"core": {}, "context": {}, "memory": {}, "return": {}},\n'
        '      "patch_ops_policy": {"allowed": [{"op": "set", "path": "return.field"}]},\n'
        '      "sys_output_format": "<corrected format>",\n'
        '      "operation_space": "<corrected operation space with evidence and example>"\n'
        "    }\n"
        "  },\n"
        '  "task_spec_patch": {\n'
        '    "init_rules": ["<corrected rule>", ...]\n'
        "  }\n"
        "}\n"
    )
