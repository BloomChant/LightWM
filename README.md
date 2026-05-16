# LightWM

**LightWM** is a lightweight working-memory framework for small language models in interactive text environments.

## Core Idea

LightWM implements the **ESSA** (Explicit State SLM-based Agent) architecture:

- **MacroStateInitializer** initializes a structured task state from the goal and initial observation.
- **StateUpdater** updates subtask-level state via patch operations.
- **Executor** proposes the next environment action from the current subtask state.
- **SubtaskCaller / ReturnApplier** pass state between subtasks without additional LLM calls.

The main design goal is to let small language models reason over compact, structured state objects instead of long unstructured trajectories.

## Repository Layout

```text
agent/
  ESSAAgent.py                 # Core ESSA logic, environment agnostic
  ReactAgent.py                # ReAct baseline agent
env/
  base.py                      # BaseEnvObserver protocol
  alfworld/
    AlfworldEnv.py             # ALFWorld environment wrapper
    AlfworldObserver.py        # ALFWorld observation parser
    simple_config.yaml
prompt/
  ESSA_prompts.py              # Runtime prompt templates
  ReactAgent_prompt.py         # ReAct baseline prompt
memory/
  ESSA/
    task_specs.json            # Task state schemas and subtask sequences
    subtask_specs.json         # Subtask state schemas and policies
offline_induction/
  prompts.py                   # Prompt templates used for offline spec induction experiments
examples/
  alfworld_essa_run.py         # Recommended ALFWorld ESSA entry point
  alfworld_react_run.py        # Recommended ALFWorld ReAct baseline entry point
  task_lists/alfworld/         # Example ALFWorld task lists
config/
  settings.py                  # API keys and endpoint settings
```

## Installation

```bash
pip install -e ".[alfworld]"
```

ALFWorld also requires its own setup and game files. See the ALFWorld project documentation for environment installation details. Benchmark data is not included in this repository; prepare it separately under your local ALFWorld installation.

## Configuration

LightWM uses OpenAI-compatible chat completion endpoints. Set one of the following endpoint configurations:

```bash
export DASH_API_KEY=your_api_key
export DASH_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

or:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=http://localhost:8000/v1
```

## Run ESSA on ALFWorld

Use the plain-text ALFWorld entry point:

```bash
python examples/alfworld_essa_run.py   --model-name qwen3-4b-instruct-2507   --split valid_seen   --task-file examples/task_lists/alfworld/alfworld_valid_seen_full.txt   --max-tasks 5
```

The script writes one log file per run:

```text
logs/alfworld_essa_run_YYYYMMDD_HHMMSS.log
```

The log records task metadata, each selected action, each environment observation, episode results, and an aggregate summary at the end.

Useful options:

```bash
--task-type auto                         # infer task family from ALFWorld path/goal
--task-type pick_and_place_simple        # force a specific task spec
--max-steps 50
--max-subtask-steps 6
--trace-mode first                       # first | all | random
--action-space-mode base                 # base | full
--state-update-mode patch                # patch | full_state
```

## Run the ReAct Baseline

```bash
python examples/alfworld_react_run.py \
  --model-name qwen3-4b \
  --split valid_seen \
  --task-file examples/task_lists/alfworld/alfworld_valid_seen_full.txt \
  --max-tasks 5
```

## Extending to New Environments

To plug LightWM into a new text environment, implement `BaseEnvObserver`:

```python
from LightWM.env.base import BaseEnvObserver

class MyEnvObserver(BaseEnvObserver):
    def extract_state_patch(self, last_action: str, observation: str) -> dict:
        return {"agent_position": ..., "inventory": ...}

    def normalize_action(self, action: str) -> str:
        return action.strip().lower()

    def detect_action_verb(self, action: str) -> str:
        ...

    def get_full_action_space(self) -> list[dict]:
        return []
```

Then pass the observer to `ESSAAgent`:

```python
from LightWM.agent.ESSAAgent import ESSAAgent

agent = ESSAAgent(model_name="...", env_observer=MyEnvObserver())
```

## Offline Induction Prompts

`offline_induction/prompts.py` contains prompt templates used in our offline spec induction experiments. The open-source package keeps these templates as reference material.

## License

MIT
