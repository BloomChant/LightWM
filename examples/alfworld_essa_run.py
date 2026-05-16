#!/usr/bin/env python3
"""
Run ESSA on ALFWorld and write a execution log.

Usage:
  python examples/alfworld_essa_run.py \
    --model-name qwen3-4b-instruct-2507 \
    --split valid_seen \
    --task-file examples/task_lists/alfworld/alfworld_valid_seen_full.txt \
    --max-tasks 5
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_REPO_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

from LightWM.agent.ESSAAgent import ESSAAgent
from LightWM.env.alfworld.AlfworldEnv import AlfworldEnv
from LightWM.env.alfworld.AlfworldObserver import AlfworldObserver


class TextRunLog:
    def __init__(self, log_dir: str = "logs", base_name: str = "alfworld_essa_run") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"{base_name}_{self.timestamp}.log")
        self._handle = open(self.path, "w", encoding="utf-8")

    def write(self, text: str = "") -> None:
        self._handle.write(text + "\n")
        self._handle.flush()

    def section(self, title: str) -> None:
        self.write()
        self.write("=" * 80)
        self.write(title)
        self.write("=" * 80)

    def close(self) -> None:
        self._handle.close()


def _clip(text: Any, limit: int = 1200) -> str:
    if text is None:
        return ""
    text = str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 20)] + "\n... [truncated]"


def _format_dict(d: Dict[str, Any]) -> str:
    if not d:
        return "{}"
    return "\n".join(f"{key}: {d[key]}" for key in sorted(d.keys()))


def load_fixed_tasks(path: str) -> List[Tuple[int, str]]:
    tasks: List[Tuple[int, str]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Task list not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                parts = line.split("#", 1)
                tid = int(parts[0].strip().split()[0])
                desc = parts[1].split("::", 1)[1].strip() if "::" in parts[1] else parts[1].strip()
                tasks.append((tid, desc))
            except Exception:
                continue
    return tasks


def infer_task_type_from_game_path(game_path: str) -> str:
    low = (game_path or "").lower()
    for key in [
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
        "pick_and_place_simple",
    ]:
        if key in low:
            return key
    return ""


def infer_task_type_from_goal(goal: str) -> str:
    low = (goal or "").lower()
    if "clean" in low:
        return "pick_clean_then_place_in_recep"
    if "heat" in low or "microwave" in low:
        return "pick_heat_then_place_in_recep"
    if "cool" in low or "fridge" in low:
        return "pick_cool_then_place_in_recep"
    if "two" in low and "and" in low and ("put" in low or "place" in low or "move" in low):
        return "pick_two_obj_and_place"
    if "lamp" in low or "desklamp" in low or "light" in low or "examine" in low:
        return "look_at_obj_in_light"
    return "pick_and_place_simple"


def run_single_task(
    *,
    env: AlfworldEnv,
    agent: ESSAAgent,
    tid: int,
    task_desc: str,
    requested_task_type: str,
    max_steps: int,
    max_subtask_steps: int,
    episode_index: int,
    log: TextRunLog,
    trial_name: Optional[str] = None,
) -> Dict[str, Any]:
    game_path, initial = env.init_task(tid, trial_name=trial_name)
    initial_obs = initial["observation"]
    admissible = initial.get("admissible_commands", [])
    goal = task_desc or initial.get("goal") or initial.get("task_description") or "Unknown task"

    task_type = requested_task_type
    if requested_task_type == "auto":
        task_type = infer_task_type_from_game_path(game_path) or infer_task_type_from_goal(goal)

    log.section(f"Episode {episode_index}")
    log.write(f"Task ID: {tid}")
    log.write(f"Game path: {game_path}")
    log.write(f"Goal: {goal}")
    log.write(f"Task type: {task_type}")
    if trial_name:
        log.write(f"Trial: {trial_name}")
    log.write()
    log.write("Initial observation:")
    log.write(_clip(initial_obs))

    print(f"\n=== Episode {episode_index} | Task ID {tid} ===")
    print(f"Goal: {goal}")
    print(f"Game path: {game_path}")

    agent.reset(goal_text=goal, initial_observation=initial_obs, task_type=task_type)
    task_state = {}
    if isinstance(getattr(agent, "task_spec", None), dict):
        task_state = agent.task_spec.get("task_state") or {}
    log.write()
    log.write("[MacroStateInitializer]")
    log.write(_format_dict(task_state if isinstance(task_state, dict) else {}))

    observation = initial_obs
    last_action = "(init)"
    total_steps = 0
    task_done = False
    task_won = False

    for subtask_index, subtask_spec in enumerate(agent.subtask_flow):
        subtask_spec["status"] = "running"
        subtask_state = agent.subtask_states.get(subtask_index, {})
        stype = subtask_spec.get("type") or subtask_spec.get("subtask_type") or "UNKNOWN"
        subtask_goal = subtask_spec.get("goal", "")
        log.write()
        log.write(f"[Subtask {subtask_index + 1}] {stype}")
        if subtask_goal:
            log.write(f"Subtask goal: {subtask_goal}")

        try:
            subtask_state = agent.subtask_caller_prepare(subtask_spec=subtask_spec, subtask_state=subtask_state)
            agent.subtask_states[subtask_index] = subtask_state
        except Exception as exc:
            log.write(f"Subtask caller error: {type(exc).__name__}: {exc}")

        for _ in range(max_subtask_steps):
            if total_steps >= max_steps:
                task_done = True
                break

            update = agent.state_update(
                subtask_spec=subtask_spec,
                subtask_state=subtask_state,
                last_action=last_action,
                observation=observation,
            )
            subtask_state = update.get("subtask_state", subtask_state)
            subtask_state.setdefault("core", {})
            try:
                prev_steps = int(subtask_state["core"].get("step_count", 0) or 0)
            except Exception:
                prev_steps = 0
            subtask_state["core"]["step_count"] = prev_steps + 1
            subtask_state["core"]["latest_observation"] = observation
            agent.subtask_states[subtask_index] = subtask_state

            if update.get("done"):
                try:
                    patch = agent.macro_status_update(subtask_spec=subtask_spec, subtask_state=subtask_state)
                except Exception:
                    patch = {}
                subtask_spec["status"] = "done"
                log.write(f"Subtask done before next env action. Macro patch: {patch}")
                break

            exec_result = agent.executor(
                subtask_state=subtask_state,
                goal_text=goal,
                admissible_commands=admissible,
                last_action=last_action,
                observation=observation,
            )
            action_intent = exec_result.get("action_intent", {})
            raw_action = exec_result.get("action", "look")
            action, select_meta = agent.select_legal_action(
                executor_action=raw_action,
                action_intent=action_intent if isinstance(action_intent, dict) else {},
                admissible_commands=admissible,
            )

            env_result = env.submit_action(action)
            observation = env_result.get("observation", "")
            admissible = env_result.get("admissible_commands", [])
            done = bool(env_result.get("done", False))
            won = bool(env_result.get("won", False))
            reward = env_result.get("reward")
            score = env_result.get("score")
            last_action = action

            try:
                agent.observe(last_action=action, observation=observation)
            except Exception:
                pass

            total_steps += 1
            print(f"[Episode {episode_index} Step {total_steps}] {action}")
            log.write()
            log.write(f"[Step {total_steps}]")
            log.write(f"Subtask: {stype}")
            if raw_action != action:
                log.write(f"Raw action: {raw_action}")
                log.write(f"Selected action: {action}")
                if select_meta:
                    log.write(f"Selection: {select_meta}")
            else:
                log.write(f"Action: {action}")
            if score is not None:
                log.write(f"Score: {score}")
            if reward is not None:
                log.write(f"Reward: {reward}")
            log.write("Observation:")
            log.write(_clip(observation))

            if done:
                task_done = True
                task_won = won
                log.write(f"Environment done: won={task_won}")
                break

        if task_done:
            break
        if subtask_spec.get("status") != "done":
            subtask_spec["status"] = "failed"
            task_done = True
            log.write(f"Subtask failed after step cap: {stype}")
            break

    result = {
        "success": task_won,
        "total_steps": total_steps,
        "task_id": tid,
        "goal": goal,
        "game_path": game_path,
        "task_type": task_type,
    }
    log.write()
    log.write("Episode result:")
    log.write(_format_dict(result))
    return result


def run(
    *,
    model_name: str,
    split: str,
    task_file: str,
    task_type: str,
    max_steps: int,
    max_subtask_steps: int,
    max_tasks: int,
    action_space_mode: str,
    state_update_mode: str,
    trace_mode: str,
    max_traces_per_task: Optional[int],
    log_dir: str,
) -> str:
    log = TextRunLog(log_dir=log_dir)
    env = AlfworldEnv(split=split, max_steps=max_steps)
    agent = ESSAAgent(
        model_name=model_name,
        action_space_mode=action_space_mode,
        state_update_mode=state_update_mode,
        env_observer=AlfworldObserver(),
    )

    results: List[Dict[str, Any]] = []
    skipped = 0
    aborted = False

    log.section("Run Configuration")
    log.write(f"Model: {model_name}")
    log.write(f"Split: {split}")
    log.write(f"Task file: {task_file}")
    log.write(f"Task type: {task_type}")
    log.write(f"Max tasks: {max_tasks}")
    log.write(f"Max steps: {max_steps}")
    log.write(f"Max subtask steps: {max_subtask_steps}")
    log.write(f"Action space mode: {action_space_mode}")
    log.write(f"State update mode: {state_update_mode}")
    log.write(f"Trace mode: {trace_mode}")

    try:
        tasks = load_fixed_tasks(task_file)[:max_tasks]
        if not tasks:
            raise RuntimeError(f"No tasks loaded from {task_file}")

        episode_index = 1
        for tid, task_desc in tasks:
            try:
                if trace_mode == "first":
                    trial_names: List[Optional[str]] = [None]
                else:
                    trial_names = env.list_task_trials(tid)
                    if max_traces_per_task is not None:
                        trial_names = trial_names[: int(max_traces_per_task)]
                    if trace_mode == "random":
                        trial_names = [random.choice(trial_names)] if trial_names else [None]

                for trial_name in trial_names:
                    result = run_single_task(
                        env=env,
                        agent=agent,
                        tid=tid,
                        task_desc=task_desc,
                        requested_task_type=task_type,
                        max_steps=max_steps,
                        max_subtask_steps=max_subtask_steps,
                        episode_index=episode_index,
                        log=log,
                        trial_name=trial_name,
                    )
                    results.append(result)
                    episode_index += 1
            except ValueError as exc:
                if any(x in str(exc) for x in ["Unsupported", "movable", "Sliced"]):
                    skipped += 1
                    log.write(f"Skipped task {tid}: {exc}")
                    continue
                raise
            except Exception as exc:
                skipped += 1
                log.write(f"Task {tid} failed with exception: {type(exc).__name__}: {exc}")
                continue
    except KeyboardInterrupt:
        aborted = True
        log.write("KeyboardInterrupt received; finalizing log.")
    finally:
        env.close()
        total = len(results)
        successes = sum(1 for r in results if r.get("success"))
        success_rate = successes / total if total else 0.0
        avg_steps = sum(int(r.get("total_steps") or 0) for r in results) / total if total else 0.0

        log.section("Summary")
        log.write(f"Total episodes: {total}")
        log.write(f"Successes: {successes}")
        log.write(f"Success rate: {success_rate:.3f}")
        log.write(f"Average steps: {avg_steps:.2f}")
        log.write(f"Skipped/failed before completion: {skipped}")
        log.write(f"Aborted: {aborted}")
        log.write(f"Model: {model_name}")
        log.write(f"Split: {split}")
        log.write(f"Log file: {log.path}")
        log.close()

    return log.path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ESSA on ALFWorld with a plain-text log")
    parser.add_argument("--model-name", type=str, default="qwen3-4b-instruct-2507")
    parser.add_argument("--split", type=str, default="valid_seen", choices=["train", "valid_seen", "valid_unseen"])
    parser.add_argument("--task-file", type=str, default="examples/task_lists/alfworld/alfworld_valid_seen_full.txt")
    parser.add_argument(
        "--task-type", type=str, default="auto",
        choices=["auto", "pick_and_place_simple", "look_at_obj_in_light",
                 "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
                 "pick_cool_then_place_in_recep", "pick_two_obj_and_place"],
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-subtask-steps", type=int, default=6)
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--action-space-mode", type=str, default="base", choices=["base", "full"])
    parser.add_argument("--state-update-mode", type=str, default="patch", choices=["patch", "full_state"])
    parser.add_argument("--trace-mode", type=str, default="first", choices=["first", "all", "random"])
    parser.add_argument("--max-traces-per-task", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = run(
        model_name=args.model_name,
        split=args.split,
        task_file=args.task_file,
        task_type=args.task_type,
        max_steps=args.max_steps,
        max_subtask_steps=args.max_subtask_steps,
        max_tasks=args.max_tasks,
        action_space_mode=args.action_space_mode,
        state_update_mode=args.state_update_mode,
        trace_mode=args.trace_mode,
        max_traces_per_task=args.max_traces_per_task,
        log_dir=args.log_dir,
    )
    print(f"\nLog saved to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
