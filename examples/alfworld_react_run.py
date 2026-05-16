#!/usr/bin/env python3
"""
Run the ReAct baseline on ALFWorld and write a execution log.

Usage:
  python examples/alfworld_react_run.py \
    --model-name qwen3-4b \
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

from LightWM.agent.ReactAgent import ReactAgent
from LightWM.env.alfworld.AlfworldEnv import AlfworldEnv
from LightWM.env.alfworld.AlfworldObserver import AlfworldObserver


class TextRunLog:
    def __init__(self, log_dir: str = "logs", base_name: str = "alfworld_react_run") -> None:
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


def run_single_task(
    *,
    env: AlfworldEnv,
    agent: ReactAgent,
    tid: int,
    task_desc: str,
    max_steps: int,
    episode_index: int,
    log: TextRunLog,
    trial_name: Optional[str] = None,
) -> Dict[str, Any]:
    game_path, initial = env.init_task(tid, trial_name=trial_name)
    initial_obs = initial["observation"]
    admissible = initial.get("admissible_commands", [])
    goal = task_desc or initial.get("goal") or initial.get("task_description") or "Unknown task"

    log.section(f"Episode {episode_index}")
    log.write(f"Task ID: {tid}")
    log.write(f"Game path: {game_path}")
    log.write(f"Goal: {goal}")
    if trial_name:
        log.write(f"Trial: {trial_name}")
    log.write()
    log.write("Initial observation:")
    log.write(_clip(initial_obs))

    print(f"\n=== Episode {episode_index} | Task ID {tid} ===")
    print(f"Goal: {goal}")
    print(f"Game path: {game_path}")

    agent.reset(task_description=goal)
    observation = initial_obs
    total_steps = 0
    task_done = False
    task_won = False
    final_score = 0

    for _ in range(max_steps):
        step_result = agent.react_step(observation=observation, admissible_commands=admissible)
        raw_action = step_result.get("raw_action", "")
        action = step_result.get("action", raw_action or "look")
        thought = step_result.get("thought", "")
        correction = step_result.get("action_correction") or {}

        env_result = env.submit_action(action)
        observation = env_result.get("observation", "")
        admissible = env_result.get("admissible_commands", [])
        done = bool(env_result.get("done", False))
        won = bool(env_result.get("won", False))
        reward = env_result.get("reward")
        score = env_result.get("score")
        if score is not None:
            try:
                final_score = int(score)
            except Exception:
                pass

        total_steps += 1
        print(f"[Episode {episode_index} Step {total_steps}] {action}")
        log.write()
        log.write(f"[Step {total_steps}]")
        if thought:
            log.write("Thought:")
            log.write(_clip(thought, 800))
        if raw_action and raw_action != action:
            log.write(f"Raw action: {raw_action}")
            log.write(f"Selected action: {action}")
            if correction:
                log.write(f"Action correction: {correction}")
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

    if not task_done:
        log.write(f"Reached max_steps={max_steps} without env_done.")

    result = {
        "success": task_won,
        "total_steps": total_steps,
        "final_score": final_score,
        "task_id": tid,
        "goal": goal,
        "game_path": game_path,
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
    max_steps: int,
    max_tasks: int,
    history_k: int,
    similarity_match: bool,
    align_env_rules: bool,
    trace_mode: str,
    max_traces_per_task: Optional[int],
    log_dir: str,
) -> str:
    log = TextRunLog(log_dir=log_dir)
    env = AlfworldEnv(split=split, max_steps=max_steps)
    agent = ReactAgent(
        model_name=model_name,
        history_k=history_k,
        similarity_match=similarity_match,
        align_env_rules=align_env_rules,
        env_observer=AlfworldObserver(),
    )

    results: List[Dict[str, Any]] = []
    skipped = 0
    aborted = False

    log.section("Run Configuration")
    log.write(f"Model: {model_name}")
    log.write(f"Split: {split}")
    log.write(f"Task file: {task_file}")
    log.write(f"Max tasks: {max_tasks}")
    log.write(f"Max steps: {max_steps}")
    log.write(f"History k: {history_k}")
    log.write(f"Similarity match: {similarity_match}")
    log.write(f"Align env rules: {align_env_rules}")
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
                        max_steps=max_steps,
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
        avg_score = sum(int(r.get("final_score") or 0) for r in results) / total if total else 0.0

        log.section("Summary")
        log.write(f"Total episodes: {total}")
        log.write(f"Successes: {successes}")
        log.write(f"Success rate: {success_rate:.3f}")
        log.write(f"Average steps: {avg_steps:.2f}")
        log.write(f"Average final score: {avg_score:.2f}")
        log.write(f"Skipped/failed before completion: {skipped}")
        log.write(f"Aborted: {aborted}")
        log.write(f"Model: {model_name}")
        log.write(f"Split: {split}")
        log.write(f"Log file: {log.path}")
        log.close()

    return log.path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReAct on ALFWorld with a plain-text log")
    parser.add_argument("--model-name", type=str, default="qwen3-4b")
    parser.add_argument("--split", type=str, default="valid_seen", choices=["train", "valid_seen", "valid_unseen"])
    parser.add_argument("--task-file", type=str, default="examples/task_lists/alfworld/alfworld_valid_seen_full.txt")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--history-k", type=int, default=-1)
    parser.add_argument("--similarity-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-env-rules", action=argparse.BooleanOptionalAction, default=True)
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
        max_steps=args.max_steps,
        max_tasks=args.max_tasks,
        history_k=args.history_k,
        similarity_match=args.similarity_match,
        align_env_rules=args.align_env_rules,
        trace_mode=args.trace_mode,
        max_traces_per_task=args.max_traces_per_task,
        log_dir=args.log_dir,
    )
    print(f"\nLog saved to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
