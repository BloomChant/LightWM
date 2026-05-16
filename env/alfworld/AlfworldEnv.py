#!/usr/bin/env python3
"""
ALFWorld environment wrapper.

Provides task initialization and action execution for ALFWorld text-game runs.
"""

import os
import json
import yaml
import copy
from typing import Dict, List, Tuple, Any, Optional

class AlfworldEnv:
    """
    Core ALFWorld environment wrapper.
    """
    
    def __init__(self, split: str = "valid_seen", max_steps: int = 50):
        """
        Initialize the environment.
        
        Args:
            split: Dataset split. Defaults to "valid_seen".
            max_steps: Maximum step limit.
        """
        self.split = split
        self.max_steps = max_steps
        self.config = None
        self.alfworld_env = None
        self.all_game_list = []
        # Task state.
        self.current_task_id = None
        self.current_game_path = None
        self.task_is_run = False
        self.obs = None
        self.infos = None
        self.done = False
        self.steps = 0
        
        # Task metadata.
        self.current_task_goal = None
        self.current_task_desc = None
        self.current_trial_dir = None
        self.current_trial_index = None
        
        # Create the underlying ALFWorld instance lazily in init_task.
        self._load_config()
        self._init_game_list()
    
    def _reset_task_info(self):
        """Reset task metadata."""
        self.current_task_goal = None
        self.current_task_desc = None
        self.current_trial_dir = None
        self.current_trial_index = None

    def _list_trial_dirs(self, game_dir_path: str) -> List[str]:
        """Return trial_* subdirectories for a task in stable sorted order."""
        if not os.path.exists(game_dir_path):
            return []
        trial_dirs = [
            d
            for d in os.listdir(game_dir_path)
            if os.path.isdir(os.path.join(game_dir_path, d)) and d.startswith("trial_")
        ]
        return sorted(trial_dirs)

    def _resolve_game_dir_path(self, task_id: int) -> str:
        """Resolve a task id to the task directory path for the configured split."""
        if not self.all_game_list:
            raise ValueError("No game files available")
        if task_id >= len(self.all_game_list):
            raise IndexError(f"Task ID {task_id} is out of range")

        game_rel = self.all_game_list[task_id]

        if self.split == "train":
            root_data_path = os.path.expandvars(self.config["dataset"]["data_path"])
            if root_data_path.endswith("/train"):
                root_data_path = root_data_path[:-6]
        else:
            root_data_path = self.config["dataset"].get("eval_id_data_path", self.config["dataset"]["data_path"])
            if self.split == "valid_unseen":
                root_data_path = self.config["dataset"].get("eval_ood_data_path", root_data_path)
            root_data_path = os.path.expandvars(root_data_path)

        return os.path.join(root_data_path, game_rel)

    def list_task_trials(self, task_id: int) -> List[str]:
        """Return sorted trial_* directory names for a task."""
        game_dir_path = self._resolve_game_dir_path(task_id)
        return self._list_trial_dirs(game_dir_path)
    
    def _extract_task_goal(self, observation: str) -> str:
        """Extract the task goal from an observation.
        
        Args:
            observation: Environment observation text.
            
        Returns:
            Task goal string.
        """
        # Look for content after "Your task is to:".
        task_prefix = "Your task is to:"
        if task_prefix in observation:
            # Extract the task description.
            task_start = observation.find(task_prefix)
            if task_start != -1:
                task_start += len(task_prefix)
                # Find the end of the task description.
                task_end = observation.find('\n', task_start)
                if task_end == -1:
                    task_end = len(observation)
                task_text = observation[task_start:task_end].strip()
                return task_text
        
        # Fall back to other common task/goal line formats.
        lines = observation.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith('task') or line.lower().startswith('goal'):
                # Extract the task-related line.
                if ':' in line:
                    return line.split(':', 1)[1].strip()
        
        # Default to the first 100 characters of the observation.
        return observation[:100] + "..." if len(observation) > 100 else observation

    def _get_train_eval_mode(self) -> str:
        """
        Map this wrapper's split names to ALFWorld train_eval values.

        valid_seen must use eval_in_distribution rather than
        eval_out_of_distribution.
        """
        if self.split == "train":
            return "train"
        if self.split == "valid_seen":
            return "eval_in_distribution"
        # valid_unseen
        return "eval_out_of_distribution"
    
    def _load_config(self):
        """Load the ALFWorld config without creating the underlying environment."""
        config_path = os.path.join(os.path.dirname(__file__), 'simple_config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def _ensure_game_solvable_key(self, game_file_path: str) -> None:
        """
        Ensure the current game file has ALFWorld's expected solvable field.

        AlfredTWEnv skips game.tw-pddl files that lack this field, which can
        leave the environment with zero games. Only the current task file is
        patched.
        """
        try:
            with open(game_file_path, "r", encoding="utf-8") as f:
                gamedata = json.load(f)
            if isinstance(gamedata, dict) and ("solvable" not in gamedata):
                gamedata["solvable"] = True
                tmp_path = f"{game_file_path}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(gamedata, f, ensure_ascii=False)
                os.replace(tmp_path, game_file_path)
        except Exception as exc:
            # Let ALFWorld surface a clearer error later if this compatibility step fails.
            print(f"[WARN] Failed to ensure solvable key for {game_file_path}: {exc}")
    
    def _init_game_list(self):
        """Initialize the game list from the configured data directory."""
        data_path = self.config['dataset']['data_path']
        if self.split != "train":
            data_path = self.config['dataset'].get('eval_id_data_path', data_path)
            if self.split == "valid_unseen":
                data_path = self.config['dataset'].get('eval_ood_data_path', data_path)

        data_path = os.path.expandvars(data_path)
        game_dirs = []
        if os.path.exists(data_path):
            for root, dirs, files in os.walk(data_path):
                if 'game.tw-pddl' in files:
                    # Store the task-type directory that contains the trials.
                    game_dirs.append(os.path.dirname(root))

        root_data_path = data_path.replace('/train', '') if '/train' in data_path else data_path
        # Deduplicate and sort paths so task ids are stable.
        unique_dirs = sorted(set(game_dirs))
        self.all_game_list = [
            os.path.relpath(game_dir, root_data_path) 
            for game_dir in unique_dirs
        ]
    
    def init_task(self, task_id: int, trial_idx: Optional[int] = None, trial_name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Initialize a specific task.
        
        Args:
            task_id: Task id.
            trial_idx: Index into the task's sorted trial_* directories.
                trial_name takes precedence.
            trial_name: Explicit trial_* directory name, for example
                trial_T2019.... Takes precedence over trial_idx.
            
        Returns:
            Tuple of game path and initial observation dictionary.
        """
        self.current_task_id = task_id
        self.current_game_path = self.all_game_list[task_id]
        self.task_is_run = True
        self.steps = 0
        self.done = False
        
        # Reset task metadata.
        self._reset_task_info()

        # Resolve the task directory path.
        game_dir_path = self._resolve_game_dir_path(task_id)

        # Match AlfredTWEnv restrictions: movable and Sliced trajectories are skipped upstream.
        if ("Sliced" in game_dir_path) or ("movable" in game_dir_path):
            raise ValueError(
                f"Unsupported ALFWorld task (movable/Sliced are skipped by AlfredTWEnv): {self.current_game_path}"
            )
        
        # Select the trial directory.
        if not os.path.exists(game_dir_path):
            raise FileNotFoundError(f"Game directory does not exist: {game_dir_path}")

        trial_dirs = self._list_trial_dirs(game_dir_path)
        if not trial_dirs:
            raise FileNotFoundError(f"No trial directories found in game directory: {game_dir_path}")

        selected_trial: str
        if trial_name:
            if trial_name not in trial_dirs:
                raise FileNotFoundError(f"trial_name not found under {game_dir_path}: {trial_name}")
            selected_trial = trial_name
            selected_idx = trial_dirs.index(trial_name)
        else:
            selected_idx = 0 if trial_idx is None else int(trial_idx)
            if selected_idx < 0 or selected_idx >= len(trial_dirs):
                raise IndexError(f"trial_idx {selected_idx} out of range for task {task_id} (0..{len(trial_dirs)-1})")
            selected_trial = trial_dirs[selected_idx]

        self.current_trial_dir = selected_trial
        self.current_trial_index = selected_idx

        trial_dir_path = os.path.join(game_dir_path, selected_trial)
        game_file_path = os.path.join(trial_dir_path, "game.tw-pddl")

        # Compatibility: ensure game.tw-pddl has a solvable field.
        self._ensure_game_solvable_key(game_file_path)
        
        # Reinitialize the underlying environment to load only the current game.
        from alfworld.agents.environment import get_environment
        env_type = self.config['env']['type']
        env_class = get_environment(env_type)
        cfg = copy.deepcopy(self.config)
        # Point data_path at the selected trial so traces can run independently.
        cfg['dataset']['data_path'] = trial_dir_path
        cfg['dataset']['eval_id_data_path'] = trial_dir_path
        cfg['dataset']['eval_ood_data_path'] = trial_dir_path
        cfg['dataset']['num_eval_games'] = 1
        train_eval = self._get_train_eval_mode()
        self.alfworld_env = env_class(cfg, train_eval=train_eval)
        self.alfworld_env = self.alfworld_env.init_env(batch_size=1)
        
        # Reset the environment to get the initial observation.
        self.obs, self.infos = self.alfworld_env.reset()
        
        # Extract the task goal from the initial observation.
        task_goal = self._extract_task_goal(self.obs[0])
        
        # Set task metadata.
        self.current_task_goal = task_goal
        # Remove the split prefix so current_task_desc is just the game directory name.
        task_desc = self.current_game_path
        if task_desc.startswith(f'{self.split}/'):
            task_desc = task_desc[len(self.split) + 1:]
        self.current_task_desc = task_desc
        
        initial_obs = {
            'observation': self.obs[0],
            'admissible_commands': self.infos['admissible_commands'][0],
            'task_description': self.current_task_desc,
            'goal': self.current_task_goal,
            'game_file': self.current_game_path,
            'trial': self.current_trial_dir,
            'trial_index': self.current_trial_index,
        }
        
        # Include the trial in game_path so logs can distinguish traces.
        return f"{self.current_game_path}/{self.current_trial_dir}", initial_obs
    
    def submit_action(self, action: str) -> Dict[str, Any]:
        """
        Submit an action to the environment.
        
        Args:
            action: Action string to execute.
            
        Returns:
            Execution result dictionary.
        """
        if not self.task_is_run:
            return {
                'success': False,
                'error': 'No task is currently running',
                'observation': '',
                'admissible_commands': [],
                'reward': 0,
                'won': False,
                'done': True
            }
        
        if self.done:
            return {
                'success': False,
                'error': 'Task has already ended',
                'observation': 'Task has ended',
                'admissible_commands': [],
                'reward': 0,
                'won': False,
                'done': True
            }
        
        # Execute the action.
        actions = [action]
        self.obs, scores, dones, self.infos = self.alfworld_env.step(actions)
        
        self.done = dones[0]
        self.steps += 1
        
        # Check whether the maximum step count has been reached.
        if self.steps >= self.max_steps:
            self.done = True
        
        # Build the result.
        won = self.infos.get('won', [False])[0] if 'won' in self.infos else False
        
        result = {
            'success': True,
            'observation': self.obs[0],
            'admissible_commands': self.infos['admissible_commands'][0],
            'reward': float(scores[0]),
            'won': won,
            'done': self.done,
            'steps': self.steps
        }
        
        # Update task state.
        if self.done or self.steps >= self.max_steps:
            self.task_is_run = False
        
        return result
    
    def get_available_games(self) -> List[str]:
        """Get the available game list."""
        return self.all_game_list.copy()
    
    def get_current_task_info(self) -> Dict[str, Any]:
        """Get current task information."""
        if self.current_task_id is None:
            return {'status': 'no_task'}
        
        return {
            'task_id': self.current_task_id,
            'game_path': self.current_game_path,
            'is_running': self.task_is_run,
            'is_done': self.done,
            'steps': self.steps,
            'max_steps': self.max_steps
        }
    
    def close(self):
        """Close the environment."""
        self.task_is_run = False
        self.obs = None
        self.infos = None


def main():
    """Run the demo CLI in interactive or non-interactive mode."""
    import argparse
    
    # Command-line arguments.
    parser = argparse.ArgumentParser(description='AlfworldEnv demo tool')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Enable interactive mode')
    parser.add_argument('--split', '-s', default='valid_seen',
                       choices=['train', 'valid_seen', 'valid_unseen'],
                       help='Dataset split')
    parser.add_argument('--task', '-t', type=int, default=0,
                       help='Initial task id for demo mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.split)
    else:
        demo_mode(args.split, args.task)


def demo_mode(split: str, task_id: int):
    """Demo mode showing the basic wrapper flow."""
    print(f"=== AlfworldEnv Demo Mode ===\n")
    
    # Initialize the environment.
    print(f"Initializing environment (split: {split})")
    env = AlfworldEnv(split=split)
    
    print(f"Available games: {len(env.get_available_games())}")
    
    if env.all_game_list:
        # Initialize a task.
        print(f"Initializing task {task_id}")
        game_path, obs = env.init_task(task_id)
        
        print(f"Task: {os.path.basename(game_path)}")
        print(f"Task description: {obs['task_description']}")
        print(f"Task goal: {obs['goal']}")
        print(f"Observation: {obs['observation'][:100]}...")
        print(f"Admissible commands: {len(obs['admissible_commands'])}")
        print()
        
        # Execute one admissible action.
        commands = obs['admissible_commands']
        if commands:
            action = commands[0]
            print(f"Executing action: {action}")
            result = env.submit_action(action)
            
            print(f"Success: {result['success']}")
            print(f"Observation: {result['observation'][:100]}...")
            print(f"Reward: {result['reward']}")
            print(f"Done: {result['done']}")
            print(f"Steps: {result['steps']}")
    
    env.close()
    print("\nDemo complete")


def interactive_mode(split: str):
    """Interactive mode for manual commands."""
    print(f"=== AlfworldEnv Interactive Mode ===\n")
    print("Enter 'help' to view available commands")
    print("Enter 'quit' to exit\n")
    
    # Initialize the environment.
    print(f"Initializing environment (split: {split})")
    env = AlfworldEnv(split=split)
    
    if not env.all_game_list:
        print("No game files available")
        return
    
    print(f"Found {len(env.all_game_list)} available games\n")
    
    # Ask the user to select a task.
    task_id = select_task_interactive(env)
    if task_id is None:
        env.close()
        return
    
    # Initialize the task.
    print(f"\nInitializing task {task_id}")
    game_path, obs = env.init_task(task_id)
    print(obs)
    print(f"Loaded task: {os.path.basename(game_path)}")
    print(f"Task description: {obs['task_description']}")
    print(f"Goal: {obs['goal']}")
    
    # Show the initial state.
    show_state_info(obs)
    
    # Interactive loop.
    interactive_loop(env)


def select_task_interactive(env):
    """Interactive task selection."""
    print(f"Available tasks (0-{len(env.all_game_list)-1}):")
    print("First 10 task examples:")
    for i in range(min(10, len(env.all_game_list))):
        game_name = env.all_game_list[i]
        # Show the final game directory name.
        game_dir_name = os.path.basename(game_name) if game_name else "Unknown game"
        print(f"  {i}: {game_dir_name}")
    
    if len(env.all_game_list) > 10:
        print(f"  ... {len(env.all_game_list) - 10} more tasks")
    
    while True:
        try:
            choice = input(f"\nSelect task ID (0-{len(env.all_game_list)-1}): ").strip()
            
            if choice.lower() in ['quit', 'exit', 'q']:
                return None
            
            task_id = int(choice)
            if 0 <= task_id < len(env.all_game_list):
                return task_id
            else:
                print(f"Please enter a number from 0 to {len(env.all_game_list)-1}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            return None


def interactive_loop(env):
    """Interactive loop for user commands."""
    while True:
        try:
            # Get current state.
            task_info = env.get_current_task_info()
            
            if not task_info.get('is_running', False):
                print(f"\nTask finished. Steps: {task_info['steps']}")
                break
            
            # Show current state.
            print(f"\nState - steps: {task_info['steps']}/{task_info['max_steps']}")
            
            # Read user input.
            user_input = input("\nEnter an action (or 'help', 'commands', 'state', 'quit'): ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands.
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            elif user_input.lower() == 'help':
                show_help()
                continue
            elif user_input.lower() == 'commands':
                show_current_commands(env)
                continue
            elif user_input.lower() == 'state':
                show_detailed_state(env)
                continue
            elif user_input.lower() == 'restart':
                # Restart the current task.
                current_task_id = env.current_task_id
                if current_task_id is not None:
                    print(f"Restarting task {current_task_id}")
                    game_path, obs = env.init_task(current_task_id)
                    show_state_info(obs)
                continue
            
            # Execute the action.
            print(f"Executing: {user_input}")
            result = env.submit_action(user_input)
            
            if result['success']:
                print(f"Success")
                print(f"Observation: {result['observation']}")
                print(f"Reward: {result['reward']:.2f}")
                print(f"Steps: {result['steps']}")
                
                if result['done']:
                    print(f"Task {'won' if result['won'] else 'ended'}!")
                    break
            else:
                print(f"Failed: {result['error']}")
                
                # If the action is invalid, show the available commands.
                if 'No task is currently running' not in result['error']:
                    show_current_commands(env)
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
    
    # Show final state.
    final_info = env.get_current_task_info()
    print(f"\nFinal state: {final_info}")
    
    # Clean up resources.
    env.close()
    print("Environment closed")


def show_help():
    """Show help text."""
    print("\nAvailable commands:")
    print("  help      - Show this help")
    print("  commands  - Show current admissible commands")
    print("  state     - Show detailed state information")
    print("  restart   - Restart the current task")
    print("  quit      - Exit the program")
    print("\nEnter any other action to interact with the environment")
    print("Make sure the action is in the admissible command list")


def show_current_commands(env):
    """Show current admissible commands."""
    if env.obs is not None and env.infos is not None:
        commands = env.infos.get('admissible_commands', [[]])[0] if env.infos else []
        if commands:
            print(f"\nAdmissible commands ({len(commands)}):")
            for i, cmd in enumerate(commands, 1):
                print(f"  {i:2d}. {cmd}")
        else:
            print("\nNo admissible commands are currently available")


def show_detailed_state(env):
    """Show detailed state information."""
    task_info = env.get_current_task_info()
    print(f"\nDetailed state:")
    print(f"  Task ID: {task_info.get('task_id', 'N/A')}")
    print(f"  Game file: {os.path.basename(task_info.get('game_path', ''))}")
    print(f"  Running: {task_info.get('is_running', False)}")
    print(f"  Done: {task_info.get('is_done', False)}")
    print(f"  Current steps: {task_info.get('steps', 0)}")
    print(f"  Max steps: {task_info.get('max_steps', 50)}")
    
    if env.obs is not None:
        print(f"  Current observation: {env.obs[0] if env.obs else 'N/A'}")


def show_state_info(obs):
    """Show initial state information."""
    print(f"\nInitial state:")
    print(f"Environment observation:")
    print(f"{obs['observation']}")
    print(f"\nAdmissible commands ({len(obs['admissible_commands'])}):")
    # Show only the first 10 commands to keep output concise.
    commands = obs['admissible_commands']
    for i, cmd in enumerate(commands[:10], 1):
        print(f"  {i:2d}. {cmd}")
    if len(commands) > 10:
        print(f"  ... {len(commands) - 10} more commands")


if __name__ == "__main__":
    main()
