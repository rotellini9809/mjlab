"""Task registry system for managing environment registration and creation."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

# Private module-level registry: task_id -> (env_cfg, rl_cfg, runner_cls)
EnvRlCfgPair = tuple[
  ManagerBasedRlEnvCfg,
  RslRlOnPolicyRunnerCfg,
  type | None,
]
_REGISTRY: dict[str, EnvRlCfgPair] = {}


def register_mjlab_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlOnPolicyRunnerCfg,
  runner_cls: type | None = None,
  play_env_cfg: ManagerBasedRlEnvCfg | None = None,
) -> None:
  """Register an environment task.

  Args:
    task_id: Unique task identifier (e.g., "Mjlab-Velocity-Rough-Unitree-Go1").
    env_cfg: Environment configuration instance.
    rl_cfg: RL runner configuration.
    runner_cls: Optional custom runner class. If None, uses OnPolicyRunner.
    play_env_cfg: Optional play mode environment configuration. If provided,
      registers an additional task with "-Play" suffix.
  """
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered")
  _REGISTRY[task_id] = (env_cfg, rl_cfg, runner_cls)

  # Register play variant if provided
  if play_env_cfg is not None:
    play_task_id = f"{task_id}-Play"
    if play_task_id in _REGISTRY:
      raise ValueError(f"Task '{play_task_id}' is already registered")
    _REGISTRY[play_task_id] = (play_env_cfg, rl_cfg, runner_cls)


def list_tasks() -> list[str]:
  """List all registered task IDs."""
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str) -> ManagerBasedRlEnvCfg:
  """Load environment configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  """
  env_cfg, _, _ = _REGISTRY[task_name]
  return deepcopy(env_cfg)


def load_rl_cfg(task_name: str) -> RslRlOnPolicyRunnerCfg:
  """Load RL configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  """
  _, rl_cfg, _ = _REGISTRY[task_name]
  return deepcopy(rl_cfg)


def load_runner_cls(task_name: str) -> type | None:
  """Load the runner class for a task.

  If None, the default OnPolicyRunner will be used.
  """
  _, _, runner_cls = _REGISTRY[task_name]
  return runner_cls
