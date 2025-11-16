"""Generic tests for task config integrity."""

import pytest

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import ObservationGroupCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg


@pytest.fixture(scope="module")
def all_task_ids() -> list[str]:
  """Get all registered task IDs."""
  return list_tasks()


@pytest.fixture(scope="module")
def play_task_pairs(all_task_ids: list[str]) -> list[tuple[str, str]]:
  """Get pairs of (training_task, play_task) for comparison.

  Returns pairs where the play task is the training task with "-Play" suffix.
  """
  pairs = []
  for task_id in all_task_ids:
    if task_id.endswith("-Play"):
      training_id = task_id[: -len("-Play")]
      if training_id in all_task_ids:
        pairs.append((training_id, task_id))
  return pairs


def test_all_tasks_loadable(all_task_ids: list[str]) -> None:
  """All registered tasks should be loadable without errors."""
  for task_id in all_task_ids:
    try:
      cfg = load_env_cfg(task_id)
      assert isinstance(cfg, ManagerBasedRlEnvCfg), (
        f"Task {task_id} did not return ManagerBasedRlEnvCfg"
      )
    except Exception as e:
      pytest.fail(f"Failed to load task '{task_id}': {e}")


def test_play_task_pairs_exist(all_task_ids: list[str]) -> None:
  """All tasks should have a corresponding Play variant."""
  non_play_tasks = [
    t for t in all_task_ids if not t.endswith("-Play") and not t.endswith("-Demo")
  ]
  for task_id in non_play_tasks:
    play_id = f"{task_id}-Play"
    assert play_id in all_task_ids, f"Task {task_id} missing Play variant: {play_id}"


def test_play_mode_episode_length(play_task_pairs: list[tuple[str, str]]) -> None:
  """Play mode tasks should have infinite episode length."""
  for _, play_task_id in play_task_pairs:
    cfg = load_env_cfg(play_task_id)
    assert cfg.episode_length_s >= 1e9, (
      f"{play_task_id} episode_length_s={cfg.episode_length_s}, expected >= 1e9"
    )


def test_play_mode_observation_corruption_disabled(all_task_ids: list[str]) -> None:
  """Play/Demo mode tasks should have observation corruption disabled for policy."""
  play_demo_tasks = [
    t for t in all_task_ids if t.endswith("-Play") or t.endswith("-Demo")
  ]

  for task_id in play_demo_tasks:
    cfg = load_env_cfg(task_id)

    assert "policy" in cfg.observations, (
      f"Play/Demo task {task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["policy"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Play/Demo task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert not policy_obs.enable_corruption, (
      f"Play/Demo task {task_id} has enable_corruption=True, expected False"
    )


def test_training_mode_observation_corruption_enabled(all_task_ids: list[str]) -> None:
  """Training mode tasks should have observation corruption enabled for policy."""
  training_tasks = [
    t for t in all_task_ids if not t.endswith("-Play") and not t.endswith("-Demo")
  ]

  for task_id in training_tasks:
    cfg = load_env_cfg(task_id)

    assert "policy" in cfg.observations, (
      f"Training task {task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["policy"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Training task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert policy_obs.enable_corruption, (
      f"Training task {task_id} has enable_corruption=False, expected True"
    )


def test_critic_observation_corruption_always_disabled(all_task_ids: list[str]) -> None:
  """Critic observations should always have corruption disabled."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    if "critic" not in cfg.observations:
      continue

    critic_obs = cfg.observations["critic"]
    assert isinstance(critic_obs, ObservationGroupCfg), (
      f"Task {task_id} critic observation is not ObservationGroupCfg"
    )

    assert not critic_obs.enable_corruption, (
      f"Task {task_id} has critic enable_corruption=True, expected False"
    )


def test_play_training_observation_structure_match(
  play_task_pairs: list[tuple[str, str]],
) -> None:
  """Play and training configs should have matching observation structure."""
  for training_task_id, play_task_id in play_task_pairs:
    training_cfg = load_env_cfg(training_task_id)
    play_cfg = load_env_cfg(play_task_id)

    # Same observation groups.
    assert set(training_cfg.observations.keys()) == set(play_cfg.observations.keys()), (
      f"Observation groups mismatch between {training_task_id} and {play_task_id}"
    )

    # Same observation terms within each group.
    for obs_group_name in training_cfg.observations:
      training_terms = set(training_cfg.observations[obs_group_name].terms.keys())
      play_terms = set(play_cfg.observations[obs_group_name].terms.keys())

      assert training_terms == play_terms, (
        f"Observation terms mismatch in group '{obs_group_name}' "
        f"between {training_task_id} and {play_task_id}"
      )


def test_play_training_action_structure_match(
  play_task_pairs: list[tuple[str, str]],
) -> None:
  """Play and training configs should have matching action structure."""
  for training_task_id, play_task_id in play_task_pairs:
    training_cfg = load_env_cfg(training_task_id)
    play_cfg = load_env_cfg(play_task_id)

    assert set(training_cfg.actions.keys()) == set(play_cfg.actions.keys()), (
      f"Action structure mismatch between {training_task_id} and {play_task_id}"
    )


def test_play_mode_disables_push_robot(all_task_ids: list[str]) -> None:
  """Play/Demo mode tasks should disable push_robot event."""
  play_demo_tasks = [
    t for t in all_task_ids if t.endswith("-Play") or t.endswith("-Demo")
  ]

  for task_id in play_demo_tasks:
    cfg = load_env_cfg(task_id)
    assert "push_robot" not in cfg.events, (
      f"Play/Demo task {task_id} has push_robot event, expected it to be removed"
    )
