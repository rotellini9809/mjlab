"""Tests for task configuration integrity and correctness."""

import pytest

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import ObservationGroupCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

# Fixtures


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


def test_play_mode_episode_length(play_task_pairs: list[tuple[str, str]]) -> None:
  """Play mode tasks should have effectively infinite episode length."""
  for _, play_task_id in play_task_pairs:
    cfg = load_env_cfg(play_task_id)
    assert cfg.episode_length_s >= 1e9, (
      f"Play task {play_task_id} episode_length_s={cfg.episode_length_s}, expected >= 1e9"
    )


def test_play_mode_observation_corruption_disabled(
  play_task_pairs: list[tuple[str, str]],
) -> None:
  """Play mode tasks should have observation corruption disabled for policy."""
  for _, play_task_id in play_task_pairs:
    cfg = load_env_cfg(play_task_id)

    assert "policy" in cfg.observations, (
      f"Play task {play_task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["policy"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Play task {play_task_id} policy observation is not ObservationGroupCfg"
    )

    assert not policy_obs.enable_corruption, (
      f"Play task {play_task_id} has enable_corruption=True, expected False"
    )


def test_training_mode_observation_corruption_enabled(
  play_task_pairs: list[tuple[str, str]],
) -> None:
  """Training mode tasks should have observation corruption enabled."""
  for training_task_id, _ in play_task_pairs:
    cfg = load_env_cfg(training_task_id)

    assert "policy" in cfg.observations, (
      f"Training task {training_task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["policy"]
    assert policy_obs.enable_corruption, (
      f"Training task {training_task_id} has enable_corruption=False, expected True"
    )


def test_policy_observation_group_exists(all_task_ids: list[str]) -> None:
  """All tasks should have a 'policy' observation group."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    assert "policy" in cfg.observations, (
      f"Task {task_id} missing 'policy' observation group"
    )
    assert isinstance(cfg.observations["policy"], ObservationGroupCfg), (
      f"Task {task_id} 'policy' observation is not ObservationGroupCfg"
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


def test_no_none_observation_configs(all_task_ids: list[str]) -> None:
  """Observation configs should not have None values where unexpected."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    for obs_group_name, obs_group_cfg in cfg.observations.items():
      assert obs_group_cfg is not None, (
        f"Task {task_id} observation group '{obs_group_name}' is None"
      )

      assert obs_group_cfg.terms is not None, (
        f"Task {task_id} observation group '{obs_group_name}' has None terms"
      )

      for term_name, term_cfg in obs_group_cfg.terms.items():
        assert term_cfg is not None, (
          f"Task {task_id} observation term '{obs_group_name}.{term_name}' is None"
        )
