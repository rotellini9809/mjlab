"""Tests for tendon action functionality."""

from pathlib import Path

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import BuiltinMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import (
  TendonEffortActionCfg,
  TendonLengthActionCfg,
  TendonVelocityActionCfg,
)
from mjlab.managers.action_manager import ActionManager
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def fixtures_dir():
  """Path to test fixtures directory."""
  return Path(__file__).parent / "fixtures"


@pytest.fixture
def finger_entity(fixtures_dir, device):
  """Create tendon finger entity."""
  xml_path = fixtures_dir / "tendon_finger.xml"

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file(str(xml_path)),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=("finger_tendon",),
          transmission_type=TransmissionType.TENDON,
          effort_limit=10.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=4, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity


@pytest.fixture
def mock_env(finger_entity, device):
  """Create a mock environment for testing."""
  from unittest.mock import Mock

  env = Mock(spec=ManagerBasedRlEnv)
  env.num_envs = 4
  env.device = device
  # Make scene subscriptable like a dict.
  env.scene = {"finger": finger_entity}
  return env


def test_tendon_length_action_initialization(mock_env):
  """Test that TendonLengthAction initializes correctly."""
  cfg = TendonLengthActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
  )

  action = cfg.build(mock_env)

  assert action.action_dim == 1, "Should have 1 tendon"
  assert action.raw_action.shape == (4, 1), "Should match (num_envs, action_dim)"


def test_tendon_velocity_action_initialization(mock_env):
  """Test that TendonVelocityAction initializes correctly."""
  cfg = TendonVelocityActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
  )

  action = cfg.build(mock_env)

  assert action.action_dim == 1
  assert action.raw_action.shape == (4, 1)


def test_tendon_effort_action_initialization(mock_env):
  """Test that TendonEffortAction initializes correctly."""
  cfg = TendonEffortActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
  )

  action = cfg.build(mock_env)

  assert action.action_dim == 1
  assert action.raw_action.shape == (4, 1)


def test_tendon_action_finds_tendons_by_name(mock_env):
  """Test that tendon actions can find tendons by name."""
  cfg = TendonLengthActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
  )

  action = cfg.build(mock_env)

  assert len(action.target_names) == 1
  assert "finger_tendon" in action.target_names


def test_tendon_action_finds_tendons_by_regex(mock_env):
  """Test that tendon actions can find tendons by regex."""
  cfg = TendonLengthActionCfg(
    entity_name="finger",
    actuator_names=(".*tendon",),
  )

  action = cfg.build(mock_env)

  assert len(action.target_names) == 1
  assert "finger_tendon" in action.target_names


def test_tendon_action_with_scalar_scale_and_offset(mock_env, device):
  """Test tendon action processing with scalar scale and offset."""
  cfg = TendonLengthActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
    scale=2.0,
    offset=0.5,
  )

  action = cfg.build(mock_env)

  raw_action = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
  action.process_actions(raw_action)

  # Verify: processed = raw * scale + offset.
  expected = raw_action * 2.0 + 0.5
  assert torch.allclose(action._processed_actions, expected)


def test_tendon_action_with_dict_scale_and_offset(mock_env, device):
  """Test tendon action processing with dict-based scale and offset."""
  cfg = TendonLengthActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
    scale={"finger_tendon": 3.0},
    offset={"finger_tendon": 1.0},
  )

  action = cfg.build(mock_env)

  raw_action = torch.tensor([[2.0], [3.0], [4.0], [5.0]], device=device)
  action.process_actions(raw_action)

  expected = raw_action * 3.0 + 1.0
  assert torch.allclose(action._processed_actions, expected)


def test_tendon_length_action_sets_target(mock_env, finger_entity, device):
  """Test that TendonLengthAction sets the correct target on entity."""
  cfg = TendonLengthActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
  )

  action = cfg.build(mock_env)

  target_length = torch.tensor([[0.05], [0.06], [0.07], [0.08]], device=device)
  action.process_actions(target_length)
  action.apply_actions()

  assert torch.allclose(finger_entity.data.tendon_len_target[:, 0:1], target_length)


def test_tendon_velocity_action_sets_target(mock_env, finger_entity, device):
  """Test that TendonVelocityAction sets the correct target on entity."""
  cfg = TendonVelocityActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
  )

  action = cfg.build(mock_env)

  target_velocity = torch.tensor([[0.1], [0.2], [0.3], [0.4]], device=device)
  action.process_actions(target_velocity)
  action.apply_actions()

  assert torch.allclose(finger_entity.data.tendon_vel_target[:, 0:1], target_velocity)


def test_tendon_effort_action_sets_target(mock_env, finger_entity, device):
  """Test that TendonEffortAction sets the correct target on entity."""
  cfg = TendonEffortActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
  )

  action = cfg.build(mock_env)

  target_effort = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
  action.process_actions(target_effort)
  action.apply_actions()

  assert torch.allclose(finger_entity.data.tendon_effort_target[:, 0:1], target_effort)


def test_tendon_action_reset(mock_env, device):
  """Test that tendon action reset clears raw actions."""
  cfg = TendonLengthActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
  )

  action = cfg.build(mock_env)

  action.process_actions(torch.ones(4, 1, device=device))
  action.reset(env_ids=torch.tensor([0, 2], device=device))

  assert torch.all(action.raw_action[0] == 0.0)
  assert torch.all(action.raw_action[2] == 0.0)
  assert torch.all(action.raw_action[1] == 1.0)
  assert torch.all(action.raw_action[3] == 1.0)


def test_tendon_action_in_action_manager(mock_env, device):
  """Test tendon actions work within ActionManager."""
  from mjlab.managers.manager_term_config import ActionTermCfg

  action_cfg: dict[str, ActionTermCfg] = {
    "tendon_control": TendonLengthActionCfg(
      entity_name="finger",
      actuator_names=("finger_tendon",),
    )
  }

  manager = ActionManager(action_cfg, mock_env)

  assert manager.action.shape == (4, 1), "Action shape should match total action dim"

  action_input = torch.tensor([[0.5], [0.6], [0.7], [0.8]], device=device)
  manager.process_action(action_input)

  assert torch.allclose(manager.action, action_input)
