"""Tests for site action functionality."""

from pathlib import Path

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import BuiltinMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import SiteEffortActionCfg
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
def quadcopter_entity(fixtures_dir, device):
  """Create quadcopter entity."""
  xml_path = fixtures_dir / "quadcopter.xml"

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file(str(xml_path)),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=("rotor_.*",),
          transmission_type=TransmissionType.SITE,
          effort_limit=20.0,
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
def mock_env(quadcopter_entity, device):
  """Create a mock environment for testing."""
  from unittest.mock import Mock

  env = Mock(spec=ManagerBasedRlEnv)
  env.num_envs = 4
  env.device = device
  # Make scene subscriptable like a dict.
  env.scene = {"drone": quadcopter_entity}
  return env


def test_site_effort_action_initialization(mock_env):
  """Test that SiteEffortAction initializes correctly."""
  cfg = SiteEffortActionCfg(
    entity_name="drone",
    actuator_names=("rotor_.*",),
  )

  action = cfg.build(mock_env)

  assert action.action_dim == 4, "Should have 4 rotor sites"
  assert action.raw_action.shape == (4, 4), "Should match (num_envs, action_dim)"


def test_site_action_finds_sites_by_name(mock_env):
  """Test that site actions can find sites by exact name."""
  cfg = SiteEffortActionCfg(
    entity_name="drone",
    actuator_names=("rotor_fl", "rotor_fr"),
  )

  action = cfg.build(mock_env)

  assert len(action.target_names) == 2
  assert "rotor_fl" in action.target_names
  assert "rotor_fr" in action.target_names


def test_site_action_finds_sites_by_regex(mock_env):
  """Test that site actions can find sites by regex pattern."""
  cfg = SiteEffortActionCfg(
    entity_name="drone",
    actuator_names=("rotor_.*",),
  )

  action = cfg.build(mock_env)

  assert len(action.target_names) == 4
  assert all(
    name in action.target_names
    for name in ["rotor_fl", "rotor_fr", "rotor_rl", "rotor_rr"]
  )


def test_site_action_preserve_order(mock_env):
  """Test that preserve_order flag respects site name order."""
  cfg_sorted = SiteEffortActionCfg(
    entity_name="drone",
    actuator_names=("rotor_rr", "rotor_fl", "rotor_rl", "rotor_fr"),
    preserve_order=False,
  )
  action_sorted = cfg_sorted.build(mock_env)

  cfg_preserved = SiteEffortActionCfg(
    entity_name="drone",
    actuator_names=("rotor_rr", "rotor_fl", "rotor_rl", "rotor_fr"),
    preserve_order=True,
  )
  action_preserved = cfg_preserved.build(mock_env)

  assert action_sorted.target_names != action_preserved.target_names
  assert action_preserved.target_names == [
    "rotor_rr",
    "rotor_fl",
    "rotor_rl",
    "rotor_fr",
  ]


def test_site_action_with_scalar_scale_and_offset(mock_env, device):
  """Test site action processing with scalar scale and offset."""
  cfg = SiteEffortActionCfg(
    entity_name="drone",
    actuator_names=("rotor_.*",),
    scale=2.0,
    offset=1.0,
  )

  action = cfg.build(mock_env)

  raw_action = torch.ones(4, 4, device=device)
  action.process_actions(raw_action)

  # Verify: processed = raw * scale + offset.
  expected = raw_action * 2.0 + 1.0
  assert torch.allclose(action._processed_actions, expected)


def test_site_action_with_dict_scale_and_offset(mock_env, device):
  """Test site action processing with dict-based scale and offset."""
  cfg = SiteEffortActionCfg(
    entity_name="drone",
    actuator_names=("rotor_.*",),
    scale={"rotor_fl": 2.0, "rotor_fr": 3.0},  # Different scales for different rotors
    offset={"rotor_rl": 0.5},  # Offset for rear-left rotor
  )

  action = cfg.build(mock_env)

  raw_action = torch.ones(4, 4, device=device)
  action.process_actions(raw_action)

  # rotor_fl should have scale 2.0, rotor_fr scale 3.0, rotor_rl offset 0.5.
  processed = action._processed_actions

  fl_idx = action.target_names.index("rotor_fl")
  fr_idx = action.target_names.index("rotor_fr")
  rl_idx = action.target_names.index("rotor_rl")

  assert torch.allclose(processed[:, fl_idx], torch.tensor(2.0, device=device))
  assert torch.allclose(processed[:, fr_idx], torch.tensor(3.0, device=device))
  assert torch.allclose(processed[:, rl_idx], torch.tensor(1.5, device=device))


def test_site_effort_action_sets_target(mock_env, quadcopter_entity, device):
  """Test that SiteEffortAction sets the correct target on entity."""
  cfg = SiteEffortActionCfg(
    entity_name="drone",
    actuator_names=("rotor_.*",),
  )

  action = cfg.build(mock_env)

  target_effort = torch.tensor(
    [
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
    ],
    device=device,
  )
  action.process_actions(target_effort)
  action.apply_actions()

  assert torch.allclose(quadcopter_entity.data.site_effort_target, target_effort)


def test_site_action_reset(mock_env, device):
  """Test that site action reset clears raw actions."""
  cfg = SiteEffortActionCfg(
    entity_name="drone",
    actuator_names=("rotor_.*",),
  )

  action = cfg.build(mock_env)

  action.process_actions(torch.ones(4, 4, device=device))
  action.reset(env_ids=torch.tensor([1, 3], device=device))

  assert torch.all(action.raw_action[1] == 0.0)
  assert torch.all(action.raw_action[3] == 0.0)
  assert torch.all(action.raw_action[0] == 1.0)
  assert torch.all(action.raw_action[2] == 1.0)


def test_site_action_in_action_manager(mock_env, device):
  """Test site actions work within ActionManager."""
  from mjlab.managers.manager_term_config import ActionTermCfg

  action_cfg: dict[str, ActionTermCfg] = {
    "thrust_control": SiteEffortActionCfg(
      entity_name="drone",
      actuator_names=("rotor_.*",),
    )
  }

  manager = ActionManager(action_cfg, mock_env)

  assert manager.action.shape == (4, 4), "Action shape should match total action dim"

  action_input = torch.tensor(
    [
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
    ],
    device=device,
  )
  manager.process_action(action_input)

  assert torch.allclose(manager.action, action_input)
