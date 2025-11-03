"""Tests for MDP events functionality."""

from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.envs.mdp import events
from mjlab.managers.scene_entity_config import SceneEntityCfg


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


def test_reset_joints_by_offset(device):
  """Test that reset_joints_by_offset applies offsets and respects joint limits."""
  env = Mock()
  env.num_envs = 2
  env.device = device

  mock_entity = Mock()
  mock_entity.data.default_joint_pos = torch.zeros((2, 3), device=device)
  mock_entity.data.default_joint_vel = torch.zeros((2, 3), device=device)
  mock_entity.data.soft_joint_pos_limits = torch.tensor(
    [
      [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
      [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
    ],
    device=device,
  )
  mock_entity.write_joint_state_to_sim = Mock()

  env.scene = {"robot": mock_entity}

  # Test normal offset application.
  events.reset_joints_by_offset(
    env,
    torch.tensor([0], device=device),
    position_range=(0.3, 0.3),
    velocity_range=(0.2, 0.2),
    asset_cfg=SceneEntityCfg("robot", joint_ids=slice(None)),
  )

  call_args = mock_entity.write_joint_state_to_sim.call_args
  joint_pos, joint_vel = call_args[0][0], call_args[0][1]
  assert torch.allclose(joint_pos, torch.ones_like(joint_pos) * 0.3)
  assert torch.allclose(joint_vel, torch.ones_like(joint_vel) * 0.2)

  # Test clamping when offset exceeds limits.
  events.reset_joints_by_offset(
    env,
    torch.tensor([1], device=device),
    position_range=(1.0, 1.0),
    velocity_range=(0.0, 0.0),
    asset_cfg=SceneEntityCfg("robot", joint_ids=slice(None)),
  )

  call_args = mock_entity.write_joint_state_to_sim.call_args
  joint_pos = call_args[0][0]
  assert torch.allclose(joint_pos, torch.ones_like(joint_pos) * 0.5)
