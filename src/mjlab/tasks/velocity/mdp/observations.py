from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  forces_flat = forces.flatten(start_dim=1)  # [B, N*3]
  # Apply symlog transformation to handle large dynamic range
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def heading_error(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Return the heading error wrapped to [-pi, pi]."""
  cmd_cfg = env.command_manager.get_term_cfg(command_name)
  if not isinstance(cmd_cfg, UniformVelocityCommandCfg) or not cmd_cfg.heading_command:
    return torch.zeros((env.num_envs, 1), device=env.device)
  cmd_term = env.command_manager.get_term(command_name)
  assert isinstance(cmd_term, UniformVelocityCommand)
  return cmd_term.heading_error.unsqueeze(-1)
