from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.any(sensor.data.found, dim=-1)


def object_out_of_reach(
  env: ManagerBasedRlEnv, object_name: str, max_distance: float
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  obj: Entity = env.scene[object_name]
  robot_pos = robot.data.root_link_pos_w
  obj_pos = obj.data.root_link_pos_w
  horizontal_distance = torch.norm(obj_pos[:, :2] - robot_pos[:, :2], dim=-1)
  return horizontal_distance > max_distance
