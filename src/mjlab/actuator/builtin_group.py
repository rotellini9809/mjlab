from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import (
  BuiltinMotorActuator,
  BuiltinPositionActuator,
  BuiltinVelocityActuator,
)

if TYPE_CHECKING:
  from mjlab.actuator.actuator import Actuator
  from mjlab.entity.data import EntityData

BUILTIN_TYPES = {BuiltinMotorActuator, BuiltinPositionActuator, BuiltinVelocityActuator}


@dataclass(frozen=True)
class BuiltinActuatorGroup:
  """Groups builtin actuators for batch processing.

  Builtin actuators (position, velocity, motor) just pass through target values
  from entity data to control signals. This class pre-computes the mappings and
  enables direct writes without per-actuator overhead.
  """

  # Map from (BuiltinActuator type, transmission_type) to (target_ids, ctrl_ids).
  _index_groups: dict[tuple[type, TransmissionType], tuple[torch.Tensor, torch.Tensor]]

  @staticmethod
  def process(
    actuators: list[Actuator],
  ) -> tuple[BuiltinActuatorGroup, tuple[Actuator, ...]]:
    """Register builtin actuators and pre-compute their mappings.

    Args:
      actuators: List of initialized actuators to process.

    Returns:
      A tuple containing:
        - BuiltinActuatorGroup with pre-computed mappings.
        - List of custom (non-builtin) actuators.
    """

    builtin_groups: dict[tuple[type, TransmissionType], list[Actuator]] = {}
    custom_actuators: list[Actuator] = []

    # Group actuators by (type, transmission_type).
    for act in actuators:
      if type(act) in BUILTIN_TYPES:
        # All builtin actuators have a cfg attribute with transmission_type.
        builtin_act = act  # type: ignore[assignment]
        key: tuple[type, TransmissionType] = (
          type(act),
          builtin_act.cfg.transmission_type,  # type: ignore[attr-defined]
        )
        builtin_groups.setdefault(key, []).append(act)
      else:
        custom_actuators.append(act)

    # Return stacked indices for each (actuator_type, transmission_type) group.
    index_groups: dict[
      tuple[type, TransmissionType], tuple[torch.Tensor, torch.Tensor]
    ] = {
      k: (
        torch.cat([act.joint_ids for act in v], dim=0),
        torch.cat([act.ctrl_ids for act in v], dim=0),
      )
      for k, v in builtin_groups.items()
    }
    return BuiltinActuatorGroup(index_groups), tuple(custom_actuators)

  def apply_controls(self, data: EntityData) -> None:
    """Write builtin actuator controls directly to simulation data.

    Args:
      data: Entity data containing targets and control arrays.
    """
    for (actuator_type, transmission_type), index_group in self._index_groups.items():
      target_ids, ctrl_ids = index_group

      # Select target tensor based on actuator type and transmission type.
      if transmission_type == TransmissionType.JOINT:
        if actuator_type == BuiltinPositionActuator:
          target_tensor = data.joint_pos_target
        elif actuator_type == BuiltinVelocityActuator:
          target_tensor = data.joint_vel_target
        elif actuator_type == BuiltinMotorActuator:
          target_tensor = data.joint_effort_target
        else:
          raise ValueError(f"Unknown actuator type: {actuator_type}")
      elif transmission_type == TransmissionType.TENDON:
        if actuator_type == BuiltinPositionActuator:
          target_tensor = data.tendon_len_target
        elif actuator_type == BuiltinVelocityActuator:
          target_tensor = data.tendon_vel_target
        elif actuator_type == BuiltinMotorActuator:
          target_tensor = data.tendon_effort_target
        else:
          raise ValueError(f"Unknown actuator type: {actuator_type}")
      elif transmission_type == TransmissionType.SITE:
        # Sites only support effort control.
        if actuator_type == BuiltinMotorActuator:
          target_tensor = data.site_effort_target
        else:
          raise ValueError(
            f"Site transmission only supports motor (effort) actuators, "
            f"not {actuator_type.__name__}"
          )
      else:
        raise ValueError(f"Unknown transmission type: {transmission_type}")

      data.write_ctrl(target_tensor[:, target_ids], ctrl_ids)
