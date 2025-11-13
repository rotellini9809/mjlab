from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.actuator.actuator import Actuator
  from mjlab.entity.data import EntityData


class BuiltinActuatorGroup:
  """Groups builtin actuators for batch processing.

  Builtin actuators (position, velocity, motor) just pass through target values
  from entity data to control signals. This class pre-computes the mappings and
  enables direct writes without per-actuator overhead.
  """

  def __init__(self) -> None:
    self._position_ctrl_ids: torch.Tensor | None = None
    self._position_joint_ids: torch.Tensor | None = None
    self._velocity_ctrl_ids: torch.Tensor | None = None
    self._velocity_joint_ids: torch.Tensor | None = None
    self._motor_ctrl_ids: torch.Tensor | None = None
    self._motor_joint_ids: torch.Tensor | None = None

  def add_actuators(self, actuators: list[Actuator]) -> None:
    """Register builtin actuators and pre-compute their mappings.

    Args:
      actuators: List of initialized actuators to process.
    """
    from mjlab.actuator.builtin_actuator import (
      BuiltinMotorActuator,
      BuiltinPositionActuator,
      BuiltinVelocityActuator,
    )

    position_ctrl_ids = []
    position_joint_ids = []
    velocity_ctrl_ids = []
    velocity_joint_ids = []
    motor_ctrl_ids = []
    motor_joint_ids = []

    for act in actuators:
      if isinstance(act, BuiltinPositionActuator):
        position_ctrl_ids.append(act.ctrl_ids)
        position_joint_ids.append(act.joint_ids)
      elif isinstance(act, BuiltinVelocityActuator):
        velocity_ctrl_ids.append(act.ctrl_ids)
        velocity_joint_ids.append(act.joint_ids)
      elif isinstance(act, BuiltinMotorActuator):
        motor_ctrl_ids.append(act.ctrl_ids)
        motor_joint_ids.append(act.joint_ids)

    if position_ctrl_ids:
      self._position_ctrl_ids = torch.cat(position_ctrl_ids)
      self._position_joint_ids = torch.cat(position_joint_ids)
    if velocity_ctrl_ids:
      self._velocity_ctrl_ids = torch.cat(velocity_ctrl_ids)
      self._velocity_joint_ids = torch.cat(velocity_joint_ids)
    if motor_ctrl_ids:
      self._motor_ctrl_ids = torch.cat(motor_ctrl_ids)
      self._motor_joint_ids = torch.cat(motor_joint_ids)

  def apply_controls(self, data: EntityData) -> None:
    """Write builtin actuator controls directly to simulation data.

    Args:
      data: Entity data containing targets and control arrays.
    """
    if self._position_ctrl_ids is not None:
      data.write_ctrl(
        data.joint_pos_target[:, self._position_joint_ids],
        self._position_ctrl_ids,
      )
    if self._velocity_ctrl_ids is not None:
      data.write_ctrl(
        data.joint_vel_target[:, self._velocity_joint_ids],
        self._velocity_ctrl_ids,
      )
    if self._motor_ctrl_ids is not None:
      data.write_ctrl(
        data.joint_effort_target[:, self._motor_joint_ids],
        self._motor_ctrl_ids,
      )

  def filter_custom_actuators(self, actuators: list[Actuator]) -> list[Actuator]:
    """Filter out builtin actuators, returning only custom ones.

    Args:
      actuators: List of all actuators.

    Returns:
      List containing only custom (non-builtin) actuators.
    """
    from mjlab.actuator.builtin_actuator import (
      BuiltinMotorActuator,
      BuiltinPositionActuator,
      BuiltinVelocityActuator,
    )

    return [
      act
      for act in actuators
      if not isinstance(
        act, (BuiltinPositionActuator, BuiltinVelocityActuator, BuiltinMotorActuator)
      )
    ]
