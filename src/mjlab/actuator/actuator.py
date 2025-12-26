"""Base actuator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity


class TransmissionType(str, Enum):
  """Transmission types for actuators."""

  JOINT = "joint"
  TENDON = "tendon"
  SITE = "site"


@dataclass(kw_only=True)
class ActuatorCfg(ABC):
  target_names_expr: tuple[str, ...]
  """Targets that are part of this actuator group.

  Can be a tuple of names or tuple of regex expressions.
  Interpreted based on transmission_type (joint/tendon/site).
  """

  transmission_type: TransmissionType = TransmissionType.JOINT
  """Transmission type. Defaults to JOINT."""

  armature: float = 0.0
  """Reflected rotor inertia."""

  frictionloss: float = 0.0
  """Friction loss force limit.

  Applies a constant friction force opposing motion, independent of load or velocity.
  Also known as dry friction or load-independent friction.
  """

  def __post_init__(self) -> None:
    assert self.armature >= 0.0, "armature must be non-negative."
    assert self.frictionloss >= 0.0, "frictionloss must be non-negative."
    if self.transmission_type == TransmissionType.SITE:
      if self.armature > 0.0 or self.frictionloss > 0.0:
        raise ValueError(
          f"{self.__class__.__name__}: armature and frictionloss are not supported for "
          "SITE transmission type."
        )

  @abstractmethod
  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> Actuator:
    """Build actuator instance.

    Args:
      entity: Entity this actuator belongs to.
      target_ids: Local target indices (for indexing entity arrays).
      target_names: Target names corresponding to target_ids.

    Returns:
      Actuator instance.
    """
    raise NotImplementedError


@dataclass
class ActuatorCmd:
  """High-level actuator command with targets and current state.

  Passed to actuator's `compute()` method to generate low-level control signals.
  All tensors have shape (num_envs, num_joints).
  """

  position_target: torch.Tensor
  """Desired joint positions."""
  velocity_target: torch.Tensor
  """Desired joint velocities."""
  effort_target: torch.Tensor
  """Feedforward effort."""
  joint_pos: torch.Tensor
  """Current joint positions."""
  joint_vel: torch.Tensor
  """Current joint velocities."""


class Actuator(ABC):
  """Base actuator interface."""

  def __init__(
    self,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    self.entity = entity
    self._target_ids_list = target_ids
    self._target_names = target_names
    self._target_ids: torch.Tensor | None = None
    self._ctrl_ids: torch.Tensor | None = None
    self._mjs_actuators: list[mujoco.MjsActuator] = []

  @property
  def target_ids(self) -> torch.Tensor:
    """Local indices of targets controlled by this actuator."""
    assert self._target_ids is not None
    return self._target_ids

  @property
  def target_names(self) -> list[str]:
    """Names of targets controlled by this actuator."""
    return self._target_names

  @property
  def ctrl_ids(self) -> torch.Tensor:
    """Global indices of control inputs for this actuator."""
    assert self._ctrl_ids is not None
    return self._ctrl_ids

  @abstractmethod
  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    """Edit the MjSpec to add actuators and configure joints.

    This is called during entity construction, before the model is compiled.

    Args:
      spec: The entity's MjSpec to edit.
      joint_names: Names of joints controlled by this actuator.
    """
    raise NotImplementedError

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    """Initialize the actuator after model compilation.

    This is called after the MjSpec is compiled into an MjModel.

    Args:
      mj_model: The compiled MuJoCo model.
      model: The compiled mjwarp model.
      data: The mjwarp data arrays.
      device: Device for tensor operations (e.g., "cuda", "cpu").
    """
    del mj_model, model, data  # Unused.
    self._target_ids = torch.tensor(
      self._target_ids_list, dtype=torch.long, device=device
    )
    ctrl_ids_list = [act.id for act in self._mjs_actuators]
    self._ctrl_ids = torch.tensor(ctrl_ids_list, dtype=torch.long, device=device)

  @abstractmethod
  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    """Compute low-level actuator control signal from high-level commands.

    Args:
      cmd: High-level actuator command.

    Returns:
      Control signal tensor of shape (num_envs, num_actuators).
    """
    raise NotImplementedError

  # Optional methods.

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset actuator state for specified environments.

    Base implementation does nothing. Override in subclasses that maintain
    internal state.

    Args:
      env_ids: Environment indices to reset. If None, reset all environments.
    """
    del env_ids  # Unused.

  def update(self, dt: float) -> None:
    """Update actuator state after a simulation step.

    Base implementation does nothing. Override in subclasses that need
    per-step updates.

    Args:
      dt: Time step in seconds.
    """
    del dt  # Unused.
