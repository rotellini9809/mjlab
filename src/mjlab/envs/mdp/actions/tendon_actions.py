from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


##
# Configuration classes.
##


@dataclass(kw_only=True)
class TendonActionCfg(ActionTermCfg):
  actuator_names: tuple[str, ...]
  """Tuple of tendon names or regex expressions to map action to."""
  scale: float | dict[str, float] = 1.0
  """Scale factor (float or dict of regex expressions)."""
  offset: float | dict[str, float] = 0.0
  """Offset factor (float or dict of regex expressions)."""
  preserve_order: bool = False
  """Whether to preserve tendon name order in action output."""


@dataclass(kw_only=True)
class TendonLengthActionCfg(TendonActionCfg):
  def build(self, env: ManagerBasedRlEnv) -> TendonLengthAction:
    return TendonLengthAction(self, env)


@dataclass(kw_only=True)
class TendonVelocityActionCfg(TendonActionCfg):
  def build(self, env: ManagerBasedRlEnv) -> TendonVelocityAction:
    return TendonVelocityAction(self, env)


@dataclass(kw_only=True)
class TendonEffortActionCfg(TendonActionCfg):
  def build(self, env: ManagerBasedRlEnv) -> TendonEffortAction:
    return TendonEffortAction(self, env)


##
# Action term implementations.
##


class TendonAction(ActionTerm):
  """Base class for tendon actions."""

  _asset: Entity

  def __init__(self, cfg: TendonActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    # Find tendons using entity's find_tendons method
    tendon_ids, tendon_names = self._asset.find_tendons(
      cfg.actuator_names, preserve_order=cfg.preserve_order
    )
    self._tendon_ids = torch.tensor(tendon_ids, device=self.device, dtype=torch.long)
    self._tendon_names = tendon_names

    self._num_tendons = len(tendon_ids)
    self._action_dim = len(tendon_ids)

    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    # Handle scale parameter
    if isinstance(cfg.scale, (float, int)):
      self._scale = float(cfg.scale)
    elif isinstance(cfg.scale, dict):
      self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.scale, self._tendon_names
      )
      self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported scale type: {type(cfg.scale)}."
        " Supported types are float and dict."
      )

    # Handle offset parameter
    if isinstance(cfg.offset, (float, int)):
      self._offset = float(cfg.offset)
    elif isinstance(cfg.offset, dict):
      self._offset = torch.zeros_like(self._raw_actions)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.offset, self._tendon_names
      )
      self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported offset type: {type(cfg.offset)}."
        " Supported types are float and dict."
      )

  # Properties.

  @property
  def scale(self) -> torch.Tensor | float:
    return self._scale

  @property
  def offset(self) -> torch.Tensor | float:
    return self._offset

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def action_dim(self) -> int:
    return self._action_dim

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0


class TendonLengthAction(TendonAction):
  def __init__(self, cfg: TendonLengthActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._asset.set_tendon_len_target(
      self._processed_actions, tendon_ids=self._tendon_ids
    )


class TendonVelocityAction(TendonAction):
  def __init__(self, cfg: TendonVelocityActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._asset.set_tendon_vel_target(
      self._processed_actions, tendon_ids=self._tendon_ids
    )


class TendonEffortAction(TendonAction):
  def __init__(self, cfg: TendonEffortActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._asset.set_tendon_effort_target(
      self._processed_actions, tendon_ids=self._tendon_ids
    )
