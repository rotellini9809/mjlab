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
class SiteActionCfg(ActionTermCfg):
  actuator_names: tuple[str, ...]
  """Tuple of site names or regex expressions to map action to."""
  scale: float | dict[str, float] = 1.0
  """Scale factor (float or dict of regex expressions)."""
  offset: float | dict[str, float] = 0.0
  """Offset factor (float or dict of regex expressions)."""
  preserve_order: bool = False
  """Whether to preserve site name order in action output."""


@dataclass(kw_only=True)
class SiteEffortActionCfg(SiteActionCfg):
  def build(self, env: ManagerBasedRlEnv) -> SiteEffortAction:
    return SiteEffortAction(self, env)


##
# Action term implementations.
##


class SiteAction(ActionTerm):
  """Base class for site actions.

  Note: Sites only support effort (force) control. Sites are coordinate frames
  attached to bodies, not actuatable DOFs. The only control is applying force
  at the site location.
  """

  _asset: Entity

  def __init__(self, cfg: SiteActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    # Find sites using entity's find_sites method
    site_ids, site_names = self._asset.find_sites(
      cfg.actuator_names, preserve_order=cfg.preserve_order
    )
    self._site_ids = torch.tensor(site_ids, device=self.device, dtype=torch.long)
    self._site_names = site_names

    self._num_sites = len(site_ids)
    self._action_dim = len(site_ids)

    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    # Handle scale parameter
    if isinstance(cfg.scale, (float, int)):
      self._scale = float(cfg.scale)
    elif isinstance(cfg.scale, dict):
      self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.scale, self._site_names
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
        cfg.offset, self._site_names
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


class SiteEffortAction(SiteAction):
  """Apply effort (force) at site locations.

  This action applies a scalar force magnitude at each specified site.
  The direction and application of the force depends on the actuator
  configuration in MuJoCo.
  """

  def __init__(self, cfg: SiteEffortActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._asset.set_site_effort_target(self._processed_actions, site_ids=self._site_ids)
