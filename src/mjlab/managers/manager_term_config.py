from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, ParamSpec, TypeVar

import torch

from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.command_manager import CommandTerm
from mjlab.utils.noise.noise_cfg import NoiseCfg, NoiseModelCfg

P = ParamSpec("P")
T = TypeVar("T")


def term(term_cls: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
  return field(default_factory=lambda: term_cls(*args, **kwargs))


@dataclass
class ManagerTermBaseCfg:
  func: Any
  params: dict[str, Any] = field(default_factory=lambda: {})


##
# Action manager.
##


@dataclass(kw_only=True)
class ActionTermCfg:
  """Configuration for an action term."""

  class_type: type[ActionTerm]
  asset_name: str
  clip: dict[str, tuple] | None = None


##
# Command manager.
##


@dataclass(kw_only=True)
class CommandTermCfg:
  """Configuration for a command generator term."""

  class_type: type[CommandTerm]
  resampling_time_range: tuple[float, float]
  debug_vis: bool = False


##
# Curriculum manager.
##


@dataclass(kw_only=True)
class CurriculumTermCfg(ManagerTermBaseCfg):
  pass


##
# Event manager.
##


EventMode = Literal["startup", "reset", "interval"]


@dataclass(kw_only=True)
class EventTermCfg(ManagerTermBaseCfg):
  """Configuration for an event term."""

  mode: EventMode
  interval_range_s: tuple[float, float] | None = None
  is_global_time: bool = False
  min_step_count_between_reset: int = 0


##
# Observation manager.
##


@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
  """Configuration for an observation term.

  Processing pipeline: compute → noise → clip → scale → delay → history.
  Delay models sensor latency. History provides temporal context. Both are optional
  and can be combined.
  """

  noise: NoiseCfg | NoiseModelCfg | None = None
  """Noise model to apply to the observation. Defaults to None."""
  clip: tuple[float, float] | None = None
  """Range (min, max) to clip the observation values. Defaults to None."""
  scale: tuple[float, ...] | float | torch.Tensor | None = None
  """Scaling factor(s) to multiply the observation by. Defaults to None."""
  delay_min_lag: int = 0
  """Minimum lag (in steps) for delayed observations. Lag sampled uniformly from
  [min_lag, max_lag]. Convert to ms: lag * (1000 / control_hz). Default 0 (no delay)."""
  delay_max_lag: int = 0
  """Maximum lag (in steps) for delayed observations. Use min=max for constant delay.
  Default 0 (no delay)."""
  delay_per_env: bool = True
  """If True, each environment samples its own lag. If False, all environments share
  the same lag at each step. Default True."""
  delay_hold_prob: float = 0.0
  """Probability of reusing the previous lag instead of resampling. Useful for
  temporally correlated latency patterns. Default 0.0 (resample every update)."""
  delay_update_period: int = 0
  """Resample lag every N steps (models multi-rate sensors). If 0, update every step.
  E.g., update_period=2 at 50Hz control ≈ 30Hz sensor refresh. Default 0."""
  delay_per_env_phase: bool = True
  """If True and update_period > 0, stagger update timing across envs to avoid
  synchronized resampling. Default True."""
  history_length: int = 0
  """Number of past observations to keep in history. 0 = no history. Default 0."""
  flatten_history_dim: bool = True
  """Whether to flatten the history dimension into observation. Default True."""


@dataclass
class ObservationGroupCfg:
  """Configuration for an observation group."""

  concatenate_terms: bool = True
  concatenate_dim: int = -1
  enable_corruption: bool = False
  history_length: int | None = None
  flatten_history_dim: bool = True


##
# Reward manager.
##


@dataclass(kw_only=True)
class RewardTermCfg(ManagerTermBaseCfg):
  """Configuration for a reward term."""

  func: Any
  weight: float


##
# Termination manager.
##


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
  """Configuration for a termination term."""

  time_out: bool = False
  """Whether the term contributes towards episodic timeouts. Defaults to False."""
