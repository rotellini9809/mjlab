"""Delay buffer for stochastically delayed observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mjlab.utils.buffers import CircularBuffer


class DelayBuffer:
  """Serve stochastically delayed (stale) observations from a rolling history.

  Wraps a :class:`CircularBuffer` and returns delayed frames according to a
  lag policy (uniform sampling over a range, optional "hold" probability, and
  optional multi-rate lag updates with per-environment phase offsets).

  Shapes:
    - Internal storage (via `CircularBuffer`): `(max_lag + 1, batch_size, ...)`
    - Output of `compute()`: `(batch_size, ...)`

  Lag update policy:
    * If `update_period == 0`: a new lag l is considered every step (subject to
      `hold_prob`).
    * If `update_period > 0`: a lag l is refreshed only on steps where
      `(step_count[env] + phase_offset[env]) % update_period == 0`; otherwise the
      previous lag is kept.
    * With probability `hold_prob`, the previous lag is kept even when a refresh
      would occur.
    * If `per_env=False`, a single sampled lag is shared by all environments.

  Reset semantics:
    `reset(batch_ids=...)` clears selected rows in the inner `CircularBuffer`,
    sets their lag and counters to zero, and on the next `append()` those rows
    receive backfill (their first new value is copied across their full history).
    Until that next append, `compute()` for those rows returns zeros.

  Args:
    min_lag (int, optional): Minimum lag (inclusive). Must be >= 0. Defaults to 0.
    max_lag (int, optional): Maximum lag (inclusive). Must be >= `min_lag`.
      Defaults to 3.
    batch_size (int, optional): Number of parallel environments (leading
      dimension of inputs). Defaults to 1.
    device (str, optional): Torch device for storage and RNG (e.g., `"cpu"`,
      `"cuda"`). Defaults to `"cpu"`.
    per_env (bool, optional): If True, sample a separate lag per environment;
      otherwise sample one lag and share it across environments. Defaults to True.
    hold_prob (float, optional): Probability in `[0.0, 1.0]` to keep the previous
      lag when an update would occur. Defaults to 0.0.
    update_period (int, optional): If > 0, refresh lags every `N` steps per
      environment; if 0, consider updating every step. Defaults to 0.
    per_env_phase (bool, optional): If True and `update_period > 0`, each
      environment uses a different phase offset in `[0, update_period)`, causing
      staggered refresh steps. Defaults to True.
    generator (torch.Generator | None, optional): Optional RNG for sampling
      lags. Defaults to None.

  Notes:
    * When the buffer contains fewer than `max_lag + 1` frames, sampled lags are
      clamped to available history to ensure valid reads (oldest → newest).
    * All operations are vectorized across environments and remain on `device`.
    * Storage complexity is
      `O(batch_size x (max_lag + 1) x prod(observation_shape))`.

  Examples:
    Constant delay (lag = 2):
      >>> buf = DelayBuffer(min_lag=2, max_lag=2, batch_size=4, device="cpu")
      >>> buf.append(obs)                # obs.shape == (4, ...)
      >>> delayed = buf.compute()        # delayed[t] = obs[t-2]

    Stochastic delay (uniform 0-3):
      >>> buf = DelayBuffer(min_lag=0, max_lag=3, batch_size=4, device="cpu")
      >>> buf.append(obs)
      >>> delayed = buf.compute()        # per-env lag sampled in {0,1,2,3}
  """

  def __init__(
    self,
    min_lag: int = 0,
    max_lag: int = 3,
    batch_size: int = 1,
    device: str = "cpu",
    per_env: bool = True,
    hold_prob: float = 0.0,
    update_period: int = 0,
    per_env_phase: bool = True,
    generator: torch.Generator | None = None,
  ) -> None:
    if min_lag < 0:
      raise ValueError(f"min_lag must be >= 0, got {min_lag}")
    if max_lag < min_lag:
      raise ValueError(f"max_lag ({max_lag}) must be >= min_lag ({min_lag})")
    if not 0.0 <= hold_prob <= 1.0:
      raise ValueError(f"hold_prob must be in [0, 1], got {hold_prob}")
    if update_period < 0:
      raise ValueError(f"update_period must be >= 0, got {update_period}")

    self.min_lag = min_lag
    self.max_lag = max_lag
    self.batch_size = batch_size
    self.device = device
    self.per_env = per_env
    self.hold_prob = hold_prob
    self.update_period = update_period
    self.per_env_phase = per_env_phase
    self.generator = generator

    buffer_size = max_lag + 1 if max_lag > 0 else 1
    self._buffer = CircularBuffer(
      max_len=buffer_size, batch_size=batch_size, device=device
    )
    self._current_lags = torch.zeros(batch_size, dtype=torch.long, device=device)
    self._step_count = torch.zeros(batch_size, dtype=torch.long, device=device)

    if update_period > 0 and per_env_phase:
      self._phase_offsets = torch.randint(
        0,
        update_period,
        (batch_size,),
        dtype=torch.long,
        device=device,
        generator=generator,
      )
    else:
      self._phase_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)

  @property
  def is_initialized(self) -> bool:
    """Check if buffer has been initialized with at least one append."""
    return self._buffer.is_initialized

  @property
  def current_lags(self) -> torch.Tensor:
    """Current lag per environment. Shape: (batch_size,)."""
    return self._current_lags

  def reset(self, batch_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    """Reset specified environments to initial state.

    Args:
      batch_ids: Batch indices to reset, or None to reset all.
    """
    ids = slice(None) if batch_ids is None else batch_ids
    self._buffer.reset(batch_ids=batch_ids)
    self._current_lags[ids] = 0
    self._step_count[ids] = 0

    if self.update_period > 0 and self.per_env_phase:
      if batch_ids is None:
        self._phase_offsets = torch.randint(
          0,
          self.update_period,
          (self.batch_size,),
          dtype=torch.long,
          device=self.device,
          generator=self.generator,
        )
      else:
        new_phases = torch.randint(
          0,
          self.update_period,
          (self.batch_size,),
          dtype=torch.long,
          device=self.device,
          generator=self.generator,
        )
        self._phase_offsets[ids] = new_phases[ids]

  def append(self, data: torch.Tensor) -> None:
    """Append new observation to buffer.

    Args:
      data: Observation tensor of shape (batch_size, ...).
    """
    self._buffer.append(data)

  def compute(self) -> torch.Tensor:
    """Compute delayed observation for current step.

    Returns:
      Delayed observation with shape (batch_size, ...).
    """
    if not self.is_initialized:
      raise RuntimeError("Buffer not initialized. Call append() first.")

    self._update_lags()

    # Clamp lags to valid range [0, buffer_length - 1].
    # Buffer may not be full yet (e.g., only 2 frames but sampled lag=3).
    valid_lags = torch.minimum(self._current_lags, self._buffer.current_length - 1)
    valid_lags = torch.maximum(valid_lags, torch.zeros_like(valid_lags))

    return self._buffer[valid_lags]

  def _update_lags(self) -> None:
    """Update current lags according to configured policy."""
    if self.update_period > 0:
      phase_adjusted_count = (self._step_count + self._phase_offsets) % (
        self.update_period
      )
      should_update = phase_adjusted_count == 0
    else:
      should_update = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

    if torch.any(should_update):
      new_lags = self._sample_lags(should_update)
      self._current_lags = torch.where(should_update, new_lags, self._current_lags)

    self._step_count += 1

  def _sample_lags(self, mask: torch.Tensor) -> torch.Tensor:
    """Sample new lags for specified environments.

    Args:
      mask: Boolean mask of shape (batch_size,) indicating which envs to sample.

    Returns:
      New lags with shape (batch_size,).
    """
    if self.per_env:
      candidate_lags = torch.randint(
        self.min_lag,
        self.max_lag + 1,
        (self.batch_size,),
        dtype=torch.long,
        device=self.device,
        generator=self.generator,
      )
    else:
      shared_lag = torch.randint(
        self.min_lag,
        self.max_lag + 1,
        (1,),
        dtype=torch.long,
        device=self.device,
        generator=self.generator,
      )
      candidate_lags = shared_lag.expand(self.batch_size)

    if self.hold_prob > 0.0:
      should_sample = (
        torch.rand(self.batch_size, device=self.device, generator=self.generator)
        >= self.hold_prob
      )
    else:
      should_sample = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

    update_mask = mask & should_sample
    return torch.where(update_mask, candidate_lags, self._current_lags)
