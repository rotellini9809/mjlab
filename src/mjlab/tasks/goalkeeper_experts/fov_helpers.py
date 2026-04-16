from __future__ import annotations

import math
from collections.abc import Callable

import torch


def compute_fov_visibility(
  rel_xy: torch.Tensor,
  forward_xy: torch.Tensor,
  *,
  fov_active: bool,
  half_angle_deg: float,
  eps: float = 1.0e-6,
) -> torch.Tensor:
  """Return whether each target lies inside the configured planar FOV."""
  if not fov_active:
    return torch.ones(rel_xy.shape[0], device=rel_xy.device, dtype=torch.bool)

  dist = torch.linalg.norm(rel_xy, dim=1)
  ball_dir_xy = rel_xy / dist.unsqueeze(1).clamp_min(float(eps))
  cos_half_fov = math.cos(math.radians(float(half_angle_deg)))
  dot = torch.sum(forward_xy * ball_dir_xy, dim=1)
  return torch.where(
    dist > float(eps),
    dot >= cos_half_fov,
    torch.ones_like(dist, dtype=torch.bool),
  )


def update_last_seen_ball_state(
  env,
  *,
  visible: torch.Tensor,
  rel_pos_xyz: torch.Tensor,
  rel_vel_xyz: torch.Tensor,
  key_prefix: str,
  get_float_state_buffer: Callable[..., torch.Tensor],
  get_bool_state_buffer: Callable[..., torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Update and return last-seen XY position, XY velocity, and stale time in seconds."""
  last_pos_x = get_float_state_buffer(env, f"{key_prefix}::last_pos_x")
  last_pos_y = get_float_state_buffer(env, f"{key_prefix}::last_pos_y")
  last_vel_x = get_float_state_buffer(env, f"{key_prefix}::last_vel_x")
  last_vel_y = get_float_state_buffer(env, f"{key_prefix}::last_vel_y")
  last_seen_time = get_float_state_buffer(env, f"{key_prefix}::last_seen_time")
  has_last_seen = get_bool_state_buffer(env, f"{key_prefix}::has_last_seen")

  reset_mask = env.episode_length_buf == 0
  if reset_mask.any():
    last_pos_x[reset_mask] = 0.0
    last_pos_y[reset_mask] = 0.0
    last_vel_x[reset_mask] = 0.0
    last_vel_y[reset_mask] = 0.0
    last_seen_time[reset_mask] = 0.0
    has_last_seen[reset_mask] = False

  current_time_s = env.episode_length_buf.to(torch.float32) * float(env.step_dt)
  if visible.any():
    last_pos_x[visible] = rel_pos_xyz[visible, 0]
    last_pos_y[visible] = rel_pos_xyz[visible, 1]
    last_vel_x[visible] = rel_vel_xyz[visible, 0]
    last_vel_y[visible] = rel_vel_xyz[visible, 1]
    last_seen_time[visible] = current_time_s[visible]
    has_last_seen[visible] = True

  last_seen_pos_xy = torch.stack((last_pos_x, last_pos_y), dim=1)
  last_seen_vel_xy = torch.stack((last_vel_x, last_vel_y), dim=1)
  last_seen_secs = torch.where(
    has_last_seen,
    torch.clamp(current_time_s - last_seen_time, min=0.0),
    current_time_s,
  )
  return last_seen_pos_xy, last_seen_vel_xy, last_seen_secs
