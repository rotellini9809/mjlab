from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import torch

from mjlab.asset_zoo.robocup_assets.ball import get_robocup_ball_cfg
from mjlab.entity import Entity, EntityCfg
from mjlab.envs.mdp import *  # noqa: F401,F403
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.motor_controller_stage1.latent_action import (
  default_motor_obs_layout,
  MotorLatentActionCfg,
  motor_last_decoded_action,
)
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
)
from mjlab.envs.mdp.rewards import (
  action_rate_l2 as _base_action_rate_l2,
  joint_pos_limits as _base_joint_pos_limits,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _sample_uniform_range(
  low: float,
  high: float,
  num: int,
  device: str,
) -> torch.Tensor:
  return torch.rand(num, device=device) * (high - low) + low


def _sample_union_range(
  intervals: tuple[tuple[float, float], ...],
  num: int,
  device: str,
) -> torch.Tensor:
  """Sample uniformly within one interval chosen uniformly from an explicit union."""
  if len(intervals) == 0:
    raise ValueError("interval union must contain at least one interval")
  if len(intervals) == 1:
    lo, hi = intervals[0]
    return _sample_uniform_range(lo, hi, num, device)

  interval_idx = torch.randint(0, len(intervals), (num,), device=device)
  samples = torch.empty(num, device=device)
  for idx, (lo, hi) in enumerate(intervals):
    mask = interval_idx == idx
    if mask.any():
      samples[mask] = _sample_uniform_range(lo, hi, int(mask.sum().item()), device)
  return samples


def _normalize_xy(vec_xy: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
  norm = torch.linalg.norm(vec_xy, dim=1, keepdim=True).clamp_min(eps)
  return vec_xy / norm


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
  qw, qx, qy, qz = (
    quat_wxyz[:, 0],
    quat_wxyz[:, 1],
    quat_wxyz[:, 2],
    quat_wxyz[:, 3],
  )
  return torch.atan2(
    2.0 * (qw * qz + qx * qy),
    1.0 - 2.0 * (qy * qy + qz * qz),
  )


def _pitch_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
  qw, qx, qy, qz = (
    quat_wxyz[:, 0],
    quat_wxyz[:, 1],
    quat_wxyz[:, 2],
    quat_wxyz[:, 3],
  )
  sin_pitch = 2.0 * (qw * qy - qz * qx)
  return torch.asin(sin_pitch.clamp(-1.0, 1.0))


def _roll_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
  qw, qx, qy, qz = (
    quat_wxyz[:, 0],
    quat_wxyz[:, 1],
    quat_wxyz[:, 2],
    quat_wxyz[:, 3],
  )
  return torch.atan2(
    2.0 * (qw * qx + qy * qz),
    1.0 - 2.0 * (qx * qx + qy * qy),
  )


def _yaw_error_from_heading(
  source_pos_w_xy: torch.Tensor,
  source_yaw: torch.Tensor,
  target_pos_w: torch.Tensor,
) -> torch.Tensor:
  target_xy = target_pos_w[:, :2] - source_pos_w_xy
  target_dir_xy = _normalize_xy(target_xy)
  forward_xy = torch.stack([torch.cos(source_yaw), torch.sin(source_yaw)], dim=1)

  dot = torch.sum(forward_xy * target_dir_xy, dim=1).clamp(-1.0, 1.0)
  det = forward_xy[:, 0] * target_dir_xy[:, 1] - forward_xy[:, 1] * target_dir_xy[:, 0]
  return torch.atan2(det, dot)


def _compute_torso_yaw_error(
  robot: Entity,
  target_pos_w: torch.Tensor,
) -> torch.Tensor:
  """Signed yaw-only error between torso heading and target direction.

  In the current T1_23 model, torso frame aligns with root link (Trunk).
  """
  torso_pos_w_xy = robot.data.root_link_pos_w[:, :2]
  torso_yaw = _yaw_from_quat_wxyz(robot.data.root_link_quat_w)
  return _yaw_error_from_heading(torso_pos_w_xy, torso_yaw, target_pos_w)


def _compute_waist_yaw_error(
  env,
  robot: Entity,
  target_pos_w: torch.Tensor,
  waist_body_name: str = r"(?i)^waist$",
) -> torch.Tensor:
  """Signed yaw-only error between anatomical waist/pelvis heading and target."""
  waist_idx = _resolve_single_body_index_cached(env, robot, waist_body_name)
  waist_pos_w_xy = robot.data.body_link_pos_w[:, waist_idx, :2]
  waist_yaw = _yaw_from_quat_wxyz(robot.data.body_link_quat_w[:, waist_idx, :])
  return _yaw_error_from_heading(waist_pos_w_xy, waist_yaw, target_pos_w)


def _wrap_angle_pi(angle: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def _outside_area_violation(
  pos_xy: torch.Tensor,
  bounds: tuple[float, float, float, float],
) -> torch.Tensor:
  x_min, x_max, y_min, y_max = bounds
  x = pos_xy[:, 0]
  y = pos_xy[:, 1]
  x_low = (x_min - x).clamp_min(0.0)
  x_high = (x - x_max).clamp_min(0.0)
  y_low = (y_min - y).clamp_min(0.0)
  y_high = (y - y_max).clamp_min(0.0)
  return x_low + x_high + y_low + y_high


def _world_to_env_local_xy(env, pos_w_xy: torch.Tensor) -> torch.Tensor:
  return pos_w_xy - env.scene.env_origins[:, :2]


def _resolve_body_index_pair_cached(
  env,
  robot: Entity,
  body_name_a: str,
  body_name_b: str,
) -> tuple[int, int]:
  """Resolve and cache body indices for repeated reward computation."""
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_e1_body_index_pair_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  key = (id(robot), body_name_a, body_name_b)
  if key not in cache:
    ids, names = robot.find_bodies((body_name_a, body_name_b), preserve_order=True)
    if len(ids) != 2:
      raise ValueError(
        "Could not resolve exactly two foot bodies for stance_ortho_to_ball_reward. "
        f"Got names={names} for patterns=({body_name_a}, {body_name_b})."
      )
    cache[key] = (int(ids[0]), int(ids[1]))

  return cache[key]


def _resolve_single_body_index_cached(
  env,
  robot: Entity,
  body_name_pattern: str,
) -> int:
  """Resolve and cache a single body index for repeated reward computation."""
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_e1_body_index_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  key = (id(robot), body_name_pattern)
  if key not in cache:
    ids, names = robot.find_bodies(body_name_pattern, preserve_order=True)
    if len(ids) != 1:
      raise ValueError(
        "Could not resolve exactly one body. "
        f"Got names={names} for pattern=({body_name_pattern})."
      )
    cache[key] = int(ids[0])

  return cache[key]


def _get_waist_yaw_progress_prev_error_buffer(
  env,
  robot: Entity,
  waist_body_name: str,
) -> torch.Tensor:
  """Get/create per-env previous waist yaw error magnitude buffer."""
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_e1_waist_yaw_progress_prev_error_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  key = (id(robot), waist_body_name)
  prev = cache.get(key)
  if prev is None or prev.shape[0] != int(env.num_envs) or prev.device != robot.data.root_link_pos_w.device:
    prev = torch.zeros(env.num_envs, device=robot.data.root_link_pos_w.device)
    cache[key] = prev
  return prev


def _get_float_state_buffer(
  env,
  name: str,
  num_envs: int,
  device: torch.device | str,
) -> torch.Tensor:
  """Get/create per-env float buffer used by stateful reward shaping."""
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_e1_reward_float_state_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  buf = cache.get(name)
  if (
    buf is None
    or buf.shape != (num_envs,)
    or buf.device != torch.device(device)
    or not torch.is_floating_point(buf)
  ):
    buf = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    cache[name] = buf
  return buf


def _get_log_dict(env) -> dict[str, torch.Tensor] | None:
  extras = getattr(env, "extras", None)
  if extras is None:
    return None
  log = extras.get("log")
  if log is None:
    log = {}
    extras["log"] = log
  return log


def _posture_score_components(
  projected_gravity_b: torch.Tensor,
  roll_band: float,
  roll_sigma: float,
  pitch_target: float,
  pitch_band: float,
  pitch_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute smooth anisotropic posture score from body-frame projected gravity.

  Body-frame convention used across mjlab:
  - x: sagittal (forward/backward)
  - y: lateral (left/right)
  """
  sagittal = projected_gravity_b[:, 0]
  lateral = projected_gravity_b[:, 1]

  roll_error = torch.relu(torch.abs(lateral) - float(roll_band))
  roll_score = torch.exp(
    -torch.square(roll_error) / max(float(roll_sigma) * float(roll_sigma), 1.0e-6)
  )

  pitch_error = torch.relu(torch.abs(sagittal - float(pitch_target)) - float(pitch_band))
  pitch_score = torch.exp(
    -torch.square(pitch_error) / max(float(pitch_sigma) * float(pitch_sigma), 1.0e-6)
  )

  posture_score = roll_score * pitch_score
  return posture_score, roll_score, pitch_score, lateral, sagittal


def _yaw_reward_posture_score(
  projected_gravity_b: torch.Tensor,
  tilt_target: float,
  tilt_band: float,
  tilt_sigma: float,
) -> torch.Tensor:
  """Posture score used by E1 yaw shaping gates.

  Reuses E1's anisotropic posture convention while letting the yaw helpers
  configure the sagittal target and shared deadband/sigma.
  """
  band = max(float(tilt_band), 0.0)
  sigma = max(float(tilt_sigma), 1.0e-6)
  posture_score, _, _, _, _ = _posture_score_components(
    projected_gravity_b,
    roll_band=band,
    roll_sigma=sigma,
    pitch_target=tilt_target,
    pitch_band=band,
    pitch_sigma=sigma,
  )
  return posture_score.clamp_min(torch.finfo(projected_gravity_b.dtype).tiny)


def _standing_gate(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  h_low: float = 0.40,
  h_good: float = 0.62,
  roll_band: float = 0.1,
  roll_sigma: float = 0.12,
  pitch_target: float = 0.25,
  pitch_band: float = 0.20,
  pitch_sigma: float = 0.30,
) -> torch.Tensor:
  """Soft standing-quality gate used by legacy reward terms."""
  robot: Entity = env.scene[asset_cfg.name]
  posture_score, _, _, _, _ = _posture_score_components(
    robot.data.projected_gravity_b,
    roll_band=roll_band,
    roll_sigma=roll_sigma,
    pitch_target=pitch_target,
    pitch_band=pitch_band,
    pitch_sigma=pitch_sigma,
  )
  base_height = robot.data.root_link_pos_w[:, 2]
  height_score = torch.clamp(
    (base_height - float(h_low)) / max(float(h_good) - float(h_low), 1.0e-6),
    min=0.0,
    max=1.0,
  )
  stand_score = posture_score * height_score
  gate = torch.square(stand_score)

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_stand_score_mean"] = torch.mean(stand_score)
    log["Metrics/e1_stand_gate_mean"] = torch.mean(gate)
    log["Metrics/e1_height_score_mean"] = torch.mean(height_score)
    log["Metrics/e1_base_height_mean"] = torch.mean(base_height)

  return gate


def _apply_standing_gate_if_enabled(
  raw: torch.Tensor,
  env,
  asset_cfg: SceneEntityCfg,
  apply_standing_gate: bool,
) -> torch.Tensor:
  if not apply_standing_gate:
    return raw
  return raw * _standing_gate(env, asset_cfg=asset_cfg)


def _home_point_world_xy(
  env,
  command_name: str,
) -> torch.Tensor:
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  return env.scene.env_origins[:, :2] + torch.tensor(
    [command.cfg.home_point_x, command.cfg.home_point_y],
    device=env.device,
    dtype=torch.float32,
  )


def _mezzaluna_point_world_xy(
  env,
  command_name: str = "set_square",
) -> torch.Tensor:
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]

  center_xy = env.scene.env_origins[:, :2] + torch.tensor(
    [command.cfg.mezzaluna_center_x, command.cfg.mezzaluna_center_y],
    device=env.device,
    dtype=torch.float32,
  )
  ball_xy = ball.data.root_link_pos_w[:, :2]
  dir_xy = ball_xy - center_xy

  a = max(
    float(command.cfg.mezzaluna_center_x)
    - float(command.cfg.mezzaluna_apex_x),
    1.0e-6,
  )
  b = max(float(command.cfg.mezzaluna_half_width_y), 1.0e-6)

  dx = dir_xy[:, 0]
  dy = dir_xy[:, 1]
  denom = torch.sqrt(
    torch.square(dx / a) + torch.square(dy / b)
  ).clamp_min(1.0e-6)

  sign_x = torch.sign(dx)
  sign_y = torch.where(dy >= 0.0, torch.ones_like(dy), -torch.ones_like(dy))
  ray_sign = torch.where(torch.abs(dx) > 1.0e-6, -sign_x, sign_y)
  t = ray_sign / denom

  point_xy = center_xy + t.unsqueeze(1) * dir_xy

  use_apex = torch.linalg.norm(dir_xy, dim=1) <= 1.0e-6
  if use_apex.any():
    point_xy[use_apex, 0] = float(command.cfg.mezzaluna_apex_x) + env.scene.env_origins[
      use_apex, 0
    ]
    point_xy[use_apex, 1] = float(command.cfg.mezzaluna_center_y) + env.scene.env_origins[
      use_apex, 1
    ]

  return point_xy


def _reward_target_world_xy(
  env,
  command_name: str = "set_square",
) -> torch.Tensor:
  return _mezzaluna_point_world_xy(env, command_name)


def _stance_center_xy(
  env,
  robot: Entity,
  left_foot_body_name: str,
  right_foot_body_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  left_idx, right_idx = _resolve_body_index_pair_cached(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  body_pos_w = robot.data.body_link_pos_w
  left_xy = body_pos_w[:, left_idx, :2]
  right_xy = body_pos_w[:, right_idx, :2]
  center_xy = 0.5 * (left_xy + right_xy)
  return center_xy, left_xy, right_xy


def _reward_active_mask(
  env,
  command_name: str = "set_square",
) -> torch.Tensor:
  del command_name
  return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)


def _reward_active_float_mask(
  env,
  command_name: str = "set_square",
) -> torch.Tensor:
  return _reward_active_mask(env, command_name).to(torch.float32)


def _reward_step_count(
  env,
  command_name: str = "set_square",
) -> torch.Tensor:
  del command_name
  return env.episode_length_buf.clone()


def _apply_reward_active_mask(
  reward: torch.Tensor,
  env,
  command_name: str = "set_square",
) -> torch.Tensor:
  return reward * _reward_active_mask(env, command_name).to(reward.dtype)


def _alignment_home_ramp(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str | None = None,
  right_foot_body_name: str | None = None,
  x_scale: float = 0.30,
  y_scale: float = 0.25,
  align_floor: float = 0.2,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  if left_foot_body_name is None:
    left_foot_body_name = command.cfg.stance_left_foot_body_name
  if right_foot_body_name is None:
    right_foot_body_name = command.cfg.stance_right_foot_body_name

  center_xy, _, _ = _stance_center_xy(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  home_xy = _reward_target_world_xy(env, command_name)
  home_x_err = torch.abs(home_xy[:, 0] - center_xy[:, 0])
  home_y_err = torch.abs(home_xy[:, 1] - center_xy[:, 1])
  x_scale_safe = max(float(x_scale), 1.0e-6)
  y_scale_safe = max(float(y_scale), 1.0e-6)
  alpha_home_x = torch.clamp(1.0 - home_x_err / x_scale_safe, min=0.0, max=1.0)
  alpha_home = torch.clamp(1.0 - home_y_err / y_scale_safe, min=0.0, max=1.0)
  alpha_home = alpha_home_x * alpha_home
  align_floor = float(align_floor)
  align_mult = align_floor + (1.0 - align_floor) * alpha_home

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_align_home_ramp_mean"] = torch.mean(align_mult)
    log["Metrics/e1_home_x_err_for_align_mean"] = torch.mean(home_x_err)
    log["Metrics/e1_home_y_err_for_align_mean"] = torch.mean(home_y_err)

  return align_mult


def _stance_ortho_score(
  env,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  left_foot_body_name: str,
  right_foot_body_name: str,
  eps: float = 1.0e-6,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]

  center_xy, left_xy, right_xy = _stance_center_xy(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  stance_vec_xy = right_xy - left_xy
  stance_norm = torch.linalg.norm(stance_vec_xy, dim=1, keepdim=True).clamp_min(float(eps))
  stance_dir_xy = stance_vec_xy / stance_norm
  ball_vec_xy = ball.data.root_link_pos_w[:, :2] - center_xy
  ball_norm = torch.linalg.norm(ball_vec_xy, dim=1, keepdim=True).clamp_min(float(eps))
  ball_dir_xy = ball_vec_xy / ball_norm
  dot = torch.sum(stance_dir_xy * ball_dir_xy, dim=1).clamp(-1.0, 1.0)
  return torch.abs(dot)


def get_target_ball_cfg() -> EntityCfg:
  """Return the physical colliding RoboCup ball for E1."""
  return get_robocup_ball_cfg()


@dataclass(frozen=True)
class IntervalUnionCfg:
  """Explicit 1D union of intervals used to encode holes without rejection."""

  intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SetSquareResetStageCfg:
  """Manual reset curriculum stage for E1 set&square."""

  keeper_spawn_x_range: tuple[float, float]
  keeper_spawn_y_range: IntervalUnionCfg
  spawn_yaw_offset_range: tuple[float, float]
  target_forward_range: tuple[float, float]
  target_lateral_range: IntervalUnionCfg
  launcher_mode_probs: tuple[float, float, float]


@dataclass(kw_only=True)
class SetSquareCommandCfg(CommandTermCfg):
  """Command term for E1 keeper reset + physical ball launcher."""

  entity_name: str = "robot"
  ball_entity_name: str = "soccer_ball"
  ball_curb_sensor_name: str | None = None

  # Motor-controller command vector dimension (from Stage-1 obs layout).
  command_dim: int = 46

  # Keeper spawn (world XY, before adding env origin).
  keeper_spawn_x_range: tuple[float, float]
  keeper_spawn_y_range: tuple[float, float]

  # Safe keeper area bounds: (x_min, x_max, y_min, y_max).
  keeper_area_bounds: tuple[float, float, float, float]
  hard_area_margin: float = 0.8
  mezzaluna_center_x: float = 7.0
  mezzaluna_center_y: float = 0.0
  mezzaluna_apex_x: float = 6.15
  mezzaluna_half_width_y: float = 1.55

  # Ball spawn sampling relative to keeper spawn.
  target_forward_range: tuple[float, float] = (1.0, 2.5)
  target_lateral_range: tuple[float, float] = (-1.2, 1.2)
  target_height_min: float = 0.11
  # Exponential scale (meters): smaller -> more mass near ground.
  target_height_exp_scale: float = 0.06
  target_height_max: float | None = None
  # Temporary debug override: force constant z for target ball.
  debug_force_target_ground_z: bool = False
  debug_target_ground_z: float = 0.11

  # Kick sampler for E1-only in-play ball behavior.
  dead_ball_prob: float = 0.35
  lateral_roll_prob: float = 0.45
  dead_ball_tiny_drift_prob: float = 0.20
  dead_ball_drift_speed_range: tuple[float, float] = (0.02, 0.10)

  kick_speed_range: tuple[float, float] = (0.4, 1.6)
  kick_angle_noise_deg: float = 20.0

  dribble_num_taps_range: tuple[int, int] = (2, 5)
  dribble_tap_time_range: tuple[float, float] = (0.6, 1.8)
  dribble_tap_interval_range: tuple[float, float] = (0.2, 0.8)
  dribble_tap_speed_range: tuple[float, float] = (0.2, 0.6)
  rebound_relaunch_enabled: bool = True
  rebound_only_side_walls: bool = True
  rebound_delay_range_s: tuple[float, float] = (0.5, 1.0)
  rebound_speed_range: tuple[float, float] = (0.8, 1.8)
  rebound_angle_noise_deg: float = 60.0
  rebound_inset_m: float = 0.15
  rebound_max_events: int = 1
  field_half_width_y: float = 4.5

  # Anti-shot clamp: limit component toward defended goal.
  # If goal is at +x, keep vx <= max_toward_goal_speed.
  goal_toward_positive_x: bool = True
  max_toward_goal_speed: float = 0.25

  # Reset curriculum hook.
  p_ready: float = 0.0

  # Keeper home point (local env coords, before adding env origin).
  home_point_x: float = 6.75
  home_point_y: float = 0.0

  # Manual reset curriculum.
  curriculum_stage: int = 4
  curriculum_stages: tuple[SetSquareResetStageCfg, ...] = ()
  nominal_keeper_facing_yaw: float = math.pi

  # Keep target fixed for full episode.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False
  stance_left_foot_body_name: str = r"^left_foot_link$"
  stance_right_foot_body_name: str = r"^right_foot_link$"
  stance_ortho_w_min: float = 0.10
  stance_ortho_d_min: float = 0.35

  @dataclass
  class VizCfg:
    z_offset: float = 0.65
    desired_length: float = 0.8
    actual_length: float = 0.8
    width: float = 0.015
    desired_color: tuple[float, float, float, float] = (0.2, 0.2, 0.9, 0.75)
    actual_color: tuple[float, float, float, float] = (0.0, 0.9, 0.6, 0.75)
    stance_axis_length: float = 0.55
    stance_axis_width: float = 0.012
    stance_z_offset: float = 0.03
    foot_line_radius: float = 0.008
    foot_line_color: tuple[float, float, float, float] = (0.85, 0.85, 0.85, 0.70)
    ball_dir_color: tuple[float, float, float, float] = (0.20, 0.65, 1.0, 0.85)
    stance_target_color: tuple[float, float, float, float] = (0.95, 0.85, 0.20, 0.85)
    stance_good_color: tuple[float, float, float, float] = (0.10, 0.85, 0.20, 0.90)
    stance_bad_color: tuple[float, float, float, float] = (0.95, 0.20, 0.20, 0.90)
    stance_neutral_color: tuple[float, float, float, float] = (0.55, 0.55, 0.55, 0.70)
    stance_cue_radius: float = 0.035
    home_point_color: tuple[float, float, float, float] = (1.0, 0.75, 0.1, 0.95)
    home_point_radius: float = 0.10
    home_point_height: float = 0.01
    home_point_z: float = 0.0
    mezzaluna_point_color: tuple[float, float, float, float] = (0.85, 0.20, 0.95, 0.95)
    mezzaluna_point_radius: float = 0.05
    mezzaluna_point_z: float = 0.03

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env):
    return SetSquareCommand(self, env)


class SetSquareCommand(CommandTerm):
  cfg: SetSquareCommandCfg

  def __init__(self, cfg: SetSquareCommandCfg, env):
    super().__init__(cfg, env)
    self._robot: Entity = env.scene[cfg.entity_name]
    self._ball: Entity = env.scene[cfg.ball_entity_name]

    self._command = torch.zeros(env.num_envs, cfg.command_dim, device=self.device)
    self._spawn_pos_w = torch.zeros(env.num_envs, 3, device=self.device)
    self._target_pos_w = torch.zeros(env.num_envs, 3, device=self.device)
    self._kick_time_s = torch.zeros(env.num_envs, device=self.device)
    self._kick_applied = torch.ones(env.num_envs, device=self.device, dtype=torch.bool)
    self._kick_vel_w = torch.zeros(env.num_envs, 3, device=self.device)
    self._tap_enabled = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
    self._next_tap_time_s = torch.zeros(env.num_envs, device=self.device)
    self._remaining_taps = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
    self._last_push_dir_xy = torch.zeros(env.num_envs, 2, device=self.device)
    self._launcher_mode = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
    self._rebound_pending = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
    self._rebound_time_s = torch.zeros(env.num_envs, device=self.device)
    self._rebound_used_count = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
    self._rebound_wall_side_sign = torch.zeros(env.num_envs, device=self.device)

    if len(self.cfg.curriculum_stages) == 0:
      raise ValueError("SetSquareCommandCfg.curriculum_stages must contain at least one stage.")
    if not (1 <= int(self.cfg.curriculum_stage) <= len(self.cfg.curriculum_stages)):
      raise ValueError(
        f"curriculum_stage must be within [1, {len(self.cfg.curriculum_stages)}], "
        f"got {self.cfg.curriculum_stage}."
      )

    self._status_markdown = None
    self._gui_get_env_idx = None

    self.metrics["yaw_error_abs"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["torso_roll"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["torso_pitch"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["torso_height"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["target_distance_xy"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["stance_width"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["outside_keeper_area"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["ball_speed_xy"] = torch.zeros(env.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def spawn_pos_w(self) -> torch.Tensor:
    return self._spawn_pos_w

  @property
  def target_pos_w(self) -> torch.Tensor:
    return self._target_pos_w

  @property
  def keeper_area_bounds(self) -> tuple[float, float, float, float]:
    return self.cfg.keeper_area_bounds

  @property
  def hard_keeper_area_bounds(self) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = self.cfg.keeper_area_bounds
    m = self.cfg.hard_area_margin
    return (x_min - m, x_max + m, y_min - m, y_max + m)

  @property
  def curriculum_stage(self) -> int:
    return int(self.cfg.curriculum_stage)

  @property
  def stage_cfg(self) -> SetSquareResetStageCfg:
    return self.cfg.curriculum_stages[self.curriculum_stage - 1]

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: "Callable[[], int]",
  ) -> None:
    del name
    self._gui_get_env_idx = get_env_idx
    with server.gui.add_folder("SetSquare"):
      self._status_markdown = server.gui.add_markdown("")
    self._update_status_markdown()

  def compute(self, dt: float) -> None:
    super().compute(dt)
    self._update_status_markdown()

  _LAUNCHER_MODE_NAMES = ["dead", "lateral", "dribble"]

  def _update_status_markdown(self) -> None:
    if self._status_markdown is None or self._gui_get_env_idx is None:
      return

    env_idx = int(self._gui_get_env_idx())
    if env_idx < 0 or env_idx >= self.num_envs:
      self._status_markdown.content = "**env idx:** n/a"
      return

    mode_id = int(self._launcher_mode[env_idx].item())
    mode_name = self._LAUNCHER_MODE_NAMES[mode_id] if mode_id < len(self._LAUNCHER_MODE_NAMES) else "?"

    yaw_err = float(self.metrics["yaw_error_abs"][env_idx].item())
    torso_roll_deg = torch.rad2deg(self.metrics["torso_roll"][env_idx]).item()
    torso_pitch_deg = torch.rad2deg(self.metrics["torso_pitch"][env_idx]).item()
    torso_height = float(self.metrics["torso_height"][env_idx].item())
    ball_dist = float(self.metrics["target_distance_xy"][env_idx].item())
    stance_width = float(self.metrics["stance_width"][env_idx].item())
    ball_speed = float(self.metrics["ball_speed_xy"][env_idx].item())
    outside = float(self.metrics["outside_keeper_area"][env_idx].item())

    self._status_markdown.content = (
      f"**Stage:** {self.curriculum_stage}\n\n"
      f"**Mode:** {mode_name}\n\n"
      f"**Torso roll:** {torso_roll_deg:.1f} deg\n\n"
      f"**Torso pitch:** {torso_pitch_deg:.1f} deg\n\n"
      f"**Torso height:** {torso_height:.3f} m\n\n"
      f"**Yaw err:** {yaw_err:.3f} rad\n\n"
      f"**Ball dist XY:** {ball_dist:.2f} m\n\n"
      f"**Stance width:** {stance_width:.3f} m\n\n"
      f"**Ball speed XY:** {ball_speed:.2f} m/s\n\n"
      f"**Outside area:** {'yes' if outside > 0 else 'no'}"
    )

  def _update_metrics(self) -> None:
    self._target_pos_w[:] = self._ball.data.root_link_pos_w
    root_quat = self._robot.data.root_link_quat_w
    self.metrics["torso_roll"] = _roll_from_quat_wxyz(root_quat)
    self.metrics["torso_pitch"] = _pitch_from_quat_wxyz(root_quat)
    self.metrics["torso_height"] = self._robot.data.root_link_pos_w[:, 2]

    yaw_error = _compute_torso_yaw_error(self._robot, self._target_pos_w)
    self.metrics["yaw_error_abs"] = yaw_error.abs()

    trunk_xy = self._robot.data.root_link_pos_w[:, :2]
    target_xy = self._target_pos_w[:, :2]
    self.metrics["target_distance_xy"] = torch.linalg.norm(target_xy - trunk_xy, dim=1)
    _, left_xy, right_xy = _stance_center_xy(
      self._env,
      self._robot,
      self.cfg.stance_left_foot_body_name,
      self.cfg.stance_right_foot_body_name,
    )
    self.metrics["stance_width"] = torch.linalg.norm(right_xy - left_xy, dim=1)

    trunk_xy_local = _world_to_env_local_xy(self._env, trunk_xy)
    outside = _outside_area_violation(trunk_xy_local, self.cfg.keeper_area_bounds)
    self.metrics["outside_keeper_area"] = outside
    self.metrics["ball_speed_xy"] = torch.linalg.norm(
      self._ball.data.root_link_lin_vel_w[:, :2], dim=1
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    spawn_x, spawn_y = self._sample_keeper_spawn_xy(env_ids)
    self._reset_ball_pose(env_ids)
    self._reset_robot_pose(env_ids, spawn_x, spawn_y)
    self._sample_ball_launcher(env_ids)

    # Stage-1 decoder command input. For E1 we keep it deterministic zero.
    self._command[env_ids] = 0.0

  def _update_command(self) -> None:
    time_s = self._env.episode_length_buf.to(torch.float) * self._env.step_dt

    curb_contact = self._ball_curb_contact_mask()
    if curb_contact.any():
      env_ids = curb_contact.nonzero(as_tuple=False).flatten()
      self._set_ball_velocity_zero(env_ids)
      self._cancel_future_dribble_taps(env_ids)
      if self.cfg.rebound_relaunch_enabled:
        self._schedule_rebound_relaunch(env_ids, time_s)

    if self.cfg.rebound_relaunch_enabled:
      pending_ids = self._rebound_pending.nonzero(as_tuple=False).flatten()
      if pending_ids.numel() > 0:
        # Keep the ball frozen until the delayed throw-in style relaunch fires.
        self._set_ball_velocity_zero(pending_ids)

      to_relaunch = self._rebound_pending & (time_s >= self._rebound_time_s)
      if to_relaunch.any():
        env_ids = to_relaunch.nonzero(as_tuple=False).flatten()
        self._execute_rebound_relaunch(env_ids)

    to_kick = (~self._kick_applied) & (time_s >= self._kick_time_s)
    if to_kick.any():
      env_ids = to_kick.nonzero(as_tuple=False).flatten()
      self._set_ball_linear_velocity(env_ids, self._kick_vel_w[env_ids])
      self._kick_applied[env_ids] = True

    to_tap = (
      self._tap_enabled
      & self._kick_applied
      & (self._remaining_taps > 0)
      & (time_s >= self._next_tap_time_s)
    )
    if to_tap.any():
      env_ids = to_tap.nonzero(as_tuple=False).flatten()
      tap_dv = self._sample_velocity_around_mean_direction(
        len(env_ids),
        self.cfg.dribble_tap_speed_range,
        self._last_push_dir_xy[env_ids],
      )
      self._add_ball_linear_velocity(env_ids, tap_dv)
      self._last_push_dir_xy[env_ids] = self._unit_xy(
        tap_dv[:, :2],
        fallback_xy=self._last_push_dir_xy[env_ids],
      )
      self._remaining_taps[env_ids] -= 1

      remaining = self._remaining_taps[env_ids]
      still_mask = remaining > 0
      done_mask = ~still_mask

      if still_mask.any():
        still_ids = env_ids[still_mask]
        dt_next = _sample_uniform_range(
          self.cfg.dribble_tap_interval_range[0],
          self.cfg.dribble_tap_interval_range[1],
          len(still_ids),
          self.device,
        )
        self._next_tap_time_s[still_ids] = time_s[still_ids] + dt_next
      if done_mask.any():
        done_ids = env_ids[done_mask]
        self._tap_enabled[done_ids] = False
        self._next_tap_time_s[done_ids] = 1.0e9

  def _sample_keeper_spawn_xy(
    self,
    env_ids: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    default_root_state = self._robot.data.default_root_state
    assert default_root_state is not None
    stage_cfg = self.stage_cfg

    spawn_x = _sample_uniform_range(
      stage_cfg.keeper_spawn_x_range[0],
      stage_cfg.keeper_spawn_x_range[1],
      len(env_ids),
      self.device,
    )
    # Early curriculum stages use a disjoint union with an explicit hole around y=0.
    spawn_y = _sample_union_range(
      stage_cfg.keeper_spawn_y_range.intervals,
      len(env_ids),
      self.device,
    )

    origins = self._env.scene.env_origins[env_ids]
    spawn_pos_w = default_root_state[env_ids, :3].clone()
    spawn_pos_w[:, 0] = origins[:, 0] + spawn_x
    spawn_pos_w[:, 1] = origins[:, 1] + spawn_y
    self._spawn_pos_w[env_ids] = spawn_pos_w
    return spawn_x, spawn_y

  def _sample_spawn_yaw(
    self,
    env_ids: torch.Tensor,
  ) -> torch.Tensor:
    stage_cfg = self.stage_cfg
    yaw_lo, yaw_hi = stage_cfg.spawn_yaw_offset_range
    if abs(yaw_hi - yaw_lo) <= 1.0e-9:
      yaw_offset = torch.full((len(env_ids),), float(yaw_lo), device=self.device)
    else:
      yaw_offset = _sample_uniform_range(yaw_lo, yaw_hi, len(env_ids), self.device)
    return torch.full(
      (len(env_ids),),
      float(self.cfg.nominal_keeper_facing_yaw),
      device=self.device,
    ) + yaw_offset

  def _reset_robot_pose(
    self,
    env_ids: torch.Tensor,
    spawn_x: torch.Tensor,
    spawn_y: torch.Tensor,
  ) -> None:
    default_root_state = self._robot.data.default_root_state
    default_joint_pos = self._robot.data.default_joint_pos
    default_joint_vel = self._robot.data.default_joint_vel

    assert default_root_state is not None
    assert default_joint_pos is not None
    assert default_joint_vel is not None

    # Future reset curriculum hook.
    use_ready = (
      torch.rand(len(env_ids), device=self.device) < float(self.cfg.p_ready)
    )
    if use_ready.any():
      self._reset_to_ready_stance(env_ids[use_ready], spawn_x[use_ready], spawn_y[use_ready])
    if (~use_ready).any():
      self._reset_to_default_pose(
        env_ids[(~use_ready)],
        spawn_x[(~use_ready)],
        spawn_y[(~use_ready)],
      )

    # Joint state already written inside _reset_to_ready_stance / _reset_to_default_pose.
    # For default-pose envs (use_ready=False) we still apply the default keyframe.
    if (~use_ready).any():
      not_ready_ids = env_ids[~use_ready]
      self._robot.write_joint_state_to_sim(
        default_joint_pos[not_ready_ids],
        default_joint_vel[not_ready_ids],
        env_ids=not_ready_ids,
      )
    self._robot.reset(env_ids=env_ids)

  def _reset_to_default_pose(
    self,
    env_ids: torch.Tensor,
    spawn_x: torch.Tensor,
    spawn_y: torch.Tensor,
  ) -> None:
    if env_ids.numel() == 0:
      return

    default_root_state = self._robot.data.default_root_state
    assert default_root_state is not None

    root_state = default_root_state[env_ids].clone()
    origins = self._env.scene.env_origins[env_ids]

    root_state[:, 0] = origins[:, 0] + spawn_x
    root_state[:, 1] = origins[:, 1] + spawn_y

    base_quat = root_state[:, 3:7].clone()
    base_yaw = _yaw_from_quat_wxyz(base_quat)
    yaw = self._sample_spawn_yaw(env_ids)
    yaw_offset = yaw - base_yaw
    yaw_q = quat_from_euler_xyz(
      torch.zeros_like(yaw),
      torch.zeros_like(yaw),
      yaw_offset,
    )
    root_state[:, 3:7] = quat_mul(yaw_q, base_quat)

    root_state[:, 7:13] = 0.0
    self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

  def _reset_to_ready_stance(
    self,
    env_ids: torch.Tensor,
    spawn_x: torch.Tensor,
    spawn_y: torch.Tensor,
  ) -> None:
    from mjlab.tasks.goalkeeper_experts.e1V2_mezzaluna.config.t1_23dof.env_cfgs import (
      KEEPER_SPAWN_Z,
      READY_JOINT_POS,
      READY_ROOT_QUAT,
    )

    if env_ids.numel() == 0:
      return

    default_root_state = self._robot.data.default_root_state
    assert default_root_state is not None

    root_state = default_root_state[env_ids].clone()
    origins = self._env.scene.env_origins[env_ids]

    root_state[:, 0] = origins[:, 0] + spawn_x
    root_state[:, 1] = origins[:, 1] + spawn_y
    root_state[:, 2] = origins[:, 2] + KEEPER_SPAWN_Z

    ready_quat = torch.tensor(READY_ROOT_QUAT, dtype=torch.float32, device=self.device)
    ready_yaw = _yaw_from_quat_wxyz(ready_quat.unsqueeze(0)).squeeze(0)
    yaw = self._sample_spawn_yaw(env_ids)
    yaw_offset = yaw - ready_yaw
    yaw_q = quat_from_euler_xyz(
      torch.zeros_like(yaw),
      torch.zeros_like(yaw),
      yaw_offset,
    )
    root_state[:, 3:7] = quat_mul(
      yaw_q,
      ready_quat.unsqueeze(0).expand(len(env_ids), -1),
    )
    root_state[:, 7:13] = 0.0
    self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    ready_jp = torch.tensor(READY_JOINT_POS, dtype=torch.float32, device=self.device)
    ready_jp = ready_jp.unsqueeze(0).expand(len(env_ids), -1)
    ready_jv = torch.zeros_like(ready_jp)
    self._robot.write_joint_state_to_sim(ready_jp, ready_jv, env_ids=env_ids)

  def _reset_ball_pose(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    default_root_state = self._ball.data.default_root_state
    assert default_root_state is not None
    stage_cfg = self.stage_cfg

    forward = _sample_uniform_range(
      stage_cfg.target_forward_range[0],
      stage_cfg.target_forward_range[1],
      len(env_ids),
      self.device,
    )
    # Early curriculum stages use a disjoint union with an explicit hole around lateral=0.
    lateral = _sample_union_range(
      stage_cfg.target_lateral_range.intervals,
      len(env_ids),
      self.device,
    )

    root_state = default_root_state[env_ids].clone()
    # Sample the initial ball position relative to the keeper spawn position.
    root_state[:, 0] = self._spawn_pos_w[env_ids, 0] - forward
    root_state[:, 1] = self._spawn_pos_w[env_ids, 1] + lateral
    z_min = float(self.cfg.target_height_min)
    scale = max(float(self.cfg.target_height_exp_scale), 1.0e-6)
    u = torch.rand(len(env_ids), device=self.device)
    excess = -scale * torch.log(torch.clamp(1.0 - u, min=1.0e-6))
    z = z_min + excess

    if self.cfg.target_height_max is not None:
      z = torch.clamp(z, max=float(self.cfg.target_height_max))

    if self.cfg.debug_force_target_ground_z:
      z = torch.full_like(z, float(self.cfg.debug_target_ground_z))

    root_state[:, 2] = self._env.scene.env_origins[env_ids, 2] + z
    root_state[:, 3:7] = 0.0
    root_state[:, 3] = 1.0
    root_state[:, 7:13] = 0.0
    self._ball.write_root_state_to_sim(root_state, env_ids=env_ids)
    self._ball.reset(env_ids=env_ids)
    self._target_pos_w[env_ids] = root_state[:, :3]

  def _sample_ball_launcher(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    n = len(env_ids)
    dead_prob, lateral_prob, dribble_prob = self.stage_cfg.launcher_mode_probs
    dead_prob = max(0.0, min(1.0, float(dead_prob)))
    lateral_prob = max(0.0, min(1.0 - dead_prob, float(lateral_prob)))
    dribble_prob = max(0.0, float(dribble_prob))
    total = dead_prob + lateral_prob + dribble_prob
    if total <= 1.0e-6:
      dead_prob, lateral_prob, dribble_prob = 1.0, 0.0, 0.0
    else:
      dead_prob /= total
      lateral_prob /= total
      dribble_prob /= total

    u = torch.rand(n, device=self.device)
    dead_mask = u < dead_prob
    lateral_mask = (u >= dead_prob) & (u < (dead_prob + lateral_prob))
    dribble_mask = ~(dead_mask | lateral_mask)

    self._launcher_mode[env_ids] = 0
    self._launcher_mode[env_ids[lateral_mask]] = 1
    self._launcher_mode[env_ids[dribble_mask]] = 2

    self._kick_vel_w[env_ids] = 0.0
    self._kick_time_s[env_ids] = 1.0e9
    self._next_tap_time_s[env_ids] = 1.0e9
    self._kick_applied[env_ids] = True
    self._tap_enabled[env_ids] = False
    self._remaining_taps[env_ids] = 0
    self._last_push_dir_xy[env_ids] = 0.0
    self._rebound_pending[env_ids] = False
    self._rebound_time_s[env_ids] = 1.0e9
    self._rebound_used_count[env_ids] = 0
    self._rebound_wall_side_sign[env_ids] = 0.0

    # Dead/dribble modes must start grounded (no exponential-z spawn).
    grounded_ids = env_ids[dead_mask | dribble_mask]
    if grounded_ids.numel() > 0:
      self._force_ball_ground_spawn(grounded_ids)

    dead_ids = env_ids[dead_mask]
    if dead_ids.numel() > 0:
      drift_prob = float(self.cfg.dead_ball_tiny_drift_prob)
      drift_mask = torch.rand(len(dead_ids), device=self.device) < drift_prob
      drift_ids = dead_ids[drift_mask]
      if drift_ids.numel() > 0:
        self._kick_vel_w[drift_ids] = self._sample_lateral_velocity(
          len(drift_ids),
          self.cfg.dead_ball_drift_speed_range,
        )
        self._kick_time_s[drift_ids] = 0.0
        self._kick_applied[drift_ids] = False

    moving_ids = env_ids[lateral_mask | dribble_mask]
    if moving_ids.numel() > 0:
      self._kick_vel_w[moving_ids] = self._sample_lateral_velocity(
        len(moving_ids),
        self.cfg.kick_speed_range,
      )
      self._kick_applied[moving_ids] = False
      self._kick_time_s[moving_ids] = 0.0

    dribble_ids = env_ids[dribble_mask]
    if dribble_ids.numel() > 0:
      self._last_push_dir_xy[dribble_ids] = self._unit_xy(
        self._kick_vel_w[dribble_ids, :2]
      )
      taps_low = int(self.cfg.dribble_num_taps_range[0])
      taps_high = int(self.cfg.dribble_num_taps_range[1])
      if taps_low < 1:
        taps_low = 1
      if taps_high < taps_low:
        taps_high = taps_low

      num_taps = torch.randint(
        low=taps_low,
        high=taps_high + 1,
        size=(len(dribble_ids),),
        device=self.device,
      )
      self._tap_enabled[dribble_ids] = True
      self._remaining_taps[dribble_ids] = num_taps
    self._next_tap_time_s[dribble_ids] = _sample_uniform_range(
        self.cfg.dribble_tap_time_range[0],
        self.cfg.dribble_tap_time_range[1],
        len(dribble_ids),
        self.device,
      )

  def _force_ball_ground_spawn(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    pose_w = torch.zeros((len(env_ids), 7), device=self.device)
    # Use cached freshly-sampled XY from _reset_ball_pose (authoritative during reset).
    pose_w[:, :2] = self._target_pos_w[env_ids, :2]
    pose_w[:, 2] = self._env.scene.env_origins[env_ids, 2] + float(self.cfg.target_height_min)
    pose_w[:, 3] = 1.0

    self._ball.write_root_link_pose_to_sim(pose_w, env_ids=env_ids)
    self._target_pos_w[env_ids] = pose_w[:, :3]

  def _sample_lateral_velocity(
    self,
    num: int,
    speed_range: tuple[float, float],
  ) -> torch.Tensor:
    speed = _sample_uniform_range(
      speed_range[0],
      speed_range[1],
      num,
      self.device,
    )
    side = torch.where(
      torch.rand(num, device=self.device) < 0.5,
      torch.ones(num, device=self.device),
      -torch.ones(num, device=self.device),
    )
    noise_deg = _sample_uniform_range(
      -float(self.cfg.kick_angle_noise_deg),
      float(self.cfg.kick_angle_noise_deg),
      num,
      self.device,
    )
    angle = side * (torch.pi / 2.0) + (noise_deg * torch.pi / 180.0)
    v_x = speed * torch.cos(angle)
    v_y = speed * torch.sin(angle)
    vel = torch.stack([v_x, v_y, torch.zeros_like(v_x)], dim=1)
    return self._clamp_toward_goal_speed(vel)

  def _sample_velocity_around_mean_direction(
    self,
    num: int,
    speed_range: tuple[float, float],
    mean_dir_xy: torch.Tensor,
    angle_noise_deg: float | None = None,
  ) -> torch.Tensor:
    speed = _sample_uniform_range(
      speed_range[0],
      speed_range[1],
      num,
      self.device,
    )
    mean_dir = self._unit_xy(mean_dir_xy)
    mean_angle = torch.atan2(mean_dir[:, 1], mean_dir[:, 0])
    if angle_noise_deg is None:
      angle_noise_deg = float(self.cfg.kick_angle_noise_deg)
    noise_deg = _sample_uniform_range(
      -float(angle_noise_deg),
      float(angle_noise_deg),
      num,
      self.device,
    )
    angle = mean_angle + (noise_deg * torch.pi / 180.0)
    v_x = speed * torch.cos(angle)
    v_y = speed * torch.sin(angle)
    vel = torch.stack([v_x, v_y, torch.zeros_like(v_x)], dim=1)
    return self._clamp_toward_goal_speed(vel)

  def _unit_xy(
    self,
    vec_xy: torch.Tensor,
    fallback_xy: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if fallback_xy is None:
      fallback_xy = torch.zeros_like(vec_xy)
      fallback_xy[:, 1] = 1.0
    fallback_norm = torch.linalg.norm(fallback_xy, dim=1, keepdim=True)
    safe_fallback = torch.where(
      fallback_norm > 1.0e-6,
      fallback_xy / fallback_norm,
      torch.tensor([0.0, 1.0], device=self.device, dtype=vec_xy.dtype).expand_as(vec_xy),
    )
    norm = torch.linalg.norm(vec_xy, dim=1, keepdim=True)
    return torch.where(norm > 1.0e-6, vec_xy / norm, safe_fallback)

  def _clamp_toward_goal_speed(self, vel_w_xyz: torch.Tensor) -> torch.Tensor:
    max_goal_speed = float(self.cfg.max_toward_goal_speed)
    if self.cfg.goal_toward_positive_x:
      vel_w_xyz[:, 0] = torch.clamp(vel_w_xyz[:, 0], max=max_goal_speed)
    else:
      vel_w_xyz[:, 0] = torch.clamp(vel_w_xyz[:, 0], min=-max_goal_speed)
    return vel_w_xyz

  def _set_ball_linear_velocity(
    self,
    env_ids: torch.Tensor,
    vel_w_xyz: torch.Tensor,
  ) -> None:
    vel_w_xyz = self._clamp_toward_goal_speed(vel_w_xyz.clone())
    ball_vel = self._ball.data.root_link_vel_w[env_ids].clone()
    ball_vel[:, :3] = vel_w_xyz
    self._ball.write_root_link_velocity_to_sim(ball_vel, env_ids=env_ids)

  def _set_ball_velocity_zero(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    ball_vel = self._ball.data.root_link_vel_w[env_ids].clone()
    ball_vel[:] = 0.0
    self._ball.write_root_link_velocity_to_sim(ball_vel, env_ids=env_ids)

  def _ball_curb_contact_mask(self) -> torch.Tensor:
    sensor_name = self.cfg.ball_curb_sensor_name
    if sensor_name is None or sensor_name == "":
      return torch.zeros(self._env.num_envs, device=self.device, dtype=torch.bool)

    sensor = self._env.scene[sensor_name]
    found = sensor.data.found
    if found is None:
      return torch.zeros(self._env.num_envs, device=self.device, dtype=torch.bool)
    return torch.any(found > 0.0, dim=1)

  def _add_ball_linear_velocity(
    self,
    env_ids: torch.Tensor,
    delta_v_w_xyz: torch.Tensor,
  ) -> None:
    ball_vel = self._ball.data.root_link_vel_w[env_ids].clone()
    ball_vel[:, :3] += delta_v_w_xyz
    ball_vel[:, :3] = self._clamp_toward_goal_speed(ball_vel[:, :3])
    self._ball.write_root_link_velocity_to_sim(ball_vel, env_ids=env_ids)

  def _cancel_future_dribble_taps(self, env_ids: torch.Tensor) -> None:
    self._tap_enabled[env_ids] = False
    self._remaining_taps[env_ids] = 0
    self._next_tap_time_s[env_ids] = 1.0e9

  def _infer_rebound_side_wall_contact(
    self,
    env_ids: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    ball_pos_local = self._ball.data.root_link_pos_w[env_ids] - self._env.scene.env_origins[env_ids]
    y_local = ball_pos_local[:, 1]
    wall_band = max(float(self.cfg.rebound_inset_m) + 0.10, 0.25)
    side_contact = torch.abs(y_local) >= (float(self.cfg.field_half_width_y) - wall_band)
    wall_side_sign = torch.where(
      y_local >= 0.0,
      torch.ones_like(y_local),
      -torch.ones_like(y_local),
    )
    return side_contact, wall_side_sign

  def _schedule_rebound_relaunch(
    self,
    env_ids: torch.Tensor,
    time_s: torch.Tensor,
  ) -> None:
    if env_ids.numel() == 0:
      return
    if int(self.cfg.rebound_max_events) <= 0:
      return

    eligible = (~self._rebound_pending[env_ids]) & (
      self._rebound_used_count[env_ids] < int(self.cfg.rebound_max_events)
    )
    if not eligible.any():
      return

    eligible_ids = env_ids[eligible]
    side_contact, wall_side_sign = self._infer_rebound_side_wall_contact(eligible_ids)
    qualifying = side_contact
    if not qualifying.any():
      return

    qualifying_ids = eligible_ids[qualifying]
    qualifying_sign = wall_side_sign[qualifying]
    delay_s = _sample_uniform_range(
      self.cfg.rebound_delay_range_s[0],
      self.cfg.rebound_delay_range_s[1],
      len(qualifying_ids),
      self.device,
    )

    self._rebound_pending[qualifying_ids] = True
    self._rebound_time_s[qualifying_ids] = time_s[qualifying_ids] + delay_s
    self._rebound_used_count[qualifying_ids] += 1
    self._rebound_wall_side_sign[qualifying_ids] = qualifying_sign

  def _execute_rebound_relaunch(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    side_sign = self._rebound_wall_side_sign[env_ids]
    pose_w = self._ball.data.root_link_pose_w[env_ids].clone()
    origins = self._env.scene.env_origins[env_ids]

    inset_local_y = side_sign * (
      float(self.cfg.field_half_width_y) - float(self.cfg.rebound_inset_m)
    )
    current_local_y = pose_w[:, 1] - origins[:, 1]
    new_local_y = torch.where(
      side_sign > 0.0,
      torch.minimum(current_local_y, inset_local_y),
      torch.maximum(current_local_y, inset_local_y),
    )
    pose_w[:, 1] = origins[:, 1] + new_local_y
    self._ball.write_root_link_pose_to_sim(pose_w, env_ids=env_ids)

    mean_dir_xy = torch.zeros((len(env_ids), 2), device=self.device)
    mean_dir_xy[:, 1] = -side_sign
    relaunch_vel = self._sample_velocity_around_mean_direction(
      len(env_ids),
      self.cfg.rebound_speed_range,
      mean_dir_xy,
      angle_noise_deg=float(self.cfg.rebound_angle_noise_deg),
    )
    self._set_ball_linear_velocity(env_ids, relaunch_vel)

    self._rebound_pending[env_ids] = False
    self._rebound_time_s[env_ids] = 1.0e9
    self._rebound_wall_side_sign[env_ids] = 0.0

  def _debug_vis_impl(self, visualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    root_pos = self._robot.data.root_link_pos_w[batch]
    target_pos = self._target_pos_w[batch]

    target_xy = target_pos[:2] - root_pos[:2]
    target_norm = torch.linalg.norm(target_xy)
    if float(target_norm.item()) < 1.0e-6:
      return

    desired_dir_xy = target_xy / target_norm

    start = root_pos.clone()
    start[2] += float(self.cfg.viz.z_offset)

    desired_end = start.clone()
    desired_end[0] += desired_dir_xy[0] * float(self.cfg.viz.desired_length)
    desired_end[1] += desired_dir_xy[1] * float(self.cfg.viz.desired_length)

    q = self._robot.data.root_link_quat_w[batch]
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    yaw = torch.atan2(
      2.0 * (qw * qz + qx * qy),
      1.0 - 2.0 * (qy * qy + qz * qz),
    )
    actual_end = start.clone()
    actual_end[0] += torch.cos(yaw) * float(self.cfg.viz.actual_length)
    actual_end[1] += torch.sin(yaw) * float(self.cfg.viz.actual_length)

    visualizer.add_arrow(
      start.cpu().numpy(),
      desired_end.cpu().numpy(),
      color=self.cfg.viz.desired_color,
      width=float(self.cfg.viz.width),
      label="desired_facing",
    )
    visualizer.add_arrow(
      start.cpu().numpy(),
      actual_end.cpu().numpy(),
      color=self.cfg.viz.actual_color,
      width=float(self.cfg.viz.width),
      label="actual_facing",
    )

    # Home point marker: purple dot at the fixed home-point reference.
    home_xy = _home_point_world_xy(self._env, "set_square")[batch]
    home_point = root_pos.clone()
    home_point[0] = home_xy[0]
    home_point[1] = home_xy[1]
    home_point[2] = float(self.cfg.viz.mezzaluna_point_z)
    visualizer.add_sphere(
      home_point.cpu().numpy(),
      radius=float(self.cfg.viz.mezzaluna_point_radius),
      color=self.cfg.viz.mezzaluna_point_color,
      label="home_point",
    )

    # Mezzaluna point marker: flat cylinder with the old home-point styling.
    mezzaluna_xy = _mezzaluna_point_world_xy(self._env, "set_square")[batch]
    mezzaluna_start = root_pos.clone()
    mezzaluna_start[0] = mezzaluna_xy[0]
    mezzaluna_start[1] = mezzaluna_xy[1]
    mezzaluna_start[2] = float(self.cfg.viz.home_point_z)
    mezzaluna_end = mezzaluna_start.clone()
    mezzaluna_end[2] += float(self.cfg.viz.home_point_height)
    visualizer.add_cylinder(
      mezzaluna_start.cpu().numpy(),
      mezzaluna_end.cpu().numpy(),
      radius=float(self.cfg.viz.home_point_radius),
      color=self.cfg.viz.home_point_color,
      label="mezzaluna_point",
    )

    # Stance orthogonality cue (for stance_ortho_to_ball reward).
    try:
      left_idx, right_idx = _resolve_body_index_pair_cached(
        self._env,
        self._robot,
        self.cfg.stance_left_foot_body_name,
        self.cfg.stance_right_foot_body_name,
      )
    except ValueError:
      return

    body_pos = self._robot.data.body_link_pos_w[batch]
    left_foot = body_pos[left_idx].clone()
    right_foot = body_pos[right_idx].clone()
    left_foot[2] += float(self.cfg.viz.stance_z_offset)
    right_foot[2] += float(self.cfg.viz.stance_z_offset)

    visualizer.add_cylinder(
      left_foot.cpu().numpy(),
      right_foot.cpu().numpy(),
      radius=float(self.cfg.viz.foot_line_radius),
      color=self.cfg.viz.foot_line_color,
      label="stance_foot_line",
    )

    stance_vec_xy = right_foot[:2] - left_foot[:2]
    stance_width = torch.linalg.norm(stance_vec_xy)
    ball_vec_xy = target_pos[:2] - root_pos[:2]
    ball_dist = torch.linalg.norm(ball_vec_xy)

    if float(stance_width.item()) <= 1.0e-6 or float(ball_dist.item()) <= 1.0e-6:
      return

    stance_dir_xy = stance_vec_xy / stance_width.clamp_min(1.0e-6)
    ball_dir_xy = ball_vec_xy / ball_dist.clamp_min(1.0e-6)
    stance_ortho_dir_xy = torch.stack([-stance_dir_xy[1], stance_dir_xy[0]])
    if torch.dot(stance_ortho_dir_xy, ball_dir_xy) < 0.0:
      stance_ortho_dir_xy = -stance_ortho_dir_xy

    dot_sb = torch.dot(stance_dir_xy, ball_dir_xy).clamp(-1.0, 1.0)
    stance_ortho = 1.0 - torch.square(dot_sb)
    gate_on = (
      float(stance_width.item()) > float(self.cfg.stance_ortho_w_min)
      and float(ball_dist.item()) > float(self.cfg.stance_ortho_d_min)
    )

    cue_origin = 0.5 * (left_foot + right_foot)
    cue_origin[2] += float(self.cfg.viz.stance_z_offset)
    axis_len = float(self.cfg.viz.stance_axis_length)
    ball_end = cue_origin.clone()
    ball_end[:2] += ball_dir_xy * axis_len
    stance_ortho_end = cue_origin.clone()
    stance_ortho_end[:2] += stance_ortho_dir_xy * axis_len

    visualizer.add_arrow(
      cue_origin.cpu().numpy(),
      ball_end.cpu().numpy(),
      color=self.cfg.viz.ball_dir_color,
      width=float(self.cfg.viz.stance_axis_width),
      label="stance_ball_dir",
    )

    if gate_on:
      t = float(stance_ortho.item())
      stance_color = tuple(
        (1.0 - t) * self.cfg.viz.stance_bad_color[i]
        + t * self.cfg.viz.stance_good_color[i]
        for i in range(4)
      )
    else:
      stance_color = self.cfg.viz.stance_neutral_color

    visualizer.add_arrow(
      cue_origin.cpu().numpy(),
      stance_ortho_end.cpu().numpy(),
      color=stance_color,
      width=float(self.cfg.viz.stance_axis_width),
      label="stance_footline_ortho",
    )
    visualizer.add_sphere(
      cue_origin.cpu().numpy(),
      radius=float(self.cfg.viz.stance_cue_radius),
      color=stance_color,
      label="stance_ortho_cue",
    )


def target_direction_xy(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]
  rel_xy = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  return _normalize_xy(rel_xy)


def target_position_relative_xyz(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]
  return ball.data.root_link_pos_w - robot.data.root_link_pos_w


def ball_velocity_relative_xyz(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]
  return ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w


def desired_point_relative_xy(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  target_w_xy = _reward_target_world_xy(env, command_name)
  return target_w_xy - robot.data.root_link_pos_w[:, :2]


def yaw_alignment_waist_reward(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  k: float = 2.5,
  waist_body_name: str = r"(?i)^waist$",
  apply_standing_gate: bool = False,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]
  # Waist alignment encourages leg/foot pivot micro-steps instead of trunk-only rotation.
  yaw_error = _compute_waist_yaw_error(
    env,
    robot,
    ball.data.root_link_pos_w,
    waist_body_name=waist_body_name,
  )
  raw = torch.exp(-k * torch.square(yaw_error))
  reward = _apply_standing_gate_if_enabled(
    raw,
    env,
    asset_cfg,
    apply_standing_gate,
  )
  reward = reward * _alignment_home_ramp(env, command_name, asset_cfg)
  return _apply_reward_active_mask(reward, env, command_name)


def waist_ready_twist_abs_penalty(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
  waist_body_name: str = r"(?i)^waist$",
  k: float = 2.5,
  apply_standing_gate: bool = False,
  eps: float = 1.0e-6,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]

  center_xy, left_xy, right_xy = _stance_center_xy(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  support_vec_xy = right_xy - left_xy
  support_dir_xy = support_vec_xy / torch.linalg.norm(
    support_vec_xy, dim=1, keepdim=True
  ).clamp_min(float(eps))
  support_normal_xy = torch.stack(
    (-support_dir_xy[:, 1], support_dir_xy[:, 0]),
    dim=1,
  )

  ball_dir_xy = _normalize_xy(ball.data.root_link_pos_w[:, :2] - center_xy, eps=float(eps))
  normal_sign = torch.sign(torch.sum(support_normal_xy * ball_dir_xy, dim=1))
  normal_sign = torch.where(
    normal_sign == 0.0,
    torch.ones_like(normal_sign),
    normal_sign,
  )
  desired_normal_xy = support_normal_xy * normal_sign.unsqueeze(1)
  desired_ready_yaw = torch.atan2(desired_normal_xy[:, 1], desired_normal_xy[:, 0])

  waist_idx = _resolve_single_body_index_cached(env, robot, waist_body_name)
  waist_yaw = _yaw_from_quat_wxyz(robot.data.body_link_quat_w[:, waist_idx, :])
  twist_err = _wrap_angle_pi(waist_yaw - desired_ready_yaw)
  raw = 1.0 - torch.exp(-float(k) * torch.square(twist_err))

  penalty = _apply_standing_gate_if_enabled(
    raw,
    env,
    asset_cfg,
    apply_standing_gate,
  )
  penalty = penalty * _alignment_home_ramp(
    env,
    command_name,
    asset_cfg,
    left_foot_body_name=left_foot_body_name,
    right_foot_body_name=right_foot_body_name,
  )

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_waist_ready_twist_abs_err_mean"] = torch.mean(torch.abs(twist_err))
    log["Metrics/e1_waist_ready_twist_abs_pen_mean"] = torch.mean(penalty)

  return _apply_reward_active_mask(penalty, env, command_name)


def waist_yaw_progress_reward(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  waist_body_name: str = r"(?i)^waist$",
  err_gate: float = 0.25,
  upright_gate: float = 0.80,
  max_delta: float = 0.35,
  tilt_target: float = 0.0,
  tilt_band: float = -1.0,
  tilt_sigma: float = 0.5,
  apply_standing_gate: bool = False,
) -> torch.Tensor:
  """Reward positive reduction in absolute waist yaw error to the ball."""
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]

  yaw_err_waist = _compute_waist_yaw_error(
    env,
    robot,
    ball.data.root_link_pos_w,
    waist_body_name=waist_body_name,
  )
  err_abs = yaw_err_waist.abs()

  prev_err_abs = _get_waist_yaw_progress_prev_error_buffer(env, robot, waist_body_name)
  reward_steps = _reward_step_count(env, command_name)
  active_mask = _reward_active_mask(env, command_name)
  is_first_step = reward_steps <= 1
  effective_prev = torch.where(is_first_step, err_abs, prev_err_abs)

  prog_raw = effective_prev - err_abs
  prog = torch.clamp(prog_raw, min=0.0, max=float(max_delta))
  posture_score = _yaw_reward_posture_score(
    robot.data.projected_gravity_b,
    tilt_target=tilt_target,
    tilt_band=tilt_band,
    tilt_sigma=tilt_sigma,
  )
  ok_err = err_abs > float(err_gate)
  ok_upright = posture_score > float(upright_gate)
  reward = torch.where(ok_err & ok_upright, prog, torch.zeros_like(prog))
  reward = _apply_standing_gate_if_enabled(
    reward,
    env,
    asset_cfg,
    apply_standing_gate,
  )
  reward = reward * _alignment_home_ramp(env, command_name, asset_cfg)
  reward = _apply_reward_active_mask(reward, env, command_name)

  prev_err_abs[active_mask] = err_abs[active_mask]
  return reward


def waist_yaw_abs_penalty(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  waist_body_name: str = r"(?i)^waist$",
  upright_gate: float = 0.85,
  tilt_target: float = 0.0,
  tilt_band: float = -1.0,
  tilt_sigma: float = 0.5,
) -> torch.Tensor:
  """Penalize absolute waist yaw error to keep persistent pressure to square up."""
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]

  yaw_err_waist = _compute_waist_yaw_error(
    env,
    robot,
    ball.data.root_link_pos_w,
    waist_body_name=waist_body_name,
  )
  err_abs = yaw_err_waist.abs()
  posture_score = _yaw_reward_posture_score(
    robot.data.projected_gravity_b,
    tilt_target=tilt_target,
    tilt_band=tilt_band,
    tilt_sigma=tilt_sigma,
  )
  penalty = torch.where(
    posture_score > float(upright_gate),
    err_abs,
    torch.zeros_like(err_abs),
  )

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_waist_yaw_abs_pen_raw_mean"] = torch.mean(err_abs)

  penalty = penalty * _alignment_home_ramp(env, command_name, asset_cfg)
  return _apply_reward_active_mask(penalty, env, command_name)


def stance_ortho_to_ball_reward(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
  w_min: float = 0.10,
  d_min: float = 0.35,
  eps: float = 1.0e-6,
  neutral_when_mask_off: bool = True,
) -> torch.Tensor:
  """Reward stance axis orthogonality to ball direction in XY plane.

  stance axis: left->right foot direction
  target: dot(stance_dir, ball_dir) ~= 0
  """
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]

  left_idx, right_idx = _resolve_body_index_pair_cached(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )

  body_pos_w = robot.data.body_link_pos_w
  p_l_xy = body_pos_w[:, left_idx, :2]
  p_r_xy = body_pos_w[:, right_idx, :2]

  stance_vec_xy = p_r_xy - p_l_xy
  stance_width = torch.linalg.norm(stance_vec_xy, dim=1)
  stance_dir_xy = stance_vec_xy / stance_width.unsqueeze(1).clamp_min(float(eps))

  root_xy = robot.data.root_link_pos_w[:, :2]
  ball_xy = ball.data.root_link_pos_w[:, :2]
  ball_vec_xy = ball_xy - root_xy
  ball_dist = torch.linalg.norm(ball_vec_xy, dim=1)
  ball_dir_xy = ball_vec_xy / ball_dist.unsqueeze(1).clamp_min(float(eps))

  dot = torch.sum(stance_dir_xy * ball_dir_xy, dim=1).clamp(-1.0, 1.0)
  stance_ortho = 1.0 - torch.square(dot)

  mask = (stance_width > float(w_min)) & (ball_dist > float(d_min))
  if neutral_when_mask_off:
    return torch.where(mask, stance_ortho, torch.ones_like(stance_ortho))
  return mask.float() * stance_ortho


def upright_stability_reward(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  roll_band: float = 0.1,
  roll_sigma: float = 0.12,
  pitch_target: float = 0.10,
  pitch_band: float = 0.20,
  pitch_sigma: float = 0.30,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  posture_score, roll_score, pitch_score, lateral, sagittal = _posture_score_components(
    robot.data.projected_gravity_b,
    roll_band=roll_band,
    roll_sigma=roll_sigma,
    pitch_target=pitch_target,
    pitch_band=pitch_band,
    pitch_sigma=pitch_sigma,
  )

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_roll_score_mean"] = torch.mean(roll_score)
    log["Metrics/e1_pitch_score_mean"] = torch.mean(pitch_score)
    log["Metrics/e1_lateral_posture_component_mean"] = torch.mean(lateral)
    log["Metrics/e1_sagittal_posture_component_mean"] = torch.mean(sagittal)

  return _apply_reward_active_mask(posture_score, env)


def low_height_soft_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  h_soft: float = 0.48,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  height = robot.data.root_link_pos_w[:, 2]
  low = torch.relu(float(h_soft) - height)
  penalty = torch.square(low)
  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_low_height_soft_pen_mean"] = torch.mean(penalty)
  return _apply_reward_active_mask(penalty, env)


def stance_center_home_axis_abs_penalty(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  axis: str = "y",
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
) -> torch.Tensor:
  axis_idx = 0 if axis.lower() == "x" else 1
  robot: Entity = env.scene[asset_cfg.name]
  center_xy, _, _ = _stance_center_xy(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  home_xy = _reward_target_world_xy(env, command_name)
  err = torch.abs(home_xy[:, axis_idx] - center_xy[:, axis_idx])
  log = _get_log_dict(env)
  if log is not None:
    if axis_idx == 0:
      log["Metrics/e1_home_x_err_mean"] = torch.mean(err)
    else:
      log["Metrics/e1_home_y_err_mean"] = torch.mean(err)
  return _apply_reward_active_mask(err, env, command_name)


def stance_center_home_x_asymmetric_abs_penalty(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  coeff_behind: float = 2.0,
  coeff_forward: float = 0.5,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  center_xy, _, _ = _stance_center_xy(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  home_xy = _reward_target_world_xy(env, command_name)
  dx = center_xy[:, 0] - home_xy[:, 0]
  behind = dx < 0.0
  return torch.where(
    behind,
    float(coeff_behind) * torch.abs(dx),
    float(coeff_forward) * torch.abs(dx),
  ) * _reward_active_float_mask(env, command_name)


def stance_center_home_axis_progress_reward(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  axis: str = "y",
  max_delta: float = 0.20,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
  apply_standing_gate: bool = False,
) -> torch.Tensor:
  axis_idx = 0 if axis.lower() == "x" else 1
  robot: Entity = env.scene[asset_cfg.name]
  center_xy, _, _ = _stance_center_xy(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  home_xy = _reward_target_world_xy(env, command_name)
  err_abs = torch.abs(home_xy[:, axis_idx] - center_xy[:, axis_idx])

  prev = _get_float_state_buffer(
    env,
    name=f"home_axis_prev::{axis}",
    num_envs=env.num_envs,
    device=env.device,
  )
  reward_steps = _reward_step_count(env, command_name)
  active_mask = _reward_active_mask(env, command_name)
  is_first_step = reward_steps <= 1
  effective_prev = torch.where(is_first_step, err_abs, prev)
  prog = torch.clamp(effective_prev - err_abs, min=0.0, max=float(max_delta))
  prog = _apply_standing_gate_if_enabled(prog, env, asset_cfg, apply_standing_gate)
  prog = _apply_reward_active_mask(prog, env, command_name)

  prev[active_mask] = err_abs[active_mask]

  log = _get_log_dict(env)
  if log is not None:
    if axis_idx == 0:
      log["Metrics/e1_home_x_progress_mean"] = torch.mean(prog)
    else:
      log["Metrics/e1_home_y_progress_mean"] = torch.mean(prog)
  return prog


def stance_center_move_toward_home_reward(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
  r_deadband: float = 0.10,
  v_cap: float = 0.6,
  apply_standing_gate: bool = False,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  center_xy, _, _ = _stance_center_xy(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  left_idx, right_idx = _resolve_body_index_pair_cached(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  v_center_xy = 0.5 * (
    robot.data.body_link_lin_vel_w[:, left_idx, :2]
    + robot.data.body_link_lin_vel_w[:, right_idx, :2]
  )

  home_xy = _reward_target_world_xy(env, command_name)
  to_home_xy = home_xy - center_xy
  dist = torch.linalg.norm(to_home_xy, dim=1)
  dir_xy = to_home_xy / dist.unsqueeze(1).clamp_min(1.0e-6)
  active = dist > float(r_deadband)
  dir_xy = torch.where(active.unsqueeze(1), dir_xy, torch.zeros_like(dir_xy))

  toward_speed = torch.sum(v_center_xy * dir_xy, dim=1)
  reward = torch.clamp(toward_speed, min=0.0, max=float(v_cap))
  reward = _apply_standing_gate_if_enabled(reward, env, asset_cfg, apply_standing_gate)

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_move_toward_home_mean"] = torch.mean(reward)
  return _apply_reward_active_mask(reward, env, command_name)


def stance_ortho_abs_penalty(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
) -> torch.Tensor:
  ortho_err = _stance_ortho_score(
    env,
    command_name,
    asset_cfg,
    left_foot_body_name,
    right_foot_body_name,
  )
  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_stance_ortho_err_mean"] = torch.mean(ortho_err)
  penalty = ortho_err * _alignment_home_ramp(
    env,
    command_name,
    asset_cfg,
    left_foot_body_name=left_foot_body_name,
    right_foot_body_name=right_foot_body_name,
  )
  return _apply_reward_active_mask(penalty, env, command_name)


def stance_ortho_progress_reward(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
  max_delta: float = 0.20,
  apply_standing_gate: bool = False,
) -> torch.Tensor:
  ortho_err = _stance_ortho_score(
    env,
    command_name,
    asset_cfg,
    left_foot_body_name,
    right_foot_body_name,
  )
  prev = _get_float_state_buffer(
    env,
    name="stance_ortho_err_prev",
    num_envs=env.num_envs,
    device=env.device,
  )
  reward_steps = _reward_step_count(env, command_name)
  active_mask = _reward_active_mask(env, command_name)
  is_first_step = reward_steps <= 1
  effective_prev = torch.where(is_first_step, ortho_err, prev)
  prog = torch.clamp(effective_prev - ortho_err, min=0.0, max=float(max_delta))
  prog = _apply_standing_gate_if_enabled(prog, env, asset_cfg, apply_standing_gate)
  prog = prog * _alignment_home_ramp(
    env,
    command_name,
    asset_cfg,
    left_foot_body_name=left_foot_body_name,
    right_foot_body_name=right_foot_body_name,
  )
  prog = _apply_reward_active_mask(prog, env, command_name)
  prev[active_mask] = ortho_err[active_mask]

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_stance_ortho_err_mean"] = torch.mean(ortho_err)
    log["Metrics/e1_stance_ortho_progress_mean"] = torch.mean(prog)

  return prog


def stance_width_band_penalty(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
  w_min: float = 0.16,
  w_max: float = 0.4,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  _, left_xy, right_xy = _stance_center_xy(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  width = torch.linalg.norm(right_xy - left_xy, dim=1)
  # Keep a comfortable stance-width band: penalize both overly narrow and overly wide stances.
  narrow_penalty = torch.square(torch.relu(float(w_min) - width))
  wide_penalty = torch.square(torch.relu(width - float(w_max)))
  penalty = narrow_penalty + wide_penalty
  return _apply_reward_active_mask(penalty, env, command_name)


def pelvis_between_feet_reward(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
  waist_body_name: str = r"(?i)^waist$",
  lateral_sigma: float = 0.09,
  longitudinal_sigma: float = 0.16,
  lateral_weight: float = 1.0,
  longitudinal_weight: float = 0.35,
  apply_standing_gate: bool = False,
  eps: float = 1.0e-6,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  center_xy, left_xy, right_xy = _stance_center_xy(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  waist_idx = _resolve_single_body_index_cached(env, robot, waist_body_name)
  pelvis_xy = robot.data.body_link_pos_w[:, waist_idx, :2]

  support_vec_xy = right_xy - left_xy
  support_width = torch.linalg.norm(support_vec_xy, dim=1, keepdim=True)
  support_dir_xy = support_vec_xy / support_width.clamp_min(float(eps))
  support_normal_xy = torch.stack(
    (-support_dir_xy[:, 1], support_dir_xy[:, 0]),
    dim=1,
  )

  pelvis_offset_xy = pelvis_xy - center_xy
  longitudinal_offset = torch.sum(pelvis_offset_xy * support_dir_xy, dim=1)
  lateral_offset = torch.sum(pelvis_offset_xy * support_normal_xy, dim=1)

  lat_term = torch.square(lateral_offset / max(float(lateral_sigma), float(eps)))
  # Keep this term strictly lateral: frontal/sagittal pelvis placement no longer
  # contributes to the reward, but we still log it for inspection.
  _ = longitudinal_sigma
  _ = longitudinal_weight
  err = float(lateral_weight) * lat_term
  raw = torch.exp(-err)

  reward = _apply_standing_gate_if_enabled(
    raw,
    env,
    asset_cfg,
    apply_standing_gate,
  )
  reward = reward * _alignment_home_ramp(
    env,
    command_name,
    asset_cfg,
    left_foot_body_name=left_foot_body_name,
    right_foot_body_name=right_foot_body_name,
  )

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_pelvis_between_feet_mean"] = torch.mean(reward)
    log["Metrics/e1_pelvis_between_feet_lateral_offset_mean"] = torch.mean(
      torch.abs(lateral_offset)
    )
    log["Metrics/e1_pelvis_between_feet_longitudinal_offset_mean"] = torch.mean(
      torch.abs(longitudinal_offset)
    )

  return _apply_reward_active_mask(reward, env, command_name)


def body_ang_vel_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize torso XY angular velocity (velocity-style stabilizer)."""
  robot: Entity = env.scene[asset_cfg.name]
  ang_vel_xy = robot.data.root_link_ang_vel_w[:, :2]
  penalty = torch.sum(torch.square(ang_vel_xy), dim=1)
  return _apply_reward_active_mask(penalty, env)


def angular_momentum_penalty(
  env,
  sensor_name: str = "robot/root_angmom",
) -> torch.Tensor:
  """Penalize whole-body angular momentum magnitude (velocity-style stabilizer)."""
  try:
    angmom_sensor = cast(BuiltinSensor, env.scene[sensor_name])
  except KeyError:
    return torch.zeros(env.num_envs, device=env.device)

  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_angular_momentum_mean"] = torch.mean(torch.sqrt(angmom_magnitude_sq))
  return _apply_reward_active_mask(angmom_magnitude_sq, env)


def outside_keeper_area_penalty(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  pos_xy_local = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
  penalty = _outside_area_violation(pos_xy_local, command.keeper_area_bounds)
  return _apply_reward_active_mask(penalty, env, command_name)


def fallen_indicator(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_height: float = 0.30,
  max_tilt: float = 1.20,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  height = robot.data.root_link_pos_w[:, 2]
  tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
  fallen = (height < min_height) | (tilt > max_tilt)
  return _apply_reward_active_mask(fallen.float(), env)


def outside_keeper_area_hard(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  pos_xy_local = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
  violation = _outside_area_violation(
    pos_xy_local,
    command.hard_keeper_area_bounds,
  )
  return violation > 0.0


class FallTermination:
  """Terminate if fallen state persists for several consecutive steps."""

  def __init__(self, cfg, env):
    del cfg
    self._counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._counter[env_ids] = 0

  def __call__(
    self,
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    min_height: float = 0.30,
    max_tilt: float = 1.20,
    consecutive_steps: int = 6,
  ) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    height = robot.data.root_link_pos_w[:, 2]
    tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
    fallen_now = (height < min_height) | (tilt > max_tilt)
    self._counter = torch.where(
      fallen_now,
      self._counter + 1,
      torch.zeros_like(self._counter),
    )
    return self._counter >= int(consecutive_steps)


def action_rate_l2(
  env,
  command_name: str = "set_square",
) -> torch.Tensor:
  return _apply_reward_active_mask(_base_action_rate_l2(env), env, command_name)


def joint_pos_limits(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str = "set_square",
) -> torch.Tensor:
  return _apply_reward_active_mask(
    _base_joint_pos_limits(env, asset_cfg=asset_cfg),
    env,
    command_name,
  )
