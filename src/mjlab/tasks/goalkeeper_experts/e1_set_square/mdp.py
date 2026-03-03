from __future__ import annotations

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
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _sample_uniform_range(
  low: float,
  high: float,
  num: int,
  device: str,
) -> torch.Tensor:
  return torch.rand(num, device=device) * (high - low) + low


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


def _get_bool_state_buffer(
  env,
  name: str,
  num_envs: int,
  device: torch.device | str,
) -> torch.Tensor:
  """Get/create per-env bool buffer used by stateful reward shaping."""
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_e1_reward_bool_state_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  buf = cache.get(name)
  if (
    buf is None
    or buf.shape != (num_envs,)
    or buf.device != torch.device(device)
    or buf.dtype != torch.bool
  ):
    buf = torch.zeros((num_envs,), dtype=torch.bool, device=device)
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


def _foot_support_from_sensor(
  env,
  sensor_name: str | None,
  fz_thresh: float,
  support_sign: str = "neg",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute support-contact boolean from vertical net force with robust fallback.

  Preferred logic (E1): support = (Fz < -fz_thresh).
  Fallback logic (missing/invalid force): support = (found > 0).
  """
  if sensor_name is None or sensor_name == "":
    zeros_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    zeros_f = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return zeros_bool, zeros_f, zeros_f

  sensor = cast(ContactSensor, env.scene[sensor_name])
  found = sensor.data.found
  if found is None:
    found_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  elif found.ndim == 1:
    found_contact = found > 0.0
  else:
    found_contact = torch.any(found > 0.0, dim=1)

  force = sensor.data.force
  if force is None:
    zeros_f = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return found_contact, zeros_f, zeros_f

  # Expected shapes are [B, 3] or [B, N, 3].
  if force.ndim == 2 and force.shape[-1] == 3:
    fz = force[:, 2]
    force_norm = torch.linalg.norm(force, dim=1)
  elif force.ndim == 3 and force.shape[-1] == 3:
    fz_all = force[..., 2]
    force_norm_all = torch.linalg.norm(force, dim=-1)
    if fz_all.ndim == 1:
      fz = fz_all
      force_norm = force_norm_all
    else:
      fz = torch.max(fz_all, dim=1).values
      force_norm = torch.max(force_norm_all, dim=1).values
  else:
    zeros_f = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return found_contact, zeros_f, zeros_f

  if fz.ndim != 1:
    fz = fz.reshape(env.num_envs, -1).max(dim=1).values
  if force_norm.ndim != 1:
    force_norm = force_norm.reshape(env.num_envs, -1).max(dim=1).values
  fz = fz.to(torch.float32)
  force_norm = force_norm.to(torch.float32)

  valid_fz = torch.isfinite(fz)
  if support_sign == "neg":
    support_from_force = fz < -float(fz_thresh)
  elif support_sign == "pos":
    support_from_force = fz > float(fz_thresh)
  elif support_sign == "abs":
    support_from_force = fz.abs() > float(fz_thresh)
  else:
    raise ValueError(
      f"Unsupported support_sign='{support_sign}'. Use one of: 'neg', 'pos', 'abs'."
    )
  support = torch.where(valid_fz, support_from_force, found_contact)
  fz_clean = torch.nan_to_num(fz, nan=0.0, posinf=0.0, neginf=0.0)
  force_norm_clean = torch.nan_to_num(force_norm, nan=0.0, posinf=0.0, neginf=0.0)
  return support, fz_clean, force_norm_clean


def get_target_ball_cfg() -> EntityCfg:
  """Return the physical colliding RoboCup ball for E1."""
  return get_robocup_ball_cfg()


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

  # Anti-shot clamp: limit component toward defended goal.
  # If goal is at +x, keep vx <= max_toward_goal_speed.
  goal_toward_positive_x: bool = True
  max_toward_goal_speed: float = 0.25

  # Reset curriculum hook.
  p_ready: float = 0.0

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

  viz: VizCfg = field(default_factory=VizCfg)

  # Optional yaw jitter at spawn.
  spawn_yaw_range: tuple[float, float] = (0.0, 0.0)

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

    self.metrics["yaw_error_abs"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["target_distance_xy"] = torch.zeros(env.num_envs, device=self.device)
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

  def _update_metrics(self) -> None:
    self._target_pos_w[:] = self._ball.data.root_link_pos_w

    yaw_error = _compute_torso_yaw_error(self._robot, self._target_pos_w)
    self.metrics["yaw_error_abs"] = yaw_error.abs()

    trunk_xy = self._robot.data.root_link_pos_w[:, :2]
    target_xy = self._target_pos_w[:, :2]
    self.metrics["target_distance_xy"] = torch.linalg.norm(target_xy - trunk_xy, dim=1)

    trunk_xy_local = _world_to_env_local_xy(self._env, trunk_xy)
    outside = _outside_area_violation(trunk_xy_local, self.cfg.keeper_area_bounds)
    self.metrics["outside_keeper_area"] = outside
    self.metrics["ball_speed_xy"] = torch.linalg.norm(
      self._ball.data.root_link_lin_vel_w[:, :2], dim=1
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    self._reset_robot_pose(env_ids)
    self._reset_ball_pose(env_ids)
    self._sample_ball_launcher(env_ids)

    # Stage-1 decoder command input. For E1 we keep it deterministic zero.
    self._command[env_ids] = 0.0

  def _update_command(self) -> None:
    time_s = self._env.episode_length_buf.to(torch.float) * self._env.step_dt

    curb_contact = self._ball_curb_contact_mask()
    if curb_contact.any():
      env_ids = curb_contact.nonzero(as_tuple=False).flatten()
      self._set_ball_velocity_zero(env_ids)
      # Stop future dribble taps once curb contact happens.
      self._tap_enabled[env_ids] = False
      self._remaining_taps[env_ids] = 0
      self._next_tap_time_s[env_ids] = 1.0e9

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

  def _reset_robot_pose(self, env_ids: torch.Tensor) -> None:
    default_root_state = self._robot.data.default_root_state
    default_joint_pos = self._robot.data.default_joint_pos
    default_joint_vel = self._robot.data.default_joint_vel

    assert default_root_state is not None
    assert default_joint_pos is not None
    assert default_joint_vel is not None

    spawn_x = _sample_uniform_range(
      self.cfg.keeper_spawn_x_range[0],
      self.cfg.keeper_spawn_x_range[1],
      len(env_ids),
      self.device,
    )
    spawn_y = _sample_uniform_range(
      self.cfg.keeper_spawn_y_range[0],
      self.cfg.keeper_spawn_y_range[1],
      len(env_ids),
      self.device,
    )

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

    # Ensure joint state always starts from default keyframe for E1 stage.
    self._robot.write_joint_state_to_sim(
      default_joint_pos[env_ids],
      default_joint_vel[env_ids],
      env_ids=env_ids,
    )
    self._robot.clear_state(env_ids=env_ids)

    # Cache spawn in world frame for rewards.
    origins = self._env.scene.env_origins[env_ids]
    spawn_pos_w = default_root_state[env_ids, :3].clone()
    spawn_pos_w[:, 0] = origins[:, 0] + spawn_x
    spawn_pos_w[:, 1] = origins[:, 1] + spawn_y
    self._spawn_pos_w[env_ids] = spawn_pos_w

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

    yaw_lo, yaw_hi = self.cfg.spawn_yaw_range
    # Support both fixed-yaw resets (yaw_lo == yaw_hi) and sampled yaw ranges.
    if abs(yaw_hi - yaw_lo) <= 1.0e-9:
      yaw = torch.full((len(env_ids),), float(yaw_lo), device=self.device)
    else:
      yaw = _sample_uniform_range(yaw_lo, yaw_hi, len(env_ids), self.device)
    if torch.any(torch.abs(yaw) > 1.0e-9):
      yaw_q = quat_from_euler_xyz(
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        yaw,
      )
      root_state[:, 3:7] = quat_mul(root_state[:, 3:7], yaw_q)

    root_state[:, 7:13] = 0.0
    self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

  def _reset_to_ready_stance(
    self,
    env_ids: torch.Tensor,
    spawn_x: torch.Tensor,
    spawn_y: torch.Tensor,
  ) -> None:
    # TODO: plug ready-stance reset once ready pose distribution is provided.
    self._reset_to_default_pose(env_ids, spawn_x, spawn_y)

  def _reset_ball_pose(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    default_root_state = self._ball.data.default_root_state
    assert default_root_state is not None

    forward = _sample_uniform_range(
      self.cfg.target_forward_range[0],
      self.cfg.target_forward_range[1],
      len(env_ids),
      self.device,
    )
    lateral = _sample_uniform_range(
      self.cfg.target_lateral_range[0],
      self.cfg.target_lateral_range[1],
      len(env_ids),
      self.device,
    )

    root_state = default_root_state[env_ids].clone()
    # Keep the ball on keeper front side after 180-deg spawn heading.
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
    self._ball.clear_state(env_ids=env_ids)
    self._target_pos_w[env_ids] = root_state[:, :3]

  def _sample_ball_launcher(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    n = len(env_ids)
    dead_prob = max(0.0, min(1.0, float(self.cfg.dead_ball_prob)))
    lateral_prob = max(0.0, min(1.0 - dead_prob, float(self.cfg.lateral_roll_prob)))

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
  ) -> torch.Tensor:
    speed = _sample_uniform_range(
      speed_range[0],
      speed_range[1],
      num,
      self.device,
    )
    mean_dir = self._unit_xy(mean_dir_xy)
    mean_angle = torch.atan2(mean_dir[:, 1], mean_dir[:, 0])
    noise_deg = _sample_uniform_range(
      -float(self.cfg.kick_angle_noise_deg),
      float(self.cfg.kick_angle_noise_deg),
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


def target_relative_xy(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]
  return ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]


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


def yaw_alignment_torso_reward(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  k: float = 2.5,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]
  yaw_error = _compute_torso_yaw_error(robot, ball.data.root_link_pos_w)
  return torch.exp(-k * torch.square(yaw_error))


def yaw_alignment_waist_reward(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  k: float = 2.5,
  waist_body_name: str = r"(?i)^waist$",
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
  return torch.exp(-k * torch.square(yaw_error))


def waist_yaw_progress_reward(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  waist_body_name: str = r"(?i)^waist$",
  err_gate: float = 0.25,
  upright_gate: float = 0.80,
  max_delta: float = 0.35,
  tilt_sigma: float = 0.5,
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
  is_first_step = env.episode_length_buf <= 1
  effective_prev = torch.where(is_first_step, err_abs, prev_err_abs)

  prog_raw = effective_prev - err_abs
  prog = torch.clamp(prog_raw, min=0.0, max=float(max_delta))

  projected_gravity_b = robot.data.projected_gravity_b
  tilt = torch.linalg.norm(projected_gravity_b[:, :2], dim=1)
  upright = torch.exp(-torch.square(tilt) / max(tilt_sigma * tilt_sigma, 1.0e-6))

  mask = (upright > float(upright_gate)) & (err_abs > float(err_gate))
  reward = mask.float() * prog

  prev_err_abs.copy_(err_abs)
  return reward


def waist_yaw_abs_penalty(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  waist_body_name: str = r"(?i)^waist$",
  upright_gate: float = 0.85,
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

  projected_gravity_b = robot.data.projected_gravity_b
  tilt = torch.linalg.norm(projected_gravity_b[:, :2], dim=1)
  upright = torch.exp(-torch.square(tilt) / max(tilt_sigma * tilt_sigma, 1.0e-6))
  penalty = torch.where(
    upright > float(upright_gate),
    err_abs,
    torch.zeros_like(err_abs),
  )

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_waist_yaw_abs_pen_raw_mean"] = torch.mean(penalty)

  return penalty


def torso_waist_twist_penalty(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  waist_body_name: str = r"(?i)^waist$",
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]
  yaw_err_torso = _compute_torso_yaw_error(robot, ball.data.root_link_pos_w)
  yaw_err_waist = _compute_waist_yaw_error(
    env,
    robot,
    ball.data.root_link_pos_w,
    waist_body_name=waist_body_name,
  )
  return torch.square(yaw_err_torso - yaw_err_waist)


def foot_yaw_slip_contact_pen(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
  left_contact_sensor_name: str = "left_foot_ground_contact",
  right_contact_sensor_name: str = "right_foot_ground_contact",
  fz_thresh: float = 40.0,
  support_sign: str = "neg",
) -> torch.Tensor:
  """Penalize yaw spinning of feet while they are in contact with the ground."""
  robot: Entity = env.scene[asset_cfg.name]
  left_idx, right_idx = _resolve_body_index_pair_cached(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )

  support_l, fz_l, fnorm_l = _foot_support_from_sensor(
    env, left_contact_sensor_name, fz_thresh, support_sign=support_sign
  )
  support_r, fz_r, fnorm_r = _foot_support_from_sensor(
    env, right_contact_sensor_name, fz_thresh, support_sign=support_sign
  )

  body_ang_vel_w = robot.data.body_link_ang_vel_w
  slip_l = torch.abs(body_ang_vel_w[:, left_idx, 2])
  slip_r = torch.abs(body_ang_vel_w[:, right_idx, 2])
  penalty = 0.5 * (
    support_l.float() * torch.square(slip_l) + support_r.float() * torch.square(slip_r)
  )

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_foot_yaw_slip_contact_pen_raw_mean"] = torch.mean(penalty)
    log["Metrics/e1_left_foot_fz_mean"] = torch.mean(fz_l)
    log["Metrics/e1_right_foot_fz_mean"] = torch.mean(fz_r)
    log["Metrics/e1_left_foot_fz_min"] = torch.min(fz_l)
    log["Metrics/e1_left_foot_fz_max"] = torch.max(fz_l)
    log["Metrics/e1_right_foot_fz_min"] = torch.min(fz_r)
    log["Metrics/e1_right_foot_fz_max"] = torch.max(fz_r)
    log["Metrics/e1_left_foot_force_norm_mean"] = torch.mean(fnorm_l)
    log["Metrics/e1_right_foot_force_norm_mean"] = torch.mean(fnorm_r)
    log["Metrics/e1_left_support_frac"] = torch.mean(support_l.float())
    log["Metrics/e1_right_support_frac"] = torch.mean(support_r.float())
    # Keep legacy metric names for continuity in dashboards.
    log["Metrics/e1_foot_contact_left_frac"] = torch.mean(support_l.float())
    log["Metrics/e1_foot_contact_right_frac"] = torch.mean(support_r.float())

  return penalty


def foot_xy_slip_contact_pen(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
  left_contact_sensor_name: str = "left_foot_ground_contact",
  right_contact_sensor_name: str = "right_foot_ground_contact",
  fz_thresh: float = 40.0,
  support_sign: str = "neg",
) -> torch.Tensor:
  """Penalize xy foot slip while the foot is in support contact."""
  robot: Entity = env.scene[asset_cfg.name]
  left_idx, right_idx = _resolve_body_index_pair_cached(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )

  support_l, _, _ = _foot_support_from_sensor(
    env, left_contact_sensor_name, fz_thresh, support_sign=support_sign
  )
  support_r, _, _ = _foot_support_from_sensor(
    env, right_contact_sensor_name, fz_thresh, support_sign=support_sign
  )

  body_lin_vel_w = robot.data.body_link_lin_vel_w
  slip_l = torch.linalg.norm(body_lin_vel_w[:, left_idx, :2], dim=1)
  slip_r = torch.linalg.norm(body_lin_vel_w[:, right_idx, :2], dim=1)
  penalty = 0.5 * (
    support_l.float() * torch.square(slip_l) + support_r.float() * torch.square(slip_r)
  )

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_foot_xy_slip_contact_pen_raw_mean"] = torch.mean(penalty)

  return penalty


def foot_contact_switch_bonus(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
  left_contact_sensor_name: str = "left_foot_ground_contact",
  right_contact_sensor_name: str = "right_foot_ground_contact",
  waist_body_name: str = r"(?i)^waist$",
  upright_gate: float = 0.85,
  err_gate: float = 0.25,
  fz_thresh: float = 40.0,
  support_sign: str = "neg",
) -> torch.Tensor:
  """Small exploration bonus for contact state switches during meaningful misalignment."""
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[command.cfg.ball_entity_name]

  _resolve_body_index_pair_cached(
    env,
    robot,
    left_foot_body_name,
    right_foot_body_name,
  )
  support_l, fz_l, fnorm_l = _foot_support_from_sensor(
    env, left_contact_sensor_name, fz_thresh, support_sign=support_sign
  )
  support_r, fz_r, fnorm_r = _foot_support_from_sensor(
    env, right_contact_sensor_name, fz_thresh, support_sign=support_sign
  )

  prev_contact_l = _get_bool_state_buffer(
    env,
    name=f"support_prev::{left_contact_sensor_name}",
    num_envs=env.num_envs,
    device=env.device,
  )
  prev_contact_r = _get_bool_state_buffer(
    env,
    name=f"support_prev::{right_contact_sensor_name}",
    num_envs=env.num_envs,
    device=env.device,
  )
  is_first_step = env.episode_length_buf <= 1
  prev_l_eff = torch.where(is_first_step, support_l, prev_contact_l)
  prev_r_eff = torch.where(is_first_step, support_r, prev_contact_r)
  switch_l = torch.logical_xor(support_l, prev_l_eff)
  switch_r = torch.logical_xor(support_r, prev_r_eff)
  switch = switch_l | switch_r

  yaw_err_waist = _compute_waist_yaw_error(
    env,
    robot,
    ball.data.root_link_pos_w,
    waist_body_name=waist_body_name,
  )
  err_abs = yaw_err_waist.abs()
  upright = upright_stability_reward(env, asset_cfg=asset_cfg)
  gate = (upright > float(upright_gate)) & (err_abs > float(err_gate))
  bonus = gate.float() * switch.float()

  prev_contact_l.copy_(support_l)
  prev_contact_r.copy_(support_r)

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e1_foot_contact_switch_bonus_raw_mean"] = torch.mean(bonus)
    log["Metrics/e1_left_foot_fz_mean"] = torch.mean(fz_l)
    log["Metrics/e1_right_foot_fz_mean"] = torch.mean(fz_r)
    log["Metrics/e1_left_foot_fz_min"] = torch.min(fz_l)
    log["Metrics/e1_left_foot_fz_max"] = torch.max(fz_l)
    log["Metrics/e1_right_foot_fz_min"] = torch.min(fz_r)
    log["Metrics/e1_right_foot_fz_max"] = torch.max(fz_r)
    log["Metrics/e1_left_foot_force_norm_mean"] = torch.mean(fnorm_l)
    log["Metrics/e1_right_foot_force_norm_mean"] = torch.mean(fnorm_r)
    log["Metrics/e1_left_support_frac"] = torch.mean(support_l.float())
    log["Metrics/e1_right_support_frac"] = torch.mean(support_r.float())
    log["Metrics/e1_support_switch_frac"] = torch.mean(switch.float())
    # Keep legacy metric names for continuity in dashboards.
    log["Metrics/e1_foot_contact_switch_frac"] = torch.mean(switch.float())
    log["Metrics/e1_foot_contact_left_frac"] = torch.mean(support_l.float())
    log["Metrics/e1_foot_contact_right_frac"] = torch.mean(support_r.float())

  return bonus


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
  height_target: float = 0.62,
  height_sigma: float = 0.12,
  tilt_sigma: float = 0.5,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]

  # TEMPORARY (E1 set-square): disable height shaping, keep tilt-only upright reward.
  # height = robot.data.root_link_pos_w[:, 2]
  # height_err_sq = torch.square(height - height_target)
  # height_reward = torch.exp(-height_err_sq / max(height_sigma * height_sigma, 1.0e-6))
  del height_target, height_sigma

  projected_gravity_b = robot.data.projected_gravity_b
  tilt = torch.linalg.norm(projected_gravity_b[:, :2], dim=1)
  upright_reward = torch.exp(-torch.square(tilt) / max(tilt_sigma * tilt_sigma, 1.0e-6))

  return upright_reward


def xy_drift_deadzone(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  r_free: float = 0.10,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  delta = robot.data.root_link_pos_w[:, :2] - command.spawn_pos_w[:, :2]
  drift_mag = torch.linalg.norm(delta, dim=1)
  drift_violation = torch.relu(drift_mag - float(r_free))
  return torch.square(drift_violation)


def xy_drift_deadzone_reward(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  r_free: float = 0.10,
  sigma: float = 0.12,
) -> torch.Tensor:
  """Reward staying close to spawn; 1.0 in deadzone, smooth decay outside."""
  penalty = xy_drift_deadzone(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
    r_free=r_free,
  )
  return torch.exp(-penalty / max(float(sigma) * float(sigma), 1.0e-6))


def xy_speed_deadzone(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  v_free: float = 0.12,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  vel_xy = robot.data.root_link_lin_vel_w[:, :2]
  speed_mag = torch.linalg.norm(vel_xy, dim=1)
  speed_violation = torch.relu(speed_mag - float(v_free))
  return torch.square(speed_violation)


def xy_speed_deadzone_reward(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  v_free: float = 0.12,
  sigma: float = 0.12,
) -> torch.Tensor:
  """Reward low XY speed; 1.0 in deadzone, smooth decay outside."""
  penalty = xy_speed_deadzone(
    env,
    asset_cfg=asset_cfg,
    v_free=v_free,
  )
  return torch.exp(-penalty / max(float(sigma) * float(sigma), 1.0e-6))


def body_ang_vel_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize torso XY angular velocity (velocity-style stabilizer)."""
  robot: Entity = env.scene[asset_cfg.name]
  ang_vel_xy = robot.data.root_link_ang_vel_w[:, :2]
  return torch.sum(torch.square(ang_vel_xy), dim=1)


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
  return angmom_magnitude_sq


def outside_keeper_area_penalty(
  env,
  command_name: str = "set_square",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(SetSquareCommand, env.command_manager.get_term(command_name))
  pos_xy_local = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
  return _outside_area_violation(pos_xy_local, command.keeper_area_bounds)


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
  return fallen.float()


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
