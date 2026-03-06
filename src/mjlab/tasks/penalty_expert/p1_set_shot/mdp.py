from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp import *  # noqa: F401,F403
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.motor_controller_stage1.latent_action import (
  default_motor_obs_layout,
  MotorLatentActionCfg,
  motor_last_decoded_action,
)
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _sample_uniform_range(low: float, high: float, num: int, device: str) -> torch.Tensor:
  return torch.rand(num, device=device) * (high - low) + low


def _normalize_xy(vec_xy: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
  norm = torch.linalg.norm(vec_xy, dim=1, keepdim=True).clamp_min(eps)
  return vec_xy / norm


def _compute_yaw_error(robot: Entity, target_pos_w: torch.Tensor) -> torch.Tensor:
  trunk_pos = robot.data.root_link_pos_w
  target_xy = target_pos_w[:, :2] - trunk_pos[:, :2]
  target_dir_xy = _normalize_xy(target_xy)

  q = robot.data.root_link_quat_w  # wxyz
  qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  yaw = torch.atan2(
    2.0 * (qw * qz + qx * qy),
    1.0 - 2.0 * (qy * qy + qz * qz),
  )
  forward_xy = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)

  dot = torch.sum(forward_xy * target_dir_xy, dim=1).clamp(-1.0, 1.0)
  det = forward_xy[:, 0] * target_dir_xy[:, 1] - forward_xy[:, 1] * target_dir_xy[:, 0]
  return torch.atan2(det, dot)


def _outside_area_violation(pos_xy: torch.Tensor, bounds: tuple[float, float, float, float]) -> torch.Tensor:
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


# ---------------- Commands ----------------

@dataclass
class SetShotCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  ball_entity_name: str = "soccer_ball"

  command_dim: int = 46

  # deterministic spawn (ranges can be (v,v))
  striker_spawn_x_range: tuple[float, float] = (0.0, 0.0)
  striker_spawn_y_range: tuple[float, float] = (0.0, 0.0)
  spawn_yaw_range: tuple[float, float] = (0.0, 0.0)

  ball_spawn_x_range: tuple[float, float] = (0.0, 0.0)
  ball_spawn_y_range: tuple[float, float] = (0.0, 0.0)
  ball_spawn_z: float = 0.11

  # aim point (center goal) in env-local coordinates (before origin)
  aim_x: float = 7.3
  aim_y: float = 0.0
  aim_z: float = 0.0

  # area bounds in env-local coordinates
  striker_area_bounds: tuple[float, float, float, float] = (-1.0, 7.0, -2.0, 2.0)
  hard_area_margin: float = 0.5

  # goal check
  goal_line_x: float = 7.0
  goal_y_half: float = 1.0

  goal_z_min: float = 0.0
  goal_z_max: float = 1.85


  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = True

  def build(self, env):
    return SetShotCommand(self, env)



class SetShotCommand(CommandTerm):
  cfg: SetShotCommandCfg

  def __init__(self, cfg: SetShotCommandCfg, env):
    super().__init__(cfg, env)
    self._robot: Entity = env.scene[cfg.entity_name]
    self._ball: Entity = env.scene[cfg.ball_entity_name]

    self._command = torch.zeros(env.num_envs, cfg.command_dim, device=self.device)
    self._aim_pos_w = torch.zeros(env.num_envs, 3, device=self.device)

    self.metrics["yaw_error_abs"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["ball_dist_xy"] = torch.zeros(env.num_envs, device=self.device)

    # --- goal event state (one-shot) ---
    self._goal_scored = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
    self.metrics["goal_event"] = torch.zeros(env.num_envs, device=self.device)


  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def aim_pos_w(self) -> torch.Tensor:
    return self._aim_pos_w

  @property
  def striker_area_bounds(self) -> tuple[float, float, float, float]:
    return self.cfg.striker_area_bounds

  @property
  def hard_striker_area_bounds(self) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = self.cfg.striker_area_bounds
    m = self.cfg.hard_area_margin
    return (x_min - m, x_max + m, y_min - m, y_max + m)

  def _update_metrics(self) -> None:
    yaw_error = _compute_yaw_error(self._robot, self._aim_pos_w)
    self.metrics["yaw_error_abs"] = yaw_error.abs()

    trunk_xy = self._robot.data.root_link_pos_w[:, :2]
    ball_xy = self._ball.data.root_link_pos_w[:, :2]
    self.metrics["ball_dist_xy"] = torch.linalg.norm(ball_xy - trunk_xy, dim=1)

    ball_pos = self._ball.data.root_link_pos_w
    origins = self._env.scene.env_origins
    ball_local = ball_pos - origins  # (N,3) env-local

    x = ball_local[:, 0]
    y = ball_local[:, 1]
    z = ball_local[:, 2]

    crossed = x >= float(self.cfg.goal_line_x)
    inside_y = torch.abs(y) <= float(self.cfg.goal_y_half)
    inside_z = (z >= float(self.cfg.goal_z_min)) & (z <= float(self.cfg.goal_z_max))
    scored_now = crossed & inside_y & inside_z

    goal_event = scored_now & (~self._goal_scored)
    self._goal_scored |= scored_now
    self.metrics["goal_event"] = goal_event.to(torch.float32)


  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self._goal_scored[env_ids] = False
    self.metrics["goal_event"][env_ids] = 0.0
    self._reset_robot_pose(env_ids)
    self._reset_ball_pose(env_ids)
    self._set_aim_pose(env_ids)

    # Stage-1 command vector: deterministic zeros
    self._command[env_ids] = 0.0

  def _update_command(self) -> None:
    # fixed for whole episode
    pass

  def _set_aim_pose(self, env_ids: torch.Tensor) -> None:
    origins = self._env.scene.env_origins[env_ids]

    # alterna angolo: env pari -> +Y, env dispari -> -Y (zero random)
    signs = torch.where(
        torch.rand(len(env_ids), device=self.device) < 0.5,
        1.0,
        -1.0,
    ).to(origins.dtype)
    aim_y = signs * float(self.cfg.aim_y)


    self._aim_pos_w[env_ids, 0] = origins[:, 0] + float(self.cfg.aim_x)
    self._aim_pos_w[env_ids, 1] = origins[:, 1] + aim_y
    self._aim_pos_w[env_ids, 2] = origins[:, 2] + float(self.cfg.aim_z)

  def _reset_robot_pose(self, env_ids: torch.Tensor) -> None:
    default_root_state = self._robot.data.default_root_state
    default_joint_pos = self._robot.data.default_joint_pos
    default_joint_vel = self._robot.data.default_joint_vel
    assert default_root_state is not None
    assert default_joint_pos is not None
    assert default_joint_vel is not None

    spawn_x = _sample_uniform_range(
      self.cfg.striker_spawn_x_range[0],
      self.cfg.striker_spawn_x_range[1],
      len(env_ids),
      self.device,
    )
    spawn_y = _sample_uniform_range(
      self.cfg.striker_spawn_y_range[0],
      self.cfg.striker_spawn_y_range[1],
      len(env_ids),
      self.device,
    )

    root_state = default_root_state[env_ids].clone()
    origins = self._env.scene.env_origins[env_ids]
    root_state[:, 0] = origins[:, 0] + spawn_x
    root_state[:, 1] = origins[:, 1] + spawn_y

    yaw_lo, yaw_hi = self.cfg.spawn_yaw_range
    if abs(yaw_hi - yaw_lo) <= 1.0e-9:
      yaw = torch.full((len(env_ids),), float(yaw_lo), device=self.device)
    else:
      yaw = _sample_uniform_range(yaw_lo, yaw_hi, len(env_ids), self.device)

    if torch.any(torch.abs(yaw) > 1.0e-9):
      yaw_q = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
      root_state[:, 3:7] = quat_mul(root_state[:, 3:7], yaw_q)

    root_state[:, 7:13] = 0.0
    self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self._robot.write_joint_state_to_sim(
      default_joint_pos[env_ids],
      default_joint_vel[env_ids],
      env_ids=env_ids,
    )
    self._robot.clear_state(env_ids=env_ids)


  def _reset_ball_pose(self, env_ids: torch.Tensor) -> None:
    default_root_state = self._ball.data.default_root_state
    assert default_root_state is not None

    bx = _sample_uniform_range(
      self.cfg.ball_spawn_x_range[0],
      self.cfg.ball_spawn_x_range[1],
      len(env_ids),
      self.device,
    )
    by = _sample_uniform_range(
      self.cfg.ball_spawn_y_range[0],
      self.cfg.ball_spawn_y_range[1],
      len(env_ids),
      self.device,
    )

    root_state = default_root_state[env_ids].clone()
    origins = self._env.scene.env_origins[env_ids]
    root_state[:, 0] = origins[:, 0] + bx
    root_state[:, 1] = origins[:, 1] + by
    root_state[:, 2] = origins[:, 2] + float(self.cfg.ball_spawn_z)

    # identity quat + zero vels
    root_state[:, 3:7] = 0.0
    root_state[:, 3] = 1.0
    root_state[:, 7:13] = 0.0

    self._ball.write_root_state_to_sim(root_state, env_ids=env_ids)
    self._ball.clear_state(env_ids=env_ids)


# ---------------- Observations helpers ----------------

def target_direction_xy(env, command_name: str = "set_shot", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  rel_xy = cmd.aim_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  return _normalize_xy(rel_xy)


def ball_position_relative_xyz(env, command_name: str = "set_shot", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  return ball.data.root_link_pos_w - robot.data.root_link_pos_w


def ball_velocity_w_xy(env, ball_entity_name: str = "soccer_ball") -> torch.Tensor:
  ball: Entity = env.scene[ball_entity_name]
  return ball.data.root_link_lin_vel_w[:, :2]


def has_struck(env, command_name: str = "set_shot") -> torch.Tensor:
  struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
  return struck.to(torch.float32)

def ball_speed_to_aim_reward_3d_after_strike(env, command_name: str = "set_shot") -> torch.Tensor:
  return has_struck(env, command_name) * ball_speed_to_aim_reward_3d(env, command_name)

def ball_flight_high_and_side_reward_after_strike(env, command_name="set_shot", z_min=0.55, y_side_min=0.55) -> torch.Tensor:
  return has_struck(env, command_name) * ball_flight_high_and_side_reward(env, command_name, z_min=z_min, y_side_min=y_side_min)

# ---------------- Rewards ----------------

def yaw_alignment_reward(env, command_name: str = "set_shot", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, k: float = 2.5) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  yaw_error = _compute_yaw_error(robot, cmd.aim_pos_w)
  return torch.exp(-k * torch.square(yaw_error))


def upright_stability_reward(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  height_target: float | None = None,
  height_sigma: float = 0.12,
  tilt_sigma: float = 0.5,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  height = robot.data.root_link_pos_w[:, 2]

  # If height_target is not provided, use the robot's default root height (per-env).
  if height_target is None:
    default_root_state = getattr(robot.data, "default_root_state", None)
    if default_root_state is not None:
      target = default_root_state[:, 2].to(height.dtype)
    else:
      target = torch.mean(height).expand_as(height)
  else:
    target = torch.full_like(height, float(height_target))

  height_err_sq = torch.square((height - target) / float(height_sigma))
  tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
  tilt_err_sq = torch.square(tilt / float(tilt_sigma))
  return torch.exp(-0.5 * (height_err_sq + tilt_err_sq))


def _get_bool_state_buffer(
  env,
  key: str,
) -> torch.Tensor:
  """
  Returns a persistent per-env boolean buffer stored on env.unwrapped.
  Useful for 'latched' events like 'has_struck'.
  """
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_p1_bool_state_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  buf = cache.get(key)
  if buf is None or buf.shape != (env.num_envs,) or buf.device != env.device:
    buf = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    cache[key] = buf
  return buf

def _sensor_any_found(env, sensor_name: str) -> torch.Tensor:
  s = env.scene[sensor_name]
  found = getattr(s.data, "found", None)
  if found is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  return torch.any(found > 0.0, dim=1)

def _get_prev_ball_dist_buffer(env, command_name: str) -> torch.Tensor:
  """Per-env buffer: previous ball distance (for progress shaping)."""
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_p1_prev_ball_dist_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  prev = cache.get(command_name)
  if (
    prev is None
    or prev.shape != (env.num_envs,)
    or prev.device != env.device
    or prev.dtype != torch.float32
  ):
    prev = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    cache[command_name] = prev
  return prev



def _get_int_state_buffer(env, key: str, dtype=torch.int32) -> torch.Tensor:
  """Persistent per-env int buffer stored on env.unwrapped."""
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_p1_int_state_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  buf = cache.get(key)
  if buf is None or buf.shape != (env.num_envs,) or buf.device != env.device or buf.dtype != dtype:
    buf = torch.zeros((env.num_envs,), device=env.device, dtype=dtype)
    cache[key] = buf
  return buf



def approach_ball_progress_reward(
  env,
  command_name: str = "set_shot",
  max_delta: float = 0.06,
  upright_gate: float = 0.65,
) -> torch.Tensor:
  """
  Reward only when you reduce distance to ball (progress), not absolute distance.
  This kills the 'stand still' optimum.
  """
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  dist = cmd.metrics["ball_dist_xy"].to(torch.float32)  # computed in _update_metrics:contentReference[oaicite:4]{index=4}

  prev = _get_prev_ball_dist_buffer(env, command_name)
  is_first_step = env.episode_length_buf <= 1
  prev[is_first_step] = dist[is_first_step]

  prog = (prev - dist).clamp(min=0.0, max=float(max_delta))

  # Gate by upright so you don't get progress reward while falling
  up = upright_stability_reward(env)  # already in mdp
  prog = torch.where(up > float(upright_gate), prog, torch.zeros_like(prog))

  prev.copy_(dist)
  return prog

def no_strike_timeout_penalty(
  env,
  command_name: str = "set_shot",
  episode_length_s: float = 6.0,
) -> torch.Tensor:
  struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")

  is_first_step = env.episode_length_buf <= 1
  struck[is_first_step] = False

  t = env.episode_length_buf.to(torch.float32) * float(env.step_dt)
  last_step = t >= (float(episode_length_s) - 1.5 * float(env.step_dt))

  return (last_step & (~struck)).to(torch.float32)

def approach_ball_reward(env, command_name: str = "set_shot", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, k: float = 2.0) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  d_xy = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  dist = torch.linalg.norm(d_xy, dim=1)
  return torch.exp(-k * dist)


def behind_ball_reward(env, command_name="set_shot", asset_cfg=_DEFAULT_ASSET_CFG,
                       dx_max: float = 0.35) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    robot_xy = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
    ball_xy  = _world_to_env_local_xy(env, ball.data.root_link_pos_w[:, :2])

    dx = (ball_xy[:, 0] - robot_xy[:, 0])   # >0 se robot è dietro
    behind = dx > 0.0
    close  = dx < float(dx_max)
    return (behind & close).to(robot_xy.dtype)


def strike_event_reward(
  env,
  command_name: str = "set_shot",
  left_sensor_name: str = "p1_left_foot_ball_contact",
  right_sensor_name: str = "p1_right_foot_ball_contact",
  speed_thresh: float = 0.6,  # tenuto solo per compatibilità, non usato
) -> torch.Tensor:
  """
  1-shot strike event: primo contatto piede<->palla.
  NON richiede una soglia di velocità della palla.
  """
  touching = _sensor_any_found(env, left_sensor_name) | _sensor_any_found(env, right_sensor_name)

  prev_touch = _get_bool_state_buffer(env, key=f"p1_prev_touch::{command_name}")
  is_first_step = env.episode_length_buf <= 1
  prev_touch[is_first_step] = False

  new_touch = touching & (~prev_touch)
  prev_touch.copy_(touching)

  struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
  struck[is_first_step] = False

  reward = new_touch & (~struck)
  struck |= new_touch

  return reward.to(torch.float32)



def extra_touch_after_first_penalty(
  env,
  command_name: str = "set_shot",
  left_sensor_name: str = "p1_left_foot_ball_contact",
  right_sensor_name: str = "p1_right_foot_ball_contact",
) -> torch.Tensor:
  """Penalizza ogni NUOVO contatto piede↔palla dopo il primo (evento, non per-step)."""
  touching = _sensor_any_found(env, left_sensor_name) | _sensor_any_found(env, right_sensor_name)

  prev_touch = _get_bool_state_buffer(env, key=f"p1_touch_prev::{command_name}")
  touch_count = _get_int_state_buffer(env, key=f"p1_touch_count::{command_name}", dtype=torch.int32)

  is_first = env.episode_length_buf <= 1
  prev_touch[is_first] = False
  touch_count[is_first] = 0

  new_touch = touching & (~prev_touch)
  prev_touch.copy_(touching)

  # count touch events
  touch_count += new_touch.to(torch.int32)

  extra = new_touch & (touch_count > 1)
  return extra.to(torch.float32)


def post_strike_upright_reward(env, command_name: str = "set_shot") -> torch.Tensor:
  """Dopo lo strike, premia restare in piedi."""
  return has_struck(env, command_name) * upright_stability_reward(env)


def post_goal_upright_reward(env, command_name: str = "set_shot") -> torch.Tensor:
  """Dopo che è stato fatto goal (latched), premia restare in piedi fino a fine episodio."""
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  goal_lat = _get_bool_state_buffer(env, key=f"p1_goal_lat::{command_name}")

  is_first = env.episode_length_buf <= 1
  goal_lat[is_first] = False

  goal_event = cmd.metrics["goal_event"] > 0.5
  goal_lat |= goal_event
  return goal_lat.to(torch.float32) * upright_stability_reward(env)


def post_strike_base_speed_penalty(
  env,
  command_name: str = "set_shot",
  max_speed: float = 1.0,
) -> torch.Tensor:
  """Dopo lo strike, penalizza correre/inseguire (aiuta a non ricontattare e a non cadere)."""
  robot: Entity = env.scene["robot"]
  speed_xy = torch.linalg.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
  return has_struck(env, command_name) * (speed_xy.clamp(min=0.0, max=float(max_speed)) / float(max_speed))



def ball_speed_to_goal_reward(env, command_name: str = "set_shot") -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_pos = ball.data.root_link_pos_w
  ball_vel = ball.data.root_link_lin_vel_w

  dir_xy = _normalize_xy(cmd.aim_pos_w[:, :2] - ball_pos[:, :2])
  v_xy = ball_vel[:, :2]
  proj = torch.sum(v_xy * dir_xy, dim=1)
  return proj.clamp_min(0.0)


def goal_scored_termination(env, command_name: str = "set_shot") -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_pos = ball.data.root_link_pos_w
  origins = env.scene.env_origins
  ball_local = ball_pos - origins
  x, y, z = ball_local[:, 0], ball_local[:, 1], ball_local[:, 2]

  crossed = x >= float(cmd.cfg.goal_line_x)
  inside_y = torch.abs(y) <= float(cmd.cfg.goal_y_half)
  inside_z = (z >= float(cmd.cfg.goal_z_min)) & (z <= float(cmd.cfg.goal_z_max))
  return crossed & inside_y & inside_z



def goal_scored_reward(env, command_name: str = "set_shot") -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  return has_struck(env, command_name) * cmd.metrics["goal_event"]



def outside_striker_area_penalty(env, command_name: str = "set_shot", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  pos_xy_local = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
  return _outside_area_violation(pos_xy_local, cmd.striker_area_bounds)


def hard_outside_striker_area_termination(env, command_name: str = "set_shot", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  pos_xy_local = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
  violation = _outside_area_violation(pos_xy_local, cmd.hard_striker_area_bounds)
  return violation > 0.0


def xy_speed_l2(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  vel_xy = robot.data.root_link_lin_vel_w[:, :2]
  return torch.sum(torch.square(vel_xy), dim=1)


def fallen_indicator(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, min_height: float = 0.30, max_tilt: float = 1.20) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  height = robot.data.root_link_pos_w[:, 2]
  tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
  return ((height < float(min_height)) | (tilt > float(max_tilt))).to(torch.float32)

def ball_speed_to_aim_reward_3d(env, command_name: str = "set_shot") -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_pos = ball.data.root_link_pos_w
  ball_vel = ball.data.root_link_lin_vel_w

  dir_3d = cmd.aim_pos_w - ball_pos
  norm = torch.linalg.norm(dir_3d, dim=1, keepdim=True).clamp_min(1e-6)
  dir_3d = dir_3d / norm

  proj = torch.sum(ball_vel * dir_3d, dim=1)
  return proj.clamp_min(0.0)


def ball_flight_high_and_side_reward(
  env,
  command_name: str = "set_shot",
  z_min: float = 0.55,
  y_side_min: float = 0.55,
) -> torch.Tensor:
  # reward denso: se la palla sta andando verso la porta, premia altezza e lateralità
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_local = _world_to_env_local_xy(env, ball.data.root_link_pos_w[:, :2])
  x = ball_local[:, 0]
  y = ball_local[:, 1]
  z = ball.data.root_link_pos_w[:, 2]

  # gating: palla "in avanzamento" verso la porta e non troppo lontana
  moving = ball.data.root_link_lin_vel_w[:, 0] > 0.4
  near_goal = x > float(cmd.cfg.goal_line_x) - 2.0

  high = torch.sigmoid((z - z_min) / 0.12)
  side = torch.sigmoid((torch.abs(y) - y_side_min) / 0.12)

  return (moving & near_goal).to(z.dtype) * high * side


def goal_high_corner_reward(
  env,
  command_name: str = "set_shot",
  z_min: float = 0.55,
  y_side_min: float = 0.55,
) -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_local = _world_to_env_local_xy(env, ball.data.root_link_pos_w[:, :2])
  y = ball_local[:, 1]
  z = ball.data.root_link_pos_w[:, 2]

  good = (z >= z_min) & (torch.abs(y) >= y_side_min)
  return has_struck(env, command_name) * (cmd.metrics["goal_event"] * good.to(torch.float32))



def goal_low_or_center_penalty(
  env,
  command_name: str = "set_shot",
  z_min: float = 0.55,
  y_side_min: float = 0.55,
) -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_local = _world_to_env_local_xy(env, ball.data.root_link_pos_w[:, :2])
  y = ball_local[:, 1]
  z = ball.data.root_link_pos_w[:, 2]

  bad = (z < z_min) | (torch.abs(y) < y_side_min)
  return has_struck(env, command_name) * (cmd.metrics["goal_event"] * bad.to(torch.float32))


def trunk_tilt_l2_penalty(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  tilt_xy = robot.data.projected_gravity_b[:, :2]
  return torch.sum(torch.square(tilt_xy), dim=1)

def _support_bool_from_ground_sensor(
  env,
  sensor_name: str,
  fz_thresh: float = 5.0,
  support_sign: str = "neg",
) -> torch.Tensor:
  s = env.scene[sensor_name]
  found = getattr(s.data, "found", None)
  force = getattr(s.data, "force", None)

  if found is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  has_contact = torch.any(found > 0.0, dim=1)

  if force is None:
    return has_contact

  # netforce reduce -> force shape (N,3) in molti casi
  if force.ndim == 2 and force.shape[1] >= 3:
    fz = force[:, 2].to(torch.float32)
  else:
    return has_contact

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

  return has_contact & support_from_force


def foot_contact_switch_bonus_p1(
  env,
  command_name: str = "set_shot",
  left_contact_sensor_name: str = "p1_left_foot_ground_contact",
  right_contact_sensor_name: str = "p1_right_foot_ground_contact",
  upright_gate: float = 0.75,
  fz_thresh: float = 5.0,
  support_sign: str = "neg",
) -> torch.Tensor:
  """
  Bonus quando cambia lo stato di appoggio (switch).
  Serve per sbloccare i passi.
  """
  # pre-strike only
  gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
  up = upright_stability_reward(env)
  ok_upright = (up > float(upright_gate))

  sup_l = _support_bool_from_ground_sensor(
    env,
    left_contact_sensor_name,
    fz_thresh,
    support_sign=support_sign,
  )
  sup_r = _support_bool_from_ground_sensor(
    env,
    right_contact_sensor_name,
    fz_thresh,
    support_sign=support_sign,
  )

  prev_l = _get_bool_state_buffer(env, key=f"p1_prev_sup_l::{left_contact_sensor_name}")
  prev_r = _get_bool_state_buffer(env, key=f"p1_prev_sup_r::{right_contact_sensor_name}")
  is_first = env.episode_length_buf <= 1
  prev_l[is_first] = sup_l[is_first]
  prev_r[is_first] = sup_r[is_first]

  switch = torch.logical_xor(sup_l, prev_l) | torch.logical_xor(sup_r, prev_r)

  prev_l.copy_(sup_l)
  prev_r.copy_(sup_r)

  return (gate_pre * ok_upright.to(torch.float32) * switch.to(torch.float32))


def base_vel_towards_ball_reward(
  env,
  command_name: str = "set_shot",
  max_speed: float = 1.2,
) -> torch.Tensor:
  """
  Premia velocità in direzione della palla (pre-strike), così spinge davvero a camminare in avanti.
  """
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  robot: Entity = env.scene[cmd.cfg.entity_name]
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)

  rel = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  dir_xy = rel / torch.linalg.norm(rel, dim=1, keepdim=True).clamp_min(1e-6)

  v_xy = robot.data.root_link_lin_vel_w[:, :2]
  proj = torch.sum(v_xy * dir_xy, dim=1).clamp(min=0.0, max=float(max_speed)) / float(max_speed)
  return gate_pre * proj

def single_support_reward(
  env,
  command_name: str = "set_shot",
  left_contact_sensor_name: str = "p1_left_foot_ground_contact",
  right_contact_sensor_name: str = "p1_right_foot_ground_contact",
  upright_gate: float = 0.75,
  fz_thresh: float = 5.0,
  support_sign: str = "neg",
) -> torch.Tensor:
  gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
  up = upright_stability_reward(env)
  ok = (up > float(upright_gate))

  sup_l = _support_bool_from_ground_sensor(
    env,
    left_contact_sensor_name,
    fz_thresh,
    support_sign=support_sign,
  )
  sup_r = _support_bool_from_ground_sensor(
    env,
    right_contact_sensor_name,
    fz_thresh,
    support_sign=support_sign,
  )

  single = sup_l ^ sup_r  # XOR: uno solo a terra
  return gate_pre * ok.to(torch.float32) * single.to(torch.float32)

def strike_impulse_reward(
  env,
  command_name: str = "set_shot",
  max_speed: float = 6.0,
) -> torch.Tensor:
  """
  Reward per dare velocità alla palla subito dopo che è avvenuto lo strike.
  """
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  speed_xy = torch.linalg.norm(ball.data.root_link_lin_vel_w[:, :2], dim=1)
  speed_xy = speed_xy.clamp(min=0.0, max=float(max_speed)) / float(max_speed)

  return has_struck(env, command_name) * speed_xy


def foot_over_ball_penalty(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  right_foot_body_name: str = r"^right_foot_link$",
  xy_near: float = 0.16,
  z_margin: float = 0.015,
) -> torch.Tensor:
  """
  Penalizza quando un piede è molto vicino alla palla in XY ma si trova sopra la palla in Z.
  È il segnale più diretto per evitare di salirci sopra.
  """
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ids, _ = robot.find_bodies((left_foot_body_name, right_foot_body_name), preserve_order=True)
  if len(ids) != 2 or not hasattr(robot.data, "body_link_pos_w"):
    return torch.zeros(env.num_envs, device=env.device)

  left_idx = int(ids[0])
  right_idx = int(ids[1])

  feet_pos = robot.data.body_link_pos_w[:, [left_idx, right_idx], :]   # (N,2,3)
  ball_pos = ball.data.root_link_pos_w                                  # (N,3)

  d_xy = torch.linalg.norm(feet_pos[:, :, :2] - ball_pos[:, None, :2], dim=2)
  feet_z = feet_pos[:, :, 2]
  ball_z = ball_pos[:, 2].unsqueeze(1)

  over_ball = (d_xy < float(xy_near)) & (feet_z > (ball_z + float(z_margin)))
  return torch.any(over_ball, dim=1).to(torch.float32)


def clean_strike_reward(
  env,
  command_name: str = "set_shot",
  max_speed: float = 6.0,
  up_penalty: float = 0.45,
) -> torch.Tensor:
  """
  Premia un colpo pulito:
  - tanta velocità in avanti verso la porta
  - poca componente verticale positiva
  """
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_pos = ball.data.root_link_pos_w
  ball_vel = ball.data.root_link_lin_vel_w

  dir_xy = _normalize_xy(cmd.aim_pos_w[:, :2] - ball_pos[:, :2])
  v_forward = torch.sum(ball_vel[:, :2] * dir_xy, dim=1).clamp_min(0.0)
  v_up = ball_vel[:, 2].clamp_min(0.0)

  score = (v_forward - float(up_penalty) * v_up).clamp(min=0.0, max=float(max_speed))
  score = score / float(max_speed)

  return has_struck(env, command_name) * score

def goal_scored_event_reward(env, command_name: str = "set_shot") -> torch.Tensor:
  # stessa condizione della termination
  scored_now = goal_scored_termination(env, command_name)

  # latch per fare 1-shot event
  prev = _get_bool_state_buffer(env, key=f"p1_goal_prev::{command_name}")
  is_first = env.episode_length_buf <= 1
  prev[is_first] = False
  event = scored_now & (~prev)
  prev.copy_(scored_now)
  # opzionale: gate dopo strike (se vuoi evitare reward “random” su rimbalzi)
  return event.to(torch.float32)


def strike_from_above_penalty(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    left_sensor_name: str = "p1_left_foot_ball_contact",
    right_sensor_name: str = "p1_right_foot_ball_contact",
    left_foot_body_name: str = r"^left_foot_link$",
    right_foot_body_name: str = r"^right_foot_link$",
    xy_near: float = 0.18,
    z_margin: float = 0.02,
) -> torch.Tensor:
    # detect new_touch senza toccare p1_struck
    touching = _sensor_any_found(env, left_sensor_name) | _sensor_any_found(env, right_sensor_name)
    prev_touch = _get_bool_state_buffer(env, key=f"p1_prev_touch_above::{command_name}")
    is_first = env.episode_length_buf <= 1
    prev_touch[is_first] = False
    new_touch = touching & (~prev_touch)
    prev_touch.copy_(touching)

    # check over_ball come nella tua penalty
    robot: Entity = env.scene[asset_cfg.name]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    ids, _ = robot.find_bodies((left_foot_body_name, right_foot_body_name), preserve_order=True)
    if len(ids) != 2 or not hasattr(robot.data, "body_link_pos_w"):
        return torch.zeros(env.num_envs, device=env.device)

    feet_pos = robot.data.body_link_pos_w[:, [int(ids[0]), int(ids[1])], :]
    ball_pos = ball.data.root_link_pos_w

    d_xy = torch.linalg.norm(feet_pos[:, :, :2] - ball_pos[:, None, :2], dim=2)
    over_ball = (d_xy < float(xy_near)) & (feet_pos[:, :, 2] > (ball_pos[:, 2].unsqueeze(1) + float(z_margin)))

    bad_strike = new_touch & torch.any(over_ball, dim=1)
    return  bad_strike.to(torch.float32)

def action_rate_l2_prestrike(env, command_name: str = "set_shot") -> torch.Tensor:
    a = env.action_manager.action
    pa = env.action_manager.prev_action
    raw = torch.mean((a - pa) ** 2, dim=1)  # <-- mean, non sum
    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    return gate_pre * raw

def post_goal_upright_reward(env, command_name: str = "set_shot") -> torch.Tensor:
    # latch: una volta segnato, resta true fino a reset
    scored_now = goal_scored_termination(env, command_name)
    lat = _get_bool_state_buffer(env, key=f"p1_goal_lat::{command_name}")
    is_first = env.episode_length_buf <= 1
    lat[is_first] = False
    lat |= scored_now
    # premia la stabilità solo dopo goal
    return lat.to(torch.float32) * upright_stability_reward(env)

class FallTermination:
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
    fallen_now = (height < float(min_height)) | (tilt > float(max_tilt))
    self._counter = torch.where(
      fallen_now,
      self._counter + 1,
      torch.zeros_like(self._counter),
    )
    return self._counter >= int(consecutive_steps)

class SecondTouchContactTermination:
  """Terminate if the robot touches the ball more than once (foot contact sensors)."""

  def __init__(self, cfg, env):
    del cfg
    self._touch_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._touching = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._touch_count[env_ids] = 0
    self._touching[env_ids] = False

  def __call__(self, env, left_sensor_name: str, right_sensor_name: str) -> torch.Tensor:
    touching = _sensor_any_found(env, left_sensor_name) | _sensor_any_found(env, right_sensor_name)
    new_touch = touching & (~self._touching)

    self._touch_count += new_touch.long()
    self._touching = touching

    return self._touch_count > 1

class SecondTouchTermination:
  """
  Termina l'episodio se il robot tocca la palla più di una volta.
  Tocchi rilevati come "palla vicina a un piede" con isteresi.
  """

  def __init__(self, cfg, env):
    del cfg
    self._touch_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._touching = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    robot = env.scene["robot"]

    # Current mjlab body positions API.
    if not hasattr(robot.data, "body_link_pos_w"):
      raise RuntimeError(
        "SecondTouchTermination: expected robot.data.body_link_pos_w, "
        "but it is not available."
      )

    self._lf_i: int | None = None
    self._rf_i: int | None = None
    body_pattern_pairs = (
      (r"^left_foot_link$", r"^right_foot_link$"),
      (r"(?i)^left_foot.*$", r"(?i)^right_foot.*$"),
      (r"(?i).*left.*foot.*$", r"(?i).*right.*foot.*$"),
    )
    for left_pattern, right_pattern in body_pattern_pairs:
      ids, _ = robot.find_bodies((left_pattern, right_pattern), preserve_order=True)
      if len(ids) == 2:
        self._lf_i = int(ids[0])
        self._rf_i = int(ids[1])
        break

    if self._lf_i is None or self._rf_i is None:
      raise RuntimeError(
        "SecondTouchTermination: could not resolve left/right foot body indices. "
        f"Available bodies: {robot.body_names}"
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._touch_count[env_ids] = 0
    self._touching[env_ids] = False

  def __call__(
    self,
    env,
    command_name: str = "set_shot",  # unused, ma lo lasciamo per coerenza
    touch_enter: float = 0.16,
    touch_exit: float = 0.22,
  ) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["soccer_ball"]

    ball_xy = ball.data.root_link_pos_w[:, :2]
    body_pos_w = robot.data.body_link_pos_w
    feet_xy = body_pos_w[:, [self._lf_i, self._rf_i], :2]  # (N,2,2)

    d_l = torch.linalg.norm(ball_xy - feet_xy[:, 0, :], dim=1)
    d_r = torch.linalg.norm(ball_xy - feet_xy[:, 1, :], dim=1)
    d_min = torch.minimum(d_l, d_r)

    touching_now = torch.where(self._touching, d_min < touch_exit, d_min < touch_enter)
    new_touch = touching_now & (~self._touching)

    self._touch_count = self._touch_count + new_touch.long()
    self._touching = touching_now

    return self._touch_count > 1

def ball_out_of_play_termination(
  env,
  command_name: str = "set_shot",
  field_half_length_x: float = 7.0,
  field_half_width_y: float = 4.5,
  goal_opening_half_width: float = 1.55,
  margin: float = 0.10,
) -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  # Se ha già segnato, NON terminare per "fuori campo"
  already_scored = cmd._goal_scored

  origins = env.scene.env_origins
  ball_local = ball.data.root_link_pos_w - origins
  x, y, z = ball_local[:, 0], ball_local[:, 1], ball_local[:, 2]

  out_sidelines = torch.abs(y) > (field_half_width_y + margin)
  out_back = x < (-field_half_length_x - margin)

  crossed = x >= float(cmd.cfg.goal_line_x)

  # Tiro wide: oltre la goal line ma fuori apertura porta
  wide = crossed & (torch.abs(y) > goal_opening_half_width)

  # Dentro apertura porta ma sotto/sopra (non è goal)
  over_or_under = crossed & (torch.abs(y) <= goal_opening_half_width) & (
    (z < float(cmd.cfg.goal_z_min)) | (z > float(cmd.cfg.goal_z_max))
  )

  out = out_sidelines | out_back | wide | over_or_under
  return (~already_scored) & out