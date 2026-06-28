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


def _shot_frame_basis(ball_xy: torch.Tensor, aim_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """
  Build shot-frame basis from ball -> aim direction.
  Returns:
    shot_dir: unit vector toward aim
    side_dir: left-orthogonal unit vector [-shot_y, shot_x]
  """
  raw = aim_xy - ball_xy
  norm = torch.linalg.norm(raw, dim=1, keepdim=True)
  fallback = torch.zeros_like(raw)
  fallback[:, 0] = 1.0
  shot_dir = torch.where(norm > 1.0e-6, raw / norm.clamp_min(1.0e-6), fallback)
  side_dir = torch.stack([-shot_dir[:, 1], shot_dir[:, 0]], dim=1)
  return shot_dir, side_dir


def _project_point_to_shot_frame(
  point_xy: torch.Tensor,
  ball_xy: torch.Tensor,
  shot_dir: torch.Tensor,
  side_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  rel = point_xy - ball_xy
  dx_shot = torch.sum(rel * shot_dir, dim=1)
  dy_shot = torch.sum(rel * side_dir, dim=1)
  return dx_shot, dy_shot


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
  # Spawn mode:
  # - "world_ranges": legacy behavior using x/y ranges above
  # - "shot_line": spawn behind ball on current ball->aim line
  striker_spawn_mode: str = "world_ranges"
  setup_side_sign: float = 1.0
  striker_distance_behind_ball: float = 0.38
  striker_lateral_offset: float = 0.0
  striker_longitudinal_jitter: float = 0.01
  striker_lateral_jitter: float = 0.0

  ball_spawn_x_range: tuple[float, float] = (0.0, 0.0)
  ball_spawn_y_range: tuple[float, float] = (0.0, 0.0)
  ball_spawn_z: float = 0.11

  # aim point (center goal) in env-local coordinates (before origin)
  aim_x: float = 7.3
  aim_y: float = 0.0
  aim_z: float = 0.0
  # Explicit visual-corner mapping for lateral target selection.
  visual_left_corner_y: float = -1.0
  visual_right_corner_y: float = 1.0
  lateral_target_mode: str = "fixed"   # allowed: "fixed", "random_binary"
  fixed_target_corner: str = "left"    # allowed: "left", "right"

  # area bounds in env-local coordinates
  striker_area_bounds: tuple[float, float, float, float] = (-1.0, 7.0, -2.0, 2.0)
  hard_area_margin: float = 0.5

  # goal check
  goal_line_x: float = 7.0
  goal_y_half: float = 1.0

  goal_z_min: float = 0.0
  goal_z_max: float = 1.85


  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  kick_only_reset_prob: float = 0.0
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
    self._sampled_aim_y = torch.zeros(env.num_envs, device=self.device)
    self._sampled_aim_y_valid = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
    # Last sampled ball spawn position in world XY for deterministic shot-line spawns.
    self._ball_spawn_xy_w = torch.zeros(env.num_envs, 2, device=self.device)

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
  def sampled_aim_y(self) -> torch.Tensor:
    return self._sampled_aim_y

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
    reset_kick_phase_buffers(self._env, env_ids, command_name="set_shot")
    self._goal_scored[env_ids] = False
    self.metrics["goal_event"][env_ids] = 0.0
    self._reset_ball_pose(env_ids)
    self._sample_lateral_target_y(env_ids)
    self._set_aim_pose(env_ids)
    self._reset_robot_pose(env_ids)
    kick_only_flag = _get_bool_state_buffer(self._env, key="p1_kick_only_reset_flag::set_shot")
    kick_only_flag[env_ids] = False
    kick_prob = float(self.cfg.kick_only_reset_prob)
    if kick_prob > 0.0:
      sample = torch.rand((len(env_ids),), device=self.device) < kick_prob
      if torch.any(sample):
        initialize_kick_phase_state(
          self._env,
          env_ids[sample],
          command_name="set_shot",
        )

    # Stage-1 command vector: deterministic zeros
    self._command[env_ids] = 0.0

  def _update_command(self) -> None:
    # fixed for whole episode
    pass

  def _sample_lateral_target_y(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    left_y = float(self.cfg.visual_left_corner_y)
    right_y = float(self.cfg.visual_right_corner_y)
    mode = str(self.cfg.lateral_target_mode).lower()
    fixed_corner = str(self.cfg.fixed_target_corner).lower()

    if mode == "fixed":
      if fixed_corner == "left":
        sampled = torch.full((len(env_ids),), left_y, device=self.device)
      elif fixed_corner == "right":
        sampled = torch.full((len(env_ids),), right_y, device=self.device)
      else:
        raise ValueError(
          f"Invalid fixed_target_corner='{self.cfg.fixed_target_corner}'. "
          "Expected 'left' or 'right'."
        )
    elif mode == "random_binary":
      choose_left = torch.rand((len(env_ids),), device=self.device) < 0.5
      sampled = torch.where(
        choose_left,
        torch.full((len(env_ids),), left_y, device=self.device),
        torch.full((len(env_ids),), right_y, device=self.device),
      )
    else:
      raise ValueError(
        f"Invalid lateral_target_mode='{self.cfg.lateral_target_mode}'. "
        "Expected 'fixed' or 'random_binary'."
      )

    self._sampled_aim_y[env_ids] = sampled
    self._sampled_aim_y_valid[env_ids] = True

  def _set_aim_pose(self, env_ids: torch.Tensor) -> None:
    origins = self._env.scene.env_origins[env_ids]

    sampled_aim_y = self._sampled_aim_y[env_ids]
    sampled_valid = self._sampled_aim_y_valid[env_ids]
    fallback_aim_y = torch.full_like(sampled_aim_y, float(self.cfg.aim_y))
    sampled_aim_y = torch.where(sampled_valid, sampled_aim_y, fallback_aim_y)

    self._aim_pos_w[env_ids, 0] = origins[:, 0] + float(self.cfg.goal_line_x)
    self._aim_pos_w[env_ids, 1] = origins[:, 1] + sampled_aim_y
    self._aim_pos_w[env_ids, 2] = origins[:, 2] + float(self.cfg.aim_z)

  def _debug_vis_impl(self, visualizer) -> None:
    if not self.cfg.debug_vis:
      return

    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    for batch in env_indices:
      center = self._aim_pos_w[batch]
      if not torch.isfinite(center).all():
        continue

      visualizer.add_sphere(
        center.cpu().numpy(),
        radius=0.12,
        color=(1.0, 0.0, 0.0, 0.9),
        label=f"set_shot_target_sphere_{batch}",
      )

  def queue_viser_overlays(self, visualizer) -> None:
    # Reuse the command debug marker path so target cues are visible in Viser too.
    self._debug_vis_impl(visualizer)

  def _reset_robot_pose(self, env_ids: torch.Tensor) -> None:
    default_root_state = self._robot.data.default_root_state
    default_joint_pos = self._robot.data.default_joint_pos
    default_joint_vel = self._robot.data.default_joint_vel
    assert default_root_state is not None
    assert default_joint_pos is not None
    assert default_joint_vel is not None

    root_state = default_root_state[env_ids].clone()
    yaw_lo, yaw_hi = self.cfg.spawn_yaw_range
    if abs(yaw_hi - yaw_lo) <= 1.0e-9:
      yaw = torch.full((len(env_ids),), float(yaw_lo), device=self.device)
    else:
      yaw = _sample_uniform_range(yaw_lo, yaw_hi, len(env_ids), self.device)

    spawn_mode = str(self.cfg.striker_spawn_mode).lower()
    if spawn_mode == "shot_line":
      # Use the sampled reset position (written in _reset_ball_pose) to avoid stale sim reads.
      ball_xy = self._ball_spawn_xy_w[env_ids, :2]
      aim_xy = self._aim_pos_w[env_ids, :2]

      long_jitter = _sample_uniform_range(
        -float(self.cfg.striker_longitudinal_jitter),
        float(self.cfg.striker_longitudinal_jitter),
        len(env_ids),
        self.device,
      )
      lat_jitter = _sample_uniform_range(
        -float(self.cfg.striker_lateral_jitter),
        float(self.cfg.striker_lateral_jitter),
        len(env_ids),
        self.device,
      )

      shot_dir, side_dir = _shot_frame_basis(ball_xy, aim_xy)
      setup_side = float(getattr(self.cfg, "setup_side_sign", 1.0))

      spawn_xy = (
        ball_xy
        - float(self.cfg.striker_distance_behind_ball) * shot_dir
        + setup_side * float(self.cfg.striker_lateral_offset) * side_dir
        + long_jitter.unsqueeze(1) * shot_dir
        + setup_side * lat_jitter.unsqueeze(1) * side_dir
      )

      root_state[:, 0:2] = spawn_xy

      # Face the sampled shot target from the actual spawn position
      face_vec = aim_xy - spawn_xy
      face_dir = _normalize_xy(face_vec)

      yaw_base = torch.atan2(face_dir[:, 1], face_dir[:, 0])
      yaw_total = yaw_base + yaw
      yaw_q = quat_from_euler_xyz(
        torch.zeros_like(yaw_total),
        torch.zeros_like(yaw_total),
        yaw_total,
      )
      root_state[:, 3:7] = yaw_q
    else:
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
      origins = self._env.scene.env_origins[env_ids]
      root_state[:, 0] = origins[:, 0] + spawn_x
      root_state[:, 1] = origins[:, 1] + spawn_y

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
    self._ball_spawn_xy_w[env_ids] = root_state[:, 0:2]

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


def kick_phase_flag_obs(
  env,
  command_name: str = "set_shot",
) -> torch.Tensor:
  return kick_phase_mask(env, command_name).to(torch.float32).unsqueeze(-1)


def kick_only_reset_flag_obs(
  env,
  command_name: str = "set_shot",
) -> torch.Tensor:
  flag = _get_bool_state_buffer(env, key=f"p1_kick_only_reset_flag::{command_name}")
  return flag.to(torch.float32).unsqueeze(-1)


def yaw_error_abs_obs(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  yaw_error_abs = torch.abs(_compute_yaw_error(robot, cmd.aim_pos_w)).to(torch.float32)
  return yaw_error_abs.unsqueeze(-1)


def ball_dist_xy_obs(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  dist_xy = torch.linalg.norm(
    ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2],
    dim=1,
  ).to(torch.float32)
  return dist_xy.unsqueeze(-1)


def right_foot_pos_rel_ball_xy_obs(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  right_foot_body_name: str = r"^right_foot_link$",
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  zeros = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
  ids, _ = robot.find_bodies((right_foot_body_name,), preserve_order=True)
  if len(ids) != 1 or not hasattr(robot.data, "body_link_pos_w"):
    return zeros

  right_idx = int(ids[0])
  rel_xy = robot.data.body_link_pos_w[:, right_idx, :2] - ball.data.root_link_pos_w[:, :2]
  return rel_xy.to(torch.float32)


def left_foot_pos_rel_ball_xy_obs(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  zeros = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
  ids, _ = robot.find_bodies((left_foot_body_name,), preserve_order=True)
  if len(ids) != 1 or not hasattr(robot.data, "body_link_pos_w"):
    return zeros

  left_idx = int(ids[0])
  rel_xy = robot.data.body_link_pos_w[:, left_idx, :2] - ball.data.root_link_pos_w[:, :2]
  return rel_xy.to(torch.float32)


def left_support_latched_error_xy_obs(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  zeros = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
  ids, _ = robot.find_bodies((left_foot_body_name,), preserve_order=True)
  if len(ids) != 1 or not hasattr(robot.data, "body_link_pos_w"):
    return zeros

  latched_xy = _get_float_state_buffer(
    env,
    key=f"p1_left_foot_latched_xy::{command_name}",
    shape_tail=(2,),
    dtype=torch.float32,
  )
  left_idx = int(ids[0])
  current_xy = robot.data.body_link_pos_w[:, left_idx, :2].to(torch.float32)
  err_xy = current_xy - latched_xy
  support_visible = torch.maximum(
    kick_phase_mask(env, command_name),
    post_strike_support_lock_mask(env, command_name, lock_steps=12),
  ).to(torch.float32).unsqueeze(-1)
  return err_xy * support_visible


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


def pre_strike_yaw_alignment_reward(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    k: float = 3.0,
    min_height: float = 0.53,
    max_tilt: float = 0.60,
) -> torch.Tensor:
    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    posture_gate = posture_priority_gate_latched(
        env,
        asset_cfg=asset_cfg,
        min_height=min_height,
        max_tilt=max_tilt,
    )
    return gate_pre * posture_gate * yaw_alignment_reward(
        env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        k=k,
    )


def _default_root_height(robot: Entity) -> torch.Tensor:
    default_root_state = getattr(robot.data, "default_root_state", None)
    if default_root_state is not None:
        return default_root_state[:, 2].to(robot.data.root_link_pos_w.dtype)
    return robot.data.root_link_pos_w[:, 2]


def striker_posture_score(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    roll_band: float = 0.07,
    roll_sigma: float = 0.12,
    pitch_target: float = 0.14,
    pitch_band: float = 0.12,
    pitch_sigma: float = 0.25,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    proj_g = robot.data.projected_gravity_b

    sagittal = proj_g[:, 0]
    lateral = proj_g[:, 1]

    roll_error = (torch.abs(lateral) - float(roll_band)).clamp_min(0.0)
    roll_score = torch.exp(-torch.square(roll_error) / max(float(roll_sigma) ** 2, 1.0e-6))

    pitch_error = (torch.abs(sagittal - float(pitch_target)) - float(pitch_band)).clamp_min(0.0)
    pitch_score = torch.exp(-torch.square(pitch_error) / max(float(pitch_sigma) ** 2, 1.0e-6))

    return roll_score * pitch_score


def upright_stability_reward(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    height_target: float | None = None,
    height_sigma: float = 0.14,
    roll_band: float = 0.07,
    roll_sigma: float = 0.12,
    pitch_target: float = 0.14,
    pitch_band: float = 0.12,
    pitch_sigma: float = 0.25,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    height = robot.data.root_link_pos_w[:, 2]

    if height_target is None:
        target = _default_root_height(robot)
    else:
        target = torch.full_like(height, float(height_target))

    posture = striker_posture_score(
        env,
        asset_cfg=asset_cfg,
        roll_band=roll_band,
        roll_sigma=roll_sigma,
        pitch_target=pitch_target,
        pitch_band=pitch_band,
        pitch_sigma=pitch_sigma,
    )

    height_err = (height - target) / max(float(height_sigma), 1.0e-6)
    height_score = torch.exp(-torch.square(height_err))

    return posture * height_score


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


def _get_float_state_buffer(
  env,
  key: str,
  shape_tail: tuple = (),
  dtype=torch.float32,
) -> torch.Tensor:
  """Persistent per-env float buffer stored on env.unwrapped."""
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_p1_float_state_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  shape = (env.num_envs, *shape_tail)
  buf = cache.get(key)
  if (
    buf is None
    or tuple(buf.shape) != shape
    or buf.device != env.device
    or buf.dtype != dtype
  ):
    buf = torch.zeros(shape, device=env.device, dtype=dtype)
    cache[key] = buf
  return buf


def reset_kick_phase_buffers(
  env,
  env_ids: torch.Tensor,
  command_name: str = "set_shot",
):
  if env_ids.numel() == 0:
    return

  bool_keys = (
    f"p1_kick_phase::{command_name}",
    f"p1_struck::{command_name}",
    f"p1_support_ok_on_valid_strike::{command_name}",
    f"p1_kick_phase_enter_paid::{command_name}",
    f"p1_kick_only_reset_flag::{command_name}",
    f"p1_kick_only_strike_paid::{command_name}",
    f"p1_supported_r_only_prev_r::{command_name}",
    f"p1_supported_r_only_paid::{command_name}",
    f"p1_supported_strike_prev_r::{command_name}",
    f"p1_weak_touch_prev_r::{command_name}",
    f"p1_prev_touch::{command_name}",
    f"p1_prev_depart::{command_name}",
    f"p1_touch_prev::{command_name}",
    f"p1_right_touch_seen::{command_name}",
  )
  for key in bool_keys:
    _get_bool_state_buffer(env, key=key)[env_ids] = False

  int_keys = (
    f"p1_kick_good_count::{command_name}",
    f"p1_kick_phase_hold_count::{command_name}",
    f"p1_kick_phase_count::{command_name}",
    f"p1_touch_count::{command_name}",
    f"p1_post_strike_support_lock_count::{command_name}",
  )
  for key in int_keys:
    _get_int_state_buffer(env, key=key, dtype=torch.int32)[env_ids] = 0

  _get_float_state_buffer(
    env,
    key=f"p1_left_foot_latched_xy::{command_name}",
    shape_tail=(2,),
    dtype=torch.float32,
  )[env_ids] = 0.0


def initialize_kick_phase_state(
  env,
  env_ids: torch.Tensor,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  target_dx: float = -0.02,
  target_abs_dy: float = 0.11,
):
  if env_ids.numel() == 0:
    return

  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  kick_phase = _get_bool_state_buffer(env, key=f"p1_kick_phase::{command_name}")
  struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
  support_latch = _get_bool_state_buffer(
    env,
    key=f"p1_support_ok_on_valid_strike::{command_name}",
  )
  enter_paid = _get_bool_state_buffer(env, key=f"p1_kick_phase_enter_paid::{command_name}")
  kick_only_flag = _get_bool_state_buffer(env, key=f"p1_kick_only_reset_flag::{command_name}")

  good_count = _get_int_state_buffer(env, key=f"p1_kick_good_count::{command_name}", dtype=torch.int32)
  hold_count = _get_int_state_buffer(env, key=f"p1_kick_phase_hold_count::{command_name}", dtype=torch.int32)
  phase_count = _get_int_state_buffer(env, key=f"p1_kick_phase_count::{command_name}", dtype=torch.int32)

  latched_xy = _get_float_state_buffer(
    env,
    key=f"p1_left_foot_latched_xy::{command_name}",
    shape_tail=(2,),
    dtype=torch.float32,
  )

  # Build a physically coherent kick-ready pose without IK in the SHOT FRAME:
  # - left support near the ball
  # - right foot slightly behind the ball (kick preload)
  # - root facing shot target, zero root/joint velocities
  ball_xy = ball.data.root_link_pos_w[env_ids, :2]
  aim_xy = cmd.aim_pos_w[env_ids, :2]
  shot_dir, side_dir = _shot_frame_basis(ball_xy, aim_xy)

  setup_side = float(getattr(cmd.cfg, "setup_side_sign", 1.0))

  desired_left_xy = (
    ball_xy
    + float(target_dx) * shot_dir
    + setup_side * float(target_abs_dy) * side_dir
  ).to(torch.float32)

  if (
    hasattr(robot.data, "body_link_pos_w")
    and hasattr(robot.data, "root_state_w")
    and hasattr(robot.data, "default_joint_pos")
    and hasattr(robot.data, "default_joint_vel")
  ):
    body_ids, _ = robot.find_bodies((left_foot_body_name, r"^right_foot_link$"), preserve_order=True)
    if len(body_ids) == 2:
      left_idx = int(body_ids[0])
      right_idx = int(body_ids[1])

      current_left_xy = robot.data.body_link_pos_w[env_ids, left_idx, :2]
      current_right_xy = robot.data.body_link_pos_w[env_ids, right_idx, :2]
      current_root_state = robot.data.root_state_w[env_ids].clone()

      right_target_dx = float(target_dx) - 0.10
      right_target_abs_dy = max(float(target_abs_dy) - 0.03, 0.05)
      desired_right_xy = (
        ball_xy
        + right_target_dx * shot_dir
        - setup_side * right_target_abs_dy * side_dir
      ).to(torch.float32)

      # Single rigid root shift that approximately satisfies both feet targets.
      delta_left = desired_left_xy - current_left_xy
      delta_right = desired_right_xy - current_right_xy
      delta_xy = 0.6 * delta_left + 0.4 * delta_right
      current_root_state[:, 0:2] += delta_xy

      face_vec = aim_xy - current_root_state[:, 0:2]
      face_dir = _normalize_xy(face_vec)

      yaw = torch.atan2(face_dir[:, 1], face_dir[:, 0])
      yaw_quat = quat_from_euler_xyz(
        torch.zeros((env_ids.numel(),), device=env.device),
        torch.zeros((env_ids.numel(),), device=env.device),
        yaw,
      )
      current_root_state[:, 3:7] = yaw_quat
      current_root_state[:, 7:13] = 0.0
      robot.write_root_state_to_sim(current_root_state, env_ids=env_ids)

      # Small safe joint preload for a short right-foot swing.
      joint_pos = robot.data.default_joint_pos[env_ids].clone()
      joint_vel = robot.data.default_joint_vel[env_ids].clone()
      joint_vel[:] = 0.0

      def _apply_joint_offset(patterns: tuple[str, ...], delta: float) -> None:
        idx = _find_joint_idx(robot, patterns)
        if idx is None:
          return
        if idx < 0 or idx >= joint_pos.shape[1]:
          return
        joint_pos[:, idx] += float(delta)

      # Left support: more stable/extended.
      _apply_joint_offset((r"left.*hip.*roll", r"l_hip_roll"), 0.04)
      _apply_joint_offset((r"left.*knee", r"l_kne"), 0.02)
      _apply_joint_offset((r"left.*ankle.*pitch", r"l_ankle_pitch"), -0.04)

      # Right kick leg: loaded.
      _apply_joint_offset((r"right.*hip.*pitch", r"r_hip_pitch"), -0.10)
      _apply_joint_offset((r"right.*knee", r"r_kne"), 0.10)
      _apply_joint_offset((r"right.*ankle.*pitch", r"r_ankle_pitch"), -0.08)

      # Torso upright-ish (no-op if joint not present).
      _apply_joint_offset((r"waist", r"torso"), 0.0)

      robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
      robot.clear_state(env_ids=env_ids)

  kick_phase[env_ids] = True
  struck[env_ids] = False
  support_latch[env_ids] = False
  enter_paid[env_ids] = False
  kick_only_flag[env_ids] = True

  good_count[env_ids] = 0
  hold_count[env_ids] = 0
  phase_count[env_ids] = 0

  latched_xy[env_ids] = desired_left_xy


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
    ball_depart_xy: float = 0.08,
    min_vx: float = 0.25,
    min_speed_xy: float = 0.45,
    require_right_touch: bool = True,
) -> torch.Tensor:
    """
    Robust strike latch:
    - sensor new-touch OR
    - ball clearly departs from spawn with enough forward/XY speed
    """
    touching_left = _sensor_any_found(env, left_sensor_name)
    touching_right = _sensor_any_found(env, right_sensor_name)
    touching = touching_right if require_right_touch else (touching_left | touching_right)

    prev_touch = _get_bool_state_buffer(env, key=f"p1_prev_touch::{command_name}")
    is_first = env.episode_length_buf <= 1
    prev_touch[is_first] = False

    new_touch = touching & (~prev_touch)
    prev_touch.copy_(touching)

    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    origins = env.scene.env_origins
    ball_local = ball.data.root_link_pos_w - origins

    spawn_x = float(cmd.cfg.ball_spawn_x_range[0])
    spawn_y = float(cmd.cfg.ball_spawn_y_range[0])

    dx = ball_local[:, 0] - spawn_x
    dy = ball_local[:, 1] - spawn_y
    depart_xy = torch.sqrt(dx * dx + dy * dy)

    vel = ball.data.root_link_lin_vel_w
    speed_xy = torch.linalg.norm(vel[:, :2], dim=1)

    departed = (
        (depart_xy > float(ball_depart_xy))
        & (vel[:, 0] > float(min_vx))
        & (speed_xy > float(min_speed_xy))
    )

    prev_depart = _get_bool_state_buffer(env, key=f"p1_prev_depart::{command_name}")
    prev_depart[is_first] = False
    new_depart = departed & (~prev_depart)
    prev_depart.copy_(departed)

    if require_right_touch:
        right_touch_seen = _get_bool_state_buffer(env, key=f"p1_right_touch_seen::{command_name}")
        right_touch_seen[is_first] = False
        right_touch_seen |= touching_right
        new_strike = new_touch | (new_depart & right_touch_seen)
    else:
        new_strike = new_touch | new_depart

    posture_ok = posture_priority_gate_latched(
        env,
        min_height=0.53,
        max_tilt=0.60,
    ) > 0.5

    new_strike = new_strike & posture_ok

    struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
    struck[is_first] = False

    reward = new_strike & (~struck)
    struck |= new_strike

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

def post_strike_phase_mask(
    env,
    command_name: str = "set_shot",
    ball_depart_xy: float = 0.08,
    min_vx: float = 0.5,
    min_speed_xy: float = 0.8,
) -> torch.Tensor:
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    origins = env.scene.env_origins
    ball_local = ball.data.root_link_pos_w - origins

    spawn_local_xy = cmd._ball_spawn_xy_w - origins[:, :2]
    spawn_x = spawn_local_xy[:, 0]
    spawn_y = spawn_local_xy[:, 1]

    dx = ball_local[:, 0] - spawn_x
    dy = ball_local[:, 1] - spawn_y
    depart_xy = torch.sqrt(dx * dx + dy * dy)

    vel = ball.data.root_link_lin_vel_w
    speed_xy = torch.linalg.norm(vel[:, :2], dim=1)

    departed = (
        (depart_xy > float(ball_depart_xy))
        & (vel[:, 0] > float(min_vx))
        & (speed_xy > float(min_speed_xy))
    )

    return torch.maximum(
        has_struck(env, command_name),
        departed.to(torch.float32),
    )


def post_strike_upright_reward(env, command_name: str = "set_shot") -> torch.Tensor:
  return post_strike_phase_mask(env, command_name) * upright_stability_reward(env)


def post_strike_yaw_alignment_reward(
  env,
  command_name: str = "set_shot",
  **kwargs,
) -> torch.Tensor:
  return post_strike_phase_mask(env, command_name) * yaw_alignment_reward(
    env,
    command_name=command_name,
    **kwargs,
  )



def post_strike_base_speed_penalty(
  env,
  command_name: str = "set_shot",
  max_speed: float = 1.0,
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  speed_xy = torch.linalg.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
  return post_strike_phase_mask(env, command_name) * (
      speed_xy.clamp(min=0.0, max=float(max_speed)) / float(max_speed)
  )



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
  scored_now = goal_scored_termination(env, command_name)

  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_local = _world_to_env_local_xy(env, ball.data.root_link_pos_w[:, :2])
  y = ball_local[:, 1]
  z = ball.data.root_link_pos_w[:, 2]

  good = (z >= z_min) & (torch.abs(y) >= y_side_min)
  return (scored_now & good).to(torch.float32)

def goal_high_left_corner_reward(
  env,
  command_name: str = "set_shot",
  z_min: float = 1.10,
  y_left_min: float = 0.80,
  left_sign: float = 1.0,
) -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_pos = ball.data.root_link_pos_w
  origins = env.scene.env_origins
  ball_local = ball_pos - origins
  x, y, z = ball_local[:, 0], ball_local[:, 1], ball_local[:, 2]

  crossed = x >= float(cmd.cfg.goal_line_x)
  inside_y = torch.abs(y) <= float(cmd.cfg.goal_y_half)
  inside_z = (z >= float(cmd.cfg.goal_z_min)) & (z <= float(cmd.cfg.goal_z_max))
  scored_now = crossed & inside_y & inside_z

  prev = _get_bool_state_buffer(env, key=f"p1_goal_high_left_prev::{command_name}")
  is_first = env.episode_length_buf <= 1
  prev[is_first] = False
  event = scored_now & (~prev)
  prev.copy_(scored_now)

  signed_y = float(left_sign) * y
  y_score = torch.sigmoid((signed_y - float(y_left_min)) / 0.10)
  z_score = torch.sigmoid((z - float(z_min)) / 0.10)
  return event.to(torch.float32) * y_score * z_score


def goal_top_left_target_reward(
  env,
  command_name: str = "set_shot",
  target_z: float = 1.34,
  sigma_z: float = 0.16,
  target_left_y: float = 1.08,
  sigma_y: float = 0.18,
  left_sign: float = -1.0,
) -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_pos = ball.data.root_link_pos_w
  origins = env.scene.env_origins
  ball_local = ball_pos - origins
  x = ball_local[:, 0]
  y = ball_local[:, 1]
  z = ball_local[:, 2]

  crossed = x >= float(cmd.cfg.goal_line_x)
  inside_y = torch.abs(y) <= float(cmd.cfg.goal_y_half)
  inside_z = (z >= float(cmd.cfg.goal_z_min)) & (z <= float(cmd.cfg.goal_z_max))
  scored_now = crossed & inside_y & inside_z

  prev = _get_bool_state_buffer(env, key=f"p1_goal_top_left_prev::{command_name}")
  is_first = env.episode_length_buf <= 1
  prev[is_first] = False
  event = scored_now & (~prev)
  prev.copy_(scored_now)

  signed_y = float(left_sign) * y
  sigma_y_safe = max(float(sigma_y), 1.0e-6)
  sigma_z_safe = max(float(sigma_z), 1.0e-6)
  y_score = torch.exp(-0.5 * torch.square((signed_y - float(target_left_y)) / sigma_y_safe))
  z_score = torch.exp(-0.5 * torch.square((z - float(target_z)) / sigma_z_safe))

  posture_gate = posture_priority_gate_latched(env, min_height=0.53, max_tilt=0.60)
  return posture_gate * event.to(torch.float32) * y_score * z_score


def goal_target_from_command_reward(
  env,
  command_name: str = "set_shot",
  sigma_y: float = 0.18,
  sigma_z: float = 0.16,
) -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_pos = ball.data.root_link_pos_w
  origins = env.scene.env_origins
  ball_local = ball_pos - origins
  x = ball_local[:, 0]
  y = ball_local[:, 1]
  z = ball_local[:, 2]

  crossed = x >= float(cmd.cfg.goal_line_x)
  inside_y = torch.abs(y) <= float(cmd.cfg.goal_y_half)
  inside_z = (z >= float(cmd.cfg.goal_z_min)) & (z <= float(cmd.cfg.goal_z_max))
  scored_now = crossed & inside_y & inside_z

  prev = _get_bool_state_buffer(env, key=f"p1_goal_target_from_cmd_prev::{command_name}")
  is_first = env.episode_length_buf <= 1
  prev[is_first] = False
  event = scored_now & (~prev)
  prev.copy_(scored_now)

  aim_local = cmd.aim_pos_w - origins
  target_y = aim_local[:, 1]
  target_z = aim_local[:, 2]

  sigma_y_safe = max(float(sigma_y), 1.0e-6)
  sigma_z_safe = max(float(sigma_z), 1.0e-6)
  y_score = torch.exp(-0.5 * torch.square((y - target_y) / sigma_y_safe))
  z_score = torch.exp(-0.5 * torch.square((z - target_z) / sigma_z_safe))

  return event.to(torch.float32) * y_score * z_score


def goal_low_or_center_penalty(
  env,
  command_name: str = "set_shot",
  z_min: float = 0.55,
  y_side_min: float = 0.55,
) -> torch.Tensor:
  scored_now = goal_scored_termination(env, command_name)

  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_local = _world_to_env_local_xy(env, ball.data.root_link_pos_w[:, :2])
  y = ball_local[:, 1]
  z = ball.data.root_link_pos_w[:, 2]

  bad = (z < z_min) | (torch.abs(y) < y_side_min)
  return (scored_now & bad).to(torch.float32)


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


def left_support_valid_now_mask(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  left_ground_sensor_name: str = "p1_left_foot_ground_contact",
  left_ball_sensor_name: str = "p1_left_foot_ball_contact",
  target_dx: float = -0.02,
  dx_tol: float = 0.16,
  target_abs_dy: float = 0.11,
  dy_tol: float = 0.12,
  max_left_speed: float = 0.50,
  min_height: float = 0.53,
  max_tilt: float = 0.60,
) -> torch.Tensor:
  """
  Hard per-step validity mask for the LEFT support foot.
  Returns bool tensor [num_envs] and does not depend on has_struck.
  """
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ids, _ = robot.find_bodies((left_foot_body_name,), preserve_order=True)
  if (
    len(ids) != 1
    or not hasattr(robot.data, "body_link_pos_w")
    or not hasattr(robot.data, "body_com_lin_vel_w")
  ):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  left_idx = int(ids[0])
  left_pos = robot.data.body_link_pos_w[:, left_idx, :]
  left_vel = robot.data.body_com_lin_vel_w[:, left_idx, :]
  ball_pos = ball.data.root_link_pos_w

  left_on_ground = _support_bool_from_ground_sensor(
    env,
    left_ground_sensor_name,
    fz_thresh=3.0,
    support_sign="neg",
  )
  left_not_touch_ball = ~_sensor_any_found(env, left_ball_sensor_name)

  root_height = robot.data.root_link_pos_w[:, 2]
  tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
  posture_ok = (root_height >= float(min_height)) & (tilt <= float(max_tilt))

  shot_dir, side_dir = _shot_frame_basis(
    ball_pos[:, :2],
    cmd.aim_pos_w[:, :2],
  )
  dx, dy = _project_point_to_shot_frame(
    left_pos[:, :2],
    ball_pos[:, :2],
    shot_dir,
    side_dir,
  )
  dx_ok = torch.abs(dx - float(target_dx)) <= float(dx_tol)
  dy_ok = torch.abs(torch.abs(dy) - float(target_abs_dy)) <= float(dy_tol)

  left_speed_xy = torch.linalg.norm(left_vel[:, :2], dim=1)
  speed_ok = left_speed_xy <= float(max_left_speed)

  return left_on_ground & left_not_touch_ball & posture_ok & dx_ok & dy_ok & speed_ok


def left_support_ready_now_mask(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  left_ground_sensor_name: str = "p1_left_foot_ground_contact",
  left_ball_sensor_name: str = "p1_left_foot_ball_contact",
  target_dx: float = -0.02,
  dx_tol: float = 0.22,
  target_abs_dy: float = 0.11,
  dy_tol: float = 0.16,
  max_left_speed: float = 0.65,
  min_height: float = 0.50,
  max_tilt: float = 0.75,
) -> torch.Tensor:
  """
  Softer per-step readiness mask for left support foot.
  Returns bool tensor [num_envs], intentionally more permissive than
  left_support_valid_now_mask.
  """
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ids, _ = robot.find_bodies((left_foot_body_name,), preserve_order=True)
  if (
    len(ids) != 1
    or not hasattr(robot.data, "body_link_pos_w")
    or not hasattr(robot.data, "body_com_lin_vel_w")
  ):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  left_idx = int(ids[0])
  left_pos = robot.data.body_link_pos_w[:, left_idx, :]
  left_vel = robot.data.body_com_lin_vel_w[:, left_idx, :]
  ball_pos = ball.data.root_link_pos_w

  left_on_ground = _support_bool_from_ground_sensor(
    env,
    left_ground_sensor_name,
    fz_thresh=3.0,
    support_sign="neg",
  )
  left_not_touch_ball = ~_sensor_any_found(env, left_ball_sensor_name)

  root_height = robot.data.root_link_pos_w[:, 2]
  tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
  posture_ok = (root_height >= float(min_height)) & (tilt <= float(max_tilt))

  shot_dir, side_dir = _shot_frame_basis(
    ball_pos[:, :2],
    cmd.aim_pos_w[:, :2],
  )
  dx, dy = _project_point_to_shot_frame(
    left_pos[:, :2],
    ball_pos[:, :2],
    shot_dir,
    side_dir,
  )
  dx_ok = torch.abs(dx - float(target_dx)) <= float(dx_tol)
  dy_ok = torch.abs(torch.abs(dy) - float(target_abs_dy)) <= float(dy_tol)

  left_speed_xy = torch.linalg.norm(left_vel[:, :2], dim=1)
  speed_ok = left_speed_xy <= float(max_left_speed)

  return left_on_ground & left_not_touch_ball & posture_ok & dx_ok & dy_ok & speed_ok


def shot_support_gate(
  env,
  command_name: str = "set_shot",
) -> torch.Tensor:
  support_latch = _get_bool_state_buffer(
    env,
    key=f"p1_support_ok_on_valid_strike::{command_name}",
  )
  return support_latch.to(torch.float32)


def kick_phase_mask(
  env,
  command_name: str = "set_shot",
) -> torch.Tensor:
  kick_phase = _get_bool_state_buffer(env, key=f"p1_kick_phase::{command_name}")
  struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
  return (kick_phase & (~struck)).to(torch.float32)


def approach_phase_mask(
  env,
  command_name: str = "set_shot",
) -> torch.Tensor:
  kick_phase = _get_bool_state_buffer(env, key=f"p1_kick_phase::{command_name}")
  struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
  return ((~kick_phase) & (~struck)).to(torch.float32)


def latch_left_support_and_enter_kick_bonus(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  left_ground_sensor_name: str = "p1_left_foot_ground_contact",
  left_ball_sensor_name: str = "p1_left_foot_ball_contact",
  target_dx: float = -0.02,
  dx_tol: float = 0.16,
  target_abs_dy: float = 0.11,
  dy_tol: float = 0.12,
  max_left_speed: float = 0.50,
  min_height: float = 0.53,
  max_tilt: float = 0.60,
  acquire_steps: int = 2,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]

  good_count = _get_int_state_buffer(
    env,
    key=f"p1_kick_good_count::{command_name}",
    dtype=torch.int32,
  )
  kick_phase = _get_bool_state_buffer(env, key=f"p1_kick_phase::{command_name}")
  paid = _get_bool_state_buffer(env, key=f"p1_kick_phase_enter_paid::{command_name}")
  latched_xy = _get_float_state_buffer(
    env,
    key=f"p1_left_foot_latched_xy::{command_name}",
    shape_tail=(2,),
    dtype=torch.float32,
  )

  is_first = env.episode_length_buf <= 1
  good_count[is_first] = 0
  paid[is_first] = False

  ids, _ = robot.find_bodies((left_foot_body_name,), preserve_order=True)
  if len(ids) != 1 or not hasattr(robot.data, "body_link_pos_w"):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  left_idx = int(ids[0])
  left_xy = robot.data.body_link_pos_w[:, left_idx, :2]

  ready = left_support_ready_now_mask(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
    left_foot_body_name=left_foot_body_name,
    left_ground_sensor_name=left_ground_sensor_name,
    left_ball_sensor_name=left_ball_sensor_name,
    target_dx=target_dx,
    dx_tol=dx_tol,
    target_abs_dy=target_abs_dy,
    dy_tol=dy_tol,
    max_left_speed=max_left_speed,
    min_height=min_height,
    max_tilt=max_tilt,
  )

  struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
  pre_strike = ~struck

  can_count = (~kick_phase) & pre_strike
  good_count[can_count & ready] += 1
  good_count[can_count & (~ready)] = 0

  new_kick = (~kick_phase) & pre_strike & (good_count >= int(acquire_steps))
  kick_phase |= new_kick

  latched_xy[new_kick] = left_xy[new_kick]

  bonus = new_kick & (~paid)
  paid |= new_kick
  return bonus.to(torch.float32)


def approach_phase_approach_ball_progress_reward(
  env,
  command_name: str = "set_shot",
  **kwargs,
) -> torch.Tensor:
  return approach_phase_mask(env, command_name) * approach_ball_progress_reward(
    env,
    command_name=command_name,
    **kwargs,
  )


def approach_phase_base_vel_towards_ball_reward(
  env,
  command_name: str = "set_shot",
  **kwargs,
) -> torch.Tensor:
  return approach_phase_mask(env, command_name) * base_vel_towards_ball_reward(
    env,
    command_name=command_name,
    **kwargs,
  )


def approach_phase_left_support_stability_reward(
  env,
  command_name: str = "set_shot",
  **kwargs,
) -> torch.Tensor:
  return approach_phase_mask(env, command_name) * left_support_stability_reward_v2(
    env,
    command_name=command_name,
    **kwargs,
  )


def approach_phase_left_foot_beside_ball_reward(
  env,
  command_name: str = "set_shot",
  **kwargs,
) -> torch.Tensor:
  return approach_phase_mask(env, command_name) * left_foot_beside_ball_reward(
    env,
    command_name=command_name,
    **kwargs,
  )






def kick_phase_yaw_alignment_reward(
  env,
  command_name: str = "set_shot",
  **kwargs,
) -> torch.Tensor:
  return kick_phase_mask(env, command_name) * yaw_alignment_reward(
    env,
    command_name=command_name,
    **kwargs,
  )



  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  if not hasattr(robot.data, "body_link_pos_w"):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  ids, _ = robot.find_bodies((right_foot_body_name,), preserve_order=True)
  if len(ids) != 1:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  right_idx = int(ids[0])
  right_xy = robot.data.body_link_pos_w[:, right_idx, :2]
  ball_xy = ball.data.root_link_pos_w[:, :2]
  dist = torch.linalg.norm(right_xy - ball_xy, dim=1)
  dist_score = torch.exp(-torch.square(dist / max(float(sigma), 1.0e-6)))
  return kick * dist_score


def right_foot_swing_intent_score(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  right_foot_body_name: str = r"^right_foot_link$",
  max_speed: float = 6.0,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  if not hasattr(robot.data, "body_link_pos_w") or not hasattr(robot.data, "body_com_lin_vel_w"):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  ids, _ = robot.find_bodies((right_foot_body_name,), preserve_order=True)
  if len(ids) != 1:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  right_idx = int(ids[0])
  right_pos_xy = robot.data.body_link_pos_w[:, right_idx, :2]
  right_vel_xy = robot.data.body_com_lin_vel_w[:, right_idx, :2]
  ball_xy = ball.data.root_link_pos_w[:, :2]

  to_ball_xy = ball_xy - right_pos_xy
  to_ball_dir = to_ball_xy / torch.linalg.norm(to_ball_xy, dim=1, keepdim=True).clamp_min(1.0e-6)
  toward_ball_speed = torch.sum(right_vel_xy * to_ball_dir, dim=1).clamp_min(0.0)

  return torch.clamp(toward_ball_speed / float(max_speed), 0.0, 1.0).to(torch.float32)


def kick_phase_no_shot_pressure_penalty(
  env,
  command_name: str = "set_shot",
  grace_steps: int = 3,
  ramp_steps: int = 12,
  max_swing_speed: float = 6.0,
) -> torch.Tensor:
  kick = kick_phase_mask(env, command_name) > 0.5
  struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")

  count = _get_int_state_buffer(
    env,
    key=f"p1_kick_no_shot_count::{command_name}",
    dtype=torch.int32,
  )

  is_first = env.episode_length_buf <= 1
  count[is_first] = 0

  active = kick & (~struck)
  count[~active] = 0
  count[active] += 1

  excess = (count - int(grace_steps)).clamp(min=0)
  time_pressure = (excess.to(torch.float32) / float(max(int(ramp_steps), 1))).clamp(0.0, 1.0)
  swing_intent = right_foot_swing_intent_score(
    env,
    command_name=command_name,
    max_speed=max_swing_speed,
  )
  penalty = active.to(torch.float32) * time_pressure * (1.0 - swing_intent)
  return penalty.to(torch.float32)


def right_foot_swing_intent_debug_reward(
  env,
  command_name: str = "set_shot",
  max_swing_speed: float = 6.0,
) -> torch.Tensor:
  return right_foot_swing_intent_score(
    env,
    command_name=command_name,
    max_speed=max_swing_speed,
  )


def kick_phase_hold_window_mask(
  env,
  command_name: str = "set_shot",
  hold_steps: int = 10,
) -> torch.Tensor:
  kick_bool = kick_phase_mask(env, command_name) > 0.5
  hold_count = _get_int_state_buffer(
    env,
    key=f"p1_kick_phase_hold_count::{command_name}",
    dtype=torch.int32,
  )

  is_first = env.episode_length_buf <= 1
  hold_count[is_first] = 0

  hold_count[kick_bool] += 1
  hold_count[~kick_bool] = 0

  hold = kick_bool & (hold_count <= int(hold_steps))
  return hold.to(torch.float32)





def post_strike_support_lock_mask(
  env,
  command_name: str = "set_shot",
  lock_steps: int = 12,
) -> torch.Tensor:
  struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
  count = _get_int_state_buffer(
    env,
    key=f"p1_post_strike_support_lock_count::{command_name}",
    dtype=torch.int32,
  )

  is_first = env.episode_length_buf <= 1
  count[is_first] = 0

  active = struck & (~is_first)
  count[~active] = 0
  count[active] += 1

  lock = active & (count <= int(lock_steps))
  return lock.to(torch.float32)


def post_strike_left_support_move_penalty(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  deadzone: float = 0.015,
  max_dist: float = 0.12,
  lock_steps: int = 12,
) -> torch.Tensor:
  lock = post_strike_support_lock_mask(env, command_name=command_name, lock_steps=lock_steps)
  if not torch.any(lock > 0.0):
    return lock

  robot: Entity = env.scene[asset_cfg.name]
  ids, _ = robot.find_bodies((left_foot_body_name,), preserve_order=True)
  if len(ids) != 1 or not hasattr(robot.data, "body_link_pos_w"):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  latched_xy = _get_float_state_buffer(
    env,
    key=f"p1_left_foot_latched_xy::{command_name}",
    shape_tail=(2,),
    dtype=torch.float32,
  )

  left_idx = int(ids[0])
  current_xy = robot.data.body_link_pos_w[:, left_idx, :2]
  dist = torch.linalg.norm(current_xy - latched_xy, dim=1)
  denom = max(float(max_dist - deadzone), 1.0e-6)
  excess = ((dist - float(deadzone)) / denom).clamp(0.0, 1.0)
  return lock * excess


def kick_phase_left_support_move_penalty(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  deadzone: float = 0.008,
  max_dist: float = 0.06,
) -> torch.Tensor:
  kick = kick_phase_mask(env, command_name)
  if not torch.any(kick > 0.0):
    return kick

  robot: Entity = env.scene[asset_cfg.name]
  ids, _ = robot.find_bodies((left_foot_body_name,), preserve_order=True)
  if len(ids) != 1 or not hasattr(robot.data, "body_link_pos_w"):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  latched_xy = _get_float_state_buffer(
    env,
    key=f"p1_left_foot_latched_xy::{command_name}",
    shape_tail=(2,),
    dtype=torch.float32,
  )

  left_idx = int(ids[0])
  current_xy = robot.data.body_link_pos_w[:, left_idx, :2]
  dist = torch.linalg.norm(current_xy - latched_xy, dim=1)
  denom = max(float(max_dist - deadzone), 1.0e-6)
  excess = ((dist - float(deadzone)) / denom).clamp(0.0, 1.0)
  return kick * excess


def kick_phase_left_support_speed_penalty(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  max_speed: float = 0.20,
) -> torch.Tensor:
  kick = kick_phase_mask(env, command_name)
  if not torch.any(kick > 0.0):
    return kick

  robot: Entity = env.scene[asset_cfg.name]
  ids, _ = robot.find_bodies((left_foot_body_name,), preserve_order=True)
  if len(ids) != 1 or not hasattr(robot.data, "body_com_lin_vel_w"):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  left_idx = int(ids[0])
  speed_xy = torch.linalg.norm(robot.data.body_com_lin_vel_w[:, left_idx, :2], dim=1)
  return kick * torch.clamp(speed_xy / float(max_speed), 0.0, 1.0)


def kick_phase_left_support_backslide_penalty(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  deadzone: float = 0.003,
  max_backslide: float = 0.04,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  kick = kick_phase_mask(env, command_name).to(torch.float32)

  zeros = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
  ids, _ = robot.find_bodies((left_foot_body_name,), preserve_order=True)
  if len(ids) != 1 or not hasattr(robot.data, "body_link_pos_w"):
    return zeros

  left_idx = int(ids[0])
  current_left_xy = robot.data.body_link_pos_w[:, left_idx, :2].to(torch.float32)

  latched_xy = _get_float_state_buffer(
    env,
    key=f"p1_left_foot_latched_xy::{command_name}",
    shape_tail=(2,),
    dtype=torch.float32,
  )

  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  ball_xy = ball.data.root_link_pos_w[:, :2]
  aim_xy = cmd.aim_pos_w[:, :2]
  shot_dir, _ = _shot_frame_basis(ball_xy, aim_xy)

  delta_xy = current_left_xy - latched_xy
  signed_along_shot = torch.sum(delta_xy * shot_dir, dim=1)

  # Backward slide means moving opposite to shot_dir.
  backslide = (-signed_along_shot - float(deadzone)).clamp_min(0.0)
  backslide = backslide.clamp(max=float(max_backslide)) / max(float(max_backslide), 1.0e-6)

  return kick * backslide


def kick_phase_left_support_lost_ground_penalty(
  env,
  command_name: str = "set_shot",
  left_ground_sensor_name: str = "p1_left_foot_ground_contact",
) -> torch.Tensor:
  kick = kick_phase_mask(env, command_name)
  grounded = _support_bool_from_ground_sensor(
    env,
    left_ground_sensor_name,
    fz_thresh=10.0,
    support_sign="neg",
  )
  return kick * (~grounded).to(torch.float32)


def post_strike_left_support_speed_penalty(
  env,
  command_name: str = "set_shot",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  left_foot_body_name: str = r"^left_foot_link$",
  max_speed: float = 0.35,
  lock_steps: int = 12,
) -> torch.Tensor:
  lock = post_strike_support_lock_mask(env, command_name=command_name, lock_steps=lock_steps)
  if not torch.any(lock > 0.0):
    return lock

  robot: Entity = env.scene[asset_cfg.name]
  ids, _ = robot.find_bodies((left_foot_body_name,), preserve_order=True)
  if len(ids) != 1 or not hasattr(robot.data, "body_com_lin_vel_w"):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  left_idx = int(ids[0])
  speed_xy = torch.linalg.norm(robot.data.body_com_lin_vel_w[:, left_idx, :2], dim=1)
  return lock * torch.clamp(speed_xy / float(max_speed), 0.0, 1.0)


def post_strike_left_support_lost_ground_penalty(
  env,
  command_name: str = "set_shot",
  left_ground_sensor_name: str = "p1_left_foot_ground_contact",
  lock_steps: int = 12,
) -> torch.Tensor:
  lock = post_strike_support_lock_mask(env, command_name=command_name, lock_steps=lock_steps)
  grounded = _support_bool_from_ground_sensor(
    env,
    left_ground_sensor_name,
    fz_thresh=3.0,
    support_sign="neg",
  )
  return lock * (~grounded).to(torch.float32)


def kick_phase_timeout_penalty(
  env,
  command_name: str = "set_shot",
  grace_steps: int = 8,
  ramp_steps: int = 6,
) -> torch.Tensor:
  kick_bool = kick_phase_mask(env, command_name) > 0.5
  count = _get_int_state_buffer(env, key=f"p1_kick_phase_count::{command_name}", dtype=torch.int32)

  is_first = env.episode_length_buf <= 1
  count[is_first] = 0

  count[kick_bool] += 1
  count[~kick_bool] = 0

  excess = (count - int(grace_steps)).clamp(min=0)
  penalty = (excess.to(torch.float32) / float(max(int(ramp_steps), 1))).clamp(0.0, 1.0)
  return kick_bool.to(torch.float32) * penalty


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


def goal_scored_shaped_target_reward(
  env,
  command_name: str = "set_shot",
  sigma_y: float = 0.35,
  sigma_z: float = 0.18,
  base_goal: float = 0.10,
  weight_y: float = 0.20,
  weight_z: float = 0.15,
  weight_yz: float = 0.65,
) -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  origins = env.scene.env_origins
  ball_local = ball.data.root_link_pos_w - origins
  x = ball_local[:, 0]
  y = ball_local[:, 1]
  z = ball_local[:, 2]

  crossed = x >= float(cmd.cfg.goal_line_x)
  inside_y = torch.abs(y) <= float(cmd.cfg.goal_y_half)
  inside_z = (z >= float(cmd.cfg.goal_z_min)) & (z <= float(cmd.cfg.goal_z_max))
  scored_now = crossed & inside_y & inside_z

  prev = _get_bool_state_buffer(env, key=f"p1_goal_scored_shaped_prev::{command_name}")
  is_first = env.episode_length_buf <= 1
  prev[is_first] = False
  event = scored_now & (~prev)
  prev.copy_(scored_now)

  aim_local = cmd.aim_pos_w - origins
  target_y = aim_local[:, 1]
  target_z = aim_local[:, 2]

  sigma_y_safe = max(float(sigma_y), 1.0e-6)
  sigma_z_safe = max(float(sigma_z), 1.0e-6)
  y_score = torch.exp(-0.5 * torch.square((y - target_y) / sigma_y_safe))
  z_score = torch.exp(-0.5 * torch.square((z - target_z) / sigma_z_safe))
  yz_score = y_score * z_score

  w_sum = max(float(weight_y) + float(weight_z) + float(weight_yz), 1.0e-6)
  shaped = float(base_goal) + (1.0 - float(base_goal)) * (
    float(weight_y) * y_score +
    float(weight_z) * z_score +
    float(weight_yz) * yz_score
  ) / w_sum

  return event.to(torch.float32) * shaped


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
  goal_lat = _get_bool_state_buffer(env, key=f"p1_goal_lat::{command_name}")
  is_first = env.episode_length_buf <= 1
  goal_lat[is_first] = False

  scored_now = goal_scored_termination(env, command_name)
  goal_lat |= scored_now

  return goal_lat.to(torch.float32) * upright_stability_reward(env)

def ball_upward_velocity_after_strike_reward(
  env,
  command_name: str = "set_shot",
  max_vz: float = 2.5,
  min_vx: float = 0.10,
) -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  vel = ball.data.root_link_lin_vel_w

  vx_gate = torch.sigmoid((vel[:, 0] - float(min_vx)) / 0.20)
  vz_pos = vel[:, 2].clamp(min=0.0, max=float(max_vz)) / float(max_vz)

  return post_strike_phase_mask(env, command_name) * vx_gate * vz_pos


def ball_power_lift_reward_after_strike(
    env,
    command_name: str = "set_shot",
    max_speed_3d: float = 9.0,
    min_vx: float = 0.20,
    min_vz: float = 0.08,
) -> torch.Tensor:
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    vel = ball.data.root_link_lin_vel_w
    speed3d = torch.linalg.norm(vel, dim=1).clamp(max=float(max_speed_3d)) / float(max_speed_3d)

    vx_gate = torch.sigmoid((vel[:, 0] - float(min_vx)) / 0.20)
    vz_gate = torch.sigmoid((vel[:, 2] - float(min_vz)) / 0.12)

    speed_bonus = torch.pow(speed3d, 2)

    return post_strike_phase_mask(env, command_name) * speed_bonus * vx_gate * vz_gate


def ball_ground_touch_before_goal_penalty(
  env,
  command_name: str = "set_shot",
  ground_z: float = 0.115,
  min_x_progress: float = 0.30,
) -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ball_pos = ball.data.root_link_pos_w
  origins = env.scene.env_origins
  ball_local = ball_pos - origins

  x = ball_local[:, 0]
  z = ball_local[:, 2]

  before_goal = x < float(cmd.cfg.goal_line_x)
  after_leave_spot = x > float(min_x_progress)
  grounded = z <= float(ground_z)

  return has_struck(env, command_name) * before_goal.to(torch.float32) * after_leave_spot.to(torch.float32) * grounded.to(torch.float32)

def foot_speed_before_strike_reward(
    env,
    command_name: str = "set_shot",
    max_speed: float = 7.0,
):

    # gate: solo prima dello strike
    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)

    robot = env.scene["robot"]

    left_vel = robot.data.body_com_lin_vel_w[:, robot.find_bodies("^left_foot_link$")[0][0], :]
    right_vel = robot.data.body_com_lin_vel_w[:, robot.find_bodies("^right_foot_link$")[0][0], :]

    speed = torch.maximum(
        torch.norm(left_vel, dim=-1),
        torch.norm(right_vel, dim=-1),
    )

    return gate_pre * torch.clamp(speed / max_speed, 0.0, 1.0)

def ball_bounce_before_goal_penalty(
    env,
    command_name: str = "set_shot",
    ground_z: float = 0.12,
    min_x_after_strike: float = 0.35,
    require_forward_vx: float = 0.35,
) -> torch.Tensor:
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    ball_pos_w = ball.data.root_link_pos_w
    ball_vel_w = ball.data.root_link_lin_vel_w
    origins = env.scene.env_origins
    ball_local = ball_pos_w - origins

    x = ball_local[:, 0]
    z = ball_local[:, 2]
    vx = ball_vel_w[:, 0]

    before_goal = x < float(cmd.cfg.goal_line_x)
    left_spot = x > float(min_x_after_strike)
    moving_forward = vx > float(require_forward_vx)
    touched_ground = z <= float(ground_z)

    bounce_now = (
        post_strike_phase_mask(env, command_name).bool()
        & before_goal
        & left_spot
        & moving_forward
        & touched_ground
    )

    fired = _get_bool_state_buffer(env, key=f"p1_bounce_fired::{command_name}")
    is_first = env.episode_length_buf <= 1
    fired[is_first] = False

    event = bounce_now & (~fired)
    fired |= bounce_now

    return event.to(torch.float32)

def foot_to_ball_velocity_alignment(
    env,
    command_name: str = "set_shot",
    max_speed: float = 7.0,
):

    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)

    robot = env.scene["robot"]
    ball = env.scene["soccer_ball"]

    ball_pos = ball.data.root_link_pos_w

    ids,_ = robot.find_bodies(("^left_foot_link$","^right_foot_link$"), preserve_order=True)

    left_pos = robot.data.body_link_pos_w[:, ids[0]]
    right_pos = robot.data.body_link_pos_w[:, ids[1]]

    left_vel = robot.data.body_com_lin_vel_w[:, ids[0], :]
    right_vel = robot.data.body_com_lin_vel_w[:, ids[1], :]

    vec_l = ball_pos - left_pos
    vec_r = ball_pos - right_pos

    vec_l = vec_l / (torch.norm(vec_l, dim=-1, keepdim=True) + 1e-6)
    vec_r = vec_r / (torch.norm(vec_r, dim=-1, keepdim=True) + 1e-6)

    vel_l = left_vel / (torch.norm(left_vel, dim=-1, keepdim=True) + 1e-6)
    vel_r = right_vel / (torch.norm(right_vel, dim=-1, keepdim=True) + 1e-6)

    align_l = torch.sum(vec_l * vel_l, dim=-1)
    align_r = torch.sum(vec_r * vel_r, dim=-1)

    align = torch.maximum(align_l, align_r)

    speed = torch.maximum(
        torch.norm(left_vel, dim=-1),
        torch.norm(right_vel, dim=-1),
    )

    speed_scale = torch.clamp(speed / max_speed, 0.0, 1.0)

    return gate_pre * torch.clamp(align,0,1) * speed_scale


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


def impact_foot_speed_reward(
    env,
    command_name: str = "set_shot",
    left_sensor_name: str = "p1_left_foot_ball_contact",
    right_sensor_name: str = "p1_right_foot_ball_contact",
    max_speed: float = 8.0,
):
    robot = env.scene["robot"]

    ids, _ = robot.find_bodies(("^left_foot_link$", "^right_foot_link$"), preserve_order=True)

    left_vel = robot.data.body_com_lin_vel_w[:, ids[0]]
    right_vel = robot.data.body_com_lin_vel_w[:, ids[1]]

    left_speed = torch.norm(left_vel, dim=-1)
    right_speed = torch.norm(right_vel, dim=-1)

    touch_l = _sensor_any_found(env, left_sensor_name)
    touch_r = _sensor_any_found(env, right_sensor_name)

    prev_l = _get_bool_state_buffer(env, key=f"p1_impact_prev_l::{command_name}")
    prev_r = _get_bool_state_buffer(env, key=f"p1_impact_prev_r::{command_name}")
    paid   = _get_bool_state_buffer(env, key=f"p1_impact_paid::{command_name}")

    strike_l = _get_bool_state_buffer(env, key=f"p1_strike_l::{command_name}")
    strike_r = _get_bool_state_buffer(env, key=f"p1_strike_r::{command_name}")

    is_first = env.episode_length_buf <= 1
    prev_l[is_first] = False
    prev_r[is_first] = False
    paid[is_first] = False
    strike_l[is_first] = False
    strike_r[is_first] = False

    new_l = touch_l & (~prev_l)
    new_r = touch_r & (~prev_r)

    prev_l.copy_(touch_l)
    prev_r.copy_(touch_r)

    event_l = new_l & (~paid)
    event_r = new_r & (~paid)

    paid |= (event_l | event_r)
    strike_l |= event_l
    strike_r |= event_r

    reward_speed = torch.where(
        event_r,
        right_speed,
        torch.where(event_l, left_speed, torch.zeros_like(left_speed)),
    )

    return torch.clamp(reward_speed / max_speed, 0.0, 1.0)


def non_strike_foot_speed_poststrike_penalty(
    env,
    command_name: str = "set_shot",
    max_speed: float = 6.0,
):
    robot = env.scene["robot"]

    ids, _ = robot.find_bodies(("^left_foot_link$", "^right_foot_link$"), preserve_order=True)

    left_vel = robot.data.body_com_lin_vel_w[:, ids[0]]
    right_vel = robot.data.body_com_lin_vel_w[:, ids[1]]

    left_speed = torch.norm(left_vel, dim=-1)
    right_speed = torch.norm(right_vel, dim=-1)

    strike_l = _get_bool_state_buffer(env, key=f"p1_strike_l::{command_name}")
    strike_r = _get_bool_state_buffer(env, key=f"p1_strike_r::{command_name}")

    other_speed = torch.where(
        strike_r,
        left_speed,
        torch.where(strike_l, right_speed, torch.zeros_like(left_speed)),
    )

    return post_strike_phase_mask(env, command_name) * torch.clamp(other_speed / max_speed, 0.0, 1.0)


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


def ball_launch_angle_reward_after_strike(
    env,
    command_name: str = "set_shot",
    target_angle_deg: float = 20.0,
    angle_sigma_deg: float = 12.0,
    min_vx: float = 0.10,
    max_speed_3d: float = 9.0,
) -> torch.Tensor:
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    vel = ball.data.root_link_lin_vel_w
    vx = vel[:, 0].clamp(min=0.0)
    vz = vel[:, 2].clamp(min=0.0)

    speed3d = torch.linalg.norm(vel, dim=1).clamp(max=float(max_speed_3d)) / float(max_speed_3d)

    angle = torch.atan2(vz, vx.clamp(min=1.0e-6))
    target = torch.deg2rad(torch.tensor(float(target_angle_deg), device=vel.device))
    sigma = torch.deg2rad(torch.tensor(float(angle_sigma_deg), device=vel.device))

    angle_reward = torch.exp(-0.5 * torch.square((angle - target) / sigma))
    forward_gate = torch.sigmoid((vx - float(min_vx)) / 0.15)

    return post_strike_phase_mask(env, command_name) * forward_gate * speed3d * angle_reward

def action_rate_l2_poststrike(env, command_name: str = "set_shot") -> torch.Tensor:
    a = env.action_manager.action
    pa = env.action_manager.prev_action
    raw = torch.mean((a - pa) ** 2, dim=1)
    return post_strike_phase_mask(env, command_name) * raw

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


############### <Funzioni di utilità> ###############

def approach_ball_reward_simple(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    k: float = 2.0,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]
    d_xy = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    dist = torch.linalg.norm(d_xy, dim=1)
    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    return gate_pre * torch.exp(-k * dist)


def behind_ball_reward_simple(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    dx_max: float = 0.40,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    robot_xy = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
    ball_xy = _world_to_env_local_xy(env, ball.data.root_link_pos_w[:, :2])

    dx = ball_xy[:, 0] - robot_xy[:, 0]
    behind = dx > 0.0
    close = dx < float(dx_max)

    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    return gate_pre * (behind & close).to(torch.float32)



def impact_foot_speed_once_reward(
    env,
    command_name: str = "set_shot",
    left_sensor_name: str = "p1_left_foot_ball_contact",
    right_sensor_name: str = "p1_right_foot_ball_contact",
    max_speed: float = 8.0,
) -> torch.Tensor:
    robot = env.scene["robot"]

    ids, _ = robot.find_bodies(("^left_foot_link$", "^right_foot_link$"), preserve_order=True)

    left_vel = robot.data.body_com_lin_vel_w[:, ids[0]]
    right_vel = robot.data.body_com_lin_vel_w[:, ids[1]]

    left_speed = torch.norm(left_vel, dim=-1)
    right_speed = torch.norm(right_vel, dim=-1)

    touch_l = _sensor_any_found(env, left_sensor_name)
    touch_r = _sensor_any_found(env, right_sensor_name)

    prev_l = _get_bool_state_buffer(env, key=f"p1_imp_prev_l::{command_name}")
    prev_r = _get_bool_state_buffer(env, key=f"p1_imp_prev_r::{command_name}")
    paid = _get_bool_state_buffer(env, key=f"p1_imp_paid::{command_name}")

    is_first = env.episode_length_buf <= 1
    prev_l[is_first] = False
    prev_r[is_first] = False
    paid[is_first] = False

    new_l = touch_l & (~prev_l)
    new_r = touch_r & (~prev_r)

    prev_l.copy_(touch_l)
    prev_r.copy_(touch_r)

    event_l = new_l & (~paid)
    event_r = new_r & (~paid)
    paid |= (event_l | event_r)

    reward_speed = torch.where(
        event_r,
        right_speed,
        torch.where(event_l, left_speed, torch.zeros_like(left_speed)),
    )

    return torch.clamp(reward_speed / max_speed, 0.0, 1.0)


def ball_launch_angle_underbar_reward(
    env,
    command_name: str = "set_shot",
    target_angle_deg: float = 24.0,
    angle_sigma_deg: float = 8.0,
    min_vx: float = 0.10,
    max_speed_3d: float = 9.0,
) -> torch.Tensor:
    return ball_launch_angle_reward_after_strike(
        env,
        command_name=command_name,
        target_angle_deg=target_angle_deg,
        angle_sigma_deg=angle_sigma_deg,
        min_vx=min_vx,
        max_speed_3d=max_speed_3d,
    )


def underbar_goal_reward(
    env,
    command_name: str = "set_shot",
    sigma_z: float = 0.18,
) -> torch.Tensor:
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    ball_pos = ball.data.root_link_pos_w
    origins = env.scene.env_origins
    ball_local = ball_pos - origins

    x = ball_local[:, 0]
    y = ball_local[:, 1]
    z = ball_local[:, 2]

    crossed = x >= float(cmd.cfg.goal_line_x)
    inside_y = torch.abs(y) <= float(cmd.cfg.goal_y_half)
    inside_z = (z >= float(cmd.cfg.goal_z_min)) & (z <= float(cmd.cfg.goal_z_max))
    scored_now = crossed & inside_y & inside_z

    prev = _get_bool_state_buffer(env, key=f"p1_underbar_goal_prev::{command_name}")
    is_first = env.episode_length_buf <= 1
    prev[is_first] = False

    event = scored_now & (~prev)
    prev.copy_(scored_now)

    aim_local = cmd.aim_pos_w - origins
    target_z = aim_local[:, 2]

    sigma_z_safe = max(float(sigma_z), 1.0e-6)
    z_score = torch.exp(-0.5 * torch.square((z - target_z) / sigma_z_safe))
    return event.to(torch.float32) * z_score


def lateral_goal_reward(
    env,
    command_name: str = "set_shot",
    sigma_y: float = 0.18,
) -> torch.Tensor:
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    ball_pos = ball.data.root_link_pos_w
    origins = env.scene.env_origins
    ball_local = ball_pos - origins

    x = ball_local[:, 0]
    y = ball_local[:, 1]
    z = ball_local[:, 2]

    crossed = x >= float(cmd.cfg.goal_line_x)
    inside_y = torch.abs(y) <= float(cmd.cfg.goal_y_half)
    inside_z = (z >= float(cmd.cfg.goal_z_min)) & (z <= float(cmd.cfg.goal_z_max))
    scored_now = crossed & inside_y & inside_z

    prev = _get_bool_state_buffer(env, key=f"p1_lateral_goal_prev::{command_name}")
    is_first = env.episode_length_buf <= 1
    prev[is_first] = False

    event = scored_now & (~prev)
    prev.copy_(scored_now)

    aim_local = cmd.aim_pos_w - origins
    target_y = aim_local[:, 1]
    sigma_y_safe = max(float(sigma_y), 1.0e-6)
    y_score = torch.exp(-0.5 * torch.square((y - target_y) / sigma_y_safe))
    return event.to(torch.float32) * y_score


def post_strike_upright_reward_strong(
    env,
    command_name: str = "set_shot",
) -> torch.Tensor:
    return post_strike_phase_mask(env, command_name) * upright_stability_reward(env)


def action_rate_l2_poststrike_penalty(
    env,
    command_name: str = "set_shot",
) -> torch.Tensor:
    a = env.action_manager.action
    pa = env.action_manager.prev_action
    raw = torch.mean((a - pa) ** 2, dim=1)
    return post_strike_phase_mask(env, command_name) * raw

def hold_contact_after_strike_penalty(
    env,
    command_name: str = "set_shot",
    left_sensor_name: str = "p1_left_foot_ball_contact",
    right_sensor_name: str = "p1_right_foot_ball_contact",
) -> torch.Tensor:
    touching = _sensor_any_found(env, left_sensor_name) | _sensor_any_found(env, right_sensor_name)

    prev_touch = _get_bool_state_buffer(env, key=f"p1_hold_prev_touch::{command_name}")
    is_first = env.episode_length_buf <= 1
    prev_touch[is_first] = False

    new_touch = touching & (~prev_touch)
    prev_touch.copy_(touching)

    hold = post_strike_phase_mask(env, command_name).bool() & touching & (~new_touch)
    return hold.to(torch.float32)

def striker_standing_gate(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    h_low: float = 0.40,
    s_min: float = 0.45,
    roll_band: float = 0.07,
    roll_sigma: float = 0.12,
    pitch_target: float = 0.14,
    pitch_band: float = 0.12,
    pitch_sigma: float = 0.25,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    base_height = robot.data.root_link_pos_w[:, 2]
    h_good = _default_root_height(robot)

    posture = striker_posture_score(
        env,
        asset_cfg=asset_cfg,
        roll_band=roll_band,
        roll_sigma=roll_sigma,
        pitch_target=pitch_target,
        pitch_band=pitch_band,
        pitch_sigma=pitch_sigma,
    )

    denom = (h_good - float(h_low)).clamp_min(1.0e-6)
    height_score = ((base_height - float(h_low)) / denom).clamp(0.0, 1.0)

    stand_score = posture * height_score
    gate_pre = ((stand_score - float(s_min)) / max(1.0 - float(s_min), 1.0e-6)).clamp(0.0, 1.0)
    return gate_pre * gate_pre


def pre_strike_standing_reward(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    return gate_pre * striker_standing_gate(env, asset_cfg=asset_cfg)


def striker_low_height_soft_penalty(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    h_soft: float | None = None,
    scale: float = 0.06,
):
    robot: Entity = env.scene[asset_cfg.name]
    height = robot.data.root_link_pos_w[:, 2]

    if h_soft is None:
        h_soft_t = (_default_root_height(robot) - 0.06).clamp_min(0.50)
    else:
        h_soft_t = torch.full_like(height, float(h_soft))

    deficit = ((h_soft_t - height).clamp_min(0.0) / float(scale)).clamp(0.0, 2.5)
    return deficit * deficit

def left_foot_prestrike_touch_penalty(
    env,
    command_name: str = "set_shot",
    left_sensor_name: str = "p1_left_foot_ball_contact",
) -> torch.Tensor:
    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    touching_l = _sensor_any_found(env, left_sensor_name).to(torch.float32)
    return gate_pre * touching_l


def right_foot_prestrike_touch_bonus(
    env,
    command_name: str = "set_shot",
    right_sensor_name: str = "p1_right_foot_ball_contact",
) -> torch.Tensor:
    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)

    touching_r = _sensor_any_found(env, right_sensor_name)
    prev_r = _get_bool_state_buffer(env, key=f"p1_right_pre_touch_prev::{command_name}")

    is_first = env.episode_length_buf <= 1
    prev_r[is_first] = False

    new_touch_r = touching_r & (~prev_r)
    prev_r.copy_(touching_r)

    return gate_pre * new_touch_r.to(torch.float32)

def posture_priority_gate_latched(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    min_height: float | None = None,
    max_tilt: float = 0.60,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    height = robot.data.root_link_pos_w[:, 2]
    tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)

    if min_height is None:
        # più severo dell'attuale low-height soft penalty
        min_h = (_default_root_height(robot) - 0.07).clamp_min(0.52)
    else:
        min_h = torch.full_like(height, float(min_height))

    bad_now = (height < min_h) | (tilt > float(max_tilt))

    latched = _get_bool_state_buffer(env, key="p1_bad_posture_latched")
    is_first = env.episode_length_buf <= 1
    latched[is_first] = False
    latched |= bad_now

    return (~latched).to(torch.float32)

def left_support_plant_reward(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    left_ground_sensor_name: str = "p1_left_foot_ground_contact",
    left_ball_sensor_name: str = "p1_left_foot_ball_contact",
    target_dx: float = 0.0,
    dx_sigma: float = 0.05,
    target_abs_dy: float = 0.10,
    dy_sigma: float = 0.05,
    max_left_speed: float = 0.35,
) -> torch.Tensor:
    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    posture_gate = posture_priority_gate_latched(
        env,
        min_height=0.53,
        max_tilt=0.60,
    )

    robot: Entity = env.scene[asset_cfg.name]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    left_idx = int(robot.find_bodies("^left_foot_link$")[0][0])

    left_pos = robot.data.body_link_pos_w[:, left_idx, :]
    left_vel = robot.data.body_com_lin_vel_w[:, left_idx, :]
    ball_pos = ball.data.root_link_pos_w

    shot_dir, side_dir = _shot_frame_basis(
        ball_pos[:, :2],
        cmd.aim_pos_w[:, :2],
    )
    dx, dy = _project_point_to_shot_frame(
        left_pos[:, :2],
        ball_pos[:, :2],
        shot_dir,
        side_dir,
    )

    dx_score = torch.exp(-torch.square((dx - float(target_dx)) / float(dx_sigma)))
    dy_score = torch.exp(-torch.square((torch.abs(dy) - float(target_abs_dy)) / float(dy_sigma)))

    left_speed_xy = torch.linalg.norm(left_vel[:, :2], dim=1)
    stable_score = torch.exp(-torch.square(left_speed_xy / float(max_left_speed)))

    left_support = _support_bool_from_ground_sensor(
        env,
        left_ground_sensor_name,
        fz_thresh=5.0,
        support_sign="neg",
    ).to(torch.float32)

    left_ball_touch = _sensor_any_found(env, left_ball_sensor_name).to(torch.float32)

    return gate_pre * posture_gate * left_support * (1.0 - left_ball_touch) * dx_score * dy_score * stable_score

def bad_posture_at_strike_penalty(
    env,
    command_name: str = "set_shot",
    left_sensor_name: str = "p1_left_foot_ball_contact",
    right_sensor_name: str = "p1_right_foot_ball_contact",
    min_height: float = 0.53,
    max_tilt: float = 0.60,
):
    touching = _sensor_any_found(env, left_sensor_name) | _sensor_any_found(env, right_sensor_name)

    prev_touch = _get_bool_state_buffer(env, key=f"p1_bad_posture_prev_touch::{command_name}")
    paid = _get_bool_state_buffer(env, key=f"p1_bad_posture_paid::{command_name}")

    is_first = env.episode_length_buf <= 1
    prev_touch[is_first] = False
    paid[is_first] = False

    new_touch = touching & (~prev_touch)
    prev_touch.copy_(touching)

    posture_ok = posture_priority_gate_latched(
        env,
        min_height=min_height,
        max_tilt=max_tilt,
    ) > 0.5

    bad_strike = new_touch & (~posture_ok) & (~paid)
    paid |= bad_strike

    return bad_strike.to(torch.float32)

# ============================================================
# SHOOTER T1 — NEW FUNCTIONS FOR NATURAL RIGHT-FOOTED SHOT
# ============================================================

import re as _re


def _find_joint_idx(robot, patterns: tuple) -> "int | None":
    """Find index of first joint matching any regex pattern (case-insensitive)."""
    joint_names = list(robot.joint_names)
    for pat in patterns:
        for idx, name in enumerate(joint_names):
            if _re.search(pat, name, _re.IGNORECASE):
                return idx
    return None


def right_knee_straight_at_strike_reward(
    env,
    command_name: str = "set_shot",
    right_knee_patterns: tuple = (
        r"right_knee",
        r"right.*knee",
        r"r_kne",
        r"rknee",
    ),
    sigma_rad: float = 0.25,
) -> torch.Tensor:
    """
    ONE-SHOT reward at the exact frame when p1_struck first becomes True.

    A proper football shot extends the kicking leg at contact — the knee
    should be nearly STRAIGHT (joint angle close to its default standing
    value). Returns a Gaussian score in [0,1] based on how extended the
    right knee is at impact. Silent no-op if the joint can't be found.
    """
    robot: Entity = env.scene["robot"]

    knee_idx = _find_joint_idx(robot, right_knee_patterns)
    if knee_idx is None:
        return torch.zeros(env.num_envs, device=env.device)

    knee_angle = robot.data.joint_pos[:, knee_idx]
    default_knee = robot.data.default_joint_pos[:, knee_idx]
    knee_flex = (knee_angle - default_knee).abs()

    # Gaussian: max when knee is fully extended (flex == 0)
    straight_score = torch.exp(-0.5 * torch.square(knee_flex / float(sigma_rad)))

    # Fire exactly once, at the transition to struck==True
    struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
    paid = _get_bool_state_buffer(env, key=f"p1_knee_straight_paid::{command_name}")
    prev_struck = _get_bool_state_buffer(env, key=f"p1_knee_prev_struck::{command_name}")

    is_first = env.episode_length_buf <= 1
    paid[is_first] = False
    prev_struck[is_first] = False

    just_struck = struck & (~prev_struck)
    prev_struck.copy_(struck)

    fire_now = just_struck & (~paid)
    paid |= fire_now

    posture_gate = posture_priority_gate_latched(env, min_height=0.53, max_tilt=0.60)
    return posture_gate * fire_now.to(torch.float32) * straight_score


def right_foot_only_strike_bonus(
    env,
    command_name: str = "set_shot",
    right_sensor_name: str = "p1_right_foot_ball_contact",
    left_sensor_name: str = "p1_left_foot_ball_contact",
) -> torch.Tensor:
    """
    ONE-SHOT bonus when the first ball contact is with the RIGHT foot only
    (left foot must NOT be touching ball simultaneously).
    Encourages the canonical right-footed shot.
    """
    struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
    paid = _get_bool_state_buffer(env, key=f"p1_rfoot_only_paid::{command_name}")

    is_first = env.episode_length_buf <= 1
    paid[is_first] = False

    touch_r = _sensor_any_found(env, right_sensor_name)
    touch_l = _sensor_any_found(env, left_sensor_name)

    prev_r = _get_bool_state_buffer(env, key=f"p1_rfoot_only_prev_r::{command_name}")
    prev_r[is_first] = False
    new_r = touch_r & (~prev_r)
    prev_r.copy_(touch_r)

    right_only = new_r & (~touch_l) & (~paid)
    paid |= right_only

    posture_gate = posture_priority_gate_latched(env, min_height=0.53, max_tilt=0.60)
    return posture_gate * right_only.to(torch.float32)


# mdp.py
def supported_right_only_strike_bonus(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    right_sensor_name: str = "p1_right_foot_ball_contact",
    left_sensor_name: str = "p1_left_foot_ball_contact",
    left_ground_sensor_name: str = "p1_left_foot_ground_contact",
    left_ball_sensor_name: str = "p1_left_foot_ball_contact",
    target_dx: float = -0.02,
    dx_tol: float = 0.16,
    target_abs_dy: float = 0.11,
    dy_tol: float = 0.12,
    max_left_speed: float = 0.50,
    min_height: float = 0.53,
    max_tilt: float = 0.60,
    min_speed: float = 2.0,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    ids, _ = robot.find_bodies(("^right_foot_link$",), preserve_order=True)
    if len(ids) != 1:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ridx = int(ids[0])
    right_vel = robot.data.body_com_lin_vel_w[:, ridx, :]
    right_speed = torch.linalg.norm(right_vel, dim=1)

    touch_r = _sensor_any_found(env, right_sensor_name)
    touch_l = _sensor_any_found(env, left_sensor_name)

    prev_r = _get_bool_state_buffer(env, key=f"p1_supported_r_only_prev_r::{command_name}")
    paid = _get_bool_state_buffer(env, key=f"p1_supported_r_only_paid::{command_name}")

    is_first = env.episode_length_buf <= 1
    prev_r[is_first] = False
    paid[is_first] = False

    new_r = touch_r & (~prev_r)
    prev_r.copy_(touch_r)

    kick_ok = kick_phase_mask(env, command_name) > 0.5

    speed_ok = right_speed > float(min_speed)
    valid_event = new_r & (~touch_l) & kick_ok & speed_ok & (~paid)
    paid |= valid_event
    return valid_event.to(torch.float32)


def left_foot_beside_ball_reward(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    left_foot_body_name: str = r"^left_foot_link$",
    left_ground_sensor_name: str = "p1_left_foot_ground_contact",
    left_ball_sensor_name: str = "p1_left_foot_ball_contact",
    target_dx: float = -0.02,
    dx_sigma: float = 0.18,
    target_abs_dy: float = 0.12,
    dy_sigma: float = 0.12,
    max_left_speed: float = 0.60,
) -> torch.Tensor:
    """
    Dense reward for planting the left (support) foot BESIDE the ball:
    - ~0 cm behind the ball in X (slightly behind is fine)
    - ~12 cm to the side in Y
    - Planted on ground, NOT touching the ball, moving slowly.

    This is the key positioning step before a right-footed shot.
    Uses relaxed spatial tolerances vs left_support_plant_reward so the
    robot can discover the correct position more easily during exploration.
    """
    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    posture_gate = posture_priority_gate_latched(env, min_height=0.53, max_tilt=0.60)

    robot: Entity = env.scene[asset_cfg.name]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    ids, _ = robot.find_bodies(left_foot_body_name, preserve_order=True)
    if len(ids) == 0 or not hasattr(robot.data, "body_link_pos_w"):
        return torch.zeros(env.num_envs, device=env.device)

    left_idx = int(ids[0])
    left_pos = robot.data.body_link_pos_w[:, left_idx, :]
    left_vel = robot.data.body_com_lin_vel_w[:, left_idx, :]
    ball_pos = ball.data.root_link_pos_w

    shot_dir, side_dir = _shot_frame_basis(
        ball_pos[:, :2],
        cmd.aim_pos_w[:, :2],
    )
    dx, dy = _project_point_to_shot_frame(
        left_pos[:, :2],
        ball_pos[:, :2],
        shot_dir,
        side_dir,
    )

    dx_score = torch.exp(-torch.square((dx - float(target_dx)) / float(dx_sigma)))
    dy_score = torch.exp(
        -torch.square((torch.abs(dy) - float(target_abs_dy)) / float(dy_sigma))
    )
    stable_score = torch.exp(
        -torch.square(torch.linalg.norm(left_vel[:, :2], dim=1) / float(max_left_speed))
    )

    left_support = _support_bool_from_ground_sensor(
        env, left_ground_sensor_name, fz_thresh=3.0, support_sign="neg"
    ).to(torch.float32)
    left_ball_touch = _sensor_any_found(env, left_ball_sensor_name).to(torch.float32)

    return (
        gate_pre
        * posture_gate
        * left_support
        * (1.0 - left_ball_touch)
        * dx_score
        * dy_score
        * stable_score
    )


def post_strike_standing_reward(
    env,
    command_name: str = "set_shot",
) -> torch.Tensor:
    """
    After the shot, reward the robot for standing upright and STILL —
    the classic footballer post-shot pose. Tighter stability requirements
    than the general upright reward so the robot truly freezes.
    """
    return post_strike_phase_mask(env, command_name) * upright_stability_reward(
        env,
        height_sigma=0.08,
        roll_band=0.04,
        roll_sigma=0.08,
        pitch_target=0.10,
        pitch_band=0.07,
        pitch_sigma=0.13,
    )

def alive_reward(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)


def _get_prev_goal_dist_buffer(env, command_name: str) -> torch.Tensor:
    env_obj = getattr(env, "unwrapped", env)
    cache_name = "_p1_prev_goal_dist_cache"
    cache = getattr(env_obj, cache_name, None)
    if cache is None:
        cache = {}
        setattr(env_obj, cache_name, cache)

    prev = cache.get(command_name)
    if prev is None or prev.shape != (env.num_envs,) or prev.device != env.device:
        prev = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        cache[command_name] = prev
    return prev


def goal_progress_after_strike_reward(
    env,
    command_name: str = "set_shot",
    max_delta: float = 0.25,
) -> torch.Tensor:
    """
    Paper-like task reward:
    reward the reduction of ball-to-goal distance after the kick has happened.
    """
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    goal_dist = torch.linalg.norm(cmd.aim_pos_w[:, :2] - ball.data.root_link_pos_w[:, :2], dim=1)

    prev = _get_prev_goal_dist_buffer(env, command_name)
    is_first = env.episode_length_buf <= 1
    prev[is_first] = goal_dist[is_first]

    prog = (prev - goal_dist).clamp(min=0.0, max=float(max_delta))
    prev.copy_(goal_dist)

    return has_struck(env, command_name) * prog


def right_foot_toe_poke_reward_once(
    env,
    command_name: str = "set_shot",
    right_sensor_name: str = "p1_right_foot_ball_contact",
    max_forward_speed: float = 5.0,
    max_vertical_speed: float = 1.2,
    xy_sigma: float = 0.12,
    z_margin: float = 0.01,
) -> torch.Tensor:
    """
    Reward a short, fast, forward toe-poke with the RIGHT foot at first new contact.
    - high forward foot velocity toward the target
    - low vertical foot velocity
    - no 'foot above ball' geometry at contact
    """
    robot: Entity = env.scene["robot"]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    ids, _ = robot.find_bodies(("^right_foot_link$",), preserve_order=True)
    if len(ids) != 1:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ridx = int(ids[0])

    foot_pos = robot.data.body_link_pos_w[:, ridx, :]
    foot_vel = robot.data.body_com_lin_vel_w[:, ridx, :]
    ball_pos = ball.data.root_link_pos_w

    touching = _sensor_any_found(env, right_sensor_name)
    prev_touch = _get_bool_state_buffer(env, key=f"p1_toepoke_prev_touch::{command_name}")
    paid = _get_bool_state_buffer(env, key=f"p1_toepoke_paid::{command_name}")

    is_first = env.episode_length_buf <= 1
    prev_touch[is_first] = False
    paid[is_first] = False

    new_touch = touching & (~prev_touch)
    prev_touch.copy_(touching)

    event = new_touch & (~paid)
    paid |= event

    target_dir_xy = _normalize_xy(cmd.aim_pos_w[:, :2] - ball_pos[:, :2])
    v_forward = torch.sum(foot_vel[:, :2] * target_dir_xy, dim=1).clamp_min(0.0)

    v_vertical = torch.abs(foot_vel[:, 2])
    vert_gate = torch.exp(-torch.square(v_vertical / float(max_vertical_speed)))

    d_xy = torch.linalg.norm(foot_pos[:, :2] - ball_pos[:, :2], dim=1)
    near_gate = torch.exp(-torch.square(d_xy / float(xy_sigma)))

    over_ball = foot_pos[:, 2] > (ball_pos[:, 2] + float(z_margin))
    not_over_gate = (~over_ball).to(torch.float32)

    return (
        event.to(torch.float32)
        * torch.clamp(v_forward / float(max_forward_speed), 0.0, 1.0)
        * vert_gate
        * near_gate
        * not_over_gate
    )


def stuck_ball_contact_penalty(
    env,
    command_name: str = "set_shot",
    left_sensor_name: str = "p1_left_foot_ball_contact",
    right_sensor_name: str = "p1_right_foot_ball_contact",
    stagnation_speed: float = 0.20,
) -> torch.Tensor:
    """
    Penalize the bad case where a foot touches the ball but the ball is not really moving.
    This directly fights the 'foot on top of the ball and stand still' exploit.
    """
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    touching = _sensor_any_found(env, left_sensor_name) | _sensor_any_found(env, right_sensor_name)
    ball_speed_xy = torch.linalg.norm(ball.data.root_link_lin_vel_w[:, :2], dim=1)

    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    slow_factor = 1.0 - torch.clamp(ball_speed_xy / float(stagnation_speed), 0.0, 1.0)

    return gate_pre * touching.to(torch.float32) * slow_factor


def left_support_stability_reward(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    left_ground_sensor_name: str = "p1_left_foot_ground_contact",
    left_ball_sensor_name: str = "p1_left_foot_ball_contact",
    near_ball_dist: float = 0.35,
    max_left_speed: float = 0.40,
) -> torch.Tensor:
    """
    Mild, non-sparse support-foot reward:
    - only before strike
    - left foot on ground
    - left foot not touching ball
    - left foot near the ball
    - left foot not moving much
    """
    robot: Entity = env.scene[asset_cfg.name]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    left_idx = int(robot.find_bodies("^left_foot_link$")[0][0])

    left_pos = robot.data.body_link_pos_w[:, left_idx, :]
    left_vel = robot.data.body_com_lin_vel_w[:, left_idx, :]
    ball_pos = ball.data.root_link_pos_w

    left_ground = _support_bool_from_ground_sensor(
        env,
        left_ground_sensor_name,
        fz_thresh=5.0,
        support_sign="neg",
    ).to(torch.float32)

    left_ball_touch = _sensor_any_found(env, left_ball_sensor_name).to(torch.float32)

    d_xy = torch.linalg.norm(left_pos[:, :2] - ball_pos[:, :2], dim=1)
    near_gate = (d_xy < float(near_ball_dist)).to(torch.float32)

    left_speed_xy = torch.linalg.norm(left_vel[:, :2], dim=1)
    stable_gate = torch.exp(-torch.square(left_speed_xy / float(max_left_speed)))

    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)

    return gate_pre * left_ground * (1.0 - left_ball_touch) * near_gate * stable_gate

def right_foot_strike_event_contact_only(
    env,
    command_name: str = "set_shot",
    right_sensor_name: str = "p1_right_foot_ball_contact",
) -> torch.Tensor:
    touching = _sensor_any_found(env, right_sensor_name)

    prev_touch = _get_bool_state_buffer(env, key=f"p1_prev_touch_right_only::{command_name}")
    struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")

    is_first = env.episode_length_buf <= 1
    prev_touch[is_first] = False
    struck[is_first] = False

    new_touch = touching & (~prev_touch)
    prev_touch.copy_(touching)

    reward = new_touch & (~struck)
    struck |= new_touch

    return reward.to(torch.float32)


def right_foot_impact_speed_once_reward(
    env,
    command_name: str = "set_shot",
    right_sensor_name: str = "p1_right_foot_ball_contact",
    max_speed: float = 8.0,
) -> torch.Tensor:
    robot = env.scene["robot"]

    ids, _ = robot.find_bodies(("^right_foot_link$",), preserve_order=True)
    right_vel = robot.data.body_com_lin_vel_w[:, ids[0]]
    right_speed = torch.norm(right_vel, dim=-1)

    touch_r = _sensor_any_found(env, right_sensor_name)

    prev_r = _get_bool_state_buffer(env, key=f"p1_imp_prev_r_only::{command_name}")
    paid = _get_bool_state_buffer(env, key=f"p1_imp_paid_r_only::{command_name}")

    is_first = env.episode_length_buf <= 1
    prev_r[is_first] = False
    paid[is_first] = False

    new_r = touch_r & (~prev_r)
    prev_r.copy_(touch_r)

    event_r = new_r & (~paid)
    paid |= event_r

    return event_r.to(torch.float32) * torch.clamp(right_speed / max_speed, 0.0, 1.0)


def right_foot_impact_speed_target_reward(
    env,
    command_name: str = "set_shot",
    right_sensor_name: str = "p1_right_foot_ball_contact",
    target_speed: float = 4.7,
    sigma: float = 0.6,
) -> torch.Tensor:
    robot = env.scene["robot"]

    ids, _ = robot.find_bodies(("^right_foot_link$",), preserve_order=True)
    right_vel = robot.data.body_com_lin_vel_w[:, ids[0]]
    right_speed = torch.norm(right_vel, dim=-1)

    touch_r = _sensor_any_found(env, right_sensor_name)

    prev_r = _get_bool_state_buffer(env, key=f"p1_imp_prev_r_target::{command_name}")
    paid = _get_bool_state_buffer(env, key=f"p1_imp_paid_r_target::{command_name}")

    is_first = env.episode_length_buf <= 1
    prev_r[is_first] = False
    paid[is_first] = False

    new_r = touch_r & (~prev_r)
    prev_r.copy_(touch_r)

    event_r = new_r & (~paid)
    paid |= event_r

    sigma_safe = max(float(sigma), 1.0e-6)
    speed_score = torch.exp(-0.5 * torch.square((right_speed - float(target_speed)) / sigma_safe))
    return event_r.to(torch.float32) * speed_score


def ball_speed_to_goal_after_strike_reward(
    env,
    command_name: str = "set_shot",
    max_speed: float = 6.0,
) -> torch.Tensor:
    return has_struck(env, command_name) * torch.clamp(
        ball_speed_to_goal_reward(env, command_name) / float(max_speed),
        0.0,
        1.0,
    )

def right_foot_ready_reward(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_dx: float = -0.06,
    target_abs_dy: float = 0.10,
    dx_sigma: float = 0.08,
    dy_sigma: float = 0.08,
) -> torch.Tensor:
    """
    Dense pre-strike shaping:
    reward the RIGHT foot for being in a good kicking pose:
    - slightly behind the ball in x
    - slightly lateral to the ball in y
    """
    robot: Entity = env.scene[asset_cfg.name]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    ids, _ = robot.find_bodies(("^right_foot_link$",), preserve_order=True)
    if len(ids) != 1:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ridx = int(ids[0])

    right_pos = robot.data.body_link_pos_w[:, ridx, :]
    ball_pos = ball.data.root_link_pos_w

    shot_dir, side_dir = _shot_frame_basis(
        ball_pos[:, :2],
        cmd.aim_pos_w[:, :2],
    )
    dx, dy = _project_point_to_shot_frame(
        right_pos[:, :2],
        ball_pos[:, :2],
        shot_dir,
        side_dir,
    )

    dx_score = torch.exp(-torch.square((dx - float(target_dx)) / float(dx_sigma)))
    dy_score = torch.exp(-torch.square((torch.abs(dy) - float(target_abs_dy)) / float(dy_sigma)))

    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    return gate_pre * dx_score * dy_score


def right_foot_speed_before_strike_reward(
    env,
    command_name: str = "set_shot",
    max_speed: float = 6.0,
) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    ids, _ = robot.find_bodies(("^right_foot_link$",), preserve_order=True)
    if len(ids) != 1:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ridx = int(ids[0])
    v = robot.data.body_com_lin_vel_w[:, ridx, :2]
    speed = torch.linalg.norm(v, dim=1)

    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    return gate_pre * torch.clamp(speed / float(max_speed), 0.0, 1.0)


def right_foot_to_ball_velocity_alignment_reward(
    env,
    command_name: str = "set_shot",
    max_speed: float = 6.0,
) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    ids, _ = robot.find_bodies(("^right_foot_link$",), preserve_order=True)
    if len(ids) != 1:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ridx = int(ids[0])

    foot_pos = robot.data.body_link_pos_w[:, ridx, :]
    foot_vel = robot.data.body_com_lin_vel_w[:, ridx, :2]
    to_ball = ball.data.root_link_pos_w[:, :2] - foot_pos[:, :2]
    to_ball_dir = _normalize_xy(to_ball)

    speed = torch.linalg.norm(foot_vel, dim=1).clamp_min(1.0e-6)
    vel_dir = foot_vel / speed.unsqueeze(1)

    align = torch.sum(vel_dir * to_ball_dir, dim=1).clamp_min(0.0)
    speed_gate = torch.clamp(speed / float(max_speed), 0.0, 1.0)

    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)
    return gate_pre * align * speed_gate

def right_foot_strike_scaled_reward(
    env,
    command_name: str = "set_shot",
    right_sensor_name: str = "p1_right_foot_ball_contact",
    max_speed: float = 8.0,
    min_speed: float = 2.0,
):
    robot = env.scene["robot"]
    ids, _ = robot.find_bodies(("^right_foot_link$",), preserve_order=True)
    ridx = int(ids[0])

    foot_vel = robot.data.body_com_lin_vel_w[:, ridx]
    foot_speed = torch.norm(foot_vel, dim=-1)

    touching = _sensor_any_found(env, right_sensor_name)

    prev = _get_bool_state_buffer(env, key=f"p1_strike_prev_r::{command_name}")
    paid = _get_bool_state_buffer(env, key=f"p1_strike_paid_r::{command_name}")
    struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")

    is_first = env.episode_length_buf <= 1
    prev[is_first] = False
    paid[is_first] = False
    struck[is_first] = False

    new_touch = touching & (~prev)
    prev.copy_(touching)

    event = new_touch & (~paid)
    paid |= event

    good_event = event & (foot_speed > float(min_speed))
    struck |= good_event

    return good_event.to(torch.float32) * torch.clamp(foot_speed / max_speed, 0.0, 1.0)


def supported_right_foot_strike_scaled_reward(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    right_sensor_name: str = "p1_right_foot_ball_contact",
    left_ground_sensor_name: str = "p1_left_foot_ground_contact",
    left_ball_sensor_name: str = "p1_left_foot_ball_contact",
    target_dx: float = -0.02,
    dx_tol: float = 0.16,
    target_abs_dy: float = 0.11,
    dy_tol: float = 0.12,
    max_left_speed: float = 0.50,
    min_height: float = 0.53,
    max_tilt: float = 0.60,
    max_speed: float = 8.0,
    min_speed: float = 2.2,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    ids, _ = robot.find_bodies(("^right_foot_link$",), preserve_order=True)
    if len(ids) != 1 or not hasattr(robot.data, "body_com_lin_vel_w"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ridx = int(ids[0])
    right_vel = robot.data.body_com_lin_vel_w[:, ridx, :]
    right_speed = torch.linalg.norm(right_vel, dim=1)

    touching_right = _sensor_any_found(env, right_sensor_name)

    prev_r = _get_bool_state_buffer(env, key=f"p1_supported_strike_prev_r::{command_name}")
    struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
    support_latch = _get_bool_state_buffer(
        env,
        key=f"p1_support_ok_on_valid_strike::{command_name}",
    )

    is_first = env.episode_length_buf <= 1
    prev_r[is_first] = False
    struck[is_first] = False
    support_latch[is_first] = False

    new_r = touching_right & (~prev_r)
    prev_r.copy_(touching_right)

    speed_ok = right_speed > float(min_speed)
    kick_ok = kick_phase_mask(env, command_name) > 0.5

    posture_gate = posture_priority_gate_latched(
        env,
        asset_cfg=asset_cfg,
        min_height=min_height,
        max_tilt=max_tilt,
    )

    posture_ok = posture_gate > 0.5

    valid_strike = new_r & (~struck) & speed_ok & kick_ok & posture_ok

    struck |= valid_strike
    support_latch |= valid_strike

    speed_score = (
        (right_speed - float(min_speed))
        / max(float(max_speed - min_speed), 1.0e-6)
    ).clamp(0.0, 1.0)

    return posture_gate * valid_strike.to(torch.float32) * speed_score




def supported_right_foot_impact_speed_once_reward(
    env,
    command_name: str = "set_shot",
    **kwargs,
) -> torch.Tensor:
    return shot_support_gate(env, command_name) * right_foot_impact_speed_once_reward(
        env,
        command_name=command_name,
        **kwargs,
    )


def supported_clean_strike_reward(
    env,
    command_name: str = "set_shot",
    **kwargs,
) -> torch.Tensor:
    return shot_support_gate(env, command_name) * clean_strike_reward(
        env,
        command_name=command_name,
        **kwargs,
    )


def supported_ball_speed_to_goal_after_strike_reward(
    env,
    command_name: str = "set_shot",
    **kwargs,
) -> torch.Tensor:
    return shot_support_gate(env, command_name) * ball_speed_to_goal_after_strike_reward(
        env,
        command_name=command_name,
        **kwargs,
    )


def supported_goal_scored_event_reward(
    env,
    command_name: str = "set_shot",
    **kwargs,
) -> torch.Tensor:
    return shot_support_gate(env, command_name) * goal_scored_event_reward(
        env,
        command_name=command_name,
        **kwargs,
    )


def support_ready_but_no_strike_penalty(
    env,
    command_name: str = "set_shot",
    ready_steps: int = 2,
    grace_steps: int = 8,
    ramp_steps: int = 6,
    left_ground_sensor_name: str = "p1_left_foot_ground_contact",
    left_ball_sensor_name: str = "p1_left_foot_ball_contact",
    target_dx: float = -0.02,
    dx_tol: float = 0.22,
    target_abs_dy: float = 0.11,
    dy_tol: float = 0.16,
    max_left_speed: float = 0.65,
    min_height: float = 0.50,
    max_tilt: float = 0.75,
) -> torch.Tensor:
    support_ready = left_support_ready_now_mask(
        env,
        command_name=command_name,
        left_ground_sensor_name=left_ground_sensor_name,
        left_ball_sensor_name=left_ball_sensor_name,
        target_dx=target_dx,
        dx_tol=dx_tol,
        target_abs_dy=target_abs_dy,
        dy_tol=dy_tol,
        max_left_speed=max_left_speed,
        min_height=min_height,
        max_tilt=max_tilt,
    )
    struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")

    ready_count = _get_int_state_buffer(
        env,
        key=f"p1_support_ready_count::{command_name}",
        dtype=torch.int32,
    )
    stall_count = _get_int_state_buffer(
        env,
        key=f"p1_support_stall_count::{command_name}",
        dtype=torch.int32,
    )

    is_first = env.episode_length_buf <= 1
    struck[is_first] = False
    ready_count[is_first] = 0
    stall_count[is_first] = 0

    keep_ready = support_ready & (~struck)
    ready_count[keep_ready] += 1
    ready_count[~keep_ready] = 0

    ready_mask = ready_count >= int(ready_steps)

    stall_active = ready_mask & (~struck)
    stall_count[stall_active] += 1
    stall_count[~stall_active] = 0

    excess = (stall_count - int(grace_steps)).clamp(min=0)
    penalty = (excess.to(torch.float32) / float(max(int(ramp_steps), 1))).clamp(0.0, 1.0)
    return (ready_mask & (~struck)).to(torch.float32) * penalty


def weak_right_touch_penalty(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    right_sensor_name: str = "p1_right_foot_ball_contact",
    left_ground_sensor_name: str = "p1_left_foot_ground_contact",
    left_ball_sensor_name: str = "p1_left_foot_ball_contact",
    target_dx: float = -0.02,
    dx_tol: float = 0.16,
    target_abs_dy: float = 0.11,
    dy_tol: float = 0.12,
    max_left_speed: float = 0.50,
    min_height: float = 0.53,
    max_tilt: float = 0.60,
    min_speed: float = 2.2,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]

    touching_right = _sensor_any_found(env, right_sensor_name)
    prev_r = _get_bool_state_buffer(env, key=f"p1_weak_touch_prev_r::{command_name}")
    is_first = env.episode_length_buf <= 1
    prev_r[is_first] = False

    new_r = touching_right & (~prev_r)
    prev_r.copy_(touching_right)

    ids, _ = robot.find_bodies(("^right_foot_link$",), preserve_order=True)
    if len(ids) != 1 or not hasattr(robot.data, "body_com_lin_vel_w"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ridx = int(ids[0])
    right_vel = robot.data.body_com_lin_vel_w[:, ridx, :]
    right_speed = torch.linalg.norm(right_vel, dim=1)

    struck = _get_bool_state_buffer(env, key=f"p1_struck::{command_name}")
    support_ok = left_support_valid_now_mask(
        env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        left_ground_sensor_name=left_ground_sensor_name,
        left_ball_sensor_name=left_ball_sensor_name,
        target_dx=target_dx,
        dx_tol=dx_tol,
        target_abs_dy=target_abs_dy,
        dy_tol=dy_tol,
        max_left_speed=max_left_speed,
        min_height=min_height,
        max_tilt=max_tilt,
    )

    bad_touch = new_r & (~struck) & ((right_speed < float(min_speed)) | (~support_ok))
    return bad_touch.to(torch.float32)


def left_support_ready_debug_reward(
    env,
    command_name: str = "set_shot",
    **kwargs,
) -> torch.Tensor:
    return left_support_ready_now_mask(
        env,
        command_name=command_name,
        **kwargs,
    ).to(torch.float32)


def left_support_stability_reward(
    env,
    command_name: str = "set_shot",
    left_ground_sensor_name: str = "p1_left_foot_ground_contact",
    left_ball_sensor_name: str = "p1_left_foot_ball_contact",
):
    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)

    left_ground = _support_bool_from_ground_sensor(
        env,
        left_ground_sensor_name,
        fz_thresh=5.0,
        support_sign="neg",
    ).to(torch.float32)

    left_touch_ball = _sensor_any_found(env, left_ball_sensor_name).to(torch.float32)

    return gate_pre * left_ground * (1.0 - left_touch_ball)

def left_support_stability_reward_v2(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    left_ground_sensor_name: str = "p1_left_foot_ground_contact",
    left_ball_sensor_name: str = "p1_left_foot_ball_contact",
    near_ball_dist: float = 0.32,
    max_left_speed: float = 0.35,
) -> torch.Tensor:
    """
    Dense support-foot reward per il sinistro:
    - solo pre-strike
    - sinistro a terra
    - sinistro NON tocca la palla
    - sinistro vicino alla palla
    - sinistro abbastanza fermo
    """
    robot: Entity = env.scene[asset_cfg.name]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    ids, _ = robot.find_bodies(("^left_foot_link$",), preserve_order=True)
    if len(ids) != 1:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    left_idx = int(ids[0])

    left_pos = robot.data.body_link_pos_w[:, left_idx, :]
    left_vel = robot.data.body_com_lin_vel_w[:, left_idx, :]
    ball_pos = ball.data.root_link_pos_w

    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)

    left_ground = _support_bool_from_ground_sensor(
        env,
        left_ground_sensor_name,
        fz_thresh=5.0,
        support_sign="neg",
    ).to(torch.float32)

    left_touch_ball = _sensor_any_found(env, left_ball_sensor_name).to(torch.float32)

    d_xy = torch.linalg.norm(left_pos[:, :2] - ball_pos[:, :2], dim=1)
    near_gate = torch.exp(-torch.square(d_xy / float(near_ball_dist)))

    left_speed_xy = torch.linalg.norm(left_vel[:, :2], dim=1)
    stable_gate = torch.exp(-torch.square(left_speed_xy / float(max_left_speed)))

    return gate_pre * left_ground * (1.0 - left_touch_ball) * near_gate * stable_gate


def double_knee_crouch_penalty(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    left_knee_patterns: tuple = (
        r"left_knee",
        r"left.*knee",
        r"l_kne",
        r"lknee",
    ),
    right_knee_patterns: tuple = (
        r"right_knee",
        r"right.*knee",
        r"r_kne",
        r"rknee",
    ),
    near_ball_dist: float = 0.70,
    free_left_flex: float = 0.18,
    free_right_flex: float = 0.28,
    max_left_flex: float = 0.85,
    max_right_flex: float = 1.05,
    left_weight: float = 1.25,
    right_weight: float = 0.75,
) -> torch.Tensor:
    """
    Penalità densa pre-strike contro il crouch eccessivo su entrambe le ginocchia.
    Più severa sul sinistro di appoggio, più permissiva sul destro che deve comunque caricare.
    Attiva soprattutto quando il robot è vicino alla palla.
    """
    robot: Entity = env.scene[asset_cfg.name]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    left_idx = _find_joint_idx(robot, left_knee_patterns)
    right_idx = _find_joint_idx(robot, right_knee_patterns)

    if left_idx is None and right_idx is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    gate_pre = (1.0 - has_struck(env, command_name)).to(torch.float32)

    trunk_pos = robot.data.root_link_pos_w
    ball_pos = ball.data.root_link_pos_w
    d_xy = torch.linalg.norm(trunk_pos[:, :2] - ball_pos[:, :2], dim=1)
    near_gate = torch.exp(-torch.square(d_xy / float(near_ball_dist)))

    total = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    total_w = 0.0

    if left_idx is not None:
        left_angle = robot.data.joint_pos[:, left_idx]
        left_default = robot.data.default_joint_pos[:, left_idx]
        left_flex = (left_angle - left_default).abs()

        left_excess = (
            (left_flex - float(free_left_flex))
            / max(float(max_left_flex - free_left_flex), 1.0e-6)
        ).clamp(0.0, 1.0)

        total = total + float(left_weight) * torch.square(left_excess)
        total_w += float(left_weight)

    if right_idx is not None:
        right_angle = robot.data.joint_pos[:, right_idx]
        right_default = robot.data.default_joint_pos[:, right_idx]
        right_flex = (right_angle - right_default).abs()

        right_excess = (
            (right_flex - float(free_right_flex))
            / max(float(max_right_flex - free_right_flex), 1.0e-6)
        ).clamp(0.0, 1.0)

        total = total + float(right_weight) * torch.square(right_excess)
        total_w += float(right_weight)

    penalty = total / max(total_w, 1.0e-6)
    return gate_pre * near_gate * penalty

def support_plant_at_strike_bonus(
    env,
    command_name: str = "set_shot",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    left_foot_body_name: str = r"^left_foot_link$",
    left_ground_sensor_name: str = "p1_left_foot_ground_contact",
    left_ball_sensor_name: str = "p1_left_foot_ball_contact",
    right_sensor_name: str = "p1_right_foot_ball_contact",
    target_dx: float = -0.02,
    dx_sigma: float = 0.16,
    target_abs_dy: float = 0.11,
    dy_sigma: float = 0.12,
    max_left_speed: float = 0.50,
    min_right_speed: float = 2.2,
    max_right_speed: float = 8.0,
) -> torch.Tensor:
    """
    ONE-SHOT bonus:
    premia il plant corretto del sinistro ESATTAMENTE al momento di un vero strike del destro.

    - stessa geometria di left_foot_beside_ball_reward
    - paga solo su nuovo contatto del destro
    - richiede velocità minima del destro
    - il bonus cresce sia con la qualità del plant sia con la qualità dello strike
    """
    robot: Entity = env.scene[asset_cfg.name]
    cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    left_ids, _ = robot.find_bodies(left_foot_body_name, preserve_order=True)
    right_ids, _ = robot.find_bodies(("^right_foot_link$",), preserve_order=True)

    if len(left_ids) != 1 or len(right_ids) != 1 or not hasattr(robot.data, "body_link_pos_w"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    left_idx = int(left_ids[0])
    right_idx = int(right_ids[0])

    # ---------------- left plant quality ----------------
    left_pos = robot.data.body_link_pos_w[:, left_idx, :]
    left_vel = robot.data.body_com_lin_vel_w[:, left_idx, :]
    ball_pos = ball.data.root_link_pos_w

    shot_dir, side_dir = _shot_frame_basis(
        ball_pos[:, :2],
        cmd.aim_pos_w[:, :2],
    )
    dx, dy = _project_point_to_shot_frame(
        left_pos[:, :2],
        ball_pos[:, :2],
        shot_dir,
        side_dir,
    )

    dx_score = torch.exp(-torch.square((dx - float(target_dx)) / float(dx_sigma)))
    dy_score = torch.exp(
        -torch.square((torch.abs(dy) - float(target_abs_dy)) / float(dy_sigma))
    )
    stable_score = torch.exp(
        -torch.square(torch.linalg.norm(left_vel[:, :2], dim=1) / float(max_left_speed))
    )

    left_support = _support_bool_from_ground_sensor(
        env,
        left_ground_sensor_name,
        fz_thresh=3.0,
        support_sign="neg",
    ).to(torch.float32)

    left_ball_touch = _sensor_any_found(env, left_ball_sensor_name).to(torch.float32)

    posture_gate = posture_priority_gate_latched(env, min_height=0.53, max_tilt=0.60)

    plant_score = (
        posture_gate
        * left_support
        * (1.0 - left_ball_touch)
        * dx_score
        * dy_score
        * stable_score
    )

    # ---------------- right strike event ----------------
    touch_r = _sensor_any_found(env, right_sensor_name)

    prev_r = _get_bool_state_buffer(env, key=f"p1_support_strike_prev_r::{command_name}")
    paid = _get_bool_state_buffer(env, key=f"p1_support_strike_paid::{command_name}")

    is_first = env.episode_length_buf <= 1
    prev_r[is_first] = False
    paid[is_first] = False

    new_r = touch_r & (~prev_r)
    prev_r.copy_(touch_r)

    right_vel = robot.data.body_com_lin_vel_w[:, right_idx]
    right_speed = torch.norm(right_vel, dim=-1)

    good_event = new_r & (~paid) & (right_speed > float(min_right_speed))
    paid |= good_event

    strike_speed_score = (
        (right_speed - float(min_right_speed))
        / max(float(max_right_speed - min_right_speed), 1.0e-6)
    ).clamp(0.0, 1.0)

    return (
    good_event.to(torch.float32)
    * torch.pow(plant_score.clamp_min(1.0e-6), 0.75)
    * (0.55 + 0.45 * strike_speed_score)
)
