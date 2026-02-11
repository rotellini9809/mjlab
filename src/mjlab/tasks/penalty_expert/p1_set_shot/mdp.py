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
  aim_z: float = 0.11

  # area bounds in env-local coordinates
  striker_area_bounds: tuple[float, float, float, float] = (-1.0, 7.0, -2.0, 2.0)
  hard_area_margin: float = 0.5

  # goal check
  goal_line_x: float = 7.0
  goal_y_half: float = 1.0

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

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

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
    signs = torch.where((env_ids % 2) == 0, 1.0, -1.0).to(origins.dtype)
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

  def build(self, env):
        return SetShotCommand(self, env)


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


# ---------------- Rewards ----------------

def yaw_alignment_reward(env, command_name: str = "set_shot", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, k: float = 2.5) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  yaw_error = _compute_yaw_error(robot, cmd.aim_pos_w)
  return torch.exp(-k * torch.square(yaw_error))


def upright_stability_reward(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  height_target: float = 0.62,
  height_sigma: float = 0.12,
  tilt_sigma: float = 0.5,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  height = robot.data.root_link_pos_w[:, 2]
  height_err_sq = torch.square((height - float(height_target)) / float(height_sigma))
  tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
  tilt_err_sq = torch.square(tilt / float(tilt_sigma))
  return torch.exp(-0.5 * (height_err_sq + tilt_err_sq))


def approach_ball_reward(env, command_name: str = "set_shot", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, k: float = 2.0) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  d_xy = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  dist = torch.linalg.norm(d_xy, dim=1)
  return torch.exp(-k * dist)


def behind_ball_reward(env, command_name: str = "set_shot", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  # dir goal in env-local is +x; behind ball means robot_x < ball_x
  robot_x = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])[:, 0]
  ball_x  = _world_to_env_local_xy(env, ball.data.root_link_pos_w[:, :2])[:, 0]
  # reward 1 if behind, else 0 (hard, but works)
  return (robot_x < ball_x).to(robot_x.dtype)


def strike_event_reward(env, command_name: str = "set_shot", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  # proxy evento strike: palla accelera mentre sei vicino
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  rel = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  dist = torch.linalg.norm(rel, dim=1)
  speed = torch.linalg.norm(ball.data.root_link_lin_vel_w[:, :2], dim=1)

  near = dist < 0.9
  fast = speed > 0.6
  return (near & fast).to(dist.dtype)


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

  ball_local = _world_to_env_local_xy(env, ball.data.root_link_pos_w[:, :2])
  x = ball_local[:, 0]
  y = ball_local[:, 1]
  crossed = x >= float(cmd.cfg.goal_line_x)
  inside = torch.abs(y) <= float(cmd.cfg.goal_y_half)
  return crossed & inside


def goal_scored_reward(env, command_name: str = "set_shot") -> torch.Tensor:
  return goal_scored_termination(env, command_name=command_name).to(torch.float32)


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

  ok_goal = goal_scored_termination(env, command_name=command_name)

  ball_local = _world_to_env_local_xy(env, ball.data.root_link_pos_w[:, :2])
  y = ball_local[:, 1]
  z = ball.data.root_link_pos_w[:, 2]

  good = (z >= z_min) & (torch.abs(y) >= y_side_min)
  return (ok_goal & good).to(torch.float32)


def goal_low_or_center_penalty(
  env,
  command_name: str = "set_shot",
  z_min: float = 0.55,
  y_side_min: float = 0.55,
) -> torch.Tensor:
  cmd = cast(SetShotCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  ok_goal = goal_scored_termination(env, command_name=command_name)

  ball_local = _world_to_env_local_xy(env, ball.data.root_link_pos_w[:, :2])
  y = ball_local[:, 1]
  z = ball.data.root_link_pos_w[:, 2]

  bad = (z < z_min) | (torch.abs(y) < y_side_min)
  return (ok_goal & bad).to(torch.float32)

def trunk_tilt_l2_penalty(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  tilt_xy = robot.data.projected_gravity_b[:, :2]
  return torch.sum(torch.square(tilt_xy), dim=1)



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
