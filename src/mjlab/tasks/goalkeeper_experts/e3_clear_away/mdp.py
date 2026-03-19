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
  if high < low:
    low, high = high, low
  return torch.rand(num, device=device) * (high - low) + low


def _sample_categorical(
  probs: tuple[float, ...],
  num: int,
  device: str,
) -> torch.Tensor:
  p = torch.tensor([max(0.0, x) for x in probs], device=device)
  if float(p.sum().item()) <= 1.0e-8:
    p = torch.ones_like(p) / float(len(probs))
  else:
    p = p / p.sum()
  cdf = torch.cumsum(p, dim=0)
  out = torch.searchsorted(cdf, torch.rand(num, device=device), right=False)
  return torch.clamp(out, min=0, max=len(probs) - 1).to(torch.long)


def _normalize_xy(vec_xy: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
  norm = torch.linalg.norm(vec_xy, dim=1, keepdim=True).clamp_min(eps)
  return vec_xy / norm


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


def _world_to_env_local_xyz(env, pos_w_xyz: torch.Tensor) -> torch.Tensor:
  return pos_w_xyz - env.scene.env_origins


def _goal_conceded_mask_from_local(
  ball_local_xyz: torch.Tensor,
  goal_toward_positive_x: bool,
  goal_plane_x: float,
  goal_plane_y_center: float,
  goal_plane_y_half: float,
  goal_plane_z_min: float,
  goal_plane_z_max: float,
) -> torch.Tensor:
  x = ball_local_xyz[:, 0]
  y = ball_local_xyz[:, 1]
  z = ball_local_xyz[:, 2]

  if goal_toward_positive_x:
    crossed = x >= float(goal_plane_x)
  else:
    crossed = x <= float(goal_plane_x)

  inside_y = torch.abs(y - float(goal_plane_y_center)) <= float(goal_plane_y_half)
  inside_z = (z >= float(goal_plane_z_min)) & (z <= float(goal_plane_z_max))
  return crossed & inside_y & inside_z


def _ball_in_danger_zone_mask_from_local(
  ball_local_xyz: torch.Tensor,
  ball_vel_x_w: torch.Tensor,
  goal_toward_positive_x: bool,
  goal_plane_x: float,
  goal_plane_y_center: float,
  danger_zone_depth: float,
  danger_zone_half_width: float,
  require_toward_goal: bool,
  toward_goal_speed_threshold: float,
) -> torch.Tensor:
  x = ball_local_xyz[:, 0]
  y = ball_local_xyz[:, 1]

  if goal_toward_positive_x:
    in_x = x >= float(goal_plane_x - danger_zone_depth)
    toward_goal = ball_vel_x_w > float(toward_goal_speed_threshold)
  else:
    in_x = x <= float(goal_plane_x + danger_zone_depth)
    toward_goal = ball_vel_x_w < -float(toward_goal_speed_threshold)

  in_y = torch.abs(y - float(goal_plane_y_center)) <= float(danger_zone_half_width)
  if require_toward_goal:
    return in_x & in_y & toward_goal
  return in_x & in_y


def _ball_goal_distance_xy(
  ball_local_xyz: torch.Tensor,
  goal_plane_x: float,
  goal_plane_y_center: float,
) -> torch.Tensor:
  dx = ball_local_xyz[:, 0] - float(goal_plane_x)
  dy = ball_local_xyz[:, 1] - float(goal_plane_y_center)
  return torch.sqrt(dx * dx + dy * dy)


def _away_speed_from_vx(
  vel_x_w: torch.Tensor,
  goal_toward_positive_x: bool,
) -> torch.Tensor:
  if goal_toward_positive_x:
    return -vel_x_w
  return vel_x_w


def get_target_ball_cfg() -> EntityCfg:
  """Return the physical colliding RoboCup ball for E3."""
  return get_robocup_ball_cfg()


@dataclass(kw_only=True)
class ClearAwayCommandCfg(CommandTermCfg):
  """Command term for E3 clear-away reset variants and fixed motor command."""

  entity_name: str = "robot"
  ball_entity_name: str = "soccer_ball"

  # Motor-controller command vector dimension (from Stage-1 obs layout).
  command_dim: int = 46

  # Keeper spawn (world XY, before adding env origin).
  keeper_spawn_x_range: tuple[float, float]
  keeper_spawn_y_range: tuple[float, float]
  spawn_yaw_range: tuple[float, float]

  # Keeper pose noise (post-contact resets use stronger perturbation).
  keeper_joint_pos_noise: float = 0.02
  keeper_joint_vel_noise: float = 0.08
  post_contact_keeper_joint_pos_noise: float = 0.05
  post_contact_keeper_joint_vel_noise: float = 0.18

  # Keeper area bounds: (x_min, x_max, y_min, y_max).
  keeper_area_bounds: tuple[float, float, float, float]
  hard_area_margin: float = 0.35

  # Goal aperture used by goal detection.
  goal_toward_positive_x: bool = True
  goal_plane_x: float = 7.0
  goal_plane_y_center: float = 0.0
  goal_plane_y_half: float = 1.35
  goal_plane_z_min: float = 0.0
  goal_plane_z_max: float = 1.90

  # Danger zone definition.
  danger_zone_depth: float = 1.4
  danger_zone_half_width: float = 1.55
  danger_zone_require_toward_goal: bool = False
  danger_zone_toward_goal_speed_threshold: float = 0.05

  # Reset variant mix.
  loose_variant_prob: float = 0.60

  # Variant A: loose ball in danger zone.
  loose_ball_speed_range: tuple[float, float] = (0.0, 0.8)
  loose_ball_toward_goal_prob: float = 0.35
  loose_ball_angle_noise_deg: float = 65.0
  loose_ball_z_range: tuple[float, float] = (0.11, 0.18)
  loose_ball_x_margin: float = 0.06
  loose_ball_y_margin: float = 0.10

  # Variant B: post-contact rebound setup.
  # Surfaces: (foot, shin, forearm, torso_front).
  post_contact_surface_probs: tuple[float, float, float, float] = (
    0.40,
    0.25,
    0.20,
    0.15,
  )
  post_contact_x_offsets: tuple[float, float, float, float] = (
    -0.34,
    -0.26,
    -0.22,
    -0.20,
  )
  post_contact_y_offsets: tuple[float, float, float, float] = (
    0.16,
    0.20,
    0.32,
    0.10,
  )
  post_contact_z_offsets: tuple[float, float, float, float] = (
    0.11,
    0.28,
    0.88,
    0.72,
  )
  post_contact_offset_noise_xy: float = 0.06
  post_contact_offset_noise_z: float = 0.05
  post_contact_rebound_speed_range: tuple[float, float] = (0.0, 1.8)
  post_contact_rebound_zero_prob: float = 0.25
  post_contact_rebound_away_prob: float = 0.65
  post_contact_rebound_angle_noise_deg: float = 55.0
  post_contact_rebound_vz_range: tuple[float, float] = (-0.45, 0.45)

  # Clear-condition defaults (also reused by termination cfg).
  clear_progress_steps: int = 6
  clear_distance_increase_threshold: float = 0.008
  clear_strong_away_speed: float = 1.20

  # Keep command fixed for full episode.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False

  @dataclass
  class VizCfg:
    goal_plane_color: tuple[float, float, float, float] = (0.15, 0.85, 0.95, 0.85)
    danger_zone_color: tuple[float, float, float, float] = (0.95, 0.70, 0.15, 0.85)
    velocity_color: tuple[float, float, float, float] = (0.95, 0.85, 0.10, 0.85)
    line_radius: float = 0.008
    velocity_arrow_scale: float = 0.24
    velocity_arrow_width: float = 0.014

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env):
    return ClearAwayCommand(self, env)


class ClearAwayCommand(CommandTerm):
  cfg: ClearAwayCommandCfg

  def __init__(self, cfg: ClearAwayCommandCfg, env):
    super().__init__(cfg, env)
    self._robot: Entity = env.scene[cfg.entity_name]
    self._ball: Entity = env.scene[cfg.ball_entity_name]

    self._command = torch.zeros(env.num_envs, cfg.command_dim, device=self.device)
    self._variant_id = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
    self._keeper_spawn_pos_w = torch.zeros(env.num_envs, 3, device=self.device)

    self.metrics["outside_keeper_area"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["ball_speed_xy"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["goal_detected"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["in_danger_zone"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["goal_distance_xy"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["away_speed_x"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["reset_variant_id"] = torch.zeros(env.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def keeper_area_bounds(self) -> tuple[float, float, float, float]:
    return self.cfg.keeper_area_bounds

  @property
  def hard_keeper_area_bounds(self) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = self.cfg.keeper_area_bounds
    m = self.cfg.hard_area_margin
    return (x_min - m, x_max + m, y_min - m, y_max + m)

  @property
  def variant_id(self) -> torch.Tensor:
    return self._variant_id

  def _update_metrics(self) -> None:
    trunk_xy_local = _world_to_env_local_xy(
      self._env,
      self._robot.data.root_link_pos_w[:, :2],
    )
    self.metrics["outside_keeper_area"] = _outside_area_violation(
      trunk_xy_local,
      self.cfg.keeper_area_bounds,
    )

    ball_local = _world_to_env_local_xyz(self._env, self._ball.data.root_link_pos_w)
    vx = self._ball.data.root_link_lin_vel_w[:, 0]

    in_dz = _ball_in_danger_zone_mask_from_local(
      ball_local,
      vx,
      goal_toward_positive_x=self.cfg.goal_toward_positive_x,
      goal_plane_x=self.cfg.goal_plane_x,
      goal_plane_y_center=self.cfg.goal_plane_y_center,
      danger_zone_depth=self.cfg.danger_zone_depth,
      danger_zone_half_width=self.cfg.danger_zone_half_width,
      require_toward_goal=self.cfg.danger_zone_require_toward_goal,
      toward_goal_speed_threshold=self.cfg.danger_zone_toward_goal_speed_threshold,
    )

    self.metrics["ball_speed_xy"] = torch.linalg.norm(
      self._ball.data.root_link_lin_vel_w[:, :2],
      dim=1,
    )
    self.metrics["goal_detected"] = self._goal_conceded_mask().float()
    self.metrics["in_danger_zone"] = in_dz.float()
    self.metrics["goal_distance_xy"] = _ball_goal_distance_xy(
      ball_local,
      goal_plane_x=self.cfg.goal_plane_x,
      goal_plane_y_center=self.cfg.goal_plane_y_center,
    )
    self.metrics["away_speed_x"] = torch.clamp(
      _away_speed_from_vx(vx, self.cfg.goal_toward_positive_x),
      min=0.0,
    )
    self.metrics["reset_variant_id"] = self._variant_id.to(torch.float)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    n = len(env_ids)
    loose_prob = max(0.0, min(1.0, float(self.cfg.loose_variant_prob)))
    loose_mask = torch.rand(n, device=self.device) < loose_prob

    self._variant_id[env_ids] = torch.where(
      loose_mask,
      torch.zeros(n, device=self.device, dtype=torch.long),
      torch.ones(n, device=self.device, dtype=torch.long),
    )

    self._reset_robot_pose(env_ids, post_contact_mask=(~loose_mask))
    self._reset_ball_pose(env_ids, loose_mask=loose_mask)

    # Stage-1 decoder command input. For E3 we keep it deterministic zero.
    self._command[env_ids] = 0.0

  def _update_command(self) -> None:
    # E3 uses immediate reset-state dynamics (no delayed launch schedule).
    return

  def _reset_robot_pose(
    self,
    env_ids: torch.Tensor,
    post_contact_mask: torch.Tensor,
  ) -> None:
    default_root_state = self._robot.data.default_root_state
    default_joint_pos = self._robot.data.default_joint_pos
    default_joint_vel = self._robot.data.default_joint_vel

    assert default_root_state is not None
    assert default_joint_pos is not None
    assert default_joint_vel is not None

    n = len(env_ids)

    spawn_x = _sample_uniform_range(
      self.cfg.keeper_spawn_x_range[0],
      self.cfg.keeper_spawn_x_range[1],
      n,
      self.device,
    )
    spawn_y = _sample_uniform_range(
      self.cfg.keeper_spawn_y_range[0],
      self.cfg.keeper_spawn_y_range[1],
      n,
      self.device,
    )

    root_state = default_root_state[env_ids].clone()
    origins = self._env.scene.env_origins[env_ids]
    root_state[:, 0] = origins[:, 0] + spawn_x
    root_state[:, 1] = origins[:, 1] + spawn_y

    yaw_lo, yaw_hi = self.cfg.spawn_yaw_range
    if abs(yaw_hi - yaw_lo) <= 1.0e-9:
      yaw = torch.full((n,), float(yaw_lo), device=self.device)
    else:
      yaw = _sample_uniform_range(yaw_lo, yaw_hi, n, self.device)

    yaw_q = quat_from_euler_xyz(
      torch.zeros_like(yaw),
      torch.zeros_like(yaw),
      yaw,
    )
    root_state[:, 3:7] = quat_mul(root_state[:, 3:7], yaw_q)
    root_state[:, 7:13] = 0.0
    self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self._keeper_spawn_pos_w[env_ids] = root_state[:, :3]

    joint_pos = default_joint_pos[env_ids].clone()
    joint_vel = default_joint_vel[env_ids].clone()

    pos_noise_mag = torch.full(
      (n, 1),
      max(float(self.cfg.keeper_joint_pos_noise), 0.0),
      device=self.device,
    )
    vel_noise_mag = torch.full(
      (n, 1),
      max(float(self.cfg.keeper_joint_vel_noise), 0.0),
      device=self.device,
    )

    if post_contact_mask.any():
      pos_noise_mag[post_contact_mask] = max(
        float(self.cfg.post_contact_keeper_joint_pos_noise),
        0.0,
      )
      vel_noise_mag[post_contact_mask] = max(
        float(self.cfg.post_contact_keeper_joint_vel_noise),
        0.0,
      )

    if torch.any(pos_noise_mag > 0.0):
      joint_pos += (torch.rand_like(joint_pos) * 2.0 - 1.0) * pos_noise_mag
    if torch.any(vel_noise_mag > 0.0):
      joint_vel += (torch.rand_like(joint_vel) * 2.0 - 1.0) * vel_noise_mag

    self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self._robot.reset(env_ids=env_ids)

  def _reset_ball_pose(
    self,
    env_ids: torch.Tensor,
    loose_mask: torch.Tensor,
  ) -> None:
    if env_ids.numel() == 0:
      return

    loose_ids = env_ids[loose_mask]
    post_ids = env_ids[~loose_mask]

    if loose_ids.numel() > 0:
      self._reset_ball_loose(loose_ids)
    if post_ids.numel() > 0:
      self._reset_ball_post_contact(post_ids)

  def _reset_ball_loose(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    origins = self._env.scene.env_origins[env_ids]

    x_margin = max(float(self.cfg.loose_ball_x_margin), 0.0)
    y_margin = max(float(self.cfg.loose_ball_y_margin), 0.0)

    if self.cfg.goal_toward_positive_x:
      x_lo = float(self.cfg.goal_plane_x - self.cfg.danger_zone_depth + x_margin)
      x_hi = float(self.cfg.goal_plane_x - x_margin)
    else:
      x_lo = float(self.cfg.goal_plane_x + x_margin)
      x_hi = float(self.cfg.goal_plane_x + self.cfg.danger_zone_depth - x_margin)

    y_half = max(float(self.cfg.danger_zone_half_width) - y_margin, 0.15)
    y_lo = float(self.cfg.goal_plane_y_center - y_half)
    y_hi = float(self.cfg.goal_plane_y_center + y_half)

    x_local = _sample_uniform_range(x_lo, x_hi, n, self.device)
    y_local = _sample_uniform_range(y_lo, y_hi, n, self.device)
    z_local = _sample_uniform_range(
      self.cfg.loose_ball_z_range[0],
      self.cfg.loose_ball_z_range[1],
      n,
      self.device,
    )

    z_floor = float(self.cfg.loose_ball_z_range[0])
    z_local = torch.clamp(z_local, min=z_floor)

    pos_local = torch.stack([x_local, y_local, z_local], dim=1)
    pos_w = origins + pos_local

    vel_w = self._sample_loose_ball_velocity(n)
    self._write_ball_pose_and_velocity(env_ids, pos_w, vel_w)

  def _reset_ball_post_contact(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return

    if len(self.cfg.post_contact_surface_probs) != 4:
      raise ValueError("post_contact_surface_probs must have 4 entries.")

    robot_pos_w = self._keeper_spawn_pos_w[env_ids]
    origins = self._env.scene.env_origins[env_ids]

    surface = _sample_categorical(
      self.cfg.post_contact_surface_probs,
      n,
      self.device,
    )

    x_table = torch.tensor(self.cfg.post_contact_x_offsets, device=self.device)
    y_table = torch.tensor(self.cfg.post_contact_y_offsets, device=self.device)
    z_table = torch.tensor(self.cfg.post_contact_z_offsets, device=self.device)

    side = torch.where(
      torch.rand(n, device=self.device) < 0.5,
      torch.ones(n, device=self.device),
      -torch.ones(n, device=self.device),
    )

    x_off = x_table[surface]
    y_off = y_table[surface] * side
    z_off = z_table[surface]

    noise_xy = max(float(self.cfg.post_contact_offset_noise_xy), 0.0)
    noise_z = max(float(self.cfg.post_contact_offset_noise_z), 0.0)

    if noise_xy > 0.0:
      x_off += _sample_uniform_range(-noise_xy, noise_xy, n, self.device)
      y_off += _sample_uniform_range(-noise_xy, noise_xy, n, self.device)
    if noise_z > 0.0:
      z_off += _sample_uniform_range(-noise_z, noise_z, n, self.device)

    pos_w = robot_pos_w.clone()
    pos_w[:, 0] += x_off
    pos_w[:, 1] += y_off
    pos_w[:, 2] += z_off

    min_z_w = origins[:, 2] + float(self.cfg.loose_ball_z_range[0])
    pos_w[:, 2] = torch.maximum(pos_w[:, 2], min_z_w)

    vel_w = self._sample_post_contact_rebound_velocity(n)
    self._write_ball_pose_and_velocity(env_ids, pos_w, vel_w)

  def _sample_loose_ball_velocity(self, n: int) -> torch.Tensor:
    speed = _sample_uniform_range(
      self.cfg.loose_ball_speed_range[0],
      self.cfg.loose_ball_speed_range[1],
      n,
      self.device,
    )

    toward_mask = torch.rand(n, device=self.device) < float(
      self.cfg.loose_ball_toward_goal_prob
    )
    noise_deg = _sample_uniform_range(
      -float(self.cfg.loose_ball_angle_noise_deg),
      float(self.cfg.loose_ball_angle_noise_deg),
      n,
      self.device,
    )

    if self.cfg.goal_toward_positive_x:
      toward_center = 0.0
      away_center = torch.pi
    else:
      toward_center = torch.pi
      away_center = 0.0

    center = torch.where(
      toward_mask,
      torch.full((n,), toward_center, device=self.device),
      torch.full((n,), away_center, device=self.device),
    )
    angle = center + noise_deg * torch.pi / 180.0

    vx = speed * torch.cos(angle)
    vy = speed * torch.sin(angle)
    vz = torch.zeros(n, device=self.device)
    return torch.stack([vx, vy, vz], dim=1)

  def _sample_post_contact_rebound_velocity(self, n: int) -> torch.Tensor:
    speed = _sample_uniform_range(
      self.cfg.post_contact_rebound_speed_range[0],
      self.cfg.post_contact_rebound_speed_range[1],
      n,
      self.device,
    )

    zero_mask = torch.rand(n, device=self.device) < float(
      self.cfg.post_contact_rebound_zero_prob
    )
    away_mask = torch.rand(n, device=self.device) < float(
      self.cfg.post_contact_rebound_away_prob
    )

    noise_deg = _sample_uniform_range(
      -float(self.cfg.post_contact_rebound_angle_noise_deg),
      float(self.cfg.post_contact_rebound_angle_noise_deg),
      n,
      self.device,
    )

    if self.cfg.goal_toward_positive_x:
      away_center = torch.pi
      toward_center = 0.0
    else:
      away_center = 0.0
      toward_center = torch.pi

    center = torch.where(
      away_mask,
      torch.full((n,), away_center, device=self.device),
      torch.full((n,), toward_center, device=self.device),
    )

    angle = center + noise_deg * torch.pi / 180.0
    vx = speed * torch.cos(angle)
    vy = speed * torch.sin(angle)
    vz = _sample_uniform_range(
      self.cfg.post_contact_rebound_vz_range[0],
      self.cfg.post_contact_rebound_vz_range[1],
      n,
      self.device,
    )

    vel = torch.stack([vx, vy, vz], dim=1)
    vel[zero_mask] = 0.0
    return vel

  def _write_ball_pose_and_velocity(
    self,
    env_ids: torch.Tensor,
    pos_w: torch.Tensor,
    vel_w: torch.Tensor,
  ) -> None:
    default_root_state = self._ball.data.default_root_state
    assert default_root_state is not None

    root_state = default_root_state[env_ids].clone()
    root_state[:, :3] = pos_w
    root_state[:, 3:7] = 0.0
    root_state[:, 3] = 1.0
    root_state[:, 7:13] = 0.0

    self._ball.write_root_state_to_sim(root_state, env_ids=env_ids)
    self._ball.reset(env_ids=env_ids)
    self._set_ball_linear_velocity(env_ids, vel_w, clear_angular=True)

  def _set_ball_linear_velocity(
    self,
    env_ids: torch.Tensor,
    vel_w_xyz: torch.Tensor,
    clear_angular: bool,
  ) -> None:
    ball_vel = self._ball.data.root_link_vel_w[env_ids].clone()
    ball_vel[:, :3] = vel_w_xyz
    if clear_angular:
      ball_vel[:, 3:] = 0.0
    self._ball.write_root_link_velocity_to_sim(ball_vel, env_ids=env_ids)

  def _goal_conceded_mask(self) -> torch.Tensor:
    ball_local = _world_to_env_local_xyz(self._env, self._ball.data.root_link_pos_w)
    return _goal_conceded_mask_from_local(
      ball_local,
      goal_toward_positive_x=self.cfg.goal_toward_positive_x,
      goal_plane_x=self.cfg.goal_plane_x,
      goal_plane_y_center=self.cfg.goal_plane_y_center,
      goal_plane_y_half=self.cfg.goal_plane_y_half,
      goal_plane_z_min=self.cfg.goal_plane_z_min,
      goal_plane_z_max=self.cfg.goal_plane_z_max,
    )

  def _ball_in_danger_zone_mask(
    self,
    require_toward_goal: bool,
  ) -> torch.Tensor:
    ball_local = _world_to_env_local_xyz(self._env, self._ball.data.root_link_pos_w)
    vx = self._ball.data.root_link_lin_vel_w[:, 0]
    return _ball_in_danger_zone_mask_from_local(
      ball_local,
      vx,
      goal_toward_positive_x=self.cfg.goal_toward_positive_x,
      goal_plane_x=self.cfg.goal_plane_x,
      goal_plane_y_center=self.cfg.goal_plane_y_center,
      danger_zone_depth=self.cfg.danger_zone_depth,
      danger_zone_half_width=self.cfg.danger_zone_half_width,
      require_toward_goal=require_toward_goal,
      toward_goal_speed_threshold=self.cfg.danger_zone_toward_goal_speed_threshold,
    )

  def _debug_vis_impl(self, visualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    origin = self._env.scene.env_origins[batch]

    x = origin[0] + float(self.cfg.goal_plane_x)
    y0 = origin[1] + float(self.cfg.goal_plane_y_center) - float(self.cfg.goal_plane_y_half)
    y1 = origin[1] + float(self.cfg.goal_plane_y_center) + float(self.cfg.goal_plane_y_half)
    z0 = origin[2] + float(self.cfg.goal_plane_z_min)
    z1 = origin[2] + float(self.cfg.goal_plane_z_max)

    p0 = torch.tensor([x, y0, z0], device=self.device)
    p1 = torch.tensor([x, y1, z0], device=self.device)
    p2 = torch.tensor([x, y1, z1], device=self.device)
    p3 = torch.tensor([x, y0, z1], device=self.device)

    visualizer.add_cylinder(
      p0.cpu().numpy(),
      p1.cpu().numpy(),
      radius=float(self.cfg.viz.line_radius),
      color=self.cfg.viz.goal_plane_color,
      label="goal_plane_bottom",
    )
    visualizer.add_cylinder(
      p1.cpu().numpy(),
      p2.cpu().numpy(),
      radius=float(self.cfg.viz.line_radius),
      color=self.cfg.viz.goal_plane_color,
      label="goal_plane_right",
    )
    visualizer.add_cylinder(
      p2.cpu().numpy(),
      p3.cpu().numpy(),
      radius=float(self.cfg.viz.line_radius),
      color=self.cfg.viz.goal_plane_color,
      label="goal_plane_top",
    )
    visualizer.add_cylinder(
      p3.cpu().numpy(),
      p0.cpu().numpy(),
      radius=float(self.cfg.viz.line_radius),
      color=self.cfg.viz.goal_plane_color,
      label="goal_plane_left",
    )

    if self.cfg.goal_toward_positive_x:
      dz_x0 = origin[0] + float(self.cfg.goal_plane_x - self.cfg.danger_zone_depth)
      dz_x1 = origin[0] + float(self.cfg.goal_plane_x)
    else:
      dz_x0 = origin[0] + float(self.cfg.goal_plane_x)
      dz_x1 = origin[0] + float(self.cfg.goal_plane_x + self.cfg.danger_zone_depth)

    dz_y0 = origin[1] + float(self.cfg.goal_plane_y_center) - float(
      self.cfg.danger_zone_half_width
    )
    dz_y1 = origin[1] + float(self.cfg.goal_plane_y_center) + float(
      self.cfg.danger_zone_half_width
    )
    dz_z = origin[2] + 0.03

    d0 = torch.tensor([dz_x0, dz_y0, dz_z], device=self.device)
    d1 = torch.tensor([dz_x1, dz_y0, dz_z], device=self.device)
    d2 = torch.tensor([dz_x1, dz_y1, dz_z], device=self.device)
    d3 = torch.tensor([dz_x0, dz_y1, dz_z], device=self.device)

    visualizer.add_cylinder(
      d0.cpu().numpy(),
      d1.cpu().numpy(),
      radius=float(self.cfg.viz.line_radius),
      color=self.cfg.viz.danger_zone_color,
      label="danger_zone_edge_0",
    )
    visualizer.add_cylinder(
      d1.cpu().numpy(),
      d2.cpu().numpy(),
      radius=float(self.cfg.viz.line_radius),
      color=self.cfg.viz.danger_zone_color,
      label="danger_zone_edge_1",
    )
    visualizer.add_cylinder(
      d2.cpu().numpy(),
      d3.cpu().numpy(),
      radius=float(self.cfg.viz.line_radius),
      color=self.cfg.viz.danger_zone_color,
      label="danger_zone_edge_2",
    )
    visualizer.add_cylinder(
      d3.cpu().numpy(),
      d0.cpu().numpy(),
      radius=float(self.cfg.viz.line_radius),
      color=self.cfg.viz.danger_zone_color,
      label="danger_zone_edge_3",
    )

    ball_pos = self._ball.data.root_link_pos_w[batch]
    ball_vel = self._ball.data.root_link_lin_vel_w[batch]
    vel_end = ball_pos + ball_vel * float(self.cfg.viz.velocity_arrow_scale)
    visualizer.add_arrow(
      ball_pos.cpu().numpy(),
      vel_end.cpu().numpy(),
      color=self.cfg.viz.velocity_color,
      width=float(self.cfg.viz.velocity_arrow_width),
      label="ball_velocity",
    )


def _goal_conceded_mask_from_command(
  env,
  command_name: str,
) -> torch.Tensor:
  cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
  return cmd._goal_conceded_mask()


def _ball_in_danger_zone_mask_from_command(
  env,
  command_name: str,
  require_toward_goal: bool,
) -> torch.Tensor:
  cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
  return cmd._ball_in_danger_zone_mask(require_toward_goal=require_toward_goal)


def _ball_goal_distance_xy_from_command(
  env,
  command_name: str,
) -> torch.Tensor:
  cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  ball_local = _world_to_env_local_xyz(env, ball.data.root_link_pos_w)
  return _ball_goal_distance_xy(
    ball_local,
    goal_plane_x=cmd.cfg.goal_plane_x,
    goal_plane_y_center=cmd.cfg.goal_plane_y_center,
  )


# ---------------- Observations ----------------

def target_direction_xy(
  env,
  command_name: str = "clear_away",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  rel_xy = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  return _normalize_xy(rel_xy)


def ball_position_relative_xyz(
  env,
  command_name: str = "clear_away",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  return ball.data.root_link_pos_w - robot.data.root_link_pos_w


def ball_velocity_relative_xyz(
  env,
  command_name: str = "clear_away",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  return ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w


def ball_goal_distance_xy(
  env,
  command_name: str = "clear_away",
) -> torch.Tensor:
  return _ball_goal_distance_xy_from_command(env, command_name).unsqueeze(1)


def ball_in_danger_zone(
  env,
  command_name: str = "clear_away",
  require_toward_goal: bool = False,
) -> torch.Tensor:
  in_dz = _ball_in_danger_zone_mask_from_command(
    env,
    command_name,
    require_toward_goal=require_toward_goal,
  )
  return in_dz.float().unsqueeze(1)


# ---------------- Rewards ----------------

def goal_conceded_indicator(
  env,
  command_name: str = "clear_away",
) -> torch.Tensor:
  return _goal_conceded_mask_from_command(env, command_name).float()


def clear_success_reward(
  env,
  command_name: str = "clear_away",
  clear_term_name: str = "clear_condition",
) -> torch.Tensor:
  clear_done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  try:
    clear_done = env.termination_manager.get_term(clear_term_name)
  except KeyError:
    clear_done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  goal = _goal_conceded_mask_from_command(env, command_name)
  success = clear_done & (~goal)
  return success.float()


def distance_from_goal_progress_reward(
  env,
  command_name: str = "clear_away",
  clip_speed: float = 3.0,
) -> torch.Tensor:
  cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  ball_local = _world_to_env_local_xyz(env, ball.data.root_link_pos_w)

  rel = torch.stack(
    [
      ball_local[:, 0] - float(cmd.cfg.goal_plane_x),
      ball_local[:, 1] - float(cmd.cfg.goal_plane_y_center),
    ],
    dim=1,
  )
  dist = torch.linalg.norm(rel, dim=1).clamp_min(1.0e-6)
  radial_dir = rel / dist.unsqueeze(1)
  vel_xy = ball.data.root_link_lin_vel_w[:, :2]

  radial_speed = torch.sum(radial_dir * vel_xy, dim=1)
  return torch.clamp(radial_speed, min=0.0, max=float(clip_speed))


def outside_danger_zone_bonus(
  env,
  command_name: str = "clear_away",
  clip_speed: float = 2.5,
) -> torch.Tensor:
  cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  in_dz = _ball_in_danger_zone_mask_from_command(
    env,
    command_name,
    require_toward_goal=False,
  )
  outside = ~in_dz

  away_speed = _away_speed_from_vx(
    ball.data.root_link_lin_vel_w[:, 0],
    cmd.cfg.goal_toward_positive_x,
  )
  away_speed = torch.clamp(away_speed, min=0.0, max=float(clip_speed))
  return outside.float() * away_speed


def away_velocity_reward(
  env,
  command_name: str = "clear_away",
  clip_speed: float = 4.0,
) -> torch.Tensor:
  cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  away_speed = _away_speed_from_vx(
    ball.data.root_link_lin_vel_w[:, 0],
    cmd.cfg.goal_toward_positive_x,
  )
  return torch.clamp(away_speed, min=0.0, max=float(clip_speed))


def outside_keeper_area_penalty(
  env,
  command_name: str = "clear_away",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
  pos_xy_local = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
  return _outside_area_violation(pos_xy_local, cmd.keeper_area_bounds)


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


# ---------------- Terminations ----------------

def goal_conceded_termination(
  env,
  command_name: str = "clear_away",
) -> torch.Tensor:
  return _goal_conceded_mask_from_command(env, command_name)


def outside_keeper_area_hard(
  env,
  command_name: str = "clear_away",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
  pos_xy_local = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
  violation = _outside_area_violation(pos_xy_local, cmd.hard_keeper_area_bounds)
  return violation > 0.0


class ClearConditionTermination:
  """Terminate when ball is safely cleared from danger.

  Safe clear condition:
  - ball outside geometric danger zone
  - and either:
    - distance-to-goal increases for N consecutive steps
    - or away-from-goal speed exceeds a strong threshold
  """

  def __init__(self, cfg, env):
    del cfg
    self._progress_counter = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.long,
    )
    self._prev_goal_dist = torch.full(
      (env.num_envs,),
      fill_value=-1.0,
      device=env.device,
      dtype=torch.float,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._progress_counter[env_ids] = 0
    self._prev_goal_dist[env_ids] = -1.0

  def __call__(
    self,
    env,
    command_name: str = "clear_away",
    required_steps: int = 6,
    min_distance_increase: float = 0.008,
    strong_away_speed: float = 1.20,
  ) -> torch.Tensor:
    cmd = cast(ClearAwayCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    ball_local = _world_to_env_local_xyz(env, ball.data.root_link_pos_w)
    dist = _ball_goal_distance_xy(
      ball_local,
      goal_plane_x=cmd.cfg.goal_plane_x,
      goal_plane_y_center=cmd.cfg.goal_plane_y_center,
    )

    prev = self._prev_goal_dist
    has_prev = prev >= 0.0
    delta = dist - prev
    increasing = has_prev & (delta >= float(min_distance_increase))

    in_dz_geom = _ball_in_danger_zone_mask_from_local(
      ball_local,
      ball.data.root_link_lin_vel_w[:, 0],
      goal_toward_positive_x=cmd.cfg.goal_toward_positive_x,
      goal_plane_x=cmd.cfg.goal_plane_x,
      goal_plane_y_center=cmd.cfg.goal_plane_y_center,
      danger_zone_depth=cmd.cfg.danger_zone_depth,
      danger_zone_half_width=cmd.cfg.danger_zone_half_width,
      require_toward_goal=False,
      toward_goal_speed_threshold=cmd.cfg.danger_zone_toward_goal_speed_threshold,
    )
    outside_dz = ~in_dz_geom

    progress_ok = outside_dz & increasing
    self._progress_counter = torch.where(
      progress_ok,
      self._progress_counter + 1,
      torch.zeros_like(self._progress_counter),
    )

    away_speed = _away_speed_from_vx(
      ball.data.root_link_lin_vel_w[:, 0],
      cmd.cfg.goal_toward_positive_x,
    )
    strong_away = away_speed >= float(strong_away_speed)

    safe = outside_dz & (
      (self._progress_counter >= int(required_steps)) | strong_away
    )

    self._prev_goal_dist = dist
    return safe


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
