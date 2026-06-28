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
from mjlab.utils.lab_api.math import quat_from_euler_xyz

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _sample_uniform_range(low: float, high: float, num: int, device: str) -> torch.Tensor:
  return torch.rand(num, device=device) * (high - low) + low


def _normalize_xy(vec_xy: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
  return vec_xy / torch.linalg.norm(vec_xy, dim=1, keepdim=True).clamp_min(eps)


def _world_to_env_local_xy(env, pos_w_xy: torch.Tensor) -> torch.Tensor:
  return pos_w_xy - env.scene.env_origins[:, :2]


def _outside_area_violation(pos_xy: torch.Tensor, bounds: tuple[float, float, float, float]) -> torch.Tensor:
  x_min, x_max, y_min, y_max = bounds
  x, y = pos_xy[:, 0], pos_xy[:, 1]
  return (x_min - x).clamp_min(0.0) + (x - x_max).clamp_min(0.0) + (y_min - y).clamp_min(0.0) + (y - y_max).clamp_min(0.0)


def _get_bool_state_buffer(env, key: str) -> torch.Tensor:
  env_obj = getattr(env, "unwrapped", env)
  cache = getattr(env_obj, "_dribble_bool_state_cache", None)
  if cache is None:
    cache = {}
    setattr(env_obj, "_dribble_bool_state_cache", cache)
  buf = cache.get(key)
  if buf is None or buf.shape != (env.num_envs,) or buf.device != env.device:
    buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    cache[key] = buf
  return buf


def _get_float_state_buffer(env, key: str, shape_tail: tuple = (), dtype=torch.float32) -> torch.Tensor:
  env_obj = getattr(env, "unwrapped", env)
  cache = getattr(env_obj, "_dribble_float_state_cache", None)
  if cache is None:
    cache = {}
    setattr(env_obj, "_dribble_float_state_cache", cache)
  shape = (env.num_envs, *shape_tail)
  buf = cache.get(key)
  if buf is None or tuple(buf.shape) != shape or buf.device != env.device or buf.dtype != dtype:
    buf = torch.zeros(shape, device=env.device, dtype=dtype)
    cache[key] = buf
  return buf


def _sensor_any_found(env, sensor_name: str) -> torch.Tensor:
  s = env.scene[sensor_name]
  found = getattr(s.data, "found", None)
  if found is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  return torch.any(found > 0.0, dim=1)


def _default_root_height(robot: Entity) -> torch.Tensor:
  default_root_state = getattr(robot.data, "default_root_state", None)
  if default_root_state is not None:
    return default_root_state[:, 2].to(robot.data.root_link_pos_w.dtype)
  return robot.data.root_link_pos_w[:, 2]


@dataclass
class DribblingCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  ball_entity_name: str = "soccer_ball"
  obstacle_entity_prefix: str = "dribble_obstacle_"
  command_dim: int = 46

  # field/local geometry: train from midfield toward +X penalty area.
  field_half_length_x: float = 7.0
  field_half_width_y: float = 4.5
  ball_spawn_x: float = 0.0
  ball_spawn_y_range: tuple[float, float] = (-3.2, 3.2)
  ball_spawn_z: float = 0.11
  robot_distance_behind_ball: float = 0.50
  robot_spawn_y_jitter: float = 0.04
  robot_spawn_x_jitter: float = 0.02
  robot_yaw_jitter: float = 0.04

  target_x: float = 4.50              # penalty-area entry line on attacking half
  target_y_range: tuple[float, float] = (-2.8, 2.8)
  success_ball_x: float = 4.50
  success_robot_x: float = 4.15

  dribble_area_bounds: tuple[float, float, float, float] = (-0.90, 4.75, -4.0, 4.0)
  hard_area_margin: float = 0.45

  num_obstacles: int = 5
  active_obstacles_range: tuple[int, int] = (3, 5)
  obstacle_spawn_x_range: tuple[float, float] = (1.00, 4.10)
  obstacle_spawn_y_range: tuple[float, float] = (-3.25, 3.25)
  obstacle_keepout_from_start: float = 0.95
  obstacle_keepout_from_target: float = 0.75
  obstacle_min_pair_dist: float = 0.95
  inactive_obstacle_xy: tuple[float, float] = (-20.0, -20.0)
  obstacle_radius: float = 0.08
  obstacle_ball_safe_dist: float = 0.80
  obstacle_robot_hit_dist: float = 0.28

  debug_vis: bool = True
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env):
    return DribblingCommand(self, env)


class DribblingCommand(CommandTerm):
  cfg: DribblingCommandCfg

  def __init__(self, cfg: DribblingCommandCfg, env):
    super().__init__(cfg, env)
    self._robot: Entity = env.scene[cfg.entity_name]
    self._ball: Entity = env.scene[cfg.ball_entity_name]
    self._obstacles: list[Entity] = [env.scene[f"{cfg.obstacle_entity_prefix}{i}"] for i in range(cfg.num_obstacles)]
    self._command = torch.zeros(env.num_envs, cfg.command_dim, device=self.device)
    self._target_pos_w = torch.zeros(env.num_envs, 3, device=self.device)
    self._ball_spawn_xy_w = torch.zeros(env.num_envs, 2, device=self.device)
    self._obstacle_xy_w = torch.zeros(env.num_envs, cfg.num_obstacles, 2, device=self.device)
    self._obstacle_active = torch.zeros(env.num_envs, cfg.num_obstacles, device=self.device, dtype=torch.bool)
    self._trail_len = 48
    self._ball_trail_w = torch.zeros(env.num_envs, self._trail_len, 3, device=self.device)

    self.metrics["ball_target_dist"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["robot_ball_dist"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["ball_obstacle_min_dist"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["robot_obstacle_min_dist"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["success_event"] = torch.zeros(env.num_envs, device=self.device)
    self._success_latched = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def target_pos_w(self) -> torch.Tensor:
    return self._target_pos_w

  @property
  def obstacle_xy_w(self) -> torch.Tensor:
    return self._obstacle_xy_w

  @property
  def obstacle_active(self) -> torch.Tensor:
    return self._obstacle_active

  @property
  def hard_dribble_area_bounds(self) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = self.cfg.dribble_area_bounds
    m = self.cfg.hard_area_margin
    return (x_min - m, x_max + m, y_min - m, y_max + m)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self._success_latched[env_ids] = False
    self.metrics["success_event"][env_ids] = 0.0
    self._reset_ball_pose(env_ids)
    self._set_target_pose(env_ids)
    self._reset_robot_pose(env_ids)
    self._reset_obstacles(env_ids)
    self._command[env_ids] = 0.0

  def _update_command(self) -> None:
    pass

  def _set_target_pose(self, env_ids: torch.Tensor) -> None:
    origins = self._env.scene.env_origins[env_ids]
    ball_spawn_local_y = self._ball_spawn_xy_w[env_ids, 1] - origins[:, 1]

    rand = torch.rand((len(env_ids),), device=self.device)
    sign = torch.where(rand < 0.5, -torch.ones_like(rand), torch.ones_like(rand))
    delta = _sample_uniform_range(1.0, 2.4, len(env_ids), self.device)
    y = ball_spawn_local_y + sign * delta
    y = torch.clamp(y, float(self.cfg.target_y_range[0]), float(self.cfg.target_y_range[1]))

    # If clamping made target too close, flip side once.
    too_close = torch.abs(y - ball_spawn_local_y) < 0.8
    y_alt = ball_spawn_local_y - sign * delta
    y_alt = torch.clamp(y_alt, float(self.cfg.target_y_range[0]), float(self.cfg.target_y_range[1]))
    y = torch.where(too_close, y_alt, y)

    self._target_pos_w[env_ids, 0] = origins[:, 0] + float(self.cfg.target_x)
    self._target_pos_w[env_ids, 1] = origins[:, 1] + y
    self._target_pos_w[env_ids, 2] = origins[:, 2]

  def _reset_ball_pose(self, env_ids: torch.Tensor) -> None:
    default_root_state = self._ball.data.default_root_state
    assert default_root_state is not None
    root_state = default_root_state[env_ids].clone()
    origins = self._env.scene.env_origins[env_ids]
    by = _sample_uniform_range(self.cfg.ball_spawn_y_range[0], self.cfg.ball_spawn_y_range[1], len(env_ids), self.device)
    root_state[:, 0] = origins[:, 0] + float(self.cfg.ball_spawn_x)
    root_state[:, 1] = origins[:, 1] + by
    root_state[:, 2] = origins[:, 2] + float(self.cfg.ball_spawn_z)
    root_state[:, 3:7] = 0.0
    root_state[:, 3] = 1.0
    root_state[:, 7:13] = 0.0
    self._ball_spawn_xy_w[env_ids] = root_state[:, 0:2]
    self._ball.write_root_state_to_sim(root_state, env_ids=env_ids)
    self._ball.clear_state(env_ids=env_ids)
    if hasattr(self, "_ball_trail_w"):
      self._ball_trail_w[env_ids] = root_state[:, None, 0:3]

  def _reset_robot_pose(self, env_ids: torch.Tensor) -> None:
    default_root_state = self._robot.data.default_root_state
    default_joint_pos = self._robot.data.default_joint_pos
    default_joint_vel = self._robot.data.default_joint_vel
    assert default_root_state is not None and default_joint_pos is not None and default_joint_vel is not None
    root_state = default_root_state[env_ids].clone()
    bx_by = self._ball_spawn_xy_w[env_ids]
    xjit = _sample_uniform_range(-self.cfg.robot_spawn_x_jitter, self.cfg.robot_spawn_x_jitter, len(env_ids), self.device)
    yjit = _sample_uniform_range(-self.cfg.robot_spawn_y_jitter, self.cfg.robot_spawn_y_jitter, len(env_ids), self.device)
    yaw = _sample_uniform_range(-self.cfg.robot_yaw_jitter, self.cfg.robot_yaw_jitter, len(env_ids), self.device)
    root_state[:, 0] = bx_by[:, 0] - float(self.cfg.robot_distance_behind_ball) + xjit
    root_state[:, 1] = bx_by[:, 1] + yjit
    root_state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    root_state[:, 7:13] = 0.0
    self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self._robot.write_joint_state_to_sim(default_joint_pos[env_ids], default_joint_vel[env_ids], env_ids=env_ids)
    self._robot.clear_state(env_ids=env_ids)

  def _reset_obstacles(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    lo, hi = int(self.cfg.active_obstacles_range[0]), int(self.cfg.active_obstacles_range[1])
    active_count = torch.randint(lo, hi + 1, (n,), device=self.device)

    origins = self._env.scene.env_origins[env_ids, :2]
    ball_local = self._ball_spawn_xy_w[env_ids] - origins
    target_local = self._target_pos_w[env_ids, :2] - origins

    local_xy = torch.empty(n, self.cfg.num_obstacles, 2, device=self.device)
    active = torch.zeros(n, self.cfg.num_obstacles, device=self.device, dtype=torch.bool)
    inactive = torch.tensor(self.cfg.inactive_obstacle_xy, device=self.device, dtype=torch.float32)
    local_xy[:] = inactive

    # Vector-friendly rejection sampling. It is intentionally conservative: if it cannot
    # place a cone cleanly after a few tries, that cone is disabled for that env.
    for k in range(self.cfg.num_obstacles):
      want = active_count > k
      placed = torch.zeros(n, device=self.device, dtype=torch.bool)
      sample = torch.zeros(n, 2, device=self.device)

      if k == 0:
        line = target_local - ball_local
        line_len = torch.linalg.norm(line, dim=1).clamp_min(1.0e-6)
        line_dir = line / line_len[:, None]
        normal = torch.stack([-line_dir[:, 1], line_dir[:, 0]], dim=1)
        for _ in range(20):
          s_frac = _sample_uniform_range(0.35, 0.70, n, self.device)
          lateral = _sample_uniform_range(-0.20, 0.20, n, self.device)
          cand = ball_local + s_frac[:, None] * line + lateral[:, None] * normal
          cand[:, 1] = torch.clamp(
            cand[:, 1],
            float(self.cfg.obstacle_spawn_y_range[0]),
            float(self.cfg.obstacle_spawn_y_range[1]),
          )
          far_start = torch.linalg.norm(cand - ball_local, dim=1) > float(self.cfg.obstacle_keepout_from_start)
          far_target = torch.linalg.norm(cand - target_local, dim=1) > float(self.cfg.obstacle_keepout_from_target)
          ok = want & (~placed) & far_start & far_target
          sample[ok] = cand[ok]
          placed |= ok

      for _ in range(40):
        cand = torch.stack([
          _sample_uniform_range(self.cfg.obstacle_spawn_x_range[0], self.cfg.obstacle_spawn_x_range[1], n, self.device),
          _sample_uniform_range(self.cfg.obstacle_spawn_y_range[0], self.cfg.obstacle_spawn_y_range[1], n, self.device),
        ], dim=1)
        far_start = torch.linalg.norm(cand - ball_local, dim=1) > float(self.cfg.obstacle_keepout_from_start)
        far_target = torch.linalg.norm(cand - target_local, dim=1) > float(self.cfg.obstacle_keepout_from_target)
        pair_ok = torch.ones(n, device=self.device, dtype=torch.bool)
        if k > 0:
          prev = local_xy[:, :k, :]
          prev_active = active[:, :k]
          d = torch.linalg.norm(cand[:, None, :] - prev, dim=2)
          d = torch.where(prev_active, d, torch.full_like(d, 1.0e6))
          pair_ok = torch.min(d, dim=1).values > float(self.cfg.obstacle_min_pair_dist)
        ok = want & (~placed) & far_start & far_target & pair_ok
        sample[ok] = cand[ok]
        placed |= ok
      active[:, k] = placed
      local_xy[placed, k, :] = sample[placed]

    world_xy = local_xy + origins[:, None, :]
    self._obstacle_xy_w[env_ids] = world_xy
    self._obstacle_active[env_ids] = active

    for k, obstacle in enumerate(self._obstacles):
      default = obstacle.data.default_root_state
      assert default is not None
      root_state = default[env_ids].clone()
      root_state[:, 0:2] = world_xy[:, k, :]
      root_state[:, 2] = self._env.scene.env_origins[env_ids, 2] + 0.35
      root_state[:, 3:7] = 0.0
      root_state[:, 3] = 1.0
      root_state[:, 7:13] = 0.0
      obstacle.write_root_state_to_sim(root_state, env_ids=env_ids)
      obstacle.clear_state(env_ids=env_ids)

  def _update_metrics(self) -> None:
    ball_xy = self._ball.data.root_link_pos_w[:, :2]
    if hasattr(self, "_ball_trail_w"):
      self._ball_trail_w = torch.roll(self._ball_trail_w, shifts=-1, dims=1)
      self._ball_trail_w[:, -1, :] = self._ball.data.root_link_pos_w[:, :3]

    robot_xy = self._robot.data.root_link_pos_w[:, :2]
    self.metrics["ball_target_dist"] = torch.linalg.norm(self._target_pos_w[:, :2] - ball_xy, dim=1)
    self.metrics["robot_ball_dist"] = torch.linalg.norm(ball_xy - robot_xy, dim=1)

    d_ball = torch.linalg.norm(ball_xy[:, None, :] - self._obstacle_xy_w, dim=2)
    d_robot = torch.linalg.norm(robot_xy[:, None, :] - self._obstacle_xy_w, dim=2)
    inf = torch.full_like(d_ball, 1.0e6)
    d_ball = torch.where(self._obstacle_active, d_ball, inf)
    d_robot = torch.where(self._obstacle_active, d_robot, inf)
    self.metrics["ball_obstacle_min_dist"] = torch.min(d_ball, dim=1).values
    self.metrics["robot_obstacle_min_dist"] = torch.min(d_robot, dim=1).values

    origins = self._env.scene.env_origins
    ball_local = self._ball.data.root_link_pos_w - origins
    robot_local = self._robot.data.root_link_pos_w - origins
    close_control = self.metrics["robot_ball_dist"] < 0.85
    ball_inside = ball_local[:, 0] >= float(self.cfg.success_ball_x)
    robot_inside = robot_local[:, 0] >= float(self.cfg.success_robot_x)
    success_now = close_control & ball_inside & robot_inside
    event = success_now & (~self._success_latched)
    self._success_latched |= success_now
    self.metrics["success_event"] = event.to(torch.float32)

  def _debug_vis_impl(self, visualizer) -> None:
    if not self.cfg.debug_vis:
      return
    env_indices = visualizer.get_env_indices(self.num_envs)
    for batch in env_indices:
      target = self._target_pos_w[batch]
      visualizer.add_sphere(target.cpu().numpy(), radius=0.12, color=(0.1, 0.9, 0.2, 0.35), label=f"dribble_target_{batch}")
      if hasattr(self, "_ball_trail_w"):
        trail = self._ball_trail_w[batch]
        for j in range(self._trail_len):
          alpha = 0.10 + 0.45 * (j / max(self._trail_len - 1, 1))
          visualizer.add_sphere(
            trail[j].cpu().numpy(),
            radius=0.035,
            color=(0.2, 0.7, 1.0, alpha),
            label=f"dribble_ball_trail_{batch}_{j}",
          )


# ---------------- Observations ----------------

def target_direction_xy(env, command_name: str = "dribble", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  return _normalize_xy(cmd.target_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2])


def target_position_relative_xy(env, command_name: str = "dribble", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  return cmd.target_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]


def ball_position_relative_xyz(env, command_name: str = "dribble", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  return ball.data.root_link_pos_w - robot.data.root_link_pos_w


def ball_velocity_w_xy(env, ball_entity_name: str = "soccer_ball") -> torch.Tensor:
  ball: Entity = env.scene[ball_entity_name]
  return ball.data.root_link_lin_vel_w[:, :2]


def ball_dist_xy_obs(env, command_name: str = "dribble") -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  return cmd.metrics["robot_ball_dist"].unsqueeze(-1).to(torch.float32)


def obstacle_positions_relative_to_ball_obs(env, command_name: str = "dribble") -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball_xy = env.scene[cmd.cfg.ball_entity_name].data.root_link_pos_w[:, :2]
  rel = cmd.obstacle_xy_w - ball_xy[:, None, :]
  rel = torch.where(cmd.obstacle_active[..., None], rel, torch.zeros_like(rel))
  return rel.reshape(env.num_envs, -1).to(torch.float32)


def obstacle_active_mask_obs(env, command_name: str = "dribble") -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  return cmd.obstacle_active.to(torch.float32)


def min_ball_obstacle_dist_obs(env, command_name: str = "dribble") -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  return cmd.metrics["ball_obstacle_min_dist"].clamp(max=5.0).unsqueeze(-1).to(torch.float32)


def right_foot_pos_rel_ball_xy_obs(env, command_name: str = "dribble", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  ids, _ = robot.find_bodies((r"^right_foot_link$",), preserve_order=True)
  if len(ids) != 1 or not hasattr(robot.data, "body_link_pos_w"):
    return torch.zeros(env.num_envs, 2, device=env.device)
  return (robot.data.body_link_pos_w[:, int(ids[0]), :2] - ball.data.root_link_pos_w[:, :2]).to(torch.float32)


def left_foot_pos_rel_ball_xy_obs(env, command_name: str = "dribble", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  ids, _ = robot.find_bodies((r"^left_foot_link$",), preserve_order=True)
  if len(ids) != 1 or not hasattr(robot.data, "body_link_pos_w"):
    return torch.zeros(env.num_envs, 2, device=env.device)
  return (robot.data.body_link_pos_w[:, int(ids[0]), :2] - ball.data.root_link_pos_w[:, :2]).to(torch.float32)


# ---------------- Rewards ----------------

def striker_posture_score(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, roll_band: float = 0.07, roll_sigma: float = 0.12, pitch_target: float = 0.14, pitch_band: float = 0.12, pitch_sigma: float = 0.25) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  proj_g = robot.data.projected_gravity_b
  sagittal, lateral = proj_g[:, 0], proj_g[:, 1]
  roll_error = (torch.abs(lateral) - float(roll_band)).clamp_min(0.0)
  pitch_error = (torch.abs(sagittal - float(pitch_target)) - float(pitch_band)).clamp_min(0.0)
  return torch.exp(-torch.square(roll_error) / max(float(roll_sigma) ** 2, 1e-6)) * torch.exp(-torch.square(pitch_error) / max(float(pitch_sigma) ** 2, 1e-6))


def upright_stability_reward(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, height_target: float | None = None, height_sigma: float = 0.14, **kwargs) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  h = robot.data.root_link_pos_w[:, 2]
  target = _default_root_height(robot) if height_target is None else torch.full_like(h, float(height_target))
  h_score = torch.exp(-torch.square((h - target) / max(float(height_sigma), 1e-6)))
  return h_score * striker_posture_score(env, asset_cfg=asset_cfg, **kwargs)


def striker_low_height_soft_penalty(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, h_soft: float | None = None, scale: float = 0.06) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  h = robot.data.root_link_pos_w[:, 2]
  if h_soft is None:
    h_soft_t = (_default_root_height(robot) - 0.06).clamp_min(0.50)
  else:
    h_soft_t = torch.full_like(h, float(h_soft))
  return torch.square(((h_soft_t - h).clamp_min(0.0) / float(scale)).clamp(0.0, 2.5))


def fallen_indicator(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, min_height: float = 0.30, max_tilt: float = 1.20) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  h = robot.data.root_link_pos_w[:, 2]
  tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
  return ((h < float(min_height)) | (tilt > float(max_tilt))).to(torch.float32)


def double_knee_crouch_penalty(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, free_flex: float = 0.18, max_flex: float = 0.85) -> torch.Tensor:
  # generic, name-based fallback: penalize large deviation of any knee-like joint from default.
  robot: Entity = env.scene[asset_cfg.name]
  vals = []
  for i, name in enumerate(robot.joint_names):
    if "knee" in name.lower() or "kne" in name.lower():
      flex = torch.abs(robot.data.joint_pos[:, i] - robot.data.default_joint_pos[:, i])
      vals.append(torch.square(((flex - free_flex) / max(max_flex - free_flex, 1e-6)).clamp(0.0, 1.0)))
  if not vals:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.stack(vals, dim=1).mean(dim=1)


def ball_to_target_progress_reward(env, command_name: str = "dribble", max_delta: float = 0.08, upright_gate: float = 0.45) -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  dist = cmd.metrics["ball_target_dist"].to(torch.float32)
  prev = _get_float_state_buffer(env, f"dribble_prev_target_dist::{command_name}")
  first = env.episode_length_buf <= 1
  prev[first] = dist[first]
  prog = (prev - dist).clamp(0.0, float(max_delta))
  prev.copy_(dist)
  return prog * (upright_stability_reward(env) > upright_gate).to(torch.float32)


def ball_forward_velocity_reward(env, command_name: str = "dribble", max_speed: float = 1.8) -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball = env.scene[cmd.cfg.ball_entity_name]
  dir_xy = _normalize_xy(cmd.target_pos_w[:, :2] - ball.data.root_link_pos_w[:, :2])
  proj = torch.sum(ball.data.root_link_lin_vel_w[:, :2] * dir_xy, dim=1).clamp(0.0, float(max_speed))
  return proj / float(max_speed)


def ball_velocity_tracking_reward(
  env,
  command_name: str = "dribble",
  target_speed: float = 0.45,
  speed_sigma: float = 0.25,
  dir_sigma: float = 0.65,
  min_robot_ball_dist: float = 0.15,
  max_robot_ball_dist: float = 0.95,
) -> torch.Tensor:
  """
  Reward controlled ball motion toward the target.
  This is different from maximizing forward velocity:
  it rewards a desired slow speed and penalizes too much lateral drift.
  """
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball = env.scene[cmd.cfg.ball_entity_name]

  ball_xy = ball.data.root_link_pos_w[:, :2]
  ball_v = ball.data.root_link_lin_vel_w[:, :2]

  target_dir = _normalize_xy(cmd.target_pos_w[:, :2] - ball_xy)
  forward_v = torch.sum(ball_v * target_dir, dim=1)
  lateral_v = torch.linalg.norm(ball_v - forward_v[:, None] * target_dir, dim=1)

  speed_score = torch.exp(
    -torch.square((forward_v - float(target_speed)) / max(float(speed_sigma), 1.0e-6))
  )

  dir_score = torch.exp(
    -torch.square(lateral_v / max(float(dir_sigma), 1.0e-6))
  )

  d = cmd.metrics["robot_ball_dist"]
  close_gate = ((d > float(min_robot_ball_dist)) & (d < float(max_robot_ball_dist))).to(torch.float32)
  forward_gate = (forward_v > 0.05).to(torch.float32)

  return speed_score * dir_score * close_gate * forward_gate


def ball_obstacle_aware_velocity_reward(
  env,
  command_name: str = "dribble",
  target_speed: float = 0.35,
  speed_sigma: float = 0.22,
  dir_sigma: float = 0.50,
  influence_dist: float = 1.60,
  lookahead_dist: float = 2.20,
  lateral_influence: float = 0.95,
  repel_gain: float = 1.35,
  min_robot_ball_dist: float = 0.12,
  max_robot_ball_dist: float = 0.95,
) -> torch.Tensor:
  """
  Reward ball velocity along an obstacle-aware desired direction.

  Base desired direction is ball -> target.
  If an active obstacle is ahead and close to the ball-target corridor,
  add a repulsive component away from the obstacle.

  This encourages curved trajectories around obstacles instead of
  a straight kick-and-chase behavior.
  """
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball = env.scene[cmd.cfg.ball_entity_name]

  ball_xy = ball.data.root_link_pos_w[:, :2]
  ball_v = ball.data.root_link_lin_vel_w[:, :2].to(torch.float32)

  target_dir = _normalize_xy(cmd.target_pos_w[:, :2] - ball_xy)

  obs_xy = cmd.obstacle_xy_w
  active = cmd.obstacle_active

  to_obs = obs_xy - ball_xy[:, None, :]
  dist = torch.linalg.norm(to_obs, dim=2).clamp_min(1.0e-6)

  forward = torch.sum(to_obs * target_dir[:, None, :], dim=2)

  # 2D lateral distance from the ball-target ray.
  proj = forward[:, :, None] * target_dir[:, None, :]
  lateral_vec = to_obs - proj
  lateral = torch.linalg.norm(lateral_vec, dim=2)

  ahead_gate = (forward > 0.0) & (forward < float(lookahead_dist))
  near_gate = dist < float(influence_dist)
  corridor_gate = lateral < float(lateral_influence)
  gate = active & ahead_gate & near_gate & corridor_gate

  away = -to_obs / dist[:, :, None]
  strength = ((float(influence_dist) - dist) / float(influence_dist)).clamp(0.0, 1.0)
  repel = torch.sum(torch.where(gate[:, :, None], away * strength[:, :, None], torch.zeros_like(away)), dim=1)

  desired_dir = _normalize_xy(target_dir + float(repel_gain) * repel)

  forward_v = torch.sum(ball_v * desired_dir, dim=1)
  lateral_v = torch.linalg.norm(ball_v - forward_v[:, None] * desired_dir, dim=1)

  speed_score = torch.exp(
    -torch.square((forward_v - float(target_speed)) / max(float(speed_sigma), 1.0e-6))
  )
  dir_score = torch.exp(
    -torch.square(lateral_v / max(float(dir_sigma), 1.0e-6))
  )

  d = cmd.metrics["robot_ball_dist"]
  close_gate = ((d > float(min_robot_ball_dist)) & (d < float(max_robot_ball_dist))).to(torch.float32)
  forward_gate = (forward_v > 0.04).to(torch.float32)

  return speed_score * dir_score * close_gate * forward_gate


def ball_speed_limit_penalty(
  env,
  command_name: str = "dribble",
  free_speed: float = 0.85,
  hard_speed: float = 1.60,
) -> torch.Tensor:
  """
  Penalize the ball when it is kicked too hard.
  This discourages the kick-and-chase behavior.
  """
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball = env.scene[cmd.cfg.ball_entity_name]

  speed = torch.linalg.norm(ball.data.root_link_lin_vel_w[:, :2], dim=1)

  return ((speed - float(free_speed)) / max(float(hard_speed - free_speed), 1.0e-6)).clamp(0.0, 2.0)


def ball_path_lane_reward(
  env,
  command_name: str = "dribble",
  lane_sigma: float = 0.45,
  forward_margin: float = 0.15,
) -> torch.Tensor:
  """
  Reward the ball for staying close to the ideal line from ball spawn to target.
  This teaches meaningful trajectory shaping, not just final arrival.
  """
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball = env.scene[cmd.cfg.ball_entity_name]

  p0 = cmd._ball_spawn_xy_w
  p1 = cmd.target_pos_w[:, :2]
  p = ball.data.root_link_pos_w[:, :2]

  line = p1 - p0
  line_norm = torch.linalg.norm(line, dim=1).clamp_min(1.0e-6)
  line_dir = line / line_norm[:, None]

  rel = p - p0
  s = torch.sum(rel * line_dir, dim=1)

  # signed/absolute lateral distance to line in 2D
  proj = p0 + s[:, None] * line_dir
  lateral_dist = torch.linalg.norm(p - proj, dim=1)

  # Only reward after the ball has started moving forward along the path.
  forward_gate = (s > float(forward_margin)).to(torch.float32)

  return torch.exp(-torch.square(lateral_dist / max(float(lane_sigma), 1.0e-6))) * forward_gate


def robot_behind_ball_reward(
  env,
  command_name: str = "dribble",
  desired_behind_dist: float = 0.45,
  behind_sigma: float = 0.28,
  lateral_sigma: float = 0.30,
  max_ball_dist: float = 1.10,
) -> torch.Tensor:
  """
  Reward the robot for being behind the ball relative to the desired ball direction.
  This helps the robot push the ball in the right direction instead of just chasing it.
  """
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  robot = env.scene[cmd.cfg.entity_name]
  ball = env.scene[cmd.cfg.ball_entity_name]

  ball_xy = ball.data.root_link_pos_w[:, :2]
  robot_xy = robot.data.root_link_pos_w[:, :2]

  target_dir = _normalize_xy(cmd.target_pos_w[:, :2] - ball_xy)

  # robot relative to ball: positive behind means robot is opposite target_dir
  rel = robot_xy - ball_xy
  behind = -torch.sum(rel * target_dir, dim=1)
  lateral_vec = rel + behind[:, None] * target_dir
  lateral = torch.linalg.norm(lateral_vec, dim=1)

  behind_score = torch.exp(
    -torch.square((behind - float(desired_behind_dist)) / max(float(behind_sigma), 1.0e-6))
  )
  lateral_score = torch.exp(
    -torch.square(lateral / max(float(lateral_sigma), 1.0e-6))
  )

  d = cmd.metrics["robot_ball_dist"]
  close_gate = (d < float(max_ball_dist)).to(torch.float32)
  upright_gate = (upright_stability_reward(env) > 0.20).to(torch.float32)

  return behind_score * lateral_score * close_gate * upright_gate


def ball_accel_limit_penalty(
  env,
  command_name: str = "dribble",
  free_delta_speed: float = 0.28,
  hard_delta_speed: float = 0.85,
) -> torch.Tensor:
  """
  Penalize sudden ball velocity jumps.
  This discourages one strong kick and encourages smaller repeated touches.
  """
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball = env.scene[cmd.cfg.ball_entity_name]

  v = ball.data.root_link_lin_vel_w[:, :2].to(torch.float32)
  prev = _get_float_state_buffer(env, f"dribble_prev_ball_vel_xy::{command_name}", shape_tail=(2,))
  first = env.episode_length_buf <= 1
  prev[first] = v[first]

  dv = torch.linalg.norm(v - prev, dim=1)
  prev.copy_(v)

  return ((dv - float(free_delta_speed)) / max(float(hard_delta_speed - free_delta_speed), 1.0e-6)).clamp(0.0, 2.0)


def keep_ball_close_reward(env, command_name: str = "dribble", sigma: float = 0.55) -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  d = cmd.metrics["robot_ball_dist"]
  return torch.exp(-torch.square(d / float(sigma)))


def ball_too_far_penalty(env, command_name: str = "dribble", free_dist: float = 0.80, max_dist: float = 1.00) -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  d = cmd.metrics["robot_ball_dist"]
  return ((d - float(free_dist)) / max(float(max_dist - free_dist), 1e-6)).clamp(0.0, 2.0)


def obstacle_ball_clearance_reward(env, command_name: str = "dribble", safe_dist: float = 0.80, sigma: float = 0.35) -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  d = cmd.metrics["ball_obstacle_min_dist"]
  return torch.sigmoid((d - float(safe_dist)) / max(float(sigma), 1e-6))


def obstacle_ball_near_penalty(env, command_name: str = "dribble", safe_dist: float = 0.80, hard_dist: float = 0.25) -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  d = cmd.metrics["ball_obstacle_min_dist"]
  return ((float(safe_dist) - d) / max(float(safe_dist - hard_dist), 1e-6)).clamp(0.0, 1.5)


def success_event_reward(env, command_name: str = "dribble") -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  return cmd.metrics["success_event"]


def time_pressure_reward(env) -> torch.Tensor:
  # Positive alive-time is deliberately NOT used. This is a small per-step cost via negative weight.
  return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)


def action_rate_l2(env) -> torch.Tensor:
  a = env.action_manager.action
  pa = env.action_manager.prev_action
  return torch.mean((a - pa) ** 2, dim=1)


def feet_ball_control_reward(env, command_name: str = "dribble", near_dist: float = 0.28, max_foot_speed: float = 2.5) -> torch.Tensor:
  robot = env.scene["robot"]
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball = env.scene[cmd.cfg.ball_entity_name]
  ids, _ = robot.find_bodies((r"^left_foot_link$", r"^right_foot_link$"), preserve_order=True)
  if len(ids) != 2 or not hasattr(robot.data, "body_link_pos_w"):
    return torch.zeros(env.num_envs, device=env.device)
  feet = robot.data.body_link_pos_w[:, [int(ids[0]), int(ids[1])], :]
  d = torch.linalg.norm(feet[:, :, :2] - ball.data.root_link_pos_w[:, None, :2], dim=2)
  near = torch.exp(-torch.square(torch.min(d, dim=1).values / float(near_dist)))
  if hasattr(robot.data, "body_com_lin_vel_w"):
    v = robot.data.body_com_lin_vel_w[:, [int(ids[0]), int(ids[1])], :2]
    speed = torch.min(torch.linalg.norm(v, dim=2), dim=1).values
    speed_gate = torch.clamp(speed / float(max_foot_speed), 0.0, 1.0)
    return near * speed_gate
  return near


def foot_over_ball_penalty(env, command_name: str = "dribble", xy_near: float = 0.16, z_margin: float = 0.015) -> torch.Tensor:
  robot = env.scene["robot"]
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball = env.scene[cmd.cfg.ball_entity_name]
  ids, _ = robot.find_bodies((r"^left_foot_link$", r"^right_foot_link$"), preserve_order=True)
  if len(ids) != 2 or not hasattr(robot.data, "body_link_pos_w"):
    return torch.zeros(env.num_envs, device=env.device)
  feet = robot.data.body_link_pos_w[:, [int(ids[0]), int(ids[1])], :]
  d_xy = torch.linalg.norm(feet[:, :, :2] - ball.data.root_link_pos_w[:, None, :2], dim=2)
  over = (d_xy < float(xy_near)) & (feet[:, :, 2] > (ball.data.root_link_pos_w[:, 2:3] + float(z_margin)))
  return torch.any(over, dim=1).to(torch.float32)


def outside_dribble_area_penalty(env, command_name: str = "dribble") -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  robot = env.scene[cmd.cfg.entity_name]
  pos_local = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
  return _outside_area_violation(pos_local, cmd.cfg.dribble_area_bounds)


# ---------------- Terminations ----------------

class FallTermination:
  def __init__(self, cfg, env):
    del cfg
    self._counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._counter[env_ids] = 0
  def __call__(self, env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, min_height: float = 0.30, max_tilt: float = 1.20, consecutive_steps: int = 6) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    fallen = (robot.data.root_link_pos_w[:, 2] < min_height) | (torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1) > max_tilt)
    self._counter = torch.where(fallen, self._counter + 1, torch.zeros_like(self._counter))
    return self._counter >= int(consecutive_steps)


def success_termination(env, command_name: str = "dribble") -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  return cmd._success_latched


def ball_lost_termination(env, command_name: str = "dribble", max_dist: float = 1.20) -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  return cmd.metrics["robot_ball_dist"] > float(max_dist)


def hard_outside_dribble_area_termination(env, command_name: str = "dribble") -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  robot = env.scene[cmd.cfg.entity_name]
  pos_local = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
  return _outside_area_violation(pos_local, cmd.hard_dribble_area_bounds) > 0.0


def robot_obstacle_hit_termination(env, command_name: str = "dribble", hit_dist: float = 0.28) -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  return cmd.metrics["robot_obstacle_min_dist"] < float(hit_dist)


def ball_out_of_field_termination(env, command_name: str = "dribble", margin: float = 0.15) -> torch.Tensor:
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  ball = env.scene[cmd.cfg.ball_entity_name]
  local = ball.data.root_link_pos_w - env.scene.env_origins
  return (torch.abs(local[:, 0]) > (cmd.cfg.field_half_length_x + margin)) | (torch.abs(local[:, 1]) > (cmd.cfg.field_half_width_y + margin))

def robot_forward_velocity_reward(env, command_name: str = "dribble", max_speed: float = 1.2):
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  robot = env.scene[cmd.cfg.entity_name]

  dir_xy = _normalize_xy(cmd.target_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2])
  v_xy = robot.data.root_link_lin_vel_w[:, :2]
  proj = torch.sum(v_xy * dir_xy, dim=1).clamp(0.0, float(max_speed))

  upright = upright_stability_reward(env)
  return (proj / float(max_speed)) * (upright > 0.25).to(torch.float32)


def robot_follow_ball_reward(env, command_name: str = "dribble", desired_dist: float = 0.45, sigma: float = 0.35):
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  d = cmd.metrics["robot_ball_dist"]
  return torch.exp(-0.5 * torch.square((d - float(desired_dist)) / float(sigma)))


def alive_moving_reward(env, command_name: str = "dribble", min_speed: float = 0.15, max_speed: float = 1.2):
  cmd = cast(DribblingCommand, env.command_manager.get_term(command_name))
  robot = env.scene[cmd.cfg.entity_name]
  speed = torch.linalg.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
  return ((speed - min_speed) / max(max_speed - min_speed, 1e-6)).clamp(0.0, 1.0) * (upright_stability_reward(env) > 0.25).to(torch.float32)
