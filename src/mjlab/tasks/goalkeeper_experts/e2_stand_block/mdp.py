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
from mjlab.sensor import ContactSensor
from mjlab.tasks.goalkeeper_experts.launcher import (
  GoalkeeperBallLauncher,
  GoalkeeperBallLauncherCfg,
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
  return torch.rand(num, device=device) * (high - low) + low


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


def get_target_ball_cfg() -> EntityCfg:
  """Return the physical colliding RoboCup ball for E2."""
  return get_robocup_ball_cfg()


@dataclass(kw_only=True)
class StandBlockCommandCfg(CommandTermCfg):
  """Command term for E2 stand-block reset + fixed direct shot."""

  entity_name: str = "robot"
  ball_entity_name: str = "soccer_ball"
  ball_robot_contact_sensor_name: str | None = None

  # Motor-controller command vector dimension (from Stage-1 obs layout).
  command_dim: int = 46

  # Keeper spawn (world XY, before adding env origin).
  keeper_spawn_x_range: tuple[float, float]
  keeper_spawn_y_range: tuple[float, float]
  spawn_yaw_range: tuple[float, float]

  # Small noise around default standing pose.
  keeper_joint_pos_noise: float = 0.02
  keeper_joint_vel_noise: float = 0.08

  # Safe keeper area bounds: (x_min, x_max, y_min, y_max).
  keeper_area_bounds: tuple[float, float, float, float]
  hard_area_margin: float = 0.4

  # Reusable centralized launcher configuration.
  launcher_cfg: GoalkeeperBallLauncherCfg = field(
    default_factory=GoalkeeperBallLauncherCfg
  )

  # Goal-plane aperture used by goal detection.
  goal_toward_positive_x: bool = True
  goal_plane_x: float = 7.0
  goal_plane_y_center: float = 0.0
  goal_plane_y_half: float = 1.35
  goal_plane_z_min: float = 0.0
  goal_plane_z_max: float = 1.90

  # Visual cue shown for a few frames after goal detection.
  goal_termination_term_name: str = "goal_conceded"
  goal_cue_flash_steps: int = 18

  # Keep command fixed for full episode.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False

  @dataclass
  class VizCfg:
    goal_plane_color: tuple[float, float, float, float] = (0.15, 0.85, 0.95, 0.85)
    goal_cue_ok_color: tuple[float, float, float, float] = (0.15, 0.90, 0.20, 0.90)
    goal_cue_alert_color: tuple[float, float, float, float] = (0.95, 0.15, 0.15, 0.95)
    velocity_color: tuple[float, float, float, float] = (0.95, 0.85, 0.10, 0.85)
    plane_line_radius: float = 0.008
    velocity_arrow_scale: float = 0.22
    velocity_arrow_width: float = 0.014
    cue_radius: float = 0.075
    cue_z_offset: float = 0.18

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env):
    return StandBlockCommand(self, env)


class StandBlockCommand(CommandTerm):
  cfg: StandBlockCommandCfg

  def __init__(self, cfg: StandBlockCommandCfg, env):
    super().__init__(cfg, env)
    self._robot: Entity = env.scene[cfg.entity_name]
    self._ball: Entity = env.scene[cfg.ball_entity_name]
    self._launcher = GoalkeeperBallLauncher(cfg.launcher_cfg, env)

    self._command = torch.zeros(env.num_envs, cfg.command_dim, device=self.device)
    self._goal_flash_steps_left = torch.zeros(
      env.num_envs,
      device=self.device,
      dtype=torch.long,
    )

    self.metrics["outside_keeper_area"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["ball_speed_xy"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["goal_detected"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["launch_family_id"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["launch_t_goal_est_s"] = torch.zeros(env.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def launcher(self) -> GoalkeeperBallLauncher:
    return self._launcher

  @property
  def keeper_area_bounds(self) -> tuple[float, float, float, float]:
    return self.cfg.keeper_area_bounds

  @property
  def hard_keeper_area_bounds(self) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = self.cfg.keeper_area_bounds
    m = self.cfg.hard_area_margin
    return (x_min - m, x_max + m, y_min - m, y_max + m)

  def _update_metrics(self) -> None:
    trunk_xy_local = _world_to_env_local_xy(
      self._env,
      self._robot.data.root_link_pos_w[:, :2],
    )
    self.metrics["outside_keeper_area"] = _outside_area_violation(
      trunk_xy_local,
      self.cfg.keeper_area_bounds,
    )
    self.metrics["ball_speed_xy"] = torch.linalg.norm(
      self._ball.data.root_link_lin_vel_w[:, :2],
      dim=1,
    )
    self.metrics["goal_detected"] = self._goal_conceded_mask().float()
    self.metrics["launch_family_id"] = self._launcher.family_id.to(torch.float)
    self.metrics["launch_t_goal_est_s"] = self._launcher.t_goal_est_s

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    self._reset_robot_pose(env_ids)
    self._launcher.reset(env_ids)

    # Stage-1 decoder command input. For E2 we keep it deterministic zero.
    self._command[env_ids] = 0.0
    self._goal_flash_steps_left[env_ids] = 0

  def _update_command(self) -> None:
    time_s = self._env.episode_length_buf.to(torch.float) * self._env.step_dt
    self._launcher.step(time_s)

    if self.cfg.goal_cue_flash_steps <= 0:
      return

    self._goal_flash_steps_left = torch.clamp(
      self._goal_flash_steps_left - 1,
      min=0,
    )

    goal_now: torch.Tensor
    term_name = self.cfg.goal_termination_term_name
    if term_name != "":
      try:
        goal_now = self._env.termination_manager.get_term(term_name)
      except KeyError:
        goal_now = self._goal_conceded_mask()
    else:
      goal_now = self._goal_conceded_mask()

    self._goal_flash_steps_left[goal_now] = int(self.cfg.goal_cue_flash_steps)

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

    root_state = default_root_state[env_ids].clone()
    origins = self._env.scene.env_origins[env_ids]
    root_state[:, 0] = origins[:, 0] + spawn_x
    root_state[:, 1] = origins[:, 1] + spawn_y

    yaw_lo, yaw_hi = self.cfg.spawn_yaw_range
    if abs(yaw_hi - yaw_lo) <= 1.0e-9:
      yaw = torch.full((len(env_ids),), float(yaw_lo), device=self.device)
    else:
      yaw = _sample_uniform_range(yaw_lo, yaw_hi, len(env_ids), self.device)
    yaw_q = quat_from_euler_xyz(
      torch.zeros_like(yaw),
      torch.zeros_like(yaw),
      yaw,
    )
    root_state[:, 3:7] = quat_mul(root_state[:, 3:7], yaw_q)
    root_state[:, 7:13] = 0.0
    self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    joint_pos = default_joint_pos[env_ids].clone()
    joint_vel = default_joint_vel[env_ids].clone()

    pos_noise_mag = max(float(self.cfg.keeper_joint_pos_noise), 0.0)
    vel_noise_mag = max(float(self.cfg.keeper_joint_vel_noise), 0.0)
    if pos_noise_mag > 0.0:
      joint_pos += _sample_uniform_range(
        -pos_noise_mag,
        pos_noise_mag,
        joint_pos.numel(),
        self.device,
      ).view_as(joint_pos)
    if vel_noise_mag > 0.0:
      joint_vel += _sample_uniform_range(
        -vel_noise_mag,
        vel_noise_mag,
        joint_vel.numel(),
        self.device,
      ).view_as(joint_vel)

    self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self._robot.clear_state(env_ids=env_ids)

  def _goal_conceded_mask(self) -> torch.Tensor:
    ball_local = _world_to_env_local_xyz(self._env, self._ball.data.root_link_pos_w)

    x = ball_local[:, 0]
    y = ball_local[:, 1]
    z = ball_local[:, 2]

    if self.cfg.goal_toward_positive_x:
      crossed = x >= float(self.cfg.goal_plane_x)
    else:
      crossed = x <= float(self.cfg.goal_plane_x)

    inside_y = torch.abs(y - float(self.cfg.goal_plane_y_center)) <= float(
      self.cfg.goal_plane_y_half
    )
    inside_z = (z >= float(self.cfg.goal_plane_z_min)) & (
      z <= float(self.cfg.goal_plane_z_max)
    )
    return crossed & inside_y & inside_z

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
      radius=float(self.cfg.viz.plane_line_radius),
      color=self.cfg.viz.goal_plane_color,
      label="goal_plane_bottom",
    )
    visualizer.add_cylinder(
      p1.cpu().numpy(),
      p2.cpu().numpy(),
      radius=float(self.cfg.viz.plane_line_radius),
      color=self.cfg.viz.goal_plane_color,
      label="goal_plane_right",
    )
    visualizer.add_cylinder(
      p2.cpu().numpy(),
      p3.cpu().numpy(),
      radius=float(self.cfg.viz.plane_line_radius),
      color=self.cfg.viz.goal_plane_color,
      label="goal_plane_top",
    )
    visualizer.add_cylinder(
      p3.cpu().numpy(),
      p0.cpu().numpy(),
      radius=float(self.cfg.viz.plane_line_radius),
      color=self.cfg.viz.goal_plane_color,
      label="goal_plane_left",
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

    cue_pos = torch.tensor(
      [
        x,
        origin[1] + float(self.cfg.goal_plane_y_center),
        z1 + float(self.cfg.viz.cue_z_offset),
      ],
      device=self.device,
    )
    cue_color = (
      self.cfg.viz.goal_cue_alert_color
      if int(self._goal_flash_steps_left[batch].item()) > 0
      else self.cfg.viz.goal_cue_ok_color
    )
    visualizer.add_sphere(
      cue_pos.cpu().numpy(),
      radius=float(self.cfg.viz.cue_radius),
      color=cue_color,
      label="goal_cue",
    )


def _goal_conceded_mask_from_command(
  env,
  command_name: str,
) -> torch.Tensor:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  return cmd._goal_conceded_mask()


def _ball_robot_contact_mask(
  env,
  sensor_name: str | None,
) -> torch.Tensor:
  if sensor_name is None or sensor_name == "":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  sensor = cast(ContactSensor, env.scene[sensor_name])
  found = sensor.data.found
  if found is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  return torch.any(found > 0.0, dim=1)


def _first_ball_robot_contact_mask(
  env,
  sensor_name: str | None,
) -> torch.Tensor:
  if sensor_name is None or sensor_name == "":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  sensor = cast(ContactSensor, env.scene[sensor_name])
  try:
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    return torch.any(first_contact, dim=1)
  except RuntimeError:
    return _ball_robot_contact_mask(env, sensor_name)


# ---------------- Observations ----------------

def target_direction_xy(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  rel_xy = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  return _normalize_xy(rel_xy)


def ball_position_relative_xyz(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  return ball.data.root_link_pos_w - robot.data.root_link_pos_w


def ball_velocity_relative_xyz(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  return ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w


def time_to_goal_plane(
  env,
  command_name: str = "stand_block",
  max_time: float = 2.0,
  min_toward_speed: float = 0.05,
) -> torch.Tensor:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  ball_local = _world_to_env_local_xyz(env, ball.data.root_link_pos_w)

  x = ball_local[:, 0]
  vx = ball.data.root_link_lin_vel_w[:, 0]

  if cmd.cfg.goal_toward_positive_x:
    dx = float(cmd.cfg.goal_plane_x) - x
    toward = vx > float(min_toward_speed)
    t = dx / torch.clamp(vx, min=float(min_toward_speed))
  else:
    dx = x - float(cmd.cfg.goal_plane_x)
    toward = vx < -float(min_toward_speed)
    t = dx / torch.clamp(-vx, min=float(min_toward_speed))

  valid = toward & (dx >= 0.0)
  t = torch.where(valid, t, torch.full_like(t, float(max_time)))
  t = torch.clamp(t, min=0.0, max=float(max_time))
  return t.unsqueeze(1)


# ---------------- Rewards ----------------

def goal_conceded_indicator(
  env,
  command_name: str = "stand_block",
) -> torch.Tensor:
  return _goal_conceded_mask_from_command(env, command_name).float()


def save_success_reward(
  env,
  command_name: str = "stand_block",
  resolution_term_name: str | None = "contact_resolution_window",
) -> torch.Tensor:
  goal = _goal_conceded_mask_from_command(env, command_name)
  resolution_done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  if resolution_term_name is not None and resolution_term_name != "":
    try:
      resolution_done = env.termination_manager.get_term(resolution_term_name)
    except KeyError:
      resolution_done = torch.zeros(
        env.num_envs,
        device=env.device,
        dtype=torch.bool,
      )

  success = resolution_done & (~goal)
  return success.float()


def deflect_away_from_goal_reward(
  env,
  command_name: str = "stand_block",
  only_on_first_contact: bool = True,
  clip_speed: float = 4.0,
) -> torch.Tensor:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  if only_on_first_contact:
    contact = _first_ball_robot_contact_mask(env, cmd.cfg.ball_robot_contact_sensor_name)
  else:
    contact = _ball_robot_contact_mask(env, cmd.cfg.ball_robot_contact_sensor_name)

  vx = ball.data.root_link_lin_vel_w[:, 0]
  if cmd.cfg.goal_toward_positive_x:
    away_speed = torch.clamp(-vx, min=0.0, max=float(clip_speed))
  else:
    away_speed = torch.clamp(vx, min=0.0, max=float(clip_speed))

  return away_speed * contact.float()


def outside_keeper_area_penalty(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
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
  command_name: str = "stand_block",
) -> torch.Tensor:
  return _goal_conceded_mask_from_command(env, command_name)


def outside_keeper_area_hard(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  pos_xy_local = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])
  violation = _outside_area_violation(
    pos_xy_local,
    cmd.hard_keeper_area_bounds,
  )
  return violation > 0.0


def first_ball_contact_termination(
  env,
  command_name: str = "stand_block",
  enabled: bool = True,
) -> torch.Tensor:
  if not enabled:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  return _first_ball_robot_contact_mask(env, cmd.cfg.ball_robot_contact_sensor_name)


class ContactResolutionTermination:
  """Terminate once a fixed window elapsed after first keeper-ball contact.

  - Stores first contact time per env once (t_contact), ignores subsequent contacts.
  - Returns done when current_time - t_contact >= resolution_window_s.
  - Goal-conceded termination remains separate and immediate.
  """

  def __init__(self, cfg, env):
    del cfg
    self._t_contact = torch.full(
      (env.num_envs,),
      fill_value=-1.0,
      device=env.device,
      dtype=torch.float,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._t_contact[env_ids] = -1.0

  def __call__(
    self,
    env,
    command_name: str = "stand_block",
    resolution_window_s: float = 0.8,
  ) -> torch.Tensor:
    cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
    time_s = env.episode_length_buf.to(torch.float) * env.step_dt

    first_contact_now = _first_ball_robot_contact_mask(
      env,
      cmd.cfg.ball_robot_contact_sensor_name,
    )
    unset = self._t_contact < 0.0
    set_mask = unset & first_contact_now
    self._t_contact = torch.where(set_mask, time_s, self._t_contact)

    has_contact = self._t_contact >= 0.0
    elapsed = time_s - self._t_contact
    return has_contact & (elapsed >= float(resolution_window_s))


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
