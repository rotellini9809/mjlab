from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from mjlab.asset_zoo.robocup_assets.ball import get_robocup_ball_cfg
from mjlab.entity import Entity, EntityCfg
from mjlab.envs.mdp import *  # noqa: F401,F403
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.motor_controller_stage1.latent_action import (  # noqa: F401
  MotorLatentActionCfg,
  default_motor_obs_layout,
  motor_last_decoded_action,
)
from mjlab.sensor import ContactSensor
from mjlab.envs.mdp.rewards import (
  action_rate_l2 as _base_action_rate_l2,
  joint_pos_limits as _base_joint_pos_limits,
)
from mjlab.tasks.goalkeeper_experts.fov_helpers import (
  compute_fov_visibility,
  update_last_seen_ball_state,
)
from mjlab.tasks.goalkeeper_experts.launcher import (
  CROSS_FAMILY,
    LAUNCH_FAMILY_NAMES,
  LOB_CHIP_FAMILY,
  LONG_DRIVEN_FAMILY,
  GoalkeeperBallLauncher,
  GoalkeeperBallLauncherCfg,
)
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
)

if TYPE_CHECKING:
  import viser

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
E2V2_MEZZALUNA_STAGE1_GROUND_ONLY = "e2v2_mezzaluna_stage1_ground_only"
E2V2_MEZZALUNA_STAGE2_GROUND_LONG_DRIVEN = (
  "e2v2_mezzaluna_stage2_ground_long_driven"
)
E2V2_MEZZALUNA_STAGE3_LONG_DRIVEN_ONLY = (
  "e2v2_mezzaluna_stage3_long_driven_only"
)
E2V2_MEZZALUNA_LAUNCHER_CURRICULUM_PRESET_NAMES = (
  E2V2_MEZZALUNA_STAGE1_GROUND_ONLY,
  E2V2_MEZZALUNA_STAGE2_GROUND_LONG_DRIVEN,
  E2V2_MEZZALUNA_STAGE3_LONG_DRIVEN_ONLY,
)


def get_e2v2_mezzaluna_launcher_curriculum_stage_index(
  preset_name: str,
) -> int | None:
  try:
    return E2V2_MEZZALUNA_LAUNCHER_CURRICULUM_PRESET_NAMES.index(preset_name) + 1
  except ValueError:
    return None


def get_e2v2_mezzaluna_launcher_curriculum_preset_name(stage_index: int) -> str:
  if not (1 <= int(stage_index) <= len(E2V2_MEZZALUNA_LAUNCHER_CURRICULUM_PRESET_NAMES)):
    raise ValueError(
      "E2V2 mezzaluna curriculum stage must be within "
      f"[1, {len(E2V2_MEZZALUNA_LAUNCHER_CURRICULUM_PRESET_NAMES)}], "
      f"got {stage_index}."
    )
  return E2V2_MEZZALUNA_LAUNCHER_CURRICULUM_PRESET_NAMES[int(stage_index) - 1]


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


def _world_to_env_local_xy(env, pos_w_xy: torch.Tensor) -> torch.Tensor:
  return pos_w_xy - env.scene.env_origins[:, :2]


def _world_to_env_local_xyz(env, pos_w_xyz: torch.Tensor) -> torch.Tensor:
  return pos_w_xyz - env.scene.env_origins


def _goal_line_center_world_xy(
  env,
  command_name: str,
) -> torch.Tensor:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  return env.scene.env_origins[:, :2] + torch.tensor(
    [cmd.cfg.goal_plane_x, cmd.cfg.goal_plane_y_center],
    device=env.device,
    dtype=torch.float32,
  )


def _mezzaluna_point_world_xy_from_ball_xy(
  env,
  ball_xy_w: torch.Tensor,
  center_x: float,
  center_y: float,
  apex_x: float,
  half_width_y: float,
) -> torch.Tensor:
  center_xy = env.scene.env_origins[:, :2] + torch.tensor(
    [center_x, center_y],
    device=env.device,
    dtype=torch.float32,
  )
  dir_xy = ball_xy_w - center_xy

  a = max(float(center_x) - float(apex_x), 1.0e-6)
  b = max(float(half_width_y), 1.0e-6)

  dx = dir_xy[:, 0]
  dy = dir_xy[:, 1]
  proj_dx = torch.where(dx <= 0.0, dx, -dx)
  proj_dy = dy
  denom = torch.sqrt(
    torch.square(proj_dx / a) + torch.square(proj_dy / b)
  ).clamp_min(1.0e-6)
  proj_dir_xy = torch.stack((proj_dx, proj_dy), dim=1)
  point_xy = center_xy + proj_dir_xy / denom.unsqueeze(1)

  use_apex = torch.linalg.norm(dir_xy, dim=1) <= 1.0e-6
  if use_apex.any():
    point_xy[use_apex, 0] = float(apex_x) + env.scene.env_origins[use_apex, 0]
    point_xy[use_apex, 1] = float(center_y) + env.scene.env_origins[use_apex, 1]

  return point_xy


def _resolve_body_index_pair_cached(
  env,
  robot: Entity,
  body_name_a: str,
  body_name_b: str,
) -> tuple[int, int]:
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_e2_body_index_pair_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  key = (id(robot), body_name_a, body_name_b)
  if key not in cache:
    ids, names = robot.find_bodies((body_name_a, body_name_b), preserve_order=True)
    if len(ids) != 2:
      raise ValueError(
        "Could not resolve exactly two bodies. "
        f"Got names={names} for patterns=({body_name_a}, {body_name_b})."
      )
    cache[key] = (int(ids[0]), int(ids[1]))

  return cache[key]


def _resolve_single_body_index_cached(
  env,
  robot: Entity,
  body_name: str,
) -> int:
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_e2_single_body_index_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  key = (id(robot), body_name)
  if key not in cache:
    ids, names = robot.find_bodies(body_name)
    if len(ids) != 1:
      raise ValueError(
        "Could not resolve exactly one body. "
        f"Got names={names} for pattern={body_name}."
      )
    cache[key] = int(ids[0])

  return cache[key]


def _get_log_dict(env) -> dict[str, torch.Tensor] | None:
  extras = getattr(env, "extras", None)
  if extras is None:
    return None
  log = extras.get("log")
  if log is None:
    log = {}
    extras["log"] = log
  return log


def _get_bool_state_buffer(
  env,
  key: str,
) -> torch.Tensor:
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_e2_bool_state_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  buf = cache.get(key)
  if buf is None or buf.shape != (env.num_envs,) or buf.device != env.device:
    buf = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    cache[key] = buf
  return buf


def _get_float_state_buffer(
  env,
  key: str,
  *,
  fill_value: float = 0.0,
) -> torch.Tensor:
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_e2_float_state_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  buf = cache.get(key)
  if buf is None or buf.shape != (env.num_envs,) or buf.device != env.device:
    buf = torch.full(
      (env.num_envs,),
      fill_value=float(fill_value),
      device=env.device,
      dtype=torch.float32,
    )
    cache[key] = buf
  return buf


def _get_long_state_buffer(
  env,
  key: str,
  *,
  fill_value: int = 0,
) -> torch.Tensor:
  env_obj = getattr(env, "unwrapped", env)
  cache_name = "_e2_long_state_cache"
  cache = getattr(env_obj, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env_obj, cache_name, cache)

  buf = cache.get(key)
  if buf is None or buf.shape != (env.num_envs,) or buf.device != env.device:
    buf = torch.full(
      (env.num_envs,),
      fill_value=int(fill_value),
      device=env.device,
      dtype=torch.long,
    )
    cache[key] = buf
  return buf


def _reward_active_mask(
  env,
  command_name: str = "stand_block",
) -> torch.Tensor:
  del command_name
  return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)


def _apply_reward_active_mask(
  reward: torch.Tensor,
  env,
  command_name: str = "stand_block",
) -> torch.Tensor:
  return reward * _reward_active_mask(env, command_name).to(reward.dtype)


def _posture_score_components(
  projected_gravity_b: torch.Tensor,
  roll_band: float,
  roll_sigma: float,
  pitch_target: float,
  pitch_band: float,
  pitch_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  sagittal = projected_gravity_b[:, 0]
  lateral = projected_gravity_b[:, 1]
  vertical = projected_gravity_b[:, 2]

  roll_error = torch.relu(torch.abs(lateral) - float(roll_band))
  roll_score = torch.exp(
    -torch.square(roll_error) / max(float(roll_sigma) * float(roll_sigma), 1.0e-6)
  )

  pitch_error = torch.relu(
    torch.abs(sagittal - float(pitch_target)) - float(pitch_band)
  )
  pitch_score = torch.exp(
    -torch.square(pitch_error) / max(float(pitch_sigma) * float(pitch_sigma), 1.0e-6)
  )

  upright_sign_score = torch.clamp(-vertical, min=0.0, max=1.0)
  posture_score = roll_score * pitch_score * upright_sign_score
  return posture_score, roll_score, pitch_score, lateral, sagittal


def _standing_gate(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  h_low: float = 0.36,
  h_good: float = 0.56,
  roll_band: float = 0.1,
  roll_sigma: float = 0.12,
  pitch_target: float = 0.25,
  pitch_band: float = 0.20,
  pitch_sigma: float = 0.30,
) -> torch.Tensor:
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
  gate = stand_score

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e2_stand_score_mean"] = torch.mean(stand_score)
    log["Metrics/e2_stand_gate_mean"] = torch.mean(gate)
    log["Metrics/e2_height_score_mean"] = torch.mean(height_score)
    log["Metrics/e2_base_height_mean"] = torch.mean(base_height)

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
  mezzaluna_center_x: float = 6.8
  mezzaluna_center_y: float = 0.0
  mezzaluna_apex_x: float = 6.15
  mezzaluna_half_width_y: float = 1.65
  mezzaluna_spawn_xy_jitter_range: tuple[float, float] = (-0.10, 0.10)
  mezzaluna_spawn_radial_jitter_range: tuple[float, float] = (-0.04, 0.04)

  # Small noise around default standing pose.
  keeper_joint_pos_noise: float = 0.02
  keeper_joint_vel_noise: float = 0.08

  # Reusable centralized launcher configuration.
  launcher_cfg: GoalkeeperBallLauncherCfg = field(
    default_factory=GoalkeeperBallLauncherCfg
  )
  curriculum_stage: int | None = None

  # Goal-plane aperture used by goal detection.
  goal_toward_positive_x: bool = True
  goal_plane_x: float = 7.0
  goal_plane_y_center: float = 0.0
  goal_plane_y_half: float = 1.35
  goal_plane_z_min: float = 0.0
  goal_plane_z_max: float = 1.90
  danger_area_bounds: tuple[float, float, float, float] = (6.0, 7.6, -2.0, 2.0)
  keeper_area_bounds: tuple[float, float, float, float] = (6.0, 7.6, -2.0, 2.0)

  # Keep command fixed for full episode.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  fov_active: bool = True
  ball_fov_half_angle_deg: float = 90.0
  debug_vis: bool = False

  @dataclass
  class VizCfg:
    goal_plane_color: tuple[float, float, float, float] = (0.15, 0.85, 0.95, 0.85)
    velocity_color: tuple[float, float, float, float] = (0.95, 0.85, 0.10, 0.85)
    plane_line_radius: float = 0.008
    velocity_arrow_scale: float = 0.22
    velocity_arrow_width: float = 0.014

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

    self.metrics["torso_roll"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["torso_pitch"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["torso_height"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["ball_speed_xy"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["goal_detected"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["launch_family_id"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["launch_t_goal_est_s"] = torch.zeros(env.num_envs, device=self.device)
    self.metrics["curriculum_stage_num"] = torch.zeros(
      env.num_envs, device=self.device
    )
    self._status_markdown = None
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def launcher(self) -> GoalkeeperBallLauncher:
    return self._launcher

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    del name
    self._gui_get_env_idx = get_env_idx
    with server.gui.add_folder("Launcher"):
      self._status_markdown = server.gui.add_markdown("")
    self._update_status_markdown()

  def compute(self, dt: float) -> None:
    super().compute(dt)
    self._update_status_markdown()

  @property
  def launcher_preset_name(self) -> str:
    return self.cfg.launcher_cfg.active_preset_name or "custom"

  @property
  def launcher_curriculum_stage(self) -> int | None:
    if self.cfg.curriculum_stage is not None:
      return int(self.cfg.curriculum_stage)
    return get_e2v2_mezzaluna_launcher_curriculum_stage_index(
      self.launcher_preset_name
    )

  def _update_status_markdown(self) -> None:
    if self._status_markdown is None or self._gui_get_env_idx is None:
      return

    env_idx = int(self._gui_get_env_idx())
    if env_idx < 0 or env_idx >= self.num_envs:
      self._status_markdown.content = (
        "**Stage:** n/a\n\n"
        "**Preset:** n/a\n\n"
        "**Launch family:** n/a\n\n"
        "**T goal est.:** n/a\n\n"
        "**Launched:** n/a\n\n"
        "**Deflection:** n/a"
      )
      return

    family_id = int(self._launcher.family_id[env_idx].item())
    if 0 <= family_id < len(LAUNCH_FAMILY_NAMES):
      family_name = LAUNCH_FAMILY_NAMES[family_id].replace("_", " ")
    else:
      family_name = "n/a"

    t_goal_est_s = float(self._launcher.t_goal_est_s[env_idx].item())
    launched = bool(self._launcher.has_launched[env_idx].item())
    has_deflection = bool(self._launcher.has_deflection[env_idx].item())
    has_deflected = bool(self._launcher.has_deflected[env_idx].item())
    torso_roll_deg = torch.rad2deg(self.metrics["torso_roll"][env_idx]).item()
    torso_pitch_deg = torch.rad2deg(self.metrics["torso_pitch"][env_idx]).item()
    torso_height = float(self.metrics["torso_height"][env_idx].item())
    deflection_status = (
      "scheduled"
      if has_deflection and not has_deflected
      else "applied"
      if has_deflected
      else "none"
    )
    stage_index = self.launcher_curriculum_stage
    stage_text = str(stage_index) if stage_index is not None else "baseline/custom"

    self._status_markdown.content = (
      f"**Stage:** {stage_text}\n\n"
      f"**Preset:** {self.launcher_preset_name}\n\n"
      f"**Launch family:** {family_name}\n\n"
      f"**Torso roll:** {torso_roll_deg:.1f} deg\n\n"
      f"**Torso pitch:** {torso_pitch_deg:.1f} deg\n\n"
      f"**Torso height:** {torso_height:.3f} m\n\n"
      f"**T goal est.:** {t_goal_est_s:.3f} s\n\n"
      f"**Launched:** {'yes' if launched else 'no'}\n\n"
      f"**Deflection:** {deflection_status}"
    )

  def _update_metrics(self) -> None:
    root_quat = self._robot.data.root_link_quat_w
    self.metrics["torso_roll"] = _roll_from_quat_wxyz(root_quat)
    self.metrics["torso_pitch"] = _pitch_from_quat_wxyz(root_quat)
    self.metrics["torso_height"] = self._robot.data.root_link_pos_w[:, 2]
    self.metrics["ball_speed_xy"] = torch.linalg.norm(
      self._ball.data.root_link_lin_vel_w[:, :2],
      dim=1,
    )
    self.metrics["goal_detected"] = self._goal_conceded_mask().float()
    self.metrics["launch_family_id"] = self._launcher.family_id.to(torch.float)
    self.metrics["launch_t_goal_est_s"] = self._launcher.t_goal_est_s
    stage_index = self.launcher_curriculum_stage
    stage_value = 0.0 if stage_index is None else float(stage_index)
    self.metrics["curriculum_stage_num"].fill_(stage_value)
    log = _get_log_dict(self._env)
    if log is not None:
      log["Metrics/e2_curriculum_stage_num"] = torch.tensor(
        stage_value, device=self.device, dtype=torch.float32
      )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    self._launcher.reset(env_ids)
    self._reset_robot_pose(env_ids)

    # Stage-1 decoder command input. For E2 we keep it deterministic zero.
    self._command[env_ids] = 0.0

  def _update_command(self) -> None:
    time_s = self._env.episode_length_buf.to(torch.float) * self._env.step_dt
    self._launcher.step(time_s)

  def _reset_robot_pose(self, env_ids: torch.Tensor) -> None:
    from mjlab.tasks.goalkeeper_experts.e2v2_mezzaluna.config.t1_23dof.env_cfgs import (
      KEEPER_SPAWN_Z,
      READY_JOINT_POS,
      READY_ROOT_QUAT,
      READY_ROOT_YAW,
    )

    default_root_state = self._robot.data.default_root_state
    default_joint_vel = self._robot.data.default_joint_vel

    assert default_root_state is not None
    assert default_joint_vel is not None

    root_state = default_root_state[env_ids].clone()
    origins = self._env.scene.env_origins[env_ids]
    ball_xy = self._launcher.spawn_pos_w[env_ids, :2]
    mezzaluna_point_xy = _mezzaluna_point_world_xy_from_ball_xy(
      self._env,
      self._launcher.spawn_pos_w[:, :2],
      center_x=float(self.cfg.mezzaluna_center_x),
      center_y=float(self.cfg.mezzaluna_center_y),
      apex_x=float(self.cfg.mezzaluna_apex_x),
      half_width_y=float(self.cfg.mezzaluna_half_width_y),
    )[env_ids]
    center_xy = self._env.scene.env_origins[env_ids, :2] + torch.tensor(
      [self.cfg.mezzaluna_center_x, self.cfg.mezzaluna_center_y],
      device=self.device,
      dtype=torch.float32,
    )
    radial_dir_xy = _normalize_xy(mezzaluna_point_xy - center_xy)
    jitter_lo, jitter_hi = self.cfg.mezzaluna_spawn_xy_jitter_range
    if abs(jitter_hi - jitter_lo) <= 1.0e-9:
      jitter_xy = torch.full(
        (len(env_ids), 2),
        float(jitter_lo),
        device=self.device,
      )
    else:
      jitter_xy = _sample_uniform_range(
        jitter_lo,
        jitter_hi,
        len(env_ids) * 2,
        self.device,
      ).view(len(env_ids), 2)
    radial_lo, radial_hi = self.cfg.mezzaluna_spawn_radial_jitter_range
    if abs(radial_hi - radial_lo) <= 1.0e-9:
      radial_jitter = torch.full((len(env_ids),), float(radial_lo), device=self.device)
    else:
      radial_jitter = _sample_uniform_range(
        radial_lo,
        radial_hi,
        len(env_ids),
        self.device,
      )
    spawn_xy = (
      mezzaluna_point_xy
      + jitter_xy
      + radial_jitter.unsqueeze(1) * radial_dir_xy
    )
    root_state[:, 0:2] = spawn_xy
    root_state[:, 2] = origins[:, 2] + float(KEEPER_SPAWN_Z)

    # Face the sampled ball spawn point in XY, with optional additive yaw offset.
    robot_xy = root_state[:, :2]
    to_ball_xy = ball_xy - robot_xy
    yaw_face_ball = torch.atan2(to_ball_xy[:, 1], to_ball_xy[:, 0])

    yaw_lo, yaw_hi = self.cfg.spawn_yaw_range
    if abs(yaw_hi - yaw_lo) <= 1.0e-9:
      yaw_offset = torch.full((len(env_ids),), float(yaw_lo), device=self.device)
    else:
      yaw_offset = _sample_uniform_range(yaw_lo, yaw_hi, len(env_ids), self.device)
    yaw = yaw_face_ball + yaw_offset

    ready_quat = torch.tensor(READY_ROOT_QUAT, dtype=torch.float32, device=self.device)
    yaw_delta = yaw - float(READY_ROOT_YAW)

    yaw_q = quat_from_euler_xyz(
      torch.zeros_like(yaw),
      torch.zeros_like(yaw),
      yaw_delta,
    )
    root_state[:, 3:7] = quat_mul(
      yaw_q,
      ready_quat.unsqueeze(0).expand(len(env_ids), -1),
    )
    root_state[:, 7:13] = 0.0
    self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    joint_pos = torch.tensor(READY_JOINT_POS, dtype=torch.float32, device=self.device)
    joint_pos = joint_pos.unsqueeze(0).expand(len(env_ids), -1).clone()
    joint_vel = torch.zeros_like(joint_pos)

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
    self._robot.reset(env_ids=env_ids)

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
    y0 = (
      origin[1]
      + float(self.cfg.goal_plane_y_center)
      - float(self.cfg.goal_plane_y_half)
    )
    y1 = (
      origin[1]
      + float(self.cfg.goal_plane_y_center)
      + float(self.cfg.goal_plane_y_half)
    )
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


def _goal_conceded_mask_from_command(
  env,
  command_name: str,
) -> torch.Tensor:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  return cmd._goal_conceded_mask()


def _ball_in_danger_area_mask_from_command(
  env,
  command_name: str,
) -> torch.Tensor:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  ball_local = _world_to_env_local_xyz(env, ball.data.root_link_pos_w)
  x = ball_local[:, 0]
  y = ball_local[:, 1]

  x_min, x_max, y_min, y_max = cmd.cfg.danger_area_bounds
  return (
    (x >= float(x_min))
    & (x <= float(x_max))
    & (y >= float(y_min))
    & (y <= float(y_max))
  )


def _contact_time_buffer_from_termination(
  env,
  resolution_term_name: str,
) -> torch.Tensor:
  unset = torch.full(
    (env.num_envs,),
    fill_value=-1.0,
    device=env.device,
    dtype=torch.float,
  )
  try:
    term_cfg = env.termination_manager.get_term_cfg(resolution_term_name)
  except ValueError:
    return unset

  t_contact = getattr(term_cfg.func, "_t_contact", None)
  if not isinstance(t_contact, torch.Tensor):
    return unset
  return t_contact


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


def _contact_rising_edge_event(
  env,
  sensor_name: str,
  *,
  state_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  contact_now = _ball_robot_contact_mask(env, sensor_name)
  prev_contact = _get_bool_state_buffer(env, key=state_key)

  is_first = env.episode_length_buf <= 1
  prev_contact[is_first] = False

  new_contact = contact_now & (~prev_contact)
  prev_contact.copy_(contact_now)
  return contact_now, new_contact


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


def _high_throw_mask_from_command(
  env,
  command_name: str,
  high_throw_target_z_min: float = 0.55,
) -> torch.Tensor:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  family_id = cmd.launcher.family_id
  valid_family = family_id >= 0
  family_mask = (
    (family_id == LOB_CHIP_FAMILY)
    | (family_id == CROSS_FAMILY)
    | (family_id == LONG_DRIVEN_FAMILY)
  )
  fallback_height_mask = cmd.launcher.target_pos_w[:, 2] >= float(high_throw_target_z_min)
  return torch.where(valid_family, family_mask, fallback_height_mask)


def _toward_goal_speed_from_command(
  env,
  command_name: str,
) -> torch.Tensor:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  vx = ball.data.root_link_lin_vel_w[:, 0]
  if cmd.cfg.goal_toward_positive_x:
    toward_speed = vx
  else:
    toward_speed = -vx
  return torch.nan_to_num(toward_speed, nan=0.0, posinf=0.0, neginf=0.0)


def _project_ball_crossing_to_goal_plane(
  env,
  command_name: str,
  *,
  min_forward_speed: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  ball_local = _world_to_env_local_xyz(env, ball.data.root_link_pos_w)
  vel_w = ball.data.root_link_lin_vel_w

  x = ball_local[:, 0]
  y = ball_local[:, 1]
  z = ball_local[:, 2]
  vx = vel_w[:, 0]
  vy = vel_w[:, 1]
  vz = vel_w[:, 2]

  min_speed = max(float(min_forward_speed), 1.0e-6)
  if cmd.cfg.goal_toward_positive_x:
    dx = float(cmd.cfg.goal_plane_x) - x
    toward_speed = vx
  else:
    dx = x - float(cmd.cfg.goal_plane_x)
    toward_speed = -vx

  valid = (toward_speed >= min_speed) & (dx >= 0.0)
  t_cross = torch.zeros_like(dx)
  t_cross = torch.where(valid, dx / toward_speed.clamp_min(min_speed), t_cross)

  gravity = float(cmd.launcher.cfg.gravity)
  y_proj = y + vy * t_cross
  z_proj = z + vz * t_cross - 0.5 * gravity * torch.square(t_cross)

  finite = (
    torch.isfinite(t_cross)
    & torch.isfinite(y_proj)
    & torch.isfinite(z_proj)
  )
  valid = valid & finite & (t_cross >= 0.0)

  return valid, t_cross, y_proj, z_proj


def _goal_projection_score_from_command(
  env,
  command_name: str,
  *,
  min_forward_speed: float,
  projection_margin_y: float,
  projection_margin_z: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  valid, _t_cross, y_proj, z_proj = _project_ball_crossing_to_goal_plane(
    env,
    command_name,
    min_forward_speed=min_forward_speed,
  )

  y_err = torch.relu(
    torch.abs(y_proj - float(cmd.cfg.goal_plane_y_center)) - float(cmd.cfg.goal_plane_y_half)
  )
  z_err_low = torch.relu(float(cmd.cfg.goal_plane_z_min) - z_proj)
  z_err_high = torch.relu(z_proj - float(cmd.cfg.goal_plane_z_max))
  z_err = torch.maximum(z_err_low, z_err_high)

  if float(projection_margin_y) > 0.0:
    y_score = 1.0 - torch.clamp(y_err / float(projection_margin_y), min=0.0, max=1.0)
    y_near = y_err <= float(projection_margin_y)
  else:
    y_score = (y_err <= 0.0).to(torch.float32)
    y_near = y_err <= 0.0

  if float(projection_margin_z) > 0.0:
    z_score = 1.0 - torch.clamp(z_err / float(projection_margin_z), min=0.0, max=1.0)
    z_near = z_err <= float(projection_margin_z)
  else:
    z_score = (z_err <= 0.0).to(torch.float32)
    z_near = z_err <= 0.0

  near_mask = valid & y_near & z_near
  score = torch.where(near_mask, y_score * z_score, torch.zeros_like(y_score))
  score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
  return score.clamp(0.0, 1.0), near_mask


def _danger_score_from_command(
  env,
  command_name: str,
  *,
  v_ref: float,
  min_forward_speed: float,
  projection_margin_y: float,
  projection_margin_z: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  toward_speed = _toward_goal_speed_from_command(env, command_name)
  forward_speed_score = torch.clamp(
    toward_speed / max(float(v_ref), 1.0e-6),
    min=0.0,
    max=1.0,
  )
  goal_projection_score, projection_near_mask = _goal_projection_score_from_command(
    env,
    command_name,
    min_forward_speed=min_forward_speed,
    projection_margin_y=projection_margin_y,
    projection_margin_z=projection_margin_z,
  )
  danger_area_mask = _ball_in_danger_area_mask_from_command(env, command_name)
  danger_area_score = danger_area_mask.to(torch.float32)

  danger = (
    0.5 * forward_speed_score
    + 0.3 * goal_projection_score
    + 0.2 * danger_area_score
  )
  danger = torch.nan_to_num(danger, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

  pre_threat_mask = (
    (toward_speed >= float(min_forward_speed))
    & projection_near_mask
    & danger_area_mask
  )
  return danger, pre_threat_mask


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


def visible_target_direction_xy(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  visible = _ball_visibility_mask(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  target_dir_xy = target_direction_xy(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  return target_dir_xy * visible.unsqueeze(1).to(target_dir_xy.dtype)


def _ball_visibility_mask(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  eps: float = 1.0e-6,
) -> torch.Tensor:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  robot: Entity = env.scene[asset_cfg.name]
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  rel_xy = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  forward_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(
    env.num_envs, -1
  )
  forward_w = quat_apply(robot.data.root_link_quat_w, forward_local)
  forward_xy = _normalize_xy(forward_w[:, :2])
  return compute_fov_visibility(
    rel_xy,
    forward_xy,
    fov_active=bool(cmd.cfg.fov_active),
    half_angle_deg=float(cmd.cfg.ball_fov_half_angle_deg),
    eps=float(eps),
  )


def _ball_visibility_obs_context(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  rel_pos_xyz = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  rel_vel_xyz = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
  visible = _ball_visibility_mask(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )

  last_seen_pos_xy, last_seen_vel_xy, last_seen_secs = update_last_seen_ball_state(
    env,
    visible=visible,
    rel_pos_xyz=rel_pos_xyz,
    rel_vel_xyz=rel_vel_xyz,
    key_prefix=f"e2_ball_visibility::{command_name}",
    get_float_state_buffer=_get_float_state_buffer,
    get_bool_state_buffer=_get_bool_state_buffer,
  )

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e2_ball_visible_mean"] = torch.mean(visible.to(torch.float32))
    log["Metrics/e2_ball_last_seen_secs_mean"] = torch.mean(last_seen_secs)

  return visible, rel_pos_xyz, rel_vel_xyz, last_seen_pos_xy, last_seen_vel_xy, last_seen_secs


def ball_visible(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  visible, _, _, _, _, _ = _ball_visibility_obs_context(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  return visible.to(torch.float32).unsqueeze(1)


def visible_ball_position_relative_xyz(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  visible, rel_pos_xyz, _, _, _, _ = _ball_visibility_obs_context(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  return rel_pos_xyz * visible.unsqueeze(1).to(rel_pos_xyz.dtype)


def visible_ball_velocity_relative_xyz(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  visible, _, rel_vel_xyz, _, _, _ = _ball_visibility_obs_context(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  return rel_vel_xyz * visible.unsqueeze(1).to(rel_vel_xyz.dtype)


def last_seen_ball_position_relative_xy(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  _, _, _, last_seen_pos_xy, _, _ = _ball_visibility_obs_context(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  return last_seen_pos_xy


def last_seen_ball_velocity_relative_xy(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  _, _, _, _, last_seen_vel_xy, _ = _ball_visibility_obs_context(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  return last_seen_vel_xy


def last_seen_ball_secs(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  _, _, _, _, _, last_seen_secs = _ball_visibility_obs_context(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  return last_seen_secs.unsqueeze(1)


def ball_position_relative_xyz(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  return ball.data.root_link_pos_w - robot.data.root_link_pos_w


def robot_position_relative_goal_line_xy(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.root_link_pos_w[:, :2] - _goal_line_center_world_xy(env, command_name)


def ball_velocity_relative_xyz(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  return ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w


def visible_time_to_goal_plane(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  max_time: float = 2.0,
  min_toward_speed: float = 0.05,
) -> torch.Tensor:
  visible = _ball_visibility_mask(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  t_goal = time_to_goal_plane(
    env,
    command_name=command_name,
    max_time=max_time,
    min_toward_speed=min_toward_speed,
  )
  return t_goal * visible.unsqueeze(1).to(t_goal.dtype)


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


def desired_point_relative_xy(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  target_w_xy = _mezzaluna_point_world_xy_from_ball_xy(
    env,
    ball.data.root_link_pos_w[:, :2],
    center_x=float(cmd.cfg.mezzaluna_center_x),
    center_y=float(cmd.cfg.mezzaluna_center_y),
    apex_x=float(cmd.cfg.mezzaluna_apex_x),
    half_width_y=float(cmd.cfg.mezzaluna_half_width_y),
  )
  return target_w_xy - robot.data.root_link_pos_w[:, :2]


def visible_desired_point_relative_xy(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  visible = _ball_visibility_mask(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  desired_point_xy = desired_point_relative_xy(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  return desired_point_xy * visible.unsqueeze(1).to(desired_point_xy.dtype)


# ---------------- Rewards ----------------


def goal_conceded_indicator(
  env,
  command_name: str = "stand_block",
) -> torch.Tensor:
  return _goal_conceded_mask_from_command(env, command_name).float()


def action_rate_l2(
  env,
  command_name: str = "stand_block",
) -> torch.Tensor:
  return _apply_reward_active_mask(_base_action_rate_l2(env), env, command_name)


def joint_pos_limits(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str = "stand_block",
) -> torch.Tensor:
  return _apply_reward_active_mask(
    _base_joint_pos_limits(env, asset_cfg=asset_cfg),
    env,
    command_name,
  )


class DangerReductionOnFirstContactReward:
  def __init__(self, cfg, env):
    del cfg
    self._env = env
    self._key_pre = "e2_danger_pre_contact::{}"
    self._key_paid = "e2_danger_bridge_paid::{}"
    self.reset(env_ids=slice(None))

  def reset(
    self,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    env = self._env
    pre = _get_float_state_buffer(
      env,
      self._key_pre.format("stand_block"),
      fill_value=float("nan"),
    )
    paid = _get_bool_state_buffer(env, self._key_paid.format("stand_block"))
    pre[env_ids] = float("nan")
    paid[env_ids] = False

  def __call__(
    self,
    env,
    command_name: str = "stand_block",
    resolution_term_name: str = "contact_resolution_window",
    v_ref: float = 6.0,
    min_forward_speed: float = 1.0,
    projection_margin_y: float = 0.35,
    projection_margin_z: float = 0.25,
    post_contact_delay_steps: int = 2,
  ) -> torch.Tensor:
    key_pre = self._key_pre.format(command_name)
    key_paid = self._key_paid.format(command_name)

    pre_buf = _get_float_state_buffer(env, key_pre, fill_value=float("nan"))
    paid_buf = _get_bool_state_buffer(env, key_paid)

    is_first = env.episode_length_buf <= 1
    pre_buf[is_first] = float("nan")
    paid_buf[is_first] = False

    danger_now, pre_threat_mask = _danger_score_from_command(
      env,
      command_name,
      v_ref=v_ref,
      min_forward_speed=min_forward_speed,
      projection_margin_y=projection_margin_y,
      projection_margin_z=projection_margin_z,
    )

    t_contact = _contact_time_buffer_from_termination(env, resolution_term_name)
    has_contact = t_contact >= 0.0
    not_armed = (~has_contact) & (~paid_buf)
    updated_pre = torch.where(
      not_armed & pre_threat_mask,
      danger_now,
      pre_buf,
    )

    time_s = env.episode_length_buf.to(torch.float32) * env.step_dt
    delay_steps = max(int(post_contact_delay_steps), 0)
    delay_s = float(delay_steps) * float(env.step_dt)
    due = (
      has_contact
      & (~paid_buf)
      & ((time_s - t_contact) >= delay_s)
    )
    active_mask = due & torch.isfinite(updated_pre)

    danger_post = torch.where(due, danger_now, torch.zeros_like(danger_now))
    danger_delta = torch.where(
      torch.isfinite(updated_pre),
      torch.clamp(updated_pre - danger_post, min=0.0, max=1.0),
      torch.zeros_like(danger_now),
    )
    raw = active_mask.to(torch.float32) * danger_delta
    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

    updated_paid = paid_buf | due

    pre_buf.copy_(updated_pre)
    paid_buf.copy_(updated_paid)

    log = _get_log_dict(env)
    if log is not None:
      safe_pre = torch.where(
        torch.isfinite(updated_pre),
        updated_pre,
        torch.zeros_like(updated_pre),
      )
      log["Metrics/e2_danger_pre_mean"] = torch.mean(safe_pre)
      log["Metrics/e2_danger_post_mean"] = torch.mean(danger_post)
      log["Metrics/e2_danger_delta_mean"] = torch.mean(
        torch.where(active_mask, danger_delta, torch.zeros_like(danger_delta))
      )
      log["Metrics/e2_danger_bridge_raw_mean"] = torch.mean(raw)

    return raw


class ClearanceQualityReward:
  def __init__(self, cfg, env):
    del cfg
    self._exit_latch = _ConfirmedDangerExitLatchState(env)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._exit_latch.reset(env_ids)

  def __call__(
    self,
    env,
    command_name: str = "stand_block",
    resolution_term_name: str = "contact_resolution_window",
    t_ref: float = 1.5,
    t_clear_clip: float = 0.5,
    clip_away_speed: float = 4.0,
    outside_steps_required: int = 2,
  ) -> torch.Tensor:
    in_danger_now, t_contact, exit_event, _post_exit_active = self._exit_latch.update(
      env,
      command_name=command_name,
      resolution_term_name=resolution_term_name,
      outside_steps_required=outside_steps_required,
    )
    cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]
    vx = ball.data.root_link_lin_vel_w[:, 0]
    if cmd.cfg.goal_toward_positive_x:
      v_away = torch.clamp(-vx, min=0.0, max=float(clip_away_speed))
    else:
      v_away = torch.clamp(vx, min=0.0, max=float(clip_away_speed))

    time_s = env.episode_length_buf.to(torch.float) * env.step_dt
    t_clear = torch.clamp(time_s - t_contact, min=0.0)
    t_clear_reward = torch.clamp(t_clear, max=float(t_clear_clip))
    time_factor = torch.clamp(
      1.0 - t_clear_reward / max(float(t_ref), 1.0e-6),
      min=0.0,
      max=1.0,
    )
    vel_factor = torch.clamp(
      v_away / max(float(clip_away_speed), 1.0e-6),
      min=0.0,
      max=1.0,
    )

    raw = exit_event.float() * time_factor * vel_factor

    log = _get_log_dict(env)
    if log is not None:
      exit_time_mean = torch.zeros((), device=env.device, dtype=torch.float)
      if torch.any(exit_event):
        exit_time_mean = torch.mean(t_clear[exit_event])
      log["Metrics/e2_ball_in_danger_mean"] = torch.mean(in_danger_now.float())
      log["Metrics/e2_clearance_exit_event_mean"] = torch.mean(exit_event.float())
      log["Metrics/e2_clearance_exit_time_mean"] = exit_time_mean
      log["Metrics/e2_clearance_quality_raw_mean"] = torch.mean(raw)

    return raw


def _log_e2_clearance_exit_metrics(
  env,
  *,
  t_contact: torch.Tensor,
  exit_event: torch.Tensor,
  in_danger_now: torch.Tensor | None = None,
  raw: torch.Tensor | None = None,
) -> None:
  log = _get_log_dict(env)
  if log is None:
    return

  time_s = env.episode_length_buf.to(torch.float) * env.step_dt
  t_clear = torch.clamp(time_s - t_contact, min=0.0)
  exit_time_mean = torch.zeros((), device=env.device, dtype=torch.float)
  if torch.any(exit_event):
    exit_time_mean = torch.mean(t_clear[exit_event])

  if in_danger_now is not None:
    log["Metrics/e2_ball_in_danger_mean"] = torch.mean(in_danger_now.float())
  log["Metrics/e2_clearance_exit_event_mean"] = torch.mean(exit_event.float())
  log["Metrics/e2_clearance_exit_time_mean"] = exit_time_mean
  if raw is not None:
    log["Metrics/e2_clearance_quality_raw_mean"] = torch.mean(raw)


class _ConfirmedDangerExitLatchState:
  def __init__(self, env):
    self._prev_in_danger = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._outside_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._post_exit_active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._prev_in_danger[env_ids] = False
    self._outside_counter[env_ids] = 0
    self._post_exit_active[env_ids] = False

  def update(
    self,
    env,
    *,
    command_name: str,
    resolution_term_name: str,
    outside_steps_required: int,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    in_danger_now = _ball_in_danger_area_mask_from_command(env, command_name)
    t_contact = _contact_time_buffer_from_termination(env, resolution_term_name)
    has_contact = t_contact >= 0.0

    continue_outside = (~in_danger_now) & (
      self._prev_in_danger | (self._outside_counter > 0)
    )
    self._outside_counter = torch.where(
      has_contact & continue_outside,
      self._outside_counter + 1,
      torch.zeros_like(self._outside_counter),
    )

    exit_event = (
      has_contact
      & (~self._post_exit_active)
      & (self._outside_counter == int(outside_steps_required))
    )
    self._post_exit_active |= exit_event
    self._prev_in_danger = in_danger_now & (~self._post_exit_active)
    return in_danger_now, t_contact, exit_event, self._post_exit_active


class StabilizeAfterExitReward:
  def __init__(self, cfg, env):
    del cfg
    self._exit_latch = _ConfirmedDangerExitLatchState(env)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._exit_latch.reset(env_ids)

  def __call__(
    self,
    env,
    command_name: str = "stand_block",
    resolution_term_name: str = "contact_resolution_window",
    outside_steps_required: int = 2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    left_foot_body_name: str = r"^left_foot_link$",
    right_foot_body_name: str = r"^right_foot_link$",
    stance_w_min: float = 0.23,
    stance_w_max: float = 0.45,
    roll_band: float = 0.1,
    roll_sigma: float = 0.12,
    pitch_target: float = 0.25,
    pitch_band: float = 0.20,
    pitch_sigma: float = 0.30,
  ) -> torch.Tensor:
    in_danger_now, t_contact, exit_event, post_exit_active = self._exit_latch.update(
      env,
      command_name=command_name,
      resolution_term_name=resolution_term_name,
      outside_steps_required=outside_steps_required,
    )
    robot: Entity = env.scene[asset_cfg.name]
    upright_score, _, _, _, _ = _posture_score_components(
      robot.data.projected_gravity_b,
      roll_band=roll_band,
      roll_sigma=roll_sigma,
      pitch_target=pitch_target,
      pitch_band=pitch_band,
      pitch_sigma=pitch_sigma,
    )
    base_height = robot.data.root_link_pos_w[:, 2]
    height_score = torch.clamp(
      (base_height - 0.40) / (0.58 - 0.40),
      min=0.0,
      max=1.0,
    )
    left_idx, right_idx = _resolve_body_index_pair_cached(
      env,
      robot,
      left_foot_body_name,
      right_foot_body_name,
    )
    foot_pos_w = robot.data.body_link_pos_w
    stance_width = torch.linalg.norm(
      foot_pos_w[:, right_idx, :2] - foot_pos_w[:, left_idx, :2],
      dim=1,
    )
    stance_width_pen = torch.square(torch.relu(float(stance_w_min) - stance_width))
    stance_width_pen += torch.square(torch.relu(stance_width - float(stance_w_max)))
    lin_vel_xy = robot.data.root_link_lin_vel_w[:, :2]
    lin_speed_pen = torch.sum(torch.square(lin_vel_xy), dim=1)
    ang_vel = robot.data.root_link_ang_vel_w
    ang_speed_pen = (
      torch.square(ang_vel[:, 0])
      + torch.square(ang_vel[:, 1])
      + 1.5 * torch.square(ang_vel[:, 2])
    )
    stabilize_raw = (
      0.6 * upright_score
      + 0.4 * height_score
      - 0.30 * stance_width_pen
      - 0.15 * lin_speed_pen
      - 0.10 * ang_speed_pen
    )
    log = _get_log_dict(env)
    if log is not None:
      _log_e2_clearance_exit_metrics(
        env,
        t_contact=t_contact,
        exit_event=exit_event,
        in_danger_now=in_danger_now,
      )
      log["Metrics/e2_stabilize_after_exit_height_score_mean"] = torch.mean(height_score)
      log["Metrics/e2_stabilize_after_exit_stance_width_mean"] = torch.mean(stance_width)
      log["Metrics/e2_stabilize_after_exit_stance_width_pen_mean"] = torch.mean(
        stance_width_pen
      )
    return post_exit_active.to(stabilize_raw.dtype) * stabilize_raw


class FaceBallAfterExitReward:
  def __init__(self, cfg, env):
    del cfg
    self._env = env
    self._exit_latch = _ConfirmedDangerExitLatchState(env)
    self._key_target_x = "e2_face_ball_after_exit_latched_target_x"
    self._key_target_y = "e2_face_ball_after_exit_latched_target_y"
    self._key_active = "e2_face_ball_after_exit_latched_active"
    self._key_steps = "e2_face_ball_after_exit_latched_steps"

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env = self._env
    self._exit_latch.reset(env_ids)
    _get_float_state_buffer(env, self._key_target_x)[env_ids] = 0.0
    _get_float_state_buffer(env, self._key_target_y)[env_ids] = 0.0
    _get_bool_state_buffer(env, self._key_active)[env_ids] = False
    _get_long_state_buffer(env, self._key_steps)[env_ids] = 0

  def __call__(
    self,
    env,
    command_name: str = "stand_block",
    resolution_term_name: str = "contact_resolution_window",
    outside_steps_required: int = 2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    waist_body_name: str = r"(?i)^waist$",
    deadband_deg: float = 12.0,
    sigma_deg: float = 25.0,
    stage1_scale: float = 1.0,
    stage2_scale: float = 1.0,
    stage3_scale: float = 0.0,
  ) -> torch.Tensor:
    in_danger_now, t_contact, exit_event, post_exit_active = self._exit_latch.update(
      env,
      command_name=command_name,
      resolution_term_name=resolution_term_name,
      outside_steps_required=outside_steps_required,
    )
    cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[cmd.cfg.ball_entity_name]

    waist_idx = _resolve_single_body_index_cached(env, robot, waist_body_name)
    waist_xy = robot.data.body_link_pos_w[:, waist_idx, :2]
    ball_xy_w = ball.data.root_link_pos_w[:, :2]
    ball_xy_local = _world_to_env_local_xy(env, ball_xy_w)
    ball_in_play_area = (
      (ball_xy_local[:, 0] >= -7.0)
      & (ball_xy_local[:, 0] <= 7.0)
      & (ball_xy_local[:, 1] >= -4.5)
      & (ball_xy_local[:, 1] <= 4.5)
    )

    latched_target_x = _get_float_state_buffer(env, self._key_target_x)
    latched_target_y = _get_float_state_buffer(env, self._key_target_y)
    latched_active = _get_bool_state_buffer(env, self._key_active)
    latched_steps = _get_long_state_buffer(env, self._key_steps)

    latch_horizon_steps = max(
      int(torch.ceil(torch.tensor(0.3 / float(env.step_dt))).item()),
      1,
    )

    target_dir_xy = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    live_target_dir_xy = _normalize_xy(ball_xy_w - waist_xy)

    if ball_in_play_area.any():
      target_dir_xy[ball_in_play_area] = live_target_dir_xy[ball_in_play_area]
      latched_target_x[ball_in_play_area] = live_target_dir_xy[ball_in_play_area, 0]
      latched_target_y[ball_in_play_area] = live_target_dir_xy[ball_in_play_area, 1]
      latched_active[ball_in_play_area] = True
      latched_steps[ball_in_play_area] = latch_horizon_steps

    use_latched_target = (~ball_in_play_area) & latched_active & (latched_steps > 0)
    if use_latched_target.any():
      target_dir_xy[use_latched_target, 0] = latched_target_x[use_latched_target]
      target_dir_xy[use_latched_target, 1] = latched_target_y[use_latched_target]
      latched_steps[use_latched_target] -= 1
      latch_expired = use_latched_target & (latched_steps <= 0)
      latched_active[latch_expired] = False

    target_active = ball_in_play_area | use_latched_target

    forward_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(
      env.num_envs, -1
    )
    waist_quat_w = robot.data.body_link_quat_w[:, waist_idx, :]
    forward_w = quat_apply(waist_quat_w, forward_local)
    forward_xy = _normalize_xy(forward_w[:, :2])

    cos_err = torch.sum(forward_xy * target_dir_xy, dim=1).clamp(-1.0, 1.0)
    yaw_err = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    if target_active.any():
      yaw_err[target_active] = torch.acos(cos_err[target_active])

    deadband_rad = float(torch.deg2rad(torch.tensor(deadband_deg)).item())
    sigma_rad = max(float(torch.deg2rad(torch.tensor(sigma_deg)).item()), 1.0e-6)
    shaped_err = torch.relu(yaw_err - deadband_rad)
    facing_score = torch.exp(-torch.square(shaped_err) / (sigma_rad * sigma_rad))
    facing_score = target_active.to(facing_score.dtype) * facing_score
    stage_index = cmd.launcher_curriculum_stage
    if stage_index == 1:
      stage_scale = float(stage1_scale)
    elif stage_index == 2:
      stage_scale = float(stage2_scale)
    elif stage_index == 3:
      stage_scale = float(stage3_scale)
    else:
      stage_scale = 1.0
    raw = post_exit_active.to(facing_score.dtype) * facing_score * stage_scale

    log = _get_log_dict(env)
    if log is not None:
      _log_e2_clearance_exit_metrics(
        env,
        t_contact=t_contact,
        exit_event=exit_event,
        in_danger_now=in_danger_now,
      )
      log["Metrics/e2_face_ball_after_exit_yaw_err_mean"] = torch.mean(yaw_err)
      log["Metrics/e2_face_ball_after_exit_score_mean"] = torch.mean(facing_score)
      log["Metrics/e2_face_ball_after_exit_raw_mean"] = torch.mean(raw)

    return raw


def save_success_reward(
  env,
  command_name: str = "stand_block",
  resolution_term_name: str | None = "contact_resolution_window",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  apply_standing_gate: bool = False,
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

  success = (resolution_done & (~goal)).float()
  return _apply_standing_gate_if_enabled(success, env, asset_cfg, apply_standing_gate)


def deflect_away_from_goal_reward(
  env,
  command_name: str = "stand_block",
  only_on_first_contact: bool = True,
  clip_speed: float = 4.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  apply_standing_gate: bool = False,
) -> torch.Tensor:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  ball: Entity = env.scene[cmd.cfg.ball_entity_name]

  if only_on_first_contact:
    contact = _first_ball_robot_contact_mask(
      env, cmd.cfg.ball_robot_contact_sensor_name
    )
  else:
    contact = _ball_robot_contact_mask(env, cmd.cfg.ball_robot_contact_sensor_name)

  vx = ball.data.root_link_lin_vel_w[:, 0]
  if cmd.cfg.goal_toward_positive_x:
    away_speed = torch.clamp(-vx, min=0.0, max=float(clip_speed))
  else:
    away_speed = torch.clamp(vx, min=0.0, max=float(clip_speed))

  raw = away_speed * contact.float()
  return _apply_standing_gate_if_enabled(raw, env, asset_cfg, apply_standing_gate)


def upright_stability_reward(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  roll_band: float = 0.1,
  roll_sigma: float = 0.12,
  pitch_target: float = 0.25,
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
    log["Metrics/e2_roll_score_mean"] = torch.mean(roll_score)
    log["Metrics/e2_pitch_score_mean"] = torch.mean(pitch_score)
    log["Metrics/e2_lateral_posture_component_mean"] = torch.mean(lateral)
    log["Metrics/e2_sagittal_posture_component_mean"] = torch.mean(sagittal)

  return posture_score


def body_ang_vel_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str = "stand_block",
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  ang_vel_xy = robot.data.root_link_ang_vel_w[:, :2]
  penalty = torch.sum(torch.square(ang_vel_xy), dim=1)
  return _apply_reward_active_mask(penalty, env, command_name)


def head_contact_penalty(
  env,
  head_sensor_name: str,
) -> torch.Tensor:
  contact_now, new_contact = _contact_rising_edge_event(
    env,
    head_sensor_name,
    state_key=f"e2_prev_head_contact::{head_sensor_name}",
  )
  raw = new_contact.to(torch.float32)

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e2_head_contact_active_mean"] = torch.mean(contact_now.to(torch.float32))
    log["Metrics/e2_head_contact_event_mean"] = torch.mean(raw)

  return raw


def arm_high_throw_deflect_reward(
  env,
  arm_sensor_name: str,
  command_name: str = "stand_block",
  high_throw_target_z_min: float = 0.55,
) -> torch.Tensor:
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  high_throw_mask = _high_throw_mask_from_command(
    env,
    command_name,
    high_throw_target_z_min=high_throw_target_z_min,
  )
  _contact_now, new_contact = _contact_rising_edge_event(
    env,
    arm_sensor_name,
    state_key=f"e2_prev_arm_contact::{arm_sensor_name}",
  )
  used = _get_bool_state_buffer(env, key=f"e2_arm_contact_reward_used::{arm_sensor_name}")
  is_first = env.episode_length_buf <= 1
  used[is_first] = False

  first_arm_contact_mask = new_contact & (~used)
  used |= first_arm_contact_mask

  ball: Entity = env.scene[cmd.cfg.ball_entity_name]
  vx = ball.data.root_link_lin_vel_w[:, 0]
  if cmd.cfg.goal_toward_positive_x:
    away_speed = torch.clamp(-vx, min=0.0)
  else:
    away_speed = torch.clamp(vx, min=0.0)
  vel_factor = torch.tanh(away_speed / 2.0)
  raw = (
    high_throw_mask.to(torch.float32)
    * first_arm_contact_mask.to(torch.float32)
    * (0.2 + 0.8 * vel_factor)
  )

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e2_high_throw_mask_mean"] = torch.mean(high_throw_mask.to(torch.float32))
    log["Metrics/e2_first_arm_contact_event_mean"] = torch.mean(
      first_arm_contact_mask.to(torch.float32)
    )
    log["Metrics/e2_arm_deflect_away_speed_mean"] = torch.mean(away_speed)
    log["Metrics/e2_arm_high_throw_deflect_raw_mean"] = torch.mean(raw)

  return raw


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
    log["Metrics/e2_low_height_soft_pen_mean"] = torch.mean(penalty)

  return penalty


def outside_area_penalty(
  env,
  command_name: str = "stand_block",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  cmd = cast(StandBlockCommand, env.command_manager.get_term(command_name))
  pos_xy_local = _world_to_env_local_xy(env, robot.data.root_link_pos_w[:, :2])

  x_min, x_max, y_min, y_max = cmd.cfg.keeper_area_bounds
  base_x = pos_xy_local[:, 0]
  base_y = pos_xy_local[:, 1]

  dx_low = torch.relu(float(x_min) - base_x)
  dx_high = torch.relu(base_x - float(x_max))
  dy_low = torch.relu(float(y_min) - base_y)
  dy_high = torch.relu(base_y - float(y_max))

  x_out = dx_low + dx_high
  y_out = dy_low + dy_high
  penalty = torch.square(x_out) + torch.square(y_out)

  log = _get_log_dict(env)
  if log is not None:
    log["Metrics/e2_outside_area_penalty_mean"] = torch.mean(penalty)

  return penalty


def fallen_indicator(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_height: float = 0.30,
  max_roll_deg: float = 100.0,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  height = robot.data.root_link_pos_w[:, 2]
  torso_roll_deg = torch.abs(torch.rad2deg(_roll_from_quat_wxyz(robot.data.root_link_quat_w)))
  fallen = (height < min_height) | (torso_roll_deg > float(max_roll_deg))
  return fallen.float()


# ---------------- Terminations ----------------


def goal_conceded_termination(
  env,
  command_name: str = "stand_block",
) -> torch.Tensor:
  return _goal_conceded_mask_from_command(env, command_name)


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
    resolution_window_s: float = 1.5,
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
    max_roll_deg: float = 100.0,
    consecutive_steps: int = 6,
  ) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    height = robot.data.root_link_pos_w[:, 2]
    torso_roll_deg = torch.abs(torch.rad2deg(_roll_from_quat_wxyz(robot.data.root_link_quat_w)))
    fallen_now = (height < min_height) | (torso_roll_deg > float(max_roll_deg))
    self._counter = torch.where(
      fallen_now,
      self._counter + 1,
      torch.zeros_like(self._counter),
    )
    return self._counter >= int(consecutive_steps)
