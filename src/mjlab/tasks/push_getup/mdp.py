from __future__ import annotations

from dataclasses import dataclass
import re
import warnings

import torch

from mjlab.envs.mdp import *  # noqa: F403
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


@dataclass(kw_only=True)
class FixedMotionCommandCfg(CommandTermCfg):
  """Fixed zero command matching tracking command dimensionality."""

  entity_name: str = "robot"
  command_dim: int | None = None
  push_delay_range_s: tuple[float, float] = (0.2, 1.0)
  push_speed_range: tuple[float, float] = (2.0, 3.8)
  push_yaw_rate_range: tuple[float, float] = (-0.35, 0.35)
  limp_gain_scale: float = 0.02
  recover_gain_scale: float = 1.0
  post_fall_delay_s: float = 0.0
  trunk_height_target: float = 0.62
  trunk_height_min: float = 0.35
  fallen_height: float = 0.42
  fallen_angle: float = 0.95
  upright_height: float = 0.62
  upright_angle: float = 0.45
  upright_hysteresis_steps: int = 8
  # Curriculum (two-stage get-up) configuration.
  curriculum_enabled: bool = False
  curriculum_success_rate_threshold: float = 0.8
  curriculum_success_rate_ema_alpha: float = 0.05
  curriculum_min_episodes: int = 200
  curriculum_min_steps: int = 0
  curriculum_time_to_upright_threshold_s: float = 0.0
  curriculum_stage1_mix_initial: float = 1.0
  curriculum_stage1_mix_after_unlock: float = 0.2
  curriculum_stage1_mix_min: float = 0.1
  curriculum_stage1_mix_decay: float = 0.0
  stage1_push_enabled: bool = False
  stage2_push_enabled: bool = True
  stage1_post_fall_delay_s: float = 0.0
  stage2_post_fall_delay_s: float | None = None
  stage1_reset_pose_mode: str = "prone"
  stage1_fallen_root_pose: tuple[float, float, float, float, float, float] = (
    0.0,
    0.0,
    -0.50,
    0.0,
    1.57,
    0.0,
  )
  stage1_fallen_pose_noise: tuple[float, float, float, float, float, float] = (
    0.02,
    0.02,
    0.02,
    0.05,
    0.05,
    0.2,
  )
  stage1_fallen_joint_pos: tuple[float, ...] | None = None
  stage1_fallen_joint_noise: float = 0.02
  stand_pose_ramp_success_threshold: float = 0.8
  stand_pose_ramp_steps: int = 2000
  stand_pose_ramp_standing_threshold: float = 0.5
  stand_pose_ramp_standing_ema_alpha: float = 0.05
  stand_pose_height_threshold: float = 0.9
  stand_pose_feet_sensor_name: str = "feet_ground_contact"
  stand_pose_pelvis_sensor_name: str | None = "pelvis_ground_contact"
  stand_pose_feet_normal_threshold: float | None = None
  stand_pose_require_pelvis_off: bool = True

  def build(self, env):
    return FixedMotionCommand(self, env)


class FixedMotionCommand(CommandTerm):
  """Outputs a constant zero command and manages push/limp/recovery phases."""

  cfg: FixedMotionCommandCfg

  def __init__(self, cfg: FixedMotionCommandCfg, env):
    super().__init__(cfg, env)
    robot = env.scene[cfg.entity_name]
    command_dim = cfg.command_dim or robot.num_joints * 2
    self._command = torch.zeros(env.num_envs, command_dim, device=self.device)

    self._push_delay = torch.zeros(env.num_envs, device=self.device)
    self._push_applied = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
    self._fallen_once = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
    self._upright_counter = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
    self._control_enabled = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
    self._control_enabled_step = torch.full(
      (env.num_envs,), -1, device=self.device, dtype=torch.long
    )
    self._fall_time = torch.zeros(env.num_envs, device=self.device)
    self._trunk_height_norm = torch.zeros(env.num_envs, device=self.device)
    self._prev_trunk_height_norm = torch.zeros(env.num_envs, device=self.device)
    self._default_action_scale: torch.Tensor | float | None = None
    self._last_gain_scale = torch.ones(env.num_envs, device=self.device)
    self._stage = torch.full(
      (env.num_envs,), 2, device=self.device, dtype=torch.long
    )
    self._episode_step_counter = torch.zeros(
      env.num_envs, device=self.device, dtype=torch.long
    )
    self._success_rate_ema = torch.tensor(0.0, device=self.device)
    self._upright_time_ema_s = torch.tensor(0.0, device=self.device)
    self._episodes_seen = 0
    self._episodes_since_unlock = 0.0
    self._stage2_unlocked = False
    self._stage1_mix_prob = (
      float(self.cfg.curriculum_stage1_mix_initial)
      if self.cfg.curriculum_enabled
      else 0.0
    )
    self._stand_pose_ramp_start_step = -1
    self._stand_pose_multiplier = 0.0
    self._standing_mask_ema = torch.tensor(0.0, device=self.device)

    self._asset_cfg = SceneEntityCfg(cfg.entity_name)
    self._init_metrics()

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    self._prev_trunk_height_norm = self._trunk_height_norm
    self._trunk_height_norm = self._compute_trunk_height_norm()
    self._update_phase()
    self._update_curriculum_stats()
    self._update_logging_metrics()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self._stage[env_ids] = self._sample_stage(env_ids)
    lower, upper = self.cfg.push_delay_range_s
    self._push_delay[env_ids] = (
      torch.rand(len(env_ids), device=self.device) * (upper - lower) + lower
    )
    self._push_applied[env_ids] = False
    self._fallen_once[env_ids] = False
    self._upright_counter[env_ids] = 0
    self._control_enabled[env_ids] = False
    self._control_enabled_step[env_ids] = -1
    self._fall_time[env_ids] = 0.0
    stage1_mask = self._stage[env_ids] == 1
    stage2_mask = ~stage1_mask

    if stage1_mask.any():
      stage1_ids = env_ids[stage1_mask]
      self._apply_fallen_pose(stage1_ids)
      self._fallen_once[stage1_ids] = True
      self._control_enabled[stage1_ids] = True
      self._control_enabled_step[stage1_ids] = self._env.episode_length_buf[stage1_ids]
      self._push_applied[stage1_ids] = True
      self._set_gains(stage1_ids, self.cfg.recover_gain_scale)
      self._set_action_enabled(stage1_ids, enabled=True)

    if stage2_mask.any():
      stage2_ids = env_ids[stage2_mask]
      self._set_gains(stage2_ids, self.cfg.limp_gain_scale)
      self._set_action_enabled(stage2_ids, enabled=False)

    curr_height = self._compute_trunk_height_norm()[env_ids]
    self._trunk_height_norm[env_ids] = curr_height
    self._prev_trunk_height_norm[env_ids] = curr_height

  def _update_command(self) -> None:
    # Command remains zero.
    pass

  def _update_phase(self) -> None:
    time_s = self._env.episode_length_buf.to(torch.float) * self._env.step_dt

    push_enabled = self._push_enabled_mask()
    to_push = (
      (~self._push_applied)
      & (~self._fallen_once)
      & (time_s >= self._push_delay)
      & push_enabled
    )
    if to_push.any():
      env_ids = to_push.nonzero(as_tuple=False).flatten()
      self._apply_random_horizontal_push(env_ids)
      self._push_applied[env_ids] = True

    fallen_now = self._is_fallen()
    newly_fallen = fallen_now & ~self._fallen_once
    if newly_fallen.any():
      env_ids = newly_fallen.nonzero(as_tuple=False).flatten()
      self._fallen_once[env_ids] = True
      self._fall_time[env_ids] = time_s[env_ids]

    ready_to_recover = self._fallen_once & ~self._control_enabled
    post_fall_delay = self._post_fall_delay_s()
    if torch.any(post_fall_delay > 0.0):
      ready_to_recover = ready_to_recover & (
        (time_s - self._fall_time) >= post_fall_delay
      )
    if ready_to_recover.any():
      env_ids = ready_to_recover.nonzero(as_tuple=False).flatten()
      self._control_enabled[env_ids] = True
      self._control_enabled_step[env_ids] = self._env.episode_length_buf[env_ids]
      self._set_gains(env_ids, self.cfg.recover_gain_scale)
      self._set_action_enabled(env_ids, enabled=True)

    upright_now = self._is_upright()
    self._upright_counter = torch.where(
      self._fallen_once & upright_now,
      self._upright_counter + 1,
      torch.zeros_like(self._upright_counter),
    )

  def _is_fallen(self) -> torch.Tensor:
    asset = self._env.scene[self.cfg.entity_name]
    height = asset.data.root_link_pos_w[:, 2]
    return (height < self.cfg.fallen_height) | (self._tilt_angle() > self.cfg.fallen_angle)

  def _is_upright(self) -> torch.Tensor:
    asset = self._env.scene[self.cfg.entity_name]
    height = asset.data.root_link_pos_w[:, 2]
    return (height > self.cfg.upright_height) & (self._tilt_angle() < self.cfg.upright_angle)

  def _tilt_angle(self) -> torch.Tensor:
    asset = self._env.scene[self.cfg.entity_name]
    projected_gravity = asset.data.projected_gravity_b
    xy_norm = torch.linalg.norm(projected_gravity[:, :2], dim=1)
    return torch.atan2(xy_norm, -projected_gravity[:, 2])

  def _compute_trunk_height_norm(self) -> torch.Tensor:
    asset = self._env.scene[self.cfg.entity_name]
    height = asset.data.root_link_pos_w[:, 2]
    denom = max(self.cfg.trunk_height_target - self.cfg.trunk_height_min, 1.0e-6)
    return torch.clamp(
      (height - self.cfg.trunk_height_min) / denom, min=0.0, max=1.0
    )

  def _apply_random_horizontal_push(self, env_ids: torch.Tensor) -> None:
    speed_lo, speed_hi = self.cfg.push_speed_range
    speeds = torch.rand(len(env_ids), device=self.device) * (speed_hi - speed_lo) + speed_lo
    angles = torch.rand(len(env_ids), device=self.device) * (2.0 * torch.pi)
    yaw_lo, yaw_hi = self.cfg.push_yaw_rate_range
    yaw_rate = torch.rand(len(env_ids), device=self.device) * (yaw_hi - yaw_lo) + yaw_lo
    asset = self._env.scene[self.cfg.entity_name]
    vel_w = asset.data.root_link_vel_w[env_ids].clone()
    vel_w[:, 0] += speeds * torch.cos(angles)
    vel_w[:, 1] += speeds * torch.sin(angles)
    vel_w[:, 5] += yaw_rate
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)

  def _apply_fallen_pose(self, env_ids: torch.Tensor) -> None:
    mode = self.cfg.stage1_reset_pose_mode
    if mode in ("none", "default", "", None):
      return
    asset = self._env.scene[self.cfg.entity_name]
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None

    base_pose = torch.tensor(
      self.cfg.stage1_fallen_root_pose, device=self.device, dtype=torch.float
    )
    pose_noise = torch.tensor(
      self.cfg.stage1_fallen_pose_noise, device=self.device, dtype=torch.float
    )
    pose_noise = torch.where(
      pose_noise >= 0.0, pose_noise, torch.zeros_like(pose_noise)
    )
    pose_samples = base_pose + (
      torch.rand((len(env_ids), 6), device=self.device) * 2.0 - 1.0
    ) * pose_noise

    root_states = default_root_state[env_ids].clone()
    positions = (
      root_states[:, 0:3] + pose_samples[:, 0:3] + self._env.scene.env_origins[env_ids]
    )
    orientations_delta = quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)
    asset.write_root_link_pose_to_sim(
      torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(
      torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
    )

    default_joint_pos = asset.data.default_joint_pos
    default_joint_vel = asset.data.default_joint_vel
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    if (
      default_joint_pos is not None
      and default_joint_vel is not None
      and soft_joint_pos_limits is not None
    ):
      joint_pos = default_joint_pos[env_ids].clone()
      if self.cfg.stage1_fallen_joint_pos is not None:
        desired = torch.tensor(
          self.cfg.stage1_fallen_joint_pos, device=self.device, dtype=joint_pos.dtype
        )
        if desired.numel() == joint_pos.shape[1]:
          joint_pos[:] = desired
        else:
          if not getattr(self._env, "_push_getup_stage1_joint_pos_warned", False):
            setattr(self._env, "_push_getup_stage1_joint_pos_warned", True)
            warnings.warn(
              "[push_getup] stage1_fallen_joint_pos length does not match "
              "num_joints; ignoring.",
              RuntimeWarning,
            )
      if self.cfg.stage1_fallen_joint_noise > 0.0:
        joint_pos += sample_uniform(
          -self.cfg.stage1_fallen_joint_noise,
          self.cfg.stage1_fallen_joint_noise,
          joint_pos.shape,
          device=self.device,
        )
      joint_limits = soft_joint_pos_limits[env_ids]
      joint_pos = joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])
      joint_vel = torch.zeros_like(joint_pos)
      asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

  def _set_gains(self, env_ids: torch.Tensor, scale: float) -> None:
    if env_ids.numel() == 0:
      return
    self._last_gain_scale[env_ids] = float(scale)
    self._set_pd_gains_scale(env_ids, float(scale))

  def _set_pd_gains_scale(self, env_ids: torch.Tensor, scale: float) -> None:
    from mjlab.actuator import (
      BuiltinPositionActuator,
      IdealPdActuator,
      XmlPositionActuator,
    )
    from mjlab.actuator.delayed_actuator import DelayedActuator

    asset = self._env.scene[self.cfg.entity_name]
    env_ids = env_ids.to(self.device, dtype=torch.int)

    if isinstance(self._asset_cfg.actuator_ids, list):
      actuators = [asset.actuators[i] for i in self._asset_cfg.actuator_ids]
    elif isinstance(self._asset_cfg.actuator_ids, slice):
      actuators = asset.actuators[self._asset_cfg.actuator_ids]
    else:
      actuators = [asset.actuators[self._asset_cfg.actuator_ids]]

    actuators = [
      a.base_actuator if isinstance(a, DelayedActuator) else a for a in actuators
    ]

    default_gainprm = None
    default_biasprm = None
    scale_tensor = torch.tensor(scale, device=self.device)

    for actuator in actuators:
      ctrl_ids = actuator.ctrl_ids
      if isinstance(actuator, (BuiltinPositionActuator, XmlPositionActuator)):
        if default_gainprm is None:
          default_gainprm = self._env.sim.get_default_field("actuator_gainprm")
          default_biasprm = self._env.sim.get_default_field("actuator_biasprm")
        self._env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
          default_gainprm[ctrl_ids, 0] * scale_tensor
        )
        self._env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
          default_biasprm[ctrl_ids, 1] * scale_tensor
        )
        self._env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
          default_biasprm[ctrl_ids, 2] * scale_tensor
        )
      elif isinstance(actuator, IdealPdActuator):
        assert actuator.default_stiffness is not None
        assert actuator.default_damping is not None
        actuator.set_gains(
          env_ids,
          kp=actuator.default_stiffness[env_ids] * scale_tensor,
          kd=actuator.default_damping[env_ids] * scale_tensor,
        )

  def _set_action_enabled(self, env_ids: torch.Tensor, enabled: bool) -> None:
    try:
      action_term = self._env.action_manager.get_term("joint_pos")
    except Exception:
      return

    # Convert to per-env scale so we can hard-disable policy control before fallen_once.
    scale = action_term.scale
    if isinstance(scale, (int, float)):
      action_term._scale = torch.full(
        (self._env.num_envs, action_term.action_dim),
        float(scale),
        device=self.device,
      )
      scale = action_term._scale
    elif torch.is_tensor(scale):
      if scale.ndim == 1:
        action_term._scale = scale.unsqueeze(0).expand(self._env.num_envs, -1).clone()
        scale = action_term._scale
    else:
      return

    # No public per-env setter; fall back to internal _scale and validate shape.
    assert isinstance(scale, torch.Tensor)
    assert scale.shape == (self._env.num_envs, action_term.action_dim)

    if self._default_action_scale is None:
      self._default_action_scale = scale.clone()

    env_ids = env_ids.to(torch.long)
    if enabled:
      if isinstance(self._default_action_scale, torch.Tensor):
        action_term._scale[env_ids] = self._default_action_scale[env_ids]
      else:
        action_term._scale[env_ids] = float(self._default_action_scale)
    else:
      action_term._scale[env_ids] = 0.0

  @property
  def fallen(self) -> torch.Tensor:
    return self._fallen_once

  @property
  def fallen_once(self) -> torch.Tensor:
    return self._fallen_once

  @property
  def control_enabled(self) -> torch.Tensor:
    return self._control_enabled

  @property
  def control_enabled_step(self) -> torch.Tensor:
    return self._control_enabled_step

  @property
  def trunk_height_norm(self) -> torch.Tensor:
    return self._trunk_height_norm

  @property
  def prev_trunk_height_norm(self) -> torch.Tensor:
    return self._prev_trunk_height_norm

  @property
  def upright_stable(self) -> torch.Tensor:
    return self._upright_counter >= self.cfg.upright_hysteresis_steps

  @property
  def stage(self) -> torch.Tensor:
    return self._stage

  def _push_enabled_mask(self) -> torch.Tensor:
    if not self.cfg.curriculum_enabled:
      return torch.ones(self._env.num_envs, device=self.device, dtype=torch.bool)
    stage1_enabled = torch.tensor(
      self.cfg.stage1_push_enabled, device=self.device, dtype=torch.bool
    )
    stage2_enabled = torch.tensor(
      self.cfg.stage2_push_enabled, device=self.device, dtype=torch.bool
    )
    return torch.where(self._stage == 1, stage1_enabled, stage2_enabled)

  def _post_fall_delay_s(self) -> torch.Tensor:
    stage2_delay = (
      self.cfg.post_fall_delay_s
      if self.cfg.stage2_post_fall_delay_s is None
      else self.cfg.stage2_post_fall_delay_s
    )
    if not self.cfg.curriculum_enabled:
      return torch.full(
        (self._env.num_envs,), float(stage2_delay), device=self.device
      )
    stage1_delay = float(self.cfg.stage1_post_fall_delay_s)
    return torch.where(
      self._stage == 1,
      torch.tensor(stage1_delay, device=self.device),
      torch.tensor(float(stage2_delay), device=self.device),
    )

  def _sample_stage(self, env_ids: torch.Tensor) -> torch.Tensor:
    if not self.cfg.curriculum_enabled:
      return torch.full(
        (len(env_ids),), 2, device=self.device, dtype=torch.long
      )
    if not self._stage2_unlocked:
      return torch.ones((len(env_ids),), device=self.device, dtype=torch.long)
    mix_prob = max(0.0, min(1.0, float(self._stage1_mix_prob)))
    if mix_prob <= 0.0:
      return torch.full(
        (len(env_ids),), 2, device=self.device, dtype=torch.long
      )
    stage1 = torch.rand(len(env_ids), device=self.device) < mix_prob
    return torch.where(
      stage1,
      torch.ones_like(stage1, dtype=torch.long),
      torch.full_like(stage1, 2, dtype=torch.long),
    )

  def _update_curriculum_stats(self) -> None:
    reset_buf = getattr(self._env, "reset_buf", None)
    if reset_buf is None:
      self._episode_step_counter += 1
      self._update_stand_pose_ramp()
      return
    done_env_ids = reset_buf.nonzero(as_tuple=False).flatten()
    if done_env_ids.numel() > 0:
      term_success = None
      try:
        term_success = self._env.termination_manager.get_term("success")
      except Exception:
        term_success = None
      if term_success is not None:
        success_flags = term_success[done_env_ids].float()
        success_mean = success_flags.mean().item()
        alpha = float(self.cfg.curriculum_success_rate_ema_alpha)
        if alpha <= 0.0:
          self._success_rate_ema = torch.tensor(
            success_mean, device=self.device
          )
        else:
          self._success_rate_ema = (
            (1.0 - alpha) * self._success_rate_ema
            + alpha * torch.tensor(success_mean, device=self.device)
          )
        if torch.any(success_flags > 0.0):
          episode_steps = self._episode_step_counter[done_env_ids]
          step_mean = episode_steps[success_flags > 0.0].float().mean().item()
          time_s = step_mean * float(self._env.step_dt)
          if self._upright_time_ema_s.item() == 0.0:
            self._upright_time_ema_s = torch.tensor(time_s, device=self.device)
          else:
            self._upright_time_ema_s = (
              (1.0 - alpha) * self._upright_time_ema_s
              + alpha * torch.tensor(time_s, device=self.device)
            )
      self._episodes_seen += int(done_env_ids.numel())
      if self.cfg.curriculum_enabled and not self._stage2_unlocked:
        if self._episodes_seen >= int(self.cfg.curriculum_min_episodes):
          if self._env.common_step_counter >= int(self.cfg.curriculum_min_steps):
            time_ok = True
            if self.cfg.curriculum_time_to_upright_threshold_s > 0.0:
              time_ok = (
                self._upright_time_ema_s.item()
                > 0.0
                and self._upright_time_ema_s.item()
                <= self.cfg.curriculum_time_to_upright_threshold_s
              )
            if (
              self._success_rate_ema.item()
              >= self.cfg.curriculum_success_rate_threshold
              and time_ok
            ):
              self._stage2_unlocked = True
              self._stage1_mix_prob = float(
                self.cfg.curriculum_stage1_mix_after_unlock
              )
      if self.cfg.curriculum_enabled and self._stage2_unlocked:
        self._episodes_since_unlock += float(done_env_ids.numel()) / float(
          max(1, self._env.num_envs)
        )
        if self.cfg.curriculum_stage1_mix_decay > 0.0:
          self._stage1_mix_prob = max(
            float(self.cfg.curriculum_stage1_mix_min),
            float(self.cfg.curriculum_stage1_mix_after_unlock)
            - float(self.cfg.curriculum_stage1_mix_decay)
            * float(self._episodes_since_unlock),
          )
        else:
          self._stage1_mix_prob = max(
            float(self.cfg.curriculum_stage1_mix_min),
            float(self.cfg.curriculum_stage1_mix_after_unlock),
          )
      self._episode_step_counter[done_env_ids] = 0

    self._episode_step_counter += 1
    self._update_stand_pose_ramp()

  def _init_metrics(self) -> None:
    self.metrics = {
      "Curriculum/push_getup_stage": torch.zeros(
        self._env.num_envs, device=self.device
      ),
      "Curriculum/push_getup_stage1_mix_prob": torch.zeros(
        self._env.num_envs, device=self.device
      ),
      "Curriculum/push_getup_success_rate_ema": torch.zeros(
        self._env.num_envs, device=self.device
      ),
      "Curriculum/push_getup_upright_time_ema_s": torch.zeros(
        self._env.num_envs, device=self.device
      ),
      "Curriculum/stand_pose_multiplier": torch.zeros(
        self._env.num_envs, device=self.device
      ),
      "Curriculum/standing_mask_frac": torch.zeros(
        self._env.num_envs, device=self.device
      ),
      "Curriculum/push_getup_gain_scale": torch.zeros(
        self._env.num_envs, device=self.device
      ),
      "Curriculum/push_getup_control_enabled_frac": torch.zeros(
        self._env.num_envs, device=self.device
      ),
      "Curriculum/push_getup_fallen_once_frac": torch.zeros(
        self._env.num_envs, device=self.device
      ),
      "Curriculum/push_getup_stage2_unlocked": torch.zeros(
        self._env.num_envs, device=self.device
      ),
    }

  def _update_logging_metrics(self) -> None:
    stage = self._stage.to(torch.float)
    self.metrics["Curriculum/push_getup_stage"] = stage
    self.metrics["Curriculum/push_getup_stage1_mix_prob"] = torch.full(
      (self._env.num_envs,), float(self._stage1_mix_prob), device=self.device
    )
    self.metrics["Curriculum/push_getup_success_rate_ema"] = torch.full(
      (self._env.num_envs,), float(self._success_rate_ema.item()), device=self.device
    )
    self.metrics["Curriculum/push_getup_upright_time_ema_s"] = torch.full(
      (self._env.num_envs,),
      float(self._upright_time_ema_s.item()),
      device=self.device,
    )
    self.metrics["Curriculum/stand_pose_multiplier"] = torch.full(
      (self._env.num_envs,), float(self._stand_pose_multiplier), device=self.device
    )
    self.metrics["Curriculum/standing_mask_frac"] = self._compute_standing_mask().float()
    self.metrics["Curriculum/push_getup_gain_scale"] = self._last_gain_scale
    self.metrics["Curriculum/push_getup_control_enabled_frac"] = (
      self._control_enabled.float()
    )
    self.metrics["Curriculum/push_getup_fallen_once_frac"] = (
      self._fallen_once.float()
    )
    self.metrics["Curriculum/push_getup_stage2_unlocked"] = torch.full(
      (self._env.num_envs,), float(self._stage2_unlocked), device=self.device
    )

  def _update_stand_pose_ramp(self) -> None:
    mask = self._compute_standing_mask()
    mask_mean = float(mask.float().mean().item())
    ema_alpha = float(self.cfg.stand_pose_ramp_standing_ema_alpha)
    if ema_alpha <= 0.0:
      self._standing_mask_ema = torch.tensor(mask_mean, device=self.device)
    else:
      self._standing_mask_ema = (
        (1.0 - ema_alpha) * self._standing_mask_ema
        + ema_alpha * torch.tensor(mask_mean, device=self.device)
      )

    threshold = float(self.cfg.stand_pose_ramp_standing_threshold)
    ramp_steps = int(self.cfg.stand_pose_ramp_steps)
    condition_met = threshold <= 0.0 or self._standing_mask_ema.item() >= threshold

    if condition_met and self._stand_pose_ramp_start_step < 0:
      self._stand_pose_ramp_start_step = int(self._env.common_step_counter)

    if self._stand_pose_ramp_start_step < 0:
      self._stand_pose_multiplier = 0.0
      return

    if ramp_steps <= 0:
      self._stand_pose_multiplier = 1.0
      return

    elapsed = int(self._env.common_step_counter) - self._stand_pose_ramp_start_step
    self._stand_pose_multiplier = max(0.0, min(1.0, elapsed / float(ramp_steps)))

  @property
  def stand_pose_multiplier(self) -> float:
    return self._stand_pose_multiplier

  def _compute_standing_mask(self) -> torch.Tensor:
    trunk_height_norm = self._trunk_height_norm
    gate = self._fallen_once & self._control_enabled
    height_mask = trunk_height_norm >= float(self.cfg.stand_pose_height_threshold)

    # Feet contact.
    both_feet = torch.zeros(self._env.num_envs, device=self.device, dtype=torch.bool)
    try:
      feet_sensor: ContactSensor = self._env.scene[self.cfg.stand_pose_feet_sensor_name]
      if feet_sensor.data.found is not None:
        feet_found = feet_sensor.data.found.squeeze(-1).clone()
        if (
          self.cfg.stand_pose_feet_normal_threshold is not None
          and feet_sensor.data.normal is not None
        ):
          normal_z = feet_sensor.data.normal[..., 2]
          feet_found = feet_found * (
            torch.abs(normal_z) >= float(self.cfg.stand_pose_feet_normal_threshold)
          )
        if feet_found.shape[1] >= 2:
          both_feet = (feet_found[:, 0] > 0) & (feet_found[:, 1] > 0)
    except Exception:
      pass

    # Pelvis contact.
    pelvis_clear = torch.ones(
      self._env.num_envs, device=self.device, dtype=torch.bool
    )
    if self.cfg.stand_pose_require_pelvis_off and self.cfg.stand_pose_pelvis_sensor_name:
      try:
        pelvis_sensor: ContactSensor = self._env.scene[
          self.cfg.stand_pose_pelvis_sensor_name
        ]
        if pelvis_sensor.data.found is not None:
          pelvis_contact = (pelvis_sensor.data.found.squeeze(-1) > 0).any(dim=1)
          pelvis_clear = ~pelvis_contact
      except Exception:
        pass

    return gate & height_mask & both_feet & pelvis_clear


class UprightSuccess:
  """Terminate when upright and above height for K consecutive steps."""

  def __init__(
    self,
    height_threshold: float,
    angle_threshold: float,
    consecutive_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    require_fallen: bool = True,
    command_name: str | None = None,
  ) -> None:
    self.height_threshold = height_threshold
    self.angle_threshold = angle_threshold
    self.consecutive_steps = consecutive_steps
    self.asset_cfg = asset_cfg
    self.require_fallen = require_fallen
    self.command_name = command_name
    self._counter: torch.Tensor | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._counter is None:
      return
    if env_ids is None:
      env_ids = slice(None)
    self._counter[env_ids] = 0

  def __call__(self, env) -> torch.Tensor:
    asset = env.scene[self.asset_cfg.name]
    if self._counter is None or self._counter.shape[0] != env.num_envs:
      self._counter = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
      )
    if self.require_fallen:
      cmd = _get_command_term(env, self.command_name)
      fallen = getattr(cmd, "fallen", None) if cmd is not None else None
      if fallen is None or not torch.any(fallen):
        self._counter[:] = 0
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
      not_fallen = ~fallen
      if torch.any(not_fallen):
        self._counter[not_fallen] = 0
    height = asset.data.root_link_pos_w[:, 2]
    projected_gravity = asset.data.projected_gravity_b
    angle = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0))
    upright = (height >= self.height_threshold) & (angle <= self.angle_threshold)
    self._counter = torch.where(
      upright, self._counter + 1, torch.zeros_like(self._counter)
    )
    return self._counter >= self.consecutive_steps


class LowMotionTermination:
  """Terminate when movement is below thresholds for K consecutive steps."""

  def __init__(
    self,
    lin_vel_threshold: float,
    ang_vel_threshold: float,
    consecutive_steps: int,
    min_recovery_steps: int = 0,
    min_height_norm: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
  ) -> None:
    self.lin_vel_threshold = lin_vel_threshold
    self.ang_vel_threshold = ang_vel_threshold
    self.consecutive_steps = consecutive_steps
    self.min_recovery_steps = min_recovery_steps
    self.min_height_norm = min_height_norm
    self.asset_cfg = asset_cfg
    self.command_name = command_name
    self._counter: torch.Tensor | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._counter is None:
      return
    if env_ids is None:
      env_ids = slice(None)
    self._counter[env_ids] = 0

  def __call__(self, env) -> torch.Tensor:
    asset = env.scene[self.asset_cfg.name]
    if self._counter is None or self._counter.shape[0] != env.num_envs:
      self._counter = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
      )

    lin_vel = asset.data.root_link_lin_vel_w
    ang_vel = asset.data.root_link_ang_vel_w
    lin_speed = torch.linalg.norm(lin_vel, dim=1)
    ang_speed = torch.linalg.norm(ang_vel, dim=1)

    recovery_steps = _recovery_steps(env, self.command_name)
    trunk_height_norm = _trunk_height_norm(env, self.command_name)
    gate = _recovery_gate_mask(env, self.command_name)

    active = gate
    if self.min_recovery_steps > 0:
      active = active & (recovery_steps >= int(self.min_recovery_steps))
    if self.min_height_norm > 0.0:
      active = active & (trunk_height_norm >= float(self.min_height_norm))

    low_motion = (lin_speed <= self.lin_vel_threshold) & (
      ang_speed <= self.ang_vel_threshold
    )
    stalled = active & low_motion

    self._counter = torch.where(
      stalled, self._counter + 1, torch.zeros_like(self._counter)
    )
    return self._counter >= self.consecutive_steps


def _warn_missing_command_term_once(env, reason: str) -> None:
  if getattr(env, "_push_getup_missing_command_warned", False):
    return
  setattr(env, "_push_getup_missing_command_warned", True)
  warnings.warn(
    f"[push_getup] No suitable command term found ({reason}); "
    "reward gating disabled.",
    RuntimeWarning,
  )


def _get_command_term(env, command_name: str | None) -> CommandTerm | None:
  if not hasattr(env, "command_manager"):
    return None
  mgr = env.command_manager
  if command_name:
    try:
      return mgr.get_term(command_name)
    except Exception:
      _warn_missing_command_term_once(env, f"term '{command_name}' not found")
      return None
  try:
    for name in getattr(mgr, "active_terms", []):
      try:
        term = mgr.get_term(name)
      except Exception:
        continue
      if any(
        hasattr(term, attr)
        for attr in (
          "fallen_once",
          "fallen",
          "control_enabled",
          "upright_stable",
          "trunk_height_norm",
        )
      ):
        return term
  except Exception:
    _warn_missing_command_term_once(env, "failed to scan active terms")
    return None
  _warn_missing_command_term_once(env, "no term with required attributes")
  return None


def _fallen_once_mask(env, command_name: str | None = None) -> torch.Tensor:
  cmd = _get_command_term(env, command_name)
  if cmd is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  fallen_once = getattr(cmd, "fallen_once", None)
  if fallen_once is None:
    fallen_once = getattr(cmd, "fallen", None)
  if fallen_once is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return fallen_once


def _control_enabled_mask(env, command_name: str | None = None) -> torch.Tensor:
  cmd = _get_command_term(env, command_name)
  if cmd is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  control_enabled = getattr(cmd, "control_enabled", None)
  if control_enabled is None:
    return _fallen_once_mask(env, command_name)
  return control_enabled


def _recovery_gate_mask(env, command_name: str | None = None) -> torch.Tensor:
  fallen_once = _fallen_once_mask(env, command_name)
  control_enabled = _control_enabled_mask(env, command_name)
  return fallen_once & control_enabled


def _trunk_height_norm(env, command_name: str | None = None) -> torch.Tensor:
  cmd = _get_command_term(env, command_name)
  if cmd is None:
    return torch.zeros(env.num_envs, device=env.device)
  trunk_height_norm = getattr(cmd, "trunk_height_norm", None)
  if trunk_height_norm is None:
    return torch.zeros(env.num_envs, device=env.device)
  return trunk_height_norm


def _recovery_steps(env, command_name: str | None = None) -> torch.Tensor:
  cmd = _get_command_term(env, command_name)
  if cmd is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
  control_enabled_step = getattr(cmd, "control_enabled_step", None)
  if control_enabled_step is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
  steps = env.episode_length_buf - control_enabled_step
  steps = torch.where(control_enabled_step >= 0, steps, torch.zeros_like(steps))
  return steps


def _stand_pose_multiplier(env, command_name: str | None = None) -> torch.Tensor:
  cmd = _get_command_term(env, command_name)
  if cmd is None:
    return torch.zeros(env.num_envs, device=env.device)
  multiplier = getattr(cmd, "stand_pose_multiplier", 0.0)
  return torch.full((env.num_envs,), float(multiplier), device=env.device)


def _standing_mask(env, command_name: str | None = None) -> torch.Tensor:
  cmd = _get_command_term(env, command_name)
  if cmd is not None and hasattr(cmd, "_compute_standing_mask"):
    try:
      mask = cmd._compute_standing_mask()
      if torch.is_tensor(mask):
        return mask
    except Exception:
      pass
  return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def _stage_id(env, command_name: str | None = None) -> torch.Tensor:
  cmd = _get_command_term(env, command_name)
  if cmd is None:
    return torch.full((env.num_envs,), 2, device=env.device, dtype=torch.long)
  stage = getattr(cmd, "stage", None)
  if stage is None:
    return torch.full((env.num_envs,), 2, device=env.device, dtype=torch.long)
  return stage


def _stage_masks(
  env, command_name: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  stage = _stage_id(env, command_name)
  stage1 = stage == 1
  return stage1, ~stage1


def success_reward(
  env, term_name: str = "success", command_name: str | None = None
) -> torch.Tensor:
  term = env.termination_manager.get_term(term_name)
  return term.float() * _recovery_gate_mask(env, command_name).float()


def upright_reward(env, command_name: str | None = None) -> torch.Tensor:
  asset = env.scene["robot"]
  projected_gravity = asset.data.projected_gravity_b
  uprightness = torch.clamp(-projected_gravity[:, 2], min=0.0, max=1.0)
  trunk_height_norm = _trunk_height_norm(env, command_name)
  return uprightness * trunk_height_norm * _recovery_gate_mask(env, command_name).float()


def trunk_height_reward(
  env,
  target_height: float,
  min_height: float = 0.10,
  command_name: str | None = None,
) -> torch.Tensor:
  asset = env.scene["robot"]
  height = asset.data.root_link_pos_w[:, 2]
  denom = max(target_height - min_height, 1.0e-6)
  normalized = torch.clamp((height - min_height) / denom, min=0.0, max=1.0)
  return normalized * _recovery_gate_mask(env, command_name).float()


def angular_velocity_penalty(env, command_name: str | None = None) -> torch.Tensor:
  asset = env.scene["robot"]
  ang_vel = asset.data.root_link_ang_vel_b
  penalty = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
  return penalty * _recovery_gate_mask(env, command_name).float()


def recovery_step_penalty(env, command_name: str | None = None) -> torch.Tensor:
  return _recovery_gate_mask(env, command_name).float()


def support_points_reward(
  env,
  feet_sensor_name: str,
  hands_sensor_name: str,
  height_threshold: float,
  foot_weight: float = 1.0,
  hand_weight: float = 1.0,
  dh_clip: float = 0.05,
  feet_normal_threshold: float | None = None,
  command_name: str | None = None,
) -> torch.Tensor:
  cmd = _get_command_term(env, command_name)
  if cmd is None:
    return torch.zeros(env.num_envs, device=env.device)
  trunk_height_norm = getattr(cmd, "trunk_height_norm", None)
  prev_height_norm = getattr(cmd, "prev_trunk_height_norm", None)
  if trunk_height_norm is None or prev_height_norm is None:
    return torch.zeros(env.num_envs, device=env.device)
  early_mask = trunk_height_norm < height_threshold
  feet_sensor: ContactSensor = env.scene[feet_sensor_name]
  hands_sensor: ContactSensor = env.scene[hands_sensor_name]
  assert feet_sensor.data.found is not None
  assert hands_sensor.data.found is not None
  feet_found = (feet_sensor.data.found.squeeze(-1) > 0).float()
  if feet_normal_threshold is not None and feet_sensor.data.normal is not None:
    normal_z = feet_sensor.data.normal[..., 2]
    feet_found = feet_found * (torch.abs(normal_z) >= float(feet_normal_threshold))
  hand_found = (hands_sensor.data.found.squeeze(-1) > 0).float()
  if feet_found.numel() > 0:
    feet_frac = feet_found.mean(dim=1)
  else:
    feet_frac = torch.zeros(env.num_envs, device=env.device)
  if hand_found.numel() > 0:
    hand_frac = hand_found.mean(dim=1)
  else:
    hand_frac = torch.zeros(env.num_envs, device=env.device)
  denom = max(foot_weight + hand_weight, 1.0e-6)
  support_frac = (foot_weight * feet_frac + hand_weight * hand_frac) / denom
  delta_h = torch.clamp(trunk_height_norm - prev_height_norm, min=0.0)
  if dh_clip > 0.0:
    delta_h = torch.clamp(delta_h, max=dh_clip)
  reward = support_frac * delta_h
  return reward * early_mask.float() * _recovery_gate_mask(env, command_name).float()


def hand_push_reward(
  env,
  hands_sensor_name: str,
  height_threshold: float,
  dh_clip: float = 0.05,
  command_name: str | None = None,
) -> torch.Tensor:
  cmd = _get_command_term(env, command_name)
  if cmd is None:
    return torch.zeros(env.num_envs, device=env.device)
  trunk_height_norm = getattr(cmd, "trunk_height_norm", None)
  prev_height_norm = getattr(cmd, "prev_trunk_height_norm", None)
  if trunk_height_norm is None or prev_height_norm is None:
    return torch.zeros(env.num_envs, device=env.device)
  early_mask = trunk_height_norm < height_threshold
  hands_sensor: ContactSensor = env.scene[hands_sensor_name]
  assert hands_sensor.data.found is not None
  hand_found = (hands_sensor.data.found.squeeze(-1) > 0).float()
  if hand_found.numel() > 0:
    hand_frac = hand_found.mean(dim=1)
  else:
    hand_frac = torch.zeros(env.num_envs, device=env.device)
  delta_h = torch.clamp(trunk_height_norm - prev_height_norm, min=0.0)
  if dh_clip > 0.0:
    delta_h = torch.clamp(delta_h, max=dh_clip)
  reward = hand_frac * delta_h
  return reward * early_mask.float() * _recovery_gate_mask(env, command_name).float()


def both_feet_reward(
  env,
  feet_sensor_name: str,
  height_threshold: float,
  feet_normal_threshold: float | None = None,
  command_name: str | None = None,
) -> torch.Tensor:
  trunk_height_norm = _trunk_height_norm(env, command_name)
  late_mask = trunk_height_norm >= height_threshold
  feet_sensor: ContactSensor = env.scene[feet_sensor_name]
  assert feet_sensor.data.found is not None
  feet_found = feet_sensor.data.found.squeeze(-1)
  if feet_normal_threshold is not None and feet_sensor.data.normal is not None:
    normal_z = feet_sensor.data.normal[..., 2]
    feet_found = feet_found * (torch.abs(normal_z) >= float(feet_normal_threshold))
  if feet_found.shape[1] < 2:
    return torch.zeros(env.num_envs, device=env.device)
  left_contact = feet_found[:, 0] > 0
  right_contact = feet_found[:, 1] > 0
  both_feet = (left_contact & right_contact).float()
  return both_feet * late_mask.float() * _recovery_gate_mask(env, command_name).float()


def hands_contact_penalty(
  env,
  hands_sensor_name: str,
  feet_sensor_name: str,
  height_threshold: float,
  command_name: str | None = None,
) -> torch.Tensor:
  trunk_height_norm = _trunk_height_norm(env, command_name)
  late_mask = trunk_height_norm >= height_threshold
  hands_sensor: ContactSensor = env.scene[hands_sensor_name]
  assert hands_sensor.data.found is not None
  standing_mask = _standing_mask(env, command_name)
  if not torch.any(standing_mask):
    feet_sensor: ContactSensor = env.scene[feet_sensor_name]
    assert feet_sensor.data.found is not None
    feet_found = feet_sensor.data.found.squeeze(-1)
    if feet_found.shape[1] < 2:
      standing_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
      standing_mask = (feet_found[:, 0] > 0) & (feet_found[:, 1] > 0)
  hand_contacts = (hands_sensor.data.found.squeeze(-1) > 0).float().mean(dim=1)
  penalty = hand_contacts * late_mask.float() * standing_mask.float()
  return penalty * _recovery_gate_mask(env, command_name).float()


def pelvis_contact_penalty(
  env,
  pelvis_sensor_name: str,
  height_threshold: float,
  min_recovery_steps: int = 0,
  command_name: str | None = None,
) -> torch.Tensor:
  trunk_height_norm = _trunk_height_norm(env, command_name)
  recovery_steps = _recovery_steps(env, command_name)
  if min_recovery_steps <= 0:
    steps_mask = torch.ones_like(trunk_height_norm, dtype=torch.bool)
  else:
    steps_mask = recovery_steps >= int(min_recovery_steps)
  gate_mask = (trunk_height_norm >= height_threshold) | steps_mask
  pelvis_sensor: ContactSensor = env.scene[pelvis_sensor_name]
  assert pelvis_sensor.data.found is not None
  pelvis_contacts = (pelvis_sensor.data.found.squeeze(-1) > 0).float().sum(dim=1)
  return pelvis_contacts * gate_mask.float() * _recovery_gate_mask(env, command_name).float()


def head_contact_penalty(
  env,
  head_sensor_name: str,
  height_threshold: float,
  early_scale: float = 0.2,
  late_scale: float = 1.0,
  command_name: str | None = None,
) -> torch.Tensor:
  trunk_height_norm = _trunk_height_norm(env, command_name)
  head_sensor: ContactSensor = env.scene[head_sensor_name]
  assert head_sensor.data.found is not None
  head_contacts = (head_sensor.data.found.squeeze(-1) > 0).float()
  if head_contacts.numel() > 0:
    head_contact_frac = head_contacts.mean(dim=1)
  else:
    head_contact_frac = torch.zeros(env.num_envs, device=env.device)
  early_mask = trunk_height_norm < height_threshold
  scale = torch.where(
    early_mask,
    torch.tensor(float(early_scale), device=env.device),
    torch.tensor(float(late_scale), device=env.device),
  )
  return head_contact_frac * scale * _recovery_gate_mask(env, command_name).float()


def stand_pose_penalty(
  env,
  feet_sensor_name: str,
  pelvis_sensor_name: str | None,
  height_threshold: float,
  q_scale: float = 1.0,
  require_pelvis_off: bool = True,
  exclude_joint_patterns: tuple[str, ...] | None = None,
  command_name: str | None = None,
) -> torch.Tensor:
  trunk_height_norm = _trunk_height_norm(env, command_name)
  feet_sensor: ContactSensor = env.scene[feet_sensor_name]
  assert feet_sensor.data.found is not None
  feet_found = feet_sensor.data.found.squeeze(-1)
  if feet_found.shape[1] < 2:
    both_feet = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  else:
    both_feet = (feet_found[:, 0] > 0) & (feet_found[:, 1] > 0)
  pelvis_clear = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
  if pelvis_sensor_name and require_pelvis_off:
    pelvis_sensor: ContactSensor = env.scene[pelvis_sensor_name]
    assert pelvis_sensor.data.found is not None
    pelvis_contact = (pelvis_sensor.data.found.squeeze(-1) > 0).any(dim=1)
    pelvis_clear = ~pelvis_contact

  standing_mask = (
    (trunk_height_norm >= height_threshold)
    & both_feet
    & pelvis_clear
    & _recovery_gate_mask(env, command_name)
  )

  asset = env.scene["robot"]
  q = asset.data.joint_pos
  q_ref = asset.data.default_joint_pos
  if q_ref is None:
    return torch.zeros(env.num_envs, device=env.device)
  if exclude_joint_patterns:
    cache_key = "_stand_pose_include_mask"
    mask = getattr(env, cache_key, None)
    if (
      mask is None
      or not torch.is_tensor(mask)
      or mask.shape[0] != q.shape[1]
      or getattr(env, "_stand_pose_exclude_patterns", None) != exclude_joint_patterns
    ):
      names = asset.joint_names
      include = []
      for name in names:
        exclude = False
        for pattern in exclude_joint_patterns:
          if re.fullmatch(pattern, name):
            exclude = True
            break
        include.append(not exclude)
      mask = torch.tensor(include, device=env.device, dtype=torch.bool)
      setattr(env, cache_key, mask)
      setattr(env, "_stand_pose_exclude_patterns", exclude_joint_patterns)
    include_idx = torch.nonzero(mask, as_tuple=False).flatten()
    if include_idx.numel() == 0:
      return torch.zeros(env.num_envs, device=env.device)
    q = q[:, include_idx]
    q_ref = q_ref[:, include_idx]
  scale = max(float(q_scale), 1.0e-6)
  pose_err = torch.mean(((q - q_ref) / scale) ** 2, dim=1)
  multiplier = _stand_pose_multiplier(env, command_name)
  return pose_err * multiplier * standing_mask.float()


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  stage1_scale: float = 1.0,
  stage2_scale: float = 1.0,
  stage1_sensor_name: str | None = None,
) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  stage1_mask, _ = _stage_masks(env, command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  stage2_cost = sensor.data.found.squeeze(-1)
  if stage1_sensor_name is not None:
    stage1_sensor: ContactSensor = env.scene[stage1_sensor_name]
    assert stage1_sensor.data.found is not None
    stage1_cost = stage1_sensor.data.found.squeeze(-1)
  else:
    stage1_cost = stage2_cost

  cost = torch.where(stage1_mask, stage1_cost, stage2_cost)
  scale = torch.where(
    stage1_mask,
    torch.tensor(stage1_scale, device=env.device),
    torch.tensor(stage2_scale, device=env.device),
  )
  return cost * scale * _recovery_gate_mask(env, command_name).float()


def trunk_height_progress_reward(
  env,
  k: float,
  dh_clip: float,
  command_name: str | None = None,
  k_stage1: float | None = None,
  k_stage2: float | None = None,
  dh_clip_stage1: float | None = None,
  dh_clip_stage2: float | None = None,
) -> torch.Tensor:
  """Reward for positive progress in normalized trunk height."""
  cmd = _get_command_term(env, command_name)
  if cmd is None:
    return torch.zeros(env.num_envs, device=env.device)
  curr = getattr(cmd, "trunk_height_norm", None)
  prev = getattr(cmd, "prev_trunk_height_norm", None)
  if curr is None or prev is None:
    return torch.zeros(env.num_envs, device=env.device)

  stage1_mask, _ = _stage_masks(env, command_name)
  k1 = k if k_stage1 is None else k_stage1
  k2 = k if k_stage2 is None else k_stage2
  dh1 = dh_clip if dh_clip_stage1 is None else dh_clip_stage1
  dh2 = dh_clip if dh_clip_stage2 is None else dh_clip_stage2
  k_tensor = torch.where(
    stage1_mask,
    torch.tensor(k1, device=env.device),
    torch.tensor(k2, device=env.device),
  )
  dh_tensor = torch.where(
    stage1_mask,
    torch.tensor(dh1, device=env.device),
    torch.tensor(dh2, device=env.device),
  )
  delta = curr - prev
  progress = torch.clamp(delta, min=0.0)
  progress = torch.minimum(progress, dh_tensor)
  return k_tensor * progress * _recovery_gate_mask(env, command_name).float()
