from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from mjlab.entity import Entity

GROUND_SHOT_FAMILY = 0
ONE_BOUNCE_FAMILY = 1
LOB_CHIP_FAMILY = 2
CROSS_FAMILY = 3
LONG_DRIVEN_FAMILY = 4
NUM_LAUNCH_FAMILIES = 5

LAUNCH_FAMILY_NAMES = (
  "ground_shot",
  "one_bounce_shot",
  "lob_chip",
  "cross",
  "long_driven",
)

E2_STAGE1_BASIC = "e2_stage1_basic"
E2_STAGE2_LATERAL = "e2_stage2_lateral"
E2_STAGE3_VERTICAL_PACE = "e2_stage3_vertical_pace"
E2_STAGE4_FULL_GEOMETRY = "e2_stage4_full_geometry"
E2_STAGE5_FINAL_HARDER = "e2_stage5_final_harder"

E2_LAUNCHER_CURRICULUM_PRESET_NAMES = (
  E2_STAGE1_BASIC,
  E2_STAGE2_LATERAL,
  E2_STAGE3_VERTICAL_PACE,
  E2_STAGE4_FULL_GEOMETRY,
  E2_STAGE5_FINAL_HARDER,
)


@dataclass(frozen=True, kw_only=True)
class GoalkeeperLauncherCurriculumPreset:
  """Named launcher-only sampling preset for E2 stand-block."""

  name: str
  family_weights: tuple[float, float, float, float, float]
  delay_range: tuple[float, float]
  t_goal_band: tuple[float, float]
  shot_target_mode_probs: tuple[float, float, float]
  shot_nearpost_abs_y_range: tuple[float, float]
  shot_farpost_abs_y_range: tuple[float, float]
  shot_center_y_range: tuple[float, float]
  shot_low_z_range: tuple[float, float]
  shot_mid_z_range: tuple[float, float]
  shot_low_z_prob: float
  lob_tof_range: tuple[float, float]
  lob_target_z_range: tuple[float, float]
  cross_target_x_range: tuple[float, float]
  cross_driven_tof_range: tuple[float, float]
  cross_lofted_tof_range: tuple[float, float]
  long_driven_target_z_range: tuple[float, float]
  deflection_prob: float
  deflection_dv_mag_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class GoalkeeperLauncherPromotionDecision:
  promoted: bool
  from_stage_index: int
  from_preset_name: str
  to_stage_index: int
  to_preset_name: str
  save_rate: float
  fall_rate: float
  required_save_rate: float | None
  max_fall_rate: float
  reason: str


def _e2_time_tiers_from_band(
  band: tuple[float, float],
  *,
  slow_weight: float,
  mid_weight: float,
  fast_weight: float,
  slow_start_frac: float,
  mid_start_frac: float,
  mid_end_frac: float,
  fast_end_frac: float,
) -> tuple[tuple[float, float, float], ...]:
  lo, hi = band
  span = max(float(hi) - float(lo), 1.0e-4)
  return (
    (
      slow_weight,
      lo + slow_start_frac * span,
      hi,
    ),
    (
      mid_weight,
      lo + mid_start_frac * span,
      lo + mid_end_frac * span,
    ),
    (
      fast_weight,
      lo,
      lo + fast_end_frac * span,
    ),
  )


def _e2_ground_time_tiers(
  band: tuple[float, float],
) -> tuple[tuple[float, float, float], ...]:
  return _e2_time_tiers_from_band(
    band,
    slow_weight=0.30,
    mid_weight=0.45,
    fast_weight=0.25,
    slow_start_frac=0.45,
    mid_start_frac=0.20,
    mid_end_frac=0.62,
    fast_end_frac=0.30,
  )


def _e2_one_bounce_time_tiers(
  band: tuple[float, float],
) -> tuple[tuple[float, float, float], ...]:
  return _e2_time_tiers_from_band(
    band,
    slow_weight=0.35,
    mid_weight=0.45,
    fast_weight=0.20,
    slow_start_frac=0.42,
    mid_start_frac=0.18,
    mid_end_frac=0.56,
    fast_end_frac=0.28,
  )


def _e2_long_driven_time_tiers(
  band: tuple[float, float],
) -> tuple[tuple[float, float, float], ...]:
  return _e2_time_tiers_from_band(
    band,
    slow_weight=0.25,
    mid_weight=0.45,
    fast_weight=0.30,
    slow_start_frac=0.42,
    mid_start_frac=0.16,
    mid_end_frac=0.52,
    fast_end_frac=0.22,
  )


_E2_LAUNCHER_PRESETS: dict[str, GoalkeeperLauncherCurriculumPreset] = {
  E2_STAGE1_BASIC: GoalkeeperLauncherCurriculumPreset(
    name=E2_STAGE1_BASIC,
    family_weights=(0.90, 0.10, 0.00, 0.00, 0.00),
    delay_range=(0.18, 0.30),
    t_goal_band=(0.55, 1.00),
    shot_target_mode_probs=(0.18, 0.00, 0.82),
    shot_nearpost_abs_y_range=(0.38, 0.68),
    shot_farpost_abs_y_range=(0.72, 0.92),
    shot_center_y_range=(-0.22, 0.22),
    shot_low_z_range=(0.11, 0.16),
    shot_mid_z_range=(0.18, 0.24),
    shot_low_z_prob=1.0,
    lob_tof_range=(0.55, 1.05),
    lob_target_z_range=(0.75, 1.15),
    cross_target_x_range=(5.8, 6.95),
    cross_driven_tof_range=(0.45, 0.75),
    cross_lofted_tof_range=(0.75, 1.20),
    long_driven_target_z_range=(0.75, 1.10),
    deflection_prob=0.00,
  ),
  E2_STAGE2_LATERAL: GoalkeeperLauncherCurriculumPreset(
    name=E2_STAGE2_LATERAL,
    family_weights=(0.75, 0.25, 0.00, 0.00, 0.00),
    delay_range=(0.15, 0.28),
    t_goal_band=(0.50, 0.95),
    shot_target_mode_probs=(0.24, 0.08, 0.68),
    shot_nearpost_abs_y_range=(0.55, 0.95),
    shot_farpost_abs_y_range=(0.88, 1.05),
    shot_center_y_range=(-0.34, 0.34),
    shot_low_z_range=(0.11, 0.18),
    shot_mid_z_range=(0.22, 0.34),
    shot_low_z_prob=0.88,
    lob_tof_range=(0.55, 1.05),
    lob_target_z_range=(0.78, 1.18),
    cross_target_x_range=(5.8, 6.95),
    cross_driven_tof_range=(0.45, 0.75),
    cross_lofted_tof_range=(0.75, 1.20),
    long_driven_target_z_range=(0.78, 1.12),
    deflection_prob=0.00,
  ),
  E2_STAGE3_VERTICAL_PACE: GoalkeeperLauncherCurriculumPreset(
    name=E2_STAGE3_VERTICAL_PACE,
    family_weights=(0.55, 0.20, 0.15, 0.00, 0.10),
    delay_range=(0.12, 0.24),
    t_goal_band=(0.42, 0.85),
    shot_target_mode_probs=(0.28, 0.18, 0.54),
    shot_nearpost_abs_y_range=(0.68, 1.12),
    shot_farpost_abs_y_range=(0.92, 1.22),
    shot_center_y_range=(-0.48, 0.48),
    shot_low_z_range=(0.11, 0.20),
    shot_mid_z_range=(0.30, 0.82),
    shot_low_z_prob=0.58,
    lob_tof_range=(0.44, 0.78),
    lob_target_z_range=(0.88, 1.24),
    cross_target_x_range=(5.8, 6.95),
    cross_driven_tof_range=(0.45, 0.75),
    cross_lofted_tof_range=(0.75, 1.20),
    long_driven_target_z_range=(0.74, 1.18),
    deflection_prob=0.00,
  ),
  E2_STAGE4_FULL_GEOMETRY: GoalkeeperLauncherCurriculumPreset(
    name=E2_STAGE4_FULL_GEOMETRY,
    family_weights=(0.45, 0.15, 0.15, 0.10, 0.15),
    delay_range=(0.10, 0.22),
    t_goal_band=(0.38, 0.80),
    shot_target_mode_probs=(0.34, 0.26, 0.40),
    shot_nearpost_abs_y_range=(0.75, 1.20),
    shot_farpost_abs_y_range=(0.95, 1.28),
    shot_center_y_range=(-0.45, 0.45),
    shot_low_z_range=(0.11, 0.20),
    shot_mid_z_range=(0.28, 1.02),
    shot_low_z_prob=0.45,
    lob_tof_range=(0.40, 0.78),
    lob_target_z_range=(0.80, 1.45),
    cross_target_x_range=(6.78, 6.98),
    cross_driven_tof_range=(0.34, 0.54),
    cross_lofted_tof_range=(0.48, 0.64),
    long_driven_target_z_range=(0.78, 1.30),
    deflection_prob=0.03,
  ),
  E2_STAGE5_FINAL_HARDER: GoalkeeperLauncherCurriculumPreset(
    name=E2_STAGE5_FINAL_HARDER,
    family_weights=(0.40, 0.15, 0.15, 0.10, 0.20),
    delay_range=(0.10, 0.20),
    t_goal_band=(0.35, 0.75),
    shot_target_mode_probs=(0.34, 0.34, 0.32),
    shot_nearpost_abs_y_range=(0.75, 1.20),
    shot_farpost_abs_y_range=(0.95, 1.30),
    shot_center_y_range=(-0.35, 0.35),
    shot_low_z_range=(0.11, 0.18),
    shot_mid_z_range=(0.22, 1.10),
    shot_low_z_prob=0.38,
    lob_tof_range=(0.36, 0.72),
    lob_target_z_range=(0.75, 1.55),
    cross_target_x_range=(6.84, 6.99),
    cross_driven_tof_range=(0.30, 0.48),
    cross_lofted_tof_range=(0.44, 0.60),
    long_driven_target_z_range=(0.80, 1.35),
    deflection_prob=0.06,
  ),
}


def get_e2_launcher_preset(name: str) -> GoalkeeperLauncherCurriculumPreset:
  try:
    return _E2_LAUNCHER_PRESETS[name]
  except KeyError as exc:
    supported = ", ".join(sorted(_E2_LAUNCHER_PRESETS))
    raise ValueError(
      f"Unknown E2 launcher preset '{name}'. Supported presets: {supported}."
    ) from exc


def get_e2_launcher_curriculum_stage_index(preset_name: str) -> int | None:
  try:
    return E2_LAUNCHER_CURRICULUM_PRESET_NAMES.index(preset_name) + 1
  except ValueError:
    return None


def get_e2_launcher_curriculum_preset_name(stage_index: int) -> str:
  if not (1 <= int(stage_index) <= len(E2_LAUNCHER_CURRICULUM_PRESET_NAMES)):
    raise ValueError(
      f"E2 curriculum stage must be within [1, {len(E2_LAUNCHER_CURRICULUM_PRESET_NAMES)}], "
      f"got {stage_index}."
    )
  return E2_LAUNCHER_CURRICULUM_PRESET_NAMES[int(stage_index) - 1]


def apply_e2_launcher_preset(
  cfg: "GoalkeeperBallLauncherCfg",
  preset_name: str,
) -> "GoalkeeperBallLauncherCfg":
  preset = get_e2_launcher_preset(preset_name)
  return replace(
    cfg,
    active_preset_name=preset.name,
    family_weights=preset.family_weights,
    delay_range=preset.delay_range,
    t_goal_band=preset.t_goal_band,
    shot_target_mode_probs=preset.shot_target_mode_probs,
    shot_nearpost_abs_y_range=preset.shot_nearpost_abs_y_range,
    shot_farpost_abs_y_range=preset.shot_farpost_abs_y_range,
    shot_center_y_range=preset.shot_center_y_range,
    shot_low_z_range=preset.shot_low_z_range,
    shot_mid_z_range=preset.shot_mid_z_range,
    shot_low_z_prob=preset.shot_low_z_prob,
    lob_tof_range=preset.lob_tof_range,
    ground_time_tiers=_e2_ground_time_tiers(preset.t_goal_band),
    one_bounce_time_tiers=_e2_one_bounce_time_tiers(preset.t_goal_band),
    lob_target_z_range=preset.lob_target_z_range,
    cross_target_x_range=preset.cross_target_x_range,
    cross_driven_tof_range=preset.cross_driven_tof_range,
    cross_lofted_tof_range=preset.cross_lofted_tof_range,
    long_driven_time_tiers=_e2_long_driven_time_tiers(preset.t_goal_band),
    long_driven_target_z_range=preset.long_driven_target_z_range,
    deflection_prob=preset.deflection_prob,
    deflection_dv_mag_range=(
      preset.deflection_dv_mag_range
      if preset.deflection_dv_mag_range is not None
      else cfg.deflection_dv_mag_range
    ),
  )


@dataclass(kw_only=True)
class GoalkeeperLauncherCurriculumManager:
  """Simple no-demotion stage manager for E2 launcher presets.

  Promotion requires both save-rate success and a low fall rate.
  Exploit checks remain a human review item.
  """

  current_stage_index: int = 1
  stage_preset_names: tuple[str, ...] = E2_LAUNCHER_CURRICULUM_PRESET_NAMES
  promotion_save_rate_thresholds: tuple[float, ...] = (0.85, 0.80, 0.70, 0.65)
  max_fall_rate: float = 0.10

  def __post_init__(self) -> None:
    if len(self.stage_preset_names) < 1:
      raise ValueError("stage_preset_names must contain at least one entry.")
    if len(self.promotion_save_rate_thresholds) != len(self.stage_preset_names) - 1:
      raise ValueError(
        "promotion_save_rate_thresholds must have length len(stage_preset_names) - 1."
      )
    if not (1 <= int(self.current_stage_index) <= len(self.stage_preset_names)):
      raise ValueError(
        f"current_stage_index must be within [1, {len(self.stage_preset_names)}], "
        f"got {self.current_stage_index}."
      )

  @property
  def current_preset_name(self) -> str:
    return self.stage_preset_names[self.current_stage_index - 1]

  def maybe_promote(
    self,
    *,
    save_rate: float,
    fall_rate: float,
  ) -> GoalkeeperLauncherPromotionDecision:
    from_stage_index = int(self.current_stage_index)
    from_preset_name = self.current_preset_name

    if from_stage_index >= len(self.stage_preset_names):
      return GoalkeeperLauncherPromotionDecision(
        promoted=False,
        from_stage_index=from_stage_index,
        from_preset_name=from_preset_name,
        to_stage_index=from_stage_index,
        to_preset_name=from_preset_name,
        save_rate=float(save_rate),
        fall_rate=float(fall_rate),
        required_save_rate=None,
        max_fall_rate=float(self.max_fall_rate),
        reason="Already at final curriculum stage. No auto-demotion is supported.",
      )

    required_save_rate = float(
      self.promotion_save_rate_thresholds[from_stage_index - 1]
    )
    if float(save_rate) < required_save_rate:
      return GoalkeeperLauncherPromotionDecision(
        promoted=False,
        from_stage_index=from_stage_index,
        from_preset_name=from_preset_name,
        to_stage_index=from_stage_index,
        to_preset_name=from_preset_name,
        save_rate=float(save_rate),
        fall_rate=float(fall_rate),
        required_save_rate=required_save_rate,
        max_fall_rate=float(self.max_fall_rate),
        reason=(
          f"Save rate {float(save_rate):.3f} is below required threshold "
          f"{required_save_rate:.3f}."
        ),
      )

    if float(fall_rate) > float(self.max_fall_rate):
      return GoalkeeperLauncherPromotionDecision(
        promoted=False,
        from_stage_index=from_stage_index,
        from_preset_name=from_preset_name,
        to_stage_index=from_stage_index,
        to_preset_name=from_preset_name,
        save_rate=float(save_rate),
        fall_rate=float(fall_rate),
        required_save_rate=required_save_rate,
        max_fall_rate=float(self.max_fall_rate),
        reason=(
          f"Fall rate {float(fall_rate):.3f} exceeds max allowed "
          f"{float(self.max_fall_rate):.3f}."
        ),
      )

    to_stage_index = from_stage_index + 1
    to_preset_name = self.stage_preset_names[to_stage_index - 1]
    self.current_stage_index = to_stage_index
    return GoalkeeperLauncherPromotionDecision(
      promoted=True,
      from_stage_index=from_stage_index,
      from_preset_name=from_preset_name,
      to_stage_index=to_stage_index,
      to_preset_name=to_preset_name,
      save_rate=float(save_rate),
      fall_rate=float(fall_rate),
      required_save_rate=required_save_rate,
      max_fall_rate=float(self.max_fall_rate),
      reason=(
        "Promotion approved by save-rate and fall-rate thresholds. "
        "Exploit checks still require human review."
      ),
    )


@dataclass(kw_only=True)
class GoalkeeperBallLauncherCfg:
  """Reusable launcher config for goalkeeper expert tasks.

  Sampling is fully vectorized. All episode decisions are sampled at reset and stored
  in per-env buffers. `step()` applies launch/deflection events when episode time reaches
  their scheduled timestamps.
  """

  ball_entity_name: str = "soccer_ball"
  active_preset_name: str | None = None

  # Goal orientation + aperture.
  goal_toward_positive_x: bool = True
  goal_plane_x: float = 7.0
  goal_y_center: float = 0.0
  goal_y_half: float = 1.35
  goal_z_min: float = 0.0
  goal_z_max: float = 1.90

  # Ball + gravity.
  ball_radius: float = 0.11
  gravity: float = 9.81

  # Global timing / delay.
  delay_range: tuple[float, float] = (0.10, 0.35)
  t_goal_band: tuple[float, float] = (0.35, 1.00)

  # Family enable + weights: (ground, one_bounce, lob_chip, cross, long_driven).
  enabled_families: tuple[bool, bool, bool, bool, bool] = (True, True, True, True, True)
  family_weights: tuple[float, float, float, float, float] = (
    0.50,
    0.15,
    0.15,
    0.10,
    0.10,
  )

  # Launch stability clamps.
  max_speed: float = 8.5
  max_abs_vz: float = 5.5
  min_toward_goal_speed: float = 0.8
  rejection_iters: int = 4

  # Launch application behavior.
  clear_angular_velocity_on_launch: bool = True

  # ---------------- Ground-shot controls ----------------
  # Origin zones: depth (near/far) x lateral channels (center/left/right).
  ground_near_depth_prob: float = 0.55
  ground_near_x_range: tuple[float, float] = (5.4, 6.4)
  ground_far_x_range: tuple[float, float] = (3.8, 5.3)
  ground_center_y_range: tuple[float, float] = (-0.45, 0.45)
  ground_left_y_range: tuple[float, float] = (0.65, 1.85)
  ground_right_y_range: tuple[float, float] = (-1.85, -0.65)
  # (center, left, right)
  ground_channel_probs: tuple[float, float, float] = (0.45, 0.275, 0.275)

  # Goal mouth target distribution for shots (near-post / far-post / center).
  shot_target_mode_probs: tuple[float, float, float] = (0.34, 0.34, 0.32)
  shot_nearpost_abs_y_range: tuple[float, float] = (0.75, 1.20)
  shot_farpost_abs_y_range: tuple[float, float] = (0.95, 1.30)
  shot_center_y_range: tuple[float, float] = (-0.35, 0.35)

  # Optional low/mid target-z buckets (mostly low for E2).
  shot_low_z_range: tuple[float, float] = (0.11, 0.18)
  shot_mid_z_range: tuple[float, float] = (0.22, 0.42)
  shot_low_z_prob: float = 0.85

  # Tiered commit-window timings for ground family: (prob, t_min, t_max).
  ground_time_tiers: tuple[tuple[float, float, float], ...] = (
    (0.30, 0.80, 1.00),
    (0.45, 0.55, 0.80),
    (0.25, 0.35, 0.55),
  )

  # ---------------- One-bounce controls ----------------
  one_bounce_x_range: tuple[float, float] = (4.2, 6.1)
  one_bounce_y_range: tuple[float, float] = (-1.7, 1.7)
  one_bounce_time_tiers: tuple[tuple[float, float, float], ...] = (
    (0.35, 0.70, 0.95),
    (0.45, 0.50, 0.75),
    (0.20, 0.38, 0.55),
  )
  one_bounce_vz_range: tuple[float, float] = (0.9, 2.1)
  # Require first bounce to occur before/near goal (fraction of x distance to goal).
  one_bounce_fraction_range: tuple[float, float] = (0.35, 1.15)

  # ---------------- Lob/chip controls ----------------
  lob_x_range: tuple[float, float] = (4.0, 6.0)
  lob_y_range: tuple[float, float] = (-1.8, 1.8)
  lob_tof_range: tuple[float, float] = (0.55, 1.05)
  lob_target_z_range: tuple[float, float] = (0.75, 1.55)

  # ---------------- Cross controls ----------------
  cross_spawn_x_range: tuple[float, float] = (4.0, 6.0)
  cross_left_spawn_y_range: tuple[float, float] = (2.4, 3.9)
  cross_right_spawn_y_range: tuple[float, float] = (-3.9, -2.4)
  cross_target_x_range: tuple[float, float] = (5.8, 6.95)
  cross_corridor_y_range: tuple[float, float] = (-0.80, 0.80)
  cross_farpost_abs_y_range: tuple[float, float] = (0.95, 1.30)
  cross_farpost_mode_prob: float = 0.60
  cross_driven_prob: float = 0.55
  cross_driven_tof_range: tuple[float, float] = (0.45, 0.75)
  cross_lofted_tof_range: tuple[float, float] = (0.75, 1.20)
  cross_driven_target_z_range: tuple[float, float] = (0.45, 0.90)
  cross_lofted_target_z_range: tuple[float, float] = (0.95, 1.55)

  # ---------------- Long-driven controls ----------------
  # Farther origin than ground-shot, mostly central with mild lateral variation.
  long_driven_x_range: tuple[float, float] = (3.2, 4.9)
  long_driven_center_y_range: tuple[float, float] = (-0.55, 0.55)
  long_driven_left_y_range: tuple[float, float] = (0.70, 1.50)
  long_driven_right_y_range: tuple[float, float] = (-1.50, -0.70)
  # (center, left, right)
  long_driven_channel_probs: tuple[float, float, float] = (0.65, 0.175, 0.175)
  # Faster commit-window timing profile.
  long_driven_time_tiers: tuple[tuple[float, float, float], ...] = (
    (0.25, 0.70, 0.95),
    (0.45, 0.55, 0.75),
    (0.30, 0.42, 0.58),
  )
  # Keep long-driven shots airborne at the goal plane.
  long_driven_target_z_range: tuple[float, float] = (0.80, 1.35)
  # Ensure this family does not collapse into slow long shots.
  long_driven_min_toward_goal_speed: float = 2.0

  # ---------------- Rare deflection modifier ----------------
  deflection_prob: float = 0.06
  deflection_time_after_launch_range: tuple[float, float] = (0.08, 0.22)
  deflection_dv_mag_range: tuple[float, float] = (0.35, 1.25)
  deflection_dv_z_range: tuple[float, float] = (-0.25, 0.35)
  # Positive value biases the deflection away from goal.
  deflection_away_goal_bias: float = 0.25


class GoalkeeperBallLauncher:
  """Centralized vectorized ball launcher for goalkeeper expert tasks."""

  cfg: GoalkeeperBallLauncherCfg

  def __init__(self, cfg: GoalkeeperBallLauncherCfg, env):
    self.cfg = cfg
    self._env = env
    self.device = env.device

    self._ball: Entity = env.scene[cfg.ball_entity_name]

    self.family_id = torch.full(
      (env.num_envs,),
      fill_value=-1,
      device=self.device,
      dtype=torch.long,
    )

    self.spawn_pos_w = torch.zeros(env.num_envs, 3, device=self.device)
    self.target_pos_w = torch.zeros(env.num_envs, 3, device=self.device)
    self.launch_vel_w = torch.zeros(env.num_envs, 3, device=self.device)
    self.launch_time_s = torch.zeros(env.num_envs, device=self.device)
    self.t_goal_est_s = torch.zeros(env.num_envs, device=self.device)

    self.has_launched = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)

    self.has_deflection = torch.zeros(
      env.num_envs, device=self.device, dtype=torch.bool
    )
    self.deflect_time_s = torch.zeros(env.num_envs, device=self.device)
    self.deflect_dv_w = torch.zeros(env.num_envs, 3, device=self.device)
    self.has_deflected = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor) -> None:
    """Sample full launch plan once per reset for selected envs."""
    if env_ids.numel() == 0:
      return

    n = len(env_ids)
    origins = self._env.scene.env_origins[env_ids]

    family = self._sample_family_ids(n)
    self.family_id[env_ids] = family

    spawn_w, target_w, vel_w = self._sample_family_plans(family, origins)
    vel_w = self._clamp_velocity(vel_w)

    # Estimate time-to-goal from x crossing (used for diagnostics and constraints).
    t_goal = self._estimate_time_to_goal(spawn_w[:, 0], vel_w[:, 0])

    self.spawn_pos_w[env_ids] = spawn_w
    self.target_pos_w[env_ids] = target_w
    self.launch_vel_w[env_ids] = vel_w
    self.t_goal_est_s[env_ids] = t_goal

    self.launch_time_s[env_ids] = self._sample_uniform(self.cfg.delay_range, n)
    self.has_launched[env_ids] = False

    has_defl = torch.rand(n, device=self.device) < float(self.cfg.deflection_prob)
    self.has_deflection[env_ids] = has_defl
    self.has_deflected[env_ids] = False

    self.deflect_time_s[env_ids] = self.launch_time_s[env_ids] + self._sample_uniform(
      self.cfg.deflection_time_after_launch_range,
      n,
    )
    self.deflect_dv_w[env_ids] = self._sample_deflection_dv(n)

    # Ball is visible and still during pre-launch delay.
    default_root_state = self._ball.data.default_root_state
    assert default_root_state is not None

    root_state = default_root_state[env_ids].clone()
    root_state[:, :3] = spawn_w
    root_state[:, 3:7] = 0.0
    root_state[:, 3] = 1.0
    root_state[:, 7:13] = 0.0
    self._ball.write_root_state_to_sim(root_state, env_ids=env_ids)
    self._ball.reset(env_ids=env_ids)

  def step(self, time_s: torch.Tensor) -> None:
    """Apply scheduled launch / rare deflection events."""
    # Launch event.
    to_launch = (~self.has_launched) & (time_s >= self.launch_time_s)
    if to_launch.any():
      env_ids = to_launch.nonzero(as_tuple=False).flatten()
      self._set_ball_linear_velocity(
        env_ids,
        self.launch_vel_w[env_ids],
        clear_angular=self.cfg.clear_angular_velocity_on_launch,
      )
      self.has_launched[env_ids] = True

    # Optional rare deflection event.
    to_deflect = (
      self.has_launched
      & self.has_deflection
      & (~self.has_deflected)
      & (time_s >= self.deflect_time_s)
    )
    if to_deflect.any():
      env_ids = to_deflect.nonzero(as_tuple=False).flatten()
      self._add_ball_linear_velocity(env_ids, self.deflect_dv_w[env_ids])
      self.has_deflected[env_ids] = True

  def mode_histogram(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Return per-family counts for the selected envs."""
    if env_ids is None:
      fam = self.family_id
    else:
      fam = self.family_id[env_ids]
    valid = fam >= 0
    if not valid.any():
      return torch.zeros(NUM_LAUNCH_FAMILIES, device=self.device, dtype=torch.long)
    return torch.bincount(fam[valid], minlength=NUM_LAUNCH_FAMILIES)

  def toward_goal_speed(self) -> torch.Tensor:
    if self.cfg.goal_toward_positive_x:
      return self.launch_vel_w[:, 0]
    return -self.launch_vel_w[:, 0]

  def validation_report(
    self, env_ids: torch.Tensor | None = None
  ) -> dict[str, float | int]:
    """Aggregate reset-plan diagnostics for quick validation."""
    if env_ids is None:
      env_ids = torch.arange(self._env.num_envs, device=self.device)

    hist = self.mode_histogram(env_ids)
    vel = self.launch_vel_w[env_ids]
    speed = torch.linalg.norm(vel, dim=1)
    toward = self.toward_goal_speed()[env_ids]
    t_goal = self.t_goal_est_s[env_ids]

    t_lo, t_hi = self.cfg.t_goal_band
    report: dict[str, float | int] = {
      "num_envs": int(env_ids.numel()),
      "ground_shot": int(hist[GROUND_SHOT_FAMILY].item()),
      "one_bounce_shot": int(hist[ONE_BOUNCE_FAMILY].item()),
      "lob_chip": int(hist[LOB_CHIP_FAMILY].item()),
      "cross": int(hist[CROSS_FAMILY].item()),
      "long_driven": int(hist[LONG_DRIVEN_FAMILY].item()),
      "pct_speed_ok": float(
        (speed <= self.cfg.max_speed + 1.0e-4).float().mean().item()
      ),
      "pct_vz_ok": float(
        (torch.abs(vel[:, 2]) <= self.cfg.max_abs_vz + 1.0e-4).float().mean().item()
      ),
      "pct_toward_ok": float(
        (toward >= self.cfg.min_toward_goal_speed - 1.0e-4).float().mean().item()
      ),
      "pct_t_goal_ok": float(
        ((t_goal >= t_lo - 1.0e-4) & (t_goal <= t_hi + 1.0e-4)).float().mean().item()
      ),
      "t_goal_min": float(t_goal.min().item()),
      "t_goal_max": float(t_goal.max().item()),
      "speed_min": float(speed.min().item()),
      "speed_max": float(speed.max().item()),
    }
    return report

  # ---------------------------------------------------------------------------
  # Sampling core
  # ---------------------------------------------------------------------------

  def _sample_family_ids(self, n: int) -> torch.Tensor:
    enabled = torch.tensor(
      self.cfg.enabled_families, device=self.device, dtype=torch.bool
    )
    weights = torch.tensor(
      self.cfg.family_weights, device=self.device, dtype=torch.float32
    )
    weights = torch.where(
      enabled, torch.clamp(weights, min=0.0), torch.zeros_like(weights)
    )

    total = torch.sum(weights)
    if float(total.item()) <= 1.0e-8:
      # Safe fallback: ground shots.
      return torch.full((n,), GROUND_SHOT_FAMILY, device=self.device, dtype=torch.long)

    probs = weights / total
    cdf = torch.cumsum(probs, dim=0)
    u = torch.rand(n, device=self.device)
    family = torch.searchsorted(cdf, u, right=False)
    return torch.clamp(family, min=0, max=NUM_LAUNCH_FAMILIES - 1).to(torch.long)

  def _sample_family_plans(
    self,
    family: torch.Tensor,
    origins: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(family)
    spawn_w = torch.zeros(n, 3, device=self.device)
    target_w = torch.zeros(n, 3, device=self.device)
    vel_w = torch.zeros(n, 3, device=self.device)

    for fam_id in (
      GROUND_SHOT_FAMILY,
      ONE_BOUNCE_FAMILY,
      LOB_CHIP_FAMILY,
      CROSS_FAMILY,
      LONG_DRIVEN_FAMILY,
    ):
      mask = family == fam_id
      if not mask.any():
        continue

      fam_origins = origins[mask]
      if fam_id == GROUND_SHOT_FAMILY:
        s, t, v = self._sample_ground_shot(fam_origins)
      elif fam_id == ONE_BOUNCE_FAMILY:
        s, t, v = self._sample_one_bounce_shot(fam_origins)
      elif fam_id == LOB_CHIP_FAMILY:
        s, t, v = self._sample_lob_chip(fam_origins)
      elif fam_id == CROSS_FAMILY:
        s, t, v = self._sample_cross(fam_origins)
      else:
        s, t, v = self._sample_long_driven(fam_origins)

      spawn_w[mask] = s
      target_w[mask] = t
      vel_w[mask] = v

    return spawn_w, target_w, vel_w

  def _sample_ground_shot(
    self,
    origins: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _sample_once(
      m: int,
      local_origins: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      near_mask = torch.rand(m, device=self.device) < float(
        self.cfg.ground_near_depth_prob
      )
      x_local = torch.where(
        near_mask,
        self._sample_uniform(self.cfg.ground_near_x_range, m),
        self._sample_uniform(self.cfg.ground_far_x_range, m),
      )

      channel = self._sample_categorical(self.cfg.ground_channel_probs, m)
      y_local = torch.zeros(m, device=self.device)
      center_m = channel == 0
      left_m = channel == 1
      right_m = channel == 2
      if center_m.any():
        y_local[center_m] = self._sample_uniform(
          self.cfg.ground_center_y_range, int(center_m.sum().item())
        )
      if left_m.any():
        y_local[left_m] = self._sample_uniform(
          self.cfg.ground_left_y_range, int(left_m.sum().item())
        )
      if right_m.any():
        y_local[right_m] = self._sample_uniform(
          self.cfg.ground_right_y_range, int(right_m.sum().item())
        )

      z_local = torch.full((m,), float(self.cfg.ball_radius), device=self.device)
      spawn_local = torch.stack([x_local, y_local, z_local], dim=1)
      spawn_w = local_origins + spawn_local

      target_local = self._sample_goal_target_local(
        source_y_local=y_local,
        m=m,
        use_mid_z=True,
      )
      target_w = local_origins + target_local

      t_goal = self._sample_tiered(self.cfg.ground_time_tiers, m)
      vel = (target_w - spawn_w) / t_goal.unsqueeze(1)
      return spawn_w, target_w, vel

    return self._sample_with_rejection(origins, _sample_once)

  def _sample_one_bounce_shot(
    self,
    origins: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _sample_once(
      m: int,
      local_origins: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      x_local = self._sample_uniform(self.cfg.one_bounce_x_range, m)
      y_local = self._sample_uniform(self.cfg.one_bounce_y_range, m)
      z_local = torch.full((m,), float(self.cfg.ball_radius), device=self.device)

      spawn_local = torch.stack([x_local, y_local, z_local], dim=1)
      spawn_w = local_origins + spawn_local

      target_local = self._sample_goal_target_local(
        source_y_local=y_local,
        m=m,
        use_mid_z=False,
      )
      # Keep bounce family mostly low toward goal.
      target_local[:, 2] = float(self.cfg.ball_radius)
      target_w = local_origins + target_local

      t_goal = self._sample_tiered(self.cfg.one_bounce_time_tiers, m)
      vel_xy = (target_w[:, :2] - spawn_w[:, :2]) / t_goal.unsqueeze(1)
      vel_z = self._sample_uniform(self.cfg.one_bounce_vz_range, m)
      vel = torch.zeros(m, 3, device=self.device)
      vel[:, :2] = vel_xy
      vel[:, 2] = vel_z
      return spawn_w, target_w, vel

    return self._sample_with_rejection(
      origins,
      _sample_once,
      extra_valid_fn=self._one_bounce_valid,
    )

  def _sample_lob_chip(
    self,
    origins: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _sample_once(
      m: int,
      local_origins: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      x_local = self._sample_uniform(self.cfg.lob_x_range, m)
      y_local = self._sample_uniform(self.cfg.lob_y_range, m)
      z_local = torch.full((m,), float(self.cfg.ball_radius), device=self.device)

      spawn_local = torch.stack([x_local, y_local, z_local], dim=1)
      spawn_w = local_origins + spawn_local

      target_local = self._sample_goal_target_local(
        source_y_local=y_local,
        m=m,
        use_mid_z=False,
      )
      target_z = self._sample_uniform(self.cfg.lob_target_z_range, m)
      z_cap = float(self.cfg.goal_z_max - self.cfg.ball_radius)
      target_local[:, 2] = torch.clamp(target_z, max=z_cap)
      target_w = local_origins + target_local

      tof = self._sample_uniform(self.cfg.lob_tof_range, m)
      vel = self._ballistic_velocity(spawn_w, target_w, tof)
      return spawn_w, target_w, vel

    return self._sample_with_rejection(origins, _sample_once)

  def _sample_cross(
    self,
    origins: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _sample_once(
      m: int,
      local_origins: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      x_local = self._sample_uniform(self.cfg.cross_spawn_x_range, m)
      from_left = torch.rand(m, device=self.device) < 0.5
      y_local = torch.where(
        from_left,
        self._sample_uniform(self.cfg.cross_left_spawn_y_range, m),
        self._sample_uniform(self.cfg.cross_right_spawn_y_range, m),
      )
      z_local = torch.full((m,), float(self.cfg.ball_radius), device=self.device)

      spawn_local = torch.stack([x_local, y_local, z_local], dim=1)
      spawn_w = local_origins + spawn_local

      target_x_local = self._sample_uniform(self.cfg.cross_target_x_range, m)
      farpost_mode = torch.rand(m, device=self.device) < float(
        self.cfg.cross_farpost_mode_prob
      )

      side_sign = torch.sign(y_local)
      zero_side = torch.abs(side_sign) < 1.0e-5
      random_side = torch.where(
        torch.rand(m, device=self.device) < 0.5,
        torch.ones(m, device=self.device),
        -torch.ones(m, device=self.device),
      )
      side_sign = torch.where(zero_side, random_side, side_sign)

      target_y_local = torch.where(
        farpost_mode,
        -side_sign * self._sample_uniform(self.cfg.cross_farpost_abs_y_range, m),
        self._sample_uniform(self.cfg.cross_corridor_y_range, m),
      )

      driven = torch.rand(m, device=self.device) < float(self.cfg.cross_driven_prob)
      tof = torch.where(
        driven,
        self._sample_uniform(self.cfg.cross_driven_tof_range, m),
        self._sample_uniform(self.cfg.cross_lofted_tof_range, m),
      )
      target_z_local = torch.where(
        driven,
        self._sample_uniform(self.cfg.cross_driven_target_z_range, m),
        self._sample_uniform(self.cfg.cross_lofted_target_z_range, m),
      )

      target_local = torch.stack(
        [target_x_local, target_y_local, target_z_local], dim=1
      )
      target_w = local_origins + target_local
      vel = self._ballistic_velocity(spawn_w, target_w, tof)
      return spawn_w, target_w, vel

    return self._sample_with_rejection(origins, _sample_once)

  def _sample_long_driven(
    self,
    origins: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _sample_once(
      m: int,
      local_origins: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      x_local = self._sample_uniform(self.cfg.long_driven_x_range, m)

      channel = self._sample_categorical(self.cfg.long_driven_channel_probs, m)
      y_local = torch.zeros(m, device=self.device)
      center_m = channel == 0
      left_m = channel == 1
      right_m = channel == 2
      if center_m.any():
        y_local[center_m] = self._sample_uniform(
          self.cfg.long_driven_center_y_range,
          int(center_m.sum().item()),
        )
      if left_m.any():
        y_local[left_m] = self._sample_uniform(
          self.cfg.long_driven_left_y_range,
          int(left_m.sum().item()),
        )
      if right_m.any():
        y_local[right_m] = self._sample_uniform(
          self.cfg.long_driven_right_y_range,
          int(right_m.sum().item()),
        )

      z_local = torch.full((m,), float(self.cfg.ball_radius), device=self.device)
      spawn_local = torch.stack([x_local, y_local, z_local], dim=1)
      spawn_w = local_origins + spawn_local

      target_local = self._sample_goal_target_local(
        source_y_local=y_local,
        m=m,
        use_mid_z=True,
      )
      target_z = self._sample_uniform(self.cfg.long_driven_target_z_range, m)
      target_local[:, 2] = torch.clamp(
        target_z,
        min=float(self.cfg.goal_z_min + self.cfg.ball_radius),
        max=float(self.cfg.goal_z_max - self.cfg.ball_radius),
      )
      target_w = local_origins + target_local

      t_goal = self._sample_tiered(self.cfg.long_driven_time_tiers, m)
      vel = self._ballistic_velocity(spawn_w, target_w, t_goal)
      return spawn_w, target_w, vel

    return self._sample_with_rejection(
      origins,
      _sample_once,
      extra_valid_fn=self._long_driven_valid,
    )

  def _sample_with_rejection(
    self,
    origins: torch.Tensor,
    sample_once,
    extra_valid_fn=None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(origins)
    spawn_w, target_w, vel_w = sample_once(n, origins)
    vel_w = self._clamp_velocity(vel_w)
    valid = self._base_valid_mask(spawn_w, vel_w)
    if extra_valid_fn is not None:
      valid = valid & extra_valid_fn(spawn_w, vel_w)

    for _ in range(max(int(self.cfg.rejection_iters), 0)):
      invalid = ~valid
      if not invalid.any():
        break

      invalid_origins = origins[invalid]
      s_i, t_i, v_i = sample_once(len(invalid_origins), invalid_origins)
      v_i = self._clamp_velocity(v_i)
      spawn_w[invalid] = s_i
      target_w[invalid] = t_i
      vel_w[invalid] = v_i

      valid_i = self._base_valid_mask(s_i, v_i)
      if extra_valid_fn is not None:
        valid_i = valid_i & extra_valid_fn(s_i, v_i)
      valid[invalid] = valid_i

    return spawn_w, target_w, vel_w

  # ---------------------------------------------------------------------------
  # Validation / constraints
  # ---------------------------------------------------------------------------

  def _base_valid_mask(
    self, spawn_w: torch.Tensor, vel_w: torch.Tensor
  ) -> torch.Tensor:
    speed = torch.linalg.norm(vel_w, dim=1)
    toward = self._toward_speed(vel_w[:, 0])
    t_goal = self._estimate_time_to_goal(spawn_w[:, 0], vel_w[:, 0])

    t_lo, t_hi = self.cfg.t_goal_band
    return (
      (speed <= float(self.cfg.max_speed))
      & (torch.abs(vel_w[:, 2]) <= float(self.cfg.max_abs_vz))
      & (toward >= float(self.cfg.min_toward_goal_speed))
      & (t_goal >= float(t_lo))
      & (t_goal <= float(t_hi))
    )

  def _one_bounce_valid(
    self, spawn_w: torch.Tensor, vel_w: torch.Tensor
  ) -> torch.Tensor:
    g = float(self.cfg.gravity)
    vz = torch.clamp(vel_w[:, 2], min=0.0)
    t_bounce = 2.0 * vz / max(g, 1.0e-6)

    toward = self._toward_speed(vel_w[:, 0])
    if self.cfg.goal_toward_positive_x:
      dist_to_goal = float(self.cfg.goal_plane_x) - spawn_w[:, 0]
    else:
      dist_to_goal = spawn_w[:, 0] - float(self.cfg.goal_plane_x)
    dist_to_goal = torch.clamp(dist_to_goal, min=1.0e-4)

    frac = (toward * t_bounce) / dist_to_goal
    lo, hi = self.cfg.one_bounce_fraction_range
    return (frac >= float(lo)) & (frac <= float(hi))

  def _long_driven_valid(
    self, spawn_w: torch.Tensor, vel_w: torch.Tensor
  ) -> torch.Tensor:
    del spawn_w
    toward = self._toward_speed(vel_w[:, 0])
    return toward >= float(self.cfg.long_driven_min_toward_goal_speed)

  def _estimate_time_to_goal(
    self, x_w: torch.Tensor, vx_w: torch.Tensor
  ) -> torch.Tensor:
    eps = 1.0e-4
    if self.cfg.goal_toward_positive_x:
      dx = float(self.cfg.goal_plane_x) - x_w
      toward = torch.clamp(vx_w, min=eps)
      valid = (dx >= 0.0) & (vx_w > eps)
    else:
      dx = x_w - float(self.cfg.goal_plane_x)
      toward = torch.clamp(-vx_w, min=eps)
      valid = (dx >= 0.0) & (vx_w < -eps)

    t = dx / toward
    t_hi = max(float(self.cfg.t_goal_band[1]), 1.0) * 4.0
    return torch.where(valid, t, torch.full_like(t, t_hi))

  def _toward_speed(self, vx_w: torch.Tensor) -> torch.Tensor:
    if self.cfg.goal_toward_positive_x:
      return vx_w
    return -vx_w

  def _clamp_velocity(self, vel_w: torch.Tensor) -> torch.Tensor:
    vel = vel_w.clone()

    vel[:, 2] = torch.clamp(
      vel[:, 2],
      min=-float(self.cfg.max_abs_vz),
      max=float(self.cfg.max_abs_vz),
    )

    if self.cfg.goal_toward_positive_x:
      vel[:, 0] = torch.clamp(vel[:, 0], min=float(self.cfg.min_toward_goal_speed))
    else:
      vel[:, 0] = torch.clamp(vel[:, 0], max=-float(self.cfg.min_toward_goal_speed))

    speed = torch.linalg.norm(vel, dim=1, keepdim=True).clamp_min(1.0e-6)
    max_speed = float(self.cfg.max_speed)
    scale = torch.clamp(max_speed / speed, max=1.0)
    vel = vel * scale

    vel[:, 2] = torch.clamp(
      vel[:, 2],
      min=-float(self.cfg.max_abs_vz),
      max=float(self.cfg.max_abs_vz),
    )
    return vel

  # ---------------------------------------------------------------------------
  # Primitive samplers
  # ---------------------------------------------------------------------------

  def _sample_goal_target_local(
    self,
    source_y_local: torch.Tensor,
    m: int,
    use_mid_z: bool,
  ) -> torch.Tensor:
    target_x = torch.full((m,), float(self.cfg.goal_plane_x), device=self.device)

    mode = self._sample_categorical(self.cfg.shot_target_mode_probs, m)

    side_sign = torch.sign(source_y_local)
    near_zero = torch.abs(side_sign) < 1.0e-5
    random_side = torch.where(
      torch.rand(m, device=self.device) < 0.5,
      torch.ones(m, device=self.device),
      -torch.ones(m, device=self.device),
    )
    side_sign = torch.where(near_zero, random_side, side_sign)

    target_y = torch.zeros(m, device=self.device)
    near_m = mode == 0
    far_m = mode == 1
    center_m = mode == 2

    if near_m.any():
      count = int(near_m.sum().item())
      y_abs = self._sample_uniform(self.cfg.shot_nearpost_abs_y_range, count)
      target_y[near_m] = side_sign[near_m] * y_abs

    if far_m.any():
      count = int(far_m.sum().item())
      y_abs = self._sample_uniform(self.cfg.shot_farpost_abs_y_range, count)
      target_y[far_m] = -side_sign[far_m] * y_abs

    if center_m.any():
      target_y[center_m] = self._sample_uniform(
        self.cfg.shot_center_y_range,
        int(center_m.sum().item()),
      )

    target_y = torch.clamp(
      target_y,
      min=float(self.cfg.goal_y_center - self.cfg.goal_y_half),
      max=float(self.cfg.goal_y_center + self.cfg.goal_y_half),
    )

    low_z = self._sample_uniform(self.cfg.shot_low_z_range, m)
    if use_mid_z:
      mid_z = self._sample_uniform(self.cfg.shot_mid_z_range, m)
      low_mask = torch.rand(m, device=self.device) < float(self.cfg.shot_low_z_prob)
      target_z = torch.where(low_mask, low_z, mid_z)
    else:
      target_z = low_z

    target_z = torch.clamp(
      target_z,
      min=float(self.cfg.goal_z_min + self.cfg.ball_radius),
      max=float(self.cfg.goal_z_max - self.cfg.ball_radius),
    )

    return torch.stack([target_x, target_y, target_z], dim=1)

  def _ballistic_velocity(
    self,
    start_w: torch.Tensor,
    target_w: torch.Tensor,
    tof: torch.Tensor,
  ) -> torch.Tensor:
    tof = torch.clamp(tof, min=1.0e-3)
    delta = target_w - start_w

    vel = delta / tof.unsqueeze(1)
    vel[:, 2] = (delta[:, 2] + 0.5 * float(self.cfg.gravity) * torch.square(tof)) / tof
    return vel

  def _sample_deflection_dv(self, n: int) -> torch.Tensor:
    mag = self._sample_uniform(self.cfg.deflection_dv_mag_range, n)
    z_comp = self._sample_uniform(self.cfg.deflection_dv_z_range, n)

    yaw = self._sample_uniform((-torch.pi, torch.pi), n)
    xy_mag = torch.sqrt(torch.clamp(1.0 - z_comp * z_comp, min=1.0e-6))

    dir_x = xy_mag * torch.cos(yaw)
    dir_y = xy_mag * torch.sin(yaw)
    direction = torch.stack([dir_x, dir_y, z_comp], dim=1)

    # Bias away from goal for plausible deflections.
    away_x = -1.0 if self.cfg.goal_toward_positive_x else 1.0
    direction[:, 0] = (
      direction[:, 0] + float(self.cfg.deflection_away_goal_bias) * away_x
    )

    direction = direction / torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(
      1.0e-6
    )
    return direction * mag.unsqueeze(1)

  def _sample_uniform(self, bounds: tuple[float, float], n: int) -> torch.Tensor:
    low, high = bounds
    if high < low:
      low, high = high, low
    return torch.rand(n, device=self.device) * (high - low) + low

  def _sample_tiered(
    self,
    tiers: tuple[tuple[float, float, float], ...],
    n: int,
  ) -> torch.Tensor:
    probs = torch.tensor([max(0.0, t[0]) for t in tiers], device=self.device)
    if float(probs.sum().item()) <= 1.0e-8:
      # Fallback to global band.
      return self._sample_uniform(self.cfg.t_goal_band, n)

    probs = probs / probs.sum()
    cdf = torch.cumsum(probs, dim=0)
    sel = torch.searchsorted(cdf, torch.rand(n, device=self.device), right=False)
    sel = torch.clamp(sel, min=0, max=len(tiers) - 1)

    low = torch.tensor([tiers[i][1] for i in range(len(tiers))], device=self.device)
    high = torch.tensor([tiers[i][2] for i in range(len(tiers))], device=self.device)

    lo_sel = low[sel]
    hi_sel = high[sel]
    return torch.rand(n, device=self.device) * (hi_sel - lo_sel) + lo_sel

  def _sample_categorical(self, probs: tuple[float, ...], n: int) -> torch.Tensor:
    p = torch.tensor([max(0.0, x) for x in probs], device=self.device)
    if float(p.sum().item()) <= 1.0e-8:
      p = torch.ones_like(p) / float(len(probs))
    else:
      p = p / p.sum()
    cdf = torch.cumsum(p, dim=0)
    out = torch.searchsorted(cdf, torch.rand(n, device=self.device), right=False)
    return torch.clamp(out, min=0, max=len(probs) - 1)

  # ---------------------------------------------------------------------------
  # Ball velocity application
  # ---------------------------------------------------------------------------

  def _set_ball_linear_velocity(
    self,
    env_ids: torch.Tensor,
    vel_w_xyz: torch.Tensor,
    *,
    clear_angular: bool,
  ) -> None:
    vel = self._clamp_velocity(vel_w_xyz)
    ball_vel = self._ball.data.root_link_vel_w[env_ids].clone()
    ball_vel[:, :3] = vel
    if clear_angular:
      ball_vel[:, 3:] = 0.0
    self._ball.write_root_link_velocity_to_sim(ball_vel, env_ids=env_ids)

  def _add_ball_linear_velocity(
    self,
    env_ids: torch.Tensor,
    delta_v_w_xyz: torch.Tensor,
  ) -> None:
    ball_vel = self._ball.data.root_link_vel_w[env_ids].clone()
    ball_vel[:, :3] = self._clamp_velocity(ball_vel[:, :3] + delta_v_w_xyz)
    self._ball.write_root_link_velocity_to_sim(ball_vel, env_ids=env_ids)
