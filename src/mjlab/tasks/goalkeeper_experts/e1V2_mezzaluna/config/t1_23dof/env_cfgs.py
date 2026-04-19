from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import mujoco
import torch
import yaml

from mjlab.asset_zoo.robocup_assets.field import get_robocup_field_cfg
from mjlab.asset_zoo.robocup_assets.goalpost import get_robocup_goalpost_cfg
from mjlab.asset_zoo.robots import T1_23_ACTION_SCALE, get_t1_23_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.motor_controller_stage1.latent_action import get_wandb_run_name
from mjlab.tasks.goalkeeper_experts.e1V2_mezzaluna import mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

GOAL_X_LINE = 7
GOALPOST_X = 7.3
# Old coherent E1 convention:
# - home point sits slightly more forward into the field
# - spawn band is centered around that home point, not around the goal line
# This keeps the home-point shaping meaningful at episode start.
KEEPER_HOME_POINT_X = 6.55
KEEPER_HOME_POINT_Y = 0.0
KEEPER_SPAWN_X_RANGE = (6.50, 6.90)
KEEPER_SPAWN_Y_RANGE = (-0.6, 0.6)
# Raw spawn height from crouch_stance_1 tracking policy run k0zgfxdw near the
# start (frame 11/66, ~= 1/6 of the clip), extracted in the MJLab controller.
KEEPER_SPAWN_Z = 0.7
# Root quaternion (w, x, y, z) and joint positions from the same snapshot.
READY_ROOT_QUAT = (0.850692629814148, -0.01908080279827118, 0.048677168786525726, 0.5230569243431091)
READY_JOINT_POS = [
  -0.006449203006923199,   # AAHead_yaw
  -0.06991329044103622,    # Head_pitch
  -0.06792288273572922,    # Left_Shoulder_Pitch
  -1.2981278896331787,     # Left_Shoulder_Roll
   0.5135267376899719,     # Left_Elbow_Pitch
  -0.5423100590705872,     # Left_Elbow_Yaw
  -0.0499301552772522,     # Right_Shoulder_Pitch
   1.2249946594238281,     # Right_Shoulder_Roll
   0.5443910956382751,     # Right_Elbow_Pitch
   0.6937841773033142,     # Right_Elbow_Yaw
   0.07025951147079468,    # Waist
  -0.3184724450111389,     # Left_Hip_Pitch
  -0.03504209965467453,    # Left_Hip_Roll
   0.06286874413490295,    # Left_Hip_Yaw
   0.2646978199481964,     # Left_Knee_Pitch
  -0.04678475856781006,    # Left_Ankle_Pitch
   0.015397579409182072,   # Left_Ankle_Roll
  -0.3520301878452301,     # Right_Hip_Pitch
  -0.004032755270600319,   # Right_Hip_Roll
  -0.08282425254583359,    # Right_Hip_Yaw
   0.4212573170661926,     # Right_Knee_Pitch
  -0.17360135912895203,    # Right_Ankle_Pitch
  -0.040193840861320496,   # Right_Ankle_Roll
]

# Safe keeper area bounds (x_min, x_max, y_min, y_max).
KEEPER_AREA_BOUNDS = (GOAL_X_LINE - 1, GOAL_X_LINE + 0.5, -2, 2)
KEEPER_AREA_HARD_MARGIN = 0.3

# Target ball spawn in absolute world XY.
TARGET_SPAWN_X_RANGE = (-1.0, 4.0)
TARGET_SPAWN_Y_RANGE = (-3.8, 3.8)
# Ball geometry in robocup asset: radius=0.11 -> height=0.22.
TARGET_BALL_HEIGHT = 0.22
# Enforce lower bound: z cannot go below ball_height / 2.
TARGET_BALL_Z_MIN = TARGET_BALL_HEIGHT / 2.0
# Exponential distribution scale (meters):
# smaller -> more concentration near ground, larger -> more high balls.
TARGET_BALL_Z_EXP_SCALE = 0.08
# Cap aerial spawn height (ball center z in world frame).
TARGET_BALL_Z_MAX = 0.35
# Temporary debug: force constant z to verify ground spawning.
DEBUG_FORCE_TARGET_BALL_GROUND_Z = False
DEBUG_TARGET_BALL_GROUND_Z = TARGET_BALL_HEIGHT / 2.0
BALL_FOV_ACTIVE = True
BALL_FOV_HALF_ANGLE_DEG = 50.0

# E1 kick sampler (single kick, lateral-biased, anti-shot).
KICK_DEAD_BALL_PROB = 0.40
KICK_LATERAL_ROLL_PROB = 0.40
KICK_DRIBBLE_PROB = 0.20
KICK_DEAD_BALL_TINY_DRIFT_PROB = 0.20
KICK_DEAD_BALL_DRIFT_SPEED_RANGE = (0.02, 0.10)
KICK_SPEED_RANGE = (0.4, 3.0)
KICK_ANGLE_NOISE_DEG = 75.0
SIDELINE_THROW_SPAWN_X_RANGE = (1.0, 2.0)
SIDELINE_THROW_SPAWN_Y_INTERVALS = ((-4.2, -2.5), (2.5, 4.2))
SIDELINE_THROW_SPEED_RANGE = (0.0, KICK_SPEED_RANGE[1])
SIDELINE_THROW_ANGLE_NOISE_DEG = 3.0
CORNER_THROW_SPAWN_X_RANGE = (5.9, 6.7)
CORNER_THROW_SPAWN_Y_INTERVALS = ((-4.05, -3.55), (3.55, 4.05))
CORNER_THROW_SPEED_RANGE = (1.2, 3.0)
CORNER_THROW_ANGLE_NOISE_DEG = 6.0
CORNER_THROW_TARGET_X_RANGE = (3.5, 5.0)
DRIBBLE_NUM_TAPS_RANGE = (2, 4)
DRIBBLE_TAP_TIME_RANGE = (0.5, 1.4)
DRIBBLE_TAP_INTERVAL_RANGE = (0.16, 0.64)
DRIBBLE_TAP_SPEED_RANGE = (0.2, 0.6)
REBOUND_RELAUNCH_ENABLED = True
REBOUND_ONLY_SIDE_WALLS = True
REBOUND_DELAY_RANGE_S = (0.5, 1.0)
REBOUND_SPEED_RANGE = (0.8, 1.8)
REBOUND_ANGLE_NOISE_DEG = 60.0
REBOUND_INSET_M = 0.15
REBOUND_MAX_EVENTS = 1
MAX_TOWARD_GOAL_VX = 0.5

# Spawn from the ready pose on half of resets; otherwise use default pose.
P_READY = 0.5
E1_STAGE1_KEEPER_SPAWN_X_RANGE = (3.5, 6.8)
E1_STAGE1_KEEPER_SPAWN_Y_RANGE = (-2.5, 2.5)
E1_STAGE2_KEEPER_SPAWN_X_RANGE = (5.2, 6.8)
E1_STAGE2_KEEPER_SPAWN_Y_RANGE = (-2.3, 2.3)
E1_KEEPER_SPAWN_VIS_GROUP = 5
E1_KEEPER_SPAWN_RGBA = (0.10, 0.95, 0.35, 0.16)

# Manual reset curriculum.
E1_DEFAULT_RESET_CURRICULUM_STAGE = 1
# Goalkeeper nominal facing in field coordinates: look out into the field, not at the sampled ball.
KEEPER_NOMINAL_FACING_YAW = math.pi
E1_SPAWN_YAW_RANDOMIZATION_DEG = 80.0

# Reset curriculum for E1V2 mezzaluna.
E1_RESET_STAGE_CFGS = (
  mdp.SetSquareResetStageCfg(
    keeper_spawn_x_range=E1_STAGE1_KEEPER_SPAWN_X_RANGE,
    keeper_spawn_y_range=mdp.IntervalUnionCfg(intervals=(E1_STAGE1_KEEPER_SPAWN_Y_RANGE,)),
    spawn_yaw_offset_range=(
      -math.radians(E1_SPAWN_YAW_RANDOMIZATION_DEG),
      math.radians(E1_SPAWN_YAW_RANDOMIZATION_DEG),
    ),
    target_spawn_x_range=TARGET_SPAWN_X_RANGE,
    target_spawn_y_range=mdp.IntervalUnionCfg(intervals=(TARGET_SPAWN_Y_RANGE,)),
    launcher_mode_probs=(0.3, 0.3, 0.2, 0.2, 0.0),
  ),
  mdp.SetSquareResetStageCfg(
    keeper_spawn_x_range=E1_STAGE2_KEEPER_SPAWN_X_RANGE,
    keeper_spawn_y_range=mdp.IntervalUnionCfg(intervals=(E1_STAGE2_KEEPER_SPAWN_Y_RANGE,)),
    spawn_yaw_offset_range=(
      -math.radians(E1_SPAWN_YAW_RANDOMIZATION_DEG),
      math.radians(E1_SPAWN_YAW_RANDOMIZATION_DEG),
    ),
    target_spawn_x_range=TARGET_SPAWN_X_RANGE,
    target_spawn_y_range=mdp.IntervalUnionCfg(intervals=(TARGET_SPAWN_Y_RANGE,)),
    launcher_mode_probs=(0.1, 0.35, 0.2, 0.2, 0.15),
  ),
  mdp.SetSquareResetStageCfg(
    keeper_spawn_x_range=E1_STAGE2_KEEPER_SPAWN_X_RANGE,
    keeper_spawn_y_range=mdp.IntervalUnionCfg(intervals=(E1_STAGE2_KEEPER_SPAWN_Y_RANGE,)),
    spawn_yaw_offset_range=(
      -math.radians(E1_SPAWN_YAW_RANDOMIZATION_DEG),
      math.radians(E1_SPAWN_YAW_RANDOMIZATION_DEG),
    ),
    target_spawn_x_range=TARGET_SPAWN_X_RANGE,
    target_spawn_y_range=mdp.IntervalUnionCfg(intervals=(TARGET_SPAWN_Y_RANGE,)),
    launcher_mode_probs=(0.0, 0.0, 0.0, 0.0, 1.0),
  ),
)

# Upright posture shaping (anisotropic): strict lateral, tolerant sagittal with slight forward lean.
UPRIGHT_ROLL_BAND = 0.1
UPRIGHT_ROLL_SIGMA = 0.12
UPRIGHT_PITCH_TARGET = 0.25
UPRIGHT_PITCH_BAND = 0.20
UPRIGHT_PITCH_SIGMA = 0.30

# Stage-1 command dimension used in motor-observation layout.
MOTOR_COMMAND_DIM = 46
# IMPORTANT:
#   Stage-1 decoder expects joint/action dims over actuated joints (23 for T1_23),
#   not the number of regex groups in T1_23_ACTION_SCALE.
MOTOR_ACT_DIM = 23

EPISODE_LENGTH_S = 4.0
SIM_TIMESTEP_S = 0.005
CONTROL_DECIMATION = 4
E1_TASK_ID = "Mjlab-GK-Expert-E1V2-Mezzaluna-Booster-T1_23"
E1_EXPERIMENT_NAME = "gk_expert_e1V2_mezzaluna_booster_t1_23"

# E1 test walls around the real 14x9 playable area:
# - 2 continuous walls on long sides (y = +/- 4.5),
# - 2 segmented walls per short side (x = +/- 7.0) leaving goal opening at y ~= 0.
# Walls are centered on this boundary.
FIELD_HALF_LENGTH_X = 7.0
FIELD_HALF_WIDTH_Y = 4.5
E1_WALL_THICKNESS = 0.16
E1_WALL_HEIGHT = 0.07
E1_GOAL_OPENING_HALF_WIDTH = 1.55
E1_WALL_GOALPOST_CORNER_CLEARANCE = 0.25
E1_WALL_RGBA = (0.92, 0.18, 0.18, 0.45)
E1_WALL_FRICTION = (1.2, 0.02, 0.002)
E1_WALL_SOLREF = (0.02, 1.5)
E1_WALL_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
E1_AREA_OVERLAY_HALF_THICKNESS = 0.0015
E1_HARD_AREA_OVERLAY_Z = 0.0015
E1_KEEPER_AREA_OVERLAY_Z = 0.0035
E1_HARD_AREA_RGBA = (0.95, 0.55, 0.10, 0.22)
E1_KEEPER_AREA_RGBA = (0.05, 0.60, 0.95, 0.30)
E1_MEZZALUNA_VIS_GROUP = 2
E1_MEZZALUNA_RGBA = (0.56, 0.95, 0.62, 0.80)
E1_MEZZALUNA_HALF_WIDTH = 0.02
E1_MEZZALUNA_HALF_THICKNESS = 0.003
E1_MEZZALUNA_Z = 0.006
E1_MEZZALUNA_SEGMENTS = 40
E1_MEZZALUNA_APEX_GOAL_OFFSET_M = 0.15
E1_MEZZALUNA_CENTER_X = GOAL_X_LINE - 0.20
E1_MEZZALUNA_CENTER_Y = 0.0
E1_MEZZALUNA_APEX_X = KEEPER_AREA_BOUNDS[0] + E1_MEZZALUNA_APEX_GOAL_OFFSET_M - 0.20
E1_MEZZALUNA_HALF_WIDTH_Y = E1_GOAL_OPENING_HALF_WIDTH + 0.10
BALL_CURB_CONTACT_SENSOR_NAME = "ball_curb_contact"
LEFT_FOOT_GROUND_CONTACT_SENSOR_NAME = "left_foot_ground_contact"
RIGHT_FOOT_GROUND_CONTACT_SENSOR_NAME = "right_foot_ground_contact"
STANCE_ORTHO_LEFT_FOOT_BODY = r"^left_foot_link$"
STANCE_ORTHO_RIGHT_FOOT_BODY = r"^right_foot_link$"
STANCE_ORTHO_W_MIN = 0.10
STANCE_ORTHO_D_MIN = 0.20
WAIST_BODY_NAME_REGEX = r"(?i)^waist$"
# Shared home-point band used by reward shaping.
HOME_POINT_BAND_RADIUS = 0.20


def _get_cli_flag_value(flag: str) -> str | None:
  flag_eq = f"{flag}="
  argv = sys.argv[1:]
  for index, arg in enumerate(argv):
    if arg == flag and index + 1 < len(argv):
      return argv[index + 1]
    if arg.startswith(flag_eq):
      return arg[len(flag_eq) :]
  return None


def _is_e1_play_cli_invocation() -> bool:
  script_stem = Path(sys.argv[0]).stem.lower()
  if "play" not in script_stem:
    return False
  return len(sys.argv) > 1 and sys.argv[1] == E1_TASK_ID


def _download_wandb_run_file(log_root: Path, run_path: str, filename: str) -> Path:
  import wandb

  run_id = run_path.split("/")[-1]
  download_dir = log_root / "wandb_checkpoints" / run_id
  target = download_dir / filename
  if target.exists():
    return target

  download_dir.mkdir(parents=True, exist_ok=True)

  api = wandb.Api()
  run = api.run(run_path)
  files = {f.name for f in run.files()}
  if filename not in files:
    raise FileNotFoundError(
      f"Required file '{filename}' not found in W&B run {run_path}."
    )
  run.file(filename).download(str(download_dir), replace=True)
  return target


def _try_download_wandb_run_file(
  log_root: Path, run_path: str, filename: str
) -> Path | None:
  try:
    return _download_wandb_run_file(log_root, run_path, filename)
  except FileNotFoundError:
    return None


def _extract_saved_e1_curriculum_stage(env_yaml_path: Path) -> int | None:
  with env_yaml_path.open("r", encoding="utf-8") as handle:
    env_data = yaml.safe_load(handle) or {}

  return _extract_e1_curriculum_stage_from_env_data(env_data)


def _extract_saved_e1_stage1_run_from_env_data(
  env_data: object,
) -> tuple[str | None, str | None]:
  if not isinstance(env_data, dict):
    return None, None
  actions = env_data.get("actions")
  if not isinstance(actions, dict):
    return None, None
  motor_latent_cfg = actions.get("motor_latent")
  if not isinstance(motor_latent_cfg, dict):
    return None, None

  run_path = motor_latent_cfg.get("stage1_wandb_run_path")
  run_name = motor_latent_cfg.get("stage1_wandb_run_name")
  resolved_run_path = run_path.strip() if isinstance(run_path, str) else None
  resolved_run_name = run_name.strip() if isinstance(run_name, str) else None
  return resolved_run_path or None, resolved_run_name or None


def _normalize_e1_curriculum_stage(stage_index: int) -> int | None:
  if 1 <= stage_index <= len(E1_RESET_STAGE_CFGS):
    return stage_index
  return None


def _is_e1_fov_enabled(curriculum_stage: int) -> bool:
  return bool(BALL_FOV_ACTIVE) and int(curriculum_stage) > 1


def _get_e1_reset_curriculum_stage_env() -> str:
  return os.environ.get("MJLAB_E1_RESET_CURRICULUM_STAGE", "").strip()


def _extract_e1_curriculum_stage_from_env_data(env_data: object) -> int | None:
  if not isinstance(env_data, dict):
    return None
  commands = env_data.get("commands")
  if not isinstance(commands, dict):
    return None
  set_square_cfg = commands.get("set_square")
  if not isinstance(set_square_cfg, dict):
    return None

  stage_value = set_square_cfg.get("curriculum_stage")
  try:
    stage_index = int(stage_value)
  except (TypeError, ValueError):
    return None

  return _normalize_e1_curriculum_stage(stage_index)


def _extract_saved_e1_curriculum_stage_from_wandb_config(
  config_yaml_path: Path,
) -> int | None:
  with config_yaml_path.open("r", encoding="utf-8") as handle:
    config_data = yaml.safe_load(handle) or {}

  if not isinstance(config_data, dict):
    return None
  env_cfg = config_data.get("env_cfg")
  if not isinstance(env_cfg, dict):
    return None
  env_cfg_value = env_cfg.get("value")
  return _extract_e1_curriculum_stage_from_env_data(env_cfg_value)


def _extract_saved_e1_stage1_run_from_wandb_config(
  config_yaml_path: Path,
) -> tuple[str | None, str | None]:
  with config_yaml_path.open("r", encoding="utf-8") as handle:
    config_data = yaml.safe_load(handle) or {}

  if not isinstance(config_data, dict):
    return None, None
  env_cfg = config_data.get("env_cfg")
  if not isinstance(env_cfg, dict):
    return None, None
  env_cfg_value = env_cfg.get("value")
  return _extract_saved_e1_stage1_run_from_env_data(env_cfg_value)


def _resolve_saved_e1_play_curriculum_stage() -> int | None:
  if not _is_e1_play_cli_invocation():
    return None

  checkpoint_file = _get_cli_flag_value("--checkpoint-file")
  if checkpoint_file:
    env_yaml_path = (
      Path(checkpoint_file).expanduser().resolve().parent / "params" / "env.yaml"
    )
    if not env_yaml_path.exists():
      print(
        "[WARN]: Saved E1 env config was not found next to the checkpoint; "
        "using the default play stage."
      )
      return None
    return _extract_saved_e1_curriculum_stage(env_yaml_path)

  wandb_run_path = _get_cli_flag_value("--wandb-run-path")
  if not wandb_run_path:
    return None

  log_root = (Path("logs") / "rsl_rl" / E1_EXPERIMENT_NAME).resolve()
  config_yaml_path = _try_download_wandb_run_file(
    log_root, wandb_run_path, "config.yaml"
  )
  if config_yaml_path is not None:
    stage_index = _extract_saved_e1_curriculum_stage_from_wandb_config(
      config_yaml_path
    )
    if stage_index is not None:
      return stage_index
    print(
      "[WARN]: E1 curriculum stage was not found in the W&B config; "
      "falling back to the saved env config if available."
    )

  env_yaml_path = _try_download_wandb_run_file(
    log_root, wandb_run_path, "params/env.yaml"
  )
  if env_yaml_path is None:
    print(
      "[WARN]: Saved E1 env config was not found in the W&B run; "
      "using the default play stage."
    )
    return None
  return _extract_saved_e1_curriculum_stage(env_yaml_path)


def _resolve_saved_e1_play_stage1_run() -> tuple[str | None, str | None]:
  if not _is_e1_play_cli_invocation():
    return None, None

  checkpoint_file = _get_cli_flag_value("--checkpoint-file")
  if checkpoint_file:
    env_yaml_path = (
      Path(checkpoint_file).expanduser().resolve().parent / "params" / "env.yaml"
    )
    if not env_yaml_path.exists():
      return None, None
    with env_yaml_path.open("r", encoding="utf-8") as handle:
      env_data = yaml.safe_load(handle) or {}
    return _extract_saved_e1_stage1_run_from_env_data(env_data)

  wandb_run_path = _get_cli_flag_value("--wandb-run-path")
  if not wandb_run_path:
    return None, None

  log_root = (Path("logs") / "rsl_rl" / E1_EXPERIMENT_NAME).resolve()
  config_yaml_path = _try_download_wandb_run_file(
    log_root, wandb_run_path, "config.yaml"
  )
  if config_yaml_path is not None:
    run_path, run_name = _extract_saved_e1_stage1_run_from_wandb_config(
      config_yaml_path
    )
    if run_path is not None:
      return run_path, run_name

  env_yaml_path = _try_download_wandb_run_file(
    log_root, wandb_run_path, "params/env.yaml"
  )
  if env_yaml_path is None:
    return None, None
  with env_yaml_path.open("r", encoding="utf-8") as handle:
    env_data = yaml.safe_load(handle) or {}
  return _extract_saved_e1_stage1_run_from_env_data(env_data)


def _add_e1_test_walls(
  spec: mujoco.MjSpec,
  *,
  curriculum_stage: int,
) -> None:
  overlay_body = next(
    (body for body in spec.bodies if body.name == "e1_field_overlays"), None
  )
  if overlay_body is None:
    overlay_body = spec.worldbody.add_body(name="e1_field_overlays")

  half_t = E1_WALL_THICKNESS / 2.0
  half_h = E1_WALL_HEIGHT / 2.0
  wall_z = half_h

  def _add_wall(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
  ) -> None:
    wall = overlay_body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=size,
    )
    wall.name = name
    wall.rgba = E1_WALL_RGBA
    wall.friction = E1_WALL_FRICTION
    wall.solref = E1_WALL_SOLREF
    wall.solimp = E1_WALL_SOLIMP
    # Ball-only collider: the RoboCup ball has default contype=1/conaffinity=1,
    # while the T1 robot collision geoms use contype=0/conaffinity=1.
    wall.contype = 0
    wall.conaffinity = 1

  # Long sides (unchanged, continuous walls).
  long_side_wall_y = FIELD_HALF_WIDTH_Y
  _add_wall(
    "e1_wall_long_pos_y",
    (0.0, long_side_wall_y, wall_z),
    (FIELD_HALF_LENGTH_X, half_t, half_h),
  )
  _add_wall(
    "e1_wall_long_neg_y",
    (0.0, -long_side_wall_y, wall_z),
    (FIELD_HALF_LENGTH_X, half_t, half_h),
  )

  # Short sides split in two per side, leaving opening for goalpost.
  short_side_opening_half_width = (
    E1_GOAL_OPENING_HALF_WIDTH + E1_WALL_GOALPOST_CORNER_CLEARANCE
  )
  short_side_segment_half_y = (FIELD_HALF_WIDTH_Y - short_side_opening_half_width) / 2.0
  if short_side_segment_half_y <= 0.0:
    raise ValueError("Short-side wall opening is too large for field width.")
  short_side_segment_center_y = short_side_opening_half_width + short_side_segment_half_y
  short_side_wall_x = FIELD_HALF_LENGTH_X

  _add_wall(
    "e1_wall_short_pos_x_upper",
    (short_side_wall_x, short_side_segment_center_y, wall_z),
    (half_t, short_side_segment_half_y, half_h),
  )
  _add_wall(
    "e1_wall_short_pos_x_lower",
    (short_side_wall_x, -short_side_segment_center_y, wall_z),
    (half_t, short_side_segment_half_y, half_h),
  )
  _add_wall(
    "e1_wall_short_neg_x_upper",
    (-short_side_wall_x, short_side_segment_center_y, wall_z),
    (half_t, short_side_segment_half_y, half_h),
  )
  _add_wall(
    "e1_wall_short_neg_x_lower",
    (-short_side_wall_x, -short_side_segment_center_y, wall_z),
    (half_t, short_side_segment_half_y, half_h),
  )

  keeper_front_x = float(E1_MEZZALUNA_APEX_X)
  goal_opening_half_y = float(E1_MEZZALUNA_HALF_WIDTH_Y)
  ellipse_a = max(float(E1_MEZZALUNA_CENTER_X) - keeper_front_x, 1.0e-6)
  ellipse_b = max(goal_opening_half_y, 1.0e-6)
  theta = torch.linspace(
    -0.5 * math.pi,
    0.5 * math.pi,
    steps=int(E1_MEZZALUNA_SEGMENTS) + 1,
    dtype=torch.float32,
  )
  x = float(E1_MEZZALUNA_CENTER_X) - ellipse_a * torch.cos(theta)
  y = float(E1_MEZZALUNA_CENTER_Y) + ellipse_b * torch.sin(theta)

  for index in range(int(E1_MEZZALUNA_SEGMENTS)):
    p0 = torch.tensor([x[index], y[index]], dtype=torch.float32)
    p1 = torch.tensor([x[index + 1], y[index + 1]], dtype=torch.float32)
    center = 0.5 * (p0 + p1)
    delta = p1 - p0
    half_len = max(0.5 * torch.linalg.norm(delta).item(), 1.0e-6)
    angle = math.atan2(float(delta[1].item()), float(delta[0].item()))
    segment = overlay_body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=(float(center[0].item()), float(center[1].item()), E1_MEZZALUNA_Z),
      size=(half_len, E1_MEZZALUNA_HALF_WIDTH, E1_MEZZALUNA_HALF_THICKNESS),
      quat=(math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)),
    )
    segment.name = f"e1v2_mezzaluna_segment_{index:02d}"
    segment.rgba = E1_MEZZALUNA_RGBA
    segment.group = E1_MEZZALUNA_VIS_GROUP
    segment.contype = 0
    segment.conaffinity = 0

  stage_cfg = E1_RESET_STAGE_CFGS[int(curriculum_stage) - 1]
  spawn_x_min, spawn_x_max = stage_cfg.keeper_spawn_x_range
  for interval_index, (spawn_y_min, spawn_y_max) in enumerate(
    stage_cfg.keeper_spawn_y_range.intervals
  ):
    spawn_overlay = overlay_body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=(
        0.5 * (float(spawn_x_min) + float(spawn_x_max)),
        0.5 * (float(spawn_y_min) + float(spawn_y_max)),
        E1_KEEPER_AREA_OVERLAY_Z,
      ),
      size=(
        0.5 * (float(spawn_x_max) - float(spawn_x_min)),
        0.5 * (float(spawn_y_max) - float(spawn_y_min)),
        E1_AREA_OVERLAY_HALF_THICKNESS,
      ),
    )
    spawn_overlay.name = (
      f"e1_stage{int(curriculum_stage)}_keeper_spawn_overlay_{interval_index:02d}"
    )
    spawn_overlay.rgba = E1_KEEPER_SPAWN_RGBA
    spawn_overlay.group = E1_KEEPER_SPAWN_VIS_GROUP
    spawn_overlay.contype = 0
    spawn_overlay.conaffinity = 0


def get_e1_field_cfg_with_test_walls(curriculum_stage: int) -> EntityCfg:
  field_cfg = get_robocup_field_cfg()
  base_spec_fn = field_cfg.spec_fn
  resolved_stage = _normalize_e1_curriculum_stage(int(curriculum_stage))
  if resolved_stage is None:
    raise ValueError(
      f"curriculum_stage must be within [1, {len(E1_RESET_STAGE_CFGS)}], got {curriculum_stage}."
    )

  def _spec_fn() -> mujoco.MjSpec:
    spec = base_spec_fn()
    _add_e1_test_walls(spec, curriculum_stage=resolved_stage)
    return spec

  field_cfg.spec_fn = _spec_fn
  return field_cfg


def _disable_e1_ball_robot_collision(spec: mujoco.MjSpec) -> None:
  # Scene.attach prefixes body names with "<entity_name>/", so exclude the
  # ball body against the robot root body tree at the combined scene-spec level.
  spec.add_exclude(name="e1_ball_robot_exclude", bodyname1="robot/Trunk", bodyname2="soccer_ball/ball")


def booster_t1_23_gk_expert_e1V2_mezzaluna_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  curriculum_stage = E1_DEFAULT_RESET_CURRICULUM_STAGE
  env_stage_raw = _get_e1_reset_curriculum_stage_env()
  if play:
    saved_stage = _resolve_saved_e1_play_curriculum_stage()
    if saved_stage is not None:
      curriculum_stage = saved_stage
      print(
        f"[INFO]: Auto-selected E1 play curriculum stage from saved run: "
        f"{curriculum_stage}"
      )
    elif env_stage_raw:
      env_stage = _normalize_e1_curriculum_stage(int(env_stage_raw))
      if env_stage is None:
        raise ValueError(
          f"MJLAB_E1_RESET_CURRICULUM_STAGE must be within [1, {len(E1_RESET_STAGE_CFGS)}]."
        )
      curriculum_stage = env_stage
  elif env_stage_raw:
    env_stage = _normalize_e1_curriculum_stage(int(env_stage_raw))
    if env_stage is None:
      raise ValueError(
        f"MJLAB_E1_RESET_CURRICULUM_STAGE must be within [1, {len(E1_RESET_STAGE_CFGS)}]."
      )
    curriculum_stage = env_stage

  robot_cfg = get_t1_23_robot_cfg()
  # Keep default keyframe pose but place initial robot near goal.
  robot_cfg.init_state.pos = (GOAL_X_LINE - 0.3, 0.0, KEEPER_SPAWN_Z)
  # Rotate keeper by 180 deg around z so it faces the opposite field side.
  robot_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)

  goal_left_cfg = get_robocup_goalpost_cfg()
  goal_left_cfg.init_state.pos = (GOALPOST_X, 0.0, 0.0)
  goal_left_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)

  goal_right_cfg = get_robocup_goalpost_cfg()
  goal_right_cfg.init_state.pos = (-GOALPOST_X, 0.0, 0.0)
  goal_right_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

  soccer_ball_cfg = mdp.get_target_ball_cfg()

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="plane",
  )
  cfg.scene.num_envs = 4096 if not play else 1
  cfg.scene.entities = {
    "robot": robot_cfg,
    "soccer_field": get_e1_field_cfg_with_test_walls(curriculum_stage),
    "goalpost_left": goal_left_cfg,
    "goalpost_right": goal_right_cfg,
    "soccer_ball": soccer_ball_cfg,
  }
  cfg.sim.mujoco.ccd_iterations = 100
  ball_curb_contact_cfg = ContactSensorCfg(
    name=BALL_CURB_CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern="e1_wall_.*", entity="soccer_field"),
    secondary=ContactMatch(mode="geom", pattern="ball_collision", entity="soccer_ball"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  left_foot_ground_contact_cfg = ContactSensorCfg(
    name=LEFT_FOOT_GROUND_CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="body", pattern=STANCE_ORTHO_LEFT_FOOT_BODY, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  right_foot_ground_contact_cfg = ContactSensorCfg(
    name=RIGHT_FOOT_GROUND_CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="body", pattern=STANCE_ORTHO_RIGHT_FOOT_BODY, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ball_curb_contact_cfg,
    left_foot_ground_contact_cfg,
    right_foot_ground_contact_cfg,
  )
  cfg.scene.spec_fn = _disable_e1_ball_robot_collision

  motor_obs_terms, motor_obs_term_dims = mdp.default_motor_obs_layout(
    act_dim=MOTOR_ACT_DIM,
  )
  stage1_goalkeeper_run_path = os.environ.get("MJLAB_STAGE1_WANDB_RUN_PATH_GOALKEEPER")
  stage1_goalkeeper_run_name = (
    get_wandb_run_name(stage1_goalkeeper_run_path)
    if stage1_goalkeeper_run_path
    else None
  )
  if play:
    saved_stage1_run_path, saved_stage1_run_name = _resolve_saved_e1_play_stage1_run()
    if saved_stage1_run_path is not None:
      env_stage1_run_name = stage1_goalkeeper_run_name
      resolved_saved_stage1_run_name = (
        saved_stage1_run_name or get_wandb_run_name(saved_stage1_run_path)
      )
      print(
        "[INFO]: Auto-selected E1 play Stage-1 controller from saved run: "
        f"{resolved_saved_stage1_run_name or '<unknown>'} "
        f"({saved_stage1_run_path})"
      )
      if (
        stage1_goalkeeper_run_path is not None
        and stage1_goalkeeper_run_path != saved_stage1_run_path
      ):
        print(
          "[WARN]: E1 play saved Stage-1 controller run differs from "
          "MJLAB_STAGE1_WANDB_RUN_PATH_GOALKEEPER "
          f"(saved={resolved_saved_stage1_run_name or '<unknown>'} "
          f"[{saved_stage1_run_path}], env={env_stage1_run_name or '<unknown>'} "
          f"[{stage1_goalkeeper_run_path}]). "
          "Using the saved run."
        )
      stage1_goalkeeper_run_path = saved_stage1_run_path
      stage1_goalkeeper_run_name = resolved_saved_stage1_run_name

  cfg.actions = {
    "motor_latent": mdp.MotorLatentActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=T1_23_ACTION_SCALE,
      use_default_offset=True,
      command_name="set_square",
      stage1_wandb_run_path=stage1_goalkeeper_run_path,
      stage1_wandb_run_name=stage1_goalkeeper_run_name,
      motor_obs_terms=motor_obs_terms,
      motor_obs_term_dims=motor_obs_term_dims,
      strict_obs_layout=True,
    )
  }

  cfg.commands = {
    "set_square": mdp.SetSquareCommandCfg(
      entity_name="robot",
      ball_entity_name="soccer_ball",
      ball_curb_sensor_name=BALL_CURB_CONTACT_SENSOR_NAME,
      command_dim=MOTOR_COMMAND_DIM,
      keeper_spawn_x_range=KEEPER_SPAWN_X_RANGE,
      keeper_spawn_y_range=KEEPER_SPAWN_Y_RANGE,
      keeper_area_bounds=KEEPER_AREA_BOUNDS,
      hard_area_margin=KEEPER_AREA_HARD_MARGIN,
      mezzaluna_center_x=E1_MEZZALUNA_CENTER_X,
      mezzaluna_center_y=E1_MEZZALUNA_CENTER_Y,
      mezzaluna_apex_x=E1_MEZZALUNA_APEX_X,
      mezzaluna_half_width_y=E1_MEZZALUNA_HALF_WIDTH_Y,
      target_spawn_x_range=TARGET_SPAWN_X_RANGE,
      target_spawn_y_range=TARGET_SPAWN_Y_RANGE,
      target_height_min=TARGET_BALL_Z_MIN,
      target_height_exp_scale=TARGET_BALL_Z_EXP_SCALE,
      target_height_max=TARGET_BALL_Z_MAX,
      debug_force_target_ground_z=DEBUG_FORCE_TARGET_BALL_GROUND_Z,
      debug_target_ground_z=DEBUG_TARGET_BALL_GROUND_Z,
      dead_ball_prob=KICK_DEAD_BALL_PROB,
      lateral_roll_prob=KICK_LATERAL_ROLL_PROB,
      dead_ball_tiny_drift_prob=KICK_DEAD_BALL_TINY_DRIFT_PROB,
      dead_ball_drift_speed_range=KICK_DEAD_BALL_DRIFT_SPEED_RANGE,
      kick_speed_range=KICK_SPEED_RANGE,
      kick_angle_noise_deg=KICK_ANGLE_NOISE_DEG,
      sideline_throw_spawn_x_range=SIDELINE_THROW_SPAWN_X_RANGE,
      sideline_throw_spawn_y_range=mdp.IntervalUnionCfg(intervals=SIDELINE_THROW_SPAWN_Y_INTERVALS),
      sideline_throw_speed_range=SIDELINE_THROW_SPEED_RANGE,
      sideline_throw_angle_noise_deg=SIDELINE_THROW_ANGLE_NOISE_DEG,
      corner_throw_spawn_x_range=CORNER_THROW_SPAWN_X_RANGE,
      corner_throw_spawn_y_range=mdp.IntervalUnionCfg(intervals=CORNER_THROW_SPAWN_Y_INTERVALS),
      corner_throw_speed_range=CORNER_THROW_SPEED_RANGE,
      corner_throw_angle_noise_deg=CORNER_THROW_ANGLE_NOISE_DEG,
      corner_throw_target_x_range=CORNER_THROW_TARGET_X_RANGE,
      dribble_num_taps_range=DRIBBLE_NUM_TAPS_RANGE,
      dribble_tap_time_range=DRIBBLE_TAP_TIME_RANGE,
      dribble_tap_interval_range=DRIBBLE_TAP_INTERVAL_RANGE,
      dribble_tap_speed_range=DRIBBLE_TAP_SPEED_RANGE,
      rebound_relaunch_enabled=REBOUND_RELAUNCH_ENABLED,
      rebound_only_side_walls=REBOUND_ONLY_SIDE_WALLS,
      rebound_delay_range_s=REBOUND_DELAY_RANGE_S,
      rebound_speed_range=REBOUND_SPEED_RANGE,
      rebound_angle_noise_deg=REBOUND_ANGLE_NOISE_DEG,
      rebound_inset_m=REBOUND_INSET_M,
      rebound_max_events=REBOUND_MAX_EVENTS,
      field_half_width_y=FIELD_HALF_WIDTH_Y,
      goal_toward_positive_x=True,
      max_toward_goal_speed=MAX_TOWARD_GOAL_VX,
      p_ready=P_READY,
      home_point_x=KEEPER_HOME_POINT_X,
      home_point_y=KEEPER_HOME_POINT_Y,
      goal_line_x=GOAL_X_LINE,
      goal_line_y_center=0.0,
      curriculum_stage=curriculum_stage,
      curriculum_stages=E1_RESET_STAGE_CFGS,
      nominal_keeper_facing_yaw=KEEPER_NOMINAL_FACING_YAW,
      stance_left_foot_body_name=STANCE_ORTHO_LEFT_FOOT_BODY,
      stance_right_foot_body_name=STANCE_ORTHO_RIGHT_FOOT_BODY,
      stance_ortho_w_min=STANCE_ORTHO_W_MIN,
      stance_ortho_d_min=STANCE_ORTHO_D_MIN,
      fov_active=_is_e1_fov_enabled(curriculum_stage),
      ball_fov_half_angle_deg=BALL_FOV_HALF_ANGLE_DEG,
      viz=mdp.SetSquareCommandCfg.VizCfg(home_point_radius=HOME_POINT_BAND_RADIUS),
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
    )
  }

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "decoded_actions": ObservationTermCfg(
      func=mdp.motor_last_decoded_action,
      params={"action_name": "motor_latent"},
    ),
    "target_dir_xy": ObservationTermCfg(
      func=mdp.visible_target_direction_xy,
      params={"command_name": "set_square"},
    ),
    "robot_pos_rel_goal_line_xy": ObservationTermCfg(
      func=mdp.robot_position_relative_goal_line_xy,
      params={"command_name": "set_square"},
    ),
    "desired_point_rel_xy": ObservationTermCfg(
      func=mdp.visible_desired_point_relative_xy,
      params={"command_name": "set_square"},
    ),
    "ball_visible": ObservationTermCfg(
      func=mdp.ball_visible,
      params={"command_name": "set_square"},
    ),
    "visible_ball_pos_rel_xyz": ObservationTermCfg(
      func=mdp.visible_ball_position_relative_xyz,
      params={"command_name": "set_square"},
    ),
    "visible_ball_vel_rel_xyz": ObservationTermCfg(
      func=mdp.visible_ball_velocity_relative_xyz,
      params={"command_name": "set_square"},
    ),
    "last_seen_ball_pos_rel_xy": ObservationTermCfg(
      func=mdp.last_seen_ball_position_relative_xy,
      params={"command_name": "set_square"},
    ),
    "last_seen_ball_vel_rel_xy": ObservationTermCfg(
      func=mdp.last_seen_ball_velocity_relative_xy,
      params={"command_name": "set_square"},
    ),
    "last_seen_ball_secs": ObservationTermCfg(
      func=mdp.last_seen_ball_secs,
      params={"command_name": "set_square"},
    ),
    "t_goal": ObservationTermCfg(
      func=mdp.visible_time_to_goal_plane,
      params={"command_name": "set_square", "max_time": 2.0},
    ),
  }

  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "decoded_actions": ObservationTermCfg(
      func=mdp.motor_last_decoded_action,
      params={"action_name": "motor_latent"},
    ),
    "target_dir_xy": ObservationTermCfg(
      func=mdp.target_direction_xy,
      params={"command_name": "set_square"},
    ),
    "robot_pos_rel_goal_line_xy": ObservationTermCfg(
      func=mdp.robot_position_relative_goal_line_xy,
      params={"command_name": "set_square"},
    ),
    "desired_point_rel_xy": ObservationTermCfg(
      func=mdp.desired_point_relative_xy,
      params={"command_name": "set_square"},
    ),
    "ball_pos_rel_xyz": ObservationTermCfg(
      func=mdp.target_position_relative_xyz,
      params={"command_name": "set_square"},
    ),
    "ball_vel_rel_xyz": ObservationTermCfg(
      func=mdp.ball_velocity_relative_xyz,
      params={"command_name": "set_square"},
    ),
    "t_goal": ObservationTermCfg(
      func=mdp.time_to_goal_plane,
      params={"command_name": "set_square", "max_time": 2.0},
    ),
  }

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="warn",
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="warn",
    ),
  }

  fallen_weight = -20.0

  cfg.rewards = {
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.004),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_ang_vel_penalty,
      weight=0.0,
      params={},
    ),
    "fallen": RewardTermCfg(
      func=mdp.fallen_indicator,
      weight=fallen_weight,
      params={
        "min_height": 0.32,
        "max_roll_deg": 100.0,
      },
    ),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "low_height_soft_penalty": RewardTermCfg(
      func=mdp.low_height_soft_penalty,
      weight=-1.6,
      params={"h_soft": 0.48},
    ),
    # Chosen near the old combined x/y penalty scale so the unified XY term
    # remains important without overpowering posture and ball-alignment rewards.
    "stance_center_target_xy_abs_pen": RewardTermCfg(
      func=mdp.stance_center_target_xy_abs_penalty,
      weight=-0.45,
      params={
        "command_name": "set_square",
        "left_foot_body_name": STANCE_ORTHO_LEFT_FOOT_BODY,
        "right_foot_body_name": STANCE_ORTHO_RIGHT_FOOT_BODY,
      },
    ),
    "stance_center_target_progress": RewardTermCfg(
      func=mdp.stance_center_target_progress_reward,
      weight=0.0,
      params={
        "command_name": "set_square",
        "left_foot_body_name": STANCE_ORTHO_LEFT_FOOT_BODY,
        "right_foot_body_name": STANCE_ORTHO_RIGHT_FOOT_BODY,
      },
    ),
    "stance_ortho_to_ball_reward": RewardTermCfg(
      func=mdp.stance_ortho_to_ball_reward,
      weight=0.65,
      params={
        "command_name": "set_square",
        "ortho_deadband": 0.10,
        "left_foot_body_name": STANCE_ORTHO_LEFT_FOOT_BODY,
        "right_foot_body_name": STANCE_ORTHO_RIGHT_FOOT_BODY,
      },
    ),
    "stance_width_band_pen": RewardTermCfg(
      func=mdp.stance_width_band_penalty,
      weight=-0.3,
      params={
        "command_name": "set_square",
        "w_min": 0.23,
        "w_max": 0.45,
        "left_foot_body_name": STANCE_ORTHO_LEFT_FOOT_BODY,
        "right_foot_body_name": STANCE_ORTHO_RIGHT_FOOT_BODY,
      },
    ),
    "pelvis_between_feet": RewardTermCfg(
      func=mdp.pelvis_between_feet_reward,
      weight=0.0,
      params={
        "command_name": "set_square",
        "left_foot_body_name": STANCE_ORTHO_LEFT_FOOT_BODY,
        "right_foot_body_name": STANCE_ORTHO_RIGHT_FOOT_BODY,
        "waist_body_name": WAIST_BODY_NAME_REGEX,
        "apply_standing_gate": True,
      },
    ),
    "upright": RewardTermCfg(
      func=mdp.upright_stability_reward,
      weight=0.8,
      params={
        "roll_band": UPRIGHT_ROLL_BAND,
        "roll_sigma": UPRIGHT_ROLL_SIGMA,
        "pitch_target": UPRIGHT_PITCH_TARGET,
        "pitch_band": UPRIGHT_PITCH_BAND,
        "pitch_sigma": UPRIGHT_PITCH_SIGMA,
      },
    ),
    "waist_ready_twist_abs_pen": RewardTermCfg(
      func=mdp.waist_ready_twist_abs_penalty,
      weight=-0.10,
      params={
        "command_name": "set_square",
        "k": 2.5,
        "left_foot_body_name": STANCE_ORTHO_LEFT_FOOT_BODY,
        "right_foot_body_name": STANCE_ORTHO_RIGHT_FOOT_BODY,
        "waist_body_name": WAIST_BODY_NAME_REGEX,
        "apply_standing_gate": True,
      },
    ),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan_detection": TerminationTermCfg(func=mdp.nan_detection),
    "fallen": TerminationTermCfg(
      func=mdp.FallTermination,
      params={
        "min_height": 0.32,
        "max_roll_deg": 100.0,
        "consecutive_steps": 6,
      },
    ),
  }

  cfg.curriculum = {}
  cfg.events = {
    "reset_scene_to_default": EventTermCfg(
      mode="reset",
      func=reset_scene_to_default,
    ),
  }

  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    lookat=(GOAL_X_LINE - 1.2, 0.0, 0.8),
    distance=8.0,
    elevation=-20.0,
    azimuth=180.0,
  )

  cfg.sim.mujoco.timestep = SIM_TIMESTEP_S
  cfg.decimation = CONTROL_DECIMATION
  cfg.episode_length_s = EPISODE_LENGTH_S

  return cfg
