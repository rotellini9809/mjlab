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
from mjlab.motor_controller_stage1.latent_action import get_wandb_run_name
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.goalkeeper_experts.e2v2_mezzaluna import mdp
from mjlab.tasks.goalkeeper_experts.launcher import (
  E2_STAGE1_BASIC,
  E2_STAGE3_VERTICAL_PACE,
  apply_e2_launcher_preset,
)
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

GOAL_X_LINE = 7.0
GOALPOST_X = 7.3
E1_HOME_POINT_X = 6.70
E1_HOME_POINT_Y = 0.0
E1_HOME_POINT_BAND_RADIUS = 0.10

# Legacy fixed ranges retained for compatibility; E2V2 reset now spawns from the
# mezzaluna using the sampled ball spawn instead of using these directly.
KEEPER_SPAWN_X_RANGE = (6.4, 6.8)
KEEPER_SPAWN_Y_RANGE = (-0.4, 0.4)
# Matches E1's ready-pose spawn after compensating for the measured
# foot-collision clearance above the field collider.
KEEPER_SPAWN_Z = 0.6728
# Ready pose imported from crouch_stance_1 tracking run k0zgfxdw near the start
# (frame 11/66, ~= 1/6 of the clip), extracted in the MJLab controller.
READY_ROOT_QUAT = (
  0.850692629814148,
  -0.01908080279827118,
  0.048677168786525726,
  0.5230569243431091,
)
READY_ROOT_YAW = 1.1035051026418754
READY_JOINT_POS = [
  -0.006449203006923199,
  -0.06991329044103622,
  -0.06792288273572922,
  -1.2981278896331787,
  0.5135267376899719,
  -0.5423100590705872,
  -0.0499301552772522,
  1.2249946594238281,
  0.5443910956382751,
  0.6937841773033142,
  0.07025951147079468,
  -0.3184724450111389,
  -0.03504209965467453,
  0.06286874413490295,
  0.2646978199481964,
  -0.04678475856781006,
  0.015397579409182072,
  -0.3520301878452301,
  -0.004032755270600319,
  -0.08282425254583359,
  0.4212573170661926,
  -0.17360135912895203,
  -0.040193840861320496,
]
# Yaw offset applied around the "face the ball" heading sampled at reset.
SPAWN_YAW_RANGE = (-0.1, 0.1)

# Upright posture shaping (same convention as E1): strict lateral, tolerant
# sagittal with a slight forward lean.
UPRIGHT_ROLL_BAND = 0.1
UPRIGHT_ROLL_SIGMA = 0.12
UPRIGHT_PITCH_TARGET = 0.10
UPRIGHT_PITCH_BAND = 0.25
UPRIGHT_PITCH_SIGMA = 0.30

# Keep reset close to standing/default with light noise.
KEEPER_JOINT_POS_NOISE = 0.02
KEEPER_JOINT_VEL_NOISE = 0.08
MEZZALUNA_CENTER_X = GOAL_X_LINE - 0.20
MEZZALUNA_CENTER_Y = 0.0
MEZZALUNA_APEX_X = (GOAL_X_LINE - 1.0) + 0.15 - 0.20
MEZZALUNA_HALF_WIDTH_Y = 1.55 + 0.10
MEZZALUNA_SPAWN_XY_JITTER_RANGE = (-0.10, 0.10)
MEZZALUNA_SPAWN_RADIAL_JITTER_RANGE = (-0.04, 0.04)

# Reusable launcher defaults for E2. Presets only override launcher sampling.
E2V2_MEZZALUNA_RESET_CURRICULUM_STAGE = os.environ.get(
  "MJLAB_E2V2_MEZZALUNA_RESET_CURRICULUM_STAGE", ""
).strip()
E2V2_MEZZALUNA_DEFAULT_LAUNCHER_PRESET_NAME = (
  mdp.E2V2_MEZZALUNA_STAGE2_GROUND_AIR
)
E2_DEFLECTION_TIME_AFTER_LAUNCH_RANGE = (0.08, 0.22)
E2_DEFLECTION_DV_MAG_RANGE = (0.35, 1.25)
E2_LAUNCH_MAX_SPEED = 8.5
E2_LAUNCH_MAX_ABS_VZ = 5.5
E2_MIN_TOWARD_GOAL_SPEED = 0.8
E2V2_GROUND_NEAR_X_RANGE = (4.8, 5.9)
E2V2_GROUND_FAR_X_RANGE = (3.2, 4.8)
E2V2_ONE_BOUNCE_X_RANGE = (3.6, 5.4)
E2V2_LOB_X_RANGE = (3.5, 5.3)

# Goal-plane aperture (used for both detection and visualization).
GOAL_PLANE_X = GOAL_X_LINE
GOAL_PLANE_Y_CENTER = 0.0
GOAL_PLANE_Y_HALF = 1.30
GOAL_PLANE_Z_MIN = 0.0
GOAL_PLANE_Z_MAX = 1.85

GOAL_PLANE_VIS_HALF_THICKNESS = 0.005
GOAL_PLANE_VIS_RGBA = (0.15, 0.85, 0.95, 0.08)
GOAL_PLANE_VIS_GROUP = 3
MEZZALUNA_VIS_GROUP = 2
MEZZALUNA_VIS_RGBA = (0.56, 0.95, 0.62, 0.80)
MEZZALUNA_HALF_WIDTH = 0.02
MEZZALUNA_HALF_THICKNESS = 0.003
MEZZALUNA_Z = 0.006
MEZZALUNA_SEGMENTS = 40

# E2 ball danger area used by clearance-quality shaping.
E2_DANGER_AREA_BOUNDS = (GOAL_X_LINE - 2.3, GOAL_X_LINE + 0.3, -2.5, 2.5)
E2_DANGER_AREA_OVERLAY_HALF_THICKNESS = 0.0015
E2_DANGER_AREA_OVERLAY_Z = 0.003
E2_DANGER_AREA_OVERLAY_RGBA = (0.95, 0.18, 0.18, 0.22)
E2_DANGER_AREA_VIS_GROUP = 3

# E2 keeper area used by the outside-area penalty. Matches E3's keeper-area
# bounds and local goal-line anchoring convention.
E2_KEEPER_AREA_BOUNDS = (GOAL_X_LINE - 1.8, GOAL_X_LINE + 0.6, -2.0, 2.0)
E2_KEEPER_AREA_OVERLAY_HALF_THICKNESS = 0.0015
E2_KEEPER_AREA_OVERLAY_Z = 0.005
E2_KEEPER_AREA_OVERLAY_RGBA = (0.95, 0.85, 0.10, 0.24)
E2_KEEPER_AREA_VIS_GROUP = 3

BALL_ROBOT_CONTACT_SENSOR_NAME = "ball_robot_contact"
HEAD_BALL_CONTACT_SENSOR_NAME = "head_ball_contact"
ARM_BALL_CONTACT_SENSOR_NAME = "arm_ball_contact"
HEAD_BODIES = ("H1", "H2")
ARM_CONTACT_BODIES = ("AL1", "AL2", "AL3", "left_hand_link", "AR1", "AR2", "AR3", "right_hand_link")
RESOLUTION_WINDOW_S = 1.5

# Stage-1 command dimension used in motor-observation layout.
MOTOR_COMMAND_DIM = 46
MOTOR_ACT_DIM = 23

EPISODE_LENGTH_S = 6.0
SIM_TIMESTEP_S = 0.005
CONTROL_DECIMATION = 4
E2V2_MEZZALUNA_TASK_ID = "Mjlab-GK-Expert-E2V2-Mezzaluna-Booster-T1_23"
E2V2_MEZZALUNA_EXPERIMENT_NAME = "gk_expert_e2v2_mezzaluna_booster_t1_23"


def _get_cli_flag_value(flag: str) -> str | None:
  flag_eq = f"{flag}="
  argv = sys.argv[1:]
  for index, arg in enumerate(argv):
    if arg == flag and index + 1 < len(argv):
      return argv[index + 1]
    if arg.startswith(flag_eq):
      return arg[len(flag_eq) :]
  return None


def _is_e2_play_cli_invocation() -> bool:
  script_stem = Path(sys.argv[0]).stem.lower()
  if "play" not in script_stem:
    return False
  return len(sys.argv) > 1 and sys.argv[1] == E2V2_MEZZALUNA_TASK_ID


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


def _extract_saved_e2_launcher_preset_name(env_yaml_path: Path) -> str | None:
  with env_yaml_path.open("r", encoding="utf-8") as handle:
    env_data = yaml.safe_load(handle) or {}

  return _extract_e2_launcher_preset_name_from_env_data(env_data)


def _extract_saved_e2_stage1_run_from_env_data(
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


def _extract_e2_launcher_preset_name_from_env_data(env_data: object) -> str | None:
  if not isinstance(env_data, dict):
    return None
  commands = env_data.get("commands")
  if not isinstance(commands, dict):
    return None
  stand_block_cfg = commands.get("stand_block")
  if not isinstance(stand_block_cfg, dict):
    return None
  launcher_cfg = stand_block_cfg.get("launcher_cfg")
  if not isinstance(launcher_cfg, dict):
    return None
  preset_name = launcher_cfg.get("active_preset_name")
  if not isinstance(preset_name, str):
    return None
  preset_name = preset_name.strip()
  return preset_name or None


def _extract_saved_e2_launcher_preset_name_from_wandb_config(
  config_yaml_path: Path,
) -> str | None:
  with config_yaml_path.open("r", encoding="utf-8") as handle:
    config_data = yaml.safe_load(handle) or {}

  if not isinstance(config_data, dict):
    return None
  env_cfg = config_data.get("env_cfg")
  if not isinstance(env_cfg, dict):
    return None
  env_cfg_value = env_cfg.get("value")
  return _extract_e2_launcher_preset_name_from_env_data(env_cfg_value)


def _extract_saved_e2_stage1_run_from_wandb_config(
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
  return _extract_saved_e2_stage1_run_from_env_data(env_cfg_value)


def _resolve_saved_e2_play_launcher_preset_name() -> str | None:
  if not _is_e2_play_cli_invocation():
    return None

  checkpoint_file = _get_cli_flag_value("--checkpoint-file")
  if checkpoint_file:
    env_yaml_path = Path(checkpoint_file).expanduser().resolve().parent / "params" / "env.yaml"
    if not env_yaml_path.exists():
      print(
        "[WARN]: Saved E2 env config was not found next to the checkpoint; "
        "using the default play preset."
      )
      return None
    return _extract_saved_e2_launcher_preset_name(env_yaml_path)

  wandb_run_path = _get_cli_flag_value("--wandb-run-path")
  if not wandb_run_path:
    return None

  log_root = (Path("logs") / "rsl_rl" / E2V2_MEZZALUNA_EXPERIMENT_NAME).resolve()
  config_yaml_path = _try_download_wandb_run_file(
    log_root, wandb_run_path, "config.yaml"
  )
  if config_yaml_path is None:
    print(
      "[WARN]: Saved E2 W&B config was not found in the run; "
      "using the default play preset."
    )
    return None
  preset_name = _extract_saved_e2_launcher_preset_name_from_wandb_config(
    config_yaml_path
  )
  if preset_name is None:
    print(
      "[WARN]: E2 launcher preset was not found in the W&B config; "
      "using the default play preset."
    )
    return None
  return preset_name


def _resolve_saved_e2_play_stage1_run() -> tuple[str | None, str | None]:
  if not _is_e2_play_cli_invocation():
    return None, None

  checkpoint_file = _get_cli_flag_value("--checkpoint-file")
  if checkpoint_file:
    env_yaml_path = Path(checkpoint_file).expanduser().resolve().parent / "params" / "env.yaml"
    if not env_yaml_path.exists():
      return None, None
    with env_yaml_path.open("r", encoding="utf-8") as handle:
      env_data = yaml.safe_load(handle) or {}
    return _extract_saved_e2_stage1_run_from_env_data(env_data)

  wandb_run_path = _get_cli_flag_value("--wandb-run-path")
  if not wandb_run_path:
    return None, None

  log_root = (Path("logs") / "rsl_rl" / E2V2_MEZZALUNA_EXPERIMENT_NAME).resolve()
  config_yaml_path = _try_download_wandb_run_file(
    log_root, wandb_run_path, "config.yaml"
  )
  if config_yaml_path is not None:
    run_path, run_name = _extract_saved_e2_stage1_run_from_wandb_config(
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
  return _extract_saved_e2_stage1_run_from_env_data(env_data)


def _add_field_overlays(spec: mujoco.MjSpec) -> None:
  overlay_body = next(
    (body for body in spec.bodies if body.name == "e2_field_overlays"), None
  )
  if overlay_body is None:
    overlay_body = spec.worldbody.add_body(name="e2_field_overlays")

  center_z = 0.5 * (GOAL_PLANE_Z_MIN + GOAL_PLANE_Z_MAX)
  half_z = max(0.5 * (GOAL_PLANE_Z_MAX - GOAL_PLANE_Z_MIN), 1.0e-3)

  overlay = overlay_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(GOAL_PLANE_X, GOAL_PLANE_Y_CENTER, center_z),
    size=(GOAL_PLANE_VIS_HALF_THICKNESS, GOAL_PLANE_Y_HALF, half_z),
  )
  overlay.name = "e2_goal_plane_overlay"
  overlay.rgba = GOAL_PLANE_VIS_RGBA
  # Put the purely-visual goal plane in the same hidden-by-default viewer group
  # used for collision/debug geometry so Viser does not show it unless requested.
  overlay.group = GOAL_PLANE_VIS_GROUP
  overlay.contype = 0
  overlay.conaffinity = 0

  danger_x_min, danger_x_max, danger_y_min, danger_y_max = E2_DANGER_AREA_BOUNDS
  danger_center_x = 0.5 * (danger_x_min + danger_x_max)
  danger_center_y = 0.5 * (danger_y_min + danger_y_max)
  danger_half_x = max(0.5 * (danger_x_max - danger_x_min), 1.0e-3)
  danger_half_y = max(0.5 * (danger_y_max - danger_y_min), 1.0e-3)

  danger_overlay = overlay_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(danger_center_x, danger_center_y, E2_DANGER_AREA_OVERLAY_Z),
    size=(
      danger_half_x,
      danger_half_y,
      E2_DANGER_AREA_OVERLAY_HALF_THICKNESS,
    ),
  )
  danger_overlay.name = "e2_danger_area_overlay"
  danger_overlay.rgba = E2_DANGER_AREA_OVERLAY_RGBA
  danger_overlay.group = E2_DANGER_AREA_VIS_GROUP
  danger_overlay.contype = 0
  danger_overlay.conaffinity = 0

  keeper_x_min, keeper_x_max, keeper_y_min, keeper_y_max = E2_KEEPER_AREA_BOUNDS
  keeper_center_x = 0.5 * (keeper_x_min + keeper_x_max)
  keeper_center_y = 0.5 * (keeper_y_min + keeper_y_max)
  keeper_half_x = max(0.5 * (keeper_x_max - keeper_x_min), 1.0e-3)
  keeper_half_y = max(0.5 * (keeper_y_max - keeper_y_min), 1.0e-3)

  keeper_overlay = overlay_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(keeper_center_x, keeper_center_y, E2_KEEPER_AREA_OVERLAY_Z),
    size=(
      keeper_half_x,
      keeper_half_y,
      E2_KEEPER_AREA_OVERLAY_HALF_THICKNESS,
    ),
  )
  keeper_overlay.name = "e2_keeper_area_overlay"
  keeper_overlay.rgba = E2_KEEPER_AREA_OVERLAY_RGBA
  keeper_overlay.group = E2_KEEPER_AREA_VIS_GROUP
  keeper_overlay.contype = 0
  keeper_overlay.conaffinity = 0

  ellipse_a = max(float(MEZZALUNA_CENTER_X) - float(MEZZALUNA_APEX_X), 1.0e-6)
  ellipse_b = max(float(MEZZALUNA_HALF_WIDTH_Y), 1.0e-6)
  theta = torch.linspace(
    -0.5 * math.pi,
    0.5 * math.pi,
    steps=int(MEZZALUNA_SEGMENTS) + 1,
    dtype=torch.float32,
  )
  x = float(MEZZALUNA_CENTER_X) - ellipse_a * torch.cos(theta)
  y = float(MEZZALUNA_CENTER_Y) + ellipse_b * torch.sin(theta)
  for index in range(int(MEZZALUNA_SEGMENTS)):
    p0 = torch.tensor([x[index], y[index]], dtype=torch.float32)
    p1 = torch.tensor([x[index + 1], y[index + 1]], dtype=torch.float32)
    center = 0.5 * (p0 + p1)
    delta = p1 - p0
    half_len = max(0.5 * torch.linalg.norm(delta).item(), 1.0e-6)
    angle = math.atan2(float(delta[1].item()), float(delta[0].item()))
    segment = overlay_body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=(float(center[0].item()), float(center[1].item()), MEZZALUNA_Z),
      size=(half_len, MEZZALUNA_HALF_WIDTH, MEZZALUNA_HALF_THICKNESS),
      quat=(math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)),
    )
    segment.name = f"e2v2_mezzaluna_segment_{index:02d}"
    segment.rgba = MEZZALUNA_VIS_RGBA
    segment.group = MEZZALUNA_VIS_GROUP
    segment.contype = 0
    segment.conaffinity = 0


def get_e2_field_cfg_with_goal_plane() -> EntityCfg:
  field_cfg = get_robocup_field_cfg()
  base_spec_fn = field_cfg.spec_fn

  def _spec_fn() -> mujoco.MjSpec:
    spec = base_spec_fn()
    _add_field_overlays(spec)
    return spec

  field_cfg.spec_fn = _spec_fn
  return field_cfg


def _apply_e2v2_mezzaluna_launcher_preset(
  cfg: mdp.GoalkeeperBallLauncherCfg,
  preset_name: str,
) -> mdp.GoalkeeperBallLauncherCfg:
  cfg.ground_near_x_range = E2V2_GROUND_NEAR_X_RANGE
  cfg.ground_far_x_range = E2V2_GROUND_FAR_X_RANGE
  cfg.one_bounce_x_range = E2V2_ONE_BOUNCE_X_RANGE
  cfg.lob_x_range = E2V2_LOB_X_RANGE
  if preset_name == mdp.E2V2_MEZZALUNA_STAGE1_GROUND:
    cfg = apply_e2_launcher_preset(cfg, E2_STAGE1_BASIC)
    cfg.active_preset_name = preset_name
    cfg.enabled_families = (True, False, False, False, False)
    cfg.family_weights = (1.0, 0.0, 0.0, 0.0, 0.0)
    cfg.shot_low_z_prob = 1.0
    cfg.ground_near_x_range = E2V2_GROUND_NEAR_X_RANGE
    cfg.ground_far_x_range = E2V2_GROUND_FAR_X_RANGE
    cfg.one_bounce_x_range = E2V2_ONE_BOUNCE_X_RANGE
    cfg.lob_x_range = E2V2_LOB_X_RANGE
    return cfg
  if preset_name == mdp.E2V2_MEZZALUNA_STAGE2_GROUND_AIR:
    cfg = apply_e2_launcher_preset(cfg, E2_STAGE3_VERTICAL_PACE)
    cfg.active_preset_name = preset_name
    cfg.enabled_families = (True, True, True, True, True)
    cfg.family_weights = (0.45, 0.20, 0.15, 0.10, 0.10)
    cfg.ground_near_x_range = E2V2_GROUND_NEAR_X_RANGE
    cfg.ground_far_x_range = E2V2_GROUND_FAR_X_RANGE
    cfg.one_bounce_x_range = E2V2_ONE_BOUNCE_X_RANGE
    cfg.lob_x_range = E2V2_LOB_X_RANGE
    return cfg
  supported = ", ".join(mdp.E2V2_MEZZALUNA_LAUNCHER_CURRICULUM_PRESET_NAMES)
  raise ValueError(
    f"Unknown E2V2 mezzaluna launcher preset '{preset_name}'. Supported presets: {supported}."
  )


def booster_t1_23_gk_expert_e2v2_mezzaluna_env_cfg(
  play: bool = False,
  launcher_preset_name: str | None = None,
  launcher_curriculum_stage: int | None = None,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  robot_cfg = get_t1_23_robot_cfg()
  robot_cfg.init_state.pos = (E1_HOME_POINT_X, E1_HOME_POINT_Y, KEEPER_SPAWN_Z)
  # Face toward field center from the defended +x goal side.
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
  cfg.scene.num_envs = 512 if not play else 1
  cfg.scene.entities = {
    "robot": robot_cfg,
    "soccer_field": get_e2_field_cfg_with_goal_plane(),
    "goalpost_left": goal_left_cfg,
    "goalpost_right": goal_right_cfg,
    "soccer_ball": soccer_ball_cfg,
  }
  cfg.sim.mujoco.ccd_iterations = 100

  ball_robot_contact_cfg = ContactSensorCfg(
    name=BALL_ROBOT_CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern="ball_collision", entity="soccer_ball"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
    track_air_time=True,
  )
  head_ball_contact_cfg = ContactSensorCfg(
    name=HEAD_BALL_CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="body", pattern=HEAD_BODIES, entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="ball_collision", entity="soccer_ball"),
    fields=("found",),
    reduce="none",
    num_slots=1,
    track_air_time=True,
  )
  arm_ball_contact_cfg = ContactSensorCfg(
    name=ARM_BALL_CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="body", pattern=ARM_CONTACT_BODIES, entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="ball_collision", entity="soccer_ball"),
    fields=("found",),
    reduce="none",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ball_robot_contact_cfg,
    head_ball_contact_cfg,
    arm_ball_contact_cfg,
  )

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
    saved_stage1_run_path, saved_stage1_run_name = _resolve_saved_e2_play_stage1_run()
    if saved_stage1_run_path is not None:
      env_stage1_run_name = stage1_goalkeeper_run_name
      resolved_saved_stage1_run_name = (
        saved_stage1_run_name or get_wandb_run_name(saved_stage1_run_path)
      )
      print(
        "[INFO]: Auto-selected E2 play Stage-1 controller from saved run: "
        f"{resolved_saved_stage1_run_name or '<unknown>'} "
        f"({saved_stage1_run_path})"
      )
      if (
        stage1_goalkeeper_run_path is not None
        and stage1_goalkeeper_run_path != saved_stage1_run_path
      ):
        print(
          "[WARN]: E2 play saved Stage-1 controller run differs from "
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
      command_name="stand_block",
      stage1_wandb_run_path=stage1_goalkeeper_run_path,
      stage1_wandb_run_name=stage1_goalkeeper_run_name,
      motor_obs_terms=motor_obs_terms,
      motor_obs_term_dims=motor_obs_term_dims,
      strict_obs_layout=True,
    )
  }

  resolved_preset_name = launcher_preset_name
  if resolved_preset_name is None and launcher_curriculum_stage is not None:
    resolved_preset_name = mdp.get_e2v2_mezzaluna_launcher_curriculum_preset_name(
      launcher_curriculum_stage
    )
  if resolved_preset_name is None and play:
    resolved_preset_name = _resolve_saved_e2_play_launcher_preset_name()
    if resolved_preset_name is not None:
      print(
        f"[INFO]: Auto-selected E2 play launcher preset from saved run: "
        f"{resolved_preset_name}"
      )
  if resolved_preset_name is None and E2V2_MEZZALUNA_RESET_CURRICULUM_STAGE:
    resolved_preset_name = mdp.get_e2v2_mezzaluna_launcher_curriculum_preset_name(
      int(E2V2_MEZZALUNA_RESET_CURRICULUM_STAGE)
    )
  if resolved_preset_name is None:
    resolved_preset_name = E2V2_MEZZALUNA_DEFAULT_LAUNCHER_PRESET_NAME

  launcher_cfg = mdp.GoalkeeperBallLauncherCfg(
    ball_entity_name="soccer_ball",
    goal_toward_positive_x=True,
    goal_plane_x=GOAL_PLANE_X,
    goal_y_center=GOAL_PLANE_Y_CENTER,
    goal_y_half=GOAL_PLANE_Y_HALF,
    goal_z_min=GOAL_PLANE_Z_MIN,
    goal_z_max=GOAL_PLANE_Z_MAX,
    max_speed=E2_LAUNCH_MAX_SPEED,
    max_abs_vz=E2_LAUNCH_MAX_ABS_VZ,
    min_toward_goal_speed=E2_MIN_TOWARD_GOAL_SPEED,
    deflection_time_after_launch_range=E2_DEFLECTION_TIME_AFTER_LAUNCH_RANGE,
    deflection_dv_mag_range=E2_DEFLECTION_DV_MAG_RANGE,
  )
  launcher_cfg = _apply_e2v2_mezzaluna_launcher_preset(launcher_cfg, resolved_preset_name)

  cfg.commands = {
    "stand_block": mdp.StandBlockCommandCfg(
      entity_name="robot",
      ball_entity_name="soccer_ball",
      ball_robot_contact_sensor_name=BALL_ROBOT_CONTACT_SENSOR_NAME,
      command_dim=MOTOR_COMMAND_DIM,
      keeper_spawn_x_range=KEEPER_SPAWN_X_RANGE,
      keeper_spawn_y_range=KEEPER_SPAWN_Y_RANGE,
      spawn_yaw_range=SPAWN_YAW_RANGE,
      mezzaluna_center_x=MEZZALUNA_CENTER_X,
      mezzaluna_center_y=MEZZALUNA_CENTER_Y,
      mezzaluna_apex_x=MEZZALUNA_APEX_X,
      mezzaluna_half_width_y=MEZZALUNA_HALF_WIDTH_Y,
      mezzaluna_spawn_xy_jitter_range=MEZZALUNA_SPAWN_XY_JITTER_RANGE,
      mezzaluna_spawn_radial_jitter_range=MEZZALUNA_SPAWN_RADIAL_JITTER_RANGE,
      keeper_joint_pos_noise=KEEPER_JOINT_POS_NOISE,
      keeper_joint_vel_noise=KEEPER_JOINT_VEL_NOISE,
      launcher_cfg=launcher_cfg,
      goal_toward_positive_x=True,
      goal_plane_x=GOAL_PLANE_X,
      goal_plane_y_center=GOAL_PLANE_Y_CENTER,
      goal_plane_y_half=GOAL_PLANE_Y_HALF,
      goal_plane_z_min=GOAL_PLANE_Z_MIN,
      goal_plane_z_max=GOAL_PLANE_Z_MAX,
      danger_area_bounds=E2_DANGER_AREA_BOUNDS,
      keeper_area_bounds=E2_KEEPER_AREA_BOUNDS,
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
      func=mdp.target_direction_xy,
      params={"command_name": "stand_block"},
    ),
    "robot_pos_rel_goal_line_xy": ObservationTermCfg(
      func=mdp.robot_position_relative_goal_line_xy,
      params={"command_name": "stand_block"},
    ),
    "ball_pos_rel_xyz": ObservationTermCfg(
      func=mdp.ball_position_relative_xyz,
      params={"command_name": "stand_block"},
    ),
    "ball_vel_rel_xyz": ObservationTermCfg(
      func=mdp.ball_velocity_relative_xyz,
      params={"command_name": "stand_block"},
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
      params={"command_name": "stand_block"},
    ),
    "robot_pos_rel_goal_line_xy": ObservationTermCfg(
      func=mdp.robot_position_relative_goal_line_xy,
      params={"command_name": "stand_block"},
    ),
    "ball_pos_rel_xyz": ObservationTermCfg(
      func=mdp.ball_position_relative_xyz,
      params={"command_name": "stand_block"},
    ),
    "ball_vel_rel_xyz": ObservationTermCfg(
      func=mdp.ball_velocity_relative_xyz,
      params={"command_name": "stand_block"},
    ),
    "t_goal": ObservationTermCfg(
      func=mdp.time_to_goal_plane,
      params={"command_name": "stand_block", "max_time": 2.0},
    ),
  }

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  cfg.rewards = {
    "goal_conceded": RewardTermCfg(
      func=mdp.goal_conceded_indicator,
      weight=-500.0,
      params={"command_name": "stand_block"},
    ),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.005,
      params={"command_name": "stand_block"},
    ),
    "save_success": RewardTermCfg(
      func=mdp.save_success_reward,
      weight=180.0,
      params={
        "command_name": "stand_block",
        "resolution_term_name": "contact_resolution_window",
        "apply_standing_gate": False,
      },
    ),
    "deflect_away": RewardTermCfg(
      func=mdp.deflect_away_from_goal_reward,
      weight=0.0,
      params={
        "command_name": "stand_block",
        "only_on_first_contact": True,
      },
    ),
    "arm_high_throw_deflect_reward": RewardTermCfg(
      func=mdp.arm_high_throw_deflect_reward,
      weight=6.0,
      params={
        "command_name": "stand_block",
        "arm_sensor_name": ARM_BALL_CONTACT_SENSOR_NAME,
      },
    ),
    "clearance_quality": RewardTermCfg(
      func=mdp.ClearanceQualityReward,
      weight=8.0,
      params={
        "command_name": "stand_block",
        "t_clear_clip": 0.5,
        "clip_away_speed": 2.5,
      },
    ),
    "stabilize_after_exit": RewardTermCfg(
      func=mdp.StabilizeAfterExitReward,
      weight=2.5,
      params={"command_name": "stand_block"},
    ),
    "face_ball_after_exit_reward": RewardTermCfg(
      func=mdp.FaceBallAfterExitReward,
      weight=1.0,
      params={"command_name": "stand_block"},
    ),
    "low_height_soft_penalty": RewardTermCfg(
      func=mdp.low_height_soft_penalty,
      weight=-3.5,
      params={"h_soft": 0.55},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-0.35,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "stand_block",
      },
    ),
    "upright": RewardTermCfg(
      func=mdp.upright_stability_reward,
      weight=2.0,
      params={
        "roll_band": UPRIGHT_ROLL_BAND,
        "roll_sigma": UPRIGHT_ROLL_SIGMA,
        "pitch_target": UPRIGHT_PITCH_TARGET,
        "pitch_band": UPRIGHT_PITCH_BAND,
        "pitch_sigma": UPRIGHT_PITCH_SIGMA,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_ang_vel_penalty,
      weight=-0.01,
      params={"command_name": "stand_block"},
    ),
    "head_contact_penalty": RewardTermCfg(
      func=mdp.head_contact_penalty,
      weight=-6.0,
      params={"head_sensor_name": HEAD_BALL_CONTACT_SENSOR_NAME},
    ),
    "outside_area": RewardTermCfg(
      func=mdp.outside_area_penalty,
      weight=-10.0,
      params={"command_name": "stand_block"},
    ),
    "fallen": RewardTermCfg(
      func=mdp.fallen_indicator,
      weight=-90.0,
      params={
        "min_height": 0.32,
        "max_roll_deg": 100.0,
      },
    ),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan_detection": TerminationTermCfg(func=mdp.nan_detection),
    "goal_conceded": TerminationTermCfg(
      func=mdp.goal_conceded_termination,
      params={"command_name": "stand_block"},
    ),
    "contact_resolution_window": TerminationTermCfg(
      func=mdp.ContactResolutionTermination,
      params={
        "command_name": "stand_block",
        "resolution_window_s": RESOLUTION_WINDOW_S,
      },
    ),
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
    lookat=(GOAL_X_LINE - 1.4, 0.0, 0.9),
    distance=6.5,
    elevation=-18.0,
    azimuth=178.0,
  )

  cfg.sim.mujoco.timestep = SIM_TIMESTEP_S
  cfg.decimation = CONTROL_DECIMATION
  cfg.episode_length_s = EPISODE_LENGTH_S

  return cfg
