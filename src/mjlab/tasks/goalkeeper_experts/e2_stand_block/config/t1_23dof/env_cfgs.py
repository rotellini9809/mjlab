from __future__ import annotations

import os

import mujoco

from mjlab.asset_zoo.robots import T1_23_ACTION_SCALE, get_t1_23_robot_cfg
from mjlab.asset_zoo.robocup_assets.field import get_robocup_field_cfg
from mjlab.asset_zoo.robocup_assets.goalpost import get_robocup_goalpost_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.goalkeeper_experts.e2_stand_block import mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.viewer import ViewerConfig

GOAL_X_LINE = 7.0
GOALPOST_X = 7.3

# E2 keeper spawn near the defended goal line (small lateral error).
KEEPER_SPAWN_X_RANGE = (GOAL_X_LINE - 0.15, GOAL_X_LINE + 0.15)
KEEPER_SPAWN_Y_RANGE = (-0.25, 0.25)
KEEPER_SPAWN_Z = 0.658
SPAWN_YAW_RANGE = (
  3.141592653589793 - 0.17453292519943295,
  3.141592653589793 + 0.17453292519943295,
)

# Keep reset close to standing/default with light noise.
KEEPER_JOINT_POS_NOISE = 0.02
KEEPER_JOINT_VEL_NOISE = 0.08

# Keeper area bounds (x_min, x_max, y_min, y_max).
KEEPER_AREA_BOUNDS = (GOAL_X_LINE - 1.0, GOAL_X_LINE + 0.6, -2.0, 2.0)
KEEPER_AREA_HARD_MARGIN = 0.35

# Reusable launcher defaults for E2 (requested mix).
E2_LAUNCHER_FAMILY_WEIGHTS = (0.45, 0.15, 0.15, 0.10, 0.15)
E2_LAUNCH_DELAY_RANGE = (0.10, 0.35)
E2_T_GOAL_BAND = (0.35, 1.00)
E2_DEFLECTION_PROB = 0.06
E2_DEFLECTION_TIME_AFTER_LAUNCH_RANGE = (0.08, 0.22)
E2_DEFLECTION_DV_MAG_RANGE = (0.35, 1.25)
E2_LAUNCH_MAX_SPEED = 8.5
E2_LAUNCH_MAX_ABS_VZ = 5.5
E2_MIN_TOWARD_GOAL_SPEED = 0.8

# Goal-plane aperture (used for both detection and visualization).
GOAL_PLANE_X = GOAL_X_LINE
GOAL_PLANE_Y_CENTER = 0.0
GOAL_PLANE_Y_HALF = 1.30
GOAL_PLANE_Z_MIN = 0.0
GOAL_PLANE_Z_MAX = 1.85

GOAL_PLANE_VIS_HALF_THICKNESS = 0.005
GOAL_PLANE_VIS_RGBA = (0.15, 0.85, 0.95, 0.08)
E2_AREA_OVERLAY_HALF_THICKNESS = 0.0015
E2_HARD_AREA_OVERLAY_Z = 0.0015
E2_KEEPER_AREA_OVERLAY_Z = 0.0035
E2_HARD_AREA_RGBA = (0.95, 0.55, 0.10, 0.22)
E2_KEEPER_AREA_RGBA = (0.05, 0.60, 0.95, 0.30)

BALL_ROBOT_CONTACT_SENSOR_NAME = "ball_robot_contact"
RESOLUTION_WINDOW_S = 0.8

# Stage-1 command dimension used in motor-observation layout.
MOTOR_COMMAND_DIM = 46
MOTOR_ACT_DIM = 23

EPISODE_LENGTH_S = 2.0
SIM_TIMESTEP_S = 0.005
CONTROL_DECIMATION = 4


def _add_goal_plane_overlay(spec: mujoco.MjSpec) -> None:
  field_body = next((body for body in spec.bodies if body.name == "field"), None)
  if field_body is None:
    field_body = spec.worldbody.add_body(name="field")

  center_z = 0.5 * (GOAL_PLANE_Z_MIN + GOAL_PLANE_Z_MAX)
  half_z = max(0.5 * (GOAL_PLANE_Z_MAX - GOAL_PLANE_Z_MIN), 1.0e-3)

  overlay = field_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(GOAL_PLANE_X, GOAL_PLANE_Y_CENTER, center_z),
    size=(GOAL_PLANE_VIS_HALF_THICKNESS, GOAL_PLANE_Y_HALF, half_z),
  )
  overlay.name = "e2_goal_plane_overlay"
  overlay.rgba = GOAL_PLANE_VIS_RGBA
  overlay.contype = 0
  overlay.conaffinity = 0

  def _add_area_overlay(
    name: str,
    bounds: tuple[float, float, float, float],
    z_center: float,
    rgba: tuple[float, float, float, float],
  ) -> None:
    x_min, x_max, y_min, y_max = bounds
    half_x = max(0.5 * (x_max - x_min), 1.0e-3)
    half_y = max(0.5 * (y_max - y_min), 1.0e-3)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)

    area = field_body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=(center_x, center_y, z_center),
      size=(half_x, half_y, E2_AREA_OVERLAY_HALF_THICKNESS),
    )
    area.name = name
    area.rgba = rgba
    area.contype = 0
    area.conaffinity = 0

  hard_bounds = (
    KEEPER_AREA_BOUNDS[0] - KEEPER_AREA_HARD_MARGIN,
    KEEPER_AREA_BOUNDS[1] + KEEPER_AREA_HARD_MARGIN,
    KEEPER_AREA_BOUNDS[2] - KEEPER_AREA_HARD_MARGIN,
    KEEPER_AREA_BOUNDS[3] + KEEPER_AREA_HARD_MARGIN,
  )
  _add_area_overlay(
    "e2_keeper_area_hard_overlay",
    hard_bounds,
    E2_HARD_AREA_OVERLAY_Z,
    E2_HARD_AREA_RGBA,
  )
  _add_area_overlay(
    "e2_keeper_area_overlay",
    KEEPER_AREA_BOUNDS,
    E2_KEEPER_AREA_OVERLAY_Z,
    E2_KEEPER_AREA_RGBA,
  )


def get_e2_field_cfg_with_goal_plane() -> EntityCfg:
  field_cfg = get_robocup_field_cfg()
  base_spec_fn = field_cfg.spec_fn

  def _spec_fn() -> mujoco.MjSpec:
    spec = base_spec_fn()
    _add_goal_plane_overlay(spec)
    return spec

  field_cfg.spec_fn = _spec_fn
  return field_cfg


def booster_t1_23_gk_expert_stand_block_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  robot_cfg = get_t1_23_robot_cfg()
  robot_cfg.init_state.pos = (GOAL_X_LINE - 0.2, 0.0, KEEPER_SPAWN_Z)
  # Face toward field center from the defended +x goal side.
  robot_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)

  goal_left_cfg = get_robocup_goalpost_cfg()
  goal_left_cfg.init_state.pos = (GOALPOST_X, 0.0, 0.0)
  goal_left_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)

  goal_right_cfg = get_robocup_goalpost_cfg()
  goal_right_cfg.init_state.pos = (-GOALPOST_X, 0.0, 0.0)
  goal_right_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

  soccer_ball_cfg = mdp.get_target_ball_cfg()

  cfg.scene.terrain = None
  cfg.scene.num_envs = 512 if not play else 1
  cfg.scene.entities = {
    "robot": robot_cfg,
    "soccer_field": get_e2_field_cfg_with_goal_plane(),
    "goalpost_left": goal_left_cfg,
    "goalpost_right": goal_right_cfg,
    "soccer_ball": soccer_ball_cfg,
  }

  ball_robot_contact_cfg = ContactSensorCfg(
    name=BALL_ROBOT_CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern="ball_collision", entity="soccer_ball"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, ball_robot_contact_cfg)

  motor_obs_terms, motor_obs_term_dims = mdp.default_motor_obs_layout(
    act_dim=MOTOR_ACT_DIM,
  )

  cfg.actions = {
    "motor_latent": mdp.MotorLatentActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=T1_23_ACTION_SCALE,
      use_default_offset=True,
      command_name="stand_block",
      stage1_wandb_run_path=os.environ.get("MJLAB_STAGE1_WANDB_RUN_PATH_GOALKEEPER"),
      motor_obs_terms=motor_obs_terms,
      motor_obs_term_dims=motor_obs_term_dims,
      strict_obs_layout=True,
    )
  }

  launcher_cfg = mdp.GoalkeeperBallLauncherCfg(
    ball_entity_name="soccer_ball",
    goal_toward_positive_x=True,
    goal_plane_x=GOAL_PLANE_X,
    goal_y_center=GOAL_PLANE_Y_CENTER,
    goal_y_half=GOAL_PLANE_Y_HALF,
    goal_z_min=GOAL_PLANE_Z_MIN,
    goal_z_max=GOAL_PLANE_Z_MAX,
    delay_range=E2_LAUNCH_DELAY_RANGE,
    t_goal_band=E2_T_GOAL_BAND,
    family_weights=E2_LAUNCHER_FAMILY_WEIGHTS,
    max_speed=E2_LAUNCH_MAX_SPEED,
    max_abs_vz=E2_LAUNCH_MAX_ABS_VZ,
    min_toward_goal_speed=E2_MIN_TOWARD_GOAL_SPEED,
    deflection_prob=E2_DEFLECTION_PROB,
    deflection_time_after_launch_range=E2_DEFLECTION_TIME_AFTER_LAUNCH_RANGE,
    deflection_dv_mag_range=E2_DEFLECTION_DV_MAG_RANGE,
  )

  cfg.commands = {
    "stand_block": mdp.StandBlockCommandCfg(
      entity_name="robot",
      ball_entity_name="soccer_ball",
      ball_robot_contact_sensor_name=BALL_ROBOT_CONTACT_SENSOR_NAME,
      command_dim=MOTOR_COMMAND_DIM,
      keeper_spawn_x_range=KEEPER_SPAWN_X_RANGE,
      keeper_spawn_y_range=KEEPER_SPAWN_Y_RANGE,
      spawn_yaw_range=SPAWN_YAW_RANGE,
      keeper_joint_pos_noise=KEEPER_JOINT_POS_NOISE,
      keeper_joint_vel_noise=KEEPER_JOINT_VEL_NOISE,
      keeper_area_bounds=KEEPER_AREA_BOUNDS,
      hard_area_margin=KEEPER_AREA_HARD_MARGIN,
      launcher_cfg=launcher_cfg,
      goal_toward_positive_x=True,
      goal_plane_x=GOAL_PLANE_X,
      goal_plane_y_center=GOAL_PLANE_Y_CENTER,
      goal_plane_y_half=GOAL_PLANE_Y_HALF,
      goal_plane_z_min=GOAL_PLANE_Z_MIN,
      goal_plane_z_max=GOAL_PLANE_Z_MAX,
      goal_termination_term_name="goal_conceded",
      goal_cue_flash_steps=18,
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
      weight=-300.0,
      params={"command_name": "stand_block"},
    ),
    "save_success": RewardTermCfg(
      func=mdp.save_success_reward,
      weight=120.0,
      params={
        "command_name": "stand_block",
        "resolution_term_name": "contact_resolution_window",
      },
    ),
    "deflect_away": RewardTermCfg(
      func=mdp.deflect_away_from_goal_reward,
      weight=40.0,
      params={"command_name": "stand_block", "only_on_first_contact": True},
    ),
    "outside_area": RewardTermCfg(
      func=mdp.outside_keeper_area_penalty,
      weight=-15.0,
      params={"command_name": "stand_block"},
    ),
    "fallen": RewardTermCfg(
      func=mdp.fallen_indicator,
      weight=-60.0,
      params={
        "min_height": 0.32,
        "max_tilt": 1.25,
      },
    ),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
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
        "max_tilt": 1.25,
        "consecutive_steps": 6,
      },
    ),
    "out_of_area_hard": TerminationTermCfg(
      func=mdp.outside_keeper_area_hard,
      params={"command_name": "stand_block"},
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
