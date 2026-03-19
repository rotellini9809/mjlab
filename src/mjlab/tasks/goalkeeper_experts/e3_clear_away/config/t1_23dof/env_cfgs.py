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
from mjlab.tasks.goalkeeper_experts.e3_clear_away import mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

GOAL_X_LINE = 7.0
GOALPOST_X = 7.3

# E3 keeper spawn near defended goal side, with mild lateral spread.
KEEPER_SPAWN_X_RANGE = (GOAL_X_LINE - 0.20, GOAL_X_LINE + 0.15)
KEEPER_SPAWN_Y_RANGE = (-0.50, 0.50)
KEEPER_SPAWN_Z = 0.658
SPAWN_YAW_RANGE = (
  3.141592653589793 - 0.20,
  3.141592653589793 + 0.20,
)

KEEPER_JOINT_POS_NOISE = 0.02
KEEPER_JOINT_VEL_NOISE = 0.08
POST_CONTACT_KEEPER_JOINT_POS_NOISE = 0.05
POST_CONTACT_KEEPER_JOINT_VEL_NOISE = 0.18

# Keeper area bounds (x_min, x_max, y_min, y_max).
KEEPER_AREA_BOUNDS = (GOAL_X_LINE - 1.0, GOAL_X_LINE + 0.6, -2.0, 2.0)
KEEPER_AREA_HARD_MARGIN = 0.35

# Goal-plane aperture.
GOAL_PLANE_X = GOAL_X_LINE
GOAL_PLANE_Y_CENTER = 0.0
GOAL_PLANE_Y_HALF = 1.30
GOAL_PLANE_Z_MIN = 0.0
GOAL_PLANE_Z_MAX = 1.85

# Danger zone for clear-away.
DANGER_ZONE_DEPTH = 1
DANGER_ZONE_HALF_WIDTH = 1.55
DANGER_ZONE_REQUIRE_TOWARD_GOAL = False
DANGER_ZONE_TOWARD_GOAL_SPEED_THRESHOLD = 0.05

# E3 reset mix.
LOOSE_VARIANT_PROB = 0.60
LOOSE_BALL_SPEED_RANGE = (0.0, 0.8)
LOOSE_BALL_TOWARD_GOAL_PROB = 0.35
LOOSE_BALL_ANGLE_NOISE_DEG = 65.0
LOOSE_BALL_Z_RANGE = (0.11, 0.18)
LOOSE_BALL_X_MARGIN = 0.06
LOOSE_BALL_Y_MARGIN = 0.10

POST_CONTACT_SURFACE_PROBS = (0.40, 0.25, 0.20, 0.15)
POST_CONTACT_X_OFFSETS = (-0.34, -0.26, -0.22, -0.20)
POST_CONTACT_Y_OFFSETS = (0.16, 0.20, 0.32, 0.10)
POST_CONTACT_Z_OFFSETS = (0.11, 0.28, 0.88, 0.72)
POST_CONTACT_OFFSET_NOISE_XY = 0.06
POST_CONTACT_OFFSET_NOISE_Z = 0.05
POST_CONTACT_REBOUND_SPEED_RANGE = (0.0, 1.8)
POST_CONTACT_REBOUND_ZERO_PROB = 0.25
POST_CONTACT_REBOUND_AWAY_PROB = 0.65
POST_CONTACT_REBOUND_ANGLE_NOISE_DEG = 55.0
POST_CONTACT_REBOUND_VZ_RANGE = (-0.45, 0.45)

# Clear-condition thresholds.
CLEAR_PROGRESS_STEPS = 6
CLEAR_DISTANCE_INCREASE_THRESHOLD = 0.008
CLEAR_STRONG_AWAY_SPEED = 1.20

# Visual overlays.
GOAL_PLANE_VIS_HALF_THICKNESS = 0.005
GOAL_PLANE_VIS_RGBA = (0.15, 0.85, 0.95, 0.08)
DANGER_ZONE_OVERLAY_HALF_THICKNESS = 0.0018
DANGER_ZONE_OVERLAY_Z = 0.005
DANGER_ZONE_OVERLAY_RGBA = (0.95, 0.10, 0.10, 0.24)
E3_AREA_OVERLAY_HALF_THICKNESS = 0.0015
E3_HARD_AREA_OVERLAY_Z = 0.0015
E3_KEEPER_AREA_OVERLAY_Z = 0.0035
E3_HARD_AREA_RGBA = (0.95, 0.55, 0.10, 0.22)
E3_KEEPER_AREA_RGBA = (0.05, 0.60, 0.95, 0.30)

# Stage-1 command dimension used in motor-observation layout.
MOTOR_COMMAND_DIM = 46
MOTOR_ACT_DIM = 23

EPISODE_LENGTH_S = 2.5
SIM_TIMESTEP_S = 0.005
CONTROL_DECIMATION = 4


def _add_goal_danger_and_area_overlays(spec: mujoco.MjSpec) -> None:
  overlay_body = next(
    (body for body in spec.bodies if body.name == "e3_field_overlays"), None
  )
  if overlay_body is None:
    overlay_body = spec.worldbody.add_body(name="e3_field_overlays")

  center_z = 0.5 * (GOAL_PLANE_Z_MIN + GOAL_PLANE_Z_MAX)
  half_z = max(0.5 * (GOAL_PLANE_Z_MAX - GOAL_PLANE_Z_MIN), 1.0e-3)

  goal_overlay = overlay_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(GOAL_PLANE_X, GOAL_PLANE_Y_CENTER, center_z),
    size=(GOAL_PLANE_VIS_HALF_THICKNESS, GOAL_PLANE_Y_HALF, half_z),
  )
  goal_overlay.name = "e3_goal_plane_overlay"
  goal_overlay.rgba = GOAL_PLANE_VIS_RGBA
  goal_overlay.contype = 0
  goal_overlay.conaffinity = 0


def get_e3_field_cfg_with_overlays() -> EntityCfg:
  field_cfg = get_robocup_field_cfg()
  base_spec_fn = field_cfg.spec_fn

  def _spec_fn() -> mujoco.MjSpec:
    spec = base_spec_fn()
    _add_goal_danger_and_area_overlays(spec)
    return spec

  field_cfg.spec_fn = _spec_fn
  return field_cfg


def booster_t1_23_gk_expert_clear_away_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  robot_cfg = get_t1_23_robot_cfg()
  robot_cfg.init_state.pos = (GOAL_X_LINE - 0.2, 0.0, KEEPER_SPAWN_Z)
  # Face toward field center from defended +x goal side.
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
    "soccer_field": get_e3_field_cfg_with_overlays(),
    "goalpost_left": goal_left_cfg,
    "goalpost_right": goal_right_cfg,
    "soccer_ball": soccer_ball_cfg,
  }

  motor_obs_terms, motor_obs_term_dims = mdp.default_motor_obs_layout(
    act_dim=MOTOR_ACT_DIM,
  )

  cfg.actions = {
    "motor_latent": mdp.MotorLatentActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=T1_23_ACTION_SCALE,
      use_default_offset=True,
      command_name="clear_away",
      stage1_wandb_run_path=os.environ.get("MJLAB_STAGE1_WANDB_RUN_PATH_GOALKEEPER"),
      motor_obs_terms=motor_obs_terms,
      motor_obs_term_dims=motor_obs_term_dims,
      strict_obs_layout=True,
    )
  }

  cfg.commands = {
    "clear_away": mdp.ClearAwayCommandCfg(
      entity_name="robot",
      ball_entity_name="soccer_ball",
      command_dim=MOTOR_COMMAND_DIM,
      keeper_spawn_x_range=KEEPER_SPAWN_X_RANGE,
      keeper_spawn_y_range=KEEPER_SPAWN_Y_RANGE,
      spawn_yaw_range=SPAWN_YAW_RANGE,
      keeper_joint_pos_noise=KEEPER_JOINT_POS_NOISE,
      keeper_joint_vel_noise=KEEPER_JOINT_VEL_NOISE,
      post_contact_keeper_joint_pos_noise=POST_CONTACT_KEEPER_JOINT_POS_NOISE,
      post_contact_keeper_joint_vel_noise=POST_CONTACT_KEEPER_JOINT_VEL_NOISE,
      keeper_area_bounds=KEEPER_AREA_BOUNDS,
      hard_area_margin=KEEPER_AREA_HARD_MARGIN,
      goal_toward_positive_x=True,
      goal_plane_x=GOAL_PLANE_X,
      goal_plane_y_center=GOAL_PLANE_Y_CENTER,
      goal_plane_y_half=GOAL_PLANE_Y_HALF,
      goal_plane_z_min=GOAL_PLANE_Z_MIN,
      goal_plane_z_max=GOAL_PLANE_Z_MAX,
      danger_zone_depth=DANGER_ZONE_DEPTH,
      danger_zone_half_width=DANGER_ZONE_HALF_WIDTH,
      danger_zone_require_toward_goal=DANGER_ZONE_REQUIRE_TOWARD_GOAL,
      danger_zone_toward_goal_speed_threshold=DANGER_ZONE_TOWARD_GOAL_SPEED_THRESHOLD,
      loose_variant_prob=LOOSE_VARIANT_PROB,
      loose_ball_speed_range=LOOSE_BALL_SPEED_RANGE,
      loose_ball_toward_goal_prob=LOOSE_BALL_TOWARD_GOAL_PROB,
      loose_ball_angle_noise_deg=LOOSE_BALL_ANGLE_NOISE_DEG,
      loose_ball_z_range=LOOSE_BALL_Z_RANGE,
      loose_ball_x_margin=LOOSE_BALL_X_MARGIN,
      loose_ball_y_margin=LOOSE_BALL_Y_MARGIN,
      post_contact_surface_probs=POST_CONTACT_SURFACE_PROBS,
      post_contact_x_offsets=POST_CONTACT_X_OFFSETS,
      post_contact_y_offsets=POST_CONTACT_Y_OFFSETS,
      post_contact_z_offsets=POST_CONTACT_Z_OFFSETS,
      post_contact_offset_noise_xy=POST_CONTACT_OFFSET_NOISE_XY,
      post_contact_offset_noise_z=POST_CONTACT_OFFSET_NOISE_Z,
      post_contact_rebound_speed_range=POST_CONTACT_REBOUND_SPEED_RANGE,
      post_contact_rebound_zero_prob=POST_CONTACT_REBOUND_ZERO_PROB,
      post_contact_rebound_away_prob=POST_CONTACT_REBOUND_AWAY_PROB,
      post_contact_rebound_angle_noise_deg=POST_CONTACT_REBOUND_ANGLE_NOISE_DEG,
      post_contact_rebound_vz_range=POST_CONTACT_REBOUND_VZ_RANGE,
      clear_progress_steps=CLEAR_PROGRESS_STEPS,
      clear_distance_increase_threshold=CLEAR_DISTANCE_INCREASE_THRESHOLD,
      clear_strong_away_speed=CLEAR_STRONG_AWAY_SPEED,
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
      params={"command_name": "clear_away"},
    ),
    "ball_pos_rel_xyz": ObservationTermCfg(
      func=mdp.ball_position_relative_xyz,
      params={"command_name": "clear_away"},
    ),
    "ball_vel_rel_xyz": ObservationTermCfg(
      func=mdp.ball_velocity_relative_xyz,
      params={"command_name": "clear_away"},
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
      params={"command_name": "clear_away"},
    ),
    "ball_pos_rel_xyz": ObservationTermCfg(
      func=mdp.ball_position_relative_xyz,
      params={"command_name": "clear_away"},
    ),
    "ball_vel_rel_xyz": ObservationTermCfg(
      func=mdp.ball_velocity_relative_xyz,
      params={"command_name": "clear_away"},
    ),
    "ball_goal_dist_xy": ObservationTermCfg(
      func=mdp.ball_goal_distance_xy,
      params={"command_name": "clear_away"},
    ),
    "ball_in_dz": ObservationTermCfg(
      func=mdp.ball_in_danger_zone,
      params={"command_name": "clear_away", "require_toward_goal": False},
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
      weight=-320.0,
      params={"command_name": "clear_away"},
    ),
    "clear_success": RewardTermCfg(
      func=mdp.clear_success_reward,
      weight=150.0,
      params={"command_name": "clear_away", "clear_term_name": "clear_condition"},
    ),
    "clear_progress": RewardTermCfg(
      func=mdp.distance_from_goal_progress_reward,
      weight=18.0,
      params={"command_name": "clear_away", "clip_speed": 3.0},
    ),
    "outside_dz_bonus": RewardTermCfg(
      func=mdp.outside_danger_zone_bonus,
      weight=6.0,
      params={"command_name": "clear_away", "clip_speed": 2.5},
    ),
    "away_velocity": RewardTermCfg(
      func=mdp.away_velocity_reward,
      weight=10.0,
      params={"command_name": "clear_away", "clip_speed": 4.0},
    ),
    "outside_area": RewardTermCfg(
      func=mdp.outside_keeper_area_penalty,
      weight=-14.0,
      params={"command_name": "clear_away"},
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
      params={"command_name": "clear_away"},
    ),
    "clear_condition": TerminationTermCfg(
      func=mdp.ClearConditionTermination,
      params={
        "command_name": "clear_away",
        "required_steps": CLEAR_PROGRESS_STEPS,
        "min_distance_increase": CLEAR_DISTANCE_INCREASE_THRESHOLD,
        "strong_away_speed": CLEAR_STRONG_AWAY_SPEED,
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
      params={"command_name": "clear_away"},
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
    lookat=(GOAL_X_LINE - 1.2, 0.0, 0.9),
    distance=6.8,
    elevation=-18.0,
    azimuth=178.0,
  )

  cfg.sim.mujoco.timestep = SIM_TIMESTEP_S
  cfg.decimation = CONTROL_DECIMATION
  cfg.episode_length_s = EPISODE_LENGTH_S

  return cfg
