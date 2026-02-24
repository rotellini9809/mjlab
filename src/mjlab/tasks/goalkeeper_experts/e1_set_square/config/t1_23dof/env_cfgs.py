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
from mjlab.tasks.goalkeeper_experts.e1_set_square import mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.viewer import ViewerConfig

GOAL_X_LINE = 7
GOALPOST_X = 7.3
# E1 keeper spawn near goal mouth (world frame, before env origins).
KEEPER_SPAWN_X_RANGE = (GOAL_X_LINE - 0.2, GOAL_X_LINE + 0.2)
KEEPER_SPAWN_Y_RANGE = (-0.6, 0.6)

# Safe keeper area bounds (x_min, x_max, y_min, y_max).
KEEPER_AREA_BOUNDS = (GOAL_X_LINE - 1, GOAL_X_LINE + 0.5, -2, 2)
KEEPER_AREA_HARD_MARGIN = 0.3

# Target ball spawn relative to keeper spawn.
# With keeper spawn centered at x~=+7, mean forward=7 centers mean ball spawn at field x~=0.
TARGET_SPAWN_FORWARD_RANGE = (3.0, 11.0)
TARGET_SPAWN_LATERAL_RANGE = (-3.8, 3.8)
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

# E1 kick sampler (single kick, lateral-biased, anti-shot).
KICK_DEAD_BALL_PROB = 0.40
KICK_LATERAL_ROLL_PROB = 0.40
KICK_DEAD_BALL_TINY_DRIFT_PROB = 0.20
KICK_DEAD_BALL_DRIFT_SPEED_RANGE = (0.02, 0.10)
KICK_SPEED_RANGE = (0.4, 1.6)
KICK_ANGLE_NOISE_DEG = 75.0
DRIBBLE_NUM_TAPS_RANGE = (2, 4)
DRIBBLE_TAP_TIME_RANGE = (0.5, 1.4)
DRIBBLE_TAP_INTERVAL_RANGE = (0.16, 0.64)
DRIBBLE_TAP_SPEED_RANGE = (0.2, 0.6)
MAX_TOWARD_GOAL_VX = 0.2

# Future reset curriculum hook (for now always default pose).
P_READY = 0.0

# Keeper spawn yaw (rad). Use pi to face opposite field side.
SPAWN_YAW_RANGE = (3.141592653589793, 3.141592653589793)

# Stage-1 command dimension used in motor-observation layout.
MOTOR_COMMAND_DIM = 46
# IMPORTANT:
#   Stage-1 decoder expects joint/action dims over actuated joints (23 for T1_23),
#   not the number of regex groups in T1_23_ACTION_SCALE.
MOTOR_ACT_DIM = 23

EPISODE_LENGTH_S = 4.0

# E1 test walls around the real 14x9 playable area:
# - 2 continuous walls on long sides (y = +/- 4.5),
# - 2 segmented walls per short side (x = +/- 7.0) leaving goal opening at y ~= 0.
# Walls are centered on this boundary.
FIELD_HALF_LENGTH_X = 7.0
FIELD_HALF_WIDTH_Y = 4.5
E1_WALL_THICKNESS = 0.16
E1_WALL_HEIGHT = 0.07
E1_GOAL_OPENING_HALF_WIDTH = 1.55
E1_WALL_RGBA = (0.92, 0.18, 0.18, 0.45)
E1_WALL_FRICTION = (1.2, 0.02, 0.002)
E1_WALL_SOLREF = (0.02, 1.5)
E1_WALL_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
E1_AREA_OVERLAY_HALF_THICKNESS = 0.0015
E1_HARD_AREA_OVERLAY_Z = 0.0015
E1_KEEPER_AREA_OVERLAY_Z = 0.0035
E1_HARD_AREA_RGBA = (0.95, 0.55, 0.10, 0.22)
E1_KEEPER_AREA_RGBA = (0.05, 0.60, 0.95, 0.30)
BALL_CURB_CONTACT_SENSOR_NAME = "ball_curb_contact"


def _add_e1_test_walls(spec: mujoco.MjSpec) -> None:
  field_body = next((body for body in spec.bodies if body.name == "field"), None)
  if field_body is None:
    field_body = spec.worldbody.add_body(name="field")

  half_t = E1_WALL_THICKNESS / 2.0
  half_h = E1_WALL_HEIGHT / 2.0
  wall_z = half_h

  def _add_wall(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
  ) -> None:
    wall = field_body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=size,
    )
    wall.name = name
    wall.rgba = E1_WALL_RGBA
    wall.friction = E1_WALL_FRICTION
    wall.solref = E1_WALL_SOLREF
    wall.solimp = E1_WALL_SOLIMP

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
  short_side_segment_half_y = (FIELD_HALF_WIDTH_Y - E1_GOAL_OPENING_HALF_WIDTH) / 2.0
  if short_side_segment_half_y <= 0.0:
    raise ValueError("E1_GOAL_OPENING_HALF_WIDTH is too large for field width.")
  short_side_segment_center_y = E1_GOAL_OPENING_HALF_WIDTH + short_side_segment_half_y
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

    overlay = field_body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=(center_x, center_y, z_center),
      size=(half_x, half_y, E1_AREA_OVERLAY_HALF_THICKNESS),
    )
    overlay.name = name
    overlay.rgba = rgba
    overlay.contype = 0
    overlay.conaffinity = 0

  hard_bounds = (
    KEEPER_AREA_BOUNDS[0] - KEEPER_AREA_HARD_MARGIN,
    KEEPER_AREA_BOUNDS[1] + KEEPER_AREA_HARD_MARGIN,
    KEEPER_AREA_BOUNDS[2] - KEEPER_AREA_HARD_MARGIN,
    KEEPER_AREA_BOUNDS[3] + KEEPER_AREA_HARD_MARGIN,
  )
  _add_area_overlay(
    "e1_keeper_area_hard_overlay",
    hard_bounds,
    E1_HARD_AREA_OVERLAY_Z,
    E1_HARD_AREA_RGBA,
  )
  _add_area_overlay(
    "e1_keeper_area_overlay",
    KEEPER_AREA_BOUNDS,
    E1_KEEPER_AREA_OVERLAY_Z,
    E1_KEEPER_AREA_RGBA,
  )


def get_e1_field_cfg_with_test_walls() -> EntityCfg:
  field_cfg = get_robocup_field_cfg()
  base_spec_fn = field_cfg.spec_fn

  def _spec_fn() -> mujoco.MjSpec:
    spec = base_spec_fn()
    _add_e1_test_walls(spec)
    return spec

  field_cfg.spec_fn = _spec_fn
  return field_cfg


def booster_t1_23_gk_expert_set_square_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  robot_cfg = get_t1_23_robot_cfg()
  # Keep default keyframe pose but place initial robot near goal.
  robot_cfg.init_state.pos = (GOAL_X_LINE - 0.3, 0.0, robot_cfg.init_state.pos[2])
  # Rotate keeper by 180 deg around z so it faces the opposite field side.
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
    "soccer_field": get_e1_field_cfg_with_test_walls(),
    "goalpost_left": goal_left_cfg,
    "goalpost_right": goal_right_cfg,
    "soccer_ball": soccer_ball_cfg,
  }
  ball_curb_contact_cfg = ContactSensorCfg(
    name=BALL_CURB_CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern="e1_wall_.*", entity="soccer_field"),
    secondary=ContactMatch(mode="geom", pattern="ball_collision", entity="soccer_ball"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, ball_curb_contact_cfg)

  motor_obs_terms, motor_obs_term_dims = mdp.default_motor_obs_layout(
    act_dim=MOTOR_ACT_DIM,
  )

  cfg.actions = {
    "motor_latent": mdp.MotorLatentActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=T1_23_ACTION_SCALE,
      use_default_offset=True,
      command_name="set_square",
      stage1_wandb_run_path=os.environ.get("MJLAB_STAGE1_WANDB_RUN_PATH"),
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
      target_forward_range=TARGET_SPAWN_FORWARD_RANGE,
      target_lateral_range=TARGET_SPAWN_LATERAL_RANGE,
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
      dribble_num_taps_range=DRIBBLE_NUM_TAPS_RANGE,
      dribble_tap_time_range=DRIBBLE_TAP_TIME_RANGE,
      dribble_tap_interval_range=DRIBBLE_TAP_INTERVAL_RANGE,
      dribble_tap_speed_range=DRIBBLE_TAP_SPEED_RANGE,
      goal_toward_positive_x=True,
      max_toward_goal_speed=MAX_TOWARD_GOAL_VX,
      p_ready=P_READY,
      spawn_yaw_range=SPAWN_YAW_RANGE,
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
    "ball_pos_rel_xyz": ObservationTermCfg(
      func=mdp.target_position_relative_xyz,
      params={"command_name": "set_square"},
    ),
    "ball_vel_rel_xyz": ObservationTermCfg(
      func=mdp.ball_velocity_relative_xyz,
      params={"command_name": "set_square"},
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
    "yaw_align": RewardTermCfg(
      func=mdp.yaw_alignment_reward,
      weight=2.5,
      params={"command_name": "set_square", "k": 2.5},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright_stability_reward,
      weight=1.0,
      params={"height_target": robot_cfg.init_state.pos[2]},
    ),
    "drift": RewardTermCfg(
      func=mdp.xy_drift_l2,
      weight=-0.45,
      params={"command_name": "set_square"},
    ),
    "xy_speed": RewardTermCfg(
      func=mdp.xy_speed_l2,
      weight=-0.06,
      params={},
    ),
    "outside_area": RewardTermCfg(
      func=mdp.outside_keeper_area_penalty,
      weight=-2.0,
      params={"command_name": "set_square"},
    ),
    "fallen": RewardTermCfg(
      func=mdp.fallen_indicator,
      weight=-8.0,
      params={
        "min_height": 0.32,
        "max_tilt": 1.25,
      },
    ),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
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
      params={"command_name": "set_square"},
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

  cfg.episode_length_s = EPISODE_LENGTH_S

  return cfg
