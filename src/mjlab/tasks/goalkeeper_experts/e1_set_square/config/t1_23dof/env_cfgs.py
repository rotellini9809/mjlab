from __future__ import annotations

import os

from mjlab.asset_zoo.robots import T1_23_ACTION_SCALE, get_t1_23_robot_cfg
from mjlab.asset_zoo.robocup_assets.field import get_robocup_field_cfg
from mjlab.asset_zoo.robocup_assets.goalpost import get_robocup_goalpost_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.goalkeeper_experts.e1_set_square import mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.viewer import ViewerConfig

GOAL_X_LINE = 7
GOALPOST_X = 7.3
# E1 keeper spawn near goal mouth (world frame, before env origins).
KEEPER_SPAWN_X_RANGE = (GOAL_X_LINE - 0.2, GOAL_X_LINE + 0.2)
KEEPER_SPAWN_Y_RANGE = (-0.6, 0.6)

# Safe keeper area bounds (x_min, x_max, y_min, y_max).
KEEPER_AREA_BOUNDS = (GOAL_X_LINE - 1, GOAL_X_LINE, -2, 2)
KEEPER_AREA_HARD_MARGIN = 0.7

# Target ball spawn relative to keeper spawn.
TARGET_SPAWN_FORWARD_RANGE = (1.0, 2.7)
TARGET_SPAWN_LATERAL_RANGE = (-3, 3)
# Ball geometry in robocup asset: radius=0.11 -> height=0.22.
TARGET_BALL_HEIGHT = 0.22
# Enforce lower bound: z cannot go below ball_height / 2.
TARGET_BALL_Z_MIN = TARGET_BALL_HEIGHT / 2.0
# Exponential distribution scale (meters):
# smaller -> more concentration near ground, larger -> more high balls.
TARGET_BALL_Z_EXP_SCALE = 0.06
# Optional upper cap to avoid very high outliers.
TARGET_BALL_Z_MAX = 2.0
# Temporary debug: force constant z to verify ground spawning.
DEBUG_FORCE_TARGET_BALL_GROUND_Z = False
DEBUG_TARGET_BALL_GROUND_Z = TARGET_BALL_HEIGHT / 2.0

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

EPISODE_LENGTH_S = 8.0


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

  target_ball_cfg = mdp.get_target_ball_cfg()

  cfg.scene.terrain = None
  cfg.scene.num_envs = 512 if not play else 1
  cfg.scene.entities = {
    "robot": robot_cfg,
    "soccer_field": get_robocup_field_cfg(),
    "goalpost_left": goal_left_cfg,
    "goalpost_right": goal_right_cfg,
    "target_ball": target_ball_cfg,
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
      marker_entity_name="target_ball",
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
      p_ready=P_READY,
      spawn_yaw_range=SPAWN_YAW_RANGE,
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
    )
  }

  policy_terms = {
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
  }

  cfg.observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
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
