from copy import deepcopy

from mjlab.asset_zoo.robots import (
  T1_23_ACTION_SCALE,
  get_t1_23_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.push_getup import mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

TRUNK_HEIGHT_TARGET = 0.62
TRUNK_HEIGHT_MIN = 0.15
CURRICULUM_ENABLED = True
CURRICULUM_SUCCESS_RATE_THRESHOLD = 0.8
CURRICULUM_SUCCESS_RATE_EMA_ALPHA = 0.05
CURRICULUM_MIN_EPISODES = 200
CURRICULUM_MIN_STEPS = 0
CURRICULUM_TIME_TO_UPRIGHT_THRESHOLD_S = 0.0
CURRICULUM_STAGE1_MIX_INITIAL = 1.0
CURRICULUM_STAGE1_MIX_AFTER_UNLOCK = 0.2
CURRICULUM_STAGE1_MIX_MIN = 0.1
CURRICULUM_STAGE1_MIX_DECAY = 0.0005

STAGE1_RESET_POSE_MODE = "prone"
STAGE1_FALLEN_ROOT_POSE = (0.0, 0.0, -0.60, 0.0, 1.57, 0.0)
STAGE1_FALLEN_POSE_NOISE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
STAGE1_FALLEN_JOINT_POS = (
  -0.00012,
  -0.06750,
  0.24115,
  0.24163,
  0.59975,
  -0.01671,
  0.24115,
  -0.24163,
  0.59975,
  0.01671,
  0.00008,
  -0.18067,
  0.01076,
  -0.00197,
  0.62887,
  -0.37332,
  -0.00000,
  -0.18060,
  -0.01080,
  0.00183,
  0.62914,
  -0.37332,
  0.00000,
)
STAGE1_FALLEN_JOINT_NOISE = 0.0
SHOULDER_PITCH_INIT = 1.0
STAGE1_PUSH_ENABLED = False
STAGE1_POST_FALL_DELAY_S = 0.0
STAGE1_PROGRESS_K = 1.0
STAGE1_PROGRESS_DH_CLIP = 0.03
STAGE1_SELF_COLLISION_SCALE = 0.0
STAGE1_SELF_COLLISION_SUBTREE = "Waist"
CONTACT_HEIGHT_SWITCH = 0.7
SUPPORT_POINTS_WEIGHT = 0.6
SUPPORT_POINTS_FOOT_WEIGHT = 0.6
SUPPORT_POINTS_HAND_WEIGHT = 1.4
SUPPORT_POINTS_DH_CLIP = 0.04
FEET_CONTACT_NORMAL_THRESHOLD = 0.7
BOTH_FEET_WEIGHT = 0.5
HANDS_CONTACT_PENALTY_WEIGHT = 0.0
PELVIS_CONTACT_PENALTY_WEIGHT = -0.8
PELVIS_CONTACT_MIN_RECOVERY_STEPS = 20
HEAD_CONTACT_PENALTY_WEIGHT = -0.5
HEAD_CONTACT_HEIGHT_SWITCH = 0.25
HEAD_CONTACT_EARLY_SCALE = 0.2
HEAD_CONTACT_LATE_SCALE = 1.0
HAND_PUSH_WEIGHT = 0.0
HAND_PUSH_DH_CLIP = 0.04
STAND_POSE_W_FINAL = 0.4
STAND_POSE_HEIGHT_THRESHOLD = 0.8
STAND_POSE_Q_SCALE = 1.0
STAND_POSE_REQUIRE_PELVIS_OFF = True
STAND_POSE_EXCLUDE_JOINT_PATTERNS = (
  ".*Shoulder.*",
  ".*Elbow.*",
  ".*Wrist.*",
  ".*Hand.*",
)
STAND_POSE_RAMP_SUCCESS_THRESHOLD = 0.1
STAND_POSE_RAMP_STEPS = 2000
STAND_POSE_RAMP_STANDING_THRESHOLD = 0.5
STAND_POSE_RAMP_STANDING_EMA_ALPHA = 0.05
ENABLE_LOW_MOTION_TERMINATION = False
LOW_MOTION_LIN_VEL_THRESHOLD = 0.05
LOW_MOTION_ANG_VEL_THRESHOLD = 0.2
LOW_MOTION_CONSECUTIVE_STEPS = 40
LOW_MOTION_MIN_RECOVERY_STEPS = 50
LOW_MOTION_MIN_HEIGHT_NORM = 0.25

FEET_CONTACT_GEOMS = ("left_foot_collision", "right_foot_collision")
HAND_FOREARM_BODIES = ("left_hand_link", "right_hand_link", "AL3", "AR3")
PELVIS_BODIES = ("Waist", "Hip_Pitch_Left", "Hip_Pitch_Right")
HEAD_BODIES = ("H1", "H2")


def booster_t1_23_push_getup_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 push-fall-getup environment configuration."""
  cfg = make_tracking_env_cfg()

  # Robot: T1 23 DoF
  cfg.scene.entities = {"robot": get_t1_23_robot_cfg()}
  robot_cfg = cfg.scene.entities["robot"]
  if robot_cfg.init_state is not None:
    robot_cfg.init_state = deepcopy(robot_cfg.init_state)
    joint_pos = dict(robot_cfg.init_state.joint_pos or {})
    joint_pos["Left_Shoulder_Pitch"] = float(SHOULDER_PITCH_INIT)
    joint_pos["Right_Shoulder_Pitch"] = float(SHOULDER_PITCH_INIT)
    robot_cfg.init_state.joint_pos = joint_pos

  # Self-collision on waist subtree (pelvis equivalent for this model).
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Waist", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Waist", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  self_collision_stage1_cfg = ContactSensorCfg(
    name="self_collision_stage1",
    primary=ContactMatch(
      mode="subtree", pattern=STAGE1_SELF_COLLISION_SUBTREE, entity="robot"
    ),
    secondary=ContactMatch(
      mode="subtree", pattern=STAGE1_SELF_COLLISION_SUBTREE, entity="robot"
    ),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FEET_CONTACT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "normal"),
    reduce="maxforce",
    num_slots=1,
  )
  hand_forearm_ground_cfg = ContactSensorCfg(
    name="hand_forearm_ground_contact",
    primary=ContactMatch(mode="body", pattern=HAND_FOREARM_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  pelvis_ground_cfg = ContactSensorCfg(
    name="pelvis_ground_contact",
    primary=ContactMatch(mode="body", pattern=PELVIS_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  head_ground_cfg = ContactSensorCfg(
    name="head_ground_contact",
    primary=ContactMatch(mode="body", pattern=HEAD_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (
    self_collision_cfg,
    self_collision_stage1_cfg,
    feet_ground_cfg,
    hand_forearm_ground_cfg,
    pelvis_ground_cfg,
    head_ground_cfg,
  )

  # Actions: scale derived from T1 actuators.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = T1_23_ACTION_SCALE

  # Commands: fixed zero command (same dim as tracking command).
  cfg.commands = {
    "motion": mdp.FixedMotionCommandCfg(
      entity_name="robot",
      command_dim=46,
      push_delay_range_s=(0.2, 1.0),
      push_speed_range=(2.0, 3.8),
      push_yaw_rate_range=(-0.35, 0.35),
      limp_gain_scale=0.02,
      recover_gain_scale=1.0,
      post_fall_delay_s=2.0,
      trunk_height_target=TRUNK_HEIGHT_TARGET,
      trunk_height_min=TRUNK_HEIGHT_MIN,
      fallen_height=0.42,
      fallen_angle=0.95,
      upright_height=0.62,
      upright_angle=0.45,
      upright_hysteresis_steps=8,
      curriculum_enabled=CURRICULUM_ENABLED,
      curriculum_success_rate_threshold=CURRICULUM_SUCCESS_RATE_THRESHOLD,
      curriculum_success_rate_ema_alpha=CURRICULUM_SUCCESS_RATE_EMA_ALPHA,
      curriculum_min_episodes=CURRICULUM_MIN_EPISODES,
      curriculum_min_steps=CURRICULUM_MIN_STEPS,
      curriculum_time_to_upright_threshold_s=CURRICULUM_TIME_TO_UPRIGHT_THRESHOLD_S,
      curriculum_stage1_mix_initial=CURRICULUM_STAGE1_MIX_INITIAL,
      curriculum_stage1_mix_after_unlock=CURRICULUM_STAGE1_MIX_AFTER_UNLOCK,
      curriculum_stage1_mix_min=CURRICULUM_STAGE1_MIX_MIN,
      curriculum_stage1_mix_decay=CURRICULUM_STAGE1_MIX_DECAY,
      stage1_push_enabled=STAGE1_PUSH_ENABLED,
      stage2_push_enabled=True,
      stage1_post_fall_delay_s=STAGE1_POST_FALL_DELAY_S,
      stage2_post_fall_delay_s=None,
      stage1_reset_pose_mode=STAGE1_RESET_POSE_MODE,
      stage1_fallen_root_pose=STAGE1_FALLEN_ROOT_POSE,
      stage1_fallen_pose_noise=STAGE1_FALLEN_POSE_NOISE,
      stage1_fallen_joint_pos=STAGE1_FALLEN_JOINT_POS,
      stage1_fallen_joint_noise=STAGE1_FALLEN_JOINT_NOISE,
      stand_pose_ramp_success_threshold=STAND_POSE_RAMP_SUCCESS_THRESHOLD,
      stand_pose_ramp_steps=STAND_POSE_RAMP_STEPS,
      stand_pose_ramp_standing_threshold=STAND_POSE_RAMP_STANDING_THRESHOLD,
      stand_pose_ramp_standing_ema_alpha=STAND_POSE_RAMP_STANDING_EMA_ALPHA,
      stand_pose_height_threshold=STAND_POSE_HEIGHT_THRESHOLD,
      stand_pose_feet_sensor_name="feet_ground_contact",
      stand_pose_pelvis_sensor_name="pelvis_ground_contact",
      stand_pose_feet_normal_threshold=FEET_CONTACT_NORMAL_THRESHOLD,
      stand_pose_require_pelvis_off=STAND_POSE_REQUIRE_PELVIS_OFF,
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=False,
    )
  }

  # Observations (actor): proprioceptive student view only (no command/anchors).
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"biased": True}),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # Events: reset to default pose, keep optional randomization, disable generic push.
  cfg.events.pop("push_robot", None)
  if "foot_friction" in cfg.events:
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
      r"^(left|right)_foot_collision$"
    )
  if "base_com" in cfg.events:
    cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.events.update(
    {
      "reset_base": EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
          "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
          "velocity_range": {},
        },
      ),
      "reset_robot_joints": EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
          "position_range": (0.0, 0.0),
          "velocity_range": (0.0, 0.0),
          "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
      ),
    }
  )

  # Rewards
  cfg.rewards = {
    "success": RewardTermCfg(
      func=mdp.success_reward,
      weight=10.0,
      params={"term_name": "success"},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright_reward,
      weight=0.25,
      params={},
    ),
    "trunk_height": RewardTermCfg(
      func=mdp.trunk_height_reward,
      weight=0.35,
      params={
        "target_height": TRUNK_HEIGHT_TARGET,
        "min_height": TRUNK_HEIGHT_MIN,
      },
    ),
    "trunk_height_progress": RewardTermCfg(
      func=mdp.trunk_height_progress_reward,
      weight=2.0,
      params={
        "k": 1.0,
        "dh_clip": 0.04,
        "k_stage1": 1.0,
        "k_stage2": 1.0,
        "dh_clip_stage1": 0.04,
        "dh_clip_stage2": 0.04,
      },
    ),
    "support_points": RewardTermCfg(
      func=mdp.support_points_reward,
      weight=0.6,
      params={
        "feet_sensor_name": "feet_ground_contact",
        "hands_sensor_name": "hand_forearm_ground_contact",
        "height_threshold": CONTACT_HEIGHT_SWITCH,
        "foot_weight": SUPPORT_POINTS_FOOT_WEIGHT,
        "hand_weight": SUPPORT_POINTS_HAND_WEIGHT,
        "dh_clip": SUPPORT_POINTS_DH_CLIP,
        "feet_normal_threshold": FEET_CONTACT_NORMAL_THRESHOLD,
      },
    ),
    "hand_push": RewardTermCfg(
      func=mdp.hand_push_reward,
      weight=HAND_PUSH_WEIGHT,
      params={
        "hands_sensor_name": "hand_forearm_ground_contact",
        "height_threshold": CONTACT_HEIGHT_SWITCH,
        "dh_clip": HAND_PUSH_DH_CLIP,
      },
    ),
    "both_feet": RewardTermCfg(
      func=mdp.both_feet_reward,
      weight=0.5,
      params={
        "feet_sensor_name": "feet_ground_contact",
        "height_threshold": CONTACT_HEIGHT_SWITCH,
        "feet_normal_threshold": FEET_CONTACT_NORMAL_THRESHOLD,
      },
    ),
    "hands_contact_penalty": RewardTermCfg(
      func=mdp.hands_contact_penalty,
      weight=-0.15,
      params={
        "hands_sensor_name": "hand_forearm_ground_contact",
        "feet_sensor_name": "feet_ground_contact",
        "height_threshold": CONTACT_HEIGHT_SWITCH,
      },
    ),
    "pelvis_contact_penalty": RewardTermCfg(
      func=mdp.pelvis_contact_penalty,
      weight=-0.8,
      params={
        "pelvis_sensor_name": "pelvis_ground_contact",
        "height_threshold": CONTACT_HEIGHT_SWITCH,
        "min_recovery_steps": PELVIS_CONTACT_MIN_RECOVERY_STEPS,
      },
    ),
    "head_contact_penalty": RewardTermCfg(
      func=mdp.head_contact_penalty,
      weight=HEAD_CONTACT_PENALTY_WEIGHT,
      params={
        "head_sensor_name": "head_ground_contact",
        "height_threshold": HEAD_CONTACT_HEIGHT_SWITCH,
        "early_scale": HEAD_CONTACT_EARLY_SCALE,
        "late_scale": HEAD_CONTACT_LATE_SCALE,
      },
    ),
    "stand_pose_penalty": RewardTermCfg(
      func=mdp.stand_pose_penalty,
      weight=-STAND_POSE_W_FINAL,
      params={
        "feet_sensor_name": "feet_ground_contact",
        "pelvis_sensor_name": "pelvis_ground_contact",
        "height_threshold": STAND_POSE_HEIGHT_THRESHOLD,
        "q_scale": STAND_POSE_Q_SCALE,
        "require_pelvis_off": STAND_POSE_REQUIRE_PELVIS_OFF,
        "exclude_joint_patterns": STAND_POSE_EXCLUDE_JOINT_PATTERNS,
      },
    ),
    "ang_vel_penalty": RewardTermCfg(
      func=mdp.angular_velocity_penalty,
      weight=-0.02,
      params={},
    ),
    "recovery_step_penalty": RewardTermCfg(
      func=mdp.recovery_step_penalty,
      weight=-0.002,
      params={},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.5,
      params={
        "sensor_name": "self_collision",
        "stage1_sensor_name": "self_collision_stage1",
        "stage1_scale": STAGE1_SELF_COLLISION_SCALE,
        "stage2_scale": 1.0,
      },
    ),
  }

  # Terminations
  success_term = mdp.UprightSuccess(
    height_threshold=0.62,
    angle_threshold=0.45,
    consecutive_steps=8,
    require_fallen=True,
  )
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "success": TerminationTermCfg(func=success_term),
  }
  if ENABLE_LOW_MOTION_TERMINATION:
    low_motion_term = mdp.LowMotionTermination(
      lin_vel_threshold=LOW_MOTION_LIN_VEL_THRESHOLD,
      ang_vel_threshold=LOW_MOTION_ANG_VEL_THRESHOLD,
      consecutive_steps=LOW_MOTION_CONSECUTIVE_STEPS,
      min_recovery_steps=LOW_MOTION_MIN_RECOVERY_STEPS,
      min_height_norm=LOW_MOTION_MIN_HEIGHT_NORM,
    )
    cfg.terminations["low_motion"] = TerminationTermCfg(func=low_motion_term)

  cfg.viewer.body_name = "Trunk"
  cfg.episode_length_s = 10.0
  cfg.sim.mujoco.ccd_iterations = 100

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
