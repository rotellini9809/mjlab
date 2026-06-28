import os

import mujoco
from mjlab.entity import EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab.asset_zoo.robots import T1_23_ACTION_SCALE, get_t1_23_robot_cfg
from mjlab.asset_zoo.robocup_assets.ball import get_robocup_ball_cfg
from mjlab.asset_zoo.robocup_assets.field import get_robocup_field_cfg
from mjlab.asset_zoo.robocup_assets.goalpost import get_robocup_goalpost_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from mjlab.tasks.penalty_expert.d1_dribbling import mdp


# Task name to register outside this file:
#   Mjlab-Dribbling-Booster-T1_23 -> booster_t1_23_dribbling_env_cfg

GOAL_X_LINE = 7.0
GOALPOST_X = 7.3
FIELD_HALF_LENGTH_X = 7.0
FIELD_HALF_WIDTH_Y = 4.5
BALL_R = 0.11
BALL_Z = BALL_R

# Midfield start -> attacking penalty-area entry.
BALL_START_X = 0.0
ROBOT_BEHIND_BALL = 0.50
TARGET_X = 4.50
SUCCESS_BALL_X = 4.50
SUCCESS_ROBOT_X = 4.15

MOTOR_COMMAND_DIM = 46
MOTOR_ACT_DIM = 23
SIM_TIMESTEP_S = 0.005
CONTROL_DECIMATION = 4
EPISODE_LENGTH_S = 18.0

NUM_OBSTACLES = 5
OBSTACLE_RADIUS = 0.08
OBSTACLE_HEIGHT = 0.70
OBSTACLE_BALL_SAFE_DIST = 0.80
OBSTACLE_ROBOT_HIT_DIST = 0.28

DRIBBLE_AREA_BOUNDS = (-0.90, TARGET_X + 0.25, -4.0, 4.0)
DRIBBLE_AREA_HARD_MARGIN = 0.45

FIELD_BODY_NAME = "field"
WALL_THICKNESS = 0.16
WALL_HEIGHT = 0.07
GOAL_OPENING_HALF_WIDTH = 1.55
WALL_RGBA = (0.92, 0.18, 0.18, 0.45)
WALL_FRICTION = (1.2, 0.02, 0.002)
WALL_SOLREF = (0.02, 1.5)
WALL_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)

# Mini-grid for friction tuning (select via MJLAB_DRIBBLE_FRICTION_PRESET).
FRICTION_PRESET = os.environ.get("MJLAB_DRIBBLE_FRICTION_PRESET", "b_balanced").strip().lower()
FRICTION_PRESETS: dict[str, dict[str, tuple[float, float, float]]] = {
  "a_soft": {
    "ball": (0.85, 0.006, 0.0002),
    "foot": (1.00, 0.015, 0.0008),
    "terrain": (0.95, 0.008, 0.0008),
  },
  "b_balanced": {
    "ball": (0.90, 0.008, 0.0003),
    "foot": (1.10, 0.020, 0.0010),
    "terrain": (1.00, 0.010, 0.0010),
  },
  "c_grip_high": {
    "ball": (0.95, 0.010, 0.0005),
    "foot": (1.25, 0.030, 0.0015),
    "terrain": (1.10, 0.015, 0.0012),
  },
}
if FRICTION_PRESET not in FRICTION_PRESETS:
  raise ValueError(
    f"Invalid MJLAB_DRIBBLE_FRICTION_PRESET='{FRICTION_PRESET}'. "
    f"Valid presets: {tuple(FRICTION_PRESETS.keys())}."
  )
_friction_cfg = FRICTION_PRESETS[FRICTION_PRESET]
BALL_FRICTION = _friction_cfg["ball"]
FOOT_FRICTION = _friction_cfg["foot"]
TERRAIN_FRICTION = _friction_cfg["terrain"]


def _add_field_walls_and_area(spec: mujoco.MjSpec) -> None:
  field_body = next((body for body in spec.bodies if body.name == FIELD_BODY_NAME), None)
  if field_body is None:
    field_body = spec.worldbody.add_body(name=FIELD_BODY_NAME)

  half_t = WALL_THICKNESS / 2.0
  half_h = WALL_HEIGHT / 2.0
  z = half_h

  def _wall(name, pos, size):
    geom = field_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)
    geom.name = name
    geom.rgba = WALL_RGBA
    geom.friction = WALL_FRICTION
    geom.solref = WALL_SOLREF
    geom.solimp = WALL_SOLIMP

  _wall("dribble_wall_long_pos_y", (0.0, FIELD_HALF_WIDTH_Y, z), (FIELD_HALF_LENGTH_X, half_t, half_h))
  _wall("dribble_wall_long_neg_y", (0.0, -FIELD_HALF_WIDTH_Y, z), (FIELD_HALF_LENGTH_X, half_t, half_h))

  seg_half_y = (FIELD_HALF_WIDTH_Y - GOAL_OPENING_HALF_WIDTH) / 2.0
  seg_center_y = GOAL_OPENING_HALF_WIDTH + seg_half_y
  _wall("dribble_wall_short_pos_x_upper", (FIELD_HALF_LENGTH_X, seg_center_y, z), (half_t, seg_half_y, half_h))
  _wall("dribble_wall_short_pos_x_lower", (FIELD_HALF_LENGTH_X, -seg_center_y, z), (half_t, seg_half_y, half_h))
  _wall("dribble_wall_short_neg_x_upper", (-FIELD_HALF_LENGTH_X, seg_center_y, z), (half_t, seg_half_y, half_h))
  _wall("dribble_wall_short_neg_x_lower", (-FIELD_HALF_LENGTH_X, -seg_center_y, z), (half_t, seg_half_y, half_h))

  # transparent target/dribble corridor overlays: non-colliding visual helpers.
  def _overlay(name, bounds, zc, rgba):
    x0, x1, y0, y1 = bounds
    geom = field_body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=((x0 + x1) * 0.5, (y0 + y1) * 0.5, zc),
      size=((x1 - x0) * 0.5, (y1 - y0) * 0.5, 0.0015),
    )
    geom.name = name
    geom.rgba = rgba
    geom.contype = 0
    geom.conaffinity = 0

  x0, x1, y0, y1 = DRIBBLE_AREA_BOUNDS
  m = DRIBBLE_AREA_HARD_MARGIN
  _overlay("dribble_area_hard_overlay", (x0 - m, x1 + m, y0 - m, y1 + m), 0.0015, (0.95, 0.55, 0.10, 0.18))
  _overlay("dribble_area_overlay", DRIBBLE_AREA_BOUNDS, 0.0035, (0.05, 0.60, 0.95, 0.25))
  _overlay("penalty_entry_line_overlay", (TARGET_X - 0.035, TARGET_X + 0.035, -3.25, 3.25), 0.0060, (0.1, 0.95, 0.1, 0.45))


def get_dribbling_field_cfg() -> EntityCfg:
  field_cfg = get_robocup_field_cfg()
  base_spec_fn = field_cfg.spec_fn
  def _spec_fn() -> mujoco.MjSpec:
    spec = base_spec_fn()
    _add_field_walls_and_area(spec)
    return spec
  field_cfg.spec_fn = _spec_fn
  return field_cfg


def get_dribble_obstacle_cfg(index: int) -> EntityCfg:
  """Kinematic-looking cone/pole entity. Command reset moves each obstacle per env."""
  def _spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name=f"dribble_obstacle_{index}_root", pos=(0.0, 0.0, OBSTACLE_HEIGHT * 0.5))
    body = spec.worldbody.bodies[-1]
    body.add_freejoint(name=f"dribble_obstacle_{index}_freejoint")
    geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      name=f"dribble_obstacle_{index}_collision",
      size=(OBSTACLE_RADIUS, OBSTACLE_HEIGHT * 0.5),
      rgba=(1.0, 0.45, 0.05, 1.0),
      friction=(1.0, 0.01, 0.001),
      density=5000.0,
    )
    geom.contype = 1
    geom.conaffinity = 1
    return spec
  return EntityCfg(spec_fn=_spec_fn)


def booster_t1_23_dribbling_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  # Remove base tracking terminations tied to "motion" commands.
  to_remove = []
  for name, term in cfg.terminations.items():
    params = getattr(term, "params", {}) or {}
    if params.get("command_name") == "motion":
      to_remove.append(name)
  for name in to_remove:
    cfg.terminations.pop(name, None)

  robot_cfg = get_t1_23_robot_cfg()
  robot_cfg.init_state.pos = (BALL_START_X - ROBOT_BEHIND_BALL, 0.0, robot_cfg.init_state.pos[2])
  robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
  robot_cfg.collisions = (
    CollisionCfg(
      geom_names_expr=(r"^(left|right)_foot_collision$",),
      friction=FOOT_FRICTION,
      disable_other_geoms=False,
    ),
  )

  ball_cfg = get_robocup_ball_cfg()
  ball_cfg.init_state.pos = (BALL_START_X, 0.0, BALL_Z)
  ball_cfg.collisions = (
    CollisionCfg(
      geom_names_expr=(r"^ball_collision$",),
      friction=BALL_FRICTION,
      disable_other_geoms=False,
    ),
  )

  goal_left_cfg = get_robocup_goalpost_cfg()
  goal_left_cfg.init_state.pos = (GOALPOST_X, 0.0, 0.0)
  goal_left_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
  goal_right_cfg = get_robocup_goalpost_cfg()
  goal_right_cfg.init_state.pos = (-GOALPOST_X, 0.0, 0.0)
  goal_right_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

  entities = {
    "robot": robot_cfg,
    "soccer_field": get_dribbling_field_cfg(),
    "soccer_ball": ball_cfg,
    "goalpost_left": goal_left_cfg,
    "goalpost_right": goal_right_cfg,
  }
  for i in range(NUM_OBSTACLES):
    entities[f"dribble_obstacle_{i}"] = get_dribble_obstacle_cfg(i)

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="plane",
    collisions=(
      CollisionCfg(
        geom_names_expr=("terrain$",),
        friction=TERRAIN_FRICTION,
        disable_other_geoms=False,
      ),
    ),
  )
  cfg.scene.num_envs = 512 if not play else 1
  cfg.scene.entities = entities

  LEFT_FOOT_BALL = "dribble_left_foot_ball_contact"
  RIGHT_FOOT_BALL = "dribble_right_foot_ball_contact"
  LEFT_FOOT_GROUND = "dribble_left_foot_ground_contact"
  RIGHT_FOOT_GROUND = "dribble_right_foot_ground_contact"
  ROBOT_OBSTACLE_CONTACT = "dribble_robot_obstacle_contact"

  sensors = [
    ContactSensorCfg(
      name=LEFT_FOOT_GROUND,
      primary=ContactMatch(mode="body", pattern=r"^left_foot_link$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"), reduce="netforce", num_slots=1,
    ),
    ContactSensorCfg(
      name=RIGHT_FOOT_GROUND,
      primary=ContactMatch(mode="body", pattern=r"^right_foot_link$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"), reduce="netforce", num_slots=1,
    ),
    ContactSensorCfg(
      name=LEFT_FOOT_BALL,
      primary=ContactMatch(mode="body", pattern=r"^left_foot_link$", entity="robot"),
      secondary=ContactMatch(mode="geom", pattern="ball_collision", entity="soccer_ball"),
      fields=("found",), reduce="none", num_slots=1,
    ),
    ContactSensorCfg(
      name=RIGHT_FOOT_BALL,
      primary=ContactMatch(mode="body", pattern=r"^right_foot_link$", entity="robot"),
      secondary=ContactMatch(mode="geom", pattern="ball_collision", entity="soccer_ball"),
      fields=("found",), reduce="none", num_slots=1,
    ),
  ]
  # Optional aggregate robot-vs-obstacle contact sensor, useful for logs/debug. The hard
  # termination below is analytic too, so training does not depend on this sensor firing.
  for i in range(NUM_OBSTACLES):
    sensors.append(ContactSensorCfg(
      name=f"{ROBOT_OBSTACLE_CONTACT}_{i}",
      primary=ContactMatch(mode="body", pattern=r".*", entity="robot"),
      secondary=ContactMatch(mode="geom", pattern=f"dribble_obstacle_{i}_collision", entity=f"dribble_obstacle_{i}"),
      fields=("found",), reduce="none", num_slots=4,
    ))
  cfg.scene.sensors = (*cfg.scene.sensors, *sensors)

  cfg.curriculum = {}
  cfg.events = {"reset_scene_to_default": EventTermCfg(mode="reset", func=reset_scene_to_default)}

  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    lookat=(2.2, 0.0, 1.0),
    distance=7.0,
    elevation=-35.0,
    azimuth=0.0,
  )

  motor_obs_terms, motor_obs_term_dims = mdp.default_motor_obs_layout(act_dim=MOTOR_ACT_DIM)
  cfg.actions = {
    "motor_latent": mdp.MotorLatentActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=T1_23_ACTION_SCALE,
      use_default_offset=True,
      command_name="dribble",
      stage1_wandb_run_path=os.environ.get("MJLAB_STAGE1_WANDB_RUN_PATH_PENALTY"),
      motor_obs_terms=motor_obs_terms,
      motor_obs_term_dims=motor_obs_term_dims,
      strict_obs_layout=True,
    )
  }

  cfg.commands = {
    "dribble": mdp.DribblingCommandCfg(
      entity_name="robot",
      ball_entity_name="soccer_ball",
      obstacle_entity_prefix="dribble_obstacle_",
      command_dim=MOTOR_COMMAND_DIM,
      field_half_length_x=FIELD_HALF_LENGTH_X,
      field_half_width_y=FIELD_HALF_WIDTH_Y,
      ball_spawn_x=BALL_START_X,
      ball_spawn_y_range=(-3.2, 3.2),
      ball_spawn_z=BALL_Z,
      robot_distance_behind_ball=ROBOT_BEHIND_BALL,
      robot_spawn_y_jitter=0.04,
      robot_spawn_x_jitter=0.02,
      robot_yaw_jitter=0.04,
      target_x=TARGET_X,
      target_y_range=(-2.8, 2.8),
      success_ball_x=SUCCESS_BALL_X,
      success_robot_x=SUCCESS_ROBOT_X,
      dribble_area_bounds=DRIBBLE_AREA_BOUNDS,
      hard_area_margin=DRIBBLE_AREA_HARD_MARGIN,
      num_obstacles=NUM_OBSTACLES,
      active_obstacles_range=(1, 1),
      obstacle_spawn_x_range=(1.0, 4.10),
      obstacle_spawn_y_range=(-3.25, 3.25),
      obstacle_keepout_from_start=0.95,
      obstacle_keepout_from_target=0.75,
      obstacle_min_pair_dist=0.95,
      obstacle_radius=OBSTACLE_RADIUS,
      obstacle_ball_safe_dist=OBSTACLE_BALL_SAFE_DIST,
      obstacle_robot_hit_dist=OBSTACLE_ROBOT_HIT_DIST,
      debug_vis=True,
      resampling_time_range=(1.0e9, 1.0e9),
    )
  }

  common_terms = {
    "base_lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}),
    "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"biased": True}),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "decoded_actions": ObservationTermCfg(func=mdp.motor_last_decoded_action, params={"action_name": "motor_latent"}),
    "target_dir_xy": ObservationTermCfg(func=mdp.target_direction_xy, params={"command_name": "dribble"}),
    "target_pos_rel_xy": ObservationTermCfg(func=mdp.target_position_relative_xy, params={"command_name": "dribble"}),
    "ball_pos_rel_xyz": ObservationTermCfg(func=mdp.ball_position_relative_xyz, params={"command_name": "dribble"}),
    "ball_vel_w_xy": ObservationTermCfg(func=mdp.ball_velocity_w_xy, params={"ball_entity_name": "soccer_ball"}),
    "ball_dist_xy": ObservationTermCfg(func=mdp.ball_dist_xy_obs, params={"command_name": "dribble"}),
    "left_foot_pos_rel_ball_xy": ObservationTermCfg(func=mdp.left_foot_pos_rel_ball_xy_obs, params={"command_name": "dribble"}),
    "right_foot_pos_rel_ball_xy": ObservationTermCfg(func=mdp.right_foot_pos_rel_ball_xy_obs, params={"command_name": "dribble"}),
    "obstacle_rel_ball_xy": ObservationTermCfg(func=mdp.obstacle_positions_relative_to_ball_obs, params={"command_name": "dribble"}),
    "obstacle_active_mask": ObservationTermCfg(func=mdp.obstacle_active_mask_obs, params={"command_name": "dribble"}),
    "min_ball_obstacle_dist": ObservationTermCfg(func=mdp.min_ball_obstacle_dist_obs, params={"command_name": "dribble"}),
  }

  cfg.observations = {
    "actor": ObservationGroupCfg(terms=common_terms, concatenate_terms=True, enable_corruption=not play),
    "critic": ObservationGroupCfg(terms=common_terms, concatenate_terms=True, enable_corruption=False),
  }

  cfg.rewards = {
    # Main task: move the ball, not just the body, into the penalty-area line.
    "success": RewardTermCfg(func=mdp.success_event_reward, weight=35.0, params={"command_name": "dribble"}),
    "ball_to_target_progress": RewardTermCfg(func=mdp.ball_to_target_progress_reward, weight=12.0, params={"command_name": "dribble", "max_delta": 0.04, "upright_gate": 0.15}),
    "ball_forward_velocity": RewardTermCfg(func=mdp.ball_forward_velocity_reward, weight=0.4, params={"command_name": "dribble", "max_speed": 0.8}),
    "ball_velocity_tracking": RewardTermCfg(
      func=mdp.ball_velocity_tracking_reward,
      weight=5.0,
      params={
        "command_name": "dribble",
        "target_speed": 0.38,
        "speed_sigma": 0.22,
        "dir_sigma": 0.50,
        "min_robot_ball_dist": 0.15,
        "max_robot_ball_dist": 0.85,
      },
    ),
    "ball_speed_limit": RewardTermCfg(
      func=mdp.ball_speed_limit_penalty,
      weight=-6.0,
      params={
        "command_name": "dribble",
        "free_speed": 0.70,
        "hard_speed": 1.25,
      },
    ),
    "ball_path_lane": RewardTermCfg(
      func=mdp.ball_path_lane_reward,
      weight=1.0,
      params={
        "command_name": "dribble",
        "lane_sigma": 0.45,
        "forward_margin": 0.15,
      },
    ),
    "ball_obstacle_aware_velocity": RewardTermCfg(
      func=mdp.ball_obstacle_aware_velocity_reward,
      weight=5.0,
      params={
        "command_name": "dribble",
        "target_speed": 0.35,
        "speed_sigma": 0.22,
        "dir_sigma": 0.50,
        "influence_dist": 1.60,
        "lookahead_dist": 2.20,
        "lateral_influence": 0.95,
        "repel_gain": 1.35,
        "min_robot_ball_dist": 0.12,
        "max_robot_ball_dist": 0.95,
      },
    ),
    "robot_behind_ball": RewardTermCfg(
      func=mdp.robot_behind_ball_reward,
      weight=4.0,
      params={
        "command_name": "dribble",
        "desired_behind_dist": 0.45,
        "behind_sigma": 0.28,
        "lateral_sigma": 0.30,
        "max_ball_dist": 1.10,
      },
    ),
    "ball_accel_limit": RewardTermCfg(
      func=mdp.ball_accel_limit_penalty,
      weight=-3.0,
      params={
        "command_name": "dribble",
        "free_delta_speed": 0.28,
        "hard_delta_speed": 0.85,
      },
    ),

    # Palla al piede: dense close-control + hard pressure near 1 m.
    "keep_ball_close": RewardTermCfg(func=mdp.keep_ball_close_reward, weight=2.5, params={"command_name": "dribble", "sigma": 0.65}),
    "ball_too_far": RewardTermCfg(func=mdp.ball_too_far_penalty, weight=-5.5, params={"command_name": "dribble", "free_dist": 0.75, "max_dist": 1.25}),
    "feet_ball_control": RewardTermCfg(func=mdp.feet_ball_control_reward, weight=0.05, params={"command_name": "dribble", "near_dist": 0.34, "max_foot_speed": 2.0}),
    "foot_over_ball": RewardTermCfg(func=mdp.foot_over_ball_penalty, weight=-1.8, params={"command_name": "dribble", "xy_near": 0.16, "z_margin": 0.015}),

    # Ostacoli: the ball must stay 80 cm away. Robot collisions terminate.
    "obstacle_ball_clearance": RewardTermCfg(func=mdp.obstacle_ball_clearance_reward, weight=0.8, params={"command_name": "dribble", "safe_dist": OBSTACLE_BALL_SAFE_DIST, "sigma": 0.35}),
    "obstacle_ball_near": RewardTermCfg(func=mdp.obstacle_ball_near_penalty, weight=-6.0, params={"command_name": "dribble", "safe_dist": OBSTACLE_BALL_SAFE_DIST, "hard_dist": 0.25}),

    # Posture and structure: strong enough to avoid low crouch / broken walking.
    "upright": RewardTermCfg(func=mdp.upright_stability_reward, weight=0.75, params={"height_target": None, "height_sigma": 0.14, "roll_band": 0.07, "roll_sigma": 0.12, "pitch_target": 0.14, "pitch_band": 0.12, "pitch_sigma": 0.25}),
    "double_knee_crouch": RewardTermCfg(func=mdp.double_knee_crouch_penalty, weight=-4.5, params={"free_flex": 0.18, "max_flex": 0.85}),
    "fallen": RewardTermCfg(func=mdp.fallen_indicator, weight=-10.0, params={"min_height": 0.30, "max_tilt": 1.20}),
    "outside_dribble_area": RewardTermCfg(func=mdp.outside_dribble_area_penalty, weight=-2.0, params={"command_name": "dribble"}),

    # Complete fast, but not at the cost of posture/close-control because the above terms dominate.
    "time_pressure": RewardTermCfg(func=mdp.time_pressure_reward, weight=0.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),

    "robot_forward_velocity": RewardTermCfg(
        func=mdp.robot_forward_velocity_reward,
        weight=1.0,
        params={"command_name": "dribble", "max_speed": 0.60},
    ),

    "robot_follow_ball": RewardTermCfg(
        func=mdp.robot_follow_ball_reward,
        weight=2.0,
        params={"command_name": "dribble", "desired_dist": 0.48, "sigma": 0.25},
    ),

    "alive_moving": RewardTermCfg(
        func=mdp.alive_moving_reward,
        weight=0.25,
        params={"command_name": "dribble", "min_speed": 0.08, "max_speed": 0.60},
),
  }

  cfg.terminations.pop("ee_body_pos", None)
  cfg.terminations.update({
    "success": TerminationTermCfg(func=mdp.success_termination, params={"command_name": "dribble"}),
    "fallen": TerminationTermCfg(func=mdp.FallTermination, params={"min_height": 0.30, "max_tilt": 1.20, "consecutive_steps": 6}),
    "ball_lost": TerminationTermCfg(func=mdp.ball_lost_termination, params={"command_name": "dribble", "max_dist": 1.30}),
    "hard_outside_area": TerminationTermCfg(func=mdp.hard_outside_dribble_area_termination, params={"command_name": "dribble"}),
    "robot_obstacle_hit": TerminationTermCfg(func=mdp.robot_obstacle_hit_termination, params={"command_name": "dribble", "hit_dist": OBSTACLE_ROBOT_HIT_DIST}),
    "ball_out": TerminationTermCfg(func=mdp.ball_out_of_field_termination, params={"command_name": "dribble", "margin": 0.15}),
  })

  cfg.sim.mujoco.timestep = SIM_TIMESTEP_S
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 512
  cfg.decimation = CONTROL_DECIMATION
  cfg.episode_length_s = EPISODE_LENGTH_S

  if play:
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
  return cfg
  
