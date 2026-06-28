import os

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg  # solo se vuoi anche il sensore curb

from mjlab.asset_zoo.robots import T1_23_ACTION_SCALE, get_t1_23_robot_cfg
from mjlab.asset_zoo.robocup_assets.ball import get_robocup_ball_cfg
from mjlab.asset_zoo.robocup_assets.field import get_robocup_field_cfg
from mjlab.asset_zoo.robocup_assets.goalpost import get_robocup_goalpost_cfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from mjlab.tasks.penalty_expert.p1_set_shot import mdp




# ---------------- Goal geometry (same convention as GK) ----------------
GOAL_X_LINE = 7.0
GOALPOST_X  = 7.3



# ---------------- Ball geometry ----------------
BALL_R = 0.11                 # robocup ball radius
BALL_Z = BALL_R               # center height above ground

# ---------------- Penalty spot center ----------------
PENALTY_DIST_FROM_GOAL = 2.5
BALL_X = GOAL_X_LINE - PENALTY_DIST_FROM_GOAL
BALL_Y = 0.0

# ---------------- Robot spawn: behind the ball ----------------
ROBOT_BEHIND_BALL = 0.38     # meters (tune later if needed)
ROBOT_X = BALL_X - ROBOT_BEHIND_BALL
ROBOT_Y = 0.0

# ---------------- Keep-out / safety bounds (optional but recommended) ----------------
# Keep robot in a corridor around the penalty spot up to near the goal line.
STRIKER_AREA_BOUNDS = (ROBOT_X - 0.5, GOAL_X_LINE - 0.1, -1.0, 1.0)
STRIKER_AREA_HARD_MARGIN = 0.5

# ---------------- Motor controller layout dims ----------------
MOTOR_COMMAND_DIM = 46
MOTOR_ACT_DIM = 23

EPISODE_LENGTH_S = 5.0
SIM_TIMESTEP_S = 0.005
CONTROL_DECIMATION = 4
KICK_ONLY_RESET_PROB = 0.6

# ---------------- P1 test walls around the real 14x9 playable area ----------------
# - 2 continuous walls on long sides (y = +/- 4.5),
# - 2 segmented walls per short side (x = +/- 7.0) leaving goal opening at y ~= 0.
# Walls are centered on this boundary.

P1_FIELD_HALF_LENGTH_X = 7.0
P1_FIELD_HALF_WIDTH_Y = 4.5
P1_WALL_THICKNESS = 0.16
P1_WALL_HEIGHT = 0.07
P1_GOAL_OPENING_HALF_WIDTH = 1.55

P1_WALL_RGBA = (0.92, 0.18, 0.18, 0.45)
P1_WALL_FRICTION = (1.2, 0.02, 0.002)
P1_WALL_SOLREF = (0.02, 1.5)
P1_WALL_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)

# Overlay (area shading)
P1_AREA_OVERLAY_HALF_THICKNESS = 0.0015
P1_HARD_AREA_OVERLAY_Z = 0.0015
P1_STRIKER_AREA_OVERLAY_Z = 0.0035
P1_HARD_AREA_RGBA = (0.95, 0.55, 0.10, 0.22)
P1_STRIKER_AREA_RGBA = (0.05, 0.60, 0.95, 0.30)

# (optional) curb contact sensor name
P1_BALL_CURB_CONTACT_SENSOR_NAME = "p1_ball_curb_contact"
P1_FIELD_BODY_NAME = "field"

# Grip presets for penalty strike tuning.
# Select with:
#   MJLAB_PENALTY_GRIP_PRESET=a
#   MJLAB_PENALTY_GRIP_PRESET=b
#   MJLAB_PENALTY_GRIP_PRESET=c
#   MJLAB_PENALTY_GRIP_PRESET=random_abc
GRIP_PRESET = os.environ.get("MJLAB_PENALTY_GRIP_PRESET", "b").strip().lower()
GRIP_PRESETS: dict[str, dict[str, tuple[float, float, float]]] = {
  "a": {
    "ball": (0.85, 0.006, 0.0002),
    "foot": (1.00, 0.015, 0.0008),
    "terrain": (0.95, 0.008, 0.0008),
  },
  "b": {
    "ball": (0.90, 0.008, 0.0003),
    "foot": (1.10, 0.020, 0.0010),
    "terrain": (1.00, 0.010, 0.0010),
  },
  "c": {
    "ball": (0.95, 0.010, 0.0005),
    "foot": (1.25, 0.030, 0.0015),
    "terrain": (1.10, 0.015, 0.0012),
  },
}
if GRIP_PRESET == "random_abc":
  _static_grip_key = "b"
elif GRIP_PRESET in GRIP_PRESETS:
  _static_grip_key = GRIP_PRESET
else:
  raise ValueError(
    f"Invalid MJLAB_PENALTY_GRIP_PRESET='{GRIP_PRESET}'. "
    "Valid values: 'a', 'b', 'c', 'random_abc'."
  )
_grip_cfg = GRIP_PRESETS[_static_grip_key]
BALL_FRICTION = _grip_cfg["ball"]
FOOT_FRICTION = _grip_cfg["foot"]
TERRAIN_FRICTION = _grip_cfg["terrain"]


def _add_p1_test_walls(
  spec: mujoco.MjSpec,
  striker_area_bounds: tuple[float, float, float, float] | None = None,
  hard_margin: float = 0.5,
) -> None:
  field_body = next((body for body in spec.bodies if body.name == P1_FIELD_BODY_NAME), None)
  if field_body is None:
    field_body = spec.worldbody.add_body(name=P1_FIELD_BODY_NAME)

  half_t = P1_WALL_THICKNESS / 2.0
  half_h = P1_WALL_HEIGHT / 2.0
  wall_z = half_h

  def _add_wall(name: str, pos: tuple[float, float, float], size: tuple[float, float, float]) -> None:
    wall = field_body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=size,
    )
    wall.name = name
    wall.rgba = P1_WALL_RGBA
    wall.friction = P1_WALL_FRICTION
    wall.solref = P1_WALL_SOLREF
    wall.solimp = P1_WALL_SOLIMP

  # Long sides: continuous walls.
  long_side_wall_y = P1_FIELD_HALF_WIDTH_Y
  _add_wall(
    "p1_wall_long_pos_y",
    (0.0, long_side_wall_y, wall_z),
    (P1_FIELD_HALF_LENGTH_X, half_t, half_h),
  )
  _add_wall(
    "p1_wall_long_neg_y",
    (0.0, -long_side_wall_y, wall_z),
    (P1_FIELD_HALF_LENGTH_X, half_t, half_h),
  )

  # Short sides split in two per side, leaving opening for goal.
  short_side_segment_half_y = (P1_FIELD_HALF_WIDTH_Y - P1_GOAL_OPENING_HALF_WIDTH) / 2.0
  if short_side_segment_half_y <= 0.0:
    raise ValueError("P1_GOAL_OPENING_HALF_WIDTH is too large for field width.")
  short_side_segment_center_y = P1_GOAL_OPENING_HALF_WIDTH + short_side_segment_half_y
  short_side_wall_x = P1_FIELD_HALF_LENGTH_X

  _add_wall(
    "p1_wall_short_pos_x_upper",
    (short_side_wall_x, short_side_segment_center_y, wall_z),
    (half_t, short_side_segment_half_y, half_h),
  )
  _add_wall(
    "p1_wall_short_pos_x_lower",
    (short_side_wall_x, -short_side_segment_center_y, wall_z),
    (half_t, short_side_segment_half_y, half_h),
  )
  _add_wall(
    "p1_wall_short_neg_x_upper",
    (-short_side_wall_x, short_side_segment_center_y, wall_z),
    (half_t, short_side_segment_half_y, half_h),
  )
  _add_wall(
    "p1_wall_short_neg_x_lower",
    (-short_side_wall_x, -short_side_segment_center_y, wall_z),
    (half_t, short_side_segment_half_y, half_h),
  )

  # Optional: shaded overlays (striker area + hard margin)
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
      size=(half_x, half_y, P1_AREA_OVERLAY_HALF_THICKNESS),
    )
    overlay.name = name
    overlay.rgba = rgba
    overlay.contype = 0
    overlay.conaffinity = 0

  if striker_area_bounds is not None:
    x_min, x_max, y_min, y_max = striker_area_bounds
    hard_bounds = (
      x_min - hard_margin,
      x_max + hard_margin,
      y_min - hard_margin,
      y_max + hard_margin,
    )
    _add_area_overlay(
      "p1_striker_area_hard_overlay",
      hard_bounds,
      P1_HARD_AREA_OVERLAY_Z,
      P1_HARD_AREA_RGBA,
    )
    _add_area_overlay(
      "p1_striker_area_overlay",
      striker_area_bounds,
      P1_STRIKER_AREA_OVERLAY_Z,
      P1_STRIKER_AREA_RGBA,
    )


def get_p1_field_cfg_with_test_walls(
  striker_area_bounds: tuple[float, float, float, float] | None = None,
  hard_margin: float = 0.5,
) -> EntityCfg:
  field_cfg = get_robocup_field_cfg()
  base_spec_fn = field_cfg.spec_fn

  def _spec_fn() -> mujoco.MjSpec:
    spec = base_spec_fn()
    _add_p1_test_walls(spec, striker_area_bounds=striker_area_bounds, hard_margin=hard_margin)
    return spec

  field_cfg.spec_fn = _spec_fn
  return field_cfg


@requires_model_fields("geom_friction")
def randomize_penalty_grip_abc(
  env,
  env_ids: torch.Tensor | slice | None,
  robot_entity_name: str = "robot",
  ball_entity_name: str = "soccer_ball",
  terrain_entity_name: str = "terrain",
  foot_geom_expr: str = r"^(left|right)_foot_collision$",
  ball_geom_expr: str = r"^ball_collision$",
  terrain_geom_expr: str = r"terrain$",
) -> None:
  if env_ids is None:
    env_ids_t = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  elif isinstance(env_ids, slice):
    env_ids_t = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
  else:
    env_ids_t = env_ids.to(device=env.device, dtype=torch.long)
  if env_ids_t.numel() == 0:
    return

  robot = env.scene[robot_entity_name]
  ball = env.scene[ball_entity_name]
  terrain = env.scene[terrain_entity_name]

  foot_local_ids, _ = robot.find_geoms(foot_geom_expr, preserve_order=True)
  ball_local_ids, _ = ball.find_geoms(ball_geom_expr, preserve_order=True)
  terrain_local_ids, _ = terrain.find_geoms(terrain_geom_expr, preserve_order=True)
  if not foot_local_ids or not ball_local_ids or not terrain_local_ids:
    raise ValueError(
      "randomize_penalty_grip_abc could not resolve required geoms "
      f"(foot={foot_local_ids}, ball={ball_local_ids}, terrain={terrain_local_ids})."
    )

  foot_ids = robot.indexing.geom_ids[
    torch.as_tensor(foot_local_ids, device=env.device, dtype=torch.long)
  ]
  ball_ids = ball.indexing.geom_ids[
    torch.as_tensor(ball_local_ids, device=env.device, dtype=torch.long)
  ]
  terrain_ids = terrain.indexing.geom_ids[
    torch.as_tensor(terrain_local_ids, device=env.device, dtype=torch.long)
  ]

  preset_keys = ("a", "b", "c")
  ball_table = torch.tensor(
    [GRIP_PRESETS[k]["ball"] for k in preset_keys],
    dtype=torch.float32,
    device=env.device,
  )
  foot_table = torch.tensor(
    [GRIP_PRESETS[k]["foot"] for k in preset_keys],
    dtype=torch.float32,
    device=env.device,
  )
  terrain_table = torch.tensor(
    [GRIP_PRESETS[k]["terrain"] for k in preset_keys],
    dtype=torch.float32,
    device=env.device,
  )
  sampled = torch.randint(0, len(preset_keys), (env_ids_t.numel(),), device=env.device)

  geom_friction = env.sim.model.geom_friction
  if geom_friction.ndim == 2:
    k = int(sampled[0].item())
    geom_friction[foot_ids, :] = foot_table[k]
    geom_friction[ball_ids, :] = ball_table[k]
    geom_friction[terrain_ids, :] = terrain_table[k]
    return

  geom_friction[env_ids_t[:, None], foot_ids[None, :], :] = foot_table[sampled][:, None, :]
  geom_friction[env_ids_t[:, None], ball_ids[None, :], :] = ball_table[sampled][:, None, :]
  geom_friction[env_ids_t[:, None], terrain_ids[None, :], :] = terrain_table[sampled][:, None, :]



def booster_t1_23_penalty_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_env_cfg()

    to_remove = []
    for name, term in cfg.terminations.items():
        params = getattr(term, "params", {}) or {}
        if params.get("command_name") == "motion":
            to_remove.append(name)

    for name in to_remove:
        cfg.terminations.pop(name, None)
   



    PENALTY_DIST_FROM_GOAL = 2.5
    ball_x = GOAL_X_LINE - PENALTY_DIST_FROM_GOAL

    ROBOT_BEHIND_BALL = 0.38
    ROBOT_Y_BIAS = 0.04
    robot_x = ball_x - ROBOT_BEHIND_BALL

    BALL_JITTER_X = 0.01
    ROBOT_JITTER_X = 0.01
    ROBOT_JITTER_Y = 0.01
    ROBOT_YAW_JITTER = 0.03

    ball_spawn_x_range = (ball_x - BALL_JITTER_X, ball_x + BALL_JITTER_X)
    ball_spawn_y_range = (0.0, 0.0)

    striker_spawn_x_range = (robot_x - ROBOT_JITTER_X, robot_x + ROBOT_JITTER_X)
    striker_spawn_y_range = (ROBOT_Y_BIAS - ROBOT_JITTER_Y, ROBOT_Y_BIAS + ROBOT_JITTER_Y)

    spawn_yaw_range = (-ROBOT_YAW_JITTER, ROBOT_YAW_JITTER)

    striker_area_bounds = (robot_x - 0.8, GOAL_X_LINE + 0.6, -1.5, 1.5)
    hard_area_margin = 0.5


    # ------------------ robot ------------------
    robot_cfg = get_t1_23_robot_cfg()
    robot_cfg.init_state.pos = (robot_x, ROBOT_Y_BIAS, robot_cfg.init_state.pos[2])  # tieni z default
    # se il robot di default NON guarda verso +x, allora ruotalo:
    # robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    robot_cfg.collisions = (
        CollisionCfg(
            geom_names_expr=(r"^(left|right)_foot_collision$",),
            friction=FOOT_FRICTION,
            disable_other_geoms=False,
        ),
    )

    # ------------------ ball -------------------
    ball_cfg = get_robocup_ball_cfg()
    ball_cfg.init_state.pos = (ball_x, 0.0, BALL_R)
    ball_cfg.collisions = (
        CollisionCfg(
            geom_names_expr=(r"^ball_collision$",),
            friction=BALL_FRICTION,
            disable_other_geoms=False,
        ),
    )
    # ------------------ goals ------------------
    goal_left_cfg = get_robocup_goalpost_cfg()
    goal_left_cfg.init_state.pos = (GOALPOST_X, 0.0, 0.0)
    goal_left_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)

    goal_right_cfg = get_robocup_goalpost_cfg()
    goal_right_cfg.init_state.pos = (-GOALPOST_X, 0.0, 0.0)
    goal_right_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    # ------------------ scene ------------------
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
    cfg.scene.entities = {
        "robot": robot_cfg,
        "soccer_field": get_p1_field_cfg_with_test_walls(
            striker_area_bounds=striker_area_bounds,
            hard_margin=hard_area_margin,
        ),
        "soccer_ball": ball_cfg,
        "goalpost_left": goal_left_cfg,
        "goalpost_right": goal_right_cfg,
    }

    # ------------------ Contact sensors: foot <-> ball ------------------
    P1_LEFT_FOOT_BALL_CONTACT_SENSOR_NAME  = "p1_left_foot_ball_contact"
    P1_RIGHT_FOOT_BALL_CONTACT_SENSOR_NAME = "p1_right_foot_ball_contact"
    P1_LEFT_FOOT_GROUND_CONTACT_SENSOR_NAME  = "p1_left_foot_ground_contact"
    P1_RIGHT_FOOT_GROUND_CONTACT_SENSOR_NAME = "p1_right_foot_ground_contact"


    left_foot_ground_contact_cfg = ContactSensorCfg(
        name=P1_LEFT_FOOT_GROUND_CONTACT_SENSOR_NAME,
        primary=ContactMatch(mode="body", pattern=r"^left_foot_link$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    right_foot_ground_contact_cfg = ContactSensorCfg(
        name=P1_RIGHT_FOOT_GROUND_CONTACT_SENSOR_NAME,
        primary=ContactMatch(mode="body", pattern=r"^right_foot_link$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    left_foot_ball_contact_cfg = ContactSensorCfg(
        name=P1_LEFT_FOOT_BALL_CONTACT_SENSOR_NAME,
        primary=ContactMatch(mode="body", pattern=r"^left_foot_link$", entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="ball_collision", entity="soccer_ball"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    right_foot_ball_contact_cfg = ContactSensorCfg(
        name=P1_RIGHT_FOOT_BALL_CONTACT_SENSOR_NAME,
        primary=ContactMatch(mode="body", pattern=r"^right_foot_link$", entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="ball_collision", entity="soccer_ball"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg.scene.sensors = (
        *cfg.scene.sensors,
        left_foot_ball_contact_cfg,
        right_foot_ball_contact_cfg,
        left_foot_ground_contact_cfg,
        right_foot_ground_contact_cfg,
    )

    # tieni il reset (userà init_state che abbiamo appena settato)
    cfg.curriculum = {}
    cfg.events = {
        "reset_scene_to_default": EventTermCfg(mode="reset", func=reset_scene_to_default),
    }
    if GRIP_PRESET == "random_abc":
        random_grip_params = {
            "robot_entity_name": "robot",
            "ball_entity_name": "soccer_ball",
            "terrain_entity_name": "terrain",
            "foot_geom_expr": r"^(left|right)_foot_collision$",
            "ball_geom_expr": r"^ball_collision$",
            "terrain_geom_expr": r"terrain$",
        }
        cfg.events["randomize_penalty_grip_startup"] = EventTermCfg(
            mode="startup",
            func=randomize_penalty_grip_abc,
            params=random_grip_params,
        )
        cfg.events["randomize_penalty_grip_reset"] = EventTermCfg(
            mode="reset",
            func=randomize_penalty_grip_abc,
            params=random_grip_params,
        )


    cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=(robot_x + 1.0, 0.0, 1.0),
        distance=5.0,
        elevation=-25.0,
        azimuth=0.0,
    )


    motor_obs_terms, motor_obs_term_dims = mdp.default_motor_obs_layout(
        act_dim=MOTOR_ACT_DIM,
    )

    cfg.actions = {
    "motor_latent": mdp.MotorLatentActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=T1_23_ACTION_SCALE,
      use_default_offset=True,
      command_name="set_shot",
      stage1_wandb_run_path=os.environ.get("MJLAB_STAGE1_WANDB_RUN_PATH_PENALTY"),
      motor_obs_terms=motor_obs_terms,
      motor_obs_term_dims=motor_obs_term_dims,
      strict_obs_layout=True,
    )
  }
    # IMPORTANT:
    # visual_left/right_corner_y are GOAL-LINE Y targets (x = GOAL_X_LINE).
    # The command can use fixed or random_binary corner selection with explicit semantics.
    VISUAL_LEFT_CORNER_Y = 1.0
    VISUAL_RIGHT_CORNER_Y = 0.0
    TARGET_MODE = "random_binary"        # allowed: "fixed", "random_binary"
    FIXED_TARGET_CORNER = "left" # used only when TARGET_MODE == "fixed"
    # aim_z is the target Z on the goal line.
    AIM_Z = 1.45

    cfg.commands = {
        "set_shot": mdp.SetShotCommandCfg(
            entity_name="robot",
            ball_entity_name="soccer_ball",
            command_dim=MOTOR_COMMAND_DIM,
            
            striker_spawn_mode="shot_line",
            setup_side_sign=-1.0,
            striker_distance_behind_ball=ROBOT_BEHIND_BALL,
            striker_lateral_offset=ROBOT_Y_BIAS,   
            striker_longitudinal_jitter=ROBOT_JITTER_X,
            striker_lateral_jitter=ROBOT_JITTER_Y,

            striker_spawn_x_range=striker_spawn_x_range,
            striker_spawn_y_range=striker_spawn_y_range,
            spawn_yaw_range=spawn_yaw_range,

            goal_line_x=GOAL_X_LINE,
            goal_y_half=1.55,
            goal_z_min=0.0,
            goal_z_max=1.85,

            ball_spawn_x_range=ball_spawn_x_range,
            ball_spawn_y_range=ball_spawn_y_range,
            ball_spawn_z=BALL_Z,

            striker_area_bounds=striker_area_bounds,
            hard_area_margin=hard_area_margin,

            aim_x=GOAL_X_LINE,
            # Fallback only; active lateral target comes from sampled visual corner config below.
            aim_y=0.0,
            aim_z=AIM_Z,
            visual_left_corner_y=VISUAL_LEFT_CORNER_Y,
            visual_right_corner_y=VISUAL_RIGHT_CORNER_Y,
            lateral_target_mode=TARGET_MODE,
            fixed_target_corner=FIXED_TARGET_CORNER,

            resampling_time_range=(1.0e9, 1.0e9),
            kick_only_reset_prob=KICK_ONLY_RESET_PROB,
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

    # last decoded motor actions (stability)
    "decoded_actions": ObservationTermCfg(
        func=mdp.motor_last_decoded_action,
        params={"action_name": "motor_latent"},
    ),

    # --- penalty extras ---
    "target_dir_xy": ObservationTermCfg(
        func=mdp.target_direction_xy,
        params={"command_name": "set_shot"},
    ),
    "ball_pos_rel_xyz": ObservationTermCfg(
        func=mdp.ball_position_relative_xyz,
        params={"command_name": "set_shot"},
    ),
    "ball_vel_w_xy": ObservationTermCfg(
        func=mdp.ball_velocity_w_xy,
        params={"ball_entity_name": "soccer_ball"},
    ),
    "kick_phase_flag": ObservationTermCfg(
        func=mdp.kick_phase_flag_obs,
        params={"command_name": "set_shot"},
    ),
    "kick_only_reset_flag": ObservationTermCfg(
        func=mdp.kick_only_reset_flag_obs,
        params={"command_name": "set_shot"},
    ),
    "yaw_error_abs": ObservationTermCfg(
        func=mdp.yaw_error_abs_obs,
        params={"command_name": "set_shot"},
    ),
    "ball_dist_xy": ObservationTermCfg(
        func=mdp.ball_dist_xy_obs,
        params={"command_name": "set_shot"},
    ),
    "right_foot_pos_rel_ball_xy": ObservationTermCfg(
        func=mdp.right_foot_pos_rel_ball_xy_obs,
        params={"command_name": "set_shot"},
    ),
    "left_foot_pos_rel_ball_xy": ObservationTermCfg(
        func=mdp.left_foot_pos_rel_ball_xy_obs,
        params={"command_name": "set_shot"},
    ),
    "left_support_latched_error_xy": ObservationTermCfg(
        func=mdp.left_support_latched_error_xy_obs,
        params={"command_name": "set_shot"},
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
        params={"command_name": "set_shot"},
    ),
    "ball_pos_rel_xyz": ObservationTermCfg(
        func=mdp.ball_position_relative_xyz,
        params={"command_name": "set_shot"},
    ),
    "ball_vel_w_xy": ObservationTermCfg(
        func=mdp.ball_velocity_w_xy,
        params={"ball_entity_name": "soccer_ball"},
    ),
    "kick_phase_flag": ObservationTermCfg(
        func=mdp.kick_phase_flag_obs,
        params={"command_name": "set_shot"},
    ),
    "kick_only_reset_flag": ObservationTermCfg(
        func=mdp.kick_only_reset_flag_obs,
        params={"command_name": "set_shot"},
    ),
    "yaw_error_abs": ObservationTermCfg(
        func=mdp.yaw_error_abs_obs,
        params={"command_name": "set_shot"},
    ),
    "ball_dist_xy": ObservationTermCfg(
        func=mdp.ball_dist_xy_obs,
        params={"command_name": "set_shot"},
    ),
    "right_foot_pos_rel_ball_xy": ObservationTermCfg(
        func=mdp.right_foot_pos_rel_ball_xy_obs,
        params={"command_name": "set_shot"},
    ),
    "left_foot_pos_rel_ball_xy": ObservationTermCfg(
        func=mdp.left_foot_pos_rel_ball_xy_obs,
        params={"command_name": "set_shot"},
    ),
    "left_support_latched_error_xy": ObservationTermCfg(
        func=mdp.left_support_latched_error_xy_obs,
        params={"command_name": "set_shot"},
    ),
    }

    cfg.observations = {
    "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
    ),
    }

    cfg.rewards = {
        "strike_event": RewardTermCfg(
            func=mdp.strike_event_reward,
            weight=12.0,
            params={
                "command_name": "set_shot",
                "left_sensor_name": P1_LEFT_FOOT_BALL_CONTACT_SENSOR_NAME,
                "right_sensor_name": P1_RIGHT_FOOT_BALL_CONTACT_SENSOR_NAME,
                "ball_depart_xy": 0.04,
                "min_vx": 0.08,
                "min_speed_xy": 0.18,
                "require_right_touch": True,
            },
        ),

        "impact_foot_speed": RewardTermCfg(
            func=mdp.right_foot_impact_speed_target_reward,
            weight=4.0,
            params={
                "command_name": "set_shot",
                "right_sensor_name": P1_RIGHT_FOOT_BALL_CONTACT_SENSOR_NAME,
                "target_speed": 4.7,
                "sigma": 0.6,
            },
        ),
        "ball_speed_to_left_aim_3d": RewardTermCfg(
            func=mdp.ball_speed_to_aim_reward_3d_after_strike,
            weight=4.0,
            params={"command_name": "set_shot"},
        ),

        "goal_scored": RewardTermCfg(
            func=mdp.goal_scored_shaped_target_reward,
            weight=12.0,
            params={
                "command_name": "set_shot",
                "sigma_y": 0.20,
                "sigma_z": 0.18,
                "base_goal": 0.10,
                "weight_y": 0.20,
                "weight_z": 0.15,
                "weight_yz": 0.65,
            },
        ),

        # 10) kick-phase left support penalties

        "post_strike_left_support_move": RewardTermCfg(
            func=mdp.post_strike_left_support_move_penalty,
            weight=-3.0,
            params={
                "command_name": "set_shot",
                "deadzone": 0.02,
                "max_dist": 0.15,
                "lock_steps": 12,
            },
        ),
        "post_strike_left_support_speed": RewardTermCfg(
            func=mdp.post_strike_left_support_speed_penalty,
            weight=-1.5,
            params={
                "command_name": "set_shot",
                "max_speed": 0.45,
                "lock_steps": 12,
            },
        ),
        "post_strike_left_support_lost_ground": RewardTermCfg(
            func=mdp.post_strike_left_support_lost_ground_penalty,
            weight=-2.0,
            params={
                "command_name": "set_shot",
                "left_ground_sensor_name": P1_LEFT_FOOT_GROUND_CONTACT_SENSOR_NAME,
                "lock_steps": 12,
            },
        ),
        # 11) final general terms
        "upright": RewardTermCfg(
            func=mdp.upright_stability_reward,
            weight=0.55,
            params={
                "height_target": None,
                "height_sigma": 0.14,
                "roll_band": 0.07,
                "roll_sigma": 0.12,
                "pitch_target": 0.14,
                "pitch_band": 0.12,
                "pitch_sigma": 0.25,
            },
        ),

        "low_height_soft_penalty": RewardTermCfg(
            func=mdp.striker_low_height_soft_penalty,
            weight=-2.2,
            params={
                "h_soft": 0.60,
            },
        ),

        "double_knee_crouch": RewardTermCfg(
            func=mdp.double_knee_crouch_penalty,
            weight=-10.0,
            params={
                "command_name": "set_shot",
                "near_ball_dist": 0.70,
                "free_left_flex": 0.10,
                "free_right_flex": 0.16,
                "max_left_flex": 0.65,
                "max_right_flex": 0.80,
                "left_weight": 1.35,
                "right_weight": 1.0,
            },
        ),

        "fallen": RewardTermCfg(
            func=mdp.fallen_indicator,
            weight=-4.0,
            params={
                "min_height": 0.30,
                "max_tilt": 1.20,
            },
        ),

        "left_pre_touch": RewardTermCfg(
            func=mdp.left_foot_prestrike_touch_penalty,
            weight=-1.0,
            params={
                "command_name": "set_shot",
                "left_sensor_name": P1_LEFT_FOOT_BALL_CONTACT_SENSOR_NAME,
            },
        ),

        "foot_over_ball": RewardTermCfg(
            func=mdp.foot_over_ball_penalty,
            weight=-2.0,
            params={
                "command_name": "set_shot",
                "xy_near": 0.18,
                "z_margin": 0.0,
            },
        ),

         "foot_contact_switch": RewardTermCfg(
            func=mdp.foot_contact_switch_bonus_p1,
            weight=0.2,
            params={
                "command_name": "set_shot",
                "left_contact_sensor_name": P1_LEFT_FOOT_GROUND_CONTACT_SENSOR_NAME,
                "right_contact_sensor_name": P1_RIGHT_FOOT_GROUND_CONTACT_SENSOR_NAME,
                "upright_gate": 0.75,
                "fz_thresh": 5.0,
                "support_sign": "neg",
            },
        ),

        "support_plant_at_strike": RewardTermCfg(
            func=mdp.support_plant_at_strike_bonus,
            weight=14.0,
            params={
                "command_name": "set_shot",
                "left_ground_sensor_name": P1_LEFT_FOOT_GROUND_CONTACT_SENSOR_NAME,
                "left_ball_sensor_name": P1_LEFT_FOOT_BALL_CONTACT_SENSOR_NAME,
                "right_sensor_name": P1_RIGHT_FOOT_BALL_CONTACT_SENSOR_NAME,
                "target_dx": -0.02,
                "dx_sigma": 0.10,
                "target_abs_dy": 0.11,
                "dy_sigma": 0.08,
                "max_left_speed": 0.28,
                "min_right_speed": 2.2,
                "max_right_speed": 5.5,
            },
        ),
        "right_knee_straight_at_strike": RewardTermCfg(
            func=mdp.right_knee_straight_at_strike_reward,
            weight=6.0,
            params={
                "command_name": "set_shot",
                "sigma_rad": 0.20,
            },
        ),
        "bad_posture_at_strike": RewardTermCfg(
            func=mdp.bad_posture_at_strike_penalty,
            weight=-30.0,
            params={
                "command_name": "set_shot",
                "left_sensor_name": P1_LEFT_FOOT_BALL_CONTACT_SENSOR_NAME,
                "right_sensor_name": P1_RIGHT_FOOT_BALL_CONTACT_SENSOR_NAME,
                "min_height": 0.56,
                "max_tilt": 0.55,
            },
        ),

        "underbar_launch": RewardTermCfg(
            func=mdp.ball_launch_angle_underbar_reward,
            weight=10.0,
            params={
                "command_name": "set_shot",
                "target_angle_deg": 24.0,
                "angle_sigma_deg": 8.0,
                "min_vx": 0.10,
                "max_speed_3d": 9.0,
            },
        ),
        "ball_power_lift": RewardTermCfg(
            func=mdp.ball_power_lift_reward_after_strike,
            weight=6.5,
            params={
                "command_name": "set_shot",
                "max_speed_3d": 9.0,
                "min_vx": 0.20,
                "min_vz": 0.12,
            },
        ),
        "ball_ground_touch_before_goal": RewardTermCfg(
            func=mdp.ball_ground_touch_before_goal_penalty,
            weight=-6.0,
            params={
                "command_name": "set_shot",
                "ground_z": 0.115,
                "min_x_progress": 0.50,
            },
        ),
        "ball_bounce_before_goal": RewardTermCfg(
            func=mdp.ball_bounce_before_goal_penalty,
            weight=-4.0,
            params={
                "command_name": "set_shot",
                "ground_z": 0.12,
                "min_x_after_strike": 0.45,
                "require_forward_vx": 0.35,
            },
        ),

########### DIREZIONE TIRO #############

        "underbar_goal": RewardTermCfg(
            func=mdp.underbar_goal_reward,
            weight=8.0,
            params={
                "command_name": "set_shot",
                "sigma_z": 0.18,
            },
        ),



        "goal_target_from_command": RewardTermCfg(
            func=mdp.goal_target_from_command_reward,
            weight=14.0,
            params={
                "command_name": "set_shot",
                "sigma_y": 0.35,
                "sigma_z": 0.18,
            },
        ),

                "lateral_goal": RewardTermCfg(
            func=mdp.lateral_goal_reward,
            weight=12.0,
            params={
                "command_name": "set_shot",
                "sigma_y": 0.15,
            },
        ),
    }

    cfg.terminations.pop("ee_body_pos", None)

    cfg.terminations.update({
        "fallen": TerminationTermCfg(
            func=mdp.FallTermination,
            params={"min_height": 0.30, "max_tilt": 1.20, "consecutive_steps": 6},
        ),

        "hard_outside_area": TerminationTermCfg(
            func=mdp.hard_outside_striker_area_termination,
            params={"command_name": "set_shot"},
        ),

        "ball_out": TerminationTermCfg(
            func=mdp.ball_out_of_play_termination,
            params={
                "command_name": "set_shot",
                "field_half_length_x": 7.0,
                "field_half_width_y": 4.5,
                "goal_opening_half_width": 1.55,
                "margin": 0.10,
            },
        ),

    }
    )


    cfg.sim.mujoco.timestep = SIM_TIMESTEP_S
    cfg.decimation = CONTROL_DECIMATION
    cfg.episode_length_s = EPISODE_LENGTH_S  # es. 6.0

    # niente random: resampling_time_range infinito già nella command
    # e range (v,v) nei parametri command.

    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = EPISODE_LENGTH_S
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)  # se esiste nel base
    return cfg
