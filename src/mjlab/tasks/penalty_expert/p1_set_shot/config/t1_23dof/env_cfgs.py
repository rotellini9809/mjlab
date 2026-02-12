import os

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
from mjlab.viewer import ViewerConfig

from mjlab.tasks.penalty_expert.p1_set_shot import mdp


# ---------------- Goal geometry (same convention as GK) ----------------
GOAL_X_LINE = 7.0
GOALPOST_X  = 7.3

# ---------------- Ball geometry ----------------
BALL_R = 0.11                 # robocup ball radius
BALL_Z = BALL_R               # center height above ground

# ---------------- Penalty spot (2 m from goal line) ----------------
PENALTY_DIST_FROM_GOAL = 2.5
BALL_X = GOAL_X_LINE - PENALTY_DIST_FROM_GOAL
BALL_Y = 0.0

BALL_SPAWN_X_RANGE = (BALL_X, BALL_X)
BALL_SPAWN_Y_RANGE = (BALL_Y, BALL_Y)

# ---------------- Robot spawn: behind the ball ----------------
ROBOT_BEHIND_BALL = 1      # meters (tune later if needed)
ROBOT_X = BALL_X - ROBOT_BEHIND_BALL
ROBOT_Y = 0.0

STRIKER_SPAWN_X_RANGE = (ROBOT_X, ROBOT_X)
STRIKER_SPAWN_Y_RANGE = (ROBOT_Y, ROBOT_Y)

# Face the goal: +x direction -> yaw = 0 rad
SPAWN_YAW_RANGE = (0.0, 0.0)

# ---------------- Keep-out / safety bounds (optional but recommended) ----------------
# Keep robot in a corridor around the penalty spot up to near the goal line.
STRIKER_AREA_BOUNDS = (ROBOT_X - 0.5, GOAL_X_LINE - 0.1, -1.0, 1.0)
STRIKER_AREA_HARD_MARGIN = 0.5

# ---------------- Motor controller layout dims ----------------
MOTOR_COMMAND_DIM = 46
MOTOR_ACT_DIM = 23

EPISODE_LENGTH_S = 8.0




def booster_t1_23_penalty_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_env_cfg()

    to_remove = []
    for name, term in cfg.terminations.items():
        params = getattr(term, "params", {}) or {}
        if params.get("command_name") == "motion":
            to_remove.append(name)

    for name in to_remove:
        cfg.terminations.pop(name, None)
   



    # distanza "dischetto" dalla porta (metti quello che ti torna bene)
    PENALTY_DIST_FROM_GOAL = 2.5  # metri circa nel tuo mondo
    ball_x = GOAL_X_LINE- PENALTY_DIST_FROM_GOAL

    # robot dietro la palla
    ROBOT_BEHIND_BALL = 1.2
    robot_x = ball_x - ROBOT_BEHIND_BALL

    # ------------------ robot ------------------
    robot_cfg = get_t1_23_robot_cfg()
    robot_cfg.init_state.pos = (robot_x, 0.0, robot_cfg.init_state.pos[2])  # tieni z default
    # se il robot di default NON guarda verso +x, allora ruotalo:
    # robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    # ------------------ ball -------------------
    ball_cfg = get_robocup_ball_cfg()
    ball_cfg.init_state.pos = (ball_x, 0.0, BALL_R)

    # ------------------ goals ------------------
    goal_left_cfg = get_robocup_goalpost_cfg()
    goal_left_cfg.init_state.pos = (GOAL_X_LINE, 0.0, 0.0)
    goal_left_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)

    goal_right_cfg = get_robocup_goalpost_cfg()
    goal_right_cfg.init_state.pos = (-GOAL_X_LINE, 0.0, 0.0)
    goal_right_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    # ------------------ scene ------------------
    cfg.scene.terrain = None
    cfg.scene.num_envs = 512 if not play else 1
    cfg.scene.entities = {
        "robot": robot_cfg,
        "soccer_field": get_robocup_field_cfg(),
        "soccer_ball": ball_cfg,
        "goalpost_left": goal_left_cfg,
        "goalpost_right": goal_right_cfg,
    }

    # tieni il reset (userà init_state che abbiamo appena settato)
    cfg.curriculum = {}
    cfg.events = {
        "reset_scene_to_default": EventTermCfg(mode="reset", func=reset_scene_to_default),
    }


    cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=(0.0, 0.0, 0.0),
        distance=12.0,
        elevation=-60.0,
        azimuth=90.0,
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
      stage1_wandb_run_path=os.environ.get("MJLAB_STAGE1_WANDB_RUN_PATH"),
      motor_obs_terms=motor_obs_terms,
      motor_obs_term_dims=motor_obs_term_dims,
      strict_obs_layout=True,
    )
  }
    AIM_Y = 0.75      # angolo destro (metti -0.75 per sinistro)
    AIM_Z = 0.90      # mira alto (ball center)
    
    cfg.commands = {
    "set_shot": mdp.SetShotCommandCfg(
        entity_name="robot",
        ball_entity_name="soccer_ball",     # la palla fisica che calci
        command_dim=MOTOR_COMMAND_DIM,

        striker_spawn_x_range=STRIKER_SPAWN_X_RANGE,
        striker_spawn_y_range=STRIKER_SPAWN_Y_RANGE,
        spawn_yaw_range=SPAWN_YAW_RANGE,

        goal_line_x=GOAL_X_LINE,
        goal_y_half=1.0,

        ball_spawn_x_range=BALL_SPAWN_X_RANGE,
        ball_spawn_y_range=BALL_SPAWN_Y_RANGE,
        ball_spawn_z=BALL_Z,

        striker_area_bounds=STRIKER_AREA_BOUNDS,
        hard_area_margin=STRIKER_AREA_HARD_MARGIN,

        # Se vuoi anche un "aim point" fisso (centro porta), puoi passarlo così:
        aim_x=GOALPOST_X,
        aim_y=AIM_Y,
        aim_z=AIM_Z,

        # no resampling within episode
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
    }

    cfg.observations = {
    "policy": ObservationGroupCfg(
        terms=policy_terms,
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
    # yaw-only verso l'aim (che ora alterna i 2 angoli)
    "yaw_align": RewardTermCfg(
        func=mdp.yaw_alignment_reward,
        weight=3.0,
        params={"command_name": "set_shot", "k": 2.5},
    ),

    # stabilità: più severa + tilt penalty esplicita (anti “sbilanciato in avanti”)
    "upright": RewardTermCfg(
        func=mdp.upright_stability_reward,
        weight=1.6,
        params={"height_target": 0.62, "height_sigma": 0.12, "tilt_sigma": 0.35},
    ),
    "tilt_penalty": RewardTermCfg(
        func=mdp.trunk_tilt_l2_penalty,
        weight=-1.0,
    ),

    # setup / approach
    "approach_ball": RewardTermCfg(
        func=mdp.approach_ball_reward,
        weight=2.0,
        params={"command_name": "set_shot"},
    ),
    "behind_ball": RewardTermCfg(
        func=mdp.behind_ball_reward,
        weight=1.0,
        params={"command_name": "set_shot"},
    ),

    # strike proxy
    "strike_event": RewardTermCfg(
        func=mdp.strike_event_reward,
        weight=4.0,
        params={"command_name": "set_shot"},
    ),

    # >>> al posto di ball_to_goal_speed: velocità verso AIM in 3D (spinge in alto se aim_z è alto)
    "ball_to_aim_speed_3d": RewardTermCfg(
        func=mdp.ball_speed_to_aim_reward_3d,
        weight=3.0,
        params={"command_name": "set_shot"},
    ),

    # shaping: mentre vola verso porta, premia ALTO+LATO
    "ball_flight_high_side": RewardTermCfg(
        func=mdp.ball_flight_high_and_side_reward,
        weight=4.0,
        params={"command_name": "set_shot", "z_min": 0.55, "y_side_min": 0.55},
    ),

    # goal “buono” (alto+angolato): premio grande
    "goal_high_corner": RewardTermCfg(
        func=mdp.goal_high_corner_reward,
        weight=30.0,
        params={"command_name": "set_shot", "z_min": 0.55, "y_side_min": 0.55},
    ),

    # goal “cattivo” (raso o centrale): penalità
    "goal_bad_penalty": RewardTermCfg(
        func=mdp.goal_low_or_center_penalty,
        weight=-10.0,
        params={"command_name": "set_shot", "z_min": 0.55, "y_side_min": 0.55},
    ),

    # goal generico: lascialo ma molto basso (non deve competere col corner high)
    "goal_scored": RewardTermCfg(
        func=mdp.goal_scored_reward,
        weight=2.0,
        params={"command_name": "set_shot"},
    ),

    # penalties
    "outside_area": RewardTermCfg(
        func=mdp.outside_striker_area_penalty,
        weight=-0.5,
        params={"command_name": "set_shot"},
    ),
    "fallen": RewardTermCfg(
        func=mdp.fallen_indicator,
        weight=-8.0,
        params={"min_height": 0.30, "max_tilt": 1.20},
    ),
    "xy_speed": RewardTermCfg(
        func=mdp.xy_speed_l2,
        weight=-0.03,
    ),
    }


 
    cfg.terminations.pop("ee_body_pos", None)

    # lascia il time_out che arriva dal make_tracking_env_cfg (di solito c’è già)
    # aggiungi/override le tue
    cfg.terminations.update({
    "fallen": TerminationTermCfg(
        func=mdp.FallTermination,
        params={"min_height": 0.30, "max_tilt": 1.20, "consecutive_steps": 6},
    ),
    "success_goal": TerminationTermCfg(
        func=mdp.goal_scored_termination,
        params={"command_name": "set_shot"},
    ),
    "hard_outside_area": TerminationTermCfg(
        func=mdp.hard_outside_striker_area_termination,
        params={"command_name": "set_shot"},
    ),
    })


    cfg.episode_length_s = EPISODE_LENGTH_S  # es. 8.0

    # niente random: resampling_time_range infinito già nella command
    # e range (v,v) nei parametri command.

    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)  # se esiste nel base

    return cfg
