from mjlab.asset_zoo.robots import (
    T1_23_ACTION_SCALE,
    get_t1_23_robot_cfg,
)
from mjlab.asset_zoo.robocup_field import get_robocup_field_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.viewer import ViewerConfig


def booster_t1_23_flat_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    # Robot: T1 23 DoF
    cfg.scene.entities = {"robot": get_t1_23_robot_cfg()}

    # Self-collision sul sottoalbero del tronco
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    # Azioni: scala derivata dagli attuatori T1
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = T1_23_ACTION_SCALE

    # Motion command: anchor + lista di body coerente col T1
    assert cfg.commands is not None
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "Trunk"
    motion_cmd.body_names = (
        # tronco
        "Trunk",
        # head
        "H1",
        "H2",
        # bacino / gambe
        "Waist",
        "Hip_Roll_Left",
        "Shank_Left",
        "left_foot_link",
        "Hip_Roll_Right",
        "Shank_Right",
        "right_foot_link",
        # braccio sinistro
        "AL2",            # spalla roll
        "AL3",            # gomito pitch
        "left_hand_link", # mano sinistra
        # braccio destro
        "AR2",
        "AR3",
        "right_hand_link",
    )

    # Eventi di contatto piedi
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

    # End-effectors: piedi + mani
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_foot_link",
        "right_foot_link",
        "left_hand_link",
        "right_hand_link",
    )

    # Viewer: segui il tronco
    cfg.viewer.body_name = "Trunk"

    # ------------------------------------------------------------------
    # Booster T1_23: rimuovi osservazioni che richiedono IMU non presenti
    # ------------------------------------------------------------------
    # Il tracking env di base aggiunge "base_lin_vel" e "base_ang_vel"
    # che usano i sensori "robot/imu_lin_vel" e "robot/imu_ang_vel".
    # Nel nostro T1_23 ci sono solo "robot/orientation" e "robot/angular-velocity",
    # quindi togliamo questi termini sia per policy che per critic.
    for group_name in ("policy", "critic"):
        obs_group = cfg.observations[group_name]
        new_terms = {
            name: term
            for name, term in obs_group.terms.items()
            if name not in ("base_lin_vel", "base_ang_vel")
        }
        cfg.observations[group_name] = ObservationGroupCfg(
            terms=new_terms,
            concatenate_terms=obs_group.concatenate_terms,
            enable_corruption=obs_group.enable_corruption,
        )

    # No state estimation: togli solo motion_anchor_pos_b dalla policy
    if not has_state_estimation:
        obs_group = cfg.observations["policy"]
        new_policy_terms = {
            k: v
            for k, v in obs_group.terms.items()
            if k not in ["motion_anchor_pos_b"]
        }
        cfg.observations["policy"] = ObservationGroupCfg(
            terms=new_policy_terms,
            concatenate_terms=obs_group.concatenate_terms,
            enable_corruption=obs_group.enable_corruption,
        )

    # Modalità play: niente randomizzazioni, episodi infiniti
    if play:
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg


def booster_t1_23_portiere_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 RoboCup portiere environment configuration."""
    cfg = make_tracking_env_cfg()

    cfg.scene.terrain = None
    cfg.scene.entities = {
        "robot": get_t1_23_robot_cfg(),
        "soccer_field": get_robocup_field_cfg(),
    }

    cfg.commands = {}
    cfg.observations = {}
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.curriculum = {}
    cfg.events = {
        "reset_scene_to_default": EventTermCfg(
            mode="reset",
            func=reset_scene_to_default,
        ),
    }

    cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=(0.0, 0.0, 0.0),
        distance=12.0,
        elevation=-60.0,
        azimuth=90.0,
    )

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = T1_23_ACTION_SCALE

    if play:
        cfg.episode_length_s = int(1e9)

    return cfg
