"""T1 Booster humanoid 23-DOF for MJLAB."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
    ElectricActuator,
    reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


##
# MJCF and assets
##
# NOTE: aggiorna questo path in base a dove metti T1_23dof.xml nel repo.
T1_23DOF_XML: Path = (
    MJLAB_SRC_PATH
    / "asset_zoo"
    / "robots"
    / "booster_t1_23dof"
    / "xmls"
    / "T1_23dof.xml"
)
assert T1_23DOF_XML.exists(), f"Cannot find T1 23-DOF XML at {T1_23DOF_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Carica le mesh del T1 23-DOF.

    Nel MJCF a 23 DOF il compiler usa meshdir="meshes/", quindi qui assumiamo
    che le mesh reali siano in `<...>/xmls/meshes/`.
    """
    assets: dict[str, bytes] = {}
    update_assets(assets, T1_23DOF_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    """Create an MjSpec from the T1 23-DOF XML and attach embedded assets."""
    spec = mujoco.MjSpec.from_file(str(T1_23DOF_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config (stesso modello del T1 29-DOF / G1)
##

# Motor specs (from Unitree G1 – riusati come modello per il T1).
ROTOR_INERTIAS_5020 = (
    0.139e-4,
    0.017e-4,
    0.169e-4,
)
GEARS_5020 = (
    1,
    1 + (46 / 18),
    1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
    0.489e-4,
    0.098e-4,
    0.533e-4,
)
GEARS_7520_14 = (
    1,
    4.5,
    1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
    0.489e-4,
    0.109e-4,
    0.738e-4,
)
GEARS_7520_22 = (
    1,
    4.5,
    5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
    0.068e-4,
    0.0,
    0.0,
)
GEARS_4010 = (
    1,
    5,
    5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_4010, GEARS_4010
)

# Stesso modello di attuatore elettrico del G1 / T1-29DOF
ACTUATOR_5020 = ElectricActuator(
    reflected_inertia=ARMATURE_5020,
    velocity_limit=37.0,
    effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
    reflected_inertia=ARMATURE_7520_14,
    velocity_limit=32.0,
    effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
    reflected_inertia=ARMATURE_7520_22,
    velocity_limit=20.0,
    effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
    reflected_inertia=ARMATURE_4010,
    velocity_limit=22.0,
    effort_limit=5.0,
)

# Parametri PD (come nel G1 / T1 29-DOF)
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


# ---- Mapping giunti T1-23DOF -> gruppi attuatori tipo G1 ----
#
# Joint names dal T1_23dof.xml:
#   AAHead_yaw, Head_pitch,
#   Left_Shoulder_Pitch, Left_Shoulder_Roll, Left_Elbow_Pitch, Left_Elbow_Yaw,
#   Right_Shoulder_Pitch, Right_Shoulder_Roll, Right_Elbow_Pitch, Right_Elbow_Yaw,
#   Waist,
#   Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw, Left_Knee_Pitch,
#   Left_Ankle_Pitch, Left_Ankle_Roll,
#   Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw, Right_Knee_Pitch,
#   Right_Ankle_Pitch, Right_Ankle_Roll
#
# Rispetto al 29-DOF mancano:
#   *_Wrist_Pitch, *_Wrist_Yaw, *_Hand_Roll
# Quindi i piccoli attuatori tipo 4010 (per polsi/mano) non hanno più giunti
# da pilotare nel modello a 23 DOF.

T1_23_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
    # Testa, spalle, gomiti, vita, caviglie (medium-strength)
    target_names_expr=(
        "AAHead_yaw",
        "Head_pitch",
        ".*_Shoulder_Pitch",
        ".*_Shoulder_Roll",
        ".*_Elbow_Pitch",
        ".*_Elbow_Yaw",
        "Waist",
        ".*_Ankle_Pitch",
        ".*_Ankle_Roll",
    ),
    stiffness=STIFFNESS_5020,
    damping=DAMPING_5020,
    effort_limit=ACTUATOR_5020.effort_limit,
    armature=ACTUATOR_5020.reflected_inertia,
)

T1_23_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
    # Hip pitch / yaw (grossi attuatori, tipo 7520_14)
    target_names_expr=(
        ".*_Hip_Pitch",
        ".*_Hip_Yaw",
    ),
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    effort_limit=ACTUATOR_7520_14.effort_limit,
    armature=ACTUATOR_7520_14.reflected_inertia,
)

T1_23_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
    # Hip roll + knee (più coppia, tipo 7520_22)
    target_names_expr=(
        ".*_Hip_Roll",
        ".*_Knee_Pitch",
    ),
    stiffness=STIFFNESS_7520_22,
    damping=DAMPING_7520_22,
    effort_limit=ACTUATOR_7520_22.effort_limit,
    armature=ACTUATOR_7520_22.reflected_inertia,
)

# NB: definiamo comunque l'attuatore 4010 per coerenza,
# ma nel 23-DOF non lo usiamo perché non ci sono giunti di polso/mano.
T1_23_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_Wrist_Pitch",
        ".*_Wrist_Yaw",
        ".*_Hand_Roll",
    ),
    stiffness=STIFFNESS_4010,
    damping=DAMPING_4010,
    effort_limit=ACTUATOR_4010.effort_limit,
    armature=ACTUATOR_4010.reflected_inertia,
)


##
# Initial state (stile KNEES_BENT del G1, ma con nomi T1-23DOF)
##

T1_23_KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.76),
    joint_pos={
        ".*_Hip_Pitch": -0.312,
        ".*_Knee_Pitch": 0.669,
        ".*_Ankle_Pitch": -0.363,
        ".*_Elbow_Pitch": 0.6,
        "Left_Shoulder_Roll": 0.2,
        "Left_Shoulder_Pitch": 0.2,
        "Right_Shoulder_Roll": -0.2,
        "Right_Shoulder_Pitch": 0.2,
    },
    joint_vel={".*": 0.0},
)

T1_23_HOME_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.655),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)


##
# Collision config
##
# Nel T1_23dof.xml non ci sono nomi espliciti per le geometrie del piede,
# quindi non possiamo selezionarle via regex come per L_rail_* / R_rail_* del 29-DOF.
# Usiamo semplicemente la collisione di default del MJCF, senza override.
T1_23_FEET_COLLISION = CollisionCfg(
    geom_names_expr=(),  # nessuna selezione esplicita; rimane tutto di default
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)


##
# Articulation / final config
##

T1_23_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        T1_23_ACTUATOR_5020,
        T1_23_ACTUATOR_7520_14,
        T1_23_ACTUATOR_7520_22,
        # T1_23_ACTUATOR_4010 è definito ma NON inserito qui, perché
        # nel modello a 23 DOF non esistono più i giunti di polso/mano.
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_t1_23_robot_cfg() -> EntityCfg:
    """Return an EntityCfg for the T1 23-DOF robot (G1-style config)."""
    return EntityCfg(
        init_state=T1_23_KNEES_BENT_KEYFRAME,
        collisions=(),  # nessun override esplicito; usa le collisioni di default
        spec_fn=get_spec,
        articulation=T1_23_ARTICULATION,
    )


# Action scale in stile G1: 0.25 * effort_limit / stiffness per ogni gruppo.
T1_23_ACTION_SCALE: dict[str, float] = {}
for a in T1_23_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        T1_23_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_t1_23_robot_cfg())
    viewer.launch(robot.spec.compile())
