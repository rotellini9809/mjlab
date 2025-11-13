"""Tests for XML actuator wrappers."""

import mujoco
import pytest
from conftest import get_test_device

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg

# Robot with 2 joints but only 1 actuator defined (underactuated).
ROBOT_XML_UNDERACTUATED = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
      <body name="link2" pos="0 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="actuator1" joint="joint1" gear="1.0"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def test_xml_actuator_underactuated_with_wildcard(device):
  """XmlActuator filters to joints with XML actuators when using wildcard."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_UNDERACTUATED),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlMotorActuatorCfg(joint_names_expr=(".*",)),)
    ),
  )
  entity = Entity(cfg)
  entity.compile()

  # Should only control joint1 (which has an XML actuator), not joint2.
  assert len(entity._actuators) == 1
  actuator = entity._actuators[0]
  assert actuator._joint_names == ["joint1"]


def test_xml_actuator_no_matching_actuators_raises_error(device):
  """XmlActuator raises error when no joints have matching XML actuators."""
  with pytest.raises(
    ValueError, match="No XML actuators found for any joints matching the patterns"
  ):
    cfg = EntityCfg(
      spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_UNDERACTUATED),
      articulation=EntityArticulationInfoCfg(
        actuators=(XmlMotorActuatorCfg(joint_names_expr=("joint2",)),)
      ),
    )
    entity = Entity(cfg)
    entity.compile()
