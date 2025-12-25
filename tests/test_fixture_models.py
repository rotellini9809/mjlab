"""Test that fixture models compile and load correctly in MuJoCo."""

from pathlib import Path

import mujoco
import pytest


@pytest.fixture
def fixtures_dir():
  """Path to test fixtures directory."""
  return Path(__file__).parent / "fixtures"


def test_tendon_finger_compiles(fixtures_dir):
  """Test that tendon_finger.xml compiles in MuJoCo."""
  xml_path = fixtures_dir / "tendon_finger.xml"
  assert xml_path.exists(), f"Fixture not found: {xml_path}"

  # Load and compile model
  model = mujoco.MjModel.from_xml_path(str(xml_path))

  # Basic sanity checks
  assert model.ntendon == 1, "Should have 1 tendon"
  assert model.nu == 3, "Should have 3 actuators (position, velocity, effort)"
  assert model.njnt == 2, "Should have 2 joints"

  # Check tendon exists
  tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "finger_tendon")
  assert tendon_id >= 0, "Tendon 'finger_tendon' not found"

  # Check actuators exist and are tendon actuators
  for actuator_name in ["tendon_position", "tendon_velocity", "tendon_effort"]:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    assert actuator_id >= 0, f"Actuator '{actuator_name}' not found"
    # Verify it's a tendon actuator (trntype should be TENDON)
    assert model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_TENDON


def test_quadcopter_compiles(fixtures_dir):
  """Test that quadcopter.xml compiles in MuJoCo."""
  xml_path = fixtures_dir / "quadcopter.xml"
  assert xml_path.exists(), f"Fixture not found: {xml_path}"

  # Load and compile model
  model = mujoco.MjModel.from_xml_path(str(xml_path))

  # Basic sanity checks
  assert model.nu == 4, "Should have 4 actuators (one per rotor)"
  assert model.nsite >= 4, "Should have at least 4 sites (rotors)"
  assert model.njnt == 1, "Should have 1 free joint"

  # Check rotor sites exist
  for site_name in ["rotor_fl", "rotor_fr", "rotor_rl", "rotor_rr"]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    assert site_id >= 0, f"Site '{site_name}' not found"

  # Check actuators exist and are site actuators
  for actuator_name in ["thrust_fl", "thrust_fr", "thrust_rl", "thrust_rr"]:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    assert actuator_id >= 0, f"Actuator '{actuator_name}' not found"
    # Verify it's a site actuator (trntype should be SITE)
    assert model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_SITE


def test_tendon_finger_simulation(fixtures_dir):
  """Test that tendon finger can be simulated."""
  xml_path = fixtures_dir / "tendon_finger.xml"
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)

  # Reset to initial state
  mujoco.mj_resetData(model, data)

  # Run a few simulation steps
  for _ in range(10):
    mujoco.mj_step(model, data)

  # Check that simulation didn't explode (positions are reasonable)
  assert all(-10 < q < 10 for q in data.qpos), "Joint positions exploded"


def test_quadcopter_simulation(fixtures_dir):
  """Test that quadcopter can be simulated."""
  xml_path = fixtures_dir / "quadcopter.xml"
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)

  # Reset to initial state
  mujoco.mj_resetData(model, data)

  # Apply some thrust to keep it from just falling
  data.ctrl[:] = 2.5  # Small upward thrust on all rotors

  # Run a few simulation steps
  for _ in range(10):
    mujoco.mj_step(model, data)

  # Check that simulation didn't explode (positions are reasonable)
  assert all(-10 < q < 10 for q in data.qpos), "Positions exploded"
