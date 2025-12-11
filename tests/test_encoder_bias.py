"""Tests for encoder offset (joint bias) modeling via qpos0.

Encoder bias: encoder_reading = true_physical_position + bias

Setting qpos0 = bias works because:
- Kinematics use (qpos - qpos0) for body positions (physical frame)
- Sensors/actuators use qpos directly (biased encoder frame)

Both sensor and actuator operate in the same biased frame, so control loops work
correctly while the physical simulation remains accurate.
"""

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.sim import Simulation, SimulationCfg

SLIDING_MASS_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="mass" pos="0 0 0">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="mass_geom" type="sphere" size="0.1" mass="1.0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pos_act" joint="slide" kp="100" kv="20"/>
  </actuator>
  <sensor>
    <jointpos name="slide_pos" joint="slide"/>
  </sensor>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def test_reset_uses_qpos0(device):
  """Verify reset_data initializes qpos to qpos0 values."""
  model = mujoco.MjModel.from_xml_string(SLIDING_MASS_XML)
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  sim.expand_model_fields(("qpos0",))

  offset = 0.5
  sim.model.qpos0[0, 0] = 0.0
  sim.model.qpos0[1, 0] = offset
  mjwarp.reset_data(sim.wp_model, sim.wp_data)

  assert torch.allclose(
    sim.data.qpos[0, 0], torch.tensor(0.0, device=device), atol=1e-6
  ), f"Expected qpos[0]=0.0, got {sim.data.qpos[0, 0]}"
  assert torch.allclose(
    sim.data.qpos[1, 0], torch.tensor(offset, device=device), atol=1e-6
  ), f"Expected qpos[1]={offset}, got {sim.data.qpos[1, 0]}"


def test_qpos_and_sensor_return_biased_values(device):
  """Verify jointpos sensor returns qpos (biased), not (qpos - qpos0)."""
  model = mujoco.MjModel.from_xml_string(SLIDING_MASS_XML)
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  sim.expand_model_fields(("qpos0",))

  offset = 0.5
  sim.model.qpos0[0, 0] = 0.0
  sim.model.qpos0[1, 0] = offset
  mjwarp.reset_data(sim.wp_model, sim.wp_data)

  assert torch.allclose(
    sim.data.qpos[0, 0], torch.tensor(0.0, device=device), atol=1e-5
  ), f"qpos[0] expected 0.0, got {sim.data.qpos[0, 0]}"
  assert torch.allclose(
    sim.data.qpos[1, 0], torch.tensor(offset, device=device), atol=1e-5
  ), f"qpos[1] expected {offset}, got {sim.data.qpos[1, 0]}"

  sim.data.ctrl[0, 0] = 0.0
  sim.data.ctrl[1, 0] = offset
  sim.step()

  assert torch.allclose(
    sim.data.qpos[0, 0], torch.tensor(0.0, device=device), atol=1e-3
  ), f"qpos[0] expected ~0.0, got {sim.data.qpos[0, 0]}"
  assert torch.allclose(
    sim.data.qpos[1, 0], torch.tensor(offset, device=device), atol=1e-3
  ), f"qpos[1] expected ~{offset}, got {sim.data.qpos[1, 0]}"

  sensor_data = sim.data.sensordata
  assert torch.allclose(sim.data.qpos[:, 0], sensor_data[:, 0], atol=1e-6), (
    f"Sensor should match data.qpos: qpos={sim.data.qpos[:, 0]}, sensor={sensor_data[:, 0]}"
  )


def test_position_actuator_uses_biased_qpos(device):
  """Verify position actuator computes error as (ctrl - qpos), ignoring qpos0."""
  model = mujoco.MjModel.from_xml_string(SLIDING_MASS_XML)
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  sim.expand_model_fields(("qpos0",))

  offset = 0.5
  sim.model.qpos0[0, 0] = 0.0
  sim.model.qpos0[1, 0] = offset
  mjwarp.reset_data(sim.wp_model, sim.wp_data)

  assert torch.allclose(
    sim.data.qpos[0, 0], torch.tensor(0.0, device=device), atol=1e-6
  )
  assert torch.allclose(
    sim.data.qpos[1, 0], torch.tensor(offset, device=device), atol=1e-6
  )

  sim.data.ctrl[:, 0] = 0.0
  sim.step()

  qfrc_actuator = sim.data.qfrc_actuator
  # Env 0: qpos=0, ctrl=0 -> no error -> no force.
  # Env 1: qpos=0.5, ctrl=0 -> error=-0.5 -> force toward 0.
  assert torch.abs(qfrc_actuator[0, 0]) < 1.0, (
    f"Env 0 should have near-zero force, got {qfrc_actuator[0, 0]}"
  )
  assert qfrc_actuator[1, 0] < -10.0, (
    f"Env 1 should have negative force, got {qfrc_actuator[1, 0]}"
  )


def test_body_position_uses_qpos_minus_qpos0(device):
  """Verify kinematics use (qpos - qpos0) so both envs are at same physical position."""
  model = mujoco.MjModel.from_xml_string(SLIDING_MASS_XML)
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  sim.expand_model_fields(("qpos0",))
  sim.create_graph()

  offset = 0.5
  sim.model.qpos0[0, 0] = 0.0
  sim.model.qpos0[1, 0] = offset
  mjwarp.reset_data(sim.wp_model, sim.wp_data)
  sim.step()

  xpos_env0 = sim.data.xpos[0, 1].cpu().numpy()
  xpos_env1 = sim.data.xpos[1, 1].cpu().numpy()
  # Both at x=0: env0 has (0-0)=0, env1 has (0.5-0.5)=0.
  np.testing.assert_allclose(xpos_env0, xpos_env1, atol=1e-5)
  np.testing.assert_allclose(xpos_env0[0], 0.0, atol=1e-5)


def test_qpos0_correctly_models_encoder_offset(device):
  """End-to-end test: same physical position, different sensor readings, consistent control."""
  model = mujoco.MjModel.from_xml_string(SLIDING_MASS_XML)
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  sim.expand_model_fields(("qpos0",))
  sim.create_graph()

  encoder_offset = 0.2
  sim.model.qpos0[0, 0] = 0.0
  sim.model.qpos0[1, 0] = encoder_offset
  mjwarp.reset_data(sim.wp_model, sim.wp_data)
  sim.step()

  # Both at same physical position.
  xpos_env0 = sim.data.xpos[0, 1].cpu().numpy()
  xpos_env1 = sim.data.xpos[1, 1].cpu().numpy()
  np.testing.assert_allclose(xpos_env0, xpos_env1, atol=1e-5)

  # But sensors read different values (encoder bias).
  sensor_env0 = sim.data.sensordata[0, 0].item()
  sensor_env1 = sim.data.sensordata[1, 0].item()
  assert abs(sensor_env0 - 0.0) < 1e-5, f"Env0 sensor should read 0, got {sensor_env0}"
  assert abs(sensor_env1 - encoder_offset) < 1e-5, (
    f"Env1 sensor should read {encoder_offset}, got {sensor_env1}"
  )

  # Command in sensor frame to hold position.
  sim.data.ctrl[0, 0] = 0.0
  sim.data.ctrl[1, 0] = encoder_offset
  for _ in range(100):
    sim.step()

  # Both stay at physical x=0.
  xpos_env0_after = sim.data.xpos[0, 1].cpu().numpy()
  xpos_env1_after = sim.data.xpos[1, 1].cpu().numpy()
  np.testing.assert_allclose(xpos_env0_after[0], 0.0, atol=0.01)
  np.testing.assert_allclose(xpos_env1_after[0], 0.0, atol=0.01)
