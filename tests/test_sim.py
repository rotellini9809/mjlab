"""Tests for sim.py."""

import mujoco
import pytest
from conftest import get_test_device

from mjlab.sim import Simulation, SimulationCfg


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def robot_xml():
  """Simple robot with geoms and joints."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1.0"
            friction="0.5 0.01 0.005"/>
          <body name="foot1" pos="0.2 0 0">
            <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="foot1_geom" type="box" size="0.05 0.05 0.05" mass="0.1"
              friction="0.5 0.01 0.005"/>
          </body>
          <body name="foot2" pos="-0.2 0 0">
            <joint name="joint2" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="foot2_geom" type="box" size="0.05 0.05 0.05" mass="0.1"
              friction="0.5 0.01 0.005"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """


def test_simulation_config_is_piped(robot_xml, device):
  """Test that SimulationCfg values are applied to warp model."""
  model = mujoco.MjModel.from_xml_string(robot_xml)

  ls_parallel = False
  maxmatch = 128

  cfg = SimulationCfg(contact_sensor_maxmatch=maxmatch, ls_parallel=ls_parallel)
  sim = Simulation(num_envs=1, cfg=cfg, model=model, device=device)

  assert sim._wp_model.opt.contact_sensor_maxmatch == maxmatch
  assert sim._wp_model.opt.ls_parallel == ls_parallel
