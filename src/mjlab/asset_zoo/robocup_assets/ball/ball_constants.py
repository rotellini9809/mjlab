"""RoboCup soccer ball asset helpers."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

ROBOCUP_BALL_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robocup_assets"
  / "ball"
  / "ball.xml"
)
assert ROBOCUP_BALL_XML.exists()


def get_assets() -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  # Pick up optional textures/meshes if they get added later.
  update_assets(assets, ROBOCUP_BALL_XML.parent, glob="*.png")
  update_assets(assets, ROBOCUP_BALL_XML.parent, glob="*.obj")
  update_assets(assets, ROBOCUP_BALL_XML.parent, glob="*.stl")
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROBOCUP_BALL_XML))
  spec.assets = get_assets()
  return spec


def get_robocup_ball_cfg() -> EntityCfg:
  """Return an EntityCfg for the RoboCup soccer ball."""
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.11)),
    spec_fn=get_spec,
  )
