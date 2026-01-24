"""RoboCup goalpost asset helpers."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

GOALPOST_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robocup_assets" / "goalpost" / "goalpost.xml"
)
assert GOALPOST_XML.exists()


def get_assets() -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, GOALPOST_XML.parent, glob="*.obj")
  update_assets(assets, GOALPOST_XML.parent, glob="*.mtl")
  update_assets(assets, GOALPOST_XML.parent, glob="*.png")
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GOALPOST_XML))
  spec.assets = get_assets()
  return spec


def get_robocup_goalpost_cfg() -> EntityCfg:
  """Return an EntityCfg for the RoboCup goalpost."""
  return EntityCfg(spec_fn=get_spec)
