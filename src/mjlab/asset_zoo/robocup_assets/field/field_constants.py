"""RoboCup soccer field asset helpers."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

ROBOCUP_FIELD_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robocup_assets"
  / "field"
  / "field.xml"
)
assert ROBOCUP_FIELD_XML.exists()


def get_assets() -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ROBOCUP_FIELD_XML.parent, glob="*.png")
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROBOCUP_FIELD_XML))
  spec.assets = get_assets()
  return spec


def get_robocup_field_cfg() -> EntityCfg:
  """Return an EntityCfg for the RoboCup soccer field ground."""
  return EntityCfg(spec_fn=get_spec)
