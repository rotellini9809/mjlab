"""Viewer script for the RoboCup portiere environment."""

import os
from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.portiere.config.t1_23dof.env_cfgs import (
  booster_t1_23_portiere_env_cfg,
)
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class ViewerArgs:
  viewer: Literal["auto", "native", "viser"] = "auto"
  device: str | None = None


def main() -> None:
  args = tyro.cli(ViewerArgs)
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = booster_t1_23_portiere_env_cfg(play=True)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  action_dim = env.unwrapped.action_manager.total_action_dim

  class ZeroPolicy:
    def __call__(self, obs) -> torch.Tensor:
      del obs
      return torch.zeros((env.num_envs, action_dim), device=env.device)

  policy = ZeroPolicy()

  if args.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = args.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


if __name__ == "__main__":
  main()
