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
  log_joint_pos: bool = True
  log_every: int = 1


def main() -> None:
  args = tyro.cli(ViewerArgs)
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = booster_t1_23_portiere_env_cfg(play=True)
  class LoggingVecEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env: ManagerBasedRlEnv, log_every: int):
      super().__init__(env)
      self._log_every = max(1, int(log_every))
      self._log_step = 0
      self._joint_names = self.env.scene["robot"].joint_names
      print("[joint_pos] Streaming joint positions (env 0).")
      print("[joint_pos] " + " ".join(self._joint_names))

    def step(
      self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
      out = super().step(actions)
      if self._log_step % self._log_every == 0:
        joint_pos = self.env.scene["robot"].data.joint_pos[0]
        joint_pos = joint_pos.detach().cpu().tolist()
        values = " ".join(f"{v:.5f}" for v in joint_pos)
        print(f"[joint_pos] step={self._log_step} {values}", flush=True)
      self._log_step += 1
      return out

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  if args.log_joint_pos:
    env = LoggingVecEnvWrapper(env, log_every=args.log_every)
  else:
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
