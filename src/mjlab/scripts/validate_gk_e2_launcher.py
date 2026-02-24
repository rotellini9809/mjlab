"""Validate centralized goalkeeper launcher sampling in E2.

Runs many environment resets, aggregates launcher-family histograms, and checks
that sampled plans satisfy E2 time-to-goal and velocity clamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.goalkeeper_experts.e2_stand_block import mdp as e2_mdp
from mjlab.tasks.goalkeeper_experts.launcher import LAUNCH_FAMILY_NAMES


@dataclass(frozen=True)
class ValidateConfig:
  task_id: str = "Mjlab-GK-Expert-StandBlock-Booster-T1_23"
  num_resets: int = 200
  num_envs: int = 512
  device: str | None = None


def main() -> None:
  # Import tasks to populate registry.
  import mjlab.tasks  # noqa: F401

  cfg = tyro.cli(ValidateConfig)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  try:
    cmd = cast(e2_mdp.StandBlockCommand, env.command_manager.get_term("stand_block"))
    launcher = cmd.launcher

    family_counts = torch.zeros(len(LAUNCH_FAMILY_NAMES), device=device, dtype=torch.long)

    speed_ok_sum = 0.0
    vz_ok_sum = 0.0
    toward_ok_sum = 0.0
    tgoal_ok_sum = 0.0

    for _ in range(cfg.num_resets):
      env.reset()

      family_counts += launcher.mode_histogram()
      report = launcher.validation_report()

      speed_ok_sum += float(report["pct_speed_ok"])
      vz_ok_sum += float(report["pct_vz_ok"])
      toward_ok_sum += float(report["pct_toward_ok"])
      tgoal_ok_sum += float(report["pct_t_goal_ok"])

    total = int(torch.sum(family_counts).item())
    print("=== E2 Launcher Reset Validation ===")
    print(f"Task: {cfg.task_id}")
    print(f"Resets: {cfg.num_resets}, Envs/reset: {cfg.num_envs}, Total samples: {total}")

    print("\nFamily histogram:")
    for idx, name in enumerate(LAUNCH_FAMILY_NAMES):
      count = int(family_counts[idx].item())
      frac = (count / max(total, 1)) * 100.0
      print(f"  {name:16s}: {count:8d} ({frac:6.2f}%)")

    denom = float(max(cfg.num_resets, 1))
    print("\nAverage constraint pass rates over resets:")
    print(f"  speed <= max_speed       : {(speed_ok_sum / denom) * 100.0:6.2f}%")
    print(f"  abs(vz) <= max_abs_vz    : {(vz_ok_sum / denom) * 100.0:6.2f}%")
    print(f"  toward-goal speed >= min : {(toward_ok_sum / denom) * 100.0:6.2f}%")
    print(f"  t_goal in E2 band        : {(tgoal_ok_sum / denom) * 100.0:6.2f}%")

    # Step-event sanity: launch and deflection scheduling using vectorized launcher.step().
    env.reset()
    has_defl = launcher.has_deflection.clone()
    launcher.step(launcher.launch_time_s + 1.0e-3)
    launched_frac = float(launcher.has_launched.float().mean().item())

    launcher.step(launcher.deflect_time_s + 1.0e-3)
    deflected_frac_all = float(launcher.has_deflected.float().mean().item())
    if has_defl.any():
      deflected_frac_cond = float(launcher.has_deflected[has_defl].float().mean().item())
    else:
      deflected_frac_cond = 0.0

    print("\nStep-event sanity:")
    print(f"  launched after launch_time: {launched_frac * 100.0:6.2f}%")
    print(f"  deflected (all envs)      : {deflected_frac_all * 100.0:6.2f}%")
    print(f"  deflected | has_deflection: {deflected_frac_cond * 100.0:6.2f}%")

  finally:
    env.close()


if __name__ == "__main__":
  main()
