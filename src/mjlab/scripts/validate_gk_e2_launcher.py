"""Validate centralized goalkeeper launcher sampling in E2.

Runs many environment resets, aggregates launcher-family histograms, and checks
that sampled plans satisfy E2 time-to-goal and velocity clamps.

This script can also make a curriculum-promotion recommendation from external
validation metrics (`save_rate`, `fall_rate`). Promotion still requires a human
exploit check; the code only enforces save-rate and fall-rate thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.goalkeeper_experts.e2_stand_block import mdp as e2_mdp
from mjlab.tasks.goalkeeper_experts.launcher import (
  LAUNCH_FAMILY_NAMES,
  GoalkeeperLauncherCurriculumManager,
  apply_e2_launcher_preset,
  get_e2_launcher_curriculum_preset_name,
  get_e2_launcher_curriculum_stage_index,
)
from mjlab.tasks.registry import load_env_cfg


@dataclass(frozen=True)
class ValidateConfig:
  task_id: str = "Mjlab-GK-Expert-StandBlock-Booster-T1_23"
  num_resets: int = 200
  num_envs: int = 512
  device: str | None = None
  launcher_preset_name: str | None = None
  curriculum_stage: int | None = None
  save_rate: float | None = None
  fall_rate: float | None = None
  max_fall_rate: float = 0.10


def _resolve_requested_preset_name(cfg: ValidateConfig) -> str | None:
  if cfg.launcher_preset_name is not None and cfg.curriculum_stage is not None:
    raise ValueError(
      "Use either --launcher-preset-name or --curriculum-stage, not both."
    )
  if cfg.curriculum_stage is not None:
    return get_e2_launcher_curriculum_preset_name(cfg.curriculum_stage)
  return cfg.launcher_preset_name


def _print_promotion_decision(
  *,
  preset_name: str,
  save_rate: float | None,
  fall_rate: float | None,
  max_fall_rate: float,
) -> None:
  if save_rate is None and fall_rate is None:
    return
  if save_rate is None or fall_rate is None:
    raise ValueError(
      "Both --save-rate and --fall-rate are required for promotion decisions."
    )

  stage_index = get_e2_launcher_curriculum_stage_index(preset_name)
  print("\nCurriculum promotion:")
  if stage_index is None:
    print(f"  preset {preset_name!r} is not part of the 5-stage E2 curriculum")
    print("  promotion decision skipped")
    return

  manager = GoalkeeperLauncherCurriculumManager(
    current_stage_index=stage_index,
    max_fall_rate=max_fall_rate,
  )
  decision = manager.maybe_promote(save_rate=save_rate, fall_rate=fall_rate)

  print(f"  current stage           : {decision.from_stage_index}")
  print(f"  current preset          : {decision.from_preset_name}")
  print(f"  save_rate               : {decision.save_rate:.3f}")
  print(f"  fall_rate               : {decision.fall_rate:.3f}")
  if decision.required_save_rate is not None:
    print(f"  required save_rate      : {decision.required_save_rate:.3f}")
  print(f"  max allowed fall_rate   : {decision.max_fall_rate:.3f}")
  print(f"  promote                 : {'yes' if decision.promoted else 'no'}")
  print(f"  next stage              : {decision.to_stage_index}")
  print(f"  next preset             : {decision.to_preset_name}")
  print(f"  reason                  : {decision.reason}")
  print("  human review            : exploit-check still required")
  if decision.promoted:
    print(f"  env override            : MJLAB_E2_RESET_CURRICULUM_STAGE={decision.to_stage_index}")


def main() -> None:
  # Import tasks to populate registry.
  import mjlab.tasks  # noqa: F401

  cfg = tyro.cli(ValidateConfig)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  requested_preset_name = _resolve_requested_preset_name(cfg)

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  cmd_cfg = cast(e2_mdp.StandBlockCommandCfg, env_cfg.commands["stand_block"])
  if requested_preset_name is not None:
    cmd_cfg.launcher_cfg = apply_e2_launcher_preset(
      cmd_cfg.launcher_cfg, requested_preset_name
    )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  try:
    cmd = cast(e2_mdp.StandBlockCommand, env.command_manager.get_term("stand_block"))
    launcher = cmd.launcher
    active_preset_name = cmd.launcher_preset_name
    active_stage_index = cmd.launcher_curriculum_stage

    family_counts = torch.zeros(
      len(LAUNCH_FAMILY_NAMES), device=device, dtype=torch.long
    )

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
    print(f"Preset: {active_preset_name}")
    print(
      "Curriculum stage: "
      f"{active_stage_index if active_stage_index is not None else 'baseline/custom'}"
    )
    print(
      f"Resets: {cfg.num_resets}, Envs/reset: {cfg.num_envs}, Total samples: {total}"
    )

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
      deflected_frac_cond = float(
        launcher.has_deflected[has_defl].float().mean().item()
      )
    else:
      deflected_frac_cond = 0.0

    print("\nStep-event sanity:")
    print(f"  launched after launch_time: {launched_frac * 100.0:6.2f}%")
    print(f"  deflected (all envs)      : {deflected_frac_all * 100.0:6.2f}%")
    print(f"  deflected | has_deflection: {deflected_frac_cond * 100.0:6.2f}%")
    _print_promotion_decision(
      preset_name=active_preset_name,
      save_rate=cfg.save_rate,
      fall_rate=cfg.fall_rate,
      max_fall_rate=cfg.max_fall_rate,
    )

  finally:
    env.close()


if __name__ == "__main__":
  main()
