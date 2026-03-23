"""Promote or fully orchestrate E1 set-square curriculum training.

Two usage modes are supported:

1. Promotion-only:
   - pass `timeout_rate` and `fall_rate`
   - the script prints the next-stage resume command
   - add `--execute` to launch that command

2. Full curriculum loop:
   - pass `--train-iterations-per-stage`
   - the script trains stage N for that many PPO iterations
   - evaluates timeout/fall rates from the latest checkpoint
   - promotes if thresholds are met, otherwise stops early
   - repeats until the final stage finishes or promotion fails
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.goalkeeper_experts.e1_set_square import mdp as e1_mdp
from mjlab.tasks.goalkeeper_experts.e1_set_square.config.t1_23dof import (
  env_cfgs as e1_env_cfgs,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class PromoteConfig:
  task_id: str = "Mjlab-GK-Expert-SetSquare-Booster-T1_23"
  current_stage: int = 1
  timeout_rate: float = 0.0
  fall_rate: float = 1.0
  max_fall_rate: float = 0.10
  promotion_timeout_rate_thresholds: tuple[float, ...] = (0.90, 0.85)

  execute: bool = False
  gpu_ids: tuple[int, ...] = (0,)
  num_envs: int = 512
  train_iterations_per_stage: int | None = None

  eval_num_envs: int = 512
  eval_num_episodes: int = 2048
  eval_device: str | None = None

  load_run: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None

  next_run_name: str | None = None
  run_prefix: str | None = None
  extra_train_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class E1EvalMetrics:
  total_episodes: int
  timeouts: int
  falls: int
  other: int
  timeout_rate: float
  fall_rate: float


@dataclass(frozen=True)
class E1PromotionDecision:
  promoted: bool
  from_stage_index: int
  to_stage_index: int
  timeout_rate: float
  fall_rate: float
  required_timeout_rate: float | None
  max_fall_rate: float
  reason: str


@dataclass(kw_only=True)
class E1CurriculumManager:
  current_stage_index: int = 1
  promotion_timeout_rate_thresholds: tuple[float, ...] = (0.90, 0.85)
  max_fall_rate: float = 0.10

  def __post_init__(self) -> None:
    num_stages = len(e1_env_cfgs.E1_RESET_STAGE_CFGS)
    if num_stages < 1:
      raise ValueError("E1 curriculum must contain at least one stage.")
    if len(self.promotion_timeout_rate_thresholds) != num_stages - 1:
      raise ValueError(
        "promotion_timeout_rate_thresholds must have length len(E1 stages) - 1."
      )
    if not (1 <= int(self.current_stage_index) <= num_stages):
      raise ValueError(
        f"current_stage_index must be within [1, {num_stages}], got {self.current_stage_index}."
      )

  def maybe_promote(
    self,
    *,
    timeout_rate: float,
    fall_rate: float,
  ) -> E1PromotionDecision:
    from_stage_index = int(self.current_stage_index)
    final_stage = len(e1_env_cfgs.E1_RESET_STAGE_CFGS)

    if from_stage_index >= final_stage:
      return E1PromotionDecision(
        promoted=False,
        from_stage_index=from_stage_index,
        to_stage_index=from_stage_index,
        timeout_rate=float(timeout_rate),
        fall_rate=float(fall_rate),
        required_timeout_rate=None,
        max_fall_rate=float(self.max_fall_rate),
        reason="Already at final curriculum stage. No auto-demotion is supported.",
      )

    required_timeout_rate = float(
      self.promotion_timeout_rate_thresholds[from_stage_index - 1]
    )
    if float(timeout_rate) < required_timeout_rate:
      return E1PromotionDecision(
        promoted=False,
        from_stage_index=from_stage_index,
        to_stage_index=from_stage_index,
        timeout_rate=float(timeout_rate),
        fall_rate=float(fall_rate),
        required_timeout_rate=required_timeout_rate,
        max_fall_rate=float(self.max_fall_rate),
        reason=(
          f"Timeout rate {float(timeout_rate):.3f} is below required threshold "
          f"{required_timeout_rate:.3f}."
        ),
      )

    if float(fall_rate) > float(self.max_fall_rate):
      return E1PromotionDecision(
        promoted=False,
        from_stage_index=from_stage_index,
        to_stage_index=from_stage_index,
        timeout_rate=float(timeout_rate),
        fall_rate=float(fall_rate),
        required_timeout_rate=required_timeout_rate,
        max_fall_rate=float(self.max_fall_rate),
        reason=(
          f"Fall rate {float(fall_rate):.3f} exceeds max allowed "
          f"{float(self.max_fall_rate):.3f}."
        ),
      )

    return E1PromotionDecision(
      promoted=True,
      from_stage_index=from_stage_index,
      to_stage_index=from_stage_index + 1,
      timeout_rate=float(timeout_rate),
      fall_rate=float(fall_rate),
      required_timeout_rate=required_timeout_rate,
      max_fall_rate=float(self.max_fall_rate),
      reason=(
        "Promotion approved by timeout-rate and fall-rate thresholds. "
        "Exploit checks still require human review."
      ),
    )


def _gpu_args(gpu_ids: tuple[int, ...]) -> list[str]:
  if len(gpu_ids) == 0:
    return ["--gpu-ids", "None"]
  gpu_ids_arg = ", ".join(str(gpu_id) for gpu_id in gpu_ids)
  return ["--gpu-ids", f"[{gpu_ids_arg}]"]


def _build_train_command(
  cfg: PromoteConfig,
  *,
  stage: int,
  run_name: str,
  resume_local_run: str | None,
  resume_wandb_run_path: str | None,
) -> tuple[list[str], dict[str, str]]:
  cmd = ["uv", "run", "train", cfg.task_id]
  cmd.extend(_gpu_args(cfg.gpu_ids))
  cmd.extend(["--env.scene.num-envs", str(cfg.num_envs)])
  cmd.extend(["--agent.run-name", run_name])

  if cfg.train_iterations_per_stage is not None:
    cmd.extend(["--agent.max-iterations", str(cfg.train_iterations_per_stage)])

  if resume_local_run is not None or resume_wandb_run_path is not None:
    cmd.extend(["--agent.resume", "True"])
    if resume_wandb_run_path is not None:
      cmd.extend(["--wandb-run-path", resume_wandb_run_path])
      if (
        cfg.wandb_checkpoint_name is not None
        and cfg.wandb_checkpoint_name.strip() != ""
      ):
        cmd.extend(["--wandb-checkpoint-name", cfg.wandb_checkpoint_name.strip()])
    else:
      assert resume_local_run is not None
      cmd.extend(["--agent.load-run", resume_local_run])

  cmd.extend(cfg.extra_train_args)

  env = os.environ.copy()
  env["MJLAB_E1_RESET_CURRICULUM_STAGE"] = str(stage)
  return cmd, env


def _resolve_run_prefix(cfg: PromoteConfig) -> str:
  if cfg.run_prefix is not None and cfg.run_prefix.strip() != "":
    return cfg.run_prefix.strip()
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  return f"e1_{timestamp}"


def _resolve_checkpoint_for_run(task_id: str, run_name: str) -> Path:
  agent_cfg = load_rl_cfg(task_id)
  log_root_path = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
  return get_checkpoint_path(
    log_path=log_root_path,
    run_dir=run_name,
    checkpoint="model_.*.pt",
  )


def _evaluate_checkpoint(
  cfg: PromoteConfig,
  *,
  stage: int,
  checkpoint_path: Path,
) -> E1EvalMetrics:
  configure_torch_backends()
  device = cfg.eval_device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = cfg.eval_num_envs
  cmd_cfg = env_cfg.commands["set_square"]
  assert isinstance(cmd_cfg, e1_mdp.SetSquareCommandCfg)
  cmd_cfg.curriculum_stage = int(stage)

  agent_cfg = load_rl_cfg(cfg.task_id)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  try:
    runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint_path),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    obs = vec_env.get_observations()
    completed = 0
    falls = 0
    timeouts = 0
    other = 0

    while completed < cfg.eval_num_episodes:
      with torch.inference_mode():
        actions = policy(obs)
        obs, _rew, dones, _extras = vec_env.step(actions)

      done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
      if done_ids.numel() == 0:
        continue

      term_mgr = vec_env.unwrapped.termination_manager
      fall_term = term_mgr.get_term("fallen")
      time_outs = term_mgr.time_outs

      for env_idx_t in done_ids:
        if completed >= cfg.eval_num_episodes:
          break

        env_idx = int(env_idx_t.item())
        fall = bool(fall_term[env_idx].item())
        timed_out = bool(time_outs[env_idx].item())

        if fall:
          falls += 1
        elif timed_out:
          timeouts += 1
        else:
          other += 1

        completed += 1

    denom = max(completed, 1)
    return E1EvalMetrics(
      total_episodes=completed,
      timeouts=timeouts,
      falls=falls,
      other=other,
      timeout_rate=float(timeouts) / float(denom),
      fall_rate=float(falls) / float(denom),
    )
  finally:
    vec_env.close()


def _print_promotion_header(
  *,
  current_stage: int,
  timeout_rate: float,
  fall_rate: float,
  max_fall_rate: float,
  promotion_timeout_rate_thresholds: tuple[float, ...],
) -> E1PromotionDecision:
  manager = E1CurriculumManager(
    current_stage_index=current_stage,
    promotion_timeout_rate_thresholds=promotion_timeout_rate_thresholds,
    max_fall_rate=max_fall_rate,
  )
  decision = manager.maybe_promote(timeout_rate=timeout_rate, fall_rate=fall_rate)

  print("=== E1 Curriculum Promotion ===")
  print(f"current stage  : {decision.from_stage_index}")
  print(f"timeout_rate   : {decision.timeout_rate:.3f}")
  print(f"fall_rate      : {decision.fall_rate:.3f}")
  if decision.required_timeout_rate is not None:
    print(f"required timeout : {decision.required_timeout_rate:.3f}")
  print(f"max fall rate  : {decision.max_fall_rate:.3f}")
  print(f"promote        : {'yes' if decision.promoted else 'no'}")
  print(f"reason         : {decision.reason}")
  print("human review   : exploit-check still required")
  return decision


def _run_single_promotion(cfg: PromoteConfig) -> None:
  if cfg.load_run and cfg.wandb_run_path:
    raise ValueError("Use either --load-run or --wandb-run-path, not both.")

  decision = _print_promotion_header(
    current_stage=cfg.current_stage,
    timeout_rate=cfg.timeout_rate,
    fall_rate=cfg.fall_rate,
    max_fall_rate=cfg.max_fall_rate,
    promotion_timeout_rate_thresholds=cfg.promotion_timeout_rate_thresholds,
  )
  if not decision.promoted:
    return

  next_run_name = cfg.next_run_name or f"e1_stage{decision.to_stage_index}"
  cmd, _env = _build_train_command(
    cfg,
    stage=decision.to_stage_index,
    run_name=next_run_name,
    resume_local_run=cfg.load_run or f"e1_stage{cfg.current_stage}",
    resume_wandb_run_path=cfg.wandb_run_path,
  )

  print("\nTrain command:")
  print(f"MJLAB_E1_RESET_CURRICULUM_STAGE={decision.to_stage_index} {' '.join(cmd)}")

  if not cfg.execute:
    return

  print("\nLaunching resumed training...")
  subprocess.run(
    cmd,
    check=True,
    env={**os.environ, "MJLAB_E1_RESET_CURRICULUM_STAGE": str(decision.to_stage_index)},
    cwd=Path.cwd(),
  )


def _run_full_curriculum(cfg: PromoteConfig) -> None:
  if cfg.train_iterations_per_stage is None or cfg.train_iterations_per_stage <= 0:
    raise ValueError("--train-iterations-per-stage must be a positive integer.")
  if not cfg.execute:
    raise ValueError("Full curriculum mode requires --execute.")
  if cfg.load_run and cfg.wandb_run_path:
    raise ValueError("Use either --load-run or --wandb-run-path, not both.")

  run_prefix = _resolve_run_prefix(cfg)
  current_stage = int(cfg.current_stage)
  resume_local_run = cfg.load_run.strip() if cfg.load_run else None
  resume_wandb_run_path = cfg.wandb_run_path.strip() if cfg.wandb_run_path else None

  print("=== E1 Curriculum Auto-Run ===")
  print(f"start stage              : {current_stage}")
  print(f"train iterations/stage   : {cfg.train_iterations_per_stage}")
  print(f"eval episodes            : {cfg.eval_num_episodes}")
  print(f"eval num_envs            : {cfg.eval_num_envs}")
  print(f"run prefix               : {run_prefix}")
  print("human review             : exploit-check still required between promotions")

  final_stage = len(e1_env_cfgs.E1_RESET_STAGE_CFGS)

  while True:
    run_name = f"{run_prefix}_stage{current_stage}"
    cmd, env = _build_train_command(
      cfg,
      stage=current_stage,
      run_name=run_name,
      resume_local_run=resume_local_run,
      resume_wandb_run_path=resume_wandb_run_path,
    )

    print(f"\n=== Training Stage {current_stage} ===")
    print(f"run name : {run_name}")
    print(f"command  : MJLAB_E1_RESET_CURRICULUM_STAGE={current_stage} {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=Path.cwd())

    checkpoint_path = _resolve_checkpoint_for_run(cfg.task_id, run_name)
    print(f"checkpoint: {checkpoint_path}")

    metrics = _evaluate_checkpoint(
      cfg,
      stage=current_stage,
      checkpoint_path=checkpoint_path,
    )
    print("\nValidation:")
    print(f"  episodes     : {metrics.total_episodes}")
    print(f"  timeouts     : {metrics.timeouts}")
    print(f"  falls        : {metrics.falls}")
    print(f"  other        : {metrics.other}")
    print(f"  timeout_rate : {metrics.timeout_rate:.3f}")
    print(f"  fall_rate    : {metrics.fall_rate:.3f}")

    if current_stage >= final_stage:
      print(f"\nCompleted final curriculum stage {current_stage}.")
      return

    manager = E1CurriculumManager(
      current_stage_index=current_stage,
      promotion_timeout_rate_thresholds=cfg.promotion_timeout_rate_thresholds,
      max_fall_rate=cfg.max_fall_rate,
    )
    decision = manager.maybe_promote(
      timeout_rate=metrics.timeout_rate,
      fall_rate=metrics.fall_rate,
    )

    print("\nPromotion:")
    print(f"  current stage : {decision.from_stage_index}")
    print(f"  next stage    : {decision.to_stage_index}")
    print(f"  promote       : {'yes' if decision.promoted else 'no'}")
    print(f"  reason        : {decision.reason}")
    print("  human review  : exploit-check still required")

    if not decision.promoted:
      print(f"\nPromotion to stage {current_stage + 1} didn't succeed. Stopping.")
      return

    resume_local_run = run_name
    resume_wandb_run_path = None
    current_stage = decision.to_stage_index


def main() -> None:
  cfg = tyro.cli(PromoteConfig)
  if cfg.train_iterations_per_stage is None:
    _run_single_promotion(cfg)
  else:
    _run_full_curriculum(cfg)


if __name__ == "__main__":
  main()
