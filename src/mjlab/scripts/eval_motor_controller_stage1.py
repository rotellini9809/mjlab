"""Metrics-only evaluation for Stage-1 motor controller (no rendering)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.motor_controller_stage1.model import LatentModelConfig, NPMPLatentMotorPrimitive
from mjlab.motor_controller_stage1.obs_views import build_student_obs
from mjlab.motor_controller_stage1.trainer import Normalizer
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvalConfig:
  wandb_run_path: str
  num_steps: int = 5000
  num_envs: int = 32
  device: str | None = None
  checkpoint: Literal["best", "last", "latest"] = "best"


def _find_repo_root(start: Path) -> Path:
  current = start if start.is_dir() else start.parent
  for parent in [current, *current.parents]:
    if (parent / "pyproject.toml").is_file():
      return parent
    if (parent / ".git").exists():
      return parent
  raise RuntimeError("Unable to locate repo root (pyproject.toml or .git).")


def _download_run_file(log_root: Path, run_path: str, filename: str) -> Path:
  import wandb

  run_id = run_path.split("/")[-1]
  download_dir = log_root / "wandb_checkpoints" / run_id
  download_dir.mkdir(parents=True, exist_ok=True)
  target = download_dir / filename
  if target.exists():
    return target

  api = wandb.Api()
  run = api.run(run_path)
  files = [f.name for f in run.files()]
  if filename not in files:
    raise FileNotFoundError(
      f"Required file '{filename}' not found in W&B run {run_path}."
    )
  run.file(filename).download(str(download_dir), replace=True)
  return target


def _try_download_run_file(log_root: Path, run_path: str, filename: str) -> Path | None:
  try:
    return _download_run_file(log_root, run_path, filename)
  except FileNotFoundError:
    return None


def _load_metadata(path: Path) -> dict[str, Any]:
  import json

  data = json.loads(path.read_text())
  required = ["dataset", "config"]
  for key in required:
    if key not in data:
      raise KeyError(f"metadata.json missing '{key}'")
  return data


def _resolve_obs_group_name(group_names: set[str]) -> str:
  if "actor" in group_names:
    return "actor"
  raise RuntimeError(f"No 'actor' observation group found: {sorted(group_names)}")


def _extract_obs_tensor(obs: Any) -> torch.Tensor:
  if isinstance(obs, dict):
    if "actor" in obs:
      obs_group = obs["actor"]
    else:
      raise RuntimeError("Actor observation group not found in observations.")
  else:
    obs_group = obs

  if torch.is_tensor(obs_group):
    return obs_group

  try:
    from tensordict import TensorDictBase  # type: ignore
  except Exception:
    TensorDictBase = None  # type: ignore

  if TensorDictBase is not None and isinstance(obs_group, TensorDictBase):
    if "actor" in obs_group.keys():
      obs_group = obs_group["actor"]
    elif "obs" in obs_group.keys():
      obs_group = obs_group["obs"]
    else:
      values = [obs_group[k] for k in sorted(obs_group.keys())]
      obs_group = torch.cat(values, dim=-1)
  elif isinstance(obs_group, dict):
    if "actor" in obs_group:
      obs_group = obs_group["actor"]
    else:
      values = [obs_group[k] for k in sorted(obs_group.keys())]
      obs_group = torch.cat(values, dim=-1)

  if not torch.is_tensor(obs_group):
    raise RuntimeError("Actor observations are not concatenated; cannot run.")
  return obs_group


def _resolve_task_id() -> str:
  task_override = os.environ.get("MJLAB_MOTOR_CONTROLLER_TASK_ID")
  if task_override:
    print(f"[INFO] Using task override: {task_override}")
    return task_override

  return "Mjlab-Tracking-Flat-Booster-T1_23"


def _load_checkpoint(model: torch.nn.Module, path: Path, device: str) -> None:
  data = torch.load(path, map_location=device)
  if isinstance(data, dict) and "state_dict" in data:
    state_dict = data["state_dict"]
  elif isinstance(data, dict) and "model" in data:
    state_dict = data["model"]
  elif isinstance(data, dict):
    state_dict = data
  else:
    raise RuntimeError(f"Unsupported checkpoint format: {path}")
  model.load_state_dict(state_dict, strict=True)


class PriorPolicy:
  def __init__(
    self,
    model: NPMPLatentMotorPrimitive,
    normalizer: Normalizer,
    num_envs: int,
    device: str,
    env: Any | None = None,
    obs_meta: dict[str, object] | None = None,
    target_obs_dim: int | None = None,
    pad_missing: bool = False,
  ) -> None:
    self.model = model
    self.normalizer = normalizer
    self.device = device
    self.env = env
    self.obs_meta = obs_meta
    self.target_obs_dim = target_obs_dim
    self.pad_missing = pad_missing
    self.z_prev = torch.zeros(num_envs, model.cfg.z_dim, device=device)
    self.obs_mean = torch.from_numpy(normalizer.obs_mean).to(device)
    self.obs_std = torch.from_numpy(normalizer.obs_std).to(device)
    self.act_mean = torch.from_numpy(normalizer.act_mean).to(device)
    self.act_std = torch.from_numpy(normalizer.act_std).to(device)
    self.last_mu_p: torch.Tensor | None = None
    self.last_logvar_p: torch.Tensor | None = None
    self.last_z: torch.Tensor | None = None

  def reset_envs(self, reset_mask: torch.Tensor) -> None:
    if reset_mask.any():
      self.z_prev = self.z_prev.clone()
      self.z_prev[reset_mask] = 0.0

  def __call__(self, obs: Any) -> torch.Tensor:
    obs_policy = _extract_obs_tensor(obs)
    obs_student, _ = build_student_obs(obs_policy, self.obs_meta)
    if self.pad_missing and self.target_obs_dim is not None:
      if obs_student.shape[-1] < self.target_obs_dim:
        pad = self.target_obs_dim - obs_student.shape[-1]
        pad_vals = self.obs_mean[-pad:].expand(obs_student.shape[0], -1)
        obs_student = torch.cat([obs_student, pad_vals], dim=-1)
    obs_policy = obs_student
    obs_norm = (obs_policy - self.obs_mean) / self.obs_std
    prior_h = self.model.prior(self.z_prev)
    mu_p, logvar_p = torch.chunk(prior_h, 2, dim=-1)
    z = mu_p
    self.last_mu_p = mu_p.detach()
    self.last_logvar_p = logvar_p.detach()
    self.last_z = z.detach()
    a_norm = self.model.decoder(obs_norm, z)
    action = a_norm * self.act_std + self.act_mean
    self.z_prev = z.detach()
    return action


def main() -> None:
  cfg = tyro.cli(EvalConfig)
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  seed = 0
  torch.manual_seed(seed)
  np.random.seed(seed)

  repo_root = _find_repo_root(Path(__file__).resolve())
  log_root = repo_root / "logs" / "motor_controller_stage1"

  print(f"[INFO] Using fixed seed: {seed}")
  print(f"[INFO] W&B run: {cfg.wandb_run_path}")

  ckpt_path: Path | None = None
  if cfg.checkpoint == "best":
    ckpt_path = _try_download_run_file(
      log_root, cfg.wandb_run_path, "model_best.pt"
    )
    if ckpt_path is None:
      ckpt_path = _try_download_run_file(
        log_root, cfg.wandb_run_path, "model_last.pt"
      )
  elif cfg.checkpoint == "last":
    ckpt_path = _try_download_run_file(
      log_root, cfg.wandb_run_path, "model_last.pt"
    )
    if ckpt_path is None:
      ckpt_path = _try_download_run_file(
        log_root, cfg.wandb_run_path, "model_best.pt"
      )
  else:
    ckpt_path = None

  if ckpt_path is None:
    ckpt_path, was_cached = get_wandb_checkpoint_path(
      log_root, Path(cfg.wandb_run_path)
    )
    cached_str = "cached" if was_cached else "downloaded"
    print(f"[INFO] Checkpoint: {ckpt_path.name} ({cached_str})")
  else:
    print(f"[INFO] Checkpoint: {ckpt_path.name} ({cfg.checkpoint})")

  metadata_path = _download_run_file(log_root, cfg.wandb_run_path, "metadata.json")
  norm_path = _download_run_file(log_root, cfg.wandb_run_path, "normalization_stats.npz")
  print(f"[INFO] metadata.json: {metadata_path}")
  print(f"[INFO] normalization_stats.npz: {norm_path}")

  metadata = _load_metadata(metadata_path)
  dataset_info = metadata.get("dataset", {})
  config_info = metadata.get("config", {})

  obs_dim = dataset_info.get("obs_dim")
  act_dim = dataset_info.get("act_dim")
  latent_type = config_info.get("latent_type")
  z_dim = config_info.get("z_dim")
  hidden_dim = config_info.get("hidden_dim")
  k_future = config_info.get("k_future")

  required = {
    "obs_dim": obs_dim,
    "act_dim": act_dim,
    "latent_type": latent_type,
    "z_dim": z_dim,
    "hidden_dim": hidden_dim,
    "k_future": k_future,
  }
  missing = [k for k, v in required.items() if v is None]
  if missing:
    raise RuntimeError(
      f"metadata.json missing required fields: {', '.join(missing)}"
    )
  if latent_type != "npmp":
    raise RuntimeError("Only latent_type=npmp is supported for prior-mode eval.")

  mcfg = LatentModelConfig(
    obs_dim=int(obs_dim),
    act_dim=int(act_dim),
    k_future=int(k_future),
    z_dim=int(z_dim),
    hidden_dim=int(hidden_dim),
  )
  model = NPMPLatentMotorPrimitive(mcfg).to(device)
  _load_checkpoint(model, ckpt_path, device)
  model.eval()

  normalizer = Normalizer.from_npz(norm_path)

  rollout_sources = metadata.get("rollout_sources")
  if not rollout_sources or not isinstance(rollout_sources, list):
    raise RuntimeError("metadata.json missing rollout_sources.")
  src = rollout_sources[0]
  motion_artifact_path = src.get("motion_artifact_path")
  motion_artifact_name = src.get("motion_artifact_name") or motion_artifact_path
  if not motion_artifact_path:
    raise RuntimeError("rollout_sources[0] missing motion_artifact_path.")

  import wandb

  # Normalize artifact path if a version/alias was duplicated (e.g., name:v0:v0).
  if isinstance(motion_artifact_path, str):
    parts = motion_artifact_path.split(":")
    if len(parts) > 2:
      motion_artifact_path = ":".join(parts[:2])

  artifact = wandb.Api().artifact(str(motion_artifact_path))
  motion_dir = Path(artifact.download())
  motion_file = motion_dir / "motion.npz"
  if not motion_file.exists():
    raise RuntimeError(f"motion.npz not found in artifact: {motion_artifact_path}")
  print(f"[INFO] Motion source: metadata.rollout_sources[0]")
  print(f"[INFO] Motion artifact: {motion_artifact_name}")
  print(f"[INFO] Motion file: {motion_file}")

  task_id = _resolve_task_id()
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  if not (hasattr(env_cfg, "commands") and env_cfg.commands and "motion" in env_cfg.commands):
    raise RuntimeError(
      "Selected task is not a tracking task. "
      "Set MJLAB_MOTOR_CONTROLLER_TASK_ID to a tracking task."
    )
  env_cfg.commands["motion"].motion_file = str(motion_file)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = seed

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  obs_manager = env.unwrapped.observation_manager
  obs_group_name = _resolve_obs_group_name(set(obs_manager.active_terms.keys()))
  obs_meta = {
    "term_order": obs_manager.active_terms.get(obs_group_name, []),
    "term_dims": obs_manager.group_obs_term_dim.get(obs_group_name, []),
    "act_dim": env.num_actions,
  }
  obs = env.get_observations()
  obs_policy = _extract_obs_tensor(obs)
  obs_student, _ = build_student_obs(obs_policy, obs_meta)
  student_dim = int(obs_student.shape[-1])
  target_dim = int(obs_dim)
  pad_missing = False
  if student_dim != target_dim:
    if student_dim < target_dim:
      pad_missing = True
      print(
        "[WARN] Student obs dim smaller than model; "
        f"padding {target_dim - student_dim} dims with mean values."
      )
    else:
      raise RuntimeError(
        f"Student obs dim mismatch: env={student_dim} vs model={target_dim}. "
        "Set MJLAB_MOTOR_CONTROLLER_TASK_ID to a compatible task."
      )

  policy = PriorPolicy(
    model,
    normalizer,
    cfg.num_envs,
    device,
    env=env,
    obs_meta=obs_meta,
    target_obs_dim=target_dim,
    pad_missing=pad_missing,
  )

  obs = env.get_observations()
  obs_policy = _extract_obs_tensor(obs)
  obs_student, _ = build_student_obs(obs_policy, obs_meta)
  if obs_student.shape[-1] != target_dim and not pad_missing:
    raise RuntimeError(
      f"Student obs dim mismatch: env={obs_student.shape[-1]} vs model={obs_dim}. "
      "Set MJLAB_MOTOR_CONTROLLER_TASK_ID to a compatible task."
    )

  total_abs_sum = 0.0
  total_abs_sumsq = 0.0
  total_abs_max = 0.0
  total_count = 0

  done_count = 0
  ep_len_sum = 0
  ep_count = 0
  ep_lens = np.zeros(cfg.num_envs, dtype=np.int64)

  prev_actions: torch.Tensor | None = None
  smooth_sum = 0.0
  smooth_count = 0

  print(
    "[INFO] Running evaluation:"
    f" task={task_id}, num_envs={cfg.num_envs}, device={device}, steps={cfg.num_steps}"
  )

  for step in range(cfg.num_steps):
    with torch.no_grad():
      actions = policy(obs)
    obs, rewards, dones, _ = env.step(actions)

    abs_actions = actions.abs()
    total_abs_sum += float(abs_actions.sum().item())
    total_abs_sumsq += float((abs_actions * abs_actions).sum().item())
    total_abs_max = max(total_abs_max, float(abs_actions.max().item()))
    total_count += abs_actions.numel()

    if prev_actions is not None:
      diff = (actions - prev_actions).abs()
      smooth_sum += float(diff.mean().item())
      smooth_count += 1
    prev_actions = actions.detach()

    ep_lens += 1
    if isinstance(dones, torch.Tensor):
      done_mask = dones.bool().cpu().numpy()
    else:
      done_mask = np.asarray(dones).astype(bool)
    if done_mask.any():
      done_count += int(done_mask.sum())
      ep_len_sum += int(ep_lens[done_mask].sum())
      ep_count += int(done_mask.sum())
      ep_lens[done_mask] = 0
      policy.reset_envs(torch.from_numpy(done_mask).to(device))

    if (step + 1) % 500 == 0:
      mean_abs = total_abs_sum / max(1, total_count)
      var_abs = total_abs_sumsq / max(1, total_count) - mean_abs * mean_abs
      std_abs = float(np.sqrt(max(var_abs, 0.0)))
      smoothness = smooth_sum / max(1, smooth_count)
      mu_p_abs = (
        float(policy.last_mu_p.abs().mean().item())
        if policy.last_mu_p is not None
        else float("nan")
      )
      var_p = (
        float(torch.exp(policy.last_logvar_p).mean().item())
        if policy.last_logvar_p is not None
        else float("nan")
      )
      z_abs = (
        float(policy.last_z.abs().mean().item())
        if policy.last_z is not None
        else float("nan")
      )
      z_std = (
        float(policy.last_z.std().item())
        if policy.last_z is not None
        else float("nan")
      )
      print(
        f"[INFO] Step {step + 1:05d} | "
        f"|a| mean={mean_abs:.4f} std={std_abs:.4f} max={total_abs_max:.4f} "
        f"| smooth={smoothness:.4f} | dones={done_count}"
      )
      print(
        f"[INFO] Prior diag | mean|mu_p|={mu_p_abs:.4f} "
        f"mean_var_p={var_p:.4f} | mean|z|={z_abs:.4f} std(z)={z_std:.4f}"
      )

  mean_abs = total_abs_sum / max(1, total_count)
  var_abs = total_abs_sumsq / max(1, total_count) - mean_abs * mean_abs
  std_abs = float(np.sqrt(max(var_abs, 0.0)))
  mean_ep_len = (ep_len_sum / ep_count) if ep_count > 0 else float("nan")
  smoothness = smooth_sum / max(1, smooth_count)

  print("[INFO] Evaluation summary:")
  print(f"  mean|action|: {mean_abs:.4f}")
  print(f"  std|action|: {std_abs:.4f}")
  print(f"  max|action|: {total_abs_max:.4f}")
  print(f"  smoothness: {smoothness:.4f}")
  print(f"  dones: {done_count}")
  if cfg.num_steps > 0:
    print(f"  dones_per_1000_steps: {done_count / (cfg.num_steps / 1000.0):.2f}")
  print(f"  mean_episode_length: {mean_ep_len:.2f}")

  env.close()


if __name__ == "__main__":
  main()
