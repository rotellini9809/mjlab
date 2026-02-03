"""Visual inspection for Stage-1 motor controller (prior-only, render)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.motor_controller_stage1.model import (
  LatentModelConfig,
  NPMPLatentMotorPrimitive,
)
from mjlab.motor_controller_stage1.trainer import Normalizer
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  wandb_run_path: str
  num_envs: int = 1
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"


class EpisodeLoggerWrapper:
  def __init__(self, env, policy: Any | None = None):
    self.env = env
    self.policy = policy
    self.num_envs = env.num_envs
    self.device = env.device
    self.cfg = env.cfg
    self.unwrapped = env.unwrapped
    self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
    self._episode_counts = np.zeros(self.num_envs, dtype=np.int64)
    self._total_episodes = 0

  def get_observations(self) -> Any:
    return self.env.get_observations()

  def reset(self) -> Any:
    self._episode_lengths[:] = 0
    if self.policy is not None and hasattr(self.policy, "reset_envs"):
      reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
      self.policy.reset_envs(reset_mask)
    return self.env.reset()

  def step(self, actions: torch.Tensor) -> Any:
    obs, rewards, dones, extras = self.env.step(actions)
    if isinstance(dones, torch.Tensor):
      done_mask_t = dones.bool()
      done_mask = done_mask_t.cpu().numpy()
    else:
      done_mask = np.asarray(dones).astype(bool)
      done_mask_t = torch.from_numpy(done_mask).to(self.device)
    if self.policy is not None and hasattr(self.policy, "reset_envs"):
      self.policy.reset_envs(done_mask_t)
    self._episode_lengths += 1
    if done_mask.any():
      done_ids = np.where(done_mask)[0]
      for env_id in done_ids:
        self._episode_counts[env_id] += 1
        self._total_episodes += 1
        length = int(self._episode_lengths[env_id])
        if self.num_envs > 1:
          print(f"[INFO] Episode {self._total_episodes} length={length} env={env_id}")
        else:
          print(f"[INFO] Episode {self._total_episodes} length={length}")
      self._episode_lengths[done_mask] = 0
    return obs, rewards, dones, extras

  def close(self) -> None:
    return self.env.close()


class PriorPolicy:
  def __init__(
    self,
    model: NPMPLatentMotorPrimitive,
    normalizer: Normalizer,
    num_envs: int,
    device: str,
  ) -> None:
    self.model = model
    self.device = device
    self.z_prev = torch.zeros(num_envs, model.cfg.z_dim, device=device)
    self.obs_mean = torch.from_numpy(normalizer.obs_mean).to(device)
    self.obs_std = torch.from_numpy(normalizer.obs_std).to(device)
    self.act_mean = torch.from_numpy(normalizer.act_mean).to(device)
    self.act_std = torch.from_numpy(normalizer.act_std).to(device)

  def reset_envs(self, reset_mask: torch.Tensor) -> None:
    if reset_mask.any():
      self.z_prev = self.z_prev.clone()
      self.z_prev[reset_mask] = 0.0

  def __call__(self, obs: Any) -> torch.Tensor:
    obs_policy = _extract_policy_tensor(obs)
    obs_norm = (obs_policy - self.obs_mean) / self.obs_std
    prior_h = self.model.prior(self.z_prev)
    mu_p, _ = torch.chunk(prior_h, 2, dim=-1)
    z = mu_p
    a_norm = self.model.decoder(obs_norm, z)
    action = a_norm * self.act_std + self.act_mean
    self.z_prev = z.detach()
    return action


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


def _load_metadata(path: Path) -> dict[str, Any]:
  import json

  data = json.loads(path.read_text())
  required = ["dataset", "config"]
  for key in required:
    if key not in data:
      raise KeyError(f"metadata.json missing '{key}'")
  return data


def _extract_policy_tensor(obs: Any) -> torch.Tensor:
  if isinstance(obs, dict):
    if "policy" not in obs:
      raise RuntimeError("Policy observation group not found in observations.")
    obs_policy = obs["policy"]
  else:
    obs_policy = obs

  if torch.is_tensor(obs_policy):
    return obs_policy

  try:
    from tensordict import TensorDictBase  # type: ignore
  except Exception:
    TensorDictBase = None  # type: ignore

  if TensorDictBase is not None and isinstance(obs_policy, TensorDictBase):
    if "policy" in obs_policy.keys():
      obs_policy = obs_policy["policy"]
    elif "obs" in obs_policy.keys():
      obs_policy = obs_policy["obs"]
    else:
      values = [obs_policy[k] for k in sorted(obs_policy.keys())]
      obs_policy = torch.cat(values, dim=-1)
  elif isinstance(obs_policy, dict):
    if "policy" in obs_policy:
      obs_policy = obs_policy["policy"]
    else:
      values = [obs_policy[k] for k in sorted(obs_policy.keys())]
      obs_policy = torch.cat(values, dim=-1)

  if not torch.is_tensor(obs_policy):
    raise RuntimeError("Policy observations are not concatenated; cannot run.")
  return obs_policy


def _resolve_task_id_from_metadata(metadata: dict[str, Any], repo_root: Path) -> str:
  rollout_sources = metadata.get("rollout_sources")
  if not rollout_sources or not isinstance(rollout_sources, list):
    return "Mjlab-Tracking-Flat-Booster-T1_23"

  data_root_cfg = metadata.get("data_root")
  if data_root_cfg is None and isinstance(metadata.get("config"), dict):
    data_root_cfg = metadata["config"].get("data_root")
  data_root = None
  if isinstance(data_root_cfg, str):
    data_root = (repo_root / data_root_cfg).resolve()

  for src in rollout_sources:
    if not isinstance(src, dict):
      continue
    for key in ("task_id", "env_task_id", "tracking_task_id", "task"):
      value = src.get(key)
      if isinstance(value, str) and value:
        return value

    output_dir = src.get("output_dir")
    run_name = src.get("run_name")
    meta_paths: list[Path] = []
    if isinstance(output_dir, str):
      meta_paths.append(Path(output_dir) / "metadata.json")
    if data_root is not None and isinstance(run_name, str):
      meta_paths.append(data_root / run_name / "metadata.json")

    for meta_path in meta_paths:
      if not meta_path.exists():
        continue
      try:
        import json

        roll_meta = json.loads(meta_path.read_text())
      except Exception:
        continue
      for key in ("task_id", "env_task_id", "tracking_task_id", "task"):
        value = roll_meta.get(key)
        if isinstance(value, str) and value:
          return value

  return "Mjlab-Tracking-Flat-Booster-T1_23"


def main() -> None:
  cfg = tyro.cli(PlayConfig)
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  seed = 0
  torch.manual_seed(seed)
  np.random.seed(seed)

  repo_root = _find_repo_root(Path(__file__).resolve())
  log_root = repo_root / "logs" / "motor_controller_stage1"

  ckpt_path = _download_run_file(log_root, cfg.wandb_run_path, "model_last.pt")
  metadata_path = _download_run_file(log_root, cfg.wandb_run_path, "metadata.json")
  norm_path = _download_run_file(
    log_root, cfg.wandb_run_path, "normalization_stats.npz"
  )

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
    raise RuntimeError("Only latent_type=npmp is supported for prior-mode play.")

  mcfg = LatentModelConfig(
    obs_dim=int(obs_dim),
    act_dim=int(act_dim),
    k_future=int(k_future),
    z_dim=int(z_dim),
    hidden_dim=int(hidden_dim),
  )
  model = NPMPLatentMotorPrimitive(mcfg).to(device)
  data = torch.load(ckpt_path, map_location=device)
  if isinstance(data, dict) and "state_dict" in data:
    state_dict = data["state_dict"]
  elif isinstance(data, dict) and "model" in data:
    state_dict = data["model"]
  elif isinstance(data, dict):
    state_dict = data
  else:
    raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")
  model.load_state_dict(state_dict, strict=True)
  model.eval()

  normalizer = Normalizer.from_npz(norm_path)

  rollout_sources = metadata.get("rollout_sources")
  if not rollout_sources or not isinstance(rollout_sources, list):
    raise RuntimeError("metadata.json missing rollout_sources.")
  src = rollout_sources[0]
  motion_artifact_path = src.get("motion_artifact_path")
  if not motion_artifact_path:
    raise RuntimeError("rollout_sources[0] missing motion_artifact_path.")

  import wandb

  if isinstance(motion_artifact_path, str):
    parts = motion_artifact_path.split(":")
    if len(parts) > 2:
      motion_artifact_path = ":".join(parts[:2])

  artifact = wandb.Api().artifact(str(motion_artifact_path))
  motion_dir = Path(artifact.download())
  motion_file = motion_dir / "motion.npz"
  if not motion_file.exists():
    raise RuntimeError(f"motion.npz not found in artifact: {motion_artifact_path}")

  task_id = _resolve_task_id_from_metadata(metadata, repo_root)
  if task_id == "Mjlab-Tracking-Flat-Booster-T1_23":
    print("[WARN] task_id missing from metadata; using default Booster T1_23 task.")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  if not (hasattr(env_cfg, "commands") and env_cfg.commands and "motion" in env_cfg.commands):
    raise RuntimeError(
      "Selected task is not a tracking task. Unable to run Stage-1 play."
    )
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(motion_file)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = seed

  policy = PriorPolicy(model, normalizer, cfg.num_envs, device)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  env = EpisodeLoggerWrapper(env, policy=policy)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


if __name__ == "__main__":
  main()
