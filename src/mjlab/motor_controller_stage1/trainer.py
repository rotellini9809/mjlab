from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from mjlab.motor_controller_stage1.dataset import RolloutDataset


@dataclass(frozen=True)
class TrainConfig:
  data_dir: str | None = None
  """Required for single-run. Recommended: ./data/motor_controller/rollouts/<name>."""
  data_root: str | None = None
  """Required for multi-run root with subfolders."""
  batch_size: int = 256
  max_iters: int = 50
  lr: float = 1e-3
  hidden_dim: int = 256
  z_dim: int = 32
  beta_kl: float = 1e-3
  beta_kl_start: float = 0.0
  beta_kl_end: float | None = None
  beta_kl_warmup_iters: int = 5000
  use_latent_model: bool = True
  seed: int = 42
  dry_run: bool = False
  k_future: int = 10
  sample_mode: Literal["step", "chunk"] = "step"
  latent_type: Literal["vae", "npmp"] = "npmp"
  chunk_len: int = 64
  val_frac: float = 0.1
  val_seed: int | None = None
  val_batches: int = 10
  log_every: int = 0
  grad_clip_norm: float = 1.0
  skip_non_finite_batches: bool = True
  run_name: str = ""
  device: str | None = None
  experiment_name: str = "motor_controller_stage1"
  wandb_project: str | None = "motor_controller_stage1"
  wandb_entity: str | None = None
  wandb_tags: str | None = None
  """Used for logs/<experiment_name>/<timestamp>."""


class Normalizer:
  def __init__(
    self,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    act_mean: np.ndarray | None = None,
    act_std: np.ndarray | None = None,
  ) -> None:
    self.obs_mean = obs_mean
    self.obs_std = obs_std
    self.act_mean = act_mean
    self.act_std = act_std

  @classmethod
  def from_npz(cls, path: Path) -> "Normalizer":
    data = np.load(path)
    obs_mean = np.asarray(data["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(data["obs_std"], dtype=np.float32)
    act_mean = np.asarray(data["act_mean"], dtype=np.float32)
    act_std = np.asarray(data["act_std"], dtype=np.float32)
    return cls(obs_mean=obs_mean, obs_std=obs_std, act_mean=act_mean, act_std=act_std)

  def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
    return (obs - self.obs_mean) / self.obs_std

  def normalize_act(self, act: np.ndarray) -> np.ndarray:
    if self.act_mean is None or self.act_std is None:
      return act
    return (act - self.act_mean) / self.act_std


class MetricsLogger:
  def __init__(self, path: Path) -> None:
    self.path = path
    self.fieldnames = [
      "iter",
      "split",
      "loss",
      "bc",
      "kl",
      "beta_kl",
      "post_mean_abs_mu",
      "post_mean_mu2",
      "post_mean_var",
      "prior_mean_abs_mu",
      "prior_mean_mu2",
      "prior_mean_var",
      "z_mean",
      "z_std",
      "mu_mse",
      "var_ratio",
      "elapsed_time_sec",
    ]
    self._write_header()

  def _write_header(self) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with self.path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=self.fieldnames)
      writer.writeheader()

  def log(self, row: dict[str, float | int | str]) -> None:
    with self.path.open("a", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=self.fieldnames)
      writer.writerow(row)


def train_stage1(cfg: TrainConfig) -> None:
  if (cfg.data_dir is None) == (cfg.data_root is None):
    print(
      "[ERROR] Provide exactly one of --data-dir or --data-root. "
      "Recommended: --data-dir ./data/motor_controller/rollouts/<name>"
    )
    raise SystemExit(2)

  data_dir = Path(cfg.data_dir).expanduser().resolve() if cfg.data_dir else None
  data_root = Path(cfg.data_root).expanduser().resolve() if cfg.data_root else None

  repo_root = _find_repo_root(Path(__file__).resolve())
  run_dir = _build_run_dir(repo_root, cfg.experiment_name, cfg.run_name)

  if data_dir is not None:
    print(f"[INFO] Data dir: {data_dir}")
  if data_root is not None:
    print(f"[INFO] Data root: {data_root}")
  print(f"[INFO] Run dir: {run_dir}")

  train_dataset, val_dataset, split_info = _build_datasets(cfg, data_dir, data_root)
  _print_stats(train_dataset, split_info)
  if val_dataset is not None:
    _print_val_info(val_dataset)

  norm_stats = _compute_norm_stats(train_dataset)
  norm_path = _save_norm_stats(run_dir, norm_stats)
  _print_norm_summary(norm_stats, norm_path)
  normalizer = Normalizer.from_npz(norm_path)
  _write_metadata(run_dir, train_dataset, cfg, data_dir, data_root, norm_path)

  batch = train_dataset.sample_batch(cfg.batch_size, np.random.default_rng(cfg.seed))
  _print_batch_shapes(batch, cfg.sample_mode)
  _print_normalized_batch_stats(batch, cfg.sample_mode, normalizer)

  if cfg.dry_run:
    print("[INFO] Dry run enabled. Exiting after dataset validation.")
    return

  start_time = time.time()
  metrics_logger = MetricsLogger(run_dir / "metrics.csv")
  wandb_run = _init_wandb(cfg, run_dir)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(cfg.seed)
  rng = np.random.default_rng(cfg.seed)
  best_metric = float("inf")

  obs_dim = train_dataset.stats.obs_dim
  act_dim = train_dataset.stats.act_dim

  from mjlab.motor_controller_stage1.model import (
    LatentModelConfig,
    LatentMotorPrimitive,
    NPMPLatentMotorPrimitive,
  )

  if cfg.use_latent_model:
    if cfg.latent_type == "vae":
      if cfg.k_future <= 0:
        raise ValueError(
          "Latent model requires k_future > 0 to build the encoder input."
        )
      mcfg = LatentModelConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        k_future=cfg.k_future,
        z_dim=cfg.z_dim,
        hidden_dim=cfg.hidden_dim,
      )
      model = LatentMotorPrimitive(mcfg).to(device)
    elif cfg.latent_type == "npmp":
      mcfg = LatentModelConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        k_future=cfg.k_future,
        z_dim=cfg.z_dim,
        hidden_dim=cfg.hidden_dim,
      )
      model = NPMPLatentMotorPrimitive(mcfg).to(device)
    else:
      raise ValueError(f"Unknown latent_type: {cfg.latent_type}")
  else:
    model = torch.nn.Sequential(
      torch.nn.Linear(obs_dim, cfg.hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(cfg.hidden_dim, act_dim),
    ).to(device)

  # TODO: replace with NPMP encoder/decoder and latent modeling.
  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
  mse = torch.nn.MSELoss()

  print(
    "[INFO] Starting Stage-1 placeholder training: "
    f"iters={cfg.max_iters}, batch_size={cfg.batch_size}, "
    f"sample_mode={cfg.sample_mode}, device={device}, "
    f"grad_clip_norm={cfg.grad_clip_norm}"
  )

  beta_kl_end = cfg.beta_kl_end if cfg.beta_kl_end is not None else cfg.beta_kl

  if cfg.use_latent_model:
    if cfg.latent_type == "vae" and cfg.sample_mode != "step":
      raise ValueError("latent_type=vae requires sample_mode=step.")
    if cfg.latent_type == "npmp" and cfg.sample_mode != "chunk":
      raise ValueError("latent_type=npmp requires sample_mode=chunk.")

  pbar = tqdm(range(cfg.max_iters), desc="Training", dynamic_ncols=True)
  for step in pbar:
    batch = train_dataset.sample_batch(cfg.batch_size, rng)
    beta_kl_current = _compute_beta_kl(
      cfg.beta_kl_start, beta_kl_end, cfg.beta_kl_warmup_iters, step
    )

    if cfg.sample_mode == "step":
      obs_t = torch.from_numpy(normalizer.normalize_obs(batch["obs_t"])).to(device)
      a_clean = torch.from_numpy(
        normalizer.normalize_act(batch["a_clean_t"])
      ).to(device)
      obs_future = torch.from_numpy(
        normalizer.normalize_obs(batch["obs_future"])
      ).to(device)

      if cfg.use_latent_model:
        out = model(obs_t, obs_future)
        bc_loss = mse(out["a_pred"], a_clean)
        kl_loss = out["kl"].mean()
        loss = bc_loss + beta_kl_current * kl_loss
      else:
        pred = model(obs_t)
        bc_loss = mse(pred, a_clean)
        kl_loss = torch.tensor(0.0, device=device)
        loss = bc_loss
    else:
      obs_chunk = torch.from_numpy(
        normalizer.normalize_obs(batch["obs_chunk"])
      ).to(device)
      a_clean = torch.from_numpy(
        normalizer.normalize_act(batch["a_clean_chunk"])
      ).to(device)
      obs_future = torch.from_numpy(
        normalizer.normalize_obs(batch["obs_future"])
      ).to(device)

      if cfg.use_latent_model:
        out = model(obs_chunk, obs_future)
        bc_loss = mse(out["a_pred"], a_clean)
        kl_loss = out["kl_t"].mean()
        loss = bc_loss + beta_kl_current * kl_loss
      else:
        obs_flat = obs_chunk.reshape(-1, obs_dim)
        act_flat = a_clean.reshape(-1, act_dim)
        pred = model(obs_flat)
        bc_loss = mse(pred, act_flat)
        kl_loss = torch.tensor(0.0, device=device)
        loss = bc_loss

    if not (
      _is_finite_scalar(loss)
      and _is_finite_scalar(bc_loss)
      and _is_finite_scalar(kl_loss)
    ):
      msg = (
        f"[WARN] Non-finite loss at iter {step}: "
        f"loss={loss.item()}, bc={bc_loss.item()}, kl={kl_loss.item()}"
      )
      if cfg.skip_non_finite_batches:
        pbar.write(f"{msg} | skipping optimizer step")
        optimizer.zero_grad(set_to_none=True)
        continue
      raise FloatingPointError(msg)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if cfg.grad_clip_norm > 0.0:
      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
      if not _is_finite_scalar(grad_norm):
        msg = f"[WARN] Non-finite gradient norm at iter {step}: grad_norm={grad_norm}"
        if cfg.skip_non_finite_batches:
          pbar.write(f"{msg} | skipping optimizer step")
          optimizer.zero_grad(set_to_none=True)
          continue
        raise FloatingPointError(msg)

    if not _model_grads_are_finite(model):
      msg = f"[WARN] Non-finite gradients detected at iter {step}"
      if cfg.skip_non_finite_batches:
        pbar.write(f"{msg} | skipping optimizer step")
        optimizer.zero_grad(set_to_none=True)
        continue
      raise FloatingPointError(msg)

    optimizer.step()

    log_interval = cfg.log_every if cfg.log_every and cfg.log_every > 0 else max(1, cfg.max_iters // 10)
    if step % log_interval == 0 or step == cfg.max_iters - 1:
      val_metrics = None
      if val_dataset is not None and cfg.val_batches > 0:
        val_metrics = _evaluate(
          model=model,
          dataset=val_dataset,
          normalizer=normalizer,
          device=device,
          batches=cfg.val_batches,
          batch_size=cfg.batch_size,
          mse=mse,
          use_latent=cfg.use_latent_model,
          latent_type=cfg.latent_type,
          beta_kl=beta_kl_current,
          seed=cfg.seed,
        )
      elapsed = time.time() - start_time
      train_diag = _latent_diag_from_out(out, cfg.latent_type) if cfg.use_latent_model else _nan_diag()
      pbar.set_postfix(
        loss=f"{loss.item():.4f}",
        bc=f"{bc_loss.item():.4f}",
        kl=f"{kl_loss.item():.4f}",
        beta=f"{beta_kl_current:.3e}",
      )
      pbar.write(
        f"[INFO] Iter {step:04d} | loss={loss.item():.6f} "
        f"| bc={bc_loss.item():.6f} | kl={kl_loss.item():.6f} "
        f"| beta_kl={beta_kl_current:.6f}"
      )
      if cfg.use_latent_model:
        if cfg.latent_type == "vae":
          _print_latent_diagnostics_vae(out, pbar.write)
        else:
          _print_latent_diagnostics_npmp(out, pbar.write)
          _print_prior_diagnostics_npmp(out, pbar.write)
      if val_metrics is not None:
        pbar.write(
          f"[INFO] Val | loss={val_metrics['loss']:.6f} "
          f"| bc={val_metrics['bc']:.6f} | kl={val_metrics['kl']:.6f}"
        )
        if cfg.use_latent_model and cfg.latent_type == "npmp":
          val_diag = val_metrics.get("diag", _nan_diag())
          pbar.write(
            "[INFO] Val prior diag | "
            f"p_mean|mu|={val_diag.get('prior_mean_abs_mu', float('nan')):.4f} "
            f"p_mean_var={val_diag.get('prior_mean_var', float('nan')):.4f} | "
            f"q_mean|mu|={val_diag.get('post_mean_abs_mu', float('nan')):.4f} "
            f"q_mean_var={val_diag.get('post_mean_var', float('nan')):.4f} | "
            f"mu_mse={val_diag.get('mu_mse', float('nan')):.4f} "
            f"var_ratio={val_diag.get('var_ratio', float('nan')):.4f}"
          )

      metrics_logger.log(
        _build_metrics_row(
          step,
          "train",
          loss.item(),
          bc_loss.item(),
          kl_loss.item(),
          beta_kl_current,
          train_diag,
          elapsed,
        )
      )
      _wandb_log_metrics(
        wandb_run,
        step,
        "train",
        loss.item(),
        bc_loss.item(),
        kl_loss.item(),
        beta_kl_current,
        train_diag,
        elapsed,
      )

      if val_metrics is not None:
        val_diag = val_metrics.get("diag", _nan_diag())
        metrics_logger.log(
          _build_metrics_row(
            step,
            "val",
            val_metrics["loss"],
            val_metrics["bc"],
            val_metrics["kl"],
            beta_kl_current,
            val_diag,
            elapsed,
          )
        )
        _wandb_log_metrics(
          wandb_run,
          step,
          "val",
          val_metrics["loss"],
          val_metrics["bc"],
          val_metrics["kl"],
          beta_kl_current,
          val_diag,
          elapsed,
        )

      candidate = (
        val_metrics["loss"]
        if val_metrics is not None
        else float(loss.item())
      )
      if candidate < best_metric:
        best_metric = candidate
        _save_checkpoint(run_dir / "model_best.pt", model, step)

  print("[INFO] Stage-1 placeholder training complete.")
  _save_checkpoint(run_dir / "model_last.pt", model, cfg.max_iters)
  run_path = _wandb_run_path(wandb_run)
  if run_path is not None:
    print(f"[INFO] W&B run saved at: {run_path}")
  _wandb_save_files(wandb_run, run_dir)
  wandb_run.finish()


def _find_repo_root(start: Path) -> Path:
  current = start if start.is_dir() else start.parent
  for parent in [current, *current.parents]:
    if (parent / "pyproject.toml").is_file():
      return parent
    if (parent / ".git").exists():
      return parent
  raise RuntimeError("Unable to locate repo root (pyproject.toml or .git).")


def _build_run_dir(repo_root: Path, experiment_name: str, run_name: str) -> Path:
  log_root = repo_root / "logs" / experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if run_name:
    log_dir_name += f"_{run_name}"
  run_dir = log_root / log_dir_name
  run_dir.mkdir(parents=True, exist_ok=True)
  return run_dir.resolve()


def _build_datasets(
  cfg: TrainConfig, data_dir: Path | None, data_root: Path | None
) -> tuple[RolloutDataset, RolloutDataset | None, dict[str, object]]:
  split_info: dict[str, object] = {}

  full_dataset = RolloutDataset(
    data_dir=data_dir,
    data_root=data_root,
    k_future=cfg.k_future,
    sample_mode=cfg.sample_mode,
    chunk_len=cfg.chunk_len,
    log_env_id_summary=False,
  )

  keys_by_run = full_dataset.episode_keys_by_run()
  total_eps = sum(len(keys) for keys in keys_by_run.values())
  val_seed = cfg.val_seed if cfg.val_seed is not None else cfg.seed

  train_keys: set[tuple[str, int | None, int]] = set()
  val_keys: set[tuple[str, int | None, int]] = set()
  per_run_train: dict[str, int] = {}
  per_run_val: dict[str, int] = {}

  rng = np.random.default_rng(val_seed)
  for run_name in sorted(keys_by_run.keys()):
    episodes = list(keys_by_run[run_name])
    if not episodes:
      continue
    rng.shuffle(episodes)
    if cfg.val_frac <= 0.0:
      val_count = 0
    else:
      val_count = int(len(episodes) * cfg.val_frac)
      if val_count == 0 and len(episodes) >= 2:
        val_count = 1
      if val_count >= len(episodes):
        val_count = len(episodes) - 1
    val_set = set(episodes[:val_count])
    train_set = set(episodes[val_count:])
    val_keys |= val_set
    train_keys |= train_set
    per_run_val[run_name] = len(val_set)
    per_run_train[run_name] = len(train_set)

  split_info["val_frac"] = cfg.val_frac
  split_info["total_episodes"] = total_eps
  split_info["train_episodes"] = len(train_keys)
  split_info["val_episodes"] = len(val_keys)
  split_info["per_run_train_episodes"] = per_run_train
  split_info["per_run_val_episodes"] = per_run_val

  train_dataset = RolloutDataset(
    data_dir=None,
    data_root=None,
    k_future=cfg.k_future,
    sample_mode=cfg.sample_mode,
    chunk_len=cfg.chunk_len,
    shards=full_dataset.shards,
    episode_allowlist=train_keys,
    log_prefix="train",
  )

  val_dataset = None
  if val_keys:
    val_dataset = RolloutDataset(
      data_dir=None,
      data_root=None,
      k_future=cfg.k_future,
      sample_mode=cfg.sample_mode,
      chunk_len=cfg.chunk_len,
      shards=full_dataset.shards,
      episode_allowlist=val_keys,
      log_prefix="val",
    )

  return train_dataset, val_dataset, split_info


def _print_stats(dataset: RolloutDataset, split_info: dict[str, object]) -> None:
  stats = dataset.stats
  print("[INFO] Rollout dataset stats:")
  print(f"  num_runs: {stats.num_runs}")
  print(f"  num_shards: {stats.num_shards}")
  print(f"  total_steps: {stats.total_steps}")
  print(f"  obs_dim: {stats.obs_dim}")
  print(f"  act_dim: {stats.act_dim}")
  print(f"  keys_found: {', '.join(stats.keys_found)}")
  print(f"  run_names: {', '.join(stats.run_names)}")
  print(f"  step_samples: {dataset.step_count}")
  print(f"  chunk_samples: {dataset.chunk_count}")
  if split_info:
    print("[INFO] Split summary:")
    for key, value in split_info.items():
      print(f"  {key}: {value}")
  print("[INFO] Sample shard shapes:")
  for key, shape in stats.sample_shapes.items():
    print(f"  {key}: {shape}")


def _print_val_info(dataset: RolloutDataset) -> None:
  stats = dataset.stats
  print("[INFO] Validation dataset stats:")
  print(f"  num_runs: {stats.num_runs}")
  print(f"  num_shards: {stats.num_shards}")
  print(f"  total_steps: {stats.total_steps}")


def _compute_norm_stats(dataset: RolloutDataset) -> dict[str, np.ndarray]:
  obs_count = 0
  obs_mean = None
  obs_m2 = None

  act_count = 0
  act_mean = None
  act_m2 = None

  for shard_idx, rows in dataset.iter_allowed_row_indices():
    shard = dataset.shards[shard_idx]
    obs_count, obs_mean, obs_m2 = _update_running_stats(
      obs_count, obs_mean, obs_m2, shard.obs[rows]
    )
    act_count, act_mean, act_m2 = _update_running_stats(
      act_count, act_mean, act_m2, shard.a_clean[rows]
    )

  if obs_count == 0 or act_count == 0:
    raise RuntimeError("Cannot compute normalization stats from empty dataset.")

  obs_var = obs_m2 / obs_count
  act_var = act_m2 / act_count
  obs_std = np.sqrt(np.maximum(obs_var, 1e-6))
  act_std = np.sqrt(np.maximum(act_var, 1e-6))

  return {
    "obs_mean": obs_mean.astype(np.float32),
    "obs_std": obs_std.astype(np.float32),
    "act_mean": act_mean.astype(np.float32),
    "act_std": act_std.astype(np.float32),
  }


def _update_running_stats(
  count: int,
  mean: np.ndarray | None,
  m2: np.ndarray | None,
  batch: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
  if batch.shape[0] == 0:
    if mean is None or m2 is None:
      raise RuntimeError("Empty batch encountered before stats initialization.")
    return count, mean, m2

  batch = batch.astype(np.float64, copy=False)
  batch_count = batch.shape[0]
  batch_mean = batch.mean(axis=0)
  batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)

  if count == 0 or mean is None or m2 is None:
    return batch_count, batch_mean, batch_m2

  total = count + batch_count
  delta = batch_mean - mean
  mean = mean + delta * batch_count / total
  m2 = m2 + batch_m2 + delta * delta * count * batch_count / total
  return total, mean, m2


def _save_norm_stats(run_dir: Path, stats: dict[str, np.ndarray]) -> Path:
  out_path = run_dir / "normalization_stats.npz"
  np.savez(out_path, **stats)
  return out_path


def _print_norm_summary(stats: dict[str, np.ndarray], path: Path) -> None:
  obs_mean = stats["obs_mean"]
  obs_std = stats["obs_std"]
  act_mean = stats["act_mean"]
  act_std = stats["act_std"]

  print("[INFO] Normalization stats:")
  print(f"  obs_mean shape: {obs_mean.shape}, obs_std shape: {obs_std.shape}")
  print(f"  act_mean shape: {act_mean.shape}, act_std shape: {act_std.shape}")
  print(f"  obs_mean[0:5]: {obs_mean[:5]}")
  print(f"  obs_std[0:5]: {obs_std[:5]}")
  print(f"  act_mean[0:5]: {act_mean[:5]}")
  print(f"  act_std[0:5]: {act_std[:5]}")
  print(f"[INFO] Saved normalization stats: {path}")


def _print_batch_shapes(batch: dict[str, np.ndarray], sample_mode: str) -> None:
  print("[INFO] Sample batch shapes:")
  if sample_mode == "step":
    print(f"  obs_t: {batch['obs_t'].shape}")
    print(f"  a_clean_t: {batch['a_clean_t'].shape}")
    print(f"  obs_future: {batch['obs_future'].shape}")
  else:
    print(f"  obs_chunk: {batch['obs_chunk'].shape}")
    print(f"  a_clean_chunk: {batch['a_clean_chunk'].shape}")
    print(f"  obs_future: {batch['obs_future'].shape}")


def _print_normalized_batch_stats(
  batch: dict[str, np.ndarray],
  sample_mode: str,
  normalizer: Normalizer,
) -> None:
  if sample_mode == "step":
    obs = normalizer.normalize_obs(batch["obs_t"])
  else:
    obs = normalizer.normalize_obs(batch["obs_chunk"]).reshape(-1, batch["obs_chunk"].shape[-1])
  mean = obs.mean(axis=0)
  std = obs.std(axis=0)
  print("[INFO] Normalized obs batch stats:")
  print(f"  mean[0:5]: {mean[:5]}")
  print(f"  std[0:5]: {std[:5]}")


def _compute_beta_kl(
  beta_start: float, beta_end: float, warmup_iters: int, step: int
) -> float:
  if warmup_iters <= 0:
    return beta_end
  progress = min(step, warmup_iters) / float(warmup_iters)
  return beta_start + (beta_end - beta_start) * progress


def _is_finite_scalar(value: torch.Tensor | float) -> bool:
  if isinstance(value, torch.Tensor):
    if value.numel() == 0:
      return False
    return bool(torch.isfinite(value).all().item())
  return bool(np.isfinite(value))


def _model_grads_are_finite(model: torch.nn.Module) -> bool:
  for param in model.parameters():
    if param.grad is None:
      continue
    if not torch.isfinite(param.grad).all():
      return False
  return True


def _evaluate(
  model: torch.nn.Module,
  dataset: RolloutDataset,
  normalizer: Normalizer,
  device: str,
  batches: int,
  batch_size: int,
  mse: torch.nn.Module,
  use_latent: bool,
  latent_type: str,
  beta_kl: float,
  seed: int,
) -> dict[str, float | dict[str, float]]:
  model.eval()
  total_loss = 0.0
  total_bc = 0.0
  total_kl = 0.0
  valid_batches = 0
  diag_values: dict[str, list[float]] = {
    "post_mean_abs_mu": [],
    "post_mean_mu2": [],
    "post_mean_var": [],
    "prior_mean_abs_mu": [],
    "prior_mean_mu2": [],
    "prior_mean_var": [],
    "z_mean": [],
    "z_std": [],
    "mu_mse": [],
    "var_ratio": [],
  }
  rng = np.random.default_rng(seed)

  with torch.no_grad():
    for _ in range(batches):
      batch = dataset.sample_batch(batch_size, rng)
      if dataset.sample_mode == "step":
        if use_latent and latent_type == "npmp":
          raise ValueError("latent_type=npmp requires sample_mode=chunk for validation.")
        obs_t = torch.from_numpy(normalizer.normalize_obs(batch["obs_t"])).to(device)
        a_clean = torch.from_numpy(
          normalizer.normalize_act(batch["a_clean_t"])
        ).to(device)
        obs_future = torch.from_numpy(
          normalizer.normalize_obs(batch["obs_future"])
        ).to(device)

        if use_latent:
          out = model(obs_t, obs_future)
          bc = mse(out["a_pred"], a_clean)
          kl = out["kl"].mean()
          loss = bc + beta_kl * kl
        else:
          pred = model(obs_t)
          bc = mse(pred, a_clean)
          kl = torch.tensor(0.0, device=device)
          loss = bc
      else:
        if use_latent and latent_type == "vae":
          raise ValueError("latent_type=vae requires sample_mode=step for validation.")
        obs_chunk = torch.from_numpy(
          normalizer.normalize_obs(batch["obs_chunk"])
        ).to(device)
        a_clean = torch.from_numpy(
          normalizer.normalize_act(batch["a_clean_chunk"])
        ).to(device)
        obs_future = torch.from_numpy(
          normalizer.normalize_obs(batch["obs_future"])
        ).to(device)

        if use_latent:
          out = model(obs_chunk, obs_future)
          bc = mse(out["a_pred"], a_clean)
          kl = out["kl_t"].mean()
          loss = bc + beta_kl * kl
        else:
          obs_flat = obs_chunk.reshape(-1, obs_chunk.shape[-1])
          act_flat = a_clean.reshape(-1, a_clean.shape[-1])
          pred = model(obs_flat)
          bc = mse(pred, act_flat)
          kl = torch.tensor(0.0, device=device)
          loss = bc

      if not (
        _is_finite_scalar(loss)
        and _is_finite_scalar(bc)
        and _is_finite_scalar(kl)
      ):
        continue

      valid_batches += 1
      total_loss += float(loss.item())
      total_bc += float(bc.item())
      total_kl += float(kl.item())
      if use_latent:
        diag = _latent_diag_from_out(out, latent_type)
        for key, value in diag.items():
          diag_values[key].append(value)

  model.train()
  if valid_batches == 0:
    result: dict[str, float | dict[str, float]] = {
      "loss": float("nan"),
      "bc": float("nan"),
      "kl": float("nan"),
    }
    if use_latent:
      result["diag"] = _nan_diag()
    return result

  denom = valid_batches
  result: dict[str, float | dict[str, float]] = {
    "loss": total_loss / denom,
    "bc": total_bc / denom,
    "kl": total_kl / denom,
  }
  if use_latent:
    diag_avg = {}
    for key, values in diag_values.items():
      finite_values = [v for v in values if np.isfinite(v)]
      diag_avg[key] = float(np.mean(finite_values)) if finite_values else float("nan")
    result["diag"] = diag_avg
  return result


def _print_latent_diagnostics_vae(
  out: dict[str, torch.Tensor], write_fn=print
) -> None:
  mu = out["mu"]
  logvar = out["logvar"]
  z = out["z"]

  mean_abs_mu = mu.abs().mean().item()
  mean_mu2 = (mu * mu).mean().item()
  mean_var = torch.exp(logvar).mean().item()

  kl_per_dim = 0.5 * (torch.exp(logvar) + mu * mu - 1.0 - logvar)
  active_units = (kl_per_dim.mean(dim=0) > 0.01).sum().item()

  z_mean = z.mean().item()
  z_std = z.std().item()

  write_fn(
    "[INFO] Latent diag | "
    f"mean|mu|={mean_abs_mu:.4f} "
    f"mean(mu^2)={mean_mu2:.4f} "
    f"mean_var={mean_var:.4f} "
    f"active_units={active_units} "
    f"z_mean={z_mean:.4f} z_std={z_std:.4f}"
  )


def _print_latent_diagnostics_npmp(
  out: dict[str, torch.Tensor], write_fn=print
) -> None:
  mu_q = out["mu_q"]
  logvar_q = out["logvar_q"]
  mu_p = out["mu_p"]
  logvar_p = out["logvar_p"]
  z = out["z"]

  mu_q_flat = mu_q.reshape(-1, mu_q.shape[-1])
  logvar_q_flat = logvar_q.reshape(-1, logvar_q.shape[-1])
  mu_p_flat = mu_p.reshape(-1, mu_p.shape[-1])
  logvar_p_flat = logvar_p.reshape(-1, logvar_p.shape[-1])
  z_flat = z.reshape(-1, z.shape[-1])

  mean_abs_mu = mu_q_flat.abs().mean().item()
  mean_mu2 = (mu_q_flat * mu_q_flat).mean().item()
  mean_var = torch.exp(logvar_q_flat).mean().item()

  kl_per_dim = 0.5 * (
    logvar_p_flat
    - logvar_q_flat
    + (torch.exp(logvar_q_flat) + (mu_q_flat - mu_p_flat) ** 2)
    / torch.exp(logvar_p_flat)
    - 1.0
  )
  active_units = (kl_per_dim.mean(dim=0) > 0.01).sum().item()

  z_mean = z_flat.mean().item()
  z_std = z_flat.std().item()

  write_fn(
    "[INFO] Latent diag | "
    f"mean|mu_q|={mean_abs_mu:.4f} "
    f"mean(mu_q^2)={mean_mu2:.4f} "
    f"mean_var_q={mean_var:.4f} "
    f"active_units={active_units} "
    f"z_mean={z_mean:.4f} z_std={z_std:.4f}"
  )


def _print_prior_diagnostics_npmp(
  out: dict[str, torch.Tensor], write_fn=print, eps: float = 1e-8
) -> None:
  mu_q = out["mu_q"]
  logvar_q = out["logvar_q"]
  mu_p = out["mu_p"]
  logvar_p = out["logvar_p"]

  mu_q_flat = mu_q.reshape(-1, mu_q.shape[-1])
  logvar_q_flat = logvar_q.reshape(-1, logvar_q.shape[-1])
  mu_p_flat = mu_p.reshape(-1, mu_p.shape[-1])
  logvar_p_flat = logvar_p.reshape(-1, logvar_p.shape[-1])

  var_q = torch.exp(logvar_q_flat)
  var_p = torch.exp(logvar_p_flat).clamp_min(eps)

  p_mean_abs = mu_p_flat.abs().mean().item()
  p_mean_var = var_p.mean().item()
  q_mean_abs = mu_q_flat.abs().mean().item()
  q_mean_var = var_q.mean().item()
  mu_mse = ((mu_q_flat - mu_p_flat) ** 2).mean().item()
  var_ratio = (var_q / var_p).mean().item()

  write_fn(
    "[INFO] Prior diag | "
    f"p_mean|mu|={p_mean_abs:.4f} p_mean_var={p_mean_var:.4f} | "
    f"q_mean|mu|={q_mean_abs:.4f} q_mean_var={q_mean_var:.4f} | "
    f"mu_mse={mu_mse:.4f} var_ratio={var_ratio:.4f}"
  )


def _write_metadata(
  run_dir: Path,
  dataset: RolloutDataset,
  cfg: TrainConfig,
  data_dir: Path | None,
  data_root: Path | None,
  norm_path: Path,
) -> None:
  rollouts_info = _collect_rollout_metadata(data_dir, data_root)
  payload = {
    "task_id": "motor_controller_stage1",
    "data_dir": str(data_dir) if data_dir is not None else None,
    "data_root": str(data_root) if data_root is not None else None,
    "run_dir": str(run_dir),
    "normalization_stats": str(norm_path),
    "rollout_sources": rollouts_info,
    "dataset": {
      "num_runs": dataset.stats.num_runs,
      "num_shards": dataset.stats.num_shards,
      "total_steps": dataset.stats.total_steps,
      "obs_dim": dataset.stats.obs_dim,
      "act_dim": dataset.stats.act_dim,
      "keys_found": list(dataset.stats.keys_found),
      "run_names": list(dataset.stats.run_names),
      "step_samples": dataset.step_count,
      "chunk_samples": dataset.chunk_count,
    },
    "config": asdict(cfg),
  }
  out_path = run_dir / "metadata.json"
  out_path.write_text(json.dumps(payload, indent=2))


def _collect_rollout_metadata(
  data_dir: Path | None, data_root: Path | None
) -> list[dict[str, object]]:
  roots: list[Path] = []
  if data_dir is not None:
    roots = [data_dir]
  elif data_root is not None:
    roots = [p for p in data_root.iterdir() if p.is_dir()]

  results: list[dict[str, object]] = []
  for root in sorted(roots):
    meta_path = root / "metadata.json"
    if not meta_path.exists():
      continue
    try:
      meta = json.loads(meta_path.read_text())
    except Exception:
      continue
    motion_artifact = meta.get("motion_artifact")
    motion_name = None
    if isinstance(motion_artifact, str):
      # Expected format: entity/project/name:version
      name_part = motion_artifact.split("/")[-1]
      motion_name = name_part.split(":")[0] if name_part else None
    results.append(
      {
        "run_name": meta.get("run_name"),
        "wandb_run_path": meta.get("wandb_run_path"),
        "motion_artifact_path": motion_artifact,
        "motion_artifact_name": motion_name,
        "output_dir": meta.get("output_dir"),
      }
    )
  return results


def _latent_diag_from_out(
  out: dict[str, torch.Tensor], latent_type: str, eps: float = 1e-8
) -> dict[str, float]:
  if latent_type == "vae":
    mu = out["mu"]
    logvar = out["logvar"]
    z = out["z"]
    mu_abs = mu.abs().mean().item()
    mu2 = (mu * mu).mean().item()
    var_q = torch.exp(logvar).mean().item()
    z_mean = z.mean().item()
    z_std = z.std().item()
    return {
      "post_mean_abs_mu": float(mu_abs),
      "post_mean_mu2": float(mu2),
      "post_mean_var": float(var_q),
      "prior_mean_abs_mu": float("nan"),
      "prior_mean_mu2": float("nan"),
      "prior_mean_var": float("nan"),
      "z_mean": float(z_mean),
      "z_std": float(z_std),
      "mu_mse": float("nan"),
      "var_ratio": float("nan"),
    }
  if latent_type == "npmp":
    mu_q = out["mu_q"]
    logvar_q = out["logvar_q"]
    mu_p = out["mu_p"]
    logvar_p = out["logvar_p"]
    z = out["z"]
    mu_q_flat = mu_q.reshape(-1, mu_q.shape[-1])
    logvar_q_flat = logvar_q.reshape(-1, logvar_q.shape[-1])
    mu_p_flat = mu_p.reshape(-1, mu_p.shape[-1])
    logvar_p_flat = logvar_p.reshape(-1, logvar_p.shape[-1])
    z_flat = z.reshape(-1, z.shape[-1])
    mu_abs = mu_q_flat.abs().mean().item()
    mu2 = (mu_q_flat * mu_q_flat).mean().item()
    var_q = torch.exp(logvar_q_flat)
    mu_p_abs = mu_p_flat.abs().mean().item()
    mu_p2 = (mu_p_flat * mu_p_flat).mean().item()
    var_p = torch.exp(logvar_p_flat).clamp_min(eps)
    z_mean = z_flat.mean().item()
    z_std = z_flat.std().item()
    mu_mse = ((mu_q_flat - mu_p_flat) ** 2).mean().item()
    var_ratio = (var_q / var_p).mean().item()
    return {
      "post_mean_abs_mu": float(mu_abs),
      "post_mean_mu2": float(mu2),
      "post_mean_var": float(var_q.mean().item()),
      "prior_mean_abs_mu": float(mu_p_abs),
      "prior_mean_mu2": float(mu_p2),
      "prior_mean_var": float(var_p.mean().item()),
      "z_mean": float(z_mean),
      "z_std": float(z_std),
      "mu_mse": float(mu_mse),
      "var_ratio": float(var_ratio),
    }
  raise ValueError(f"Unknown latent_type: {latent_type}")


def _nan_diag() -> dict[str, float]:
  return {
    "post_mean_abs_mu": float("nan"),
    "post_mean_mu2": float("nan"),
    "post_mean_var": float("nan"),
    "prior_mean_abs_mu": float("nan"),
    "prior_mean_mu2": float("nan"),
    "prior_mean_var": float("nan"),
    "z_mean": float("nan"),
    "z_std": float("nan"),
    "mu_mse": float("nan"),
    "var_ratio": float("nan"),
  }


def _build_metrics_row(
  step: int,
  split: str,
  loss: float,
  bc: float,
  kl: float,
  beta_kl: float,
  diag: dict[str, float],
  elapsed: float,
) -> dict[str, float | int | str]:
  return {
    "iter": int(step),
    "split": split,
    "loss": float(loss),
    "bc": float(bc),
    "kl": float(kl),
    "beta_kl": float(beta_kl),
    "post_mean_abs_mu": float(diag.get("post_mean_abs_mu", float("nan"))),
    "post_mean_mu2": float(diag.get("post_mean_mu2", float("nan"))),
    "post_mean_var": float(diag.get("post_mean_var", float("nan"))),
    "prior_mean_abs_mu": float(diag.get("prior_mean_abs_mu", float("nan"))),
    "prior_mean_mu2": float(diag.get("prior_mean_mu2", float("nan"))),
    "prior_mean_var": float(diag.get("prior_mean_var", float("nan"))),
    "z_mean": float(diag.get("z_mean", float("nan"))),
    "z_std": float(diag.get("z_std", float("nan"))),
    "mu_mse": float(diag.get("mu_mse", float("nan"))),
    "var_ratio": float(diag.get("var_ratio", float("nan"))),
    "elapsed_time_sec": float(elapsed),
  }


def _init_wandb(cfg: TrainConfig, run_dir: Path):
  try:
    import wandb
  except ImportError as exc:
    raise RuntimeError(
      "W&B is required but the 'wandb' package is not installed."
    ) from exc

  # Match rsl_rl-style logging: place wandb/ under the experiment log root,
  # not inside the specific run_dir.
  os.environ.setdefault("WANDB_DIR", str(run_dir.parent))

  api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("wandb_api_key")
  if api_key and not os.environ.get("WANDB_API_KEY"):
    os.environ["WANDB_API_KEY"] = api_key
  try:
    login_ok = (
      wandb.login(key=api_key, relogin=False)
      if api_key
      else wandb.login()
    )
  except Exception as exc:
    raise RuntimeError(
      "W&B is required but login failed. "
      "Set WANDB_API_KEY (or run `wandb login`) and retry."
    ) from exc
  if login_ok is False:
    raise RuntimeError(
      "W&B is required but login was not successful. "
      "Set WANDB_API_KEY (or run `wandb login`) and retry."
    )

  tags = None
  if cfg.wandb_tags:
    tags = [tag.strip() for tag in cfg.wandb_tags.split(",") if tag.strip()]

  try:
    wandb_run = wandb.init(
      project=cfg.wandb_project,
      entity=cfg.wandb_entity,
      tags=tags,
      name=run_dir.name,
      dir=str(run_dir.parent),
    )
  except Exception as exc:
    raise RuntimeError(
      "W&B is required but run initialization failed. "
      "Check project/entity permissions and network access."
    ) from exc
  if wandb_run is None:
    raise RuntimeError("W&B is required but wandb.init returned None.")
  wandb_run.config.update(asdict(cfg), allow_val_change=True)
  run_path = _wandb_run_path(wandb_run)
  if run_path is not None:
    print(f"[INFO] W&B run path: {run_path}")
  run_url = getattr(wandb_run, "url", None)
  if isinstance(run_url, str) and run_url:
    print(f"[INFO] W&B URL: {run_url}")
  return wandb_run


def _wandb_run_path(wandb_run) -> str | None:
  entity = getattr(wandb_run, "entity", None)
  project = getattr(wandb_run, "project", None)
  run_id = getattr(wandb_run, "id", None)
  if entity and project and run_id:
    return f"{entity}/{project}/{run_id}"
  return None


def _wandb_log_metrics(
  wandb_run,
  step: int,
  split: str,
  loss: float,
  bc: float,
  kl: float,
  beta_kl: float,
  diag: dict[str, float],
  elapsed: float,
) -> None:
  payload = {
    f"{split}/loss": loss,
    f"{split}/bc": bc,
    f"{split}/kl": kl,
    f"{split}/beta_kl": beta_kl,
    f"{split}/post_mean_abs_mu": diag.get("post_mean_abs_mu", float("nan")),
    f"{split}/post_mean_mu2": diag.get("post_mean_mu2", float("nan")),
    f"{split}/post_mean_var": diag.get("post_mean_var", float("nan")),
    f"{split}/prior_mean_abs_mu": diag.get("prior_mean_abs_mu", float("nan")),
    f"{split}/prior_mean_mu2": diag.get("prior_mean_mu2", float("nan")),
    f"{split}/prior_mean_var": diag.get("prior_mean_var", float("nan")),
    f"{split}/z_mean": diag.get("z_mean", float("nan")),
    f"{split}/z_std": diag.get("z_std", float("nan")),
    f"{split}/mu_q_minus_mu_p_mse": diag.get("mu_mse", float("nan")),
    f"{split}/var_ratio": diag.get("var_ratio", float("nan")),
    f"{split}/elapsed_time_sec": elapsed,
  }
  wandb_run.log(payload, step=step)


def _wandb_log_artifacts(wandb_run, run_dir: Path) -> None:
  # Artifacts are disabled for motor_controller_stage1; keep run files only.
  return


def _wandb_save_files(wandb_run, run_dir: Path) -> None:
  try:
    import wandb
  except ImportError:
    return

  for filename in [
    "metadata.json",
    "normalization_stats.npz",
    "metrics.csv",
    "model_best.pt",
    "model_last.pt",
  ]:
    path = run_dir / filename
    if path.exists():
      try:
        wandb.save(str(path), base_path=str(run_dir))
      except Exception:
        pass


def _save_checkpoint(path: Path, model: torch.nn.Module, step: int) -> None:
  payload = {
    "step": int(step),
    "state_dict": model.state_dict(),
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(payload, path)
