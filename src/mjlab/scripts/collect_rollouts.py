"""Collect Stage-1 rollouts from a trained tracking policy."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
from tqdm import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.motor_controller_stage1.obs_views import build_student_obs
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

COLLECTOR_VERSION = "stage1_npmp_v2"
STAGE1_CONTROL_DT = 0.03
DONE_REASON_NONE = "none"
DONE_REASON_CLIP_END = "clip_end"
DONE_REASON_TIMEOUT = "timeout"
DONE_REASON_NAN = "nan"
DONE_REASON_FALL = "fall"
DONE_REASON_INVALID_STATE = "invalid_state"


@dataclass(frozen=True)
class CollectRolloutsConfig:
  wandb_run_path: str
  num_envs: int | None = None
  num_episodes: int = 100
  num_steps: int | None = None
  output_dir: str | None = None
  """Required. Recommended: ./data/motor_controller_rollouts/<dataset_root>."""
  shard_size: int = 100_000
  seed: int | None = None
  device: str | None = None
  noise_std: float = 0.0
  """Used when noise_std_levels is empty."""
  noise_std_levels: tuple[float, ...] = ()
  """Optional per-episode noise levels. Overrides noise_std when provided."""
  noise_level_probs: tuple[float, ...] = ()
  """Optional probabilities for noise_std_levels."""
  stage1_chunk_len_hint: int = 32
  stage1_k_future_hint: int = 8
  stage1_start_margin: int = 4


class ShardWriter:
  def __init__(self, output_dir: Path, shard_size: int) -> None:
    self.output_dir = output_dir
    self.shard_size = shard_size
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self._buffer: dict[str, list[np.ndarray]] = {}
    self._buffer_count = 0
    self._shard_idx = 0

  @property
  def shard_count(self) -> int:
    return self._shard_idx

  def append(self, data: dict[str, np.ndarray]) -> None:
    if not self._buffer:
      self._buffer = {k: [] for k in data}
    for key, value in data.items():
      self._buffer[key].append(value)
    self._buffer_count += next(iter(data.values())).shape[0]
    self._flush_if_needed(force=False)

  def flush(self) -> None:
    self._flush_if_needed(force=True)

  def _flush_if_needed(self, force: bool) -> None:
    if self._buffer_count == 0:
      return
    if not force and self._buffer_count < self.shard_size:
      return

    combined = {k: np.concatenate(vs, axis=0) for k, vs in self._buffer.items()}
    total = next(iter(combined.values())).shape[0]
    if total == 0:
      self._buffer = {k: [] for k in self._buffer}
      self._buffer_count = 0
      return

    full_shards = total // self.shard_size
    remainder = total % self.shard_size
    num_to_write = full_shards + (1 if force and remainder > 0 else 0)

    for idx in range(num_to_write):
      start = idx * self.shard_size
      end = min(start + self.shard_size, total)
      if start >= end:
        continue
      shard = {k: v[start:end] for k, v in combined.items()}
      out_path = self.output_dir / f"rollouts_{self._shard_idx:06d}.npz"
      np.savez(out_path, **shard)
      self._shard_idx += 1

    if force or remainder == 0:
      self._buffer = {k: [] for k in self._buffer}
      self._buffer_count = 0
    else:
      start = full_shards * self.shard_size
      self._buffer = {k: [v[start:]] for k, v in combined.items()}
      self._buffer_count = remainder


def _resolve_motion_file(
  env_cfg, cfg: CollectRolloutsConfig, is_tracking_task: bool
) -> tuple[str | None, str | None]:
  if not is_tracking_task:
    return None, None
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  import wandb

  api = wandb.Api()
  wandb_run = api.run(str(cfg.wandb_run_path))
  art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
  if art is None:
    raise RuntimeError("No motion artifact found in the run.")
  motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")
  run_name = getattr(wandb_run, "name", None)
  art_path = None
  try:
    if ":" in art.name:
      art_path = f"{art.entity}/{art.project}/{art.name}"
    else:
      art_path = f"{art.entity}/{art.project}/{art.name}:{art.version}"
  except Exception:
    art_path = art.name
  return run_name, art_path


def _slugify_name(name: str) -> str:
  name = name.strip().replace(" ", "_")
  name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
  name = re.sub(r"_+", "_", name).strip("_")
  return name or "wandb_run"


def _resolve_checkpoint_path(
  agent_cfg, cfg: CollectRolloutsConfig
) -> tuple[Path, bool]:
  log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
  resume_path, was_cached = get_wandb_checkpoint_path(
    log_root_path, Path(cfg.wandb_run_path)
  )
  run_id = resume_path.parent.name
  checkpoint_name = resume_path.name
  cached_str = "cached" if was_cached else "downloaded"
  print(
    f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
  )
  return resume_path, was_cached


def _removed_dim_from_meta(obs_meta: dict[str, object]) -> int:
  removed_present_raw = obs_meta.get("removed_terms_present", [])
  if not isinstance(removed_present_raw, list):
    return 0
  removed_present = {str(term) for term in removed_present_raw}
  if not removed_present:
    return 0

  slice_map = obs_meta.get("slice_map")
  if not isinstance(slice_map, list):
    return 0

  removed_dim = 0
  for entry in slice_map:
    if not isinstance(entry, dict):
      continue
    name = str(entry.get("name", ""))
    if name not in removed_present:
      continue
    size = entry.get("size")
    if size is not None:
      removed_dim += int(size)
      continue
    start = entry.get("start")
    end = entry.get("end")
    if start is not None and end is not None:
      removed_dim += int(end) - int(start)
  return removed_dim


def _resolve_noise_levels(cfg: CollectRolloutsConfig) -> tuple[np.ndarray, np.ndarray]:
  if cfg.noise_std_levels:
    levels = np.asarray(cfg.noise_std_levels, dtype=np.float32)
  else:
    levels = np.asarray([cfg.noise_std], dtype=np.float32)

  if np.any(levels < 0.0):
    raise ValueError(f"Noise std must be non-negative. Got: {levels.tolist()}")

  if cfg.noise_level_probs:
    probs = np.asarray(cfg.noise_level_probs, dtype=np.float64)
    if probs.shape[0] != levels.shape[0]:
      raise ValueError(
        "noise_level_probs length must match noise_std_levels length. "
        f"Got {probs.shape[0]} vs {levels.shape[0]}"
      )
    if np.any(probs < 0.0):
      raise ValueError("noise_level_probs must be non-negative.")
    total = float(probs.sum())
    if total <= 0.0:
      raise ValueError("noise_level_probs must sum to > 0.")
    probs = probs / total
  else:
    probs = np.full((levels.shape[0],), 1.0 / levels.shape[0], dtype=np.float64)

  return levels, probs


def _sample_episode_noise_levels(
  rng: np.random.Generator,
  num_envs: int,
  noise_levels: np.ndarray,
  noise_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  if noise_levels.shape[0] == 1:
    level_ids = np.zeros((num_envs,), dtype=np.int64)
  else:
    level_ids = rng.choice(
      noise_levels.shape[0], size=num_envs, p=noise_probs, replace=True
    ).astype(np.int64)
  return level_ids, noise_levels[level_ids]


def _sample_episode_starts(
  rng: np.random.Generator,
  num_envs: int,
  clip_ids: np.ndarray,
  clip_start_steps: np.ndarray,
  clip_len_steps: np.ndarray,
  min_remaining: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  clip_count = int(clip_ids.shape[0])
  clip_indices = rng.integers(0, clip_count, size=num_envs, dtype=np.int64)
  sampled_clip_ids = clip_ids[clip_indices]
  sampled_clip_lens = clip_len_steps[clip_indices]
  sampled_clip_starts = clip_start_steps[clip_indices]

  start_phase = np.zeros((num_envs,), dtype=np.int64)
  for i in range(num_envs):
    clip_len = int(sampled_clip_lens[i])
    if clip_len >= min_remaining:
      max_start = clip_len - min_remaining
      start_phase[i] = int(rng.integers(0, max_start + 1, dtype=np.int64))
    else:
      start_phase[i] = 0

  start_steps = sampled_clip_starts + start_phase
  return clip_indices, sampled_clip_ids, sampled_clip_lens, start_steps


def _reset_envs_for_stage1(
  env: RslRlVecEnvWrapper,
  motion_term: MotionCommand,
  env_ids: torch.Tensor,
  rng: np.random.Generator,
  clip_ids: np.ndarray,
  clip_start_steps: np.ndarray,
  clip_len_steps: np.ndarray,
  min_remaining: int,
  noise_levels: np.ndarray,
  noise_probs: np.ndarray,
) -> tuple[TensorDict, dict[str, np.ndarray]]:
  if env_ids.numel() == 0:
    return env.get_observations(), {}

  env.unwrapped.reset(env_ids=env_ids)
  n = int(env_ids.numel())
  clip_indices, sampled_clip_ids, sampled_clip_lens, start_steps = _sample_episode_starts(
    rng=rng,
    num_envs=n,
    clip_ids=clip_ids,
    clip_start_steps=clip_start_steps,
    clip_len_steps=clip_len_steps,
    min_remaining=min_remaining,
  )

  start_steps_t = torch.from_numpy(start_steps).to(
    device=env.unwrapped.device, dtype=torch.long
  )
  motion_term.set_time_steps(env_ids, start_steps_t, apply_randomization=False)

  env.unwrapped.scene.write_data_to_sim()
  env.unwrapped.sim.forward()
  env.unwrapped.sim.sense()
  env.unwrapped.observation_manager.reset(env_ids=env_ids)
  obs_dict = env.unwrapped.observation_manager.compute(update_history=True)
  obs = TensorDict(obs_dict, batch_size=[env.num_envs])

  noise_level_id, noise_std = _sample_episode_noise_levels(
    rng=rng,
    num_envs=n,
    noise_levels=noise_levels,
    noise_probs=noise_probs,
  )

  env_ids_np = env_ids.detach().cpu().numpy().astype(np.int64, copy=False)
  info = {
    "env_id": env_ids_np,
    "clip_index": clip_indices.astype(np.int64, copy=False),
    "clip_id": sampled_clip_ids.astype(np.int64, copy=False),
    "clip_len_steps": sampled_clip_lens.astype(np.int64, copy=False),
    "start_step": start_steps.astype(np.int64, copy=False),
    "start_phase": (start_steps - clip_start_steps[clip_indices]).astype(
      np.int64, copy=False
    ),
    "noise_level_id": noise_level_id.astype(np.int64, copy=False),
    "noise_std": noise_std.astype(np.float32, copy=False),
  }
  return obs, info


def _classify_reason_from_term(term_name: str, is_timeout: bool) -> str:
  if is_timeout:
    return DONE_REASON_TIMEOUT
  name = term_name.lower()
  if "nan" in name:
    return DONE_REASON_NAN
  if any(token in name for token in ("anchor", "body", "fall", "collision", "ori")):
    return DONE_REASON_FALL
  return DONE_REASON_INVALID_STATE


def _resolve_done_flags_and_reasons(
  env: RslRlVecEnvWrapper,
  terminated_t: torch.Tensor,
  truncated_t: torch.Tensor,
  clip_end_t: torch.Tensor,
  nan_t: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  num_envs = env.num_envs
  term_mgr = env.unwrapped.termination_manager

  terminated_np = terminated_t.detach().cpu().numpy().astype(np.bool_, copy=False)
  truncated_np = truncated_t.detach().cpu().numpy().astype(np.bool_, copy=False)
  clip_end_np = clip_end_t.detach().cpu().numpy().astype(np.bool_, copy=False)
  nan_np = nan_t.detach().cpu().numpy().astype(np.bool_, copy=False)

  term_reason = np.full((num_envs,), "", dtype=object)
  trunc_reason = np.full((num_envs,), "", dtype=object)

  for name in term_mgr.active_terms:
    mask = term_mgr.get_term(name).detach().cpu().numpy().astype(np.bool_, copy=False)
    is_timeout = bool(term_mgr.get_term_cfg(name).time_out)
    if is_timeout:
      write_mask = mask & (trunc_reason == "")
      trunc_reason[write_mask] = name
    else:
      write_mask = mask & (term_reason == "")
      term_reason[write_mask] = name

  done_reason = np.full((num_envs,), DONE_REASON_NONE, dtype="<U32")

  if np.any(nan_np):
    terminated_np[nan_np] = True
    truncated_np[nan_np] = False
    done_reason[nan_np] = DONE_REASON_NAN

  terminated_only = terminated_np & ~nan_np
  if np.any(terminated_only):
    for idx in np.where(terminated_only)[0]:
      name = str(term_reason[idx]) if term_reason[idx] else ""
      if name:
        done_reason[idx] = _classify_reason_from_term(name, is_timeout=False)
      else:
        done_reason[idx] = DONE_REASON_INVALID_STATE

  clip_done = clip_end_np & ~terminated_np & ~truncated_np
  if np.any(clip_done):
    truncated_np[clip_done] = True
    done_reason[clip_done] = DONE_REASON_CLIP_END

  truncated_only = truncated_np & ~terminated_np & ~clip_done & ~nan_np
  if np.any(truncated_only):
    for idx in np.where(truncated_only)[0]:
      name = str(trunc_reason[idx]) if trunc_reason[idx] else ""
      if name:
        done_reason[idx] = _classify_reason_from_term(name, is_timeout=True)
      else:
        done_reason[idx] = DONE_REASON_TIMEOUT

  both_mask = terminated_np & truncated_np
  if np.any(both_mask):
    truncated_np[both_mask] = False

  done_np = terminated_np | truncated_np
  return terminated_np, truncated_np, done_np, done_reason


def _counter_from_values(values: np.ndarray) -> Counter[int]:
  out: Counter[int] = Counter()
  if values.size == 0:
    return out
  uniq, counts = np.unique(values.astype(np.int64, copy=False), return_counts=True)
  for key, count in zip(uniq.tolist(), counts.tolist(), strict=False):
    out[int(key)] += int(count)
  return out


def _sorted_counter_items(counter: Counter[int] | Counter[str]):
  if not counter:
    return []
  return sorted(counter.items(), key=lambda kv: str(kv[0]))


def run_collect_rollouts(task_id: str, cfg: CollectRolloutsConfig) -> None:
  configure_torch_backends()

  if cfg.output_dir is None:
    print(
      "[ERROR] --output-dir is required. Recommended: "
      "--output-dir ./data/motor_controller_rollouts/<dataset_root>"
    )
    sys.exit(2)

  if cfg.num_episodes <= 0 and cfg.num_steps is None:
    raise ValueError("num_episodes must be > 0 when num_steps is not set.")

  if cfg.stage1_chunk_len_hint <= 0:
    raise ValueError("stage1_chunk_len_hint must be > 0.")
  if cfg.stage1_k_future_hint < 0:
    raise ValueError("stage1_k_future_hint must be >= 0.")
  if cfg.stage1_start_margin < 0:
    raise ValueError("stage1_start_margin must be >= 0.")

  output_root = Path(cfg.output_dir).expanduser().resolve()
  output_root.mkdir(parents=True, exist_ok=True)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  rng = np.random.default_rng(cfg.seed)

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )
  if not is_tracking_task:
    raise RuntimeError(
      "Stage-1 rollout collector requires tracking task with `commands.motion`."
    )

  run_name, motion_artifact_path = _resolve_motion_file(env_cfg, cfg, is_tracking_task)
  if run_name is None:
    run_name = "wandb_run"
  run_dir = output_root / _slugify_name(run_name)
  run_dir.mkdir(parents=True, exist_ok=True)
  print(f"[INFO] Output dir: {run_dir}")
  resume_path, _ = _resolve_checkpoint_path(agent_cfg, cfg)

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.seed is not None:
    env_cfg.seed = cfg.seed

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  control_dt = float(env.unwrapped.step_dt)
  if not np.isclose(control_dt, STAGE1_CONTROL_DT, atol=1e-6):
    raise RuntimeError(
      f"Stage-1 collector expects control_dt={STAGE1_CONTROL_DT}, got {control_dt}."
    )

  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  maybe_motion_term = env.unwrapped.command_manager.get_term("motion")
  if not isinstance(maybe_motion_term, MotionCommand):
    raise RuntimeError("Tracking task missing MotionCommand term.")
  motion_term = maybe_motion_term

  motion = motion_term.motion
  clip_ids = motion.clip_ids.detach().cpu().numpy().astype(np.int64, copy=False)
  clip_start_steps = motion.clip_start_steps.detach().cpu().numpy().astype(
    np.int64, copy=False
  )
  clip_len_steps = motion.clip_len_steps.detach().cpu().numpy().astype(
    np.int64, copy=False
  )
  clip_min_len = int(clip_len_steps.min())
  clip_max_len = int(clip_len_steps.max())
  print(
    "[INFO] Motion clips:"
    f" count={int(motion.clip_count)}, min_len={clip_min_len}, max_len={clip_max_len}"
  )

  min_remaining = (
    cfg.stage1_chunk_len_hint + cfg.stage1_k_future_hint + cfg.stage1_start_margin
  )
  print(
    "[INFO] Stage-1 valid-start rule:"
    f" min_remaining={min_remaining} "
    f"(chunk={cfg.stage1_chunk_len_hint}, k_future={cfg.stage1_k_future_hint}, "
    f"margin={cfg.stage1_start_margin})"
  )

  noise_levels, noise_probs = _resolve_noise_levels(cfg)
  print(
    f"[INFO] Noise levels: {noise_levels.tolist()} with probs {noise_probs.tolist()}"
  )

  num_envs = env.num_envs
  env_ids_np = np.arange(num_envs, dtype=np.int64)
  all_env_ids_t = torch.arange(num_envs, device=env.unwrapped.device, dtype=torch.long)

  obs, init_info = _reset_envs_for_stage1(
    env=env,
    motion_term=motion_term,
    env_ids=all_env_ids_t,
    rng=rng,
    clip_ids=clip_ids,
    clip_start_steps=clip_start_steps,
    clip_len_steps=clip_len_steps,
    min_remaining=min_remaining,
    noise_levels=noise_levels,
    noise_probs=noise_probs,
  )

  if "policy" not in obs:
    raise RuntimeError("Policy observation group not found in observations.")

  obs_manager = env.unwrapped.observation_manager
  obs_meta = {
    "term_order": obs_manager.active_terms.get("policy", []),
    "term_dims": obs_manager.group_obs_term_dim.get("policy", []),
    "act_dim": env.num_actions,
  }
  obs_student_init, obs_student_meta = build_student_obs(obs["policy"], obs_meta)
  teacher_dim = int(
    obs_student_meta.get("teacher_obs_dim", obs_student_init.shape[-1])  # type: ignore[arg-type]
  )
  obs_dim = int(
    obs_student_meta.get("student_obs_dim", obs_student_init.shape[-1])  # type: ignore[arg-type]
  )
  anchors_stripped = bool(obs_student_meta.get("anchors_stripped", False))
  features_stripped = bool(obs_student_meta.get("features_stripped", anchors_stripped))
  removed_terms_present_raw = obs_student_meta.get("removed_terms_present", [])
  removed_terms_present = (
    [str(term) for term in removed_terms_present_raw]
    if isinstance(removed_terms_present_raw, list)
    else []
  )
  kept_terms_raw = obs_student_meta.get("kept_terms", [])
  kept_terms = (
    [str(term) for term in kept_terms_raw] if isinstance(kept_terms_raw, list) else []
  )
  removed_dim = _removed_dim_from_meta(obs_student_meta)
  act_dim = env.num_actions
  anchors_present_in_student = any(
    term in kept_terms for term in ("motion_anchor_pos_b", "motion_anchor_ori_b")
  )

  print(
    "[INFO] Student obs dims: "
    f"teacher_dim={teacher_dim}, student_dim={obs_dim}, "
    f"anchors_stripped={anchors_stripped}, features_stripped={features_stripped}"
  )
  print(
    "[INFO] Student obs terms: "
    f"removed={removed_terms_present if removed_terms_present else 'none'}, "
    f"anchors_present_in_student={anchors_present_in_student}"
  )
  if features_stripped:
    if removed_dim > 0:
      assert obs_dim == teacher_dim - removed_dim, (
        "features_stripped=True but student_dim mismatch: "
        f"teacher_dim={teacher_dim}, removed_dim={removed_dim}, student_dim={obs_dim}"
      )
    assert not anchors_present_in_student, (
      "Anchor terms are still present in student observation view: "
      f"kept_terms={kept_terms}"
    )

  writer = ShardWriter(run_dir, cfg.shard_size)

  episode_ids = np.zeros((num_envs,), dtype=np.int64)
  step_in_episode = np.zeros((num_envs,), dtype=np.int64)
  episode_len_running = np.zeros((num_envs,), dtype=np.int64)
  episode_noise_level_id = np.zeros((num_envs,), dtype=np.int64)
  episode_noise_std = np.zeros((num_envs,), dtype=np.float32)

  init_env_ids = init_info["env_id"]
  episode_noise_level_id[init_env_ids] = init_info["noise_level_id"]
  episode_noise_std[init_env_ids] = init_info["noise_std"]

  total_steps = 0
  total_episodes = 0
  terminated_count = 0
  truncated_count = 0
  episode_lengths_steps: list[int] = []
  clip_hist_steps: Counter[int] = Counter()
  clip_hist_episodes: Counter[int] = Counter()
  done_reason_hist: Counter[str] = Counter()
  noise_norm_sum = 0.0
  noise_norm_count = 0
  noise_norm_min: float | None = None
  noise_norm_max: float | None = None

  clip_hist_episodes.update(_counter_from_values(init_info["clip_id"]))

  stop_on_steps = cfg.num_steps is not None

  print(
    "[INFO] Collecting rollouts:"
    f" task={task_id}, num_envs={num_envs}, device={device}"
  )
  pbar = tqdm(
    total=cfg.num_steps,
    unit="steps",
    dynamic_ncols=True,
    desc="Collecting",
  )

  data_keys: list[str] = []
  while True:
    time_steps_t = motion_term.time_steps.clone()
    clip_idx_t = motion.step_clip_index[time_steps_t]
    clip_id_t = motion.clip_ids[clip_idx_t]
    clip_len_t = motion.clip_len_steps[clip_idx_t]
    phase_idx_t = motion.step_phase_idx[time_steps_t]
    steps_to_clip_end_t = motion.clip_end_steps[clip_idx_t] - time_steps_t - 1
    phase_norm_t = phase_idx_t.float() / torch.clamp(clip_len_t - 1, min=1).float()

    with torch.no_grad():
      a_clean = policy(obs)

    if np.any(episode_noise_std > 0.0):
      noise_std_t = torch.from_numpy(episode_noise_std).to(a_clean.device).unsqueeze(-1)
      action_noise = torch.randn_like(a_clean) * noise_std_t
    else:
      action_noise = torch.zeros_like(a_clean)

    a_exec = a_clean + action_noise
    nan_t = ~(
      torch.isfinite(a_clean).all(dim=1)
      & torch.isfinite(a_exec).all(dim=1)
      & torch.isfinite(action_noise).all(dim=1)
    )
    if nan_t.any():
      a_exec[nan_t] = 0.0

    obs_next, _, _, _ = env.step(a_exec)

    terminated_t = env.unwrapped.termination_manager.terminated.clone()
    truncated_t = env.unwrapped.termination_manager.time_outs.clone()
    clip_end_t = (steps_to_clip_end_t == 0) & ~terminated_t & ~truncated_t

    terminated_np, truncated_np, done_np, done_reason_np = _resolve_done_flags_and_reasons(
      env=env,
      terminated_t=terminated_t,
      truncated_t=truncated_t,
      clip_end_t=clip_end_t,
      nan_t=nan_t,
    )

    policy_obs = obs["policy"]
    obs_student, _ = build_student_obs(policy_obs, obs_meta)
    if torch.is_tensor(obs_student):
      obs_student_np = obs_student.detach().cpu().numpy()
    else:
      obs_student_np = np.asarray(obs_student, dtype=np.float32)

    a_clean_np = a_clean.detach().cpu().numpy().astype(np.float32, copy=False)
    a_exec_np = a_exec.detach().cpu().numpy().astype(np.float32, copy=False)
    noise_norm_np = (
      torch.linalg.vector_norm(action_noise, dim=1)
      .detach()
      .cpu()
      .numpy()
      .astype(np.float32, copy=False)
    )

    clip_id_np = clip_id_t.detach().cpu().numpy().astype(np.int64, copy=False)
    clip_len_np = clip_len_t.detach().cpu().numpy().astype(np.int64, copy=False)
    phase_idx_np = phase_idx_t.detach().cpu().numpy().astype(np.int64, copy=False)
    phase_norm_np = phase_norm_t.detach().cpu().numpy().astype(np.float32, copy=False)
    steps_to_clip_end_np = (
      steps_to_clip_end_t.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    future_valid_len_hint = np.minimum(
      steps_to_clip_end_np, cfg.stage1_k_future_hint
    ).astype(np.int64, copy=False)

    data = {
      "obs_student": obs_student_np.astype(np.float32, copy=False),
      "a_clean": a_clean_np,
      "a_exec": a_exec_np,
      "episode_id": episode_ids.copy(),
      "step_in_episode": step_in_episode.copy(),
      "env_id": env_ids_np,
      "clip_id": clip_id_np,
      "clip_len_steps": clip_len_np,
      "phase_idx": phase_idx_np,
      "phase_norm": phase_norm_np,
      "steps_to_clip_end": steps_to_clip_end_np,
      "future_valid_len_hint": future_valid_len_hint,
      "terminated": terminated_np.astype(np.bool_, copy=False),
      "truncated": truncated_np.astype(np.bool_, copy=False),
      "done_reason": done_reason_np,
      "noise_level_id": episode_noise_level_id.copy(),
      "noise_norm": noise_norm_np,
    }
    if not data_keys:
      data_keys = list(data.keys())
    writer.append(data)

    total_steps += obs_student_np.shape[0]
    done_count = int(done_np.sum())
    total_episodes += done_count

    episode_len_running += 1
    if done_count > 0:
      done_rows = np.where(done_np)[0]
      episode_lengths_steps.extend(episode_len_running[done_rows].tolist())
      episode_len_running[done_rows] = 0
      episode_ids[done_rows] += 1

    step_in_episode += 1
    step_in_episode[done_np] = 0

    terminated_count += int(terminated_np.sum())
    truncated_count += int(truncated_np.sum())
    clip_hist_steps.update(_counter_from_values(clip_id_np))
    if done_count > 0:
      for reason in done_reason_np[done_np]:
        done_reason_hist[str(reason)] += 1

    noise_norm_sum += float(noise_norm_np.sum())
    noise_norm_count += int(noise_norm_np.shape[0])
    local_min = float(noise_norm_np.min())
    local_max = float(noise_norm_np.max())
    noise_norm_min = local_min if noise_norm_min is None else min(noise_norm_min, local_min)
    noise_norm_max = local_max if noise_norm_max is None else max(noise_norm_max, local_max)

    if done_count > 0:
      done_env_ids_np = np.where(done_np)[0].astype(np.int64, copy=False)
      done_env_ids_t = torch.from_numpy(done_env_ids_np).to(
        device=env.unwrapped.device, dtype=torch.long
      )
      reset_obs, reset_info = _reset_envs_for_stage1(
        env=env,
        motion_term=motion_term,
        env_ids=done_env_ids_t,
        rng=rng,
        clip_ids=clip_ids,
        clip_start_steps=clip_start_steps,
        clip_len_steps=clip_len_steps,
        min_remaining=min_remaining,
        noise_levels=noise_levels,
        noise_probs=noise_probs,
      )
      for key in obs_next.keys():
        obs_next[key][done_env_ids_t] = reset_obs[key][done_env_ids_t]
      reset_env_ids = reset_info["env_id"]
      episode_noise_level_id[reset_env_ids] = reset_info["noise_level_id"]
      episode_noise_std[reset_env_ids] = reset_info["noise_std"]
      clip_hist_episodes.update(_counter_from_values(reset_info["clip_id"]))

    pbar.update(obs_student_np.shape[0])
    pbar.set_postfix(episodes=total_episodes)
    obs = obs_next

    if stop_on_steps and cfg.num_steps is not None:
      if total_steps >= cfg.num_steps:
        break
    elif total_episodes >= cfg.num_episodes:
      break

  pbar.close()
  writer.flush()
  env.close()

  episode_lengths_np = (
    np.asarray(episode_lengths_steps, dtype=np.int64)
    if episode_lengths_steps
    else np.zeros((0,), dtype=np.int64)
  )
  if episode_lengths_np.size > 0:
    len_min = int(episode_lengths_np.min())
    len_max = int(episode_lengths_np.max())
    len_mean = float(episode_lengths_np.mean())
  else:
    len_min = 0
    len_max = 0
    len_mean = 0.0
  len_mean_s = len_mean * control_dt
  len_min_s = len_min * control_dt
  len_max_s = len_max * control_dt

  noise_mean = noise_norm_sum / max(noise_norm_count, 1)
  noise_stats = {
    "mean": noise_mean,
    "min": 0.0 if noise_norm_min is None else noise_norm_min,
    "max": 0.0 if noise_norm_max is None else noise_norm_max,
  }

  metadata = {
    "collector_version": COLLECTOR_VERSION,
    "run_name": run_name,
    "wandb_run_path": cfg.wandb_run_path,
    "motion_artifact": motion_artifact_path,
    "output_dir": str(run_dir),
    "control_dt": STAGE1_CONTROL_DT,
    "num_envs": num_envs,
    "num_episodes_requested": cfg.num_episodes,
    "num_steps_requested": cfg.num_steps,
    "num_episodes_collected": total_episodes,
    "num_steps_collected": total_steps,
    "shard_count": writer.shard_count,
    "obs_dim": obs_dim,
    "teacher_obs_dim": teacher_dim,
    "act_dim": act_dim,
    "keys": data_keys,
    "obs_student_view": obs_student_meta,
    "stage1_chunk_len_hint": cfg.stage1_chunk_len_hint,
    "stage1_k_future_hint": cfg.stage1_k_future_hint,
    "stage1_start_margin": cfg.stage1_start_margin,
    "stage1_min_remaining": min_remaining,
    "noise_std_default": cfg.noise_std,
    "noise_std_levels": noise_levels.tolist(),
    "noise_level_probs": noise_probs.tolist(),
    "done_reason_values": sorted({str(k) for k in done_reason_hist.keys()}),
    "clip_count": int(motion.clip_count),
    "clip_len_steps_min": clip_min_len,
    "clip_len_steps_max": clip_max_len,
    "summary": {
      "terminated_count": terminated_count,
      "truncated_count": truncated_count,
      "episode_len_steps_mean": len_mean,
      "episode_len_steps_min": len_min,
      "episode_len_steps_max": len_max,
      "episode_len_seconds_mean": len_mean_s,
      "episode_len_seconds_min": len_min_s,
      "episode_len_seconds_max": len_max_s,
      "clip_hist_steps": {str(k): int(v) for k, v in _sorted_counter_items(clip_hist_steps)},
      "clip_hist_episodes": {
        str(k): int(v) for k, v in _sorted_counter_items(clip_hist_episodes)
      },
      "done_reason_hist": {
        str(k): int(v) for k, v in _sorted_counter_items(done_reason_hist)
      },
      "noise_norm": noise_stats,
    },
  }
  (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

  print("[INFO] Rollout collection complete.")
  print(f"[INFO] Episodes collected: {total_episodes}")
  print(f"[INFO] Steps collected: {total_steps}")
  print(f"[INFO] Shards written: {writer.shard_count}")
  print(
    "[INFO] Episode length steps: "
    f"mean={len_mean:.2f}, min={len_min}, max={len_max}"
  )
  print(
    "[INFO] Episode length seconds: "
    f"mean={len_mean_s:.2f}, min={len_min_s:.2f}, max={len_max_s:.2f}"
  )
  print(
    "[INFO] Done flags: "
    f"terminated={terminated_count}, truncated={truncated_count}"
  )
  if clip_hist_episodes:
    print(f"[INFO] Clip usage histogram (episodes): {dict(_sorted_counter_items(clip_hist_episodes))}")
  if clip_hist_steps:
    print(f"[INFO] Clip usage histogram (steps): {dict(_sorted_counter_items(clip_hist_steps))}")
  if done_reason_hist:
    print(f"[INFO] Done reason histogram: {dict(_sorted_counter_items(done_reason_hist))}")
  print(
    "[INFO] Noise stats (noise_norm): "
    f"mean={noise_stats['mean']:.6f}, min={noise_stats['min']:.6f}, max={noise_stats['max']:.6f}"
  )
  print(f"[INFO] Output dir: {run_dir}")


def main():
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    CollectRolloutsConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )

  run_collect_rollouts(chosen_task, args)


if __name__ == "__main__":
  main()
