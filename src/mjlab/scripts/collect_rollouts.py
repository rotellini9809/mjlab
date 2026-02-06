"""Collect rollouts from a trained policy and save dataset shards."""

from __future__ import annotations

import json
import sys
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.motor_controller_stage1.obs_views import build_student_obs
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class CollectRolloutsConfig:
  wandb_run_path: str
  num_envs: int | None = None
  num_episodes: int = 100
  num_steps: int | None = None
  output_dir: str | None = None
  """Required. Recommended: ./data/motor_controller_rollouts/<dataset_root>."""
  shard_size: int = 100_000
  noise_std: float = 0.0
  seed: int | None = None
  device: str | None = None
  save_reward: bool = False
  save_episode_id: bool = True
  save_env_id: bool = True
  """Save env_id per row (recommended for num_envs > 1)."""
  save_step_in_episode: bool = False
  """Optionally save step index within each episode."""
  save_teacher_obs: bool = False
  """Optionally save full teacher observations as obs_teacher."""


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


def run_collect_rollouts(task_id: str, cfg: CollectRolloutsConfig) -> None:
  configure_torch_backends()

  if cfg.output_dir is None:
    print(
      "[ERROR] --output-dir is required. Recommended: "
      "--output-dir ./data/motor_controller_rollouts/<dataset_root>"
    )
    sys.exit(2)

  output_root = Path(cfg.output_dir).expanduser().resolve()
  output_root.mkdir(parents=True, exist_ok=True)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
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

  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  obs = env.get_observations()
  env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)

  if "policy" not in obs:
    raise RuntimeError("Policy observation group not found in observations.")

  motion_term = None
  prev_time_steps = None
  if is_tracking_task:
    maybe_motion_term = env.unwrapped.command_manager.get_term("motion")
    if isinstance(maybe_motion_term, MotionCommand):
      motion_term = maybe_motion_term
      prev_time_steps = motion_term.time_steps.clone()

  policy_obs = obs["policy"]
  obs_manager = env.unwrapped.observation_manager
  obs_meta = {
    "term_order": obs_manager.active_terms.get("policy", []),
    "term_dims": obs_manager.group_obs_term_dim.get("policy", []),
    "act_dim": env.num_actions,
  }
  obs_student, obs_student_meta = build_student_obs(policy_obs, obs_meta)
  teacher_dim = int(obs_student_meta.get("teacher_obs_dim", obs_student.shape[-1]))
  obs_dim = int(obs_student_meta.get("student_obs_dim", obs_student.shape[-1]))
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

  num_envs = env.num_envs
  episode_ids = np.zeros(num_envs, dtype=np.int64)
  step_in_episode = np.zeros(num_envs, dtype=np.int64)

  if num_envs > 1 and not cfg.save_env_id:
    print(
      "[WARN] env_id logging is disabled with num_envs > 1. "
      "Sequence sampling may mix trajectories."
    )

  env_ids = np.arange(num_envs, dtype=np.int64) if cfg.save_env_id else None

  writer = ShardWriter(run_dir, cfg.shard_size)

  total_steps = 0
  total_episodes = 0

  stop_on_steps = cfg.num_steps is not None
  if not stop_on_steps and cfg.num_episodes <= 0:
    raise ValueError("num_episodes must be > 0 when num_steps is not set.")

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

  while True:
    with torch.no_grad():
      a_clean = policy(obs)
    if cfg.noise_std > 0.0:
      noise = torch.randn_like(a_clean) * cfg.noise_std
      actions = a_clean + noise
    else:
      actions = a_clean

    obs_next, rewards, dones, _ = env.step(actions)
    if motion_term is not None and prev_time_steps is not None:
      current_time_steps = motion_term.time_steps
      wrap_mask = current_time_steps < prev_time_steps
    else:
      wrap_mask = torch.zeros_like(dones, dtype=torch.bool)
    done_mask = dones.bool() | wrap_mask

    policy_obs = obs["policy"]
    obs_student, _ = build_student_obs(policy_obs, obs_meta)
    if torch.is_tensor(obs_student):
      obs_student_np = obs_student.detach().cpu().numpy()
    else:
      obs_student_np = np.asarray(obs_student)
    act_np = a_clean.detach().cpu().numpy()
    done_np = done_mask.detach().cpu().numpy().astype(np.bool_)

    meta_json = json.dumps(obs_student_meta)
    meta_json_arr = np.full(
      (obs_student_np.shape[0],), meta_json, dtype=object
    )
    anchors_stripped_arr = np.full(
      (obs_student_np.shape[0],), anchors_stripped, dtype=np.bool_
    )
    features_stripped_arr = np.full(
      (obs_student_np.shape[0],), features_stripped, dtype=np.bool_
    )
    teacher_dim_arr = np.full(
      (obs_student_np.shape[0],), teacher_dim, dtype=np.int64
    )
    student_dim_arr = np.full(
      (obs_student_np.shape[0],), obs_dim, dtype=np.int64
    )

    data: dict[str, np.ndarray] = {
      "obs_student": obs_student_np,
      "obs_student_meta_json": meta_json_arr,
      "obs_student_anchors_stripped": anchors_stripped_arr,
      "obs_student_features_stripped": features_stripped_arr,
      "obs_student_teacher_dim": teacher_dim_arr,
      "obs_student_dim": student_dim_arr,
      "a_clean": act_np,
      "done": done_np,
    }
    if cfg.save_teacher_obs:
      data["obs_teacher"] = policy_obs.detach().cpu().numpy()
    if cfg.save_reward:
      data["reward"] = rewards.detach().cpu().numpy()
    if cfg.save_episode_id:
      data["episode_id"] = episode_ids.copy()
    if env_ids is not None:
      data["env_id"] = env_ids
    if cfg.save_step_in_episode:
      data["step_in_episode"] = step_in_episode.copy()

    writer.append(data)

    total_steps += obs_student_np.shape[0]
    total_episodes += int(done_mask.sum().item())
    if done_mask.any():
      episode_ids[done_np] += 1
    if cfg.save_step_in_episode:
      step_in_episode += 1
      step_in_episode[done_np] = 0

    if motion_term is not None and prev_time_steps is not None:
      wrap_ids = torch.where(wrap_mask & ~dones.bool())[0]
      if wrap_ids.numel() > 0:
        reset_obs, _ = env.unwrapped.reset(env_ids=wrap_ids)
        for key, value in reset_obs.items():
          obs_next[key][wrap_ids] = value[wrap_ids]

    pbar.update(obs_student_np.shape[0])
    pbar.set_postfix(episodes=total_episodes)

    obs = obs_next
    if motion_term is not None and prev_time_steps is not None:
      prev_time_steps = motion_term.time_steps.clone()

    if stop_on_steps and cfg.num_steps is not None:
      if total_steps >= cfg.num_steps:
        break
    elif total_episodes >= cfg.num_episodes:
      break

  pbar.close()
  writer.flush()
  env.close()

  metadata = {
    "run_name": run_name,
    "wandb_run_path": cfg.wandb_run_path,
    "motion_artifact": motion_artifact_path,
    "output_dir": str(run_dir),
    "num_envs": num_envs,
    "num_episodes": cfg.num_episodes,
    "num_steps": cfg.num_steps,
    "shard_count": writer.shard_count,
    "obs_dim": obs_dim,
    "teacher_obs_dim": teacher_dim,
    "act_dim": act_dim,
    "keys": list(data.keys()) if "data" in locals() else [],
    "obs_student_view": obs_student_meta,
  }
  (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

  print("[INFO] Rollout collection complete.")
  print(f"[INFO] Total steps: {total_steps}")
  print(f"[INFO] Total episodes: {total_episodes}")
  print(f"[INFO] obs_dim: {obs_dim}, act_dim: {act_dim}")
  print(f"[INFO] Shards written: {writer.shard_count}")
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
