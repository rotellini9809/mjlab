from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

_REQUIRED_KEYS = ("a_clean", "done")
_OBS_KEYS = ("obs_student", "obs")


@dataclass(frozen=True)
class RolloutShard:
  path: Path
  run_name: str
  obs: np.ndarray
  a_clean: np.ndarray
  done: np.ndarray
  episode_id: np.ndarray
  env_id: np.ndarray | None
  extras: dict[str, np.ndarray]
  keys: tuple[str, ...]


@dataclass(frozen=True)
class RolloutDatasetStats:
  num_runs: int
  num_shards: int
  total_steps: int
  obs_dim: int
  act_dim: int
  keys_found: tuple[str, ...]
  run_names: tuple[str, ...]
  sample_shapes: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class SampleIndex:
  shard_idx: int
  env_id: int | None
  env_pos: int


@dataclass(frozen=True)
class EnvSequence:
  env_id: int | None
  row_indices: np.ndarray
  done: np.ndarray
  episode_id: np.ndarray


class RolloutDataset:
  """Sequence-ready rollout dataset loader for Stage-1 motor controller training."""

  def __init__(
    self,
    data_dir: Path | None,
    data_root: Path | None,
    k_future: int,
    sample_mode: Literal["step", "chunk"],
    chunk_len: int,
    run_allowlist: set[str] | None = None,
    shard_allowlist: set[Path] | None = None,
    episode_allowlist: set[tuple[str, int | None, int]] | None = None,
    shards: list[RolloutShard] | None = None,
    log_prefix: str | None = None,
    log_env_id_summary: bool = True,
  ) -> None:
    if shards is None and (data_dir is None) == (data_root is None):
      raise ValueError("Provide exactly one of data_dir or data_root.")

    self.data_dir = data_dir
    self.data_root = data_root
    self.k_future = k_future
    self.sample_mode = sample_mode
    self.chunk_len = chunk_len
    self._run_allowlist = run_allowlist
    self._shard_allowlist = shard_allowlist
    self._episode_allowlist = episode_allowlist
    self._log_prefix = log_prefix
    self._log_env_id_summary = log_env_id_summary

    if shards is None:
      runs = self._resolve_runs(data_dir, data_root)
      self._shards = self._load_shards(runs)
    else:
      self._shards = shards
    self._env_id_missing = sum(1 for shard in self._shards if shard.env_id is None)
    if self._log_env_id_summary:
      prefix = f"[{self._log_prefix}]" if self._log_prefix else ""
      if self._env_id_missing > 0:
        print(
          f"[WARN]{prefix} {self._env_id_missing}/{len(self._shards)} shards missing "
          "env_id. Sequence sampling may mix trajectories for multi-env rollouts."
        )
      else:
        print(
          f"[INFO]{prefix} All shards include env_id. Sequence sampling is safe for "
          "vectorized rollouts."
        )
    self._extra_keys = self._resolve_extra_keys()
    self._env_sequences = self._build_env_sequences()
    self._step_indices, self._chunk_indices = self._build_indices()
    self._stats = self._compute_stats()

  @property
  def stats(self) -> RolloutDatasetStats:
    return self._stats

  @property
  def shards(self) -> list[RolloutShard]:
    return self._shards

  @property
  def step_count(self) -> int:
    return len(self._step_indices)

  @property
  def chunk_count(self) -> int:
    return len(self._chunk_indices)

  def sample_batch(
    self, batch_size: int, rng: np.random.Generator
  ) -> dict[str, np.ndarray]:
    if self.sample_mode == "step":
      return self._sample_step(batch_size, rng)
    if self.sample_mode == "chunk":
      return self._sample_chunk(batch_size, rng)
    raise ValueError(f"Unsupported sample_mode: {self.sample_mode}")

  def _resolve_runs(
    self, data_dir: Path | None, data_root: Path | None
  ) -> list[tuple[str, Path]]:
    if data_dir is not None:
      if self._run_allowlist is not None and data_dir.name not in self._run_allowlist:
        raise ValueError(
          f"data_dir '{data_dir.name}' not in run_allowlist {self._run_allowlist}"
        )
      return [(data_dir.name, data_dir)]

    assert data_root is not None
    run_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if self._run_allowlist is not None:
      run_dirs = [p for p in run_dirs if p.name in self._run_allowlist]
    if not run_dirs:
      raise FileNotFoundError(f"No run subfolders found under {data_root}")
    return [(run_dir.name, run_dir) for run_dir in run_dirs]

  def _load_shards(self, runs: list[tuple[str, Path]]) -> list[RolloutShard]:
    shards: list[RolloutShard] = []
    obs_dim = None
    act_dim = None

    for run_name, run_dir in runs:
      shard_paths = sorted(p for p in run_dir.rglob("*.npz") if p.is_file())
      if self._shard_allowlist is not None:
        shard_paths = [
          p for p in shard_paths if p.resolve() in self._shard_allowlist
        ]
      if not shard_paths:
        continue

      for path in shard_paths:
        shard = self._load_single_shard(path, run_name)
        if obs_dim is None:
          obs_dim = shard.obs.shape[1]
          act_dim = shard.a_clean.shape[1]
        elif shard.obs.shape[1] != obs_dim or shard.a_clean.shape[1] != act_dim:
          raise ValueError(
            f"Shard {path} has mismatched dims: obs_dim={shard.obs.shape[1]}, "
            f"act_dim={shard.a_clean.shape[1]} (expected obs_dim={obs_dim}, act_dim={act_dim})"
          )
        shards.append(shard)

    if not shards:
      root = data_root = self.data_root or self.data_dir
      raise FileNotFoundError(f"No .npz shards found under {root}")

    return shards

  def _load_single_shard(self, path: Path, run_name: str) -> RolloutShard:
    with np.load(path) as data:
      keys = tuple(sorted(data.files))
      missing = [k for k in _REQUIRED_KEYS if k not in data.files]
      if missing:
        raise ValueError(f"Shard {path} missing required keys: {missing}")

      obs_key = "obs_student" if "obs_student" in data.files else None
      if obs_key is None and "obs" in data.files:
        obs_key = "obs"
      if obs_key is None:
        raise ValueError(
          f"Shard {path} missing observation key: expected one of {_OBS_KEYS}"
        )

      obs = np.asarray(data[obs_key], dtype=np.float32)
      a_clean = np.asarray(data["a_clean"], dtype=np.float32)
      done = np.asarray(data["done"], dtype=np.bool_)

      extras = {}
      ignore_keys = {
        obs_key,
        "obs_teacher",
        "obs_student_meta_json",
        "obs_student_anchors_stripped",
        "obs_student_features_stripped",
        "obs_student_teacher_dim",
        "obs_student_dim",
      }
      if obs_key != "obs":
        ignore_keys.add("obs")
      for key in data.files:
        if key in _REQUIRED_KEYS or key in ignore_keys:
          continue
        value = np.asarray(data[key])
        if value.ndim > 0 and value.shape[0] == obs.shape[0]:
          extras[key] = value

    if obs.ndim != 2:
      raise ValueError(f"Shard {path} obs must be [T, obs_dim], got {obs.shape}")
    if a_clean.ndim != 2:
      raise ValueError(
        f"Shard {path} a_clean must be [T, act_dim], got {a_clean.shape}"
      )
    if done.ndim != 1:
      raise ValueError(f"Shard {path} done must be [T], got {done.shape}")
    if obs.shape[0] != a_clean.shape[0] or obs.shape[0] != done.shape[0]:
      raise ValueError(
        f"Shard {path} length mismatch: obs={obs.shape[0]}, "
        f"a_clean={a_clean.shape[0]}, done={done.shape[0]}"
      )

    env_id = None
    if "env_id" in extras:
      env_id = np.asarray(extras.pop("env_id")).astype(np.int64, copy=False)
      if env_id.shape[0] != obs.shape[0]:
        raise ValueError(f"Shard {path} env_id must be [T], got {env_id.shape}")

    if "episode_id" in extras:
      episode_id = np.asarray(extras.pop("episode_id")).astype(np.int64, copy=False)
      if episode_id.shape[0] != obs.shape[0]:
        raise ValueError(
          f"Shard {path} episode_id must be [T], got {episode_id.shape}"
        )
    else:
      episode_id = _compute_episode_id(done, env_id)

    return RolloutShard(
      path=path,
      run_name=run_name,
      obs=obs,
      a_clean=a_clean,
      done=done,
      episode_id=episode_id,
      env_id=env_id,
      extras=extras,
      keys=keys,
    )

  def _resolve_extra_keys(self) -> tuple[str, ...]:
    if not self._shards:
      return ()
    common = set(self._shards[0].extras.keys())
    for shard in self._shards[1:]:
      common &= set(shard.extras.keys())
    return tuple(sorted(common))

  def _build_env_sequences(self) -> list[dict[int | None, EnvSequence]]:
    sequences: list[dict[int | None, EnvSequence]] = []
    for shard in self._shards:
      env_sequences: dict[int | None, EnvSequence] = {}
      if shard.env_id is None:
        row_indices = np.arange(shard.obs.shape[0], dtype=np.int64)
        env_sequences[None] = EnvSequence(
          env_id=None,
          row_indices=row_indices,
          done=shard.done[row_indices],
          episode_id=shard.episode_id[row_indices],
        )
      else:
        for env in np.unique(shard.env_id):
          row_indices = np.where(shard.env_id == env)[0]
          env_sequences[int(env)] = EnvSequence(
            env_id=int(env),
            row_indices=row_indices,
            done=shard.done[row_indices],
            episode_id=shard.episode_id[row_indices],
          )
      sequences.append(env_sequences)
    return sequences

  def _build_indices(self) -> tuple[list[SampleIndex], list[SampleIndex]]:
    step_indices: list[SampleIndex] = []
    chunk_indices: list[SampleIndex] = []

    for shard_idx, env_id, start, end, episode_id in self._iter_episode_segments():
      if not self._episode_allowed(shard_idx, env_id, episode_id):
        continue
      step_last = end - self.k_future - 1
      if step_last >= start:
        for t in range(start, step_last + 1):
          step_indices.append(
            SampleIndex(shard_idx=shard_idx, env_id=env_id, env_pos=t)
          )

      chunk_last = end - self.chunk_len - self.k_future
      if chunk_last >= start:
        for t in range(start, chunk_last + 1):
          chunk_indices.append(
            SampleIndex(shard_idx=shard_idx, env_id=env_id, env_pos=t)
          )

    return step_indices, chunk_indices

  def _sample_step(
    self, batch_size: int, rng: np.random.Generator
  ) -> dict[str, np.ndarray]:
    if not self._step_indices:
      raise RuntimeError("No valid step indices available.")

    replace = batch_size > len(self._step_indices)
    picks = rng.choice(len(self._step_indices), size=batch_size, replace=replace)

    obs_list = []
    act_list = []
    future_list = []
    done_list = []
    run_names = []
    episode_ids = []
    env_ids = []
    extras_out = {key: [] for key in self._extra_keys}

    for pick in picks:
      idx = self._step_indices[int(pick)]
      shard = self._shards[idx.shard_idx]
      seq = self._env_sequences[idx.shard_idx][idx.env_id]

      row_idx = seq.row_indices[idx.env_pos]
      obs_list.append(shard.obs[row_idx])
      act_list.append(shard.a_clean[row_idx])
      done_list.append(shard.done[row_idx])

      if self.k_future > 0:
        future_rows = seq.row_indices[
          idx.env_pos + 1 : idx.env_pos + 1 + self.k_future
        ]
        future_list.append(shard.obs[future_rows])
      else:
        future_list.append(np.zeros((0, shard.obs.shape[1]), dtype=shard.obs.dtype))

      run_names.append(shard.run_name)
      episode_ids.append(shard.episode_id[row_idx])
      env_ids.append(-1 if shard.env_id is None else shard.env_id[row_idx])
      for key in self._extra_keys:
        extras_out[key].append(shard.extras[key][row_idx])

    batch = {
      "obs_t": np.stack(obs_list, axis=0),
      "a_clean_t": np.stack(act_list, axis=0),
      "obs_future": np.stack(future_list, axis=0),
      "done": np.asarray(done_list, dtype=np.bool_),
      "run_name": np.asarray(run_names),
      "episode_id": np.asarray(episode_ids, dtype=np.int64),
      "env_id": np.asarray(env_ids, dtype=np.int64),
    }
    for key, values in extras_out.items():
      batch[key] = np.asarray(values)
    return batch

  def _sample_chunk(
    self, batch_size: int, rng: np.random.Generator
  ) -> dict[str, np.ndarray]:
    if not self._chunk_indices:
      raise RuntimeError("No valid chunk indices available.")

    replace = batch_size > len(self._chunk_indices)
    picks = rng.choice(len(self._chunk_indices), size=batch_size, replace=replace)

    obs_chunks = []
    act_chunks = []
    done_chunks = []
    future_chunks = []
    run_names = []
    episode_ids = []
    env_ids = []
    extras_out = {key: [] for key in self._extra_keys}

    for pick in picks:
      idx = self._chunk_indices[int(pick)]
      shard = self._shards[idx.shard_idx]
      seq = self._env_sequences[idx.shard_idx][idx.env_id]

      start = idx.env_pos
      end = start + self.chunk_len
      row_indices = seq.row_indices[start:end]

      obs_chunk = shard.obs[row_indices]
      act_chunk = shard.a_clean[row_indices]
      done_chunk = shard.done[row_indices]

      if self.k_future > 0:
        future_steps = []
        for offset in range(self.chunk_len):
          t = start + offset
          future_rows = seq.row_indices[t + 1 : t + 1 + self.k_future]
          future_steps.append(shard.obs[future_rows])
        future_chunks.append(np.stack(future_steps, axis=0))
      else:
        future_chunks.append(
          np.zeros((self.chunk_len, 0, shard.obs.shape[1]), dtype=shard.obs.dtype)
        )

      obs_chunks.append(obs_chunk)
      act_chunks.append(act_chunk)
      done_chunks.append(done_chunk)
      run_names.append(shard.run_name)
      episode_ids.append(shard.episode_id[row_indices])
      env_ids.append(-1 if shard.env_id is None else shard.env_id[row_indices[0]])
      for key in self._extra_keys:
        extras_out[key].append(shard.extras[key][row_indices])

    batch = {
      "obs_chunk": np.stack(obs_chunks, axis=0),
      "a_clean_chunk": np.stack(act_chunks, axis=0),
      "obs_future": np.stack(future_chunks, axis=0),
      "done": np.stack(done_chunks, axis=0),
      "run_name": np.asarray(run_names),
      "episode_id": np.stack(episode_ids, axis=0),
      "env_id": np.asarray(env_ids, dtype=np.int64),
    }
    for key, values in extras_out.items():
      batch[key] = np.stack(values, axis=0)
    return batch

  def _compute_stats(self) -> RolloutDatasetStats:
    num_shards = len(self._shards)
    if self._episode_allowlist is None:
      total_steps = sum(s.obs.shape[0] for s in self._shards)
    else:
      total_steps = 0
      for shard_idx, env_id, start, end, episode_id in self._iter_episode_segments():
        if self._episode_allowed(shard_idx, env_id, episode_id):
          total_steps += end - start
    obs_dim = self._shards[0].obs.shape[1]
    act_dim = self._shards[0].a_clean.shape[1]
    keys_found = sorted({key for shard in self._shards for key in shard.keys})
    run_names = tuple(sorted({shard.run_name for shard in self._shards}))

    sample = self._shards[0]
    sample_shapes = {
      "obs": sample.obs.shape,
      "a_clean": sample.a_clean.shape,
      "done": sample.done.shape,
    }

    return RolloutDatasetStats(
      num_runs=len(run_names),
      num_shards=num_shards,
      total_steps=total_steps,
      obs_dim=obs_dim,
      act_dim=act_dim,
      keys_found=tuple(keys_found),
      run_names=run_names,
      sample_shapes=sample_shapes,
    )

  def episode_keys_by_run(self) -> dict[str, list[tuple[str, int | None, int]]]:
    by_run: dict[str, set[tuple[str, int | None, int]]] = {}
    for shard_idx, env_id, start, _end, episode_id in self._iter_episode_segments():
      shard = self._shards[shard_idx]
      key = (shard.run_name, env_id, int(episode_id))
      by_run.setdefault(shard.run_name, set()).add(key)
    return {
      run: sorted(
        list(keys),
        key=lambda k: (k[0], -1 if k[1] is None else int(k[1]), int(k[2])),
      )
      for run, keys in by_run.items()
    }

  def iter_allowed_row_indices(self):
    for shard_idx, env_sequences in enumerate(self._env_sequences):
      for env_id, seq in env_sequences.items():
        segments = _split_segments(seq.done)
        for start, end in segments:
          episode_id = int(seq.episode_id[start])
          if not self._episode_allowed(shard_idx, env_id, episode_id):
            continue
          yield shard_idx, seq.row_indices[start:end]

  def _iter_episode_segments(self):
    for shard_idx, env_sequences in enumerate(self._env_sequences):
      for env_id, seq in env_sequences.items():
        segments = _split_segments(seq.done)
        for start, end in segments:
          episode_id = int(seq.episode_id[start])
          yield shard_idx, env_id, start, end, episode_id

  def _episode_allowed(
    self, shard_idx: int, env_id: int | None, episode_id: int
  ) -> bool:
    if self._episode_allowlist is None:
      return True
    shard = self._shards[shard_idx]
    key = (shard.run_name, env_id, int(episode_id))
    return key in self._episode_allowlist


def _compute_episode_id(done: np.ndarray, env_id: np.ndarray | None) -> np.ndarray:
  episode_id = np.zeros(done.shape[0], dtype=np.int64)
  if env_id is None:
    current = 0
    for i in range(done.shape[0]):
      episode_id[i] = current
      if done[i]:
        current += 1
    return episode_id

  counters: dict[int, int] = {}
  for i, env in enumerate(env_id):
    env_key = int(env)
    current = counters.get(env_key, 0)
    episode_id[i] = current
    if done[i]:
      counters[env_key] = current + 1
    else:
      counters[env_key] = current
  return episode_id


def _split_segments(done: np.ndarray) -> list[tuple[int, int]]:
  ends = np.where(done)[0].tolist()
  segments = []
  start = 0
  for end in ends:
    segments.append((start, end + 1))
    start = end + 1
  if start < len(done):
    segments.append((start, len(done)))
  return segments
