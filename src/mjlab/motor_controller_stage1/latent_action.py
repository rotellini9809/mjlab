from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.motor_controller_stage1.model import LatentModelConfig, NPMPLatentMotorPrimitive
from mjlab.motor_controller_stage1.trainer import Normalizer
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.string import resolve_matching_names_values

CheckpointSelector = Literal["best", "last", "latest"]

# Global Stage-1 defaults for all expert tasks.
# Set DEFAULT_STAGE1_WANDB_RUN_PATH once to avoid repeating run wiring in each expert env cfg.
DEFAULT_STAGE1_WANDB_RUN_PATH: str | None = "fratelligpt-sapienza-universit-di-roma/motor_controller_stage1/6e30hj7w"
DEFAULT_STAGE1_CHECKPOINT: CheckpointSelector = "best"
DEFAULT_MOTOR_OBS_TERMS: tuple[str, ...] = (
  "base_lin_vel",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)


def default_motor_obs_layout(*, act_dim: int) -> tuple[tuple[str, ...], tuple[int, ...]]:
  """Build the canonical no-command Stage-1 motor-observation layout."""
  if act_dim <= 0:
    raise ValueError(f"act_dim must be > 0, got {act_dim}.")
  return DEFAULT_MOTOR_OBS_TERMS, (
    3,  # base_lin_vel
    3,  # base_ang_vel
    act_dim,  # joint_pos
    act_dim,  # joint_vel
    act_dim,  # actions
  )


@dataclass(kw_only=True)
class MotorLatentActionCfg(ActionTermCfg):
  """Latent action term: z -> frozen Stage-1 motor decoder -> joint targets."""

  actuator_names: tuple[str, ...] | list[str]

  scale: float | dict[str, float] = 1.0
  offset: float | dict[str, float] = 0.0
  use_default_offset: bool = True

  # Which command term to query for optional "command" motor-observation term.
  command_name: str = "motion"

  # Stage-1 W&B run path (entity/project/run_id).
  stage1_wandb_run_path: str | None = None
  stage1_checkpoint: CheckpointSelector = DEFAULT_STAGE1_CHECKPOINT

  # Optional explicit motor-observation layout override.
  motor_obs_terms: tuple[str, ...] | None = None
  motor_obs_term_dims: tuple[int, ...] | None = None

  # If true, fail fast when expected obs layout cannot be recovered.
  strict_obs_layout: bool = True

  # Built-in sensor names used when Stage-1 student obs includes base velocity terms.
  base_lin_vel_sensor_name: str = "robot/imu_lin_vel"
  base_ang_vel_sensor_name: str = "robot/imu_ang_vel"

  def build(self, env):
    return MotorLatentAction(self, env)


@dataclass(frozen=True)
class _DecoderBundle:
  model: NPMPLatentMotorPrimitive
  normalizer: Normalizer
  obs_terms: tuple[str, ...]
  obs_dims: tuple[int, ...]
  obs_dim: int
  act_dim: int
  z_dim: int
  metadata_path: Path
  checkpoint_path: Path


def _find_repo_root(start: Path) -> Path:
  current = start if start.is_dir() else start.parent
  for parent in [current, *current.parents]:
    if (parent / "pyproject.toml").is_file():
      return parent
    if (parent / ".git").exists():
      return parent
  raise RuntimeError("Unable to locate repo root (pyproject.toml or .git).")


def _load_json(path: Path) -> dict[str, object]:
  return cast(dict[str, object], json.loads(path.read_text()))


def _stage1_wandb_cache_root(repo_root: Path) -> Path:
  return repo_root / "logs" / "motor_controller_stage1"


def _resolve_stage1_wandb_run_path(stage1_wandb_run_path: str | None) -> str:
  env_override = os.environ.get("MJLAB_STAGE1_WANDB_RUN_PATH")
  chosen = stage1_wandb_run_path or env_override or DEFAULT_STAGE1_WANDB_RUN_PATH
  if not chosen:
    raise ValueError(
      "Stage-1 W&B run path is required. Set MotorLatentActionCfg.stage1_wandb_run_path, "
      "MJLAB_STAGE1_WANDB_RUN_PATH, or DEFAULT_STAGE1_WANDB_RUN_PATH."
    )
  return chosen


def _download_run_file(cache_root: Path, run_path: str, filename: str) -> Path:
  import wandb

  run_id = run_path.split("/")[-1]
  download_dir = cache_root / "wandb_checkpoints" / run_id
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


def _try_download_run_file(
  cache_root: Path,
  run_path: str,
  filename: str,
) -> Path | None:
  try:
    return _download_run_file(cache_root, run_path, filename)
  except FileNotFoundError:
    return None


def _download_latest_model_file(cache_root: Path, run_path: str) -> Path:
  import wandb

  run_id = run_path.split("/")[-1]
  download_dir = cache_root / "wandb_checkpoints" / run_id
  download_dir.mkdir(parents=True, exist_ok=True)

  api = wandb.Api()
  run = api.run(run_path)
  model_files = [f.name for f in run.files() if f.name.startswith("model") and f.name.endswith(".pt")]
  if not model_files:
    raise FileNotFoundError(f"No model checkpoint files found in W&B run {run_path}.")

  numeric_models: list[tuple[int, str]] = []
  for name in model_files:
    if name.startswith("model_") and name.endswith(".pt"):
      stem = name[len("model_") : -len(".pt")]
      if stem.isdigit():
        numeric_models.append((int(stem), name))

  if numeric_models:
    numeric_models.sort(key=lambda x: x[0])
    filename = numeric_models[-1][1]
  elif "model_last.pt" in model_files:
    filename = "model_last.pt"
  elif "model_best.pt" in model_files:
    filename = "model_best.pt"
  else:
    filename = sorted(model_files)[-1]

  target = download_dir / filename
  if not target.exists():
    run.file(filename).download(str(download_dir), replace=True)
  return target


def _resolve_checkpoint_from_wandb(
  cache_root: Path,
  run_path: str,
  checkpoint: CheckpointSelector,
) -> Path:
  if checkpoint == "best":
    ckpt = _try_download_run_file(cache_root, run_path, "model_best.pt")
    if ckpt is not None:
      return ckpt
    ckpt = _try_download_run_file(cache_root, run_path, "model_last.pt")
    if ckpt is not None:
      return ckpt
  elif checkpoint == "last":
    ckpt = _try_download_run_file(cache_root, run_path, "model_last.pt")
    if ckpt is not None:
      return ckpt
    ckpt = _try_download_run_file(cache_root, run_path, "model_best.pt")
    if ckpt is not None:
      return ckpt

  return _download_latest_model_file(cache_root, run_path)


def _extract_obs_layout_from_view(
  view: dict[str, object],
) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
  kept_terms_raw = view.get("kept_terms")
  slice_map_raw = view.get("slice_map")
  if not isinstance(kept_terms_raw, list) or not isinstance(slice_map_raw, list):
    return None

  size_by_name: dict[str, int] = {}
  for item in slice_map_raw:
    if not isinstance(item, dict):
      continue
    name = item.get("name")
    if not isinstance(name, str):
      continue
    if "size" in item:
      size = int(item["size"])
    elif "start" in item and "end" in item:
      size = int(item["end"]) - int(item["start"])
    else:
      continue
    size_by_name[name] = size

  terms: list[str] = []
  dims: list[int] = []
  for term in kept_terms_raw:
    if not isinstance(term, str):
      return None
    if term not in size_by_name:
      return None
    terms.append(term)
    dims.append(int(size_by_name[term]))

  if not terms:
    return None
  return tuple(terms), tuple(dims)


def _candidate_mapped_output_dirs(output_dir: str, repo_root: Path) -> tuple[Path, ...]:
  raw = Path(output_dir)
  candidates = [raw]

  if output_dir.startswith("/app/"):
    rel = output_dir[len("/app/") :]
    candidates.append(repo_root / rel)
    candidates.append(repo_root.parent / rel)

  marker = "/data/"
  if marker in output_dir:
    rel_data = output_dir.split(marker, 1)[1]
    candidates.append(repo_root / "data" / rel_data)
    candidates.append(repo_root.parent / "data" / rel_data)

  uniq: list[Path] = []
  seen: set[Path] = set()
  for cand in candidates:
    c = cand.expanduser()
    if c in seen:
      continue
    seen.add(c)
    uniq.append(c)
  return tuple(uniq)


def _resolve_motor_obs_layout(
  metadata: dict[str, object],
  cfg: MotorLatentActionCfg,
  repo_root: Path,
  obs_dim: int,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
  if cfg.motor_obs_terms is not None or cfg.motor_obs_term_dims is not None:
    if cfg.motor_obs_terms is None or cfg.motor_obs_term_dims is None:
      raise ValueError(
        "motor_obs_terms and motor_obs_term_dims must be both set when overriding motor obs layout."
      )
    if len(cfg.motor_obs_terms) != len(cfg.motor_obs_term_dims):
      raise ValueError("motor_obs_terms and motor_obs_term_dims length mismatch.")
    configured_sum = sum(int(v) for v in cfg.motor_obs_term_dims)
    if configured_sum != obs_dim:
      print(
        "[DEBUG] Motor obs layout mismatch | "
        f"stage1_obs_dim={obs_dim}, configured_sum={configured_sum}, "
        f"terms={cfg.motor_obs_terms}, term_dims={cfg.motor_obs_term_dims}"
      )
      raise ValueError(
        "motor_obs_term_dims sum does not match Stage-1 obs_dim from metadata."
      )
    return tuple(cfg.motor_obs_terms), tuple(int(v) for v in cfg.motor_obs_term_dims)

  direct_view = metadata.get("obs_student_view")
  if isinstance(direct_view, dict):
    direct = _extract_obs_layout_from_view(cast(dict[str, object], direct_view))
    if direct is not None and sum(direct[1]) == obs_dim:
      return direct

  rollout_sources = metadata.get("rollout_sources")
  if isinstance(rollout_sources, list):
    for src in rollout_sources:
      if not isinstance(src, dict):
        continue
      output_dir = src.get("output_dir")
      if not isinstance(output_dir, str):
        continue
      for mapped_dir in _candidate_mapped_output_dirs(output_dir, repo_root):
        rollout_meta = mapped_dir / "metadata.json"
        if not rollout_meta.is_file():
          continue
        try:
          data = _load_json(rollout_meta)
        except Exception:
          continue
        view = data.get("obs_student_view")
        if not isinstance(view, dict):
          continue
        layout = _extract_obs_layout_from_view(cast(dict[str, object], view))
        if layout is None:
          continue
        if sum(layout[1]) != obs_dim:
          continue
        return layout

  msg = (
    "Unable to recover motor observation layout from Stage-1 metadata. "
    "Set MotorLatentActionCfg.motor_obs_terms and motor_obs_term_dims explicitly."
  )
  if cfg.strict_obs_layout:
    raise RuntimeError(msg)
  raise RuntimeError(msg)


def _load_stage1_decoder_bundle(
  cfg: MotorLatentActionCfg,
  *,
  device: str,
  expected_act_dim: int,
) -> _DecoderBundle:
  repo_root = _find_repo_root(Path(__file__).resolve())
  cache_root = _stage1_wandb_cache_root(repo_root)
  run_path = _resolve_stage1_wandb_run_path(cfg.stage1_wandb_run_path)

  metadata_path = _download_run_file(cache_root, run_path, "metadata.json")
  norm_path = _download_run_file(cache_root, run_path, "normalization_stats.npz")

  checkpoint_name = cast(
    CheckpointSelector,
    os.environ.get("MJLAB_STAGE1_CHECKPOINT", cfg.stage1_checkpoint),
  )
  print(
    "[DEBUG] Stage-1 bundle selection | "
    f"run_path={run_path}, checkpoint_selector={checkpoint_name}, device={device}"
  )
  checkpoint_path = _resolve_checkpoint_from_wandb(
    cache_root,
    run_path,
    checkpoint_name,
  )

  metadata = _load_json(metadata_path)
  dataset_info = cast(dict[str, object], metadata.get("dataset", {}))
  config_info = cast(dict[str, object], metadata.get("config", {}))

  obs_dim = int(dataset_info.get("obs_dim", -1))
  act_dim = int(dataset_info.get("act_dim", -1))
  latent_type = str(config_info.get("latent_type", ""))
  z_dim = int(config_info.get("z_dim", -1))
  hidden_dim = int(config_info.get("hidden_dim", -1))
  k_future = int(config_info.get("k_future", -1))

  if obs_dim <= 0 or act_dim <= 0 or z_dim <= 0 or hidden_dim <= 0 or k_future <= 0:
    raise RuntimeError(
      f"Stage-1 metadata missing required dims/config: {metadata_path}"
    )
  print(
    "[DEBUG] Stage-1 metadata dims | "
    f"obs_dim={obs_dim}, act_dim={act_dim}, z_dim={z_dim}, "
    f"hidden_dim={hidden_dim}, k_future={k_future}, latent_type={latent_type}"
  )
  if latent_type != "npmp":
    raise RuntimeError(
      f"Unsupported Stage-1 latent_type='{latent_type}'. Expected 'npmp'."
    )
  if act_dim != expected_act_dim:
    raise RuntimeError(
      "Stage-1 act_dim does not match robot action dim: "
      f"stage1={act_dim}, env_robot={expected_act_dim}."
    )

  obs_terms, obs_dims = _resolve_motor_obs_layout(metadata, cfg, repo_root, obs_dim)

  mcfg = LatentModelConfig(
    obs_dim=obs_dim,
    act_dim=act_dim,
    k_future=k_future,
    z_dim=z_dim,
    hidden_dim=hidden_dim,
  )
  model = NPMPLatentMotorPrimitive(mcfg).to(device)

  data = torch.load(checkpoint_path, map_location=device)
  if isinstance(data, dict) and "state_dict" in data:
    state_dict = data["state_dict"]
  elif isinstance(data, dict) and "model" in data:
    state_dict = data["model"]
  elif isinstance(data, dict):
    state_dict = data
  else:
    raise RuntimeError(f"Unsupported Stage-1 checkpoint format: {checkpoint_path}")

  model.load_state_dict(state_dict, strict=True)
  model.eval()
  model.requires_grad_(False)

  normalizer = Normalizer.from_npz(norm_path)

  return _DecoderBundle(
    model=model,
    normalizer=normalizer,
    obs_terms=obs_terms,
    obs_dims=obs_dims,
    obs_dim=obs_dim,
    act_dim=act_dim,
    z_dim=z_dim,
    metadata_path=metadata_path,
    checkpoint_path=checkpoint_path,
  )


class MotorLatentAction(ActionTerm):
  cfg: MotorLatentActionCfg

  def __init__(self, cfg: MotorLatentActionCfg, env):
    super().__init__(cfg=cfg, env=env)

    target_ids, target_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
    self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
    self._target_names = target_names
    self._num_targets = len(target_ids)

    bundle = _load_stage1_decoder_bundle(
      cfg,
      device=self.device,
      expected_act_dim=self._num_targets,
    )

    self._model = bundle.model
    self._motor_obs_terms = bundle.obs_terms
    self._motor_obs_dims = bundle.obs_dims
    self._motor_obs_dim = bundle.obs_dim

    self._obs_mean = torch.from_numpy(bundle.normalizer.obs_mean).to(self.device)
    self._obs_std = torch.from_numpy(bundle.normalizer.obs_std).to(self.device)
    self._act_mean = torch.from_numpy(bundle.normalizer.act_mean).to(self.device)
    self._act_std = torch.from_numpy(bundle.normalizer.act_std).to(self.device)

    self._action_dim = bundle.z_dim
    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._decoded_actions = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )
    self._processed_actions = torch.zeros_like(self._decoded_actions)
    self._last_decoded_actions = torch.zeros_like(self._decoded_actions)

    if isinstance(cfg.scale, (float, int)):
      self._scale = float(cfg.scale)
    elif isinstance(cfg.scale, dict):
      self._scale = torch.ones(self.num_envs, self._num_targets, device=self.device)
      ids, _, vals = resolve_matching_names_values(cfg.scale, self._target_names)
      self._scale[:, ids] = torch.tensor(vals, device=self.device)
    else:
      raise ValueError(f"Unsupported scale type: {type(cfg.scale)}")

    if isinstance(cfg.offset, (float, int)):
      self._offset = float(cfg.offset)
    elif isinstance(cfg.offset, dict):
      self._offset = torch.zeros(self.num_envs, self._num_targets, device=self.device)
      ids, _, vals = resolve_matching_names_values(cfg.offset, self._target_names)
      self._offset[:, ids] = torch.tensor(vals, device=self.device)
    else:
      raise ValueError(f"Unsupported offset type: {type(cfg.offset)}")

    if cfg.use_default_offset:
      default_joint_pos = self._entity.data.default_joint_pos
      assert default_joint_pos is not None
      self._offset = default_joint_pos[:, self._target_ids].clone()

    self._decoded_this_step = False

    print(
      "[INFO] MotorLatentAction initialized | "
      f"z_dim={self._action_dim}, act_dim={self._num_targets}, "
      f"motor_obs_dim={self._motor_obs_dim}, terms={self._motor_obs_terms}, "
      f"checkpoint={bundle.checkpoint_path}"
    )

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def last_decoded_action(self) -> torch.Tensor:
    return self._last_decoded_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape[1] != self._action_dim:
      raise ValueError(
        f"Invalid latent action shape {actions.shape}; expected (_, {self._action_dim})."
      )

    self._raw_actions[:] = actions.to(self.device)
    motor_obs = self._build_motor_obs()
    obs_norm = (motor_obs - self._obs_mean) / self._obs_std

    with torch.no_grad():
      a_norm = self._model.decoder(obs_norm, self._raw_actions)
      decoded = a_norm * self._act_std + self._act_mean

    if decoded.shape != (self.num_envs, self._num_targets):
      raise RuntimeError(
        "Decoded action shape mismatch: "
        f"decoded={tuple(decoded.shape)}, expected={(self.num_envs, self._num_targets)}"
      )

    self._decoded_actions[:] = decoded
    self._processed_actions[:] = self._decoded_actions * self._scale + self._offset
    self._decoded_this_step = True

  def apply_actions(self) -> None:
    if not self._decoded_this_step:
      raise RuntimeError("MotorLatentAction.apply_actions called before process_actions.")

    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

    self._last_decoded_actions[:] = self._decoded_actions

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
    self._decoded_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0
    self._last_decoded_actions[env_ids] = 0.0
    self._decoded_this_step = False

  def _build_motor_obs(self) -> torch.Tensor:
    term_values = {
      "base_lin_vel": self._builtin_sensor(self.cfg.base_lin_vel_sensor_name),
      "base_ang_vel": self._builtin_sensor(self.cfg.base_ang_vel_sensor_name),
      "joint_pos": self._joint_pos_rel(biased=True),
      "joint_vel": self._joint_vel_rel(),
      "actions": self._last_decoded_actions,
    }

    if "command" in self._motor_obs_terms:
      command = self._env.command_manager.get_command(self.cfg.command_name)
      if command is None:
        raise RuntimeError(
          f"Command '{self.cfg.command_name}' not found but motor obs expects 'command'."
        )
      term_values["command"] = command

    chunks: list[torch.Tensor] = []
    for term_name, expected_dim in zip(self._motor_obs_terms, self._motor_obs_dims, strict=False):
      if term_name not in term_values:
        raise RuntimeError(
          f"Motor obs term '{term_name}' is not available in MotorLatentAction."
        )
      value = term_values[term_name]
      if value.shape[-1] != expected_dim:
        raise RuntimeError(
          f"Motor obs term '{term_name}' has dim={value.shape[-1]} but expected {expected_dim}."
        )
      chunks.append(value)

    motor_obs = torch.cat(chunks, dim=-1)
    if motor_obs.shape[-1] != self._motor_obs_dim:
      raise RuntimeError(
        f"Motor obs dim mismatch: built={motor_obs.shape[-1]}, expected={self._motor_obs_dim}."
      )
    return motor_obs

  def _joint_pos_rel(self, biased: bool) -> torch.Tensor:
    default_joint_pos = self._entity.data.default_joint_pos
    assert default_joint_pos is not None
    joint_pos = self._entity.data.joint_pos_biased if biased else self._entity.data.joint_pos
    return joint_pos - default_joint_pos

  def _joint_vel_rel(self) -> torch.Tensor:
    default_joint_vel = self._entity.data.default_joint_vel
    assert default_joint_vel is not None
    return self._entity.data.joint_vel - default_joint_vel

  def _builtin_sensor(self, sensor_name: str) -> torch.Tensor:
    sensor = self._env.scene[sensor_name]
    assert isinstance(sensor, BuiltinSensor)
    return sensor.data


def motor_last_decoded_action(
  env,
  action_name: str = "motor_latent",
) -> torch.Tensor:
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, MotorLatentAction):
    raise TypeError(
      f"Action term '{action_name}' is not MotorLatentAction (got {type(term)})."
    )
  return term.last_decoded_action
