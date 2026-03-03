"""Play script with reward debug overlay for push_getup."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mujoco
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

TASK_ID = "Mjlab-PushGetUp-Flat-Booster-T1_23"


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


class PushGetupDebugViewer(NativeMujocoViewer):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, key_callback=self._on_key, **kwargs)
    self._page_idx = 0
    self._num_pages = 2

  def _on_key(self, key: int) -> None:
    # Runs on viewer thread; only mutate local state.
    from mjlab.viewer.native.keys import KEY_B, KEY_N

    if key == KEY_N:
      self._page_idx = (self._page_idx + 1) % self._num_pages
    elif key == KEY_B:
      self._page_idx = (self._page_idx - 1) % self._num_pages

  def _format_value(self, value: float | None, fmt: str = "{:.3f}") -> str:
    if value is None:
      return "n/a"
    try:
      return fmt.format(float(value))
    except Exception:
      return "n/a"

  def _bool_str(self, value: bool | None) -> str:
    if value is None:
      return "n/a"
    return "1" if value else "0"

  def _get_term_value(self, terms: dict[str, float], name: str) -> float | None:
    if name not in terms:
      return None
    return terms[name]

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    v = self.viewer
    if v is None:
      return
    try:
      env_unwrapped = self.env.unwrapped
      env_idx = self.env_idx

      total_reward = None
      if hasattr(env_unwrapped, "reward_buf") and env_unwrapped.reward_buf is not None:
        total_reward = float(env_unwrapped.reward_buf[env_idx].item())

      terms_list = env_unwrapped.reward_manager.get_active_iterable_terms(env_idx)
      terms = {name: float(arr[0]) for name, arr in terms_list}

      r_ep = None
      reward_manager = getattr(env_unwrapped, "reward_manager", None)
      if reward_manager is not None and hasattr(reward_manager, "_episode_sums"):
        try:
          r_ep = 0.0
          for value in reward_manager._episode_sums.values():
            r_ep += float(value[env_idx].item())
        except Exception:
          r_ep = None

      cmd = None
      try:
        cmd = env_unwrapped.command_manager.get_term("motion")
      except Exception:
        cmd = None

      trunk_height_norm = None
      control_enabled = None
      fallen_once = None
      stage = None
      stand_pose_multiplier = None
      standing_mask = None
      standing_mask_frac = None
      phase = "n/a"
      height_switch = None

      if cmd is not None:
        if hasattr(cmd, "trunk_height_norm"):
          trunk_height_norm = float(cmd.trunk_height_norm[env_idx].item())
        if hasattr(cmd, "control_enabled"):
          control_enabled = bool(cmd.control_enabled[env_idx].item())
        if hasattr(cmd, "fallen_once"):
          fallen_once = bool(cmd.fallen_once[env_idx].item())
        if hasattr(cmd, "stage"):
          stage = int(cmd.stage[env_idx].item())
        if hasattr(cmd, "stand_pose_multiplier"):
          try:
            stand_pose_multiplier = float(cmd.stand_pose_multiplier)
          except Exception:
            stand_pose_multiplier = None
        if hasattr(cmd, "_compute_standing_mask"):
          mask = cmd._compute_standing_mask()
          standing_mask = bool(mask[env_idx].item())
          standing_mask_frac = float(mask.float().mean().item())
        else:
          metrics = getattr(cmd, "metrics", {})
          if "Curriculum/standing_mask_frac" in metrics:
            standing_mask_frac = float(
              metrics["Curriculum/standing_mask_frac"][env_idx].item()
            )

      if reward_manager is not None:
        try:
          height_switch = reward_manager.get_term_cfg("support_points").params.get(
            "height_threshold", None
          )
        except Exception:
          height_switch = None

      if standing_mask is True:
        phase = "STANDING"
      elif trunk_height_norm is not None and height_switch is not None:
        phase = "EARLY" if trunk_height_norm < float(height_switch) else "LATE"

      if self._page_idx == 0:
        text_1 = "\n".join(
          [
            "Page 1/2 (N/B)",
            "Step",
            "r_step_total",
            "R_ep",
            "Phase",
            "standing_mask",
            "standing_mask_frac",
            "trunk_height",
            "trunk_h_prog",
            "support_pts",
            "both_feet",
            "pelvis_pen",
            "stand_pose",
            "self_coll",
          ]
        )
        text_2 = "\n".join(
          [
            "",
            str(self._step_count),
            self._format_value(total_reward),
            self._format_value(r_ep),
            phase,
            self._bool_str(standing_mask),
            self._format_value(standing_mask_frac),
            self._format_value(self._get_term_value(terms, "trunk_height")),
            self._format_value(self._get_term_value(terms, "trunk_height_progress")),
            self._format_value(self._get_term_value(terms, "support_points")),
            self._format_value(self._get_term_value(terms, "both_feet")),
            self._format_value(self._get_term_value(terms, "pelvis_contact_penalty")),
            self._format_value(self._get_term_value(terms, "stand_pose_penalty")),
            self._format_value(self._get_term_value(terms, "self_collisions")),
          ]
        )
      else:
        text_1 = "\n".join(
          [
            "Page 2/2 (N/B)",
            "trunk_h_norm",
            "control_enabled",
            "fallen_once",
            "stage",
            "stand_pose_mult",
          ]
        )
        text_2 = "\n".join(
          [
            "",
            self._format_value(trunk_height_norm),
            self._bool_str(control_enabled),
            self._bool_str(fallen_once),
            "n/a" if stage is None else str(stage),
            self._format_value(stand_pose_multiplier),
          ]
        )
      overlay = (
        mujoco.mjtFontScale.mjFONTSCALE_100.value,
        mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
        text_1,
        text_2,
      )
      v.set_texts(overlay)
    except Exception:
      return


class PushGetupViserViewer(ViserPlayViewer):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._reward_debug_text = None

  def setup(self) -> None:
    super().setup()
    self._reward_debug_text = self._server.gui.add_text(
      "Rewards (Debug)",
      "",
      multiline=True,
      disabled=True,
    )

  def _update_status_display(self) -> None:
    super()._update_status_display()
    if self._reward_debug_text is None:
      return

    env_unwrapped = self.env.unwrapped
    env_idx = self._scene.env_idx

    r_step_total = None
    if hasattr(env_unwrapped, "reward_buf") and env_unwrapped.reward_buf is not None:
      try:
        r_step_total = float(env_unwrapped.reward_buf[env_idx].item())
      except Exception:
        r_step_total = None

    r_ep = None
    reward_manager = getattr(env_unwrapped, "reward_manager", None)
    if reward_manager is not None and hasattr(reward_manager, "_episode_sums"):
      try:
        r_ep = 0.0
        for value in reward_manager._episode_sums.values():
          r_ep += float(value[env_idx].item())
      except Exception:
        r_ep = None

    trunk_height_norm = None
    standing_mask = None
    phase = "n/a"
    switch = None
    try:
      cmd = env_unwrapped.command_manager.get_term("motion")
      if hasattr(cmd, "trunk_height_norm"):
        trunk_height_norm = float(cmd.trunk_height_norm[env_idx].item())
      if hasattr(cmd, "_compute_standing_mask"):
        standing_mask = bool(cmd._compute_standing_mask()[env_idx].item())
    except Exception:
      cmd = None

    if reward_manager is not None:
      try:
        switch = reward_manager.get_term_cfg("support_points").params.get(
          "height_threshold", None
        )
      except Exception:
        switch = None

    if standing_mask is True:
      phase = "STANDING"
    elif trunk_height_norm is not None and switch is not None:
      phase = "EARLY" if trunk_height_norm < float(switch) else "LATE"

    term_lines = []
    if reward_manager is not None:
      try:
        terms = reward_manager.get_active_iterable_terms(env_idx)
        for name, arr in terms:
          try:
            val = float(arr[0])
            term_lines.append(f"{name}: {val:.3f}")
          except Exception:
            term_lines.append(f"{name}: n/a")
      except Exception:
        term_lines = []

    def _fmt(value: float | None) -> str:
      if value is None:
        return "n/a"
      return f"{value:.3f}"

    term_text = "\n".join(term_lines) if term_lines else "n/a"
    self._reward_debug_text.value = (
      "Phase: "
      + phase
      + "\n"
      + "r_step_total: "
      + _fmt(r_step_total)
      + "\n"
      + "R_ep: "
      + _fmt(r_ep)
      + "\n"
      + "Terms (weighted):\n"
      + term_text
    )

def run_play(task_id: str, cfg: PlayConfig) -> None:
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    if DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require `registry_name` when using dummy agents."
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    PushGetupDebugViewer(env, policy).run()
  elif resolved_viewer == "viser":
    PushGetupViserViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main() -> None:
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  args = tyro.cli(
    PlayConfig,
    default=PlayConfig(),
    prog=f"{Path(__file__).name}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  run_play(TASK_ID, args)


if __name__ == "__main__":
  main()
