from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.t1_23dof.env_cfgs import (
    booster_t1_23_flat_tracking_env_cfg,
)
FOOT_BOTTOM_LOCAL_Z = -0.0432508  # distanza (m) tra il center del body 'left/right_foot_link' e la suola


from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


def _is_timeout_error(exc: BaseException) -> bool:
  """Return True for timeout-like exceptions (including wrapped causes)."""
  if isinstance(exc, TimeoutError):
    return True

  try:
    import requests

    if isinstance(exc, requests.exceptions.Timeout):
      return True
  except Exception:
    pass

  msg = str(exc).lower()
  if "timed out" in msg or "read timeout" in msg or "timeout=" in msg:
    return True

  cause = getattr(exc, "__cause__", None)
  if cause is not None and cause is not exc and _is_timeout_error(cause):
    return True
  context = getattr(exc, "__context__", None)
  if context is not None and context is not exc and _is_timeout_error(context):
    return True
  return False


def _is_retryable_link_error(exc: BaseException) -> bool:
  """Return True for transient link errors that should be retried."""
  if _is_timeout_error(exc):
    return True

  try:
    import requests

    if isinstance(exc, requests.exceptions.HTTPError):
      status_code = (
        exc.response.status_code if getattr(exc, "response", None) is not None else None
      )
      if status_code in {429, 500, 502, 503, 504}:
        return True
  except Exception:
    pass

  msg = str(exc).lower()
  transient_markers = (
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "too many requests",
    "rate limit",
    "temporarily unavailable",
    "internal server error",
  )
  if any(marker in msg for marker in transient_markers):
    return True

  cause = getattr(exc, "__cause__", None)
  if cause is not None and cause is not exc and _is_retryable_link_error(cause):
    return True
  context = getattr(exc, "__context__", None)
  if context is not None and context is not exc and _is_retryable_link_error(context):
    return True
  return False


def _link_artifact_with_retry(
  run: Any,
  artifact: Any,
  target_path: str,
  *,
  max_attempts: int = 5,
  base_sleep_s: float = 2.0,
) -> bool:
  for attempt in range(1, max_attempts + 1):
    try:
      run.link_artifact(artifact=artifact, target_path=target_path)
      return True
    except Exception as exc:
      retryable = _is_retryable_link_error(exc)
      is_last = attempt == max_attempts
      if not retryable:
        raise
      if is_last:
        print(
          f"[ERROR] Failed to link artifact to registry after {max_attempts} retries "
          f"due to transient API errors: {exc}"
        )
        return False
      wait_s = base_sleep_s * (2 ** (attempt - 1))
      print(
        f"[WARN] link_artifact transient error on attempt {attempt}/{max_attempts}. "
        f"Retrying in {wait_s:.1f}s..."
      )
      time.sleep(wait_s)
  return False


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # convert to wxyz
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion."""
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  output_name,
  render,
  line_range,
  renderer: OffscreenRenderer | None = None,
  wandb_project: str = "csv_to_npz",
  wandb_entity: str | None = None,
):
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  
  # --------------------------------------------------------------
  # AUTO GROUND ALIGN: usa il primo frame per mettere le suole a z = 0
  # --------------------------------------------------------------
  # Primo frame della motion interpolata
  base_pos0 = motion.motion_base_poss[0:1].clone()   # (1, 3)
  base_rot0 = motion.motion_base_rots[0:1].clone()   # (1, 4)
  dof_pos0  = motion.motion_dof_poss[0:1].clone()    # (1, n_dof)

  # Scrivi stato root nel simulatore
  root_states = robot.data.default_root_state.clone()
  root_states[:, 0:3] = base_pos0
  root_states[:, 3:7] = base_rot0
  root_states[:, 7:10] = 0.0
  root_states[:, 10:] = 0.0
  robot.write_root_state_to_sim(root_states)

  # Scrivi stato dei joint
  joint_pos = robot.data.default_joint_pos.clone()
  joint_vel = robot.data.default_joint_vel.clone()
  joint_pos[:, robot_joint_indexes] = dof_pos0
  joint_vel[:, robot_joint_indexes] = 0.0
  robot.write_joint_state_to_sim(joint_pos, joint_vel)

  # Fai un forward per aggiornare le posizioni world
  sim.forward()
  scene.update(sim.mj_model.opt.timestep)

  # Prendi le posizioni dei body dei piedi
  foot_body_ids, _ = robot.find_bodies(
      ["left_foot_link", "right_foot_link"],
      preserve_order=True,
  )
  feet_pos = robot.data.body_link_pos_w[0, foot_body_ids, :]  # (2, 3)
  min_foot_origin_z = float(feet_pos[:, 2].min().item())

  # La suola è FOOT_BOTTOM_LOCAL_Z sotto l'origine del body del piede
  foot_bottom_z = min_foot_origin_z + FOOT_BOTTOM_LOCAL_Z

  # Offset per portare il punto più basso della suola a z = 0
  z_offset = -foot_bottom_z
  print(f"[auto-ground-align] applying z offset {z_offset:.4f}")

  # Applica l'offset a tutta la traiettoria base
  motion.motion_base_poss[:, 2]       += z_offset
  motion.motion_base_poss_input[:, 2] += z_offset
  

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  # Create progress bar
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # Update every 100 frames to avoid spam
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)

        # Keep artifact name unique (output_name), but store a standard filename
        # inside artifacts for consistency with csv_to_npz.py.
        motion_npz_path = Path("/tmp") / "motion.npz"
        print(f"Saving to {motion_npz_path}...")
        np.savez(motion_npz_path, **log)  # type: ignore[arg-type]

        print("Uploading to Weights & Biases...")
        import wandb

        COLLECTION = output_name
        run = wandb.init(project=wandb_project, entity=wandb_entity, name=COLLECTION)
        print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
        REGISTRY = "motions"
        logged_artifact = run.log_artifact(
          artifact_or_path=str(motion_npz_path), name=COLLECTION, type=REGISTRY
        )
        linked = _link_artifact_with_retry(
          run=run,
          artifact=logged_artifact,
          target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}",
        )
        if linked:
          print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")
        else:
          print(
            f"[WARN]: Artifact logged but not linked to registry: {REGISTRY}/{COLLECTION}"
          )

        if render:
          from moviepy import ImageSequenceClip

          print("Creating video...")
          clip = ImageSequenceClip(frames, fps=output_fps)
          motion_video_path = Path("/tmp") / f"{output_name}.mp4"
          clip.write_videofile(str(motion_video_path))

          print("Logging video to wandb...")
          wandb.log({"motion_video": wandb.Video(str(motion_video_path), format="mp4")})

        wandb.finish()


def _run_single_csv(
  input_file: str,
  output_name: str,
  input_fps: float,
  output_fps: float,
  device: str,
  render: bool,
  line_range: tuple[int, int] | None,
  wandb_project: str,
  wandb_entity: str | None,
) -> None:
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(booster_t1_23_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()

  run_sim(
    sim=sim,
    scene=scene,
    joint_names=[
      # 1–6: gamba sinistra
      "Left_Hip_Pitch",
      "Left_Hip_Roll",
      "Left_Hip_Yaw",
      "Left_Knee_Pitch",
      "Left_Ankle_Pitch",
      "Left_Ankle_Roll",
      # 7–12: gamba destra
      "Right_Hip_Pitch",
      "Right_Hip_Roll",
      "Right_Hip_Yaw",
      "Right_Knee_Pitch",
      "Right_Ankle_Pitch",
      "Right_Ankle_Roll",
      # 13: bacino
      "Waist",
      # 14–17: braccio sinistro
      "Left_Shoulder_Pitch",
      "Left_Shoulder_Roll",
      "Left_Elbow_Pitch",
      "Left_Elbow_Yaw",
      # 18–21: braccio destro
      "Right_Shoulder_Pitch",
      "Right_Shoulder_Roll",
      "Right_Elbow_Pitch",
      "Right_Elbow_Yaw",
      # 22–23: testa
      "AAHead_yaw",
      "Head_pitch",
    ],
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    output_name=output_name,
    render=render,
    line_range=line_range,
    renderer=renderer,
    wandb_project=wandb_project,
    wandb_entity=wandb_entity,
  )


def main(
  input_file: str | None = None,
  output_name: str | None = None,
  root_dir: str | None = None,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = True,
  line_range: tuple[int, int] | None = None,
  wandb_project: str = "csv_to_npz",
  wandb_entity: str | None = None,
):
  """Replay motion from CSV file and output to npz file.

  Args:
    input_file: Path to one input CSV file.
    output_name: Optional artifact name. Defaults to input CSV stem.
    root_dir: Optional folder; if set, process all *.csv files recursively.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    render: Whether to render the simulation and save a video.
      Default is True; disable with `--no-render`.
    line_range: Range of lines to process from the CSV file.
    wandb_project: W&B project name used for upload.
    wandb_entity: Optional W&B entity. If unset, WANDB_ENTITY env var is used.
  """
  if (input_file is None) == (root_dir is None):
    raise ValueError("Provide exactly one of `input_file` or `root_dir`.")

  if root_dir is not None and output_name is not None:
    raise ValueError("`output_name` cannot be used with `root_dir` batch mode.")

  if root_dir is not None and line_range is not None:
    raise ValueError("`line_range` is only supported in single-file mode.")

  if root_dir is not None:
    csv_files = sorted(Path(root_dir).rglob("*.csv"))
    if not csv_files:
      raise FileNotFoundError(f"No CSV files found in root_dir: {root_dir}")

    print(f"[INFO] Found {len(csv_files)} CSV file(s) under {root_dir}")
    uploaded = 0

    for csv_path in csv_files:
      collection_name = csv_path.stem
      print(f"\n[INFO] Processing: {csv_path} -> {collection_name}")

      _run_single_csv(
        input_file=str(csv_path),
        output_name=collection_name,
        input_fps=input_fps,
        output_fps=output_fps,
        device=device,
        render=render,
        line_range=None,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
      )
      uploaded += 1

    print(f"\n[INFO] Done. uploaded={uploaded}, total={len(csv_files)}")
    return

  assert input_file is not None
  input_path = Path(input_file)
  if not input_path.is_file():
    raise FileNotFoundError(f"Input CSV not found: {input_file}")

  collection_name = output_name or input_path.stem
  print(f"[INFO] Processing: {input_path} -> {collection_name}")

  _run_single_csv(
    input_file=str(input_path),
    output_name=collection_name,
    input_fps=input_fps,
    output_fps=output_fps,
    device=device,
    render=render,
    line_range=line_range,
    wandb_project=wandb_project,
    wandb_entity=wandb_entity,
  )


if __name__ == "__main__":
  tyro.cli(main)
