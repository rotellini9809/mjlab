"""
Run the tracking policy from a W&B run for half an episode and print the
robot's actual joint positions, root position and root quaternion at that point.

Run inside the mjlab docker (from /app):
  python /app/inspect_policy_midpose.py \
    --wandb-run-path fratelligpt-sapienza-universit-di-roma/mjlab/k0zgfxdw

Optionally override the time fraction to inspect:
  python /app/inspect_policy_midpose.py \
    --wandb-run-path fratelligpt-sapienza-universit-di-roma/mjlab/k0zgfxdw \
    --frac 0.5
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--wandb-run-path", required=True)
parser.add_argument("--frac", type=float, default=0.5, help="Episode fraction to inspect (0=start, 0.5=mid, 1=end)")
parser.add_argument("--device", default=None)
args = parser.parse_args()

device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

import mjlab.tasks  # noqa: F401 – populate registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

configure_torch_backends()

task_id = "Mjlab-Tracking-Flat-Booster-T1_23"
env_cfg = load_env_cfg(task_id, play=True)
agent_cfg = load_rl_cfg(task_id)

# Resolve motion artifact from the run.
import wandb
api = wandb.Api()
wandb_run = api.run(args.wandb_run_path)
art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
if art is None:
    raise RuntimeError("No motion artifact found in the run.")
motion_path = Path(art.download()) / "motion.npz"
print(f"[INFO] Motion file: {motion_path}")

motion_cmd = env_cfg.commands["motion"]
assert isinstance(motion_cmd, MotionCommandCfg)
motion_cmd.motion_file = str(motion_path)
# Force start from the beginning of the clip.
motion_cmd.sampling_mode = "start"

# Use a single env, no randomization for a clean read.
env_cfg.scene.num_envs = 1

# Resolve checkpoint.
log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
resume_path, _ = get_wandb_checkpoint_path(log_root_path, Path(args.wandb_run_path), None)
print(f"[INFO] Checkpoint: {resume_path}")

# Build env.
env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

# Figure out the total motion steps so we know when we're at the midpoint.
inner_env = env.unwrapped
motion_term = inner_env.command_manager.get_term("motion")
total_steps = int(motion_term.motion.time_step_total)
target_step = int(round(args.frac * (total_steps - 1)))
print(f"[INFO] Motion total frames: {total_steps}, target frame: {target_step} (frac={args.frac})")

# Reset and step until we reach the target frame.
obs, _ = env.reset()
for step in range(target_step):
    with torch.no_grad():
        actions = policy(obs)
    step_out = env.step(actions)
    obs = step_out[0]

# Read the robot state.
robot = inner_env.scene["robot"]
joint_pos = robot.data.joint_pos[0].cpu().numpy()
root_pos = robot.data.root_link_pos_w[0].cpu().numpy()
root_quat = robot.data.root_link_quat_w[0].cpu().numpy()

print(f"\n{'='*60}")
print(f"Robot state at episode fraction {args.frac} (motion frame {target_step}/{total_steps - 1})")
print(f"{'='*60}")
print(f"\nroot_pos  (x, y, z): {root_pos.tolist()}")
print(f"  -> z (height): {root_pos[2]:.4f} m")
print(f"\nroot_quat (w, x, y, z): {root_quat.tolist()}")
print(f"\njoint_pos ({len(joint_pos)} dof):")
joint_names = robot.joint_names
for name, val in zip(joint_names, joint_pos):
    print(f"  {name:40s}: {val:+.6f}")

print(f"\n--- Compact copy-paste form ---")
print(f"KEEPER_SPAWN_Z = {root_pos[2]:.4f}")
print(f"READY_JOINT_POS = {joint_pos.tolist()}")

env.close()
