# Distributed Training

Distributed training parallelizes RL workloads across multiple GPUs by running
independent rollouts on each device and synchronizing gradients after each
training iteration. Throughput scales nearly linearly with GPU count.

## TL;DR

Launch distributed training using `torchrun`:

```bash
uv run torchrun \
  --nproc_per_node=N \
  --no_python \
  train <task-name> \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 10_000 \
    --distributed True \
    <additional task-specific CLI args>
```

**Key points:**
- `--nproc_per_node=N` spawns N processes (one per GPU).
- `--no_python` is required when using `torchrun` to launch a console script
  (like `train`). Without it, torchrun tries to execute `python train`, which
  fails because `train` is not a Python file.
- `--distributed True` enables gradient synchronization in the training script
- Each GPU runs the full `num-envs` count (e.g., 2 GPUs × 4096 envs = 8192
  total)

**Tracking task example:**

```bash
uv run torchrun \
  --nproc_per_node=2 \
  --no_python \
  train Mjlab-Tracking-Flat-Unitree-G1 \
    --registry-name rll_humanoid/wandb-registry-Motions/lafan_cartwheel \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 10_000 \
    --distributed True
```

## How It Works

**Process isolation.** `torchrun` spawns N independent processes (one per GPU)
and sets environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) to
coordinate them. Each process executes the full training script with its
assigned GPU.

**Seed diversity.** Each process uses `seed = cfg.seed + local_rank` to ensure
different random experiences across GPUs, increasing sample diversity.

**Independent rollouts.** Each process maintains its own:
- MuJoco Warp environment (with `num-envs` parallel environments), fully
  isolated on its assigned GPU
- Policy network copy
- Experience buffer (sized `num_steps_per_env × num-envs`)

The mjwarp simulation backend wraps all operations in `wp.ScopedDevice` to
ensure complete GPU isolation. Every simulation call—initialization, stepping,
CUDA graph creation, and execution—happens within the device scope:

```python
# From mjlab/sim/sim.py
def step(self) -> None:
  with wp.ScopedDevice(self.wp_device):
    if self.use_cuda_graph and self.step_graph is not None:
      wp.capture_launch(self.step_graph)
    else:
      mjwarp.step(self.wp_model, self.wp_data)

def forward(self) -> None:
  with wp.ScopedDevice(self.wp_device):
    if self.use_cuda_graph and self.forward_graph is not None:
      wp.capture_launch(self.forward_graph)
    else:
      mjwarp.forward(self.wp_model, self.wp_data)
```

Simulation data is exposed to PyTorch via `TorchArray`, which provides
zero-copy memory sharing between Warp arrays and PyTorch tensors using
`wp.to_torch()`. This eliminates data transfer overhead between simulation and
policy networks.

**Synchronized initialization.** Before training begins, rank 0 broadcasts its
model parameters to all other ranks via `broadcast_object_list`, ensuring all
GPUs start with identical weights.

**Gradient synchronization.** After each rollout, all processes:
1. Compute policy updates independently on their local data
2. Call `backward()` to compute gradients
3. Synchronize gradients via `reduce_parameters()`:

```python
# From rsl_rl/algorithms/ppo.py
def reduce_parameters(self) -> None:
  """Collect gradients from all GPUs and average them."""
  # Flatten all gradients into a single tensor
  grads = [param.grad.view(-1) for param in self.policy.parameters()]
  all_grads = torch.cat(grads)

  # Average gradients across all GPUs
  torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
  all_grads /= self.gpu_world_size

  # Copy averaged gradients back to parameters
  # ... (gradient assignment)
```

4. Apply averaged gradients via `optimizer.step()`

This differs from `DistributedDataParallel` (DDP): gradients are manually
synchronized once per rollout (not per minibatch), giving fine-grained control.

**Adaptive learning rate.** For adaptive schedules, KL divergence is averaged
across GPUs via `all_reduce`, rank 0 computes the new learning rate, then
broadcasts it to all ranks to ensure consistent updates.

**Logging.** All ranks compute metrics locally, but only rank 0 (`global_rank
== 0`) writes to WandB and saves checkpoints, avoiding duplicate writes and
conflicts.

## Backend

RSL-RL uses PyTorch's `torch.distributed` with the NCCL backend for
GPU-to-GPU communication:

```python
# From rsl_rl/runners/on_policy_runner.py
torch.distributed.init_process_group(
  backend="nccl",
  rank=self.gpu_global_rank,
  world_size=self.gpu_world_size
)
```

NCCL (NVIDIA Collective Communications Library) provides optimized multi-GPU
primitives like `all_reduce` and `broadcast` for efficient gradient
synchronization.

## Logging

Only rank 0 logs to WandB and saves checkpoints:

```python
# From rsl_rl/runners/on_policy_runner.py
self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
```

All ranks compute metrics (rewards, episode lengths, etc.) for their local
environments, but non-zero ranks discard them. This avoids duplicate writes and
file conflicts. Total timesteps are correctly accounted for by multiplying by
`world_size`.

## Device Assignment

Each process assigns its device based on `LOCAL_RANK`:

```python
# From mjlab/scripts/train.py
if cfg.distributed:
  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  device = f"cuda:{local_rank}"

  # Offset seed for experience diversity
  seed = cfg.agent.seed + local_rank
  cfg.env.seed = seed
  cfg.agent.seed = seed

# Pass device to environment
env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
```

The device is propagated through the entire stack:

```python
# From mjlab/envs/manager_based_rl_env.py
self.scene = Scene(self.cfg.scene, device=device)
self.sim = Simulation(num_envs=..., cfg=..., model=..., device=device)

# From mjlab/sim/sim.py
self.device = device
self.wp_device = wp.get_device(self.device)

# All simulation operations wrapped in device scope
with wp.ScopedDevice(self.wp_device):
  self._wp_model = mjwarp.put_model(self._mj_model)
  self._wp_data = mjwarp.put_data(...)
```

This ensures rank 0 → `cuda:0`, rank 1 → `cuda:1`, etc. All simulation
kernels, tensors, CUDA graphs, and PyTorch tensors are isolated per device.
No cross-GPU communication occurs during rollouts—only during gradient synchronization.
