# Observation History and Delay

Observations have two temporal features: history and delay. History stacks past
frames for temporal context, while delay can be used to model sensor latency.

## TL;DR

**Add history to stack frames:**
```python
from mjlab.managers.manager_term_config import ObservationTermCfg

joint_vel: ObservationTermCfg = ObservationTermCfg(
  func=joint_vel,
  history_length=5,        # Keep last 5 frames
  flatten_history_dim=True # Flatten for MLP: (12,) * 5 = (60,)
)
```

**Add delay to model sensor latency:**
```python
# At 50Hz control (20ms/step): lag=2-3 → 40-60ms latency
camera: ObservationTermCfg = ObservationTermCfg(
  func=camera_obs,
  delay_min_lag=2,
  delay_max_lag=3,
)
```

**Combine both:**
```python
joint_pos: ObservationTermCfg = ObservationTermCfg(
  func=joint_pos,
  delay_min_lag=1,
  delay_max_lag=3,      # Delayed observations
  history_length=5,     # Stack 5 delayed frames
  flatten_history_dim=True
)
# Pipeline: compute → delay → stack → flatten
```

## Observation History

History stacks past observations to provide temporal context.

### Basic Usage

**Flattened history (for MLPs):**
```python
joint_vel: ObservationTermCfg = ObservationTermCfg(
  func=joint_vel,           # Returns (num_envs, 12)
  history_length=3,
  flatten_history_dim=True  # Output: (num_envs, 36)
)
```

**Structured history (for RNNs):**
```python
joint_vel: ObservationTermCfg = ObservationTermCfg(
  func=joint_vel,            # Returns (num_envs, 12)
  history_length=3,
  flatten_history_dim=False  # Output: (num_envs, 3, 12)
)
```

### Group-Level Override

Apply history to all terms in a group:

```python
@dataclass
class PolicyCfg(ObservationGroupCfg):
  concatenate_terms: bool = True
  history_length: int = 5           # Applied to all terms
  flatten_history_dim: bool = True

  joint_pos: ObservationTermCfg = ObservationTermCfg(func=joint_pos)
  joint_vel: ObservationTermCfg = ObservationTermCfg(func=joint_vel)
  # Both terms get 5-frame history, flattened
```

Term-level settings override group settings:

```python
@dataclass
class PolicyCfg(ObservationGroupCfg):
  history_length: int = 3  # Default for group

  joint_pos: ObservationTermCfg = ObservationTermCfg(
    func=joint_pos,
    history_length=5  # Override: use 5 instead of 3
  )
```

### Reset Behavior

History buffers are cleared on environment reset. The first observation after
reset is backfilled across all history slots, ensuring valid data from step 0.

```python
# At reset
buffer = [obs_0, obs_0, obs_0]  # Backfilled

# After 2 steps
buffer = [obs_0, obs_1, obs_2]  # Normal accumulation
```

## Observation Delay

Real robots have communication delays (WiFi, USB) and sensors that refresh at
different rates (30Hz camera, 100Hz encoders). Delay models this behavior.

### Delay Parameters

**`delay_min_lag`** / **`delay_max_lag`** (default: 0) Lag range in steps.
Sampled uniformly each update. `lag=0` means current observation, `lag=2` means
2 steps ago.

**`delay_per_env`** (default: True) If True, each environment gets a different
lag. If False, all environments share the same lag.

**`delay_hold_prob`** (default: 0.0)
Probability [0, 1] of keeping the previous lag instead of resampling.

**`delay_update_period`** (default: 0) If > 0, resample lags every N steps
(models multi-rate sensors). If 0, resample every step.

**`delay_per_env_phase`** (default: True) If True, each environment has a
different phase offset for update period (staggers refresh times).

### Computing Delays from Real-World Latency

Convert real-world latency to simulation steps:

```
delay_steps = latency_ms / (1000 / control_hz)
```

**Example at 50Hz control (20ms per step):**
- 40ms latency = 40 / 20 = 2 steps
- 60ms latency = 60 / 20 = 3 steps
- 100ms latency = 100 / 20 = 5 steps

**Example at 100Hz control (10ms per step):**
- 40ms latency = 40 / 10 = 4 steps
- 60ms latency = 60 / 10 = 6 steps

> **Note:** Delays are quantized to multiples of the control timestep. At 50Hz
> (20ms/step), you can only represent delays of 0ms, 20ms, 40ms, etc. To
> approximate a 45ms sensor, use a range like `delay_min_lag=2, delay_max_lag=3`
> to stochastically vary between 40ms and 60ms.

### Computing Multi-Rate Updates

Convert sensor refresh rate to update period:

```
update_period = control_hz / sensor_hz
```

**Example at 50Hz control:**
- 30Hz camera: update_period = 50 / 30 ≈ 2 steps
- 25Hz LiDAR: update_period = 50 / 25 = 2 steps
- 10Hz GPS: update_period = 50 / 10 = 5 steps

**Example at 100Hz control:**
- 30Hz camera: update_period = 100 / 30 ≈ 3 steps
- 50Hz IMU: update_period = 100 / 50 = 2 steps

### Examples

**Joint encoders (100Hz, no delay) at 50Hz control:**
```python
joint_pos: ObservationTermCfg = ObservationTermCfg(func=joint_pos)
# delay_min_lag=delay_max_lag=0 by default.
```

**Camera (30Hz, 40-60ms latency) at 50Hz control:**
```python
# 30Hz camera ≈ update every 2 steps (50Hz/2 = 25Hz, approximates 30Hz)
# 40-60ms latency = 2-3 steps at 50Hz (20ms/step)
camera: ObservationTermCfg = ObservationTermCfg(
  func=camera_obs,
  delay_min_lag=2,          # 40ms
  delay_max_lag=3,          # 60ms
  delay_update_period=2,    # ~30Hz refresh (actually 25Hz)
  delay_per_env_phase=True  # Staggered refresh across envs
)
```

**Mixed-rate system at 50Hz control:**
```python
@dataclass
class PolicyCfg(ObservationGroupCfg):
  # Fast encoders (no delay)
  joint_pos: ObservationTermCfg = ObservationTermCfg(
    func=joint_pos,
    # delay_min_lag=0, delay_max_lag=0 (default)
  )

  # 25Hz camera (40-80ms latency)
  camera: ObservationTermCfg = ObservationTermCfg(
    func=camera_obs,
    delay_min_lag=2,  # 40ms
    delay_max_lag=4,  # 80ms
    delay_update_period=2  # 25Hz (50Hz control / 2)
  )
```

## Processing Pipeline

Observations flow through this pipeline:

```
compute → noise → clip → scale → delay → history → flatten
```

**Why delay before history?** History stacks delayed observations. This models
real systems where you buffer old sensor readings, not future ones.

Example with both:
```python
joint_vel: ObservationTermCfg = ObservationTermCfg(
  func=joint_vel,
  scale=0.1,             # Scale raw values
  delay_min_lag=1,       # 20ms delay at 50Hz
  delay_max_lag=2,       # 40ms delay at 50Hz
  history_length=3,      # Stack 3 delayed frames
  flatten_history_dim=True
)
# Pipeline:
# 1. compute() returns (num_envs, 12)
# 2. scale: multiply by 0.1
# 3. delay: return observation from 1-2 steps ago
# 4. history: stack last 3 delayed frames → (num_envs, 3, 12)
# 5. flatten: reshape → (num_envs, 36)
```

## Performance

Delay buffers are only created when `delay_max_lag > 0`. Terms with no delay
(the default) have zero overhead. Similarly, history buffers are only created
when `history_length > 0`.
