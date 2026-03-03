# Stage-1 Motor Controller (NPMP)

This folder contains Stage-1 (motor-focused) controller training code and data loading.

## Rollout Collection

Use `collect-rollouts` to collect supervision data from a trained tracking policy:

```bash
uv run collect-rollouts Mjlab-Tracking-Flat-Booster-T1_23 \
  --wandb-run-path your-org/mjlab/your-tracking-run-id \
  --output-dir ./data/motor_controller_rollouts/my_dataset \
  --num-envs 64 \
  --num-episodes 200 \
  --stage1-chunk-len-hint 32 \
  --stage1-k-future-hint 10 \
  --stage1-start-margin 4 \
  --noise-std 0.0
```

Collect all runs in a workspace group:

```bash
uv run collect-rollouts Mjlab-Tracking-Flat-Booster-T1_23 \
  --wandb-workspace your-org/mjlab \
  --wandb-group your_group_name \
  --output-dir ./data/motor_controller_rollouts/my_dataset \
  --num-envs 64 \
  --num-episodes 200
```

Collector behavior:

- Clip-aware random start:
  - `min_remaining = stage1_chunk_len_hint + stage1_k_future_hint + stage1_start_margin`
  - If clip length is at least `min_remaining`, start is sampled uniformly from valid starts.
  - Otherwise start is `0` (short clips are still kept).
- Clip-aware truncation:
  - Natural clip completion is marked as `truncated=true` with `done_reason="clip_end"`.
- Explicit boundary labels:
  - `terminated` for failures (`fall`, `invalid_state`, `nan`, ...)
  - `truncated` for non-failure boundaries (`clip_end`, `timeout`)
  - `done_reason` as explicit reason string.
- Noise handling:
  - Policy action before noise is saved as `a_clean` (supervision target).
  - Executed action is `a_exec = a_clean + noise`.

If `motion.npz` has no `clip_id`, collector treats the full trajectory as a single clip.

## Stage-1 Rollout Shard Schema

Each `rollouts_*.npz` shard stores per-step arrays. Required keys:

- `obs_student`
- `a_clean`
- `a_exec`
- `episode_id`
- `step_in_episode`
- `clip_id`
- `clip_len_steps`
- `phase_idx`
- `phase_norm`
- `steps_to_clip_end`
- `terminated`
- `truncated`
- `done_reason`

Common optional keys:

- `env_id`
- `noise_level_id`
- `noise_norm`
- `future_valid_len_hint`

`done` is not stored in shards anymore. In data loading, use:

```python
done = terminated | truncated
```

## Metadata Contract

Collector writes `metadata.json` next to shards with:

- `collector_version`
- `control_dt` (must match rollout env step dt; default collector check is `0.02`)
- `stage1_chunk_len_hint`
- `stage1_k_future_hint`
- `stage1_start_margin`
- noise config (`noise_std_default`, `noise_std_levels`, `noise_level_probs`)
- `obs_student_view`
- `obs_dim`
- `act_dim`

## Training

Train NPMP Stage-1:

```bash
uv run train-motor-stage1 \
  --data-root ./data/motor_controller_rollouts/my_dataset \
  --latent-type npmp \
  --sample-mode chunk \
  --chunk-len 32 \
  --k-future 10
```

The Stage-1 dataset loader expects `terminated` + `truncated` and computes internal `done = terminated | truncated`.
