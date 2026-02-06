#!/usr/bin/env bash
set -euo pipefail

# ============================
# CONFIG (edit here once)
# ============================

# Rollout source run (tracking policy run on W&B)
# (For now set to supersad. Later you can change it or run the script twice.)
ROLLOUT_RUN_PATH_DEFAULT="fratelligpt-sapienza-universit-di-roma/mjlab/2rd8me48"

# Task used ONLY for rollout collection
ROLLOUT_TASK_ID_DEFAULT="Mjlab-Tracking-Flat-Booster-T1_23"

# Stage-1 trained run (eval/play)
STAGE1_RUN_PATH_DEFAULT="fratelligpt-sapienza-universit-di-roma/motor_controller_stage1/ny89vv0x"

# Dataset root (collector creates a subfolder per run automatically)
DATASET_ROOT_DEFAULT="./data/motor_controller_rollouts/dataset_v1"

# Default params
ROLLOUT_NUM_ENVS_DEFAULT="64"
ROLLOUT_NUM_EPISODES_DEFAULT="200"

TRAIN_MAX_ITERS_DEFAULT="5000"
TRAIN_CHUNK_LEN_DEFAULT="32"
TRAIN_K_FUTURE_DEFAULT="10"
TRAIN_VAL_FRAC_DEFAULT="0.1"
TRAIN_SEED_DEFAULT="0"
TRAIN_RUN_NAME_DEFAULT="stage1_npmp_chunk32_k10_seed0"
TRAIN_LOG_EVERY_DEFAULT="50"
TRAIN_BETA_KL_END_DEFAULT="1e-3"
TRAIN_BETA_KL_WARMUP_ITERS_DEFAULT="2000"

EVAL_NUM_STEPS_DEFAULT="2000"
EVAL_NUM_ENVS_DEFAULT="32"

# Switches
RUN_ROLLOUTS_DEFAULT="1"
RUN_TRAIN_DEFAULT="0"
RUN_EVAL_DEFAULT="1"
RUN_PLAY_DEFAULT="0"


# ============================
# RUNTIME OVERRIDES (optional)
# ============================
# You can override any of these by prefixing env vars when calling the script.

ROLLOUT_RUN_PATH="${ROLLOUT_RUN_PATH:-$ROLLOUT_RUN_PATH_DEFAULT}"
ROLLOUT_TASK_ID="${ROLLOUT_TASK_ID:-$ROLLOUT_TASK_ID_DEFAULT}"
STAGE1_RUN_PATH="${STAGE1_RUN_PATH:-$STAGE1_RUN_PATH_DEFAULT}"
DATASET_ROOT="${DATASET_ROOT:-$DATASET_ROOT_DEFAULT}"

ROLLOUT_NUM_ENVS="${ROLLOUT_NUM_ENVS:-$ROLLOUT_NUM_ENVS_DEFAULT}"
ROLLOUT_NUM_EPISODES="${ROLLOUT_NUM_EPISODES:-$ROLLOUT_NUM_EPISODES_DEFAULT}"

TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-$TRAIN_MAX_ITERS_DEFAULT}"
TRAIN_CHUNK_LEN="${TRAIN_CHUNK_LEN:-$TRAIN_CHUNK_LEN_DEFAULT}"
TRAIN_K_FUTURE="${TRAIN_K_FUTURE:-$TRAIN_K_FUTURE_DEFAULT}"
TRAIN_VAL_FRAC="${TRAIN_VAL_FRAC:-$TRAIN_VAL_FRAC_DEFAULT}"
TRAIN_SEED="${TRAIN_SEED:-$TRAIN_SEED_DEFAULT}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-$TRAIN_RUN_NAME_DEFAULT}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-$TRAIN_LOG_EVERY_DEFAULT}"
TRAIN_BETA_KL_END="${TRAIN_BETA_KL_END:-$TRAIN_BETA_KL_END_DEFAULT}"
TRAIN_BETA_KL_WARMUP_ITERS="${TRAIN_BETA_KL_WARMUP_ITERS:-$TRAIN_BETA_KL_WARMUP_ITERS_DEFAULT}"

EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-$EVAL_NUM_STEPS_DEFAULT}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-$EVAL_NUM_ENVS_DEFAULT}"

RUN_ROLLOUTS="${RUN_ROLLOUTS:-$RUN_ROLLOUTS_DEFAULT}"
RUN_TRAIN="${RUN_TRAIN:-$RUN_TRAIN_DEFAULT}"
RUN_EVAL="${RUN_EVAL:-$RUN_EVAL_DEFAULT}"
RUN_PLAY="${RUN_PLAY:-$RUN_PLAY_DEFAULT}"

# ============================
# ENV
# ============================
cd /app

export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Force W&B online (avoid offline runs)
wandb online || true
unset WANDB_MODE WANDB_DISABLED
export WANDB_MODE=online

echo "=============================="
echo "[INFO] ROLLOUT_RUN_PATH = $ROLLOUT_RUN_PATH"
echo "[INFO] ROLLOUT_TASK_ID  = $ROLLOUT_TASK_ID"
echo "[INFO] DATASET_ROOT     = $DATASET_ROOT"
echo "[INFO] STAGE1_RUN_PATH  = $STAGE1_RUN_PATH"
echo "[INFO] switches         : rollouts=$RUN_ROLLOUTS train=$RUN_TRAIN eval=$RUN_EVAL play=$RUN_PLAY"
echo "=============================="

mkdir -p "$DATASET_ROOT"

# IMPORTANT: do NOT override MJLAB_MOTOR_CONTROLLER_TASK_ID unless you verified compatibility.
unset MJLAB_MOTOR_CONTROLLER_TASK_ID || true


# ============================
# 1) ROLLOUTS
# ============================
if [[ "$RUN_ROLLOUTS" == "1" ]]; then
  echo "[STEP 1] Collect rollouts..."
  uv run collect-rollouts "$ROLLOUT_TASK_ID" \
    --wandb-run-path "$ROLLOUT_RUN_PATH" \
    --output-dir "$DATASET_ROOT" \
    --num-envs "$ROLLOUT_NUM_ENVS" \
    --num-episodes "$ROLLOUT_NUM_EPISODES"

  echo "[STEP 1] Dataset sanity check..."
  find "$DATASET_ROOT" -maxdepth 4 -type f -name "*.npz" | head
  find "$DATASET_ROOT" -maxdepth 4 -type f -name "metadata.json" | head
else
  echo "[STEP 1] Skipped rollouts (RUN_ROLLOUTS=0)"
fi


# ============================
# 2) TRAIN (optional)
# ============================
if [[ "$RUN_TRAIN" == "1" ]]; then
  echo "[STEP 2] Train Stage-1..."
  uv run train-motor-stage1 \
    --data-root "$DATASET_ROOT" \
    --latent-type npmp \
    --sample-mode chunk \
    --chunk-len "$TRAIN_CHUNK_LEN" \
    --k-future "$TRAIN_K_FUTURE" \
    --max-iters "$TRAIN_MAX_ITERS" \
    --val-frac "$TRAIN_VAL_FRAC" \
    --seed "$TRAIN_SEED" \
    --run-name "$TRAIN_RUN_NAME" \
    --beta-kl-end "$TRAIN_BETA_KL_END" \
    --beta-kl-warmup-iters "$TRAIN_BETA_KL_WARMUP_ITERS" \
    --log-every "$TRAIN_LOG_EVERY" \
    --wandb \
    --wandb-project motor_controller_stage1 \
    --wandb-entity fratelligpt-sapienza-universit-di-roma
else
  echo "[STEP 2] Skipped training (RUN_TRAIN=0)"
fi


# ============================
# 3) EVAL
# ============================
if [[ "$RUN_EVAL" == "1" ]]; then
  echo "[STEP 3] Eval Stage-1..."
  uv run eval-motor-stage1 \
    --wandb-run-path "$STAGE1_RUN_PATH" \
    --num-steps "$EVAL_NUM_STEPS" \
    --num-envs "$EVAL_NUM_ENVS" \
    --checkpoint best
else
  echo "[STEP 3] Skipped eval (RUN_EVAL=0)"
fi


# ============================
# 4) PLAY
# ============================
if [[ "$RUN_PLAY" == "1" ]]; then
  echo "[STEP 4] Play Stage-1..."
  VIEWER="${VIEWER:-native}"
  uv run play-motor-stage1 \
    --wandb-run-path "$STAGE1_RUN_PATH" \
    --viewer "$VIEWER"
else
  echo "[STEP 4] Skipped play (RUN_PLAY=0)"
fi

echo "[DONE]"
