#!/bin/bash
# Nightly training script for MJLab benchmarks
#
# This script runs the tracking benchmark and generates a report.
# It is designed to be called by a systemd timer or cron job.
#
# Usage:
#   ./scripts/benchmarks/nightly_train.sh
#
# Environment variables:
#   MJLAB_DIR: Path to mjlab repository (default: script directory)
#   CUDA_DEVICE: GPU device to use (default: 0)
#   WANDB_TAGS: Comma-separated tags for the run (default: nightly)
#   SKIP_TRAINING: Set to "1" to skip training and only generate report

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MJLAB_DIR="${MJLAB_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WANDB_TAGS="${WANDB_TAGS:-nightly}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"

# Training configuration
TASK="Mjlab-Tracking-Flat-Unitree-G1"
NUM_ENVS=4096
MAX_ITERATIONS=6000
REGISTRY_NAME="rll_humanoid/wandb-registry-Motions/side_kick_test"

GH_PAGES_BRANCH="gh-pages"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

# Use a worktree for training to avoid touching the main repo
WORKTREE_DIR="/tmp/mjlab-nightly-$$"
REPORT_DIR="${WORKTREE_DIR}/benchmark_results"

cleanup() {
    if [[ -d "$WORKTREE_DIR" ]]; then
        log "Cleaning up worktree..."
        git -C "$MJLAB_DIR" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
    fi
}
trap cleanup EXIT

cd "$MJLAB_DIR" || error "Failed to cd to $MJLAB_DIR"

log "Starting nightly benchmark run"
log "Task: $TASK"
log "GPU: $CUDA_DEVICE"

# Fetch latest and create worktree
git fetch origin main || log "Warning: Failed to fetch origin/main"
git worktree prune
git worktree add "$WORKTREE_DIR" origin/main --detach || error "Failed to create worktree"
cd "$WORKTREE_DIR"

log "Running on commit: $(git rev-parse HEAD)"

# Run training
if [[ "$SKIP_TRAINING" != "1" ]]; then
    log "Starting training..."

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" uv run train "$TASK" \
        --env.scene.num-envs "$NUM_ENVS" \
        --agent.max-iterations "$MAX_ITERATIONS" \
        --registry-name "$REGISTRY_NAME" \
        --agent.wandb-tags "$WANDB_TAGS"

    log "Training completed"
else
    log "Skipping training (SKIP_TRAINING=1)"
fi

# Generate report
log "Generating benchmark report..."
uv run python scripts/benchmarks/generate_report.py \
    --entity gcbc_researchers \
    --tag nightly \
    --output-dir "$REPORT_DIR"

log "Report generated at $REPORT_DIR"

# Deploy to GitHub Pages using a separate worktree
log "Deploying to GitHub Pages..."

GH_PAGES_WORKTREE="/tmp/mjlab-gh-pages-$$"
cleanup_gh_pages() {
    if [[ -d "$GH_PAGES_WORKTREE" ]]; then
        git -C "$MJLAB_DIR" worktree remove --force "$GH_PAGES_WORKTREE" 2>/dev/null || rm -rf "$GH_PAGES_WORKTREE"
    fi
}

# Checkout gh-pages in separate worktree
if git -C "$MJLAB_DIR" ls-remote --exit-code --heads origin "$GH_PAGES_BRANCH" > /dev/null 2>&1; then
    git -C "$MJLAB_DIR" fetch origin "$GH_PAGES_BRANCH"
    git -C "$MJLAB_DIR" worktree add "$GH_PAGES_WORKTREE" "origin/$GH_PAGES_BRANCH" || { log "Failed to create gh-pages worktree"; cleanup_gh_pages; exit 0; }
    cd "$GH_PAGES_WORKTREE"
    git checkout -B "$GH_PAGES_BRANCH"
else
    # Create new orphan gh-pages branch
    git -C "$MJLAB_DIR" worktree add --detach "$GH_PAGES_WORKTREE" HEAD || { log "Failed to create gh-pages worktree"; cleanup_gh_pages; exit 0; }
    cd "$GH_PAGES_WORKTREE"
    git checkout --orphan "$GH_PAGES_BRANCH"
    git reset --hard
    git clean -fdx
fi

# Copy report files
mkdir -p nightly
cp -r "$REPORT_DIR"/* nightly/

# Commit and push
git add -A
git commit -m "Update nightly benchmarks $(date '+%Y-%m-%d')" || log "No changes to commit"
git push origin "$GH_PAGES_BRANCH" || log "Failed to push"

log "Deployed to GitHub Pages"

cd "$MJLAB_DIR"
cleanup_gh_pages

log "Nightly benchmark complete"
