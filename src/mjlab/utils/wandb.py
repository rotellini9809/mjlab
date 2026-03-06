"""WandB utilities."""

from __future__ import annotations

import os
from typing import Sequence


def _infer_wandb_entity() -> str:
  """Best-effort default entity resolution."""
  entity = os.environ.get("WANDB_ENTITY", "").strip()
  if entity:
    return entity
  try:
    import wandb

    api = wandb.Api()
    inferred = getattr(api, "default_entity", "") or ""
    return str(inferred).strip()
  except Exception:
    return ""


def add_wandb_tags(tags: Sequence[str]) -> None:
  """Add tags to the current wandb run.

  Note: This function stores tags in wandb.config._wandb_tags if the run is not yet
  initialized, allowing them to be retrieved later. If the run is already initialized,
  tags are added directly.
  """
  if not tags:
    return

  try:
    import wandb

    if wandb.run is not None:
      existing_tags = list(wandb.run.tags) if wandb.run.tags else []
      new_tags = list(set(existing_tags + list(tags)))
      wandb.run.tags = new_tags
    else:
      # Store tags to be added when run is initialized.
      # This is a workaround for lazy wandb initialization in rsl_rl 3.1.0.
      current_tags = os.environ.get("WANDB_TAGS", "")
      all_tags = set(current_tags.split(",") if current_tags else [])
      all_tags.update(tags)
      os.environ["WANDB_TAGS"] = ",".join(sorted(all_tags))
  except ImportError:
    pass


def resolve_artifact_path(name: str) -> str:
  """Resolve a W&B artifact name to a fully-qualified path.

  Accepted inputs:
  - "entity/project/artifact[:alias]" (already fully-qualified)
  - "project/artifact[:alias]" (uses WANDB_ENTITY or API default entity)
  - "artifact[:alias]" (uses WANDB_ENTITY + WANDB_PROJECT, with sane defaults)
  """
  if not name:
    raise ValueError("registry_name cannot be empty.")

  base = name.split(":", 1)[0]
  parts = base.split("/")
  if len(parts) >= 3:
    return name

  entity = _infer_wandb_entity()
  if len(parts) == 2:
    if not entity:
      raise ValueError(
        "registry_name is missing entity. Set WANDB_ENTITY or pass "
        "'entity/project/artifact[:alias]'."
      )
    return f"{entity}/{name}"

  # len(parts) == 1
  project = (
    os.environ.get("WANDB_PROJECT")
    or os.environ.get("WANDB_REGISTRY_PROJECT")
    or "csv_to_npz"
  )
  if not entity:
    raise ValueError(
      "registry_name is missing entity. Set WANDB_ENTITY or pass "
      "'entity/project/artifact[:alias]'."
    )
  return f"{entity}/{project}/{name}"
