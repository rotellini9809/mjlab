from __future__ import annotations

import numpy as np
import torch

_REMOVED_TERMS = ("command", "motion_anchor_pos_b", "motion_anchor_ori_b")


def _concat_terms(values: list[torch.Tensor] | list[np.ndarray]) -> torch.Tensor | np.ndarray:
  if not values:
    raise ValueError("No observation terms to concatenate.")
  first = values[0]
  if torch.is_tensor(first):
    return torch.cat(values, dim=-1)
  return np.concatenate(values, axis=-1)


def _build_term_slices(
  term_order: list[str], term_dims: list[tuple[int, ...]]
) -> tuple[list[dict[str, object]], int]:
  slices: list[dict[str, object]] = []
  cursor = 0
  for name, dims in zip(term_order, term_dims, strict=False):
    size = int(np.prod(dims)) if len(dims) > 0 else 0
    start = cursor
    end = cursor + size
    slices.append(
      {
        "name": name,
        "start": start,
        "end": end,
        "shape": list(dims),
        "size": size,
      }
    )
    cursor = end
  return slices, cursor


def _fallback_strip_anchors(
  obs: torch.Tensor | np.ndarray, obs_meta: dict[str, object] | None
) -> tuple[torch.Tensor | np.ndarray, dict[str, object]]:
  meta: dict[str, object] = {
    "anchors_stripped": False,
    "method": "fallback",
  }
  act_dim = None if obs_meta is None else obs_meta.get("act_dim")
  if act_dim is None:
    meta["reason"] = "no_act_dim"
    return obs, meta

  teacher_dim = int(obs.shape[-1])
  act_dim = int(act_dim)
  fixed = 3 + 6 + 3 + 3 + 3 * act_dim
  cmd_dim = teacher_dim - fixed
  if cmd_dim < 0:
    meta["reason"] = "dim_mismatch"
    return obs, meta

  order = [
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  ]
  sizes = [cmd_dim, 3, 6, 3, 3, act_dim, act_dim, act_dim]
  if sizes[1] != 3 or sizes[2] != 6:
    raise RuntimeError("Fallback anchor sizes are unexpected (pos=3, ori=6).")
  term_dims = [(s,) for s in sizes]
  slices, _ = _build_term_slices(order, term_dims)
  kept_slices = [s for s in slices if s["name"] not in _REMOVED_TERMS]

  if torch.is_tensor(obs):
    student = torch.cat(
      [obs[..., int(s["start"]):int(s["end"])] for s in kept_slices], dim=-1
    )
  else:
    student = np.concatenate(
      [obs[..., int(s["start"]):int(s["end"])] for s in kept_slices], axis=-1
    )

  meta.update(
    {
      "anchors_stripped": True,
      "features_stripped": True,
      "teacher_obs_dim": teacher_dim,
      "student_obs_dim": int(student.shape[-1]),
      "removed_terms": list(_REMOVED_TERMS),
      "kept_terms": [s["name"] for s in kept_slices],
      "slice_map": slices,
      "method": "fallback_tracking_order",
    }
  )
  return student, meta


def build_student_obs(
  obs: torch.Tensor | np.ndarray | dict[str, torch.Tensor] | dict[str, np.ndarray],
  obs_meta: dict[str, object] | None = None,
) -> tuple[torch.Tensor | np.ndarray, dict[str, object]]:
  removed_terms = _REMOVED_TERMS
  term_order = None if obs_meta is None else obs_meta.get("term_order")
  term_dims = None if obs_meta is None else obs_meta.get("term_dims")

  meta: dict[str, object] = {
    "anchors_stripped": False,
    "removed_terms": list(removed_terms),
    "method": "none",
  }

  if isinstance(obs, dict):
    order = list(term_order) if isinstance(term_order, list) else list(obs.keys())
    removed_present = [term for term in removed_terms if term in obs]
    kept_keys = [k for k in order if k in obs and k not in removed_terms]
    all_keys = [k for k in order if k in obs]
    teacher_dim = int(sum(int(obs[k].shape[-1]) for k in all_keys))
    student = _concat_terms([obs[k] for k in kept_keys])
    student_dim = int(student.shape[-1])

    slice_map = []
    if isinstance(term_dims, list) and isinstance(term_order, list):
      slice_map, _ = _build_term_slices(term_order, term_dims)

    meta.update(
      {
        "anchors_stripped": bool(removed_present),
        "features_stripped": bool(removed_present),
        "teacher_obs_dim": teacher_dim,
        "student_obs_dim": student_dim,
        "removed_terms_present": removed_present,
        "kept_terms": kept_keys,
        "slice_map": slice_map,
        "method": "dict_terms",
      }
    )
    return student, meta

  teacher_dim = int(obs.shape[-1])
  if isinstance(term_order, list) and isinstance(term_dims, list):
    slices, total = _build_term_slices(term_order, term_dims)
    if total != teacher_dim:
      meta.update({"teacher_obs_dim": teacher_dim, "method": "meta_mismatch"})
      return obs, meta

    removed_present = [s["name"] for s in slices if s["name"] in removed_terms]
    if not removed_present:
      meta.update(
        {
          "teacher_obs_dim": teacher_dim,
          "student_obs_dim": teacher_dim,
          "removed_terms_present": [],
          "kept_terms": [s["name"] for s in slices],
          "slice_map": slices,
        }
      )
      return obs, meta

    kept_slices = [s for s in slices if s["name"] not in removed_terms]
    if torch.is_tensor(obs):
      student = torch.cat(
        [obs[..., int(s["start"]):int(s["end"])] for s in kept_slices], dim=-1
      )
    else:
      student = np.concatenate(
        [obs[..., int(s["start"]):int(s["end"])] for s in kept_slices], axis=-1
      )

    meta.update(
      {
        "anchors_stripped": True,
        "features_stripped": True,
        "teacher_obs_dim": teacher_dim,
        "student_obs_dim": int(student.shape[-1]),
        "removed_terms_present": removed_present,
        "kept_terms": [s["name"] for s in kept_slices],
        "slice_map": slices,
        "method": "meta_slices",
      }
    )
    return student, meta

  student, fb_meta = _fallback_strip_anchors(obs, obs_meta)
  if "teacher_obs_dim" not in fb_meta:
    fb_meta["teacher_obs_dim"] = teacher_dim
    fb_meta["student_obs_dim"] = int(student.shape[-1])
  if "removed_terms_present" not in fb_meta:
    fb_meta["removed_terms_present"] = []
  return student, fb_meta
