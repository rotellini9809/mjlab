import re
from typing import TypeVar

T = TypeVar("T", int, float)


def resolve_expr(
  pattern_map: dict[str, T],
  names: tuple[str, ...],
  default_val: T = 0.0,
) -> tuple[T, ...]:
  """Resolve a field value (scalar or dict) to a tuple of values matched by patterns."""
  patterns = [(re.compile(pat), val) for pat, val in pattern_map.items()]

  result = []
  for name in names:
    for pat, val in patterns:
      if pat.match(name):
        result.append(val)
        break
    else:
      result.append(default_val)
  return tuple(result)


def filter_exp(
  exprs: list[str] | tuple[str, ...], names: tuple[str, ...]
) -> tuple[str, ...]:
  """Filter names based on regex patterns."""
  patterns = [re.compile(expr) for expr in exprs]
  return tuple(name for name in names if any(pat.match(name) for pat in patterns))


def resolve_field(
  field: T | dict[str, T], names: tuple[str, ...], default_val: T = 0
) -> tuple[T, ...]:
  result = (
    resolve_expr(field, names, default_val)
    if isinstance(field, dict)
    else [field] * len(names)
  )


def resolve_param_to_list(
  param: float | dict[str, float], joint_names: list[str]
) -> list[float]:
  """Convert a parameter (float or dict) to a list matching joint order.

  Args:
    param: Single float or dict mapping joint names/regex patterns
      to values.
    joint_names: Ordered list of joint names.

  Returns:
    List of parameter values in the same order as joint_names.

  Raises:
    ValueError: If param is a dict and not all regex patterns match,
      or if multiple patterns match the same joint name.

  Example:
    >>> resolve_param_to_list(1.5, ["hip", "knee", "ankle"])
    [1.5, 1.5, 1.5]
    >>> resolve_param_to_list(
    ...   {"hip": 2.0, "knee": 1.5, "ankle": 1.0},
    ...   ["hip", "knee", "ankle"]
    ... )
    [2.0, 1.5, 1.0]
    >>> resolve_param_to_list(
    ...   {".*_hip": 2.0, ".*_knee": 1.5},
    ...   ["front_hip", "back_hip", "front_knee"]
    ... )
    [2.0, 2.0, 1.5]
  """
  if isinstance(param, dict):
    from mjlab.third_party.isaaclab.isaaclab.utils.string import (
      resolve_matching_names_values,
    )

    _, _, values = resolve_matching_names_values(
      param, joint_names, preserve_order=True
    )
    return values
  else:
    return [param] * len(joint_names)
