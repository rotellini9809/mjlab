"""Script to list MJLab environments."""

import gymnasium as gym
import tyro
from prettytable import PrettyTable

import mjlab.tasks  # noqa: F401 to register environments


def main(keyword: str | None = None):
  """Print all environments registered whose id contains `Mjlab-`."""
  prefix_substring = "Mjlab-"

  table = PrettyTable(["#", "Task ID", "Entry Point", "env_cfg_entry_point"])
  table.title = "Available Environments in Mjlab"
  table.align["Task ID"] = "l"
  table.align["Entry Point"] = "l"
  table.align["env_cfg_entry_point"] = "l"

  idx = 0
  for spec in gym.registry.values():
    try:
      if prefix_substring in spec.id and (keyword is None or keyword in spec.id):
        env_cfg_ep = spec.kwargs.get("env_cfg_entry_point", "")
        table.add_row([idx + 1, spec.id, spec.entry_point, env_cfg_ep])
        idx += 1
    except Exception:
      continue

  print(table)
  if idx == 0:
    msg = f"[INFO] No tasks matched filter: '{prefix_substring}'"
    if keyword:
      msg += f" and keyword '{keyword}'"
    print(msg)
  return idx


if __name__ == "__main__":
  tyro.cli(main)
