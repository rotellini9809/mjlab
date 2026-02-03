"""Stage-1 motor controller training entrypoint."""

from __future__ import annotations

import tyro

from mjlab.motor_controller_stage1.trainer import TrainConfig, train_stage1


def main() -> None:
  cfg = tyro.cli(TrainConfig)
  train_stage1(cfg)


if __name__ == "__main__":
  main()
