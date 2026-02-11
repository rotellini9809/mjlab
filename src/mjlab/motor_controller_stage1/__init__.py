"""Stage-1 motor controller training (offline, dataset-driven)."""

from mjlab.motor_controller_stage1.dataset import (  # noqa: F401
  RolloutDataset,
  RolloutDatasetStats,
)
from mjlab.motor_controller_stage1.latent_action import (  # noqa: F401
  DEFAULT_STAGE1_CHECKPOINT,
  DEFAULT_STAGE1_WANDB_RUN_PATH,
  MotorLatentAction,
  MotorLatentActionCfg,
  motor_last_decoded_action,
)
from mjlab.motor_controller_stage1.trainer import TrainConfig, train_stage1  # noqa: F401
