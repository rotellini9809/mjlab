"""Task registration for Booster T1 flat tracking."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import booster_t1_23_flat_tracking_env_cfg
from .rl_cfg import booster_t1_23_tracking_ppo_runner_cfg


# Task standard con state estimation
register_mjlab_task(
    task_id="Mjlab-Tracking-Flat-Booster-T1_23",
    env_cfg=booster_t1_23_flat_tracking_env_cfg(),
    play_env_cfg=booster_t1_23_flat_tracking_env_cfg(play=True),
    rl_cfg=booster_t1_23_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Variante senza state estimation
register_mjlab_task(
    task_id="Mjlab-Tracking-Flat-Booster-T1_23-No-State-Estimation",
    env_cfg=booster_t1_23_flat_tracking_env_cfg(has_state_estimation=False),
    play_env_cfg=booster_t1_23_flat_tracking_env_cfg(
        has_state_estimation=False,
        play=True,
    ),
    rl_cfg=booster_t1_23_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
