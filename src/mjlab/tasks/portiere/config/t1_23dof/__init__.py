"""Task registration for Booster T1 portiere environment."""

from mjlab.tasks.registry import register_mjlab_task
from .env_cfgs import booster_t1_23_portiere_env_cfg
from .rl_cfg import booster_t1_23_tracking_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Portiere-Booster-T1_23",
    env_cfg=booster_t1_23_portiere_env_cfg(),
    play_env_cfg=booster_t1_23_portiere_env_cfg(play=True),
    rl_cfg=booster_t1_23_tracking_ppo_runner_cfg(),
    runner_cls=None,
)
