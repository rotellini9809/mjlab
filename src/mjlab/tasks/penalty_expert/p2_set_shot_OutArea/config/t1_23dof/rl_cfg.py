"""RL configuration for Booster T1 penalty Out Area task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def booster_t1_23_penalty_out_area_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster T1 penalty task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            obs_normalization=True,
            hidden_dims=(512, 256, 128),
            activation="elu",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            obs_normalization=True,
            hidden_dims=(512, 256, 128),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.02,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.02,
            max_grad_norm=1.0,
        ),
        experiment_name="booster_t1_23_penalty_out_area",
        wandb_project="penalty_OutArea",
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
    )


def booster_t1_23_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Backward-compatible alias for older penalty task imports."""
    return booster_t1_23_penalty_out_area_ppo_runner_cfg()
