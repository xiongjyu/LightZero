import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
reanalyze_ratio = 0.
update_per_collect = None
replay_ratio = 0.25
max_env_step = int(5e5)
batch_size = 256
num_unroll_steps = 10
infer_context_length = 4
num_layers = 2
norm_type = 'BN'
game_segment_length = 20

# debug
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 5
# batch_size = 2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_image_unizero_config = dict(
    exp_name=f'data_unizero/lunarlander_image_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}-infer{infer_context_length}_bs{batch_size}_{norm_type}_seed0',
    env=dict(
        env_id='LunarLander-v2',
        observation_shape=(3, 64, 64),
        gray_scale=False,
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        collect_max_episode_steps=int(1000),
        eval_max_episode_steps=int(1000),
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 64, 64),
            action_space_size=4,
            norm_type=norm_type,
            world_model_cfg=dict(
                continuous_action_space=False,
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * infer_context_length,
                device='cuda',
                action_space_size=4,
                num_layers=num_layers,
                num_heads=8,
                embed_dim=768,
                obs_type='image',
                encoder_type='resnet',
                group_size=8,
                norm_type=norm_type,
                env_num=max(collector_env_num, evaluator_env_num),
                # Normalization options
                final_norm_option_in_encoder='LayerNorm',
                final_norm_option_in_obs_head='LayerNorm',
                predict_latent_loss_type='mse',
                # Task embedding (single-task, disabled)
                task_embed_option=None,
                # MoE (disabled for single-task baseline)
                moe_in_transformer=False,
                multiplication_moe_in_transformer=False,
                # Misc
                policy_entropy_weight=1e-4,
                num_simulations=num_simulations,
                game_segment_length=game_segment_length,
                rotary_emb=False,
                latent_recon_loss_weight=0.,
                perceptual_loss_weight=0.,
                decode_loss_mode=None,
            ),
        ),
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        cuda=True,
        game_segment_length=game_segment_length,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        piecewise_decay_lr_scheduler=False,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        replay_ratio=replay_ratio,
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
lunarlander_image_unizero_config = EasyDict(lunarlander_image_unizero_config)
main_config = lunarlander_image_unizero_config

lunarlander_image_unizero_create_config = dict(
    env=dict(
        type='lunarlander_image',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_image_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
lunarlander_image_unizero_create_config = EasyDict(lunarlander_image_unizero_create_config)
create_config = lunarlander_image_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)
