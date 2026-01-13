from easydict import EasyDict

def create_config(env_id, max_steps, max_action_num, action_space_size, collector_env_num, evaluator_env_num, n_episode,
                  reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length,
                  buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition, total_batch_size,
                  num_layers, model_name, replay_ratio, norm_type, update_per_collect,  
                  collect_num_simulations, eval_num_simulations):
    return EasyDict(dict(
        env=dict(
            stop_value=int(1e6),
            env_id=env_id,
            observation_shape=512,
            max_steps=max_steps,
            max_action_num=max_action_num,
            tokenizer_path=model_name,
            max_seq_len=512,
            game_path=f"./zoo/jericho/envs/z-machine-games-master/jericho-game-suite/{env_id}",
            for_unizero=True,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
        ),
        policy=dict(
            multi_gpu=True,  # Very important for ddp
            only_use_moco_stats=False,
            collect_num_simulations=collect_num_simulations,
            eval_num_simulations=eval_num_simulations,
            use_moco=False, # Whether to use MoCo for multi-task gradient adjustments
            grad_correct_params=dict(
                MoCo_beta=0.5, MoCo_beta_sigma=0.5, MoCo_gamma=0.1, MoCo_gamma_sigma=0.5, MoCo_rho=0,
                calpha=0.5, rescale=1,
            ),
            use_wandb=False,
            learn=dict(
                learner=dict(
                    hook=dict(
                        save_ckpt_after_iter=200000
                    ),
                ),
            ),
            total_task_num=len(env_id_list),
            task_num=len(env_id_list),
            task_id=0,
            model=dict(
                observation_shape=512,
                action_space_size=action_space_size,
                encoder_url=model_name,
                model_type="mlp",
                norm_type=norm_type,
                continuous_action_space=False,
                world_model_cfg=dict(
                    final_norm_option_in_obs_head='LayerNorm',
                    final_norm_option_in_encoder='LayerNorm',
                    predict_latent_loss_type='mse', 
                    share_head=False, 
                    policy_entropy_weight=5e-2, 
                    continuous_action_space=False,
                    max_blocks=num_unroll_steps,
                    max_tokens=2 * num_unroll_steps,
                    context_length=2 * infer_context_length,
                    device="cuda",
                    action_space_size=action_space_size,
                    num_layers=num_layers,
                    num_heads=24,
                    obs_type="text",  
                    env_num=max(collector_env_num, evaluator_env_num),              
                    task_embed_option=None,   
                    use_task_embed=False,
                    embed_dim=768,
                    task_num=len(env_id_list),
                    use_normal_head=True,
                    use_softmoe_head=False,
                    use_moe_head=False,
                    num_experts_in_moe_head=4,

                    moe_in_transformer=False, 
                    multiplication_moe_in_transformer=True, # Whether to use moe in transformers
                    n_shared_experts=1,
                    num_experts_per_tok=1,
                    num_experts_of_moe_in_transformer=8,

                    moe_use_lora=False, # Does moe use lora
                    lora_r= 0,
                    lora_alpha =1,
                    lora_dropout= 0.0,

                    analysis_dormant_ratio_weight_rank=False, 
                    analysis_dormant_ratio_interval=5000,
                    game_segment_length=40,
                    use_priority=False,
                ),
            ),
            optim_type='AdamW',
            # (bool) 是否启用自适应策略熵权重 (alpha)
            use_adaptive_entropy_weight=False,

            # (float) 自适应alpha优化器的学习率
            adaptive_entropy_alpha_lr=1e-4,
            target_entropy_start_ratio =0.98,
            # target_entropy_end_ratio =0.9, # TODO=====
            # target_entropy_end_ratio =0.7,
            # target_entropy_decay_steps = 100000, # 例如，在100k次迭代后达到最终值

            target_entropy_end_ratio =0.5, # for action_space=18
            target_entropy_decay_steps = 100000, # 例如，在150k次迭代 300k envsteps后达到最终值
            # target_entropy_decay_steps = 150000, # 例如，在150k次迭代 300k envsteps后达到最终值

            # ==================== START: Encoder-Clip Annealing Config ====================
            # (bool) 是否启用 encoder-clip 值的退火。
            use_encoder_clip_annealing=False,
            # (str) 退火类型。可选 'linear' 或 'cosine'。
            encoder_clip_anneal_type='cosine',
            # (float) 退火的起始 clip 值 (训练初期，较宽松)。
            encoder_clip_start_value=30.0,
            # (float) 退火的结束 clip 值 (训练后期，较严格)。
            encoder_clip_end_value=10.0,
            # (int) 完成从起始值到结束值的退火所需的训练迭代步数。
            encoder_clip_anneal_steps=30000,  # 例如，在30k次迭代后达到最终值


            # ==================== START: label smooth ====================
            policy_ls_eps_start=0.05, #TODO============= good start in Pong and MsPacman
            policy_ls_eps_end=0.01,
            policy_ls_eps_decay_steps=50000, # 50k
            label_smoothing_eps=0,  #TODO============= for value

            # ==================== [新增] 范数监控频率 ====================
            # 每隔多少个训练迭代步数，监控一次模型参数的范数。设置为0则禁用。
            monitor_norm_freq=10000,
            
            
            use_task_exploitation_weight=False, 
            task_complexity_weight=False,
            total_batch_size=total_batch_size,
            allocated_batch_sizes=False,
            use_priority=False,
            print_task_priority_logs=False,
            cuda=True,
            model_path=None,
            num_unroll_steps=num_unroll_steps,
            update_per_collect=update_per_collect, 
            action_type="varied_action_space",
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            reanalyze_ratio=reanalyze_ratio,
            learning_rate=0.0001,
            cos_lr_scheduler=False,
            fixed_temperature_value=0.25,
            manual_temperature_decay=False,
            n_episode=n_episode,
            train_start_after_envsteps=int(0),
            replay_buffer_size=int(5e5),
            eval_freq=int(5e3),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size,
            reanalyze_partition=reanalyze_partition,     
        ),
    ))


def generate_configs(env_id_list, env_configurations, collector_env_num, n_episode, evaluator_env_num,
                     reanalyze_ratio, batch_size, num_unroll_steps, infer_context_length,
                     seed, buffer_reanalyze_freq, reanalyze_batch_size, reanalyze_partition,
                     total_batch_size, num_layers, model_name, replay_ratio, norm_type, collect_num_simulations, eval_num_simulations):
    configs = []
    exp_name_prefix = f'data_scalezero/jericho_ddp_mt_moe8_{len(env_id_list)}games_tbs{total_batch_size}-nlayer{num_layers}-rr{replay_ratio}_not-share-head_encoder-final-ln_seed{seed}/'

    action_space_size_list = [v[0] for _, v in env_configurations.items()]  
    max_action_space_size = max(action_space_size_list)

    for task_id, env_id in enumerate(env_id_list):
        _, max_steps = env_configurations.get(env_id, (10, 50))  
        update_per_collect = 40  # Ensure at least one update per collect

        config = create_config(
            env_id=env_id, max_steps=max_steps, max_action_num=max_action_space_size, action_space_size=max_action_space_size,
            collector_env_num=collector_env_num, evaluator_env_num=evaluator_env_num, n_episode=n_episode,
            reanalyze_ratio=reanalyze_ratio, batch_size=batch_size,
            num_unroll_steps=num_unroll_steps, infer_context_length=infer_context_length, buffer_reanalyze_freq=buffer_reanalyze_freq,
            reanalyze_batch_size=reanalyze_batch_size, reanalyze_partition=reanalyze_partition, total_batch_size=total_batch_size,
            num_layers=num_layers, model_name=model_name, replay_ratio=replay_ratio,
            norm_type=norm_type, update_per_collect=update_per_collect, collect_num_simulations=collect_num_simulations, eval_num_simulations=eval_num_simulations,
        )
        config.policy.task_id = task_id
        config.exp_name = exp_name_prefix + f"{env_id.split('.z5')[0]}_seed{seed}"
        configs.append([task_id, [config, create_env_manager()]])
    return configs

def create_env_manager():
    return EasyDict(dict(
        env=dict(
            type='jericho',
            import_names=['zoo.jericho.envs.jericho_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(
            type='unizero_multitask',
            import_names=['lzero.policy.unizero_multitask'],
        ),
    ))

if __name__ == "__main__":
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        
        Example launch command:
        
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        torchrun --nproc_per_node=4 ./zoo/jericho/config/jericho_unizero_multitask_ddp_config.py
    """

    from lzero.entry import train_unizero_multitask_ddp
    from ding.utils import DDPContext
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    env_configurations = {
        'detective.z5': (12, 100),
        'omniquest.z5': (25, 100),
        'acorncourt.z5': (45, 50),
        'zork1.z5': (55, 500),
    }
    env_id_list = list(env_configurations.keys())
    
    # Model name or path - configurable according to the predefined model paths or names
    model_name: str = 'BAAI/bge-base-en-v1.5'
    replay_ratio = 0.1
    norm_type = 'LN'

    collector_env_num = 4
    n_episode = 4
    evaluator_env_num = 2
    max_env_step = int(5e5)
    reanalyze_ratio = 0.0

    total_batch_size =int(64 * len(env_id_list))
    batch_size = [int(total_batch_size / len(env_id_list)) for _ in range(len(env_id_list))]

    num_layers=2
    num_unroll_steps = 10
    infer_context_length = 4
    buffer_reanalyze_freq = 1 / 1000000
    reanalyze_batch_size = 160
    reanalyze_partition = 0.75
    collect_num_simulations = 50
    eval_num_simulations = 50

    for seed in [0]:
        configs = generate_configs( env_id_list=env_id_list, env_configurations=env_configurations, 
                                    collector_env_num=collector_env_num, n_episode=n_episode,
                                    evaluator_env_num=evaluator_env_num,
                                    reanalyze_ratio=reanalyze_ratio, batch_size=batch_size,
                                    num_unroll_steps=num_unroll_steps, infer_context_length=infer_context_length,
                                    seed=seed, buffer_reanalyze_freq=buffer_reanalyze_freq,
                                    reanalyze_batch_size=reanalyze_batch_size, reanalyze_partition=reanalyze_partition,
                                    total_batch_size=total_batch_size, num_layers=num_layers, 
                                    model_name=model_name, replay_ratio=replay_ratio, norm_type=norm_type,
                                    collect_num_simulations=collect_num_simulations, eval_num_simulations=eval_num_simulations)

        with DDPContext():
            train_unizero_multitask_ddp(configs, seed=seed, max_env_step=max_env_step) 
