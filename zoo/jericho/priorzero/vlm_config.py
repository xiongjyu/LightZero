"""
VLM Configuration for PriorZero with Image Input

This module provides configuration for using Vision-Language Models
to generate action priors for image-based environments (e.g., Atari).
"""
from typing import Dict, Tuple, Optional
from easydict import EasyDict
from dataclasses import dataclass, field


# ==============================================================================
# VLM Model Configuration Presets
# ==============================================================================
VLM_MODEL_CONFIGS = {
    "Qwen2.5-VL-2b": {
        "model_name": "Qwen2.5-VL",
        "model_path": "/mnt/shared-storage-user/puyuan/model/Qwen2.5-VL-2B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.25,
        "description": "Qwen2.5-VL-2B-Instruct (smaller, faster)",
    },
    "Qwen2.5-VL-7b": {
        "model_name": "Qwen2.5-VL",
        "model_path": "/mnt/shared-storage-user/puyuan/model/Qwen2.5-VL-7B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.35,
        "description": "Qwen2.5-VL-7B-Instruct (better quality)",
    },
    "Qwen3-VL-2b": {
        "model_name": "Qwen3-VL",
        "model_path": "/mnt/shared-storage-user/puyuan/model/Qwen3-VL-2B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.25,
        "description": "Qwen2.5-VL-2B-Instruct (smaller, faster)",
    },
}


def get_available_vlm_models():
    """Get list of available VLM model configurations"""
    return list(VLM_MODEL_CONFIGS.keys())


def get_vlm_model_config(model_key: str) -> Dict:
    """Get VLM model configuration by key"""
    if model_key not in VLM_MODEL_CONFIGS:
        available = ", ".join(get_available_vlm_models())
        raise ValueError(
            f"Unknown VLM model key: {model_key}\n"
            f"Available models: {available}"
        )
    return VLM_MODEL_CONFIGS[model_key]


def print_available_vlm_models():
    """Print all available VLM model configurations"""
    print("\n" + "="*80)
    print("Available VLM Model Configurations:")
    print("="*80)
    for key, config in VLM_MODEL_CONFIGS.items():
        print(f"\n  {key}:")
        print(f"    Path: {config['model_path']}")
        print(f"    Tensor Parallel Size: {config['tensor_parallel_size']}")
        print(f"    GPU Memory Utilization: {config['gpu_memory_utilization']}")
        print(f"    Description: {config['description']}")
    print("="*80 + "\n")


@dataclass
class PriorZeroVLMConfig:
    """Configuration for VLM-based PriorZero (image input)"""

    # VLM model settings
    model_name_or_path: str = "Qwen2.5-VL-7b"

    vlm_model_type: str = "qwen-vl"  # 'qwen-vl', 'llava', 'internvl'

    # Training settings (similar to LLM config)
    enable_sft: bool = False
    enable_rft: bool = True
    rft_loss_weight: float = 1.0

    # VLM inference settings
    temperature: float = 1.0
    max_new_tokens: int = 256  # Shorter than LLM since we just need action probs
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.3

    # vLLM engines 
    enable_vllm: bool = True
    enable_prefix_caching: bool = True
    use_cuda_ipc: bool = False
    vllm_sync_backend: str = "nccl" # vLLM 同步参数使用的后端
    vllm_sync_with_ray: bool = False # 是否使用 ray 来同步 vLLM 参数
    vllm_tensor_parallel_size: int = 1 # 每个vllm engine使用几张GPU张量并行 (Fixed: 1.5B model should use 1 GPU)

    vllm_enable_sleep: bool = True # 是否可以休眠
    top_p: float = 1.0
    seed: int = 0
    reduction: str = "mean"
    


    # Prior generation settings
    use_prior: bool = True  # Whether to use VLM prior
    llm_prior_temperature: float = 1.0  # Temperature for prior distribution

    attn_implementation: str = "flash_attention_2" 
    use_cot: bool = True
    prompt_max_len: int = 8192
    generate_max_len: int = 512
    bf16: bool = True

    history_length: int = 3  # Number of recent steps to include in context

    # Training settings
    colocate_all_models: bool = True
    policy_model_num_gpus: int = 1
    reference_model_num_gpus: int = 1
    deepspeed_enable_sleep: bool = True

    zero_stage: int = 2
    gradient_checkpointing: bool = False
    max_norm: float = 1.0
    ds_tensor_parallel_size: int = 1
    ring_attn_size: int = 1

    # Batch sizes
    train_batch_size: int = 640
    micro_train_batch_size: int = 8
    broadcast_every: int = 1

    # Optimizer settings
    learning_rate: float = 5e-7
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine_with_min_lr"
    lr_warmup_ratio: float = 0.03
    max_steps: int = int(1e4)

    # Loss settings
    policy_loss_type: str = "ppo"
    reward_func: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "format_reward": False,  # No format reward for Atari
    }))
    advantage_type: str = "advantage_running_norm"
    eps_clip_low_high: Tuple[float, float] = (0.2, 0.2)
    rft_kl_coef: float = 0.01
    entropy_loss_coef: float = 0.0
    kl_estimator: str = "k3"

    # Training schedule
    train_vlm_after_wm_warm_step: int = int(1e2)
    vlm_save_freq: int = 500
    save_path: str = ""

    # Value normalization
    value_norm_cfg: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        'enable_stability_optimizer': True,
        'value_norm_init_momentum': 0.9,
        'value_norm_final_momentum': 0.99,
        'value_norm_warmup_steps': 100,
        'value_norm_clip_percentile': 0.95,
        'value_norm_clip_method': "soft",
        "value_norm_history_size": 1000,
    }))

    # Prompt template
    prompt_template: str = (
        "You are an expert Atari game player. "
        "Based on the current game screen, choose the best action. "
        "Available actions: {action_list}\n"
        "Provide probabilities for each action as JSON: "
        "{{'action': probability, ...}}"
    )


def get_priorzero_vlm_config(
    env_id: str = 'PongNoFrameskip-v4',
    seed: int = 0,
    exp_name: str = None,
    vlm_model_key: Optional[str] = None,
    use_prior: bool = True,
    multi_gpu: bool = False,
    quick_test: bool = False,
) -> Tuple[EasyDict, EasyDict, PriorZeroVLMConfig]:
    """
    Generate complete PriorZero configuration with VLM for image input.

    Args:
        env_id: Atari environment ID
        seed: Random seed
        exp_name: Experiment name
        vlm_model_key: VLM model key (e.g., 'qwen-vl-chat', 'llava-1.5-7b')
        use_prior: Whether to use VLM prior
        multi_gpu: Whether to use multi-GPU training
        quick_test: Whether to use quick test configuration

    Returns:
        main_config: Main configuration dictionary
        create_config: Creation configuration
        vlm_config: VLM configuration
    """
    from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

    action_space_size = atari_env_action_space_map[env_id]

    # Base configuration parameters
    if quick_test:
        collector_env_num = 2
        num_segments = 2
        game_segment_length = 20
        evaluator_env_num = 2
        num_simulations = 5
        collect_num_simulations = 5
        eval_num_simulations = 5
        batch_size = 8
        num_layers = 1
        replay_ratio = 0.1
    else:
        collector_env_num = 8
        num_segments = 8
        game_segment_length = 20
        evaluator_env_num = 3
        num_simulations = 25
        collect_num_simulations = 25
        eval_num_simulations = 50

        batch_size = 256
        num_layers = 2
        replay_ratio = 0.1

    num_unroll_steps = 10
    infer_context_length = 4

    # Environment configuration
    env_config = dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(3, 64, 64),
        gray_scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False,),
    )

    # Policy configuration
    policy_config = dict(
        type='priorzero',
        multi_gpu=multi_gpu,
        use_wandb=False,
        learn=dict(
            learner=dict(
                hook=dict(save_ckpt_after_iter=1000000,),
            ),
        ),
        model=dict(
            observation_shape=(3, 64, 64),
            action_space_size=action_space_size,
            reward_support_range=(-300., 301., 1.),
            value_support_range=(-300., 301., 1.),
            norm_type="LN",
            num_res_blocks=1,
            num_channels=64,
            world_model_cfg=dict(
                norm_type="LN",
                final_norm_option_in_obs_head='LayerNorm',
                final_norm_option_in_encoder='LayerNorm',
                predict_latent_loss_type='mse',
                policy_entropy_weight=5e-3,
                continuous_action_space=False,
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,
                context_length=2 * infer_context_length,
                device='cuda',
                action_space_size=action_space_size,
                num_layers=num_layers,
                num_heads=8,
                embed_dim=768,
                obs_type='image',  # KEY: Image input with VLM prior
                env_num=max(collector_env_num, evaluator_env_num),
                num_simulations=num_simulations,
                game_segment_length=game_segment_length,
                encoder_type='resnet',

                decode_loss_mode=None, 
                latent_recon_loss_weight=0,
                task_embed_option=None,
                moe_in_transformer=False,
                multiplication_moe_in_transformer=False,
            )
        ),
        optim_type='AdamW_mix_lr_wdecay',
        weight_decay=1e-2,
        learning_rate=0.0001,
        num_unroll_steps=num_unroll_steps,
        update_per_collect=None,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        num_simulations=num_simulations,
        # num_segments=num_segments,
        td_steps=5,
        train_start_after_envsteps=0,
        game_segment_length=game_segment_length,
        replay_buffer_size=int(5e5),
        eval_freq=int(5e3),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,

        num_segments=collector_env_num,
        action_type="varied_action_space",
        model_path=None,
        reanalyze_ratio=0,
        cos_lr_scheduler=False,
        fixed_temperature_value=0.25,
        manual_temperature_decay=False,
        n_episode=collector_env_num,
        buffer_reanalyze_freq=1 / 1000000,
        reanalyze_batch_size=160,
        reanalyze_partition=0.75,
        device='cuda',
        
        collect_num_simulations=collect_num_simulations,
        eval_num_simulations=eval_num_simulations,
        off_policy_degree=0,
        enable_async_eval=False,
        
        # optim_type='AdamW',
        grad_clip_value=10.0,
        value_loss_weight=0.25,
        policy_loss_weight=1.0,
        reward_loss_weight=1.0,

        use_adaptive_entropy_weight=False,
        adaptive_entropy_alpha_lr=1e-4,
        use_encoder_clip_annealing=False,
        encoder_clip_anneal_type='cosine',
        encoder_clip_start_value=30.0,
        encoder_clip_end_value=10.0,
        encoder_clip_anneal_steps=100000,
        use_priority=False,  # Prioritized experience replay
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,
    )

    main_config = EasyDict(dict(
        env=env_config,
        policy=policy_config,
        exp_name=exp_name or f'data_priorzero_vlm/{env_id[:-14]}_seed{seed}',
        seed=seed
    ))

    create_config = EasyDict(dict(
        env=dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='priorzero',
            import_names=['zoo.jericho.priorzero.priorzero_policy'],
        ),
        collector=dict(
            type='priorzero_segment',
            import_names=['zoo.jericho.priorzero.priorzero_collector'],
        ),
        evaluator=dict(
            type='priorzero',
            import_names=['zoo.jericho.priorzero.priorzero_evaluator'],
        ),
        replay_buffer=dict(
            type='game_buffer_muzero',
            import_names=['lzero.mcts.buffer.game_buffer_muzero'],
        ),
    ))

    # VLM configuration
    vlm_config = PriorZeroVLMConfig(use_prior=use_prior)

    # Auto-configure VLM model
    if use_prior:
        if vlm_model_key is None:
            vlm_model_key = "qwen-vl-chat"  # Default VLM
            print(f"[Config] Using default VLM model: {vlm_model_key}")

        vlm_model_config = get_vlm_model_config(vlm_model_key)
        vlm_config.model_name_or_path = vlm_model_config["model_path"]
        vlm_config.vlm_model_type = vlm_model_config["model_name"]
        vlm_config.tensor_parallel_size = vlm_model_config["tensor_parallel_size"]
        vlm_config.gpu_memory_utilization = vlm_model_config["gpu_memory_utilization"]

        print(f"[Config] VLM configuration applied:")
        print(f"  - Model: {vlm_model_key}")
        print(f"  - Path: {vlm_config.model_name_or_path}")
        print(f"  - Tensor Parallel Size: {vlm_config.tensor_parallel_size}")
        print(f"  - GPU Memory Utilization: {vlm_config.gpu_memory_utilization}")
    else:
        print(f"[Config] VLM prior disabled (use_prior=False)")
        vlm_config = None

    return main_config, create_config, vlm_config


if __name__ == "__main__":
    # Test configuration generation
    print("PriorZero VLM Configuration")
    print("=" * 80)

    # List available models
    print_available_vlm_models()

    # Generate test config
    print("\nGenerating test configuration...")
    main_cfg, create_cfg, vlm_cfg = get_priorzero_vlm_config(
        env_id='PongNoFrameskip-v4',
        seed=0,
        vlm_model_key='qwen-vl-chat',
        use_prior=True,
        quick_test=True,
    )

    print("\n✓ Configuration generated successfully")
    print(f"  - Experiment: {main_cfg.exp_name}")
    print(f"  - Environment: {main_cfg.env.env_id}")
    print(f"  - Observation shape: {main_cfg.policy.model.observation_shape}")
    print(f"  - obs_type: {main_cfg.policy.model.world_model_cfg.obs_type}")
    if vlm_cfg:
        print(f"  - VLM model: {vlm_cfg.model_name_or_path}")
        print(f"  - Use prior: {vlm_cfg.use_prior}")
