"""
VL Configuration for PriorZero with Image Input

This module provides configuration for using Vision-Language (VL) models
to generate action priors for image-based environments (e.g., Atari).
"""
from typing import Dict, Tuple, Optional
from easydict import EasyDict
from dataclasses import dataclass, field


# ==============================================================================
# Game Descriptions for VL Prompts
# ==============================================================================
GAME_DESCRIPTIONS = {
    'PongNoFrameskip-v4': (
        "This is Pong. You control the right paddle. "
        "Move the paddle UP or DOWN to hit the ball past the opponent's paddle on the left. "
        "Score points when the opponent misses. First to 21 points wins."
    ),
    'BreakoutNoFrameskip-v4': (
        "This is Breakout. You control a paddle at the bottom of the screen. "
        "Move LEFT or RIGHT to bounce the ball upward and break the colored bricks. "
        "Each brick broken scores points. Don't let the ball fall below the paddle."
    ),
    'SpaceInvadersNoFrameskip-v4': (
        "This is Space Invaders. You control a cannon at the bottom of the screen. "
        "Move LEFT/RIGHT and FIRE to shoot the descending rows of aliens. "
        "Destroy all aliens before they reach the bottom. Use shields for cover."
    ),
    'QbertNoFrameskip-v4': (
        "This is Q*bert. You control Q*bert on a pyramid of cubes. "
        "Jump on each cube to change its color to the target color. "
        "Avoid enemies like Coily the snake. Change all cubes to complete the level."
    ),
    'MsPacmanNoFrameskip-v4': (
        "This is Ms. Pac-Man. Navigate the maze eating dots and power pellets. "
        "Avoid the ghosts unless you've eaten a power pellet, which lets you eat them. "
        "Clear all dots to advance to the next level."
    ),
    'LunarLander-v2': (
        "This is Lunar Lander. You control a spacecraft descending toward a landing pad. "
        "Use the MAIN ENGINE to slow descent, and LEFT/RIGHT engines to adjust position. "
        "Land gently on the pad between the flags. Fuel is limited. "
        "Reward: +100-140 for landing on pad, -100 for crash, -0.3 per engine fire."
    ),
}


# ==============================================================================
# VL Model Configuration Presets
# ==============================================================================
VL_MODEL_CONFIGS = {
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


def get_available_vl_models():
    """Get list of available VL model configurations"""
    return list(VL_MODEL_CONFIGS.keys())


def get_vl_model_config(model_key: str) -> Dict:
    """Get VL model configuration by key"""
    if model_key not in VL_MODEL_CONFIGS:
        available = ", ".join(get_available_vl_models())
        raise ValueError(
            f"Unknown VL model key: {model_key}\n"
            f"Available models: {available}"
        )
    return VL_MODEL_CONFIGS[model_key]


def print_available_vl_models():
    """Print all available VL model configurations"""
    print("\n" + "="*80)
    print("Available VL Model Configurations:")
    print("="*80)
    for key, config in VL_MODEL_CONFIGS.items():
        print(f"\n  {key}:")
        print(f"    Path: {config['model_path']}")
        print(f"    Tensor Parallel Size: {config['tensor_parallel_size']}")
        print(f"    GPU Memory Utilization: {config['gpu_memory_utilization']}")
        print(f"    Description: {config['description']}")
    print("="*80 + "\n")


@dataclass
class PriorZeroVLConfig:
    """Configuration for VL-based PriorZero (image input)"""

    # VL model settings
    model_name_or_path: str = "Qwen2.5-VL-7b"

    vl_model_type: str = "qwen-vl"  # 'qwen-vl', 'llava', 'internvl'

    # Game description for prompts
    game_description: str = ""

    # Training settings (similar to LLM config)
    enable_sft: bool = False
    enable_rft: bool = True
    rft_loss_weight: float = 1.0

    # VL inference settings
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
    use_prior: bool = True  # Whether to use VL prior
    llm_prior_temperature: float = 1.0  # Temperature for prior distribution

    # Evaluation settings
    eval_dict: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "world_model": True,
        "world_model_llm_prior": True,
        "llm_prior": True,
        "eval_freq": int(500),
    }))

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
    train_vl_after_wm_warm_step: int = int(1e2)
    vl_save_freq: int = 500
    save_path: str = ""

    # Alternating training schedule (matches LLM config)
    train_schedule: Optional[EasyDict] = field(default_factory=lambda: EasyDict({
        "alternate": True,
        "wm_update_iters": 1e3,
        "llm_update_iters": 1e2,
        "start_phase": "wm",
        "wm_warmup_updates": 0,
    }))

    enable_world_model: bool = True
    enable_rft: bool = True
    max_rollout_staleness: int = 1
    vl_fixed: bool = False  # If True, VL is frozen (inference only, no VL training)

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

    # Prompt template (Qwen-VL format)
    prompt_template: str = (
        "<|vision_start|><|image_pad|><|vision_end|>"
        "You are an expert Atari game player. "
        "Based on the current game screen, choose the best action. "
        "Available actions: {action_list}\n"
        "Provide probabilities for each action as JSON: "
        "{{'action': probability, ...}}"
    )


def get_priorzero_vl_config(
    env_id: str = 'PongNoFrameskip-v4',
    seed: int = 0,
    exp_name: str = None,
    vl_model_key: Optional[str] = None,
    use_prior: bool = True,
    multi_gpu: bool = False,
    quick_test: bool = False,
) -> Tuple[EasyDict, EasyDict, PriorZeroVLConfig]:
    """
    Generate complete PriorZero configuration with VL for image input.

    Args:
        env_id: Atari environment ID
        seed: Random seed
        exp_name: Experiment name
        vl_model_key: VL model key (e.g., 'qwen-vl-chat', 'llava-1.5-7b')
        use_prior: Whether to use VL prior
        multi_gpu: Whether to use multi-GPU training
        quick_test: Whether to use quick test configuration

    Returns:
        main_config: Main configuration dictionary
        create_config: Creation configuration
        vl_config: VL configuration
    """
    from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

    # Detect environment type
    is_lunarlander = 'LunarLander' in env_id

    if is_lunarlander:
        action_space_size = 4
    else:
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

    # Episode step limits
    if is_lunarlander:
        collect_max_episode_steps = int(1000)
        eval_max_episode_steps = int(1000)
    else:
        collect_max_episode_steps = int(5e3)
        eval_max_episode_steps = int(5e3)

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
        collect_max_episode_steps=collect_max_episode_steps,
        eval_max_episode_steps=eval_max_episode_steps,
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
            reward_support_range=(-50., 51., 1.),
            value_support_range=(-50., 51., 1.),
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
                obs_type='image',  # KEY: Image input with VL prior
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
        exp_name=exp_name or f'data_priorzero_vl/{env_id}_seed{seed}',
        seed=seed
    ))

    if is_lunarlander:
        env_create_cfg = dict(
            type='lunarlander_image',
            import_names=['zoo.box2d.lunarlander.envs.lunarlander_image_env'],
        )
    else:
        env_create_cfg = dict(
            type='atari_lightzero',
            import_names=['zoo.atari.envs.atari_lightzero_env'],
        )

    create_config = EasyDict(dict(
        env=env_create_cfg,
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

    # VL configuration
    vl_config = PriorZeroVLConfig(use_prior=use_prior)

    # Set game description
    vl_config.game_description = GAME_DESCRIPTIONS.get(env_id, "")

    # Auto-configure VL model
    if use_prior:
        if vl_model_key is None:
            vl_model_key = "qwen-vl-chat"  # Default VL
            print(f"[Config] Using default VL model: {vl_model_key}")

        vl_model_config = get_vl_model_config(vl_model_key)
        vl_config.model_name_or_path = vl_model_config["model_path"]
        vl_config.vl_model_type = vl_model_config["model_name"]
        vl_config.tensor_parallel_size = vl_model_config["tensor_parallel_size"]
        vl_config.gpu_memory_utilization = vl_model_config["gpu_memory_utilization"]

        print(f"[Config] VL configuration applied:")
        print(f"  - Model: {vl_model_key}")
        print(f"  - Path: {vl_config.model_name_or_path}")
        print(f"  - Tensor Parallel Size: {vl_config.tensor_parallel_size}")
        print(f"  - GPU Memory Utilization: {vl_config.gpu_memory_utilization}")
    else:
        print(f"[Config] VL prior disabled (use_prior=False)")
        vl_config = None

    return main_config, create_config, vl_config


if __name__ == "__main__":
    # Test configuration generation
    print("PriorZero VL Configuration")
    print("=" * 80)

    # List available models
    print_available_vl_models()

    # Generate test config
    print("\nGenerating test configuration...")
    main_cfg, create_cfg, vl_cfg = get_priorzero_vl_config(
        env_id='PongNoFrameskip-v4',
        seed=0,
        vl_model_key='qwen-vl-chat',
        use_prior=True,
        quick_test=True,
    )

    print("\n✓ Configuration generated successfully")
    print(f"  - Experiment: {main_cfg.exp_name}")
    print(f"  - Environment: {main_cfg.env.env_id}")
    print(f"  - Observation shape: {main_cfg.policy.model.observation_shape}")
    print(f"  - obs_type: {main_cfg.policy.model.world_model_cfg.obs_type}")
    if vl_cfg:
        print(f"  - VL model: {vl_cfg.model_name_or_path}")
        print(f"  - Use prior: {vl_cfg.use_prior}")
