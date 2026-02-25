"""
Complete PriorZero Entry with VLM Support

This is the COMPLETE implementation with full training loop.
Supports both text (LLM) and image (VLM) inputs.
"""
import sys
import os
from pathlib import Path

# Add project root to path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
if str(project_root) not in sys.path:
    print(f"[SYSTEM] Inserting project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))

import argparse
from functools import partial
from typing import Tuple, Optional, List

import torch
import torch.distributed as dist
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter
from loguru import logger

from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized
from lzero.entry.utils import calculate_update_per_collect


def all_gather_cmd(world_size, obj) -> List:
    """Gather command from all ranks."""
    if world_size <= 1:
        return [obj]
    lst = [None] * dist.get_world_size()
    dist.all_gather_object(lst, obj)
    return lst


def prepare_common_components(rank, cfg, create_cfg, seed):
    """Prepare components common to both LLM and VLM."""
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Create environments
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)

    # Create policy
    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'], exp_name=cfg.exp_name)
    logger.info(f"[Rank {rank}] Policy created")

    # Create logger and learner
    os.makedirs(f'./{cfg.exp_name}/log/', exist_ok=True)
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial')) if rank == 0 else None
    logger.info(f"[Rank {rank}] TensorBoard logger: ./{cfg.exp_name}/log/")

    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    logger.info(f"[Rank {rank}] BaseLearner created")

    # Create replay buffer
    replay_buffer = PriorZeroGameBufferOptimized(cfg.policy)
    logger.info(f"[Rank {rank}] PriorZero replay buffer created")

    return cfg, collector_env, evaluator_env, policy, learner, replay_buffer, tb_logger


def prepare_llm_components(rank, cfg, llm_cfg, strategy, collector_env, evaluator_env, policy, tb_logger, seed):
    """Prepare LLM-specific components for text input."""
    from utils import Profiler, dump_dataclass_cfg_py
    from models.actor import PolicyModel, ReferenceModel
    from vllm_utils.vllm_engine import create_vllm_engine
    from priorzero_datafactory_unified import UnifiedDataProcessor
    from priorzero_trainer import PriorZeroLLMTrainer
    from priorzero_collector_unified import PriorZeroCollector
    from priorzero_evaluator import PriorZeroEvaluator
    from prior_generator import LLMPriorGenerator

    prof = Profiler(log_interval=10, stats_file=f'./{cfg.exp_name}/log/profiler.txt', enable_profile=False)

    if rank == 0:
        dump_dataclass_cfg_py(llm_cfg, path=f"{cfg.exp_name}/llm_cfg.py")
        llm_cfg.save_path = f'./{cfg.exp_name}/llm_ckpt/'

    logger.info(f"[Rank {rank}] Initializing LLM components...")
    set_pkg_seed(seed + rank, use_cuda=True)

    # Reference model
    ref_model = ReferenceModel(strategy=strategy, pretrain=llm_cfg.model_name_or_path) if llm_cfg.rft_kl_coef > 0 else None

    # vLLM engine
    vllm_engine = create_vllm_engine(
        tensor_parallel_size=llm_cfg.vllm_tensor_parallel_size,
        pretrain=llm_cfg.model_name_or_path,
        enable_prefix_caching=llm_cfg.enable_prefix_caching,
        max_model_len=llm_cfg.prompt_max_len + llm_cfg.generate_max_len,
        gpu_memory_utilization=llm_cfg.gpu_memory_utilization,
        vllm_enable_sleep=llm_cfg.vllm_enable_sleep,
    )
    logger.info(f'[Rank {rank}] vLLM engine created')

    # Data processor
    world_size = getattr(strategy, "world_size", 1)
    data_processor = UnifiedDataProcessor(
        rank=rank,
        world_size=world_size,
        vllm_engine=vllm_engine,
        strategy=strategy,
        model_path=llm_cfg.model_name_or_path,
        exp_name=cfg.exp_name if rank == 0 else None,
        obs_type='text',
    )

    # Policy model
    policy_model = PolicyModel(
        strategy=strategy,
        pretrain=llm_cfg.model_name_or_path,
        vllm_engine=vllm_engine,
        max_steps=llm_cfg.max_steps
    )

    # Trainer
    trainer = PriorZeroLLMTrainer(
        cfg=llm_cfg,
        pretrain=llm_cfg.model_name_or_path,
        strategy=strategy,
        vllm_engine=vllm_engine,
        policy_model=policy_model,
        reference_model=ref_model,
        exp_name=cfg.exp_name if rank == 0 else None,
        tb_logger=tb_logger if rank == 0 else None,
        llm_save_freq=llm_cfg.llm_save_freq
    )

    # Prior generator
    prior_generator = LLMPriorGenerator(
        vllm_engine=vllm_engine,
        data_processor=data_processor,
        model_name=llm_cfg.model_name_or_path,
        use_cot=llm_cfg.use_cot,
    )

    # Collector
    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        llm_config=llm_cfg,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
        data_processor=data_processor,
        prior_generator=prior_generator,
        obs_type='text',
    )
    collector.prof = prof

    # Evaluator
    evaluator = PriorZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
        llm_config=llm_cfg,
        data_processor=data_processor,
    )

    logger.info(f"[Rank {rank}] ✓ LLM components initialized")

    return {
        'prior_generator': prior_generator,
        'vllm_engine': vllm_engine,
        'policy_model': policy_model,
        'ref_model': ref_model,
        'trainer': trainer,
        'data_processor': data_processor,
        'collector': collector,
        'evaluator': evaluator,
        'prof': prof,
    }


def prepare_vlm_components(rank, cfg, vlm_cfg, strategy, collector_env, evaluator_env, policy, tb_logger, seed):
    """Prepare VLM-specific components for image input."""
    from utils import Profiler, dump_dataclass_cfg_py
    from models.actor import PolicyModel, ReferenceModel
    from vlm_engine import create_vlm_engine
    from priorzero_datafactory_unified import UnifiedDataProcessor
    from priorzero_trainer import PriorZeroLLMTrainer  # Can reuse for VLM
    from priorzero_collector_unified import PriorZeroCollector
    from priorzero_evaluator import PriorZeroEvaluator
    from prior_generator import VLMPriorGenerator

    prof = Profiler(log_interval=10, stats_file=f'./{cfg.exp_name}/log/profiler.txt', enable_profile=False)

    if rank == 0:
        dump_dataclass_cfg_py(vlm_cfg, path=f"{cfg.exp_name}/vlm_cfg.py")
        vlm_cfg.save_path = f'./{cfg.exp_name}/vlm_ckpt/'

    logger.info(f"[Rank {rank}] Initializing VLM components...")
    set_pkg_seed(seed + rank, use_cuda=True)

    # Reference model
    ref_model = ReferenceModel(strategy=strategy, pretrain=vlm_cfg.model_name_or_path) if vlm_cfg.rft_kl_coef > 0 else None

    # VLM engine
    vlm_engine = create_vlm_engine(
        model_name=vlm_cfg.vlm_model_type,
        model_path=vlm_cfg.model_name_or_path,
        tensor_parallel_size=vlm_cfg.tensor_parallel_size,
        gpu_memory_utilization=vlm_cfg.gpu_memory_utilization,
    )
    logger.info(f'[Rank {rank}] VLM engine created: {vlm_cfg.vlm_model_type}')

    # Data processor
    world_size = getattr(strategy, "world_size", 1)
    data_processor = UnifiedDataProcessor(
        rank=rank,
        world_size=world_size,
        vllm_engine=vlm_engine,
        strategy=strategy,
        model_path=vlm_cfg.model_name_or_path,
        exp_name=cfg.exp_name if rank == 0 else None,
        obs_type='image',
    )

    # Policy model
    policy_model = PolicyModel(
        strategy=strategy,
        pretrain=vlm_cfg.model_name_or_path,
        vllm_engine=vlm_engine,
        max_steps=vlm_cfg.max_steps
    )

    # Trainer
    trainer = PriorZeroLLMTrainer(
        cfg=vlm_cfg,
        pretrain=vlm_cfg.model_name_or_path,
        strategy=strategy,
        vllm_engine=vlm_engine,
        policy_model=policy_model,
        reference_model=ref_model,
        exp_name=cfg.exp_name if rank == 0 else None,
        tb_logger=tb_logger if rank == 0 else None,
        llm_save_freq=vlm_cfg.vlm_save_freq
    )

    # Prior generator
    prior_generator = VLMPriorGenerator(
        vlm_engine=vlm_engine,
        model_name=vlm_cfg.model_name_or_path,
        prompt_template=vlm_cfg.prompt_template,
    )

    # Collector
    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        llm_config=vlm_cfg,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
        data_processor=data_processor,
        prior_generator=prior_generator,
        obs_type='image',
    )
    collector.prof = prof

    # Evaluator
    evaluator = PriorZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
        llm_config=vlm_cfg,
        data_processor=data_processor,
    )

    logger.info(f"[Rank {rank}] ✓ VLM components initialized")

    return {
        'prior_generator': prior_generator,
        'vlm_engine': vlm_engine,
        'policy_model': policy_model,
        'ref_model': ref_model,
        'trainer': trainer,
        'data_processor': data_processor,
        'collector': collector,
        'evaluator': evaluator,
        'prof': prof,
    }


def train_unified(
    cfg: dict,
    create_cfg: dict,
    prior_cfg,  # LLM or VLM config
    seed: int = 0,
    max_train_iter: int = int(1e6),
    max_env_step: Optional[int] = int(1e10),
    enable_profile: bool = False,
    is_text_input: bool = True,
):
    """
    Unified training function supporting both LLM and VLM.

    Args:
        cfg: Main configuration
        create_cfg: Creation configuration
        prior_cfg: LLM or VLM configuration
        seed: Random seed
        max_train_iter: Maximum training iterations
        max_env_step: Maximum environment steps
        enable_profile: Whether to enable profiling
        is_text_input: Whether using text input (True) or image input (False)
    """
    rank = int(os.environ.get("RANK", "0"))

    # Initialize strategy
    from strategy.deepspeed import get_strategy, torch_dist_barrier_and_cuda_sync
    strategy = get_strategy(prior_cfg)
    strategy.print(prior_cfg)
    strategy.setup_distributed()
    world_size = getattr(strategy, "world_size", 1)

    # Prepare common components
    cfg, collector_env, evaluator_env, policy, learner, replay_buffer, tb_logger = prepare_common_components(
        rank, cfg, create_cfg, seed
    )
    batch_size = cfg.policy.batch_size

    # Prepare input-specific components
    if is_text_input:
        components = prepare_llm_components(
            rank, cfg, prior_cfg, strategy, collector_env, evaluator_env, policy, tb_logger, seed
        )
        engine_name = "vLLM"
    else:
        components = prepare_vlm_components(
            rank, cfg, prior_cfg, strategy, collector_env, evaluator_env, policy, tb_logger, seed
        )
        engine_name = "VLM"

    # Extract components
    prior_engine = components['vllm_engine'] if is_text_input else components['vlm_engine']
    policy_model = components['policy_model']
    trainer = components['trainer']
    data_processor = components['data_processor']
    collector = components['collector']
    evaluator = components['evaluator']
    prof = components['prof']

    torch_dist_barrier_and_cuda_sync()
    learner.call_hook('before_run')

    logger.info(f"[Rank {rank}] Starting training loop with {engine_name} prior...")

    # =========================================================================
    # Main Training Loop
    # =========================================================================
    while True:
        cmd = 0
        priorzero_batch = None

        # Evaluation
        if learner.train_iter > 0 and evaluator.should_eval(learner.train_iter):
            logger.info(f"\n[Rank {rank}: Iter {learner.train_iter}] Evaluating...")
            stop, reward = evaluator.eval(
                save_ckpt_fn=learner.save_checkpoint,
                train_iter=learner.train_iter,
                envstep=collector.envstep
            )

        # Wake up engine
        if prior_cfg.vllm_enable_sleep and prior_engine is not None:
            prior_engine.wake_up()

        # Data collection
        with prof.block("collect", rank=rank):
            new_data = collector.collect(
                train_iter=learner.train_iter,
                policy_kwargs={'temperature': 0.25, 'epsilon': 0.0}
            )
        data_processor.get_llm_output_log(
            wm_train_iter=learner.train_iter,
            llm_train_iter=policy_model.train_iter
        )

        # Sleep engine
        if prior_cfg.vllm_enable_sleep and prior_engine is not None:
            prior_engine.sleep()

        # Calculate updates
        update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=world_size)

        # Push to replay buffer
        replay_buffer.push_game_segments(new_data)
        replay_buffer.remove_oldest_data_to_fit()

        num_of_transitions = replay_buffer.get_num_of_transitions()
        new_num_of_transitions = replay_buffer.get_num_of_transitions() - replay_buffer.last_pos_in_transition

        logger.info(
            f"[Data Collection] Rank {rank} | "
            f"Total transitions: {num_of_transitions} | "
            f"New transitions: {new_num_of_transitions}"
        )

        # Check if we have enough data
        if not (num_of_transitions > batch_size):
            logger.warning(
                f'  ⚠ Data insufficient: batch_size={batch_size}, buffer={num_of_transitions}'
            )
            cmd = 0
        else:
            cmd = 1

        if min(all_gather_cmd(world_size=world_size, obj=cmd)) == 0:
            continue

        # =====================================================================
        # World Model Training
        # =====================================================================
        logger.info(
            f"[World Model Training] Rank {rank} | Iter {learner.train_iter} | "
            f"Updates: {update_per_collect}"
        )

        for i in range(update_per_collect):
            with prof.block("train_world_model", rank=rank):
                train_data = replay_buffer.sample(batch_size, policy)
                train_data.append(learner.train_iter)
                log_vars = learner.train(train_data, collector.envstep)
                if cfg.policy.use_priority:
                    replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        policy.recompute_pos_emb_diff_and_clear_cache()

        # =====================================================================
        # LLM/VLM Training
        # =====================================================================
        llm_need_sample_cnt = prior_cfg.train_batch_size * prior_cfg.broadcast_every // world_size
        llm_need_transition_cnt = (llm_need_sample_cnt + cfg.policy.num_unroll_steps - 1) // cfg.policy.num_unroll_steps

        if learner.train_iter >= prior_cfg.train_vlm_after_wm_warm_step and new_num_of_transitions >= llm_need_transition_cnt:
            cmd = 1
        else:
            cmd = 0

        # Check stopping criteria
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            cmd = 2

        all_cmd = all_gather_cmd(world_size=world_size, obj=cmd)
        if max(all_cmd) == 2:
            break
        elif min(all_cmd) == 1:
            with prof.block("fetch_latest_batch", rank=rank):
                logger.info(f"[Batch Fetch] Rank {rank} | Required transitions: {llm_need_transition_cnt}")
                priorzero_batch = replay_buffer.fetch_latest_batch(batch_size=llm_need_transition_cnt, policy=policy)

            with prof.block("train_prior_model", rank=rank):
                sample_count = len(priorzero_batch[0]) if priorzero_batch and len(priorzero_batch) > 0 else 0
                logger.info(f"[{engine_name} Training] Rank {rank} | Samples: {sample_count}")

                train_samples = data_processor.make_llm_train_samples(priorzero_batch, ddp=True)
                trainer.train_batch(train_samples, collect_env_steps=collector.envstep)

                torch_dist_barrier_and_cuda_sync()
        else:
            continue

    logger.info(f"[Rank {rank}] Training completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='PriorZero with VLM Support')

    # Common arguments
    parser.add_argument('--input_type', type=str, required=True, choices=['text', 'image'])
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=int(1e6))
    parser.add_argument('--quick_test', action='store_true', default=False)
    parser.add_argument('--enable_profile', action='store_true', default=False)

    # Text-specific
    parser.add_argument('--llm_model', type=str, default='qwen2.5-1.5b')
    parser.add_argument('--use_cot', action='store_true', default=True)

    # Image-specific
    parser.add_argument('--vlm_model', type=str, default='Qwen2.5-VL-7b')
    parser.add_argument('--use_prior', action='store_true', default=True)

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"PriorZero Training with {'LLM' if args.input_type == 'text' else 'VLM'} Prior")
    print(f"{'='*80}")
    print(f"Input Type: {args.input_type}")
    print(f"Environment: {args.env_id}")
    print(f"Seed: {args.seed}")
    print(f"Quick Test: {args.quick_test}")
    print(f"{'='*80}\n")

    if args.input_type == 'text':
        from priorzero_config import get_priorzero_config, get_priorzero_debug_config

        if args.quick_test:
            main_cfg, create_cfg, llm_cfg = get_priorzero_debug_config(
                args.env_id, args.seed, use_cot=args.use_cot,
                exp_name=f'data_priorzero_complete/text_{args.env_id}_seed{args.seed}',
                model_key=args.llm_model,
            )
        else:
            main_cfg, create_cfg, llm_cfg = get_priorzero_config(
                args.env_id, args.seed, use_cot=args.use_cot,
                exp_name=f'data_priorzero_complete/text_{args.env_id}_seed{args.seed}',
                model_key=args.llm_model,
                multi_gpu=True
            )

        train_unified(
            main_cfg, create_cfg, llm_cfg,
            seed=args.seed,
            max_train_iter=args.max_iter,
            enable_profile=args.enable_profile,
            is_text_input=True,
        )

    else:
        from vlm_config import get_priorzero_vlm_config

        main_cfg, create_cfg, vlm_cfg = get_priorzero_vlm_config(
            args.env_id, args.seed,
            exp_name=f'data_priorzero_complete/image_{args.env_id[:-14]}_seed{args.seed}',
            vlm_model_key=args.vlm_model,
            use_prior=args.use_prior,
            multi_gpu=False,
            quick_test=args.quick_test,
        )

        train_unified(
            main_cfg, create_cfg, vlm_cfg,
            seed=args.seed,
            max_train_iter=args.max_iter,
            enable_profile=args.enable_profile,
            is_text_input=False,
        )


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
