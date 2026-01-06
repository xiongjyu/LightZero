import sys
import os
from pathlib import Path

# ==============================================================================
# 假设当前脚本在 .../zoo/jericho/priorzero/ 目录下
current_file_path = Path(__file__).resolve()
# 回退 4 层找到 LightZero 根目录 (priorzero -> jericho -> zoo -> LightZero)
project_root = current_file_path.parents[3] 

if str(project_root) not in sys.path:
    print(f"[SYSTEM] Inserting project root to sys.path: {project_root}")
    sys.path.insert(0, str(project_root))
# ==============================================================================


import asyncio
import os
import sys
from functools import partial
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.distributed as dist
import wandb

from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank, get_world_size
from ding.worker import create_buffer, BaseLearner
from tensorboardX import SummaryWriter
from loguru import logger
import deepspeed

from priorzero_config import (
    get_priorzero_config,
    get_priorzero_debug_config,
    get_available_models,
)
from priorzero_collector import PriorZeroCollector
from priorzero_evaluator import PriorZeroEvaluator
from priorzero_policy import *
from lzero.mcts.buffer.game_buffer_priorzero import PriorZeroGameBufferOptimized


from lzero.entry.utils import calculate_update_per_collect

def prepare_unizero(rank, cfg, create_cfg, llm_cfg, seed, data_processor=None):
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    
    policy = create_policy( cfg.policy, enable_field=['learn', 'collect', 'eval'], exp_name=cfg.exp_name)
    logger.info(f"[Rank {rank}]  Policy created")

    os.makedirs(f'./{cfg.exp_name}/log/', exist_ok=True)
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial')) if get_rank() == 0 else None
    logger.info(f"[Rank {rank}] TensorBoard logger: ./{cfg.exp_name}/log/")
    
    learner = BaseLearner(
        cfg.policy.learn.learner,
        policy.learn_mode,
        tb_logger,
        exp_name=cfg.exp_name
    )
    logger.info(f"[Rank {rank}] BaseLearner created")

    
    replay_buffer = PriorZeroGameBufferOptimized(cfg.policy)
    logger.info(f"[Rank {rank}] PriorZero replay buffer created (with game_segments support)")

    # Create collector
    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        llm_config=llm_cfg,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        data_processor=data_processor,
        policy_config=cfg.policy,
    )
    logger.info(f"[Rank {rank}] Collector created")

    # Create evaluator
    evaluator = PriorZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
    )
    logger.info(f"[Rank {rank}] Evaluator created")
    learner.call_hook('before_run')

    return cfg, replay_buffer, tb_logger, policy, collector, evaluator, learner
    
def bcast_obj(world_size, obj, rank, src=0):
    if world_size <= 1:
        return obj
    lst = [obj] if rank == src else [None]
    dist.broadcast_object_list(lst, src=src)
    return lst[0]    

def train_priorzero(
    cfg: dict,
    create_cfg: dict,
    llm_cfg,
    seed: int = 0,
    max_train_iter: int = int(1e6),
    max_env_step: Optional[int] = int(1e10),
):
    rank = int(os.environ.get("RANK", "0"))
    print(f"rank={rank}")

    # ============================================================================
    # REVERT: Only rank 0 creates World Model components (original design)
    # ============================================================================
    if rank == 0:
        cfg, replay_buffer, tb_logger, policy, collector, evaluator, learner = prepare_unizero(
            rank=rank,
            cfg=cfg,
            create_cfg=create_cfg,
            llm_cfg=llm_cfg,
            seed=seed,
            data_processor=None
        )
        batch_size = cfg.policy.batch_size
        logger.info(f"[Rank {rank}] World Model components initialized")
    # ============================================================================

    from strategy.deepspeed import get_strategy, torch_dist_barrier_and_cuda_sync
    strategy = get_strategy(llm_cfg)
    strategy.print(llm_cfg)
    
    strategy.setup_distributed()   # torchrun 下：绑定 local_rank + init_distributed
    world_size = getattr(strategy, "world_size", 1)
    
    logger.info(f"[Rank {rank}] Initializing LLM Actor...")
    set_pkg_seed(seed + rank, use_cuda=True)
    
    from models.actor import PolicyModel, ReferenceModel
    if llm_cfg.rft_kl_coef > 0:
        ref_model = ReferenceModel(
            strategy=strategy,
            pretrain=llm_cfg.model_name_or_path
        )
    else:
        ref_model = None
    
    from vllm_utils.vllm_engine import create_vllm_engine
    vllm_engine = create_vllm_engine(
        tensor_parallel_size=llm_cfg.vllm_tensor_parallel_size,
        pretrain=llm_cfg.model_name_or_path,
        enable_prefix_caching=llm_cfg.enable_prefix_caching,
        max_model_len=llm_cfg.prompt_max_len + llm_cfg.generate_max_len,
        gpu_memory_utilization=llm_cfg.gpu_memory_utilization,
        vllm_enable_sleep=llm_cfg.vllm_enable_sleep,
    )

    print(f'[Rank {rank}] Vllm engine successfully created!')
    
    from priorzero_datafactory import DataProcessor
    data_processor = DataProcessor(rank=rank, 
                                   world_size=world_size,
                                   vllm_engine=vllm_engine, 
                                   strategy=strategy, 
                                   model_path=llm_cfg.model_name_or_path,
                                   exp_name=cfg.exp_name if rank == 0 else None,
                                   )
    if rank == 0:
        collector.data_processor = data_processor
    
    policy_model = PolicyModel(
        strategy=strategy,
        pretrain=llm_cfg.model_name_or_path,
        vllm_engine=vllm_engine
    )
    from priorzero_trainer import PriorZeroLLMTrainer
    trainer = PriorZeroLLMTrainer(
        cfg=llm_cfg,
        pretrain=llm_cfg.model_name_or_path,
        strategy= strategy,
        vllm_engine = vllm_engine,
        policy_model=policy_model,
        reference_model=ref_model,
        broadcast_every=llm_cfg.broadcast_every,
        exp_name=cfg.exp_name if rank == 0 else None,
        tb_logger=tb_logger if rank == 0 else None,
    )

    # ============================================================================
    # Stability Optimizer Integration
    # ============================================================================
    stability_optimizer = None
    enable_stability_opt = getattr(llm_cfg, 'enable_stability_optimizer', False)

    if rank == 0 and enable_stability_opt:
        try:
            # from priorzero_stability_optimizer import PriorZeroStabilityOptimizer
            from priorzero_stability_optimizer_enhanced import PriorZeroStabilityOptimizer

            stability_optimizer = PriorZeroStabilityOptimizer(
                # Value normalization
                value_norm_init_momentum=getattr(llm_cfg, 'value_norm_init_momentum', 0.9),
                value_norm_final_momentum=getattr(llm_cfg, 'value_norm_final_momentum', 0.99),
                value_norm_warmup_steps=getattr(llm_cfg, 'value_norm_warmup_steps', 100),
                value_norm_clip_percentile=getattr(llm_cfg, 'value_norm_clip_percentile', 0.95),

                # Curriculum training
                wm_warmup_steps=getattr(llm_cfg, 'wm_warmup_steps', 500),
                wm_warmup_update_ratio=getattr(llm_cfg, 'wm_warmup_update_ratio', 5),
                llm_start_threshold=getattr(llm_cfg, 'llm_start_threshold', 0.3),
                min_llm_update_interval=getattr(llm_cfg, 'min_llm_update_interval', 1),
                max_llm_update_interval=getattr(llm_cfg, 'max_llm_update_interval', 10),

                # Stability monitoring
                grad_norm_threshold=getattr(llm_cfg, 'grad_norm_threshold', 100.0),
                loss_spike_threshold=getattr(llm_cfg, 'loss_spike_threshold', 3.0),
                value_shift_threshold=getattr(llm_cfg, 'value_shift_threshold', 2.0),

                # Logging
                log_interval=getattr(llm_cfg, 'stability_log_interval', 10),
                rank=rank,
            )
            logger.info("="*80)
            logger.info("✓ PriorZero Stability Optimizer initialized")
            logger.info(f"  - WM warmup steps: {llm_cfg.wm_warmup_steps}")
            logger.info(f"  - LLM start threshold: {llm_cfg.llm_start_threshold}")
            logger.info(f"  - Value norm warmup: {llm_cfg.value_norm_warmup_steps}")
            logger.info("="*80)
        except ImportError as e:
            logger.warning(f"⚠ Failed to import PriorZeroStabilityOptimizer: {e}")
            logger.warning("  Falling back to original training logic")
            stability_optimizer = None
    # ============================================================================

    torch_dist_barrier_and_cuda_sync()

    while True:
        cmd = "noop"
        priorzero_batch = None
        if rank == 0:
            if learner.train_iter > 0 and evaluator.should_eval(learner.train_iter):
                logger.info(f"\n[Rank {rank}: Iter {learner.train_iter}] Evaluating...")
                stop, reward = evaluator.eval(
                    save_ckpt_fn=learner.save_checkpoint,
                    train_iter=learner.train_iter,
                    envstep=collector.envstep
                )
                if stop:
                    cmd = "stop"

            if cmd != "stop":
                if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
                    vllm_engine.wake_up()

                new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'temperature': 0.25, 'epsilon': 0.0})
                data_processor.get_llm_output_log()

                if llm_cfg.vllm_enable_sleep and vllm_engine is not None:
                    vllm_engine.sleep()

                update_per_collect = calculate_update_per_collect(cfg, new_data, world_size=1)

                replay_buffer.push_game_segments(new_data)
                replay_buffer.remove_oldest_data_to_fit()

                num_of_transitions = replay_buffer.get_num_of_transitions()
                new_num_of_transitions = replay_buffer.get_num_of_transitions() - replay_buffer.last_pos_in_transition
                logger.info(f"[Rank {rank}] Data collected, num_of_transitions: {num_of_transitions} transitions\tnew_num_of_transitions: {new_num_of_transitions}")

                if not (num_of_transitions > batch_size):
                    logger.warning(
                        f'  ⚠ Data in replay_buffer is not sufficient: '
                        f'batch_size: {batch_size}, replay_buffer: {replay_buffer}. Continue to collect...'
                    )
                    cmd = "noop"
                    # IMPORTANT FIX: Broadcast cmd before continue to prevent deadlock
                    cmd = bcast_obj(world_size, cmd, rank, src=0)
                    continue

                # ============================================================================
                # World Model Training with Stability Optimizer
                # ============================================================================
                should_train_wm = True
                wm_update_count = update_per_collect

                if stability_optimizer is not None:
                    should_train_wm, _ = stability_optimizer.should_train_world_model(
                        new_data_collected=True
                    )

                if should_train_wm:
                    logger.info(f"[Rank {rank}: World Model] [Iter {learner.train_iter}] Training for {wm_update_count} updates...")

                    # Track WM training metrics
                    wm_losses = []
                    wm_grad_norms = []

                    for i in range(wm_update_count):
                        train_data = replay_buffer.sample(batch_size, policy)
                        train_data.append(learner.train_iter)

                        log_vars = learner.train(train_data, collector.envstep)

                        # Collect metrics
                        wm_losses.append(log_vars[0].get('wm_total_loss', 0.0))
                        wm_grad_norms.append(log_vars[0].get('wm_grad_norm', 0.0))

                        if cfg.policy.use_priority:
                            replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

                    policy.recompute_pos_emb_diff_and_clear_cache()

                    # Record WM metrics to stability optimizer
                    if stability_optimizer is not None and len(wm_losses) > 0:
                        import numpy as np
                        avg_wm_loss = np.mean(wm_losses)
                        avg_wm_grad_norm = np.mean(wm_grad_norms)

                        value_pred_error = None
                        if 'wm_target_value' in log_vars[0] and 'predicted_value' in log_vars[0]:
                            target_val = log_vars[0]['wm_target_value']
                            pred_val = log_vars[0].get('predicted_value', target_val)
                            if abs(target_val) > 1e-6:
                                value_pred_error = abs(pred_val - target_val) / (abs(target_val) + 1e-8)

                        stability_optimizer.record_training_metrics(
                            wm_loss=avg_wm_loss,
                            wm_grad_norm=avg_wm_grad_norm,
                        )
                else:
                    logger.info(f"[Rank {rank}: World Model] [Iter {learner.train_iter}] Skipped (stability optimizer decision)")

                # ============================================================================
                # LLM Training Decision with Stability Optimizer
                # ============================================================================
                should_train_llm_orig = new_num_of_transitions >= llm_cfg.llm_learn_num_samples
                should_train_llm = should_train_llm_orig
                llm_decision_info = None

                if stability_optimizer is not None:
                    value_pred_error = None
                    if len(wm_losses) > 0 and 'wm_target_value' in log_vars[0]:
                        target_val = log_vars[0]['wm_target_value']
                        pred_val = log_vars[0].get('predicted_value', target_val)
                        if abs(target_val) > 1e-6:
                            value_pred_error = abs(pred_val - target_val) / (abs(target_val) + 1e-8)

                    should_train_llm, llm_decision_info = stability_optimizer.should_train_llm(
                        new_samples=new_num_of_transitions,
                        value_prediction_error=value_pred_error,
                    )

                    if not should_train_llm:
                        logger.info(f"[Rank {rank}: LLM Training] Skipped - {llm_decision_info.get('reason', 'Unknown')}")
                    else:
                        logger.info(f"[Rank {rank}: LLM Training] Proceeding - {llm_decision_info.get('reason', 'Conditions met')}")
                else:
                    should_train_llm = should_train_llm_orig

                if should_train_llm:
                    logger.info(f"[Rank {rank}] Fetching latest batch for LLM training...")
                    priorzero_batch = replay_buffer.fetch_latest_batch(batch_size=llm_cfg.llm_learn_num_samples, policy=policy)
                    cmd = "llm"
                # ============================================================================

                if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
                    cmd = "stop"

        cmd = bcast_obj(world_size, cmd, rank, src=0)
        if cmd == "stop":
            break
        elif cmd == "llm":
            logger.info(f"[Rank {rank}] Waiting for broadcast of train_samples from Rank 0...")
            priorzero_batch = bcast_obj(world_size, priorzero_batch, rank, src=0)
            logger.info(f"[Rank {rank}] Received broadcast. train_samples count: {len(priorzero_batch[0]) if priorzero_batch and len(priorzero_batch) > 0 else 'UNKNOWN'}. Starting LLM training...")
            train_samples = data_processor.make_llm_train_samples(priorzero_batch)

            # Train LLM and get status
            llm_status = trainer.train_batch(train_samples)

            # Record LLM metrics to stability optimizer (rank 0 only)
            if rank == 0 and stability_optimizer is not None and llm_status:
                stability_optimizer.record_training_metrics(
                    llm_loss=llm_status.get('policy_loss', None),
                    llm_grad_norm=llm_status.get('grad_norm', None),  # If available
                )

            torch_dist_barrier_and_cuda_sync()

        # ============================================================================
        # Stability Optimizer Step and TensorBoard Logging
        # ============================================================================
        if rank == 0 and stability_optimizer is not None:
            stability_optimizer.step()

            # Record stability metrics to TensorBoard periodically
            if learner.train_iter % 10 == 0 and tb_logger is not None:
                tb_metrics = stability_optimizer.get_tb_metrics()
                for metric_name, metric_value in tb_metrics.items():
                    tb_logger.add_scalar(metric_name, metric_value, learner.train_iter)
        # ============================================================================
            

def main():
    """
    Main entry point with argument parsing.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='PriorZero Training with Auto Model Configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model (qwen2.5-1.5b)
  torchrun --nproc_per_node 2 priorzero_entry_sync.py

  # Use specific model
  torchrun --nproc_per_node 2 priorzero_entry_sync.py --model qwen2.5-0.5b
  torchrun --nproc_per_node 2 priorzero_entry_sync.py --model qwen2.5-7b

  # List all available models
  python priorzero_entry_sync.py --list-models

  # Different environment
  torchrun --nproc_per_node 2 priorzero_entry_sync.py --env_id zork1.z5 --model qwen2.5-1.5b
        """
    )
    parser.add_argument('--env_id', type=str, default='detective.z5', help='Jericho game ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--max_iter', type=int, default=int(1e6), help='Max training iterations')
    parser.add_argument('--quick_test', action='store_true', default=False, help='Use quick test config')
    # Model selection
    parser.add_argument('--model', type=str, default="qwen2.5-3b", choices=get_available_models())

    args = parser.parse_args()

    model_key = args.model if args.model else "qwen2.5-1.5b"
    print(f"\n{'='*80}")
    print(f"PriorZero Training Configuration")
    print(f"{'='*80}")
    print(f"Environment: {args.env_id}")
    print(f"Model: {model_key}")
    print(f"Seed: {args.seed}")
    print(f"Quick Test: {args.quick_test}")
    print(f"{'='*80}\n")

    use_cot = True 
    if args.quick_test:
        logger.info("Using quick test configuration")
        main_cfg, create_cfg, llm_cfg = get_priorzero_debug_config(
            args.env_id, args.seed, use_cot=use_cot,
            exp_name=f'data_priorzero/priorzero_sync_debug_{args.env_id}_seed0',
            model_key=model_key
        )
    else:
        main_cfg, create_cfg, llm_cfg = get_priorzero_config(
            args.env_id, args.seed, use_cot=use_cot,
            exp_name=f'data_priorzero/priorzero_ppo_{args.env_id}_seed0',
            model_key=model_key
        )

    train_priorzero(
        main_cfg,
        create_cfg,
        llm_cfg,
        seed=args.seed,
        max_train_iter=args.max_iter,
    )


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
