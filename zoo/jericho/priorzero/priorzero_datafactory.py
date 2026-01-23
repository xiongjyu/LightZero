from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import re
import torch
import torch.distributed as dist
from vllm import SamplingParams
from ding.utils import build_logger
import random
import math

_FMT_RE = re.compile(
    r'^\s*Reasoning:\s*(?P<reason>[\s\S]*?)\nAction:\s*(?P<action>[^\n\r]+)\s*$',
    flags=re.IGNORECASE
)
def _format_reward(text: str) -> int:
    """
    Return 1 if the output strictly matches:
      Reasoning: <any, may contain newlines>
      Action: <one line>
    Otherwise 0.
    """
    if not isinstance(text, str):
        return 0

    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    m = _FMT_RE.match(t)
    if m is None:
        return 0

    if len(re.findall(r'Reasoning:', t, flags=re.IGNORECASE)) != 1:
        return 0
    if len(re.findall(r'Action:', t, flags=re.IGNORECASE)) != 1:
        return 0

    # Action 必须非空（regex 已经用 + 保证非空，这里再保险）
    if m.group("action").strip() == "":
        return 0

    return 1

class DataProcessor:
    """
      - build_llm_prompt / build_chat_context
      - priorzero_batch -> samples
      - (use_cot) 批量生成 prefix_cot
      - vLLM 计算 action prior score（prompt_logprobs）
      - samples -> Dataset/Dataloader（collate_fn 做 pack）
    """

    def __init__(self, rank, world_size, vllm_engine, strategy, model_path, exp_name=None, instance_name="vllm_output", tb_logger=None):
        self.vllm_engine = vllm_engine
        self.strategy = strategy
        self.args = getattr(strategy, "args", None)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.use_cot = self.args.use_cot
        self.prompt_max_len = self.args.prompt_max_len
        self.generate_max_len = self.args.generate_max_len
        self.temperature = self.args.temperature
        self.top_p = self.args.top_p
        self.vllm_enable_sleep = self.args.vllm_enable_sleep
        self.reduction = self.args.reduction
        self.rank = rank
        self.world_size = world_size
        self.output_step = 0
        self.llm_prior_with_cot = False

        from collections import deque
        self.episode_output = []

        # Running statistics for advantage normalization
        self.value_running_mean = 0.0
        self.value_running_std = 1.0
        self.value_count = 0
        self.running_momentum = 0.99  # EMA momentum for running statistics

        # TensorBoard logger for advantage statistics
        self.tb_logger = tb_logger
        self.advantage_norm_step = 0  # Global step counter for advantage normalization

        # Prior temperature scheduler
        self.prior_temp_scheduler = None
        if hasattr(self.args, 'prior_temp_schedule') and self.args.prior_temp_schedule.enable:
            from prior_temperature_scheduler import PriorTemperatureScheduler
            cfg = self.args.prior_temp_schedule
            self.prior_temp_scheduler = PriorTemperatureScheduler(
                init_temperature=cfg.init_temperature,
                final_temperature=cfg.final_temperature,
                total_steps=self.args.max_steps,
                warmup_steps=cfg.warmup_steps,
                schedule_type=cfg.schedule_type,
                decay_rate=cfg.get('decay_rate', 0.95),
                step_size=cfg.get('step_size', 1000),
                step_gamma=cfg.get('step_gamma', 0.8),
                target_entropy=cfg.get('target_entropy', None),
                entropy_window=cfg.get('entropy_window', 100),
                entropy_lr=cfg.get('entropy_lr', 0.01),
                min_temperature=cfg.get('min_temperature', 0.1),
                max_temperature=cfg.get('max_temperature', 5.0),
            )
            if self.rank == 0:
                print(f"[Prior Temperature Scheduler] Initialized with {cfg.schedule_type} schedule: "
                      f"{cfg.init_temperature:.2f} -> {cfg.final_temperature:.2f} over {self.args.max_steps} steps")

        if self.rank == 0:
            self._logger, _ = build_logger(
                path=f'./{exp_name}/log/{instance_name}', name=instance_name, need_tb=False
            )

        if self.args.value_norm_cfg.enable_stability_optimizer:
            from models.stability_optimizer import AdaptiveValueNormalizer
            self.value_normalizer = AdaptiveValueNormalizer(
                init_momentum=self.args.value_norm_cfg.value_norm_init_momentum,
                final_momentum=self.args.value_norm_cfg.value_norm_final_momentum,
                warmup_steps=self.args.value_norm_cfg.value_norm_warmup_steps,
                clip_method=self.args.value_norm_cfg.value_norm_clip_method,
                clip_percentile=self.args.value_norm_cfg.value_norm_clip_percentile,
                min_std=1e-6,
                history_size=self.args.value_norm_cfg.value_norm_history_size,
            )
        else:
            self.value_normalizer = None
    # v1
    # def build_llm_prompt(self, current_obs: str, history: Optional[List[Tuple[str, str, float]]] = None) -> str:
    #     prompt_parts = []
        
    #     # 1. System / Role Definition
    #     prompt_parts.append(
    #         "You are an expert player in a text-based adventure game. "
    #         "Your goal is to maximize the score by choosing the best possible next action. "
    #         "You must choose exactly ONE best next action."
    #     )
        
    #     # 2. History (Context)
    #     if history is not None and len(history) > 0:
    #         history = list(history)
    #         prompt_parts.append("\n=== Recent History ===")
    #         for i, (obs, action, reward) in enumerate(history, start=1):  
    #             obs_str = obs
    #             prompt_parts.append(f"Step {i}:")
    #             prompt_parts.append(f"  Observation: {obs_str.strip()}")
    #             prompt_parts.append(f"  Action: {action.strip()}")
    #             prompt_parts.append(f"  Reward: {reward}")

    #     # 3. Current State
    #     prompt_parts.append("\n=== Current Situation ===")
    #     prompt_parts.append(current_obs.strip())

    #     # 4. Task & Format Instructions
    #     if self.use_cot:
    #         prompt_parts.append(
    #             "\n=== Task ===\n"
    #             "You must produce TWO parts in order: (1) Reasoning, then (2) Action.\n"
    #             "\n"
    #             "### Format Requirements ###\n"
    #             "Your output MUST strictly follow this format:\n"
    #             "Reasoning: <detailed analysis of the state, constraints, and why you are choosing the action>\n"
    #             "Action: <the chosen command>\n"
    #             "\n"
    #             "### Instruction ###\n"
    #             "1. Analyze the 'Current Situation' and 'Recent History'.\n"
    #             "2. In the 'Reasoning' section, explain your thought process. Do NOT output the final command string here, just the logic.\n"
    #             "3. In the 'Action' section, output the specific command to execute.\n"
    #             "### Example ###\n"
    #             "Observation: You are in a dark room. There is a lamp on the table. It is too dark to see the exit.\n"
    #             "Reasoning: The room is currently too dark to navigate safely or find an exit. I see a lamp which can provide light. Therefore, the logical first step is to acquire the light source.\n"
    #             "Action: take lamp\n"
    #             "\n"
    #         )
    #     else:
    #         prompt_parts.append(
    #             "\n=== Task ===\n"
    #             "Analyze the recent history and the current situation, and decide on the SINGLE best next action. "
    #             "Please keep the output concise, avoiding any other content."
    #         )
            
    #     return "\n".join(prompt_parts)



    def build_llm_prompt(self, current_obs: str, history: Optional[List[Tuple[str, str, float]]] = None) -> str:
        prompt_parts = []
        
        # 1. Role & Goal (保持不变，强调唯一性)
        prompt_parts.append(
            "You are an expert player in a text-based adventure game. "
            "Your goal is to maximize the score by choosing the best possible next action. "
            "You must choose exactly ONE best next action."
        )
        
        # 2. History (保持不变)
        if history is not None and len(history) > 0:
            history = list(history)
            prompt_parts.append("\n=== Recent History ===")
            for i, (obs, action, reward) in enumerate(history, start=1):  
                obs_str = obs
                prompt_parts.append(f"Step {i}:")
                prompt_parts.append(f"  Observation: {obs_str.strip()}")
                prompt_parts.append(f"  Action: {action.strip()}")
                prompt_parts.append(f"  Reward: {reward}")

        # 3. Current Situation (保持不变)
        prompt_parts.append("\n=== Current Situation ===")
        prompt_parts.append(current_obs.strip())

        # 4. Task & Format (核心优化部分)
        if self.use_cot:
            prompt_parts.append(
                "\n=== Task ===\n"
                "You must produce TWO parts in order: (1) Reasoning, then (2) Action.\n"
                "\n"
                "### Guidelines ###\n"
                "1. **Analyze**: Look at the history. If you are stuck in a loop (repeating actions), choose a DIFFERENT action (e.g., 'look', 'inventory', 'go north').\n"
                "2. **Sparse Observation**: If the 'Current Situation' is short (e.g., 'Dropped.', 'Done.'), you likely need to re-examine your surroundings. Try 'look'.\n"
                "3. **Conclude**: Your reasoning MUST end with a clear decision on what to do next.\n"
                "\n"
                "### Format Requirements ###\n"
                "Your output MUST strictly follow this format:\n"
                "Reasoning: <analysis of state -> logic -> final decision>\n"
                "Action: <the chosen command>\n"
                "\n"
                "### Example 1 ###\n"
                "Observation: Dropped.\n"
                "Reasoning: The previous action was to drop the paper, and the observation confirms it is dropped. I am currently lacking information about the room. To make a better decision, I need to see what else is here.\n"
                "Action: look\n"
                "### Example 2 ###\n"
                "Observation: You are in a dark room. There is a lamp on the table. It is too dark to see the exit.\n"
                "Reasoning: The room is currently too dark to navigate safely or find an exit. I see a lamp which can provide light. Therefore, the logical first step is to acquire the light source.\n"
                "Action: take lamp\n"
                "\n"
                "### Final Instruction ###\n"
                "Ensure you generate the 'Action:' line immediately after your reasoning."
            )
        else:
            prompt_parts.append(
                "\n=== Task ===\n"
                "Analyze the recent history and the current situation, and decide on the SINGLE best next action. "
                "Please keep the output concise, avoiding any other content."
            )
            
        return "\n".join(prompt_parts)

    # v0

    # def build_llm_prompt(self, current_obs: str, history: Optional[List[Tuple[str, str, float]]] = None) -> str:
    #     prompt_parts = []
    #     prompt_parts.append(
    #         "You are an expert player in a text-based adventure game. "
    #         "Your goal is to maximize the score by choosing the best possible next action. "
    #         "You must choose exactly ONE best next action."
    #     )
    #     if history is not None and len(history) > 0:
    #         history = list(history)
    #         prompt_parts.append("=== Recent History ===")

    #         for i, (obs, action, reward) in enumerate(history, start=1):  
    #             obs_str = obs
    #             prompt_parts.append(f"Step {i}:")
    #             prompt_parts.append(f"  Observation: {obs_str.strip()}")
    #             prompt_parts.append(f"  Action: {action.strip()}")
    #             prompt_parts.append(f"  Reward: {reward}")

    #     prompt_parts.append("=== Current Situation ===")
    #     prompt_parts.append(current_obs.strip())

    #     if self.use_cot:
    #         prompt_parts.append(
    #             "=== Task ==="
    #             "You must produce TWO parts in order: (1) Reasoning, then (2) Action.\n"
    #             "1) Reasoning:\n"
    #             "Perform a detailed reasoning process based ONLY on the current state and the recent interaction history; first analyze what environment or situation you are currently in, then identify what actions are available at this step along with the relevant constraints, and you may also discuss key observations, uncertainties, and implications of different possibilities; however, do NOT state, imply, or reveal which action will be chosen, and the reasoning section MUST be output exactly in the format: Reasoning: <your reasoning content>.\n"
    #             "2) Action:\n"
    #             "After finishing the reasoning, output exactly ONE line in the following format: Action: <the chosen action>." 
    #             "Your output MUST strictly follow this format: \nReasoning: <your reasoning content>\nAction: <the chosen action>"
    #         )
    #     else:
    #         prompt_parts.append(
    #             "\n=== Task ===\n"
    #             "Analyze the recent history and the current situation, and decide on the SINGLE best next action."
    #             "Please keep the output concise, avoiding any other content.\n"
    #         )
    #     return "\n".join(prompt_parts)

    def build_chat_context(self, user_prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_llm_samples(self,
        raw_obs_list: List[List[str]],
        history_obs_list: List[List[List[Tuple[str, str, float]]]],
        action_logprob_list: Optional[List[List[Any]]] = None,
        pred_values: Optional[torch.Tensor] = None,   # [B, T-1] 
        target_values: Optional[torch.Tensor] = None,   # [B, T-1] 
        cot_prefix_list: Optional[List[List[str]]] = None,  # CoT reuse optimization
    ) -> List[Dict[str, Any]]:
        """
        Build training samples from collected data.

        Args:
            raw_obs_list: Raw observations
            history_obs_list: History observations
            action_logprob_list: Action logprobs from collect phase
            target_values: Target values for advantage calculation
            cot_prefix_list: CoT prefixes from collect phase (CoT reuse optimization)

        Returns:
            List of sample dictionaries
        """
        samples: List[Dict[str, Any]] = []
        B = len(raw_obs_list)
        if B == 0:
            return samples
        T = len(raw_obs_list[0])

        for b in range(B):
            for t in range(T - 1):
                current_obs = raw_obs_list[b][t]
                current_hist = history_obs_list[b][t]
                next_hist = history_obs_list[b][t + 1]

                _, true_action, reward_value = next_hist[-1]
                if not true_action:
                    continue

                instruction = self.build_llm_prompt(
                    current_obs=current_obs,
                    history=current_hist,
                )
                prompt = self.build_chat_context(instruction)
                old_logprob = None
                if action_logprob_list is not None:
                    old_logprob = action_logprob_list[b][t + 1][true_action]

                # FIX: Filter samples with extreme logprobs to prevent ratio explosion
                if old_logprob is not None:
                    # Skip if empty
                    if len(old_logprob) == 0:
                        continue
                    # Skip if contains extreme values (< -50 indicates very low probability)
                    if min(old_logprob) < -50.0:
                        continue
                    # Skip if contains NaN/Inf
                    if any(math.isnan(x) or math.isinf(x) for x in old_logprob):
                        continue

                target_value = None
                if target_values is not None:
                    target_value = float(target_values[b][t].item())

                pred_value = None
                if pred_values is not None:
                    pred_value = float(pred_values[b][t].item())

                # FIX: Skip samples with NaN/Inf values
                if target_value is not None and (math.isnan(target_value) or math.isinf(target_value)):
                    continue
                if pred_value is not None and (math.isnan(pred_value) or math.isinf(pred_value)):
                    continue

                # CoT reuse optimization: get CoT prefix from stored data
                # 需要注意的是：game_segment在reset的时候，obs是第一个obs,而cot_prefix是None; 每次append的时候都是next_obs, 和当前obs的cot_prefix
                # 所有cot_prefix应该错位
                prefix_cot = None
                if self.use_cot and cot_prefix_list is not None:
                    prefix_cot = cot_prefix_list[b][t+1]

                samples.append(
                    {
                        "instruction": instruction,
                        "prompt": prompt,
                        "target": true_action,
                        "reward": float(reward_value) if reward_value is not None else 0.0,
                        "pred_value": pred_value,
                        "target_value": target_value,
                        "old_logprob": old_logprob,  # Reinforce++ ratio 需要
                        "prefix_cot": prefix_cot,  # CoT reuse optimization
                    }
                )
        return samples

    def make_llm_train_samples(self, priorzero_batch, ddp: bool = False) -> List[Dict[str, Any]]:
        """
        Convert PriorZero batch to LLM training samples.

        Args:
            priorzero_batch: Tuple of (raw_obs_list, history_obs_list, action_logprob_list, target_value, cot_prefix_list)
                            CoT prefix list is added for CoT reuse optimization.

        Returns:
            Tuple of (input_ids, attention_mask, action_mask, advantages, old_logprob)
        """
        raw_obs_list, history_obs_list, action_logprob_list, target_value, pred_value, cot_prefix_list = priorzero_batch

        assert len(raw_obs_list) == len(history_obs_list) == len(action_logprob_list) == len(target_value) == len(pred_value) == len(cot_prefix_list), \
            f"Batch size mismatch: raw_obs={len(raw_obs_list)}, history_obs={len(history_obs_list)}, action_logprob={len(action_logprob_list)}, target_value={len(target_value)}, cot_prefix={len(cot_prefix_list)}"

        # Build samples with CoT prefixes
        samples = self.build_llm_samples(
            raw_obs_list, history_obs_list, action_logprob_list, pred_value, target_value, cot_prefix_list
        )
        random.shuffle(samples)
        if ddp:
            print(f"[Rank {self.rank}] Processing {len(samples)} samples collected by Rank {self.rank}")
            real_samples = samples
        else:
            per_rank = len(samples) // self.world_size
            start = self.rank * per_rank
            end = (self.rank + 1) * per_rank if self.rank != self.world_size - 1 else len(samples)
            print(f"[Rank {self.rank}] Processing samples {start}:{end} (Total: {len(samples)} samples from Rank 0)")
            real_samples = samples[start:end]
        
        prompts_only = [s["prompt"] for s in real_samples]
        if self.use_cot:
            targets_only = [s["prefix_cot"] + " " + s["target"] + self.tokenizer.eos_token for s in real_samples]
            if self.args.reward_func.format_reward:
                fmt_rewards = torch.tensor([_format_reward(t) for t in targets_only])
            else:
                fmt_rewards = None
        else:
            targets_only = [s["target"] + self.tokenizer.eos_token for s in real_samples]
            fmt_rewards = None

        prompts_ids_list = self.tokenizer(prompts_only, add_special_tokens=False, truncation=True, max_length=self.prompt_max_len - self.generate_max_len - 20)["input_ids"]
        tgt_ids_list = self.tokenizer(targets_only, add_special_tokens=False, truncation=True)["input_ids"]

        full_ids_list = [p + t for p, t in zip(prompts_ids_list, tgt_ids_list)]
        inputs = self.tokenizer.pad({"input_ids": full_ids_list}, padding=True, return_tensors="pt")

        labels = inputs.input_ids.clone()
        labels[inputs.attention_mask == 0] = -100

        for row, p_ids in enumerate(prompts_ids_list):
            pad_len = int((inputs.attention_mask[row] == 0).sum().item())
            real_prompt_len = pad_len + len(p_ids)
            labels[row, :real_prompt_len] = -100

        action_mask_full = (labels != -100).long()
        max_tgt_len = max(len(t) for t in tgt_ids_list)
        action_mask = action_mask_full[:, -max_tgt_len:] 
        log_status_tmp = {}
        log_status = []
        
        if fmt_rewards is not None:
            fmt_weight = self.args.reward_func.format_param.format_weight
            log_status_tmp['fmt_rewards'] = fmt_rewards.tolist()
        
        # t 时刻的 target_value = td_step 步真实 r 的折扣和 + boostrap( t + td_step) 的 v
        target_value = torch.tensor([s["target_value"] for s in real_samples], dtype=torch.float32)
        # t 时刻的 pred_value = boostrap( t ) 的 v
        pred_value = torch.tensor([s["pred_value"] for s in real_samples], dtype=torch.float32)
        advantage = target_value - pred_value

        # FIX: Check for NaN/Inf in advantage before normalization
        if torch.isnan(advantage).any() or torch.isinf(advantage).any():
            if self.rank == 0:
                print(f"[WARNING] Advantage contains NaN/Inf values, replacing with zeros")
            advantage = torch.where(torch.isnan(advantage) | torch.isinf(advantage),
                                   torch.zeros_like(advantage), advantage)

        if self.args.advantage_type == "advantage":
            advantage = advantage
            log_status_tmp["advantage"] = advantage.tolist()
            if fmt_rewards is not None:
                advantage = (1 - fmt_weight) * advantage + fmt_weight * fmt_rewards

        elif self.args.advantage_type == "target_reward":
            advantage = torch.tensor([s["reward"] for s in real_samples], dtype=torch.float32)
            log_status_tmp["advantage"] = advantage.tolist()
            if fmt_rewards is not None:
                advantage = (1 - fmt_weight) * advantage + fmt_weight * fmt_rewards

        elif self.args.advantage_type == "advantage_batch_norm":
            # Legacy implementation: batch normalization (not recommended)
            # Calculate batch statistics before normalization
            batch_mean = advantage.mean().item()
            batch_std = advantage.std().item()
            batch_min = advantage.min().item()
            batch_max = advantage.max().item()
            batch_size = advantage.numel()

            # FIX: Increase epsilon for numerical stability
            advantage_normalized = (advantage - advantage.mean()) / (advantage.std() + 1e-4)

            # Calculate normalized statistics
            norm_min = advantage_normalized.min().item()
            norm_max = advantage_normalized.max().item()
            norm_mean = advantage_normalized.mean().item()
            norm_std = advantage_normalized.std().item()

            # Print detailed statistics
            if self.rank == 0:
                print(
                    f"[Advantage Batch Norm] step={self.advantage_norm_step} | "
                    f"batch_size={batch_size} | "
                    f"raw: mean={batch_mean:.3f}, std={batch_std:.3f}, min={batch_min:.3f}, max={batch_max:.3f} | "
                    f"norm: mean={norm_mean:.3f}, std={norm_std:.3f}, min={norm_min:.3f}, max={norm_max:.3f}"
                )

            # Log to TensorBoard
            if self.tb_logger is not None and self.rank == 0:
                self.tb_logger.add_scalar("advantage_norm/batch_norm/raw_mean", batch_mean, self.advantage_norm_step)
                self.tb_logger.add_scalar("advantage_norm/batch_norm/raw_std", batch_std, self.advantage_norm_step)
                self.tb_logger.add_scalar("advantage_norm/batch_norm/raw_min", batch_min, self.advantage_norm_step)
                self.tb_logger.add_scalar("advantage_norm/batch_norm/raw_max", batch_max, self.advantage_norm_step)
                self.tb_logger.add_scalar("advantage_norm/batch_norm/batch_size", batch_size, self.advantage_norm_step)
                self.tb_logger.add_scalar("advantage_norm/batch_norm/norm_mean", norm_mean, self.advantage_norm_step)
                self.tb_logger.add_scalar("advantage_norm/batch_norm/norm_std", norm_std, self.advantage_norm_step)
                self.tb_logger.add_scalar("advantage_norm/batch_norm/norm_min", norm_min, self.advantage_norm_step)
                self.tb_logger.add_scalar("advantage_norm/batch_norm/norm_max", norm_max, self.advantage_norm_step)

            advantage = advantage_normalized
            log_status_tmp["advantage"] = advantage.tolist()

            if fmt_rewards is not None:
                advantage = (1 - fmt_weight) * advantage + fmt_weight * fmt_rewards

            self.advantage_norm_step += 1

        elif self.args.advantage_type == "advantage_running_norm":
            if self.value_normalizer is not None:
                # Store raw advantage statistics before normalization
                raw_mean = advantage.mean().item()
                raw_std = advantage.std().item()
                raw_min = advantage.min().item()
                raw_max = advantage.max().item()
                batch_size = advantage.numel()

                advantage, norm_stats = self.value_normalizer.normalize(
                    advantage,
                    clip_values=True,
                    return_stats=True
                )

                # Calculate normalized statistics
                norm_min = advantage.min().item()
                norm_max = advantage.max().item()
                norm_mean = advantage.mean().item()
                norm_std = advantage.std().item()

                if self.rank == 0 and self.value_normalizer.update_count % 10 == 0:
                    print(
                        f"[Value Norm] step={self.value_normalizer.update_count} | "
                        f"batch_size={batch_size} | "
                        f"running: mean={norm_stats['running_mean']:.3f}, std={norm_stats['running_std']:.3f} | "
                        f"batch: mean={norm_stats['batch_mean']:.3f}, std={norm_stats['batch_std']:.3f} | "
                        f"raw: min={raw_min:.3f}, max={raw_max:.3f} | "
                        f"norm: min={norm_min:.3f}, max={norm_max:.3f} | "
                        f"clipped={norm_stats['clipped_count']}/{norm_stats['total_count']} | "
                        f"momentum={norm_stats['momentum']:.3f}"
                    )

                # Log to TensorBoard
                if self.tb_logger is not None and self.rank == 0:
                    step = self.value_normalizer.update_count
                    self.tb_logger.add_scalar("advantage_norm/running_norm/raw_mean", raw_mean, step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/raw_std", raw_std, step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/raw_min", raw_min, step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/raw_max", raw_max, step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/batch_size", batch_size, step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/running_mean", norm_stats['running_mean'], step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/running_std", norm_stats['running_std'], step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/batch_mean", norm_stats['batch_mean'], step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/batch_std", norm_stats['batch_std'], step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/norm_mean", norm_mean, step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/norm_std", norm_std, step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/norm_min", norm_min, step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/norm_max", norm_max, step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/clipped_count", norm_stats['clipped_count'], step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/clipped_ratio", norm_stats['clipped_count'] / max(norm_stats['total_count'], 1), step)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/momentum", norm_stats['momentum'], step)
            else:
                batch_mean = advantage.mean().item()
                batch_std = advantage.std().item()
                batch_min = advantage.min().item()
                batch_max = advantage.max().item()
                batch_size = advantage.numel()

                if self.value_count == 0:
                    self.value_running_mean = batch_mean
                    self.value_running_std = max(batch_std, 1e-8)  # Avoid zero std
                else:
                    self.value_running_mean = (
                        self.running_momentum * self.value_running_mean +
                        (1 - self.running_momentum) * batch_mean
                    )
                    self.value_running_std = (
                        self.running_momentum * self.value_running_std +
                        (1 - self.running_momentum) * max(batch_std, 1e-8)
                    )

                self.value_count += 1
                # FIX: Increase epsilon for numerical stability
                advantage = (advantage - self.value_running_mean) / (self.value_running_std + 1e-4)

                # Calculate normalized statistics
                norm_min = advantage.min().item()
                norm_max = advantage.max().item()
                norm_mean = advantage.mean().item()
                norm_std = advantage.std().item()

                if self.rank == 0 and self.value_count % 10 == 0:
                    print(
                        f"[Advantage Running Norm] step={self.value_count} | "
                        f"batch_size={batch_size} | "
                        f"running: mean={self.value_running_mean:.3f}, std={self.value_running_std:.3f} | "
                        f"batch: mean={batch_mean:.3f}, std={batch_std:.3f} | "
                        f"raw: min={batch_min:.3f}, max={batch_max:.3f} | "
                        f"norm: min={norm_min:.3f}, max={norm_max:.3f}"
                    )

                # Log to TensorBoard
                if self.tb_logger is not None and self.rank == 0:
                    self.tb_logger.add_scalar("advantage_norm/running_norm/raw_mean", batch_mean, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/raw_std", batch_std, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/raw_min", batch_min, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/raw_max", batch_max, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/batch_size", batch_size, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/running_mean", self.value_running_mean, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/running_std", self.value_running_std, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/batch_mean", batch_mean, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/batch_std", batch_std, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/norm_mean", norm_mean, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/norm_std", norm_std, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/norm_min", norm_min, self.value_count)
                    self.tb_logger.add_scalar("advantage_norm/running_norm/norm_max", norm_max, self.value_count)

            log_status_tmp["advantage"] = advantage.tolist()
            if fmt_rewards is not None:
                advantage = (1 - fmt_weight) * advantage + fmt_weight * fmt_rewards
        else:
            raise ValueError(f"Unknown advantage_type: {self.args.advantage_type}")
        
        log_status = [
            {k: log_status_tmp[k][i] for k in log_status_tmp.keys()} for i in range(len(log_status_tmp['advantage']))
        ]
        
        old_seq_max_len = max([len(s['old_logprob']) for s in real_samples])
        old_logprob = torch.zeros(len(real_samples), old_seq_max_len, dtype=torch.float32)
        for idx in range(len(real_samples)):
            logprob_token_list = real_samples[idx]['old_logprob']
            old_logprob[idx, -len(logprob_token_list):] = torch.tensor(logprob_token_list, dtype=torch.float32)


        return inputs.input_ids, inputs.attention_mask, action_mask, advantage, old_logprob, log_status
        
    @torch.no_grad()
    def _build_cot_prefix_texts(self, all_user_prompts: List[str]) -> List[str]:
        """
        生成CoT推理前缀。
        优化: 使用较短的max_tokens(128)和stop条件以减少不必要的生成。
        从最后一次出现的 "Action:" 截断出 prefix（包含 Action: 和其后的空格位置）。
        返回 prefix_cot_list，与 all_user_prompts 等长。
        """
        cot_sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=self.generate_max_len, 
            stop=["\n\n"],
            # stop=["Action:", "\n\n"]
            include_stop_str_in_output=True,
            logprobs=None,
            prompt_logprobs=None,
        )

        all_context_texts = [self.build_chat_context(p) for p in all_user_prompts]
        context_token_ids = self.tokenizer(
            all_context_texts,
            add_special_tokens=False,
            max_length=self.prompt_max_len,
            padding=False,
            truncation=True,
        )["input_ids"]

        self.vllm_engine.add_requests(sampling_params=cot_sampling_params, prompt_token_ids=context_token_ids)
        cot_outputs = self.vllm_engine.get_responses()

        prefix_cot_list, full_output = [], []
        for output in cot_outputs:
            gen_text = output.outputs[0].text
            full_output.append(gen_text)
            
            matches = list(re.finditer(r"(?mi)^\s*Action\s*:\s*", gen_text))
            if not matches:
                matches = list(re.finditer(r"action\s*:\s*", gen_text, flags=re.IGNORECASE))

            if not matches:
                prefix_cot_list.append("")
                continue

            m = matches[-1]
            prefix_piece = gen_text[: m.end()].strip() 
            
            prefix_cot_list.append(prefix_piece)

        return prefix_cot_list, full_output
    
    @torch.no_grad()
    def get_llm_prior(
        self,
        states: List[str],
        valid_actions_list: List[List[str]],
        histories: Optional[List[List[Tuple[str, str, float]]]] = None,
        return_cot: bool = False,  # CoT reuse optimization: return CoT prefixes
    ) -> List[Any]:
        """
        Get LLM prior scores for actions.

        Args:
            states: List of current state observations
            valid_actions_list: List of valid actions for each state
            histories: List of history observations
            return_cot: If True, return CoT prefixes for reuse (optimization)

        Returns:
            If return_cot=False: (llm_prior_per_seq, llm_prior_per_tok)
            If return_cot=True: (llm_prior_per_seq, llm_prior_per_tok, prefix_cots)
        """
        prompt_list = []
        assert len(states) == len(histories) == len(valid_actions_list)
        for state, history in zip(states, histories):
            prompt = self.build_llm_prompt(current_obs=state, history=history)
            prompt_list.append(prompt)

        if self.use_cot:
            prefix_cots, full_output = self._build_cot_prefix_texts(prompt_list)
        else:
            prefix_cots = [None] * len(prompt_list)
            full_output = None

        all_prompts = []
        all_labels = []
        all_prefix_cots = []

        for prompt, actions, prefix in zip(prompt_list, valid_actions_list, prefix_cots):
            actions2 = actions if "go" in actions else (actions + ["go"])   # 确保环境使用的动作都在valid actions里有对应的logprob
            for action in actions2:
                all_prompts.append(prompt)
                all_labels.append(action)
                all_prefix_cots.append(prefix)

        scores, old_action_logprob = self._score_labels_with_prompt_logprobs(all_prompts, all_labels, all_prefix_cots)
        llm_prior_per_seq, llm_prior_per_tok, idx = [],[], 0

        for prompt, actions, prefix in zip(prompt_list, valid_actions_list, prefix_cots):
            actions2 = actions if "go" in actions else (actions + ["go"])
            tmp_dict = {}
            tmp_dict2 = {}
            for action in actions2:
                tmp_dict[action] = scores[idx]
                tmp_dict2[action] = old_action_logprob[idx]
                idx = idx + 1
            llm_prior_per_seq.append(tmp_dict)
            llm_prior_per_tok.append(tmp_dict2)

        if self.use_cot:
            self.episode_output.append({
                "Instruction": prompt_list[0],
                "Response": full_output[0],
                "llm_prior_per_seq": llm_prior_per_seq[0]
            })
        # CoT reuse optimization: return CoT prefixes if requested
        if return_cot:
            return llm_prior_per_seq, llm_prior_per_tok, prefix_cots
        else:
            return llm_prior_per_seq, llm_prior_per_tok

    @torch.no_grad()
    def _score_labels_with_prompt_logprobs(self, all_prompts: List[str], all_labels: List[str], all_prefix_cots: List[str]) -> List[float]:
        assert len(all_prompts) == len(all_labels) == len(all_prefix_cots)

        # Get current temperature from scheduler
        current_temperature = self.temperature  # Default temperature
        if self.prior_temp_scheduler is not None:
            current_temperature = self.prior_temp_scheduler.get_temperature()

        sampling_params = SamplingParams(
            temperature=self.temperature,  # Keep original for generation
            top_p=self.top_p,
            max_tokens=1,
            include_stop_str_in_output=True,
            logprobs=None,
            prompt_logprobs=1,
        )

        all_context_texts = [self.build_chat_context(p) for p in all_prompts]
        context_ids = self.tokenizer(all_context_texts, add_special_tokens=False, max_length=self.prompt_max_len - self.generate_max_len - 20, padding=False, truncation=True)["input_ids"]

        if self.use_cot:
            label_texts = [pc + " " + l + self.tokenizer.eos_token for pc, l in zip(all_prefix_cots, all_labels)]
            label_texts_no_cots =  [" " + l + self.tokenizer.eos_token for l in all_labels]
        else:
            label_texts = [l + self.tokenizer.eos_token for l in all_labels]
            label_texts_no_cots = label_texts

        label_ids = self.tokenizer(label_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        label_ids_no_cots = self.tokenizer(label_texts_no_cots, add_special_tokens=False, padding=False, truncation=False)["input_ids"]

        full_ids = [c + l for c, l in zip(context_ids, label_ids)]
        p_lens = [len(x) for x in context_ids]
        l_lens = [len(x) for x in label_ids]
        l_no_cots_lens = [len(x) for x in label_ids_no_cots]

        self.vllm_engine.add_requests(sampling_params=sampling_params, prompt_token_ids=full_ids)
        outs = self.vllm_engine.get_responses()

        scores = []
        old_action_logprob = []
        all_entropies = []  # Track entropies for adaptive scheduling

        for out, ids, p_len, l_len, l_no_cots_len in zip(outs, full_ids, p_lens, l_lens, l_no_cots_lens):
            prompt_logprobs = getattr(out, "prompt_logprobs", None)

            token_lps = []
            for j in range(p_len, p_len + l_len):
                tok_id = ids[j]
                lp_dict = prompt_logprobs[j]
                if tok_id not in lp_dict:
                    # FIX: Use finite value instead of -inf to prevent ratio explosion
                    # -100 corresponds to probability ~3.7e-44, small enough but won't cause exp(+inf)
                    token_lps.append(-100.0)
                    if self.rank == 0:
                        self._logger.info(f"tok_id not in lp_dict: -100 corresponds to probability ~3.7e-44, small enough but won't cause exp(+inf)")
                else:
                    token_lps.append(lp_dict[tok_id].logprob)

            if not token_lps:
                scores.append(-100.0)
                if self.rank == 0:
                    self._logger.info(f"not token_lps: -100 corresponds to probability ~3.7e-44, small enough but won't cause exp(+inf)")
                old_action_logprob.append([])
            else:
                assert l_no_cots_len <= l_len

                # IMPORTANT: Keep original logprobs for PPO training
                original_token_lps = token_lps.copy()

                # Apply temperature scaling ONLY for prior scores (MCTS root)
                if current_temperature != 1.0:
                    import torch
                    token_lps_tensor = torch.tensor(token_lps, dtype=torch.float32)

                    # Apply temperature scaling
                    scaled_lps = token_lps_tensor / current_temperature

                    # Compute entropy before normalization (for adaptive scheduling)
                    probs = torch.exp(scaled_lps)
                    entropy = -(probs * scaled_lps).sum().item()
                    all_entropies.append(entropy)

                    # Renormalize to maintain valid probability distribution
                    scaled_lps = torch.log_softmax(scaled_lps, dim=0)

                    # Use scaled logprobs for prior scores
                    scaled_token_lps = scaled_lps.tolist()
                else:
                    # No temperature scaling
                    scaled_token_lps = token_lps

                # Prior scores: use temperature-scaled logprobs for MCTS exploration
                if self.llm_prior_with_cot:
                    scores.append(sum(scaled_token_lps) if self.reduction == "sum" else sum(scaled_token_lps) / l_len)
                else:
                    scores.append(sum(scaled_token_lps[-l_no_cots_len:]) if self.reduction == "sum" else sum(scaled_token_lps[-l_no_cots_len:]) / l_no_cots_len)

                # Old action logprob: use ORIGINAL logprobs for PPO training
                old_action_logprob.append(original_token_lps)

        # Update temperature scheduler with average entropy
        if self.prior_temp_scheduler is not None and all_entropies:
            avg_entropy = sum(all_entropies) / len(all_entropies)
            new_temperature = self.prior_temp_scheduler.step(entropy=avg_entropy)

            # Log temperature and entropy statistics
            if self.rank == 0 and self.prior_temp_scheduler.current_step % 10 == 0:
                stats = self.prior_temp_scheduler.get_stats()
                print(
                    f"[Prior Temperature] step={stats['temperature_step']} | "
                    f"temp={stats['prior_temperature']:.3f} | "
                    f"entropy={avg_entropy:.3f} | "
                    f"progress={stats['temperature_progress']:.2%}"
                )

                # Log to TensorBoard
                if self.tb_logger is not None:
                    step = stats['temperature_step']
                    self.tb_logger.add_scalar("prior_temp/temperature", stats['prior_temperature'], step)
                    self.tb_logger.add_scalar("prior_temp/entropy", avg_entropy, step)
                    self.tb_logger.add_scalar("prior_temp/progress", stats['temperature_progress'], step)
                    if 'prior_avg_entropy' in stats:
                        self.tb_logger.add_scalar("prior_temp/avg_entropy", stats['prior_avg_entropy'], step)
                    if 'prior_target_entropy' in stats:
                        self.tb_logger.add_scalar("prior_temp/target_entropy", stats['prior_target_entropy'], step)
                    if 'prior_entropy_gap' in stats:
                        self.tb_logger.add_scalar("prior_temp/entropy_gap", stats['prior_entropy_gap'], step)

        return scores, old_action_logprob

    @torch.no_grad()
    def get_llm_output_log(self, wm_train_iter: int = 0, llm_train_iter: int = 0):
        if self.rank != 0:
            return

        # Structured LLM output log header
        self._logger.info(
            f"\n{'='*80}\n"
            f"[LLM Output Log] WM Iter: {wm_train_iter} | LLM Iter: {llm_train_iter}\n"
            f"{'='*80}"
        )

        for i, tmp_dict in enumerate(self.episode_output[:15]):
            instruction = tmp_dict["Instruction"]
            response = tmp_dict["Response"]
            llm_prior = tmp_dict["llm_prior_per_seq"]

            # Structured step log
            self._logger.info(
                f"\n{'-'*80}\n"
                f"[Step {i}]\n"
                f"{'-'*80}\n"
                f"Instruction:\n{instruction}\n\n"
                f"Response:\n{response}\n\n"
                f"Action Probabilities:"
            )

            action_probs = {a: math.exp(float(lp)) for a, lp in llm_prior.items() if lp is not None and math.isfinite(float(lp))}
            all_prob = sum(action_probs.values())

            for action, prob in sorted(action_probs.items(), key=lambda x: x[1], reverse=True):
                self._logger.info(f"  {action:30s} | unnorm={prob:.6f} | norm={(prob / all_prob):.6f}")
            self._logger.info(f"  {'<other>':30s} | unnorm={1-all_prob:.6f}")

        self.episode_output = []

        
        