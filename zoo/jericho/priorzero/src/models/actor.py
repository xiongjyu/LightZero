from contextlib import contextmanager
from typing import Optional, Union, List, Dict
from collections import defaultdict
import os
import math
from tqdm import tqdm
import numpy as np
import deepspeed
from torch.optim import Optimizer
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.trainer import get_scheduler

from utils import compute_approx_kl, compute_entropy, masked_mean, torch_dist_barrier_and_cuda_sync, log_probs_from_logits


import hashlib
import torch

def _tensor_digest(t: torch.Tensor, max_elems: int = 4096):
    x = t.detach().float().contiguous().view(-1)
    if x.numel() > max_elems:
        x = x[:max_elems]
    return hashlib.md5(x.cpu().numpy().tobytes()).hexdigest()

def _param_signature(param: torch.Tensor):
    x = param.detach()
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "digest": _tensor_digest(x),
    }

def _compare_signature_dict(sig_a, sig_b, max_print=20, title="COMPARE"):
    keys_a = set(sig_a.keys())
    keys_b = set(sig_b.keys())

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    if only_a:
        print(f"[{title}] only in A: {only_a[:10]}")
    if only_b:
        print(f"[{title}] only in B: {only_b[:10]}")

    mismatch = 0
    for k in sorted(keys_a & keys_b):
        a = sig_a[k]
        b = sig_b[k]
        if a["shape"] != b["shape"] or a["digest"] != b["digest"]:
            print(f"[{title}] mismatch: {k}")
            print(f"  A: {a}")
            print(f"  B: {b}")
            mismatch += 1
            if mismatch >= max_print:
                print(f"[{title}] too many mismatches, stop early")
                break

    ok = (len(only_a) == 0 and len(only_b) == 0 and mismatch == 0)
    print(f"[{title}] ok={ok}, mismatch={mismatch}, only_a={len(only_a)}, only_b={len(only_b)}")
    return ok


def _normalize_vllm_weight_name(name: str) -> str:
    if name.startswith("base_model.model."):
        name = name[len("base_model.model."):]
    name = name.replace(".base_layer.", ".")
    return name


def _should_skip_vllm_sync_param(name: str) -> bool:
    return any(marker in name for marker in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"))


def _validate_vllm_sync_config(args, train_mode: str, vllm_engine) -> None:
    if vllm_engine is None:
        return

    ds_tensor_parallel_size = getattr(args, "ds_tensor_parallel_size", 1)
    zero_stage = getattr(args, "zero_stage", 2)

    if ds_tensor_parallel_size != 1:
        raise NotImplementedError(
            "PolicyModel._deepspeed_broadcast currently supports only ds_tensor_parallel_size == 1. "
            f"Got ds_tensor_parallel_size={ds_tensor_parallel_size}. "
            "The active vLLM sync path does not safely handle DeepSpeed tensor parallel shards yet."
        )

    if zero_stage == 3 and train_mode == "lora":
        raise NotImplementedError(
            "PolicyModel._deepspeed_broadcast does not support train_mode='lora' with zero_stage=3. "
            "This path needs adapter merge/unmerge together with ZeRO-3 sharded parameters, which is not "
            "validated in the current implementation."
        )

class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        attn_implementation (str, optional): Attention mechanism implementation to use. Defaults to "flash_attention_2".
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
    """

    def __init__(
        self,
        pretrain_or_model: str,
        attn_implementation="flash_attention_2",
        bf16=True,
        ds_config=None,
        device_map=None,
        temperature=1.0,
        train_mode_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.temperature = temperature
        self.pretrain_or_model = pretrain_or_model
        self.train_mode_cfg = train_mode_cfg if train_mode_cfg is not None else {"mode": "full"}
        self.train_mode = self.train_mode_cfg.get("mode", "full")
        attn_impl = attn_implementation

        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            _ = HfDeepSpeedConfig(ds_config)
        else:
            _ = None

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrain_or_model,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            device_map=device_map,
        )
        self.model.config.use_cache = False

        if self.train_mode == "lora":
            self.model.enable_input_require_grads()
            target_modules = self.train_mode_cfg.get("lora_target_modules")
            target_modules = list(target_modules) if target_modules else None
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.train_mode_cfg.get("lora_r", 16),
                lora_alpha=self.train_mode_cfg.get("lora_alpha", 32),
                lora_dropout=self.train_mode_cfg.get("lora_dropout", 0.05),
                bias=self.train_mode_cfg.get("lora_bias", "none"),
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
        elif self.train_mode != "full":
            raise ValueError(f"Unsupported train_mode: {self.train_mode}")
        
        self.model.config.use_cache = False

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        return_entropy=False,
    ) -> torch.Tensor:

        foward_attention_mask = attention_mask
        rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids)
        output["logits"] = output["logits"].to(torch.float32)

        if return_entropy:
            assert return_output
            entropy = compute_entropy(output["logits"])
            setattr(output, "entropy", entropy[:, :-1])

        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)

        log_probs = log_probs[:, :-1]

        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()
        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()

class ReferenceModel:
    def __init__(self, strategy, pretrain):
        self.strategy = strategy
        model = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            ds_config=strategy.get_ds_eval_config(
                offload=False
            ),
            temperature=strategy.args.temperature,
        )
        self.model = strategy.prepare(model, is_rlhf=True)
        self.model.eval()
        self.micro_train_batch_size = self.strategy.args.micro_train_batch_size
    
    @torch.no_grad()
    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return: action_log_probs [B, T_action]
        """
        device = torch.cuda.current_device()
        B = sequences.size(0)
        outs = []
        chunk_size = max(self.micro_train_batch_size, 32)
        
        sequences = sequences.to(device)
        attention_mask = attention_mask.to(device)
        action_mask = action_mask.to(device) 
        for i in range(0, B, chunk_size):
            s = sequences[i : i + chunk_size].to(device)
            am = action_mask[i : i + chunk_size].to(device)
            attn = attention_mask[i : i + chunk_size].to(device)

            out = self.model(
                s,
                action_mask=am,
                attention_mask=attn,
            )  
            outs.append(out)
        return torch.cat(outs, dim=0)

class BatchPPOTrainer:
    def __init__(
        self,
        strategy,
        actor,
        actor_optim,
        actor_scheduler,               
        micro_train_batch_size: int = 8,
        vllm_engine = None
    ):
        self.strategy = strategy
        self.args = strategy.args

        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.vllm_engine = vllm_engine
        self.use_cuda_ipc = self.args.use_cuda_ipc

        self.micro_train_batch_size = micro_train_batch_size
        from models.loss import PolicyLoss
        self.policy_loss = PolicyLoss(
            clip_eps_low=self.args.eps_clip_low_high[0],
            clip_eps_high=self.args.eps_clip_low_high[1],
            policy_loss_type=self.args.policy_loss_type,
        )
        self.train_iter = 0
        self._install_vllm_weight_recorder()

    def _install_vllm_weight_recorder(self):
        if self.vllm_engine is None:
            return
        if hasattr(self.vllm_engine, "_weight_recorder_installed"):
            return

        self.vllm_engine._broadcasted_weight_cache = {}
        original_update_weight = self.vllm_engine.update_weight

        def wrapped_update_weight(name, dtype, shape, weight, empty_cache=False, **kwargs):
            self.vllm_engine._broadcasted_weight_cache[name] = {
                "shape": tuple(shape),
                "dtype": str(dtype),
                "digest": _tensor_digest(weight),
            }
            return original_update_weight(
                name=name,
                dtype=dtype,
                shape=shape,
                weight=weight,
                empty_cache=empty_cache,
                **kwargs,
            )

        self.vllm_engine.update_weight = wrapped_update_weight
        self.vllm_engine._weight_recorder_installed = True


    def _collect_actor_sync_signature(self):
        model = self.actor.model.module if hasattr(self.actor.model, "module") else self.actor.model
        sig = {}
        with self._merged_lora_adapter(model):
            for name, param in self._iter_vllm_sync_params(model):
                sig[name] = _param_signature(param)
        return sig


    def compare_actor_vs_vllm_broadcasted(self, tag="WEIGHT_CHECK"):
        if self.vllm_engine is None:
            print(f"[{tag}] vllm_engine is None")
            return False

        actor_sig = self._collect_actor_sync_signature()
        vllm_sig = getattr(self.vllm_engine, "_broadcasted_weight_cache", None)

        if not vllm_sig:
            print(f"[{tag}] no cached weights in vllm yet")
            return False

        return _compare_signature_dict(actor_sig, vllm_sig, title=tag)

    def compute_logprob_from_vllm(self, action_log_probs, sequences: torch.LongTensor, action_mask: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.vllm_engine.wake_up()
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,                       
            logprobs=None,
            prompt_logprobs=5,
        )
        full_ids = []
        for seq, attn in zip(sequences, attention_mask):
            ids = seq[attn.bool()].tolist()
            full_ids.append(ids)
        
        action_lengths = action_mask.sum(dim=-1).tolist()
        
        self.vllm_engine.add_requests(sampling_params=sampling_params, prompt_token_ids=full_ids)
        outs = self.vllm_engine.get_responses()

        old_action_logprob = []
        old_full_logprob = []
        for i, (out, ids, action_len) in enumerate(zip(outs, full_ids, action_lengths)):            
            prompt_logprobs = getattr(out, "prompt_logprobs", None)
            token_lps = []
            
            for j in range(1, len(ids)):
                tok_id = ids[j]
                lp_dict = prompt_logprobs[j]
                
                assert tok_id in lp_dict
                token_lps.append(lp_dict[tok_id].logprob)

            old_action_logprob.append(token_lps[-action_len :])
            old_full_logprob.append(token_lps)
        max_len = max(action_lengths)
        result = torch.tensor(
            [[0.0]*(max_len - len(x)) + x for x in old_action_logprob],
            dtype=torch.float32
        )
        return result, old_full_logprob
        
    def train_batch(self, batch_data: Dict[str, torch.Tensor], kl_ctl: float, step_idx: int = 0) -> Dict[str, float]:
        device = torch.cuda.current_device()
        for k, v in batch_data.items():
            if torch.is_tensor(v):
                batch_data[k] = v.to(device)
        
        all_samples_size = batch_data["input_ids"].size(0)
        status_list = []
        pbar = tqdm(
            range(0, all_samples_size, self.micro_train_batch_size),
            desc=f"PPO batch step={step_idx}",
            disable=not self.strategy.is_rank_0(),
        )
        acc_grad_steps = self.strategy.accumulated_gradient 
        metrics_buffer = defaultdict(list) # 用于累积 micro_step 指标的缓冲区
        
        for micro_step, start_idx in enumerate(pbar):
            end_idx = min(start_idx + self.micro_train_batch_size, all_samples_size)
            micro_batch = {
                'input_ids': batch_data['input_ids'][start_idx:end_idx],
                "attention_mask": batch_data['attention_mask'][start_idx:end_idx],
                "action_mask": batch_data['action_mask'][start_idx:end_idx],
                "advantages": batch_data['advantages'][start_idx:end_idx],
                "old_action_logprob": batch_data['old_action_logprob'][start_idx:end_idx],
                "log_status": batch_data['log_status'][start_idx:end_idx]
            }
            micro_batch['ref_action_log_probs'] = batch_data['ref_action_log_probs'][start_idx:end_idx] if batch_data['ref_action_log_probs'] is not None else None
            action_log_probs, output = self.actor(
                micro_batch['input_ids'],
                micro_batch['action_mask'],
                attention_mask=micro_batch['attention_mask'],
                return_output=True,
                return_entropy=True,
            )
            # vllm_logprob, vllm_full_logprob = self.compute_logprob_from_vllm(
            #     action_log_probs=action_log_probs,
            #     sequences=micro_batch['input_ids'],
            #     action_mask=micro_batch['action_mask'],
            #     attention_mask=micro_batch['attention_mask']
            # )
            actor_loss, clipfrac, clip_ratio, approx_kl, vllm_kl = self.policy_loss(
                action_log_probs,
                micro_batch['old_action_logprob'],
                micro_batch['advantages'],
                action_mask=micro_batch['action_mask'],
            )
            
            if self.args.rft_kl_coef > 0 and micro_batch['ref_action_log_probs'] is not None:
                kl = compute_approx_kl(
                    action_log_probs,
                    micro_batch['ref_action_log_probs'],
                    kl_estimator=self.args.kl_estimator
                )
                kl_loss = masked_mean(kl, micro_batch["action_mask"])
            else:
                kl_loss = torch.tensor(0.0, device=device)
            
            loss = actor_loss + kl_loss * float(kl_ctl.value)
            
            entropy_loss = masked_mean(output.entropy[:, -micro_batch["action_mask"].shape[1] :], micro_batch["action_mask"])
            if self.args.entropy_loss_coef != 0:
                loss -= entropy_loss * self.args.entropy_loss_coef  
            
            self.strategy.backward(loss, self.actor, self.actor_optim)
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
            
            policy_loss_item = actor_loss.detach().float().item()
            clipfrac_item = clipfrac.detach().float().item()
            clip_ratio_item = clip_ratio.detach().float().item()
            approx_kl_item = approx_kl.detach().float().item()
            kl_loss_item = kl_loss.detach().float().item()
            input_response_length_item = micro_batch["attention_mask"].sum().detach().float().item() / micro_batch["attention_mask"].shape[0]
            response_length_item = micro_batch["action_mask"].sum().detach().float().item() / micro_batch["action_mask"].shape[0]
            input_length_item = input_response_length_item - response_length_item
            entropy_loss_item = entropy_loss.detach().float().item()
            
            pbar.set_postfix({
                "policy_loss": policy_loss_item,
                "clipfrac": clipfrac_item,
                "approx_kl": approx_kl_item,
                "iter": self.train_iter,
            })
            
            metrics_buffer["policy_loss"].append(policy_loss_item)
            metrics_buffer["clipfrac"].append(clipfrac_item)
            metrics_buffer["clip_ratio"].append(clip_ratio_item)
            metrics_buffer["approx_kl"].append(approx_kl_item)
            metrics_buffer["ref_kl"].append(kl_loss_item)
            metrics_buffer["input_length"].append(input_length_item)
            metrics_buffer["response_length"].append(response_length_item)
            metrics_buffer['entropy'].append(entropy_loss_item)

            log_status = micro_batch["log_status"]
            other_status = {k: [item[k] for item in log_status] for k in log_status[0].keys()}
            for k, v in other_status.items():
                metrics_buffer[k] = v
        
            if ((micro_step + 1) % acc_grad_steps == 0) or ((micro_step + 1) == pbar.total):
                self.train_iter += 1
                status = {
                    "policy_loss": np.mean(metrics_buffer['policy_loss']),
                    "clipfrac": np.mean(metrics_buffer['clipfrac']),
                    "clip_ratio": np.mean(metrics_buffer['clip_ratio']),
                    "approx_kl": np.mean(metrics_buffer['approx_kl']),
                    "ref_kl": np.mean(metrics_buffer['ref_kl']),
                    "entropy": np.mean(metrics_buffer['entropy']),
                    
                    "iter": self.train_iter,
                    "lr": self.actor_scheduler.get_last_lr()[0],
                    "global_grad_norm": self.actor_optim._global_grad_norm,

                    "input_length_max": np.max(metrics_buffer['input_length']),
                    "input_length_mean": np.mean(metrics_buffer['input_length']),
                    "input_length_min": np.min(metrics_buffer['input_length']),

                    "response_length_max": np.max(metrics_buffer['response_length']),
                    "response_length_mean": np.mean(metrics_buffer['response_length']),
                    "response_length_min": np.min(metrics_buffer['response_length']),

                    "value_advantage_max": np.max(metrics_buffer['value_advantage']),
                    "value_advantage_mean": np.mean(metrics_buffer['value_advantage']),
                    "value_advantage_min": np.min(metrics_buffer['value_advantage']),
                }
                if "final_advantage" in metrics_buffer:
                    status["final_advantage_max"] = np.max(metrics_buffer['final_advantage'])
                    status["final_advantage_mean"] = np.mean(metrics_buffer['final_advantage'])
                    status["final_advantage_min"] = np.min(metrics_buffer['final_advantage'])
                if "fmt_rewards" in metrics_buffer:
                    status["fmt_rewards"] = np.mean(metrics_buffer['fmt_rewards'])
                metrics_buffer.clear()

                status = self.strategy.all_reduce(status)
                status_list.append(status)

        return status_list
    
    def _deepspeed_broadcast(self):
        _validate_vllm_sync_config(self.strategy.args, self.actor.train_mode, self.vllm_engine)
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        if use_prefix_cache:
            self.vllm_engine.reset_prefix_cache()

        torch.cuda.empty_cache()
        model = self.actor.model.module
        with self._merged_lora_adapter(model):
            sync_params = list(self._iter_vllm_sync_params(model))
            count, num_params = 0, len(sync_params)
            for name, param in sync_params:
                count += 1  # empty_cache at last param
                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    self.vllm_engine.update_weight(name, dtype=param.dtype, shape=shape, weight=param.data, empty_cache=(count == num_params)) 
    
    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            self.vllm_engine.reset_prefix_cache()

        torch.cuda.empty_cache()
        model = self.actor.model
        count, num_params = 0, len(list(model.named_parameters()))

        def _broadcast_param(param, count, num_params):
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                self.vllm_engine.update_weight(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params) 
                
                self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())

        def _handle_cuda_ipc(param, count, num_params):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            from vllm_utils.vllm_engine import get_physical_gpu_id
            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                self.vllm_engine.update_weight_cuda_ipc(
                    name,
                    dtype=param.dtype,
                    shape=shape,
                    ipc_handles=ipc_handles,
                    empty_cache=count == num_params,
                )

            torch_dist_barrier_and_cuda_sync()

        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not self.use_cuda_ipc:
                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _broadcast_param(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _broadcast_param(param, count, num_params)
            else:
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _handle_cuda_ipc(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _handle_cuda_ipc(param, count, num_params)

        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()

    def _iter_vllm_sync_params(self, model):
        for name, param in model.named_parameters():
            if _should_skip_vllm_sync_param(name):
                continue
            yield _normalize_vllm_weight_name(name), param

    @contextmanager
    def _merged_lora_adapter(self, model):
        if isinstance(model, PeftModel):
            if not hasattr(model, "merge_adapter") or not hasattr(model, "unmerge_adapter"):
                raise RuntimeError("Current PEFT version does not support merge_adapter/unmerge_adapter required for vLLM sync.")
            model.merge_adapter()
            try:
                yield model
            finally:
                model.unmerge_adapter()
        else:
            yield model


class PolicyModel:
    def __init__(
        self,
        strategy,
        pretrain: str,
        max_steps: Optional[int] = None,
        vllm_engine=None,
    ):
        self.strategy = strategy
        args = strategy.args

        self.vllm_engine = vllm_engine
        self.max_steps = max_steps

        if getattr(args, "vllm_num_engines", 0) > 0:
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        actor = Actor(
            pretrain,
            attn_implementation=args.attn_implementation,
            bf16=args.bf16,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            temperature=args.temperature,
            train_mode_cfg=args.train_mode_dict,
        )
        strategy.print(actor)
        if args.train_mode_dict.mode == "lora":
            actor.print_trainable_parameters()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrain, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        actor_optim = strategy.create_optimizer(
            actor,
            lr=args.learning_rate,
            betas=args.adam_betas,
            weight_decay=args.weight_decay,
        )

        if max_steps is None:
            max_steps = int(getattr(args, "max_steps", 1_000_000))

        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
        )
        
        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )
    
        if strategy.args.deepspeed_enable_sleep:
            from strategy.deepspeed import offload_deepspeed_states
            offload_deepspeed_states(self.actor.model)

        self.trainer = BatchPPOTrainer(
            strategy,
            self.actor,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            vllm_engine = vllm_engine,
        )

    def fit(self, batch_data, kl_ctl: float = 0.0):
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.train_batch(batch_data, kl_ctl)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    @torch.no_grad()
    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int], torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        to_cpu: bool = False,
    ) -> torch.Tensor:
        self.actor.eval()

        if action_mask is None:
            raise ValueError("action_mask is required for returning action_log_probs")

        device = torch.cuda.current_device()
        sequences = sequences.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True) if attention_mask is not None else None
        action_mask = action_mask.to(device, non_blocking=True) if torch.is_tensor(action_mask) else action_mask

        action_log_probs = self.actor(
            sequences,
            action_mask=action_mask,
            attention_mask=attention_mask,
            ring_attn_group=self.strategy.ring_attn_group, 
        )

        self.actor.train() 
        return action_log_probs.to("cpu") if to_cpu else action_log_probs

    def broadcast_to_vllm(self):
        # self.trainer._broadcast_to_vllm()
        self.trainer._deepspeed_broadcast()

    def save_model(self):
        args = self.strategy.args
        self.strategy.save_model(
            self.actor,
            self.tokenizer,
            args.save_path,
        )
    @property
    def train_iter(self):
        return self.trainer.train_iter

    def reload_states(self):
        from strategy.deepspeed import reload_deepspeed_states
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        from strategy.deepspeed import offload_deepspeed_states
        offload_deepspeed_states(self.actor.model)