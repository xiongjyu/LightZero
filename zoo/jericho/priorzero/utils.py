import torch
from typing import List, Dict, Any, Tuple, Union, Optional
from transformers import AutoTokenizer

def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and CUDA operations.
    This function ensures that:
    1. All distributed processes reach this point (barrier)
    2. All CUDA operations are completed (synchronize)
    """
    import torch

    torch.distributed.barrier()
    torch.cuda.synchronize()


def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer

@torch.compile
def compute_entropy(logits: torch.Tensor):
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio

def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)

import time
from contextlib import contextmanager
from collections import defaultdict

class Profiler:
    def __init__(self, log_interval: int = 10, stats_file: str = None):
        self.log_interval = max(1, int(log_interval))
        self.stats_file = stats_file
        self.stats = defaultdict(lambda: {"count": 0, "total": 0.0, "max": 0.0})
        self._inited = False

    def _init_once(self):
        if self._inited:
            return
        with open(self.stats_file, "a", encoding="utf-8") as f:
            f.write("ts\tname\tcount\ttotal_s\tavg_s\tmax_s\n")
        self._inited = True

    def _record(self, name: str, elapsed: float):
        s = self.stats[name]
        s["count"] += 1
        s["total"] += elapsed
        s["max"] = max(s["max"], elapsed)
        if s["count"] % self.log_interval == 0:
            avg = s["total"] / s["count"]
            with open(self.stats_file, "a", encoding="utf-8") as f:
                f.write(f"{time.time():.3f}\t{name}\t{s['count']}\t{s['total']:.6f}\t{avg:.6f}\t{s['max']:.6f}\n")

    @contextmanager
    def block(self, name: str, enable_profile: bool = True, rank: int = 0):
        if not enable_profile or rank != 0:
            yield None
            return
        self._init_once()
        t0 = time.perf_counter()
        try:
            yield None
        finally:
            self._record(name, time.perf_counter() - t0)