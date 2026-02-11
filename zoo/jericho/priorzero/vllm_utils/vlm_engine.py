"""
vLLM-based VLM Engine for multimodal inference.

This module provides a vLLM wrapper for Vision-Language Models,
similar to the text-only vLLM engine but with multimodal support.
"""
import vllm
from typing import List, Union, Optional, Dict, Any
from PIL import Image
import numpy as np
from loguru import logger


class VLMActor:
    """
    vLLM Actor for Vision-Language Models.

    Similar to LLMActor but with multimodal support.
    """

    def __init__(
        self,
        model: str = None,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        """
        Args:
            model: Path to VLM model
            limit_mm_per_prompt: Multimodal limits (e.g., {"image": 1})
            **kwargs: Additional vLLM arguments
        """
        self.kwargs = kwargs
        self.limit_mm_per_prompt = limit_mm_per_prompt or {"image": 1}

        logger.info(f"Initializing VLMActor with model: {model}")
        logger.info(f"  Multimodal limits: {self.limit_mm_per_prompt}")

        self.llm = vllm.LLM(
            model=model,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            **self.kwargs
        )

    def sleep(self, level=1):
        """Put the engine to sleep to free GPU memory."""
        if hasattr(self.llm, 'sleep'):
            self.llm.sleep(level=level)

    def wake_up(self):
        """Wake up the engine from sleep mode."""
        if hasattr(self.llm, 'wake_up'):
            self.llm.wake_up()

    def generate(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        prompts: List[str],
        sampling_params: Any,
    ) -> List[Any]:
        """
        Generate responses for multimodal inputs.

        Args:
            images: List of images (PIL Image or numpy array)
            prompts: List of text prompts
            sampling_params: vLLM SamplingParams

        Returns:
            List of vLLM RequestOutput objects
        """
        # Prepare multimodal inputs
        inputs = []
        for image, prompt in zip(images, prompts):
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                if len(image.shape) == 3 and image.shape[0] == 3:
                    # Convert CHW to HWC
                    image = np.transpose(image, (1, 2, 0))
                image = Image.fromarray(image)

            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            })

        # Generate
        responses = self.llm.generate(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        return responses


def create_vllm_vlm_engine(
    tensor_parallel_size: int,
    pretrain: str,
    max_model_len: int,
    gpu_memory_utilization: float = 0.3,
    vllm_enable_sleep: bool = False,
    limit_mm_per_prompt: Optional[Dict[str, int]] = None,
):
    """
    Create a vLLM engine for Vision-Language Models.

    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pretrain: Path to pretrained VLM model
        max_model_len: Maximum sequence length
        gpu_memory_utilization: GPU memory utilization ratio
        vllm_enable_sleep: Whether to enable sleep mode
        limit_mm_per_prompt: Multimodal limits per prompt

    Returns:
        VLMActor instance
    """
    distributed_executor_backend = "external_launcher"

    if limit_mm_per_prompt is None:
        limit_mm_per_prompt = {"image": 1}

    logger.info("Creating vLLM VLM engine:")
    logger.info(f"  Model: {pretrain}")
    logger.info(f"  Tensor Parallel Size: {tensor_parallel_size}")
    logger.info(f"  Max Model Length: {max_model_len}")
    logger.info(f"  GPU Memory Utilization: {gpu_memory_utilization}")
    logger.info(f"  Enable Sleep: {vllm_enable_sleep}")
    logger.info(f"  Multimodal Limits: {limit_mm_per_prompt}")

    vllm_engine = VLMActor(
        model=pretrain,
        worker_extension_cls="vllm_utils.worker.WorkerWrap",
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        max_model_len=max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_sleep_mode=vllm_enable_sleep,
        limit_mm_per_prompt=limit_mm_per_prompt,
        trust_remote_code=True,
    )

    if vllm_enable_sleep:
        vllm_engine.sleep()

    logger.info("✓ vLLM VLM engine created successfully")

    return vllm_engine
