"""
Unified Prior Generator Interface

This module provides a unified interface for generating action priors
from different types of observations (text or image).
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
from PIL import Image


class PriorGenerator(ABC):
    """
    Abstract base class for prior generators.

    Subclasses should implement generate_prior() to generate action prior
    distributions from observations.
    """

    def __init__(self, model_name: str, obs_type: str):
        """
        Args:
            model_name: Name/path of the model
            obs_type: Type of observation ('text' or 'image')
        """
        self.model_name = model_name
        self.obs_type = obs_type

    @abstractmethod
    def generate_prior(
        self,
        observation: Any,
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate action prior distribution from observation.

        Args:
            observation: Observation (text string or image array/PIL Image)
            action_candidates: List of valid action strings
            history: Optional history of previous (obs, action, reward) tuples
            temperature: Temperature for sampling
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary containing:
                - 'action_probs': np.ndarray of shape (num_actions,) with probabilities
                - 'action_logits': np.ndarray of shape (num_actions,) with logits
                - 'raw_output': Raw model output (for logging/debugging)
        """
        pass

    @abstractmethod
    def batch_generate_prior(
        self,
        observations: List[Any],
        action_candidates_list: List[List[str]],
        histories: Optional[List[List]] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch version of generate_prior for efficiency.

        Args:
            observations: List of observations
            action_candidates_list: List of action candidate lists
            histories: Optional list of histories
            temperature: Temperature for sampling
            **kwargs: Additional arguments

        Returns:
            List of prior dictionaries (same format as generate_prior)
        """
        pass


class LLMPriorGenerator(PriorGenerator):
    """
    Prior generator using Language Models for text observations.

    This is a wrapper around the existing vLLM engine and DataProcessor.
    """

    def __init__(
        self,
        vllm_engine,
        data_processor,
        model_name: str,
        use_cot: bool = True,
        **kwargs
    ):
        """
        Args:
            vllm_engine: vLLM engine instance
            data_processor: DataProcessor instance
            model_name: LLM model name
            use_cot: Whether to use Chain-of-Thought
        """
        super().__init__(model_name, obs_type='text')
        self.vllm_engine = vllm_engine
        self.data_processor = data_processor
        self.use_cot = use_cot

    def generate_prior(
        self,
        observation: str,
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate prior from text observation using LLM.

        Args:
            observation: Text observation string
            action_candidates: List of valid action strings
            history: Optional history buffer
            temperature: Sampling temperature

        Returns:
            Prior dictionary with action_probs, action_logits, raw_output
        """
        # Use existing DataProcessor logic
        # This delegates to the existing implementation
        result = self.data_processor.get_action_prior_single(
            text_obs=observation,
            action_candidates=action_candidates,
            history=history,
            temperature=temperature,
            use_cot=self.use_cot,
        )

        return result

    def batch_generate_prior(
        self,
        observations: List[str],
        action_candidates_list: List[List[str]],
        histories: Optional[List[List]] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch generate priors from text observations.
        """
        # Use existing DataProcessor batch logic
        results = self.data_processor.get_action_prior_batch(
            text_obs_list=observations,
            action_candidates_list=action_candidates_list,
            histories=histories,
            temperature=temperature,
            use_cot=self.use_cot,
        )

        return results


class VLMPriorGenerator(PriorGenerator):
    """
    Prior generator using Vision-Language Models for image observations.

    Supports models like Qwen-VL, LLaVA, InternVL, etc.
    """

    def __init__(
        self,
        vlm_engine,
        model_name: str,
        prompt_template: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            vlm_engine: VLM engine instance (to be implemented)
            model_name: VLM model name
            prompt_template: Optional custom prompt template
        """
        super().__init__(model_name, obs_type='image')
        self.vlm_engine = vlm_engine
        self.prompt_template = prompt_template or self._default_prompt_template()

    def _default_prompt_template(self) -> str:
        """Default prompt template for Atari games."""
        return (
            "You are an expert Atari game player. "
            "Based on the current game screen shown in the image, "
            "choose the best action from the following options:\n"
            "{action_list}\n\n"
            "Provide a probability distribution over these actions. "
            "Consider the game state, positions of objects, and your goal. "
            "Output format: {{'action_name': probability, ...}}\n"
            "Make sure probabilities sum to 1.0."
        )

    def _convert_obs_to_pil_image(self, obs: np.ndarray) -> Image.Image:
        """
        Robustly convert observation array to PIL Image.

        Handles various input formats:
        - CHW format (C, H, W): channels first, e.g., (3, 64, 64)
        - HWC format (H, W, C): channels last, e.g., (64, 64, 3)
        - Grayscale (H, W): single channel, e.g., (64, 64)
        - Stacked frames (N, H, W): takes the last frame

        Args:
            obs: Observation array

        Returns:
            PIL Image in RGB format

        Raises:
            ValueError: If observation shape is invalid
        """
        if not isinstance(obs, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(obs)}")

        # Ensure uint8 dtype
        if obs.dtype != np.uint8:
            # Normalize to [0, 255] if needed
            if obs.max() <= 1.0:
                obs = (obs * 255).astype(np.uint8)
            else:
                obs = obs.astype(np.uint8)

        # Handle different shapes
        if obs.ndim == 2:
            # Grayscale (H, W) -> convert to RGB
            return Image.fromarray(obs, mode='L').convert('RGB')

        elif obs.ndim == 3:
            # Determine if CHW or HWC format
            c, h, w = obs.shape

            # If first dimension is small (1-4), likely CHW format
            if c <= 4 and h > c and w > c:
                # CHW format -> transpose to HWC
                if c == 1:
                    # Single channel (1, H, W) -> (H, W)
                    obs = obs[0]
                    return Image.fromarray(obs, mode='L').convert('RGB')
                elif c == 3:
                    # RGB (3, H, W) -> (H, W, 3)
                    obs = np.transpose(obs, (1, 2, 0))
                    return Image.fromarray(obs, mode='RGB')
                elif c == 4:
                    # RGBA or stacked frames
                    # Take last 3 channels as RGB
                    obs = np.transpose(obs[-3:], (1, 2, 0))
                    return Image.fromarray(obs, mode='RGB')
                else:
                    # Stacked grayscale frames (N, H, W) -> take last frame
                    obs = obs[-1]
                    return Image.fromarray(obs, mode='L').convert('RGB')

            # Otherwise, assume HWC format
            elif w <= 4 and h > w and c > w:
                # HWC format
                if w == 1:
                    # Single channel (H, W, 1) -> (H, W)
                    obs = obs[:, :, 0]
                    return Image.fromarray(obs, mode='L').convert('RGB')
                elif w == 3:
                    # RGB (H, W, 3)
                    return Image.fromarray(obs, mode='RGB')
                elif w == 4:
                    # RGBA (H, W, 4) -> take first 3 channels
                    obs = obs[:, :, :3]
                    return Image.fromarray(obs, mode='RGB')

            # Ambiguous shape - provide detailed error
            raise ValueError(
                f"Cannot determine image format from shape {obs.shape}. "
                f"Expected CHW (C, H, W) with C<=4 or HWC (H, W, C) with C<=4. "
                f"Please ensure observation is in correct format."
            )

        elif obs.ndim == 4:
            # Batch dimension (B, C, H, W) or (B, H, W, C) -> take first image
            raise ValueError(
                f"Observation has batch dimension {obs.shape}. "
                f"Please pass individual observations, not batches."
            )

        else:
            raise ValueError(
                f"Invalid observation shape {obs.shape}. "
                f"Expected 2D (H, W) or 3D (C, H, W) or (H, W, C)."
            )

    def _build_prompt(
        self,
        action_candidates: List[str],
        history: Optional[List] = None
    ) -> str:
        """
        Build prompt for VLM.

        Args:
            action_candidates: List of valid actions
            history: Optional history (for context)

        Returns:
            Formatted prompt string
        """
        # Format action list
        action_list = "\n".join([f"- {action}" for action in action_candidates])

        # Build base prompt
        prompt = self.prompt_template.format(action_list=action_list)

        # Add history context if available
        if history and len(history) > 0:
            history_text = "\n\nRecent history:\n"
            for i, (obs, action, reward) in enumerate(history[-3:]):  # Last 3 steps
                history_text += f"Step {i+1}: Action={action}, Reward={reward}\n"
            prompt = history_text + "\n" + prompt

        return prompt

    def _parse_vlm_output(
        self,
        raw_output: str,
        action_candidates: List[str]
    ) -> np.ndarray:
        """
        Parse VLM output to extract action probabilities.

        Args:
            raw_output: Raw text output from VLM
            action_candidates: List of valid actions

        Returns:
            Action probabilities as numpy array
        """
        import json
        import re

        # Try to extract JSON from output
        try:
            # Look for JSON-like structure
            json_match = re.search(r'\{[^}]+\}', raw_output)
            if json_match:
                action_probs_dict = json.loads(json_match.group())

                # Convert to array aligned with action_candidates
                probs = []
                for action in action_candidates:
                    probs.append(action_probs_dict.get(action, 0.0))

                probs = np.array(probs)

                # Normalize
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    # Fallback to uniform
                    probs = np.ones(len(action_candidates)) / len(action_candidates)

                return probs
        except:
            pass

        # Fallback: uniform distribution
        return np.ones(len(action_candidates)) / len(action_candidates)

    def generate_prior(
        self,
        observation: Union[np.ndarray, Image.Image],
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate prior from image observation using VLM.

        Args:
            observation: Image observation (numpy array or PIL Image)
            action_candidates: List of valid action strings
            history: Optional history buffer
            temperature: Sampling temperature

        Returns:
            Prior dictionary with action_probs, action_logits, raw_output
        """
        # Convert observation to PIL Image if needed
        if isinstance(observation, np.ndarray):
            # Assume (H, W, C) format
            if observation.dtype != np.uint8:
                observation = (observation * 255).astype(np.uint8)
            image = Image.fromarray(observation)
        else:
            image = observation

        # Build prompt
        prompt = self._build_prompt(action_candidates, history)

        # Generate with VLM
        raw_output = self.vlm_engine.generate(
            image=image,
            prompt=prompt,
            temperature=temperature,
            **kwargs
        )

        # Parse output to get probabilities
        action_probs = self._parse_vlm_output(raw_output, action_candidates)

        # Compute logits (inverse of softmax with temperature)
        action_logits = np.log(action_probs + 1e-10) * temperature

        return {
            'action_probs': action_probs,
            'action_logits': action_logits,
            'raw_output': raw_output,
        }

    def batch_generate_prior(
        self,
        observations: List[Union[np.ndarray, Image.Image]],
        action_candidates_list: List[List[str]],
        histories: Optional[List[List]] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch generate priors from image observations.

        For efficiency, this should use batched VLM inference.
        """
        if histories is None:
            histories = [None] * len(observations)

        # Convert all observations to PIL Images using robust conversion
        images = []
        for i, obs in enumerate(observations):
            try:
                if isinstance(obs, Image.Image):
                    # Already a PIL Image
                    images.append(obs)
                elif isinstance(obs, np.ndarray):
                    # Convert numpy array to PIL Image
                    pil_image = self._convert_obs_to_pil_image(obs)
                    images.append(pil_image)
                else:
                    raise TypeError(f"Unsupported observation type: {type(obs)}")
            except Exception as e:
                raise ValueError(
                    f"Failed to convert observation {i} with shape "
                    f"{obs.shape if isinstance(obs, np.ndarray) else 'N/A'} "
                    f"to PIL Image: {e}"
                ) from e

        # Build prompts
        prompts = [
            self._build_prompt(actions, hist)
            for actions, hist in zip(action_candidates_list, histories)
        ]

        # Batch generate with VLM
        raw_outputs = self.vlm_engine.batch_generate(
            images=images,
            prompts=prompts,
            temperature=temperature,
            **kwargs
        )

        # Parse outputs
        results = []
        for raw_output, action_candidates in zip(raw_outputs, action_candidates_list):
            action_probs = self._parse_vlm_output(raw_output, action_candidates)
            action_logits = np.log(action_probs + 1e-10) * temperature

            results.append({
                'action_probs': action_probs,
                'action_logits': action_logits,
                'raw_output': raw_output,
            })

        return results


def create_prior_generator(
    obs_type: str,
    model_config: Dict[str, Any],
    **kwargs
) -> PriorGenerator:
    """
    Factory function to create appropriate prior generator.

    Args:
        obs_type: 'text' or 'image'
        model_config: Model configuration dictionary
        **kwargs: Additional arguments

    Returns:
        PriorGenerator instance (LLMPriorGenerator or VLMPriorGenerator)
    """
    if obs_type == 'text':
        # Create LLM prior generator
        from vllm_utils.vllm_engine import create_vllm_engine

        vllm_engine = create_vllm_engine(
            tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
            pretrain=model_config['model_path'],
            enable_prefix_caching=model_config.get('enable_prefix_caching', True),
            max_model_len=model_config.get('max_model_len', 8192),
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.3),
        )

        # Note: data_processor needs to be passed separately
        # This is a placeholder - actual implementation needs data_processor
        raise NotImplementedError(
            "LLMPriorGenerator requires data_processor. "
            "Use the existing implementation or pass data_processor explicitly."
        )

    elif obs_type == 'image':
        # Create VLM prior generator
        from vlm_engine import create_vlm_engine

        vlm_engine = create_vlm_engine(
            model_name=model_config['model_name'],
            model_path=model_config['model_path'],
            tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.3),
        )

        return VLMPriorGenerator(
            vlm_engine=vlm_engine,
            model_name=model_config['model_name'],
            prompt_template=model_config.get('prompt_template', None),
        )

    else:
        raise ValueError(f"Unknown obs_type: {obs_type}. Must be 'text' or 'image'.")


if __name__ == "__main__":
    # Example usage
    print("Prior Generator Interface")
    print("=" * 80)
    print("\nThis module provides unified interface for generating action priors.")
    print("\nSupported generators:")
    print("  - LLMPriorGenerator: For text observations (Jericho games)")
    print("  - VLMPriorGenerator: For image observations (Atari games)")
    print("\nUsage:")
    print("  generator = create_prior_generator(obs_type='image', model_config={...})")
    print("  prior = generator.generate_prior(observation, action_candidates)")
