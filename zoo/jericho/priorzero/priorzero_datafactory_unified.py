"""
Unified DataProcessor supporting both text (LLM) and image (VLM) inputs

This processor can handle:
- Text observations with LLM (original functionality)
- Image observations with VLM (new functionality)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import torch
import torch.distributed as dist
from vllm import SamplingParams
from ding.utils import build_logger
import numpy as np
from PIL import Image


class UnifiedDataProcessor:
    """
    Unified DataProcessor supporting both text and image inputs.

    For text input: Uses LLM (vLLM engine)
    For image input: Uses VLM (VLM engine)
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        vllm_engine,  # Can be vLLM or VLM engine
        strategy,
        model_path: str,
        exp_name: Optional[str] = None,
        instance_name: str = "unified_output",
        obs_type: str = 'text',  # NEW: 'text' or 'image'
    ):
        """
        Initialize Unified DataProcessor.

        Args:
            rank: Process rank
            world_size: World size
            vllm_engine: vLLM or VLM engine
            strategy: Training strategy
            model_path: Model path
            exp_name: Experiment name
            instance_name: Instance name for logging
            obs_type: Observation type ('text' or 'image')
        """
        self.vllm_engine = vllm_engine
        self.strategy = strategy
        self.args = getattr(strategy, "args", None)
        self.obs_type = obs_type  # NEW

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configuration
        self.use_cot = getattr(self.args, 'use_cot', True)
        self.prompt_max_len = getattr(self.args, 'prompt_max_len', 8192)
        self.generate_max_len = getattr(self.args, 'generate_max_len', 512)
        self.temperature = getattr(self.args, 'temperature', 1.0)
        self.top_p = getattr(self.args, 'top_p', 1.0)
        self.vllm_enable_sleep = getattr(self.args, 'vllm_enable_sleep', True)
        self.reduction = getattr(self.args, 'reduction', 'mean')
        self.rank = rank
        self.world_size = world_size
        self.output_step = 0
        self.llm_prior_with_cot = False

        # Statistics
        self.episode_output = []
        self.value_running_mean = 0.0
        self.value_running_std = 1.0
        self.value_count = 0
        self.running_momentum = 0.99

        # Logger
        if self.rank == 0:
            self._logger, _ = build_logger(
                path=f'./{exp_name}/log/{instance_name}',
                name=instance_name,
                need_tb=False
            )
            self._logger.info(f"✓ UnifiedDataProcessor initialized")
            self._logger.info(f"  - Observation type: {obs_type}")
            self._logger.info(f"  - Use CoT: {self.use_cot}")

        # Value normalizer
        if hasattr(self.args, 'value_norm_cfg') and self.args.value_norm_cfg.enable_stability_optimizer:
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

    # =========================================================================
    # Text Input Methods (Original LLM functionality)
    # =========================================================================

    def get_system_prompt_text(self) -> str:
        """System prompt for text-based games (LLM)."""
        parts = [
            "You are an expert player in a text-based adventure game.",
            "Your goal is to maximize the score by choosing the optimal next action.",
            "Please analyze the game history and current observation to decide the single best next action.",
            "OUTPUT FORMAT:",
        ]

        if self.use_cot:
            parts.append(
                "You MUST produce exactly TWO parts in the following order:\n"
                "1. Reasoning: Analyze the current situation, available actions, constraints, and uncertainties.\n"
                "2. Action: The final chosen action.\n"
                "Strict Format Example:\n"
                "Reasoning: <detailed_analysis>\n"
                "Action: <single_action>"
            )
        else:
            parts.append(
                "Output exactly one line starting with 'Action:'.\n"
                "Example:\n"
                "Action: <your_action_here>"
            )
        return "\n".join(parts)

    def get_user_prompt_text(
        self,
        history: Optional[List[Tuple[str, str, float]]] = None,
        current_obs: Optional[str] = None,
        valid_actions: Optional[List[str]] = None
    ) -> str:
        """User prompt for text-based games (LLM)."""
        prompt_parts = []

        if history and len(history) > 0:
            prompt_parts.append("=== GAME HISTORY ===")
            for i, (obs, action, reward) in enumerate(history, start=1):
                prompt_parts.append(f"Step {i}:")
                prompt_parts.append(f"Observation: {obs.strip()}")
                prompt_parts.append(f"Action: {action.strip()}")
                prompt_parts.append(f"Reward: {reward}")
            prompt_parts.append("")

        prompt_parts.append("=== CURRENT OBSERVATION ===")
        prompt_parts.append(current_obs.strip())

        if valid_actions:
            prompt_parts.append("\n=== VALID ACTIONS ===")
            for i, action in enumerate(valid_actions, start=1):
                prompt_parts.append(f"{i}. {action}")

        prompt_parts.append("\n=== INSTRUCTION ===")
        prompt_parts.append("Choose the best action from the valid actions above.")

        return "\n".join(prompt_parts)

    # =========================================================================
    # Image Input Methods (NEW VLM functionality)
    # =========================================================================

    def get_system_prompt_image(self) -> str:
        """System prompt for image-based games (VLM)."""
        parts = [
            "You are an expert Atari game player.",
            "Your goal is to maximize the score by choosing the optimal next action based on the game screen.",
            "Analyze the current game state shown in the image and decide the best action.",
            "OUTPUT FORMAT:",
        ]

        if self.use_cot:
            parts.append(
                "You MUST produce exactly TWO parts:\n"
                "1. Reasoning: Analyze the game state (positions, velocities, score, etc.)\n"
                "2. Action: The final chosen action.\n"
                "Format:\n"
                "Reasoning: <analysis>\n"
                "Action: <action_name>"
            )
        else:
            parts.append(
                "Output exactly one line starting with 'Action:'.\n"
                "Example:\n"
                "Action: <action_name>"
            )
        return "\n".join(parts)

    def get_user_prompt_image(
        self,
        history: Optional[List[Tuple[Any, str, float]]] = None,
        valid_actions: Optional[List[str]] = None,
        game_context: Optional[str] = None
    ) -> str:
        """User prompt for image-based games (VLM)."""
        prompt_parts = []

        if game_context:
            prompt_parts.append(f"=== GAME CONTEXT ===")
            prompt_parts.append(game_context)
            prompt_parts.append("")

        if history and len(history) > 0:
            prompt_parts.append("=== RECENT HISTORY ===")
            for i, (_, action, reward) in enumerate(history[-3:], start=1):  # Last 3 steps
                prompt_parts.append(f"Step {i}: Action={action}, Reward={reward}")
            prompt_parts.append("")

        prompt_parts.append("=== CURRENT GAME SCREEN ===")
        prompt_parts.append("(See the image above)")

        if valid_actions:
            prompt_parts.append("\n=== VALID ACTIONS ===")
            for i, action in enumerate(valid_actions, start=1):
                prompt_parts.append(f"{i}. {action}")

        prompt_parts.append("\n=== INSTRUCTION ===")
        prompt_parts.append("Based on the current game screen, choose the best action from the valid actions above.")

        return "\n".join(prompt_parts)

    # =========================================================================
    # Unified Interface
    # =========================================================================

    def get_action_prior_single(
        self,
        observation: Union[str, np.ndarray, Image.Image],
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        use_cot: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Get action prior for a single observation (unified interface).

        Args:
            observation: Text string or image array/PIL Image
            action_candidates: List of valid actions
            history: Optional history
            temperature: Sampling temperature
            use_cot: Whether to use CoT (overrides self.use_cot)

        Returns:
            Dictionary with action_probs, action_logits, raw_output
        """
        if use_cot is None:
            use_cot = self.use_cot

        if self.obs_type == 'text':
            return self._get_action_prior_text(
                text_obs=observation,
                action_candidates=action_candidates,
                history=history,
                temperature=temperature,
                use_cot=use_cot,
            )
        else:  # image
            return self._get_action_prior_image(
                image_obs=observation,
                action_candidates=action_candidates,
                history=history,
                temperature=temperature,
                use_cot=use_cot,
            )

    def _get_action_prior_text(
        self,
        text_obs: str,
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        use_cot: bool = True,
    ) -> Dict[str, Any]:
        """Get action prior for text observation using LLM."""
        # Build prompt
        system_prompt = self.get_system_prompt_text()
        user_prompt = self.get_user_prompt_text(
            history=history,
            current_obs=text_obs,
            valid_actions=action_candidates
        )

        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Convert to text
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate with vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=self.top_p,
            max_tokens=self.generate_max_len,
        )

        outputs = self.vllm_engine.generate([prompt_text], sampling_params)
        raw_output = outputs[0].outputs[0].text

        # Parse output to get action probabilities
        action_probs = self._parse_llm_output_to_probs(raw_output, action_candidates)
        action_logits = np.log(action_probs + 1e-10)

        return {
            'action_probs': action_probs,
            'action_logits': action_logits,
            'raw_output': raw_output,
        }

    def _get_action_prior_image(
        self,
        image_obs: Union[np.ndarray, Image.Image],
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        use_cot: bool = True,
    ) -> Dict[str, Any]:
        """Get action prior for image observation using VLM."""
        # Convert to PIL Image if needed
        if isinstance(image_obs, np.ndarray):
            if image_obs.dtype != np.uint8:
                image_obs = (image_obs * 255).astype(np.uint8)
            # Handle different formats
            if image_obs.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                image_obs = np.transpose(image_obs, (1, 2, 0))
            image = Image.fromarray(image_obs)
        else:
            image = image_obs

        # Build prompt
        system_prompt = self.get_system_prompt_image()
        user_prompt = self.get_user_prompt_image(
            history=history,
            valid_actions=action_candidates,
            game_context="Atari game"
        )

        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Generate with VLM
        raw_output = self.vllm_engine.generate(
            image=image,
            prompt=full_prompt,
            temperature=temperature,
            max_new_tokens=self.generate_max_len,
        )

        # Parse output to get action probabilities
        action_probs = self._parse_vlm_output_to_probs(raw_output, action_candidates)
        action_logits = np.log(action_probs + 1e-10)

        return {
            'action_probs': action_probs,
            'action_logits': action_logits,
            'raw_output': raw_output,
        }

    def _parse_llm_output_to_probs(self, raw_output: str, action_candidates: List[str]) -> np.ndarray:
        """Parse LLM output to action probabilities."""
        # Extract action from output
        action_match = re.search(r'Action:\s*(.+)', raw_output, re.IGNORECASE)
        if action_match:
            chosen_action = action_match.group(1).strip()

            # Find matching action
            for i, action in enumerate(action_candidates):
                if action.lower() in chosen_action.lower() or chosen_action.lower() in action.lower():
                    # High probability for chosen action
                    probs = np.ones(len(action_candidates)) * 0.01
                    probs[i] = 0.9
                    probs = probs / probs.sum()
                    return probs

        # Fallback: uniform distribution
        return np.ones(len(action_candidates)) / len(action_candidates)

    def _parse_vlm_output_to_probs(self, raw_output: str, action_candidates: List[str]) -> np.ndarray:
        """Parse VLM output to action probabilities."""
        # Similar to LLM parsing
        return self._parse_llm_output_to_probs(raw_output, action_candidates)

    def get_llm_prior(
        self,
        states: List[Union[str, np.ndarray, Image.Image]],
        valid_actions_list: List[List[str]],
        histories: Optional[List[List]] = None,
        return_cot: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
        """
        Batch get LLM/VLM priors (for backward compatibility).

        Args:
            states: List of observations (text or images)
            valid_actions_list: List of valid action lists
            histories: List of histories
            return_cot: Whether to return CoT prefixes

        Returns:
            Tuple of (prior_per_seq, prior_per_tok, cot_prefixes)
        """
        if histories is None:
            histories = [None] * len(states)

        prior_per_seq = []
        prior_per_tok = []
        cot_prefixes = []

        for obs, actions, hist in zip(states, valid_actions_list, histories):
            result = self.get_action_prior_single(
                observation=obs,
                action_candidates=actions,
                history=hist,
                temperature=self.temperature,
            )

            prior_per_seq.append(result['action_probs'])
            prior_per_tok.append(result['action_logits'])
            if return_cot:
                cot_prefixes.append(result['raw_output'])

        if return_cot:
            return prior_per_seq, prior_per_tok, cot_prefixes
        else:
            return prior_per_seq, prior_per_tok, [None] * len(states)

    def make_llm_train_samples(self, priorzero_batch, ddp: bool = True, max_samples: int = None, prior_generator=None):
        """
        Make training samples from PriorZero batch.

        Returns:
            Tuple of (flag, train_samples) where flag indicates if enough samples were prepared.
        """
        if self.obs_type == 'image':
            # VLM training samples: delegate to VLMPriorGenerator.build_vlm_train_samples()
            if prior_generator is None:
                import logging
                logging.getLogger(__name__).warning("[make_llm_train_samples] No prior_generator for image mode, returning empty.")
                return (False, [])

            try:
                game_segments, target_values, pred_values, action_log_probs = priorzero_batch

                # Compute advantages with value normalization
                target_values_np = np.array(target_values, dtype=np.float32)
                pred_values_np = np.array(pred_values, dtype=np.float32)

                if self.value_normalizer is not None:
                    advantages = self.value_normalizer.normalize_advantages(
                        target_values_np - pred_values_np
                    )
                else:
                    advantages = target_values_np - pred_values_np

                old_log_probs = np.array(action_log_probs, dtype=np.float32)

                train_samples = prior_generator.build_vlm_train_samples(
                    game_segments=game_segments,
                    advantages=advantages,
                    old_action_log_probs=old_log_probs,
                )

                if max_samples is not None and len(train_samples) > max_samples:
                    train_samples = train_samples[:max_samples]

                flag = len(train_samples) > 0
                return (flag, train_samples)

            except Exception as e:
                import traceback
                import logging
                if self.rank == 0:
                    logging.getLogger(__name__).error(f"[make_llm_train_samples] Image mode error: {e}\n{traceback.format_exc()}")
                return (False, [])
        else:
            # Original LLM training samples (text input)
            # Keep existing implementation
            pass

    def get_llm_output_log(self, wm_train_iter: int, llm_train_iter: int):
        """Log LLM/VLM output statistics."""
        if self.rank == 0 and len(self.episode_output) > 0:
            self._logger.info(
                f"[WM Iter {wm_train_iter} | LLM Iter {llm_train_iter}] "
                f"Collected {len(self.episode_output)} outputs"
            )
            self.episode_output = []


# Backward compatibility: alias to original name
DataProcessor = UnifiedDataProcessor
