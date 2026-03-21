"""
Unified Prior Generator Interface

This module provides a unified interface for generating action priors
from different types of observations (text or image).
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import time
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


class VLPriorGenerator(PriorGenerator):
    """
    Prior generator using Vision-Language (VL) models for image observations.

    Supports models like Qwen-VL, LLaVA, InternVL, etc.

    Includes training sample construction for PPO optimization with advantages.
    """

    def __init__(
        self,
        vl_engine,
        model_name: str,
        use_cot: bool = True,
        tokenizer=None,
        game_description: str = "",
        **kwargs
    ):
        """
        Args:
            vl_engine: VL engine instance (to be implemented)
            model_name: VL model name
            use_cot: Whether to use Chain-of-Thought reasoning
            tokenizer: Tokenizer for building training samples
            game_description: Game-specific description for prompts
        """
        super().__init__(model_name, obs_type='image')
        self.vl_engine = vl_engine
        self.use_cot = use_cot
        self.tokenizer = tokenizer
        self.game_description = game_description

        # For logging VL outputs
        self.episode_output = []

        # Log control: only log every N calls
        self.log_interval = 100  # Log every 100 calls
        self.call_count = 0
        self.batch_call_count = 0

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
                    return Image.fromarray(obs)
                elif c == 4:
                    # RGBA or stacked frames
                    # Take last 3 channels as RGB
                    obs = np.transpose(obs[-3:], (1, 2, 0))
                    return Image.fromarray(obs)
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
                    return Image.fromarray(obs)
                elif w == 4:
                    # RGBA (H, W, 4) -> take first 3 channels
                    obs = obs[:, :, :3]
                    return Image.fromarray(obs)

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

    def get_system_prompt(self) -> str:
        """
        System prompt for VL — mirrors LLM's get_system_prompt(),
        only replacing "text-based adventure game" with image-based context.
        """
        parts = [
            "You are an expert player in an image-based game. Your goal is to maximize the score by choosing the optimal next action.",
            "Analyze the game screen and history to decide the single best next action.",
            "IMPORTANT: You MUST choose EXACTLY ONE action from the provided valid actions list. Output the action name EXACTLY as given.",
        ]

        if self.use_cot:
            parts.append(
                "OUTPUT FORMAT (you MUST follow this EXACTLY):\n"
                "Reasoning: <brief analysis in 1-3 sentences>\n"
                "Action: <exact_action_name>\n\n"
                "RULES:\n"
                "- Keep reasoning SHORT (1-3 sentences max).\n"
                "- The Action line MUST contain exactly one action name from the valid actions list.\n"
                "- Do NOT add any text after the action name."
            )
        else:
            parts.append(
                "OUTPUT FORMAT:\n"
                "Action: <exact_action_name>\n\n"
                "Output ONLY this single line. No other text."
            )
        return "\n".join(parts)

    def get_user_prompt(
        self,
        action_candidates: List[str],
        history: Optional[List] = None
    ) -> str:
        """
        User prompt for VL — mirrors LLM's get_user_prompt() structure,
        replacing text observation with image vision tokens.
        """
        prompt_parts = []

        if history and len(history) > 0:
            prompt_parts.append("=== GAME HISTORY ===")
            for entry in history:
                # Support both (obs, action, reward, timestep) and legacy (obs, action, reward)
                if len(entry) >= 4:
                    obs, action, reward, timestep = entry[0], entry[1], entry[2], entry[3]
                    prompt_parts.append(f"Step {timestep}: Action: {action}, Reward: {reward}")
                else:
                    obs, action, reward = entry[0], entry[1], entry[2]
                    prompt_parts.append(f"Action: {action}, Reward: {reward}")
            prompt_parts.append("")  # empty line separator

        prompt_parts.append("=== CURRENT OBSERVATION ===")
        # NOTE: Do NOT include <|vision_start|><|image_pad|><|vision_end|> here.
        # The image placeholder is inserted by the chat template in vl_engine.
        prompt_parts.append("[See the game screen image above]")
        if self.game_description:
            prompt_parts.append(self.game_description)

        if action_candidates and len(action_candidates) > 0:
            actions_str = ", ".join(action_candidates)
            prompt_parts.append(f"\nValid actions: [{actions_str}]")

        # Add a concrete few-shot example using the actual action names
        example_action = action_candidates[0] if action_candidates else "NOOP"
        prompt_parts.append("\n=== INSTRUCTION ===")
        if self.use_cot:
            prompt_parts.append(
                f"Choose the best action. Respond in EXACTLY this format:\n"
                f"Reasoning: <1-3 sentences>\n"
                f"Action: <one action from the valid actions list>\n\n"
                f"Example:\n"
                f"Reasoning: The lander is drifting left and descending fast, so I need to fire the right engine.\n"
                f"Action: {example_action}"
            )
        else:
            prompt_parts.append(
                f"Choose the best action. Output ONLY:\n"
                f"Action: <one action from the valid actions list>\n\n"
                f"Example:\nAction: {example_action}"
            )
        return "\n".join(prompt_parts)

    def _parse_vl_output_with_cot(
        self,
        raw_output: str,
        action_candidates: List[str]
    ) -> Tuple[str, Optional[str]]:
        """
        Parse VL output to extract action and optional CoT reasoning.

        Args:
            raw_output: Raw VL output string
            action_candidates: List of valid action names

        Returns:
            Tuple of (chosen_action, cot_prefix)
            - chosen_action: The selected action name
            - cot_prefix: The reasoning part (if use_cot=True), else None
        """
        import re

        cot_prefix = None
        chosen_action = None

        # Extract reasoning part (if present)
        if self.use_cot:
            reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=Action:|$)', raw_output, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                cot_prefix = reasoning_match.group(1).strip()

        # Strategy 1: Extract text after "Action:" and match against candidates
        # Use .+ instead of \S+ to capture multi-word or underscore-separated actions
        action_match = re.search(r'Action:\s*(.+)', raw_output, re.IGNORECASE)
        if action_match:
            action_str = action_match.group(1).strip().strip("'\"`.,:;")
            # Exact match (case-insensitive)
            for candidate in action_candidates:
                if candidate.upper() == action_str.upper():
                    chosen_action = candidate
                    break
            # If no exact match, try if candidate is contained in the extracted text
            if chosen_action is None:
                for candidate in action_candidates:
                    if candidate.upper() in action_str.upper():
                        chosen_action = candidate
                        break

        # Strategy 2: If no "Action:" line found, scan entire output for action names
        if chosen_action is None:
            # Search for exact action name mentions in the output (prefer later mentions)
            last_found = None
            for candidate in action_candidates:
                # Use word boundary to avoid partial matches
                pattern = re.escape(candidate)
                matches = list(re.finditer(pattern, raw_output, re.IGNORECASE))
                if matches:
                    pos = matches[-1].start()
                    if last_found is None or pos > last_found[1]:
                        last_found = (candidate, pos)
            if last_found is not None:
                chosen_action = last_found[0]

        # Fallback: if no valid action found, use first candidate
        if chosen_action is None:
            chosen_action = action_candidates[0] if action_candidates else "NOOP"

        return chosen_action, cot_prefix

    def _action_to_logprob(
        self,
        chosen_action: str,
        action_candidates: List[str],
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Convert chosen action to log probability distribution.

        For training, we need to store the "old" log probabilities that were used
        to select the action. This creates a peaked distribution around the chosen action.

        Args:
            chosen_action: The action selected by VL
            action_candidates: List of all valid actions
            temperature: Temperature for softening the distribution

        Returns:
            Log probability array of shape (num_actions,)
        """
        num_actions = len(action_candidates)

        # Create peaked but NOT one-hot distribution to preserve MCTS exploration.
        # Use moderate logit gap (2.0 vs 0.0) instead of extreme (10.0 vs -10.0),
        # so the prior is informative but not deterministic.
        logits = np.zeros(num_actions, dtype=np.float32)

        try:
            chosen_idx = action_candidates.index(chosen_action)
            logits[chosen_idx] = 2.0  # Moderate logit for chosen action
        except ValueError:
            # If chosen action not in candidates, uniform distribution
            logits = np.zeros(num_actions)

        # Apply temperature and convert to log probabilities (numerically stable)
        logits = logits / temperature
        max_logit = np.max(logits)
        log_probs = logits - max_logit - np.log(np.sum(np.exp(logits - max_logit)) + 1e-10)

        return log_probs

    def _parse_vl_output(
        self,
        raw_output: str,
        action_candidates: List[str]
    ) -> np.ndarray:
        """
        Parse VL output to extract action probabilities.

        Args:
            raw_output: Raw text output from VL
            action_candidates: List of valid action names (e.g., ['NOOP', 'FIRE', 'RIGHT'])

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
                    # Try exact match and case-insensitive match
                    prob = action_probs_dict.get(action,
                           action_probs_dict.get(action.upper(),
                           action_probs_dict.get(action.lower(), 0.0)))
                    probs.append(prob)

                probs = np.array(probs, dtype=np.float32)

                # Normalize
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    # Fallback to uniform
                    probs = np.ones(len(action_candidates), dtype=np.float32) / len(action_candidates)

                return probs
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to parse VL output: {e}. Using uniform prior.")
            logger.debug(f"Raw output: {raw_output}")

        # Fallback: uniform distribution
        return np.ones(len(action_candidates), dtype=np.float32) / len(action_candidates)

    def generate_prior(
        self,
        observation: Union[np.ndarray, Image.Image],
        action_candidates: List[str],
        history: Optional[List] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate prior from image observation using VL with CoT support.

        Args:
            observation: Image observation (numpy array or PIL Image)
            action_candidates: List of valid action strings
            history: Optional history buffer
            temperature: Sampling temperature

        Returns:
            Prior dictionary with action_probs, action_logits, raw_output, cot_prefix
        """
        self.call_count += 1

        # Convert observation to PIL Image if needed
        if isinstance(observation, np.ndarray):
            image = self._convert_obs_to_pil_image(observation)
        else:
            image = observation

        # Build prompt (unified: always use get_user_prompt, consistent with LLM side)
        prompt = self.get_user_prompt(action_candidates, history)

        # Log prompt preview at intervals
        if self.call_count % self.log_interval == 1:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"[VL Prior Generation] Call #{self.call_count} | "
                f"Actions: {len(action_candidates)} | "
                f"Prompt preview: {prompt[:150]}..."
            )

        # Generate with VL
        raw_output = self.vl_engine.generate(
            image=image,
            prompt=prompt,
            temperature=temperature,
            system_prompt=self.get_system_prompt(),
            **kwargs
        )

        # Parse output (unified: always use CoT-style parser which handles both formats)
        chosen_action, cot_prefix = self._parse_vl_output_with_cot(raw_output, action_candidates)

        # Convert chosen action to log probability distribution
        action_log_probs = self._action_to_logprob(chosen_action, action_candidates, temperature)
        action_probs = np.exp(action_log_probs)

        # Log output at intervals
        if self.call_count % self.log_interval == 1:
            logger.info(
                f"[VL Prior Output] Chosen: {chosen_action} | "
                f"CoT: {cot_prefix[:100] if cot_prefix else 'None'}..."
            )

        return {
            'action_probs': action_probs,
            'action_logits': action_log_probs,
            'raw_output': raw_output,
            'cot_prefix': cot_prefix,
            'chosen_action': chosen_action,
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

        For efficiency, this should use batched VL inference.
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

        # Build prompts (unified: always use get_user_prompt)
        prompts = []
        for action_candidates, history in zip(action_candidates_list, histories):
            prompt = self.get_user_prompt(action_candidates, history)
            prompts.append(prompt)

        # Increment batch call counter
        self.batch_call_count += 1

        # First-call validation logging: image shapes, dtypes, PIL sizes, prompt preview
        if self.batch_call_count == 1:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[VL Batch Validation] === FIRST CALL DATA FLOW CHECK ===")
            logger.info(f"  Batch size: {len(observations)}")
            for i, obs in enumerate(observations[:3]):
                if isinstance(obs, np.ndarray):
                    logger.info(f"  Obs[{i}]: ndarray shape={obs.shape}, dtype={obs.dtype}, min={obs.min()}, max={obs.max()}")
                elif isinstance(obs, Image.Image):
                    logger.info(f"  Obs[{i}]: PIL Image size={obs.size}, mode={obs.mode}")
            for i, img in enumerate(images[:3]):
                logger.info(f"  PIL Image[{i}]: size={img.size}, mode={img.mode}")
            logger.info(f"  Prompt[0] preview: {prompts[0][:300]}")
            logger.info(f"  Actions[0]: {action_candidates_list[0]}")
            logger.info(f"[VL Batch Validation] === END FIRST CALL CHECK ===")

        # Batch generate with VL
        _batch_start = time.monotonic()
        raw_outputs = self.vl_engine.batch_generate(
            images=images,
            prompts=prompts,
            temperature=temperature,
            system_prompt=self.get_system_prompt(),
            **kwargs
        )
        _batch_elapsed = time.monotonic() - _batch_start

        # Parse outputs (unified: always use CoT-style parser)
        results = []
        for idx, (raw_output, action_candidates) in enumerate(zip(raw_outputs, action_candidates_list)):
            chosen_action, cot_prefix = self._parse_vl_output_with_cot(raw_output, action_candidates)
            action_log_probs = self._action_to_logprob(chosen_action, action_candidates, temperature)
            action_probs = np.exp(action_log_probs)

            # Store for logging
            if idx < 15:  # Only store first 15 for logging
                history = histories[idx] if idx < len(histories) else []
                prompt = prompts[idx]

                # Build action probability dict
                action_prob_dict = {
                    action: float(action_probs[i])
                    for i, action in enumerate(action_candidates)
                }

                self.episode_output.append({
                    "Instruction": prompt,
                    "Response": raw_output,
                    "vl_prior_per_seq": action_prob_dict,
                    "chosen_action": chosen_action,
                    "cot_prefix": cot_prefix,
                })

            results.append({
                'action_probs': action_probs,
                'action_logits': action_log_probs,
                'raw_output': raw_output,
                'cot_prefix': cot_prefix,
                'chosen_action': chosen_action,
            })

        # Log batch info at intervals (every 10 batch calls)
        if self.batch_call_count % 10 == 1:
            import logging
            logger = logging.getLogger(__name__)
            _action_dist = {}
            _parse_fail = 0
            for r in results:
                _action_dist[r['chosen_action']] = _action_dist.get(r['chosen_action'], 0) + 1
                if 'Action:' not in r.get('raw_output', ''):
                    _parse_fail += 1
            logger.info(
                f"[VL Batch] #{self.batch_call_count} | "
                f"size={len(observations)} | "
                f"actions={sum(len(a) for a in action_candidates_list) / len(action_candidates_list):.0f} | "
                f"time={_batch_elapsed:.2f}s ({_batch_elapsed / max(len(observations), 1):.2f}s/obs) | "
                f"parse_fail={_parse_fail}/{len(observations)} | "
                f"action_dist={_action_dist}"
            )

        return results

    def build_vl_train_samples(
        self,
        game_segments: List,
        advantages: np.ndarray,
        old_action_log_probs: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Build training samples for VL from game segments with advantages.

        This is the VL equivalent of LLM's build_llm_samples in datafactory.

        Args:
            game_segments: List of game segments from replay buffer
            advantages: Advantage values (target_value - pred_value) for each step
            old_action_log_probs: Old action log probabilities from collection

        Returns:
            List of training samples, each containing:
            - image: PIL Image
            - prompt: Full prompt with history and actions
            - target_action: The action that was taken
            - old_log_prob: Old log probability of the action
            - advantage: Advantage value for PPO loss
            - cot_prefix: CoT reasoning (if use_cot=True)
        """
        import logging
        logger = logging.getLogger(__name__)

        train_samples = []
        total_steps = 0

        logger.info(f"[VL Training Samples] Building samples from {len(game_segments)} segments...")

        for seg_idx, segment in enumerate(game_segments):
            # Extract segment data
            raw_obs_list = segment.raw_obs_segment  # List of image observations
            history_list = segment.history_obs_segment  # List of history tuples
            action_list = segment.action_segment  # List of action indices
            llm_action_list = segment.llm_action_segment  # List of action names
            cot_prefix_list = segment.cot_prefix_segment if hasattr(segment, 'cot_prefix_segment') else [None] * len(action_list)

            # Get valid actions for this environment
            # Assume all steps have same action space
            if hasattr(segment, 'valid_actions'):
                valid_actions = segment.valid_actions
            else:
                # Fallback: extract from first history or use generic
                valid_actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

            # Build samples for each step in segment
            for step_idx in range(len(action_list)):
                # Get observation (image)
                obs = raw_obs_list[step_idx]
                if isinstance(obs, np.ndarray):
                    image = self._convert_obs_to_pil_image(obs)
                else:
                    image = obs

                # Get history
                history = history_list[step_idx] if step_idx < len(history_list) else []

                # Get action
                action_idx = action_list[step_idx]
                action_name = llm_action_list[step_idx] if step_idx < len(llm_action_list) else valid_actions[action_idx]

                # Get advantage and old log prob
                advantage = advantages[seg_idx, step_idx] if seg_idx < len(advantages) else 0.0
                old_log_prob = old_action_log_probs[seg_idx, step_idx] if seg_idx < len(old_action_log_probs) else 0.0

                # Get CoT prefix (if available)
                cot_prefix = cot_prefix_list[step_idx] if step_idx < len(cot_prefix_list) else None

                # Build prompt (unified)
                prompt = self.get_user_prompt(valid_actions, history)

                # Create training sample
                sample = {
                    'image': image,
                    'prompt': prompt,
                    'target_action': action_name,
                    'old_log_prob': float(old_log_prob),
                    'advantage': float(advantage),
                    'cot_prefix': cot_prefix,
                    'valid_actions': valid_actions,
                }

                train_samples.append(sample)
                total_steps += 1

        # Log summary
        if len(train_samples) > 0:
            avg_advantage = np.mean([s['advantage'] for s in train_samples])
            avg_old_logprob = np.mean([s['old_log_prob'] for s in train_samples])
            logger.info(
                f"[VL Training Samples] Built {len(train_samples)} samples | "
                f"Avg advantage: {avg_advantage:.4f} | "
                f"Avg old_logprob: {avg_old_logprob:.4f}"
            )

        return train_samples

    def compute_action_log_prob(
        self,
        vl_output: str,
        target_action: str,
        valid_actions: List[str],
        temperature: float = 1.0
    ) -> float:
        """
        Compute log probability of target action from VL output.

        This is used during training to compute the new log probability
        for PPO ratio calculation.

        Args:
            vl_output: Raw VL output string
            target_action: The action that was actually taken
            valid_actions: List of valid action names
            temperature: Temperature for scaling

        Returns:
            Log probability of target action
        """
        if self.use_cot:
            # Parse CoT output to get chosen action
            chosen_action, _ = self._parse_vl_output_with_cot(vl_output, valid_actions)

            # Get log prob distribution
            log_probs = self._action_to_logprob(chosen_action, valid_actions, temperature)

            # Return log prob of target action
            try:
                target_idx = valid_actions.index(target_action)
                return float(log_probs[target_idx])
            except ValueError:
                # Target action not in valid actions
                return -10.0  # Very low log prob
        else:
            # Parse probability distribution
            probs = self._parse_vl_output(vl_output, valid_actions)
            log_probs = np.log(probs + 1e-10)

            try:
                target_idx = valid_actions.index(target_action)
                return float(log_probs[target_idx])
            except ValueError:
                return -10.0


    def get_vl_output_log(
        self,
        wm_train_iter: int,
        vl_train_iter: int,
    ) -> None:
        """
        Log VL output statistics (similar to LLM's get_llm_output_log).

        Args:
            wm_train_iter: World model training iteration
            vl_train_iter: VL training iteration
        """
        import logging
        logger = logging.getLogger(__name__)

        if len(self.episode_output) == 0:
            return

        logger.info(
            f"\n{'='*80}\n"
            f"[VL Output Log] WM Iter: {wm_train_iter} | VL Iter: {vl_train_iter}\n"
            f"{'='*80}"
        )

        for i, tmp_dict in enumerate(self.episode_output[:15]):
            instruction = tmp_dict["Instruction"]
            response = tmp_dict["Response"]
            vl_prior = tmp_dict["vl_prior_per_seq"]
            chosen_action = tmp_dict.get("chosen_action", "N/A")
            cot_prefix = tmp_dict.get("cot_prefix", "")

            logger.info(
                f"\n{'-'*80}\n"
                f"[Step {i}]\n"
                f"{'-'*80}\n"
                f"Instruction:\n{instruction}\n\n"
                f"Response:\n{response}\n\n"
                f"Chosen Action: {chosen_action}\n"
            )

            if cot_prefix:
                logger.info(f"CoT Reasoning:\n{cot_prefix}\n")

            logger.info("Action Probabilities:")

            # Sort actions by probability (descending)
            sorted_actions = sorted(vl_prior.items(), key=lambda x: x[1], reverse=True)

            for action, prob in sorted_actions:
                logger.info(f"  {action:30s} | prob={prob:.6f}")

        self.episode_output = []


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
        PriorGenerator instance (LLMPriorGenerator or VLPriorGenerator)
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
        # Create VL prior generator
        from vl_engine import create_vl_engine

        vl_engine = create_vl_engine(
            model_name=model_config['model_name'],
            model_path=model_config['model_path'],
            tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.3),
        )

        return VLPriorGenerator(
            vl_engine=vl_engine,
            model_name=model_config['model_name'],
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
    print("  - VLPriorGenerator: For image observations (Atari games)")
    print("\nUsage:")
    print("  generator = create_prior_generator(obs_type='image', model_config={...})")
    print("  prior = generator.generate_prior(observation, action_candidates)")
