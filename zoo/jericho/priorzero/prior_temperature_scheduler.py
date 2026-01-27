"""
Prior Temperature Scheduler for PriorZero

This module provides temperature scheduling for prior probability extraction,
enabling exploration in early training and exploitation in later stages.

Key Features:
- Multiple scheduling strategies (linear, cosine, exponential, step)
- Entropy-based adaptive scheduling
- Robust numerical stability
- TensorBoard logging integration
- Easy configuration and extensibility
"""

import math
from typing import Optional, Dict, Literal
import torch


class PriorTemperatureScheduler:
    """
    Temperature scheduler for prior probability extraction.

    Temperature controls the sharpness of the prior distribution:
    - High temperature (e.g., 2.0): Flatter distribution, more exploration
    - Low temperature (e.g., 0.5): Sharper distribution, more exploitation
    - Temperature = 1.0: Original distribution (no modification)

    The scheduler gradually transitions from high to low temperature during training.
    """

    def __init__(
        self,
        init_temperature: float = 2.0,
        final_temperature: float = 1.0,
        total_steps: int = 10000,
        warmup_steps: int = 0,
        schedule_type: Literal["linear", "cosine", "exponential", "step", "adaptive"] = "cosine",
        # Exponential decay parameters
        decay_rate: float = 0.95,
        # Step decay parameters
        step_size: int = 1000,
        step_gamma: float = 0.8,
        # Adaptive parameters
        target_entropy: Optional[float] = None,
        entropy_window: int = 100,
        entropy_lr: float = 0.01,
        # Numerical stability
        min_temperature: float = 0.1,
        max_temperature: float = 5.0,
    ):
        """
        Args:
            init_temperature: Initial temperature (high for exploration)
            final_temperature: Final temperature (low for exploitation)
            total_steps: Total training steps
            warmup_steps: Steps to keep at init_temperature
            schedule_type: Type of scheduling strategy
            decay_rate: Decay rate for exponential schedule
            step_size: Step size for step schedule
            step_gamma: Decay factor for step schedule
            target_entropy: Target entropy for adaptive schedule (None = auto)
            entropy_window: Window size for entropy moving average
            entropy_lr: Learning rate for adaptive temperature adjustment
            min_temperature: Minimum allowed temperature (numerical stability)
            max_temperature: Maximum allowed temperature (numerical stability)
        """
        assert init_temperature > 0, "init_temperature must be positive"
        assert final_temperature > 0, "final_temperature must be positive"
        assert total_steps > 0, "total_steps must be positive"
        assert warmup_steps >= 0, "warmup_steps must be non-negative"
        assert min_temperature > 0, "min_temperature must be positive"
        assert max_temperature > min_temperature, "max_temperature must be greater than min_temperature"

        self.init_temperature = init_temperature
        self.final_temperature = final_temperature
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type

        # Exponential parameters
        self.decay_rate = decay_rate

        # Step parameters
        self.step_size = step_size
        self.step_gamma = step_gamma

        # Adaptive parameters
        self.target_entropy = target_entropy
        self.entropy_window = entropy_window
        self.entropy_lr = entropy_lr
        self.entropy_history = []

        # Numerical stability
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        # State
        self.current_step = 0
        self.current_temperature = init_temperature

    def step(self, entropy: Optional[float] = None) -> float:
        """
        Update temperature for the current step.

        Args:
            entropy: Current entropy of the prior distribution (for adaptive schedule)

        Returns:
            Current temperature value
        """
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            self.current_temperature = self.init_temperature
        else:
            progress = (self.current_step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)  # Clamp to [0, 1]

            if self.schedule_type == "linear":
                self.current_temperature = self._linear_schedule(progress)
            elif self.schedule_type == "cosine":
                self.current_temperature = self._cosine_schedule(progress)
            elif self.schedule_type == "exponential":
                self.current_temperature = self._exponential_schedule(progress)
            elif self.schedule_type == "step":
                self.current_temperature = self._step_schedule()
            elif self.schedule_type == "adaptive":
                self.current_temperature = self._adaptive_schedule(entropy, progress)
            else:
                raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        # Clamp temperature for numerical stability
        self.current_temperature = max(self.min_temperature, min(self.max_temperature, self.current_temperature))

        return self.current_temperature

    def _linear_schedule(self, progress: float) -> float:
        """Linear interpolation from init to final temperature."""
        return self.init_temperature + (self.final_temperature - self.init_temperature) * progress

    def _cosine_schedule(self, progress: float) -> float:
        """Cosine annealing from init to final temperature."""
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.final_temperature + (self.init_temperature - self.final_temperature) * cosine_decay

    def _exponential_schedule(self, progress: float) -> float:
        """Exponential decay from init to final temperature."""
        decay_steps = self.total_steps - self.warmup_steps
        current_decay_step = int(progress * decay_steps)
        decay_factor = self.decay_rate ** (current_decay_step / self.step_size)
        return max(self.final_temperature, self.init_temperature * decay_factor)

    def _step_schedule(self) -> float:
        """Step-wise decay from init to final temperature."""
        num_decays = (self.current_step - self.warmup_steps) // self.step_size
        decay_factor = self.step_gamma ** num_decays
        return max(self.final_temperature, self.init_temperature * decay_factor)

    def _adaptive_schedule(self, entropy: Optional[float], progress: float) -> float:
        """
        Adaptive temperature based on entropy feedback.

        If entropy is too low (distribution too sharp), increase temperature.
        If entropy is too high (distribution too flat), decrease temperature.
        """
        # Use cosine schedule as baseline
        baseline_temp = self._cosine_schedule(progress)

        if entropy is None:
            return baseline_temp

        # Track entropy history
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > self.entropy_window:
            self.entropy_history.pop(0)

        # Compute moving average entropy
        avg_entropy = sum(self.entropy_history) / len(self.entropy_history)

        # Auto-set target entropy if not specified
        if self.target_entropy is None:
            # Target entropy = 70% of maximum possible entropy
            # This encourages some exploration while still being decisive
            self.target_entropy = 0.7 * avg_entropy if len(self.entropy_history) < 10 else self.target_entropy
            return baseline_temp

        # Adjust temperature based on entropy gap
        entropy_gap = avg_entropy - self.target_entropy
        adjustment = self.entropy_lr * entropy_gap

        # Apply adjustment to baseline
        adjusted_temp = baseline_temp - adjustment

        return adjusted_temp

    def get_temperature(self) -> float:
        """Get current temperature without stepping."""
        return self.current_temperature

    def get_stats(self) -> Dict[str, float]:
        """Get statistics for logging."""
        stats = {
            "prior_temperature": self.current_temperature,
            "temperature_step": self.current_step,
            "temperature_progress": min(self.current_step / self.total_steps, 1.0),
        }

        if self.schedule_type == "adaptive" and self.entropy_history:
            avg_entropy = sum(self.entropy_history) / len(self.entropy_history)
            stats["prior_avg_entropy"] = avg_entropy
            if self.target_entropy is not None:
                stats["prior_target_entropy"] = self.target_entropy
                stats["prior_entropy_gap"] = avg_entropy - self.target_entropy

        return stats

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0
        self.current_temperature = self.init_temperature
        self.entropy_history = []

    def state_dict(self) -> Dict:
        """Get state for checkpointing."""
        return {
            "current_step": self.current_step,
            "current_temperature": self.current_temperature,
            "entropy_history": self.entropy_history.copy(),
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint."""
        self.current_step = state_dict.get("current_step", 0)
        self.current_temperature = state_dict.get("current_temperature", self.init_temperature)
        self.entropy_history = state_dict.get("entropy_history", [])


def apply_temperature_to_logprobs(logprobs: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to log probabilities.

    Args:
        logprobs: Log probabilities (any shape)
        temperature: Temperature value (> 0)

    Returns:
        Temperature-scaled log probabilities

    Note:
        - Temperature > 1: Flatter distribution (more exploration)
        - Temperature < 1: Sharper distribution (more exploitation)
        - Temperature = 1: No change

    Mathematical formulation:
        p_temp(x) = p(x)^(1/T) / Z
        log p_temp(x) = log p(x) / T - log Z

    For numerical stability, we work in log space:
        log p_temp(x) = (log p(x) - log_sum_exp(log p(x) / T)) / T
    """
    if temperature == 1.0:
        return logprobs

    # Scale logits by temperature
    scaled_logprobs = logprobs / temperature

    # Renormalize (important for maintaining valid probability distribution)
    # This is equivalent to: log(softmax(logits / T))
    # We use log_softmax for numerical stability
    if logprobs.dim() > 1:
        # Assume last dimension is the probability dimension
        scaled_logprobs = torch.log_softmax(scaled_logprobs, dim=-1)
    else:
        # 1D case
        scaled_logprobs = torch.log_softmax(scaled_logprobs, dim=0)

    return scaled_logprobs


def compute_entropy(logprobs: torch.Tensor) -> float:
    """
    Compute entropy of a probability distribution from log probabilities.

    Args:
        logprobs: Log probabilities (any shape)

    Returns:
        Entropy value (in nats)

    Formula:
        H(p) = -sum(p(x) * log p(x))
    """
    # Convert to probabilities
    probs = torch.exp(logprobs)

    # Compute entropy: -sum(p * log(p))
    # Use logprobs directly to avoid numerical issues
    entropy = -(probs * logprobs).sum()

    return entropy.item()


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Prior Temperature Scheduler - Examples")
    print("=" * 80)

    # Example 1: Cosine schedule
    print("\n1. Cosine Schedule (Recommended)")
    scheduler = PriorTemperatureScheduler(
        init_temperature=2.0,
        final_temperature=1.0,
        total_steps=1000,
        warmup_steps=100,
        schedule_type="cosine"
    )

    for step in [0, 100, 300, 500, 700, 900, 1000]:
        scheduler.current_step = step - 1
        temp = scheduler.step()
        print(f"  Step {step:4d}: Temperature = {temp:.3f}")

    # Example 2: Adaptive schedule
    print("\n2. Adaptive Schedule (Entropy-based)")
    scheduler = PriorTemperatureScheduler(
        init_temperature=2.0,
        final_temperature=1.0,
        total_steps=1000,
        schedule_type="adaptive",
        target_entropy=2.0,
        entropy_lr=0.01
    )

    # Simulate varying entropy
    entropies = [3.0, 2.8, 2.5, 2.2, 2.0, 1.8, 1.5]
    for i, ent in enumerate(entropies):
        temp = scheduler.step(entropy=ent)
        stats = scheduler.get_stats()
        print(f"  Step {i+1}: Entropy = {ent:.2f}, Temperature = {temp:.3f}")

    # Example 3: Temperature scaling effect
    print("\n3. Temperature Scaling Effect")
    logprobs = torch.tensor([-0.1, -0.5, -2.0, -3.0])  # Sharp distribution
    print(f"  Original logprobs: {logprobs.tolist()}")
    print(f"  Original probs: {torch.exp(logprobs).tolist()}")

    for temp in [0.5, 1.0, 2.0]:
        scaled = apply_temperature_to_logprobs(logprobs, temp)
        probs = torch.exp(scaled)
        entropy = compute_entropy(scaled)
        print(f"  Temperature {temp:.1f}: probs = {probs.tolist()}, entropy = {entropy:.3f}")

    print("\n" + "=" * 80)
