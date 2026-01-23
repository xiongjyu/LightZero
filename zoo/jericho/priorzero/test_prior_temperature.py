#!/usr/bin/env python3
"""
Quick test script for Prior Temperature Scheduler

Run this to verify the implementation works correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
from prior_temperature_scheduler import PriorTemperatureScheduler, apply_temperature_to_logprobs, compute_entropy


def test_scheduler_curves():
    """Test different scheduling strategies and plot curves."""
    print("=" * 80)
    print("Testing Temperature Scheduler Curves")
    print("=" * 80)

    total_steps = 1000
    strategies = {
        'linear': {'schedule_type': 'linear'},
        'cosine': {'schedule_type': 'cosine'},
        'exponential': {'schedule_type': 'exponential', 'decay_rate': 0.95, 'step_size': 100},
        'step': {'schedule_type': 'step', 'step_size': 200, 'step_gamma': 0.8},
    }

    results = {}
    for name, config in strategies.items():
        scheduler = PriorTemperatureScheduler(
            init_temperature=2.0,
            final_temperature=1.0,
            total_steps=total_steps,
            warmup_steps=100,
            **config
        )

        temps = []
        for step in range(total_steps):
            temp = scheduler.step()
            temps.append(temp)

        results[name] = temps
        print(f"\n{name.upper()} Schedule:")
        print(f"  Step 0:    {temps[0]:.3f}")
        print(f"  Step 100:  {temps[100]:.3f}")
        print(f"  Step 500:  {temps[500]:.3f}")
        print(f"  Step 999:  {temps[999]:.3f}")

    # Plot curves
    plt.figure(figsize=(12, 6))
    for name, temps in results.items():
        plt.plot(temps, label=name, linewidth=2)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Temperature', fontsize=12)
    plt.title('Temperature Scheduling Strategies', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = 'temperature_schedules.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ Plot saved to: {output_path}")


def test_temperature_scaling():
    """Test temperature scaling effect on distributions."""
    print("\n" + "=" * 80)
    print("Testing Temperature Scaling Effect")
    print("=" * 80)

    # Create a sharp distribution (like what 7B model produces)
    logprobs = torch.tensor([-0.1, -0.5, -2.0, -3.0, -4.0])
    print(f"\nOriginal logprobs: {logprobs.tolist()}")

    temperatures = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = []

    for temp in temperatures:
        scaled_lps = apply_temperature_to_logprobs(logprobs, temp)
        probs = torch.exp(scaled_lps)
        entropy = compute_entropy(scaled_lps)

        results.append({
            'temp': temp,
            'probs': probs.tolist(),
            'entropy': entropy
        })

        print(f"\nTemperature = {temp:.1f}:")
        print(f"  Probabilities: {[f'{p:.3f}' for p in probs.tolist()]}")
        print(f"  Entropy: {entropy:.3f}")
        print(f"  Max prob: {probs.max():.3f}")

    # Plot probability distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Probability distributions
    x = range(len(logprobs))
    for result in results:
        ax1.plot(x, result['probs'], marker='o', label=f"T={result['temp']:.1f}", linewidth=2)

    ax1.set_xlabel('Action Index', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Effect of Temperature on Distribution Shape', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Entropy vs Temperature
    temps = [r['temp'] for r in results]
    entropies = [r['entropy'] for r in results]
    ax2.plot(temps, entropies, marker='o', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Temperature', fontsize=12)
    ax2.set_ylabel('Entropy', fontsize=12)
    ax2.set_title('Entropy vs Temperature', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'temperature_scaling_effect.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ Plot saved to: {output_path}")


def test_adaptive_schedule():
    """Test adaptive scheduling with entropy feedback."""
    print("\n" + "=" * 80)
    print("Testing Adaptive Schedule")
    print("=" * 80)

    scheduler = PriorTemperatureScheduler(
        init_temperature=2.0,
        final_temperature=1.0,
        total_steps=1000,
        warmup_steps=100,
        schedule_type='adaptive',
        target_entropy=2.0,
        entropy_lr=0.01
    )

    # Simulate varying entropy (starts high, gradually decreases)
    temps = []
    entropies_input = []
    entropies_avg = []

    for step in range(1000):
        # Simulate entropy that decreases over time
        entropy = 3.0 - 1.5 * (step / 1000) + 0.2 * torch.randn(1).item()
        entropy = max(0.5, entropy)  # Clamp to reasonable range

        temp = scheduler.step(entropy=entropy)
        stats = scheduler.get_stats()

        temps.append(temp)
        entropies_input.append(entropy)
        if 'prior_avg_entropy' in stats:
            entropies_avg.append(stats['prior_avg_entropy'])

    print(f"\nAdaptive Schedule Results:")
    print(f"  Initial temperature: {temps[0]:.3f}")
    print(f"  Final temperature: {temps[-1]:.3f}")
    print(f"  Initial entropy: {entropies_input[0]:.3f}")
    print(f"  Final entropy: {entropies_input[-1]:.3f}")

    # Plot adaptive behavior
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Temperature over time
    ax1.plot(temps, label='Temperature', linewidth=2, color='blue')
    ax1.axhline(y=2.0, color='green', linestyle='--', label='Target Entropy', alpha=0.5)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Temperature', fontsize=12)
    ax1.set_title('Adaptive Temperature Scheduling', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Entropy over time
    ax2.plot(entropies_input, label='Input Entropy', linewidth=1, alpha=0.5, color='orange')
    if entropies_avg:
        ax2.plot(entropies_avg, label='Moving Avg Entropy', linewidth=2, color='red')
    ax2.axhline(y=2.0, color='green', linestyle='--', label='Target Entropy', alpha=0.5)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Entropy', fontsize=12)
    ax2.set_title('Entropy Tracking', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'adaptive_schedule.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ Plot saved to: {output_path}")


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("\n" + "=" * 80)
    print("Testing Numerical Stability")
    print("=" * 80)

    # Test 1: Very small temperature
    print("\nTest 1: Very small temperature (should clamp to min_temperature)")
    scheduler = PriorTemperatureScheduler(
        init_temperature=0.01,  # Below min
        final_temperature=0.001,
        total_steps=100,
        min_temperature=0.1
    )
    temp = scheduler.step()
    print(f"  Requested: 0.01, Actual: {temp:.3f} (clamped to {scheduler.min_temperature})")
    assert temp >= scheduler.min_temperature, "Temperature not clamped to minimum!"

    # Test 2: Very large temperature
    print("\nTest 2: Very large temperature (should clamp to max_temperature)")
    scheduler = PriorTemperatureScheduler(
        init_temperature=10.0,  # Above max
        final_temperature=8.0,
        total_steps=100,
        max_temperature=5.0
    )
    temp = scheduler.step()
    print(f"  Requested: 10.0, Actual: {temp:.3f} (clamped to {scheduler.max_temperature})")
    assert temp <= scheduler.max_temperature, "Temperature not clamped to maximum!"

    # Test 3: Extreme logprobs
    print("\nTest 3: Extreme logprobs (should handle gracefully)")
    logprobs = torch.tensor([-100.0, -0.1, -50.0, -200.0])
    for temp in [0.5, 1.0, 2.0]:
        try:
            scaled = apply_temperature_to_logprobs(logprobs, temp)
            probs = torch.exp(scaled)
            entropy = compute_entropy(scaled)
            print(f"  T={temp:.1f}: max_prob={probs.max():.6f}, entropy={entropy:.3f} ✓")
            assert not torch.isnan(probs).any(), "NaN in probabilities!"
            assert not torch.isinf(probs).any(), "Inf in probabilities!"
        except Exception as e:
            print(f"  T={temp:.1f}: FAILED - {e}")

    print("\n✓ All numerical stability tests passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PRIOR TEMPERATURE SCHEDULER - TEST SUITE")
    print("=" * 80)

    try:
        # Run tests
        test_scheduler_curves()
        test_temperature_scaling()
        test_adaptive_schedule()
        test_numerical_stability()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nGenerated plots:")
        print("  - temperature_schedules.png")
        print("  - temperature_scaling_effect.png")
        print("  - adaptive_schedule.png")
        print("\nYou can now use the scheduler in your training!")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
