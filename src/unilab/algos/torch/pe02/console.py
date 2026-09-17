"""PE02-owned console formatting for the full PPO training runner."""

from collections.abc import Mapping


def _duration(seconds: float) -> str:
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_iteration(
    metrics: Mapping[str, float],
    *,
    target_iteration: int,
    completed_iterations: int,
    iteration_seconds: float,
    elapsed_seconds: float,
    policy_dt: float,
) -> str:
    """Format existing metrics without changing their TensorBoard/JSONL semantics."""
    iteration = int(metrics["iteration"])
    lines = ["", f"PE02 | Learning iteration {iteration}/{target_iteration}".center(80), ""]

    def row(label: str, value: str) -> None:
        lines.append(f"{label:>38}: {value}")

    row("Total steps", f"{metrics['samples']:.0f}")
    row("Steps per second", f"{metrics['samples_per_second']:.0f}")
    row("Collection time", f"{metrics['collection_seconds']:.3f}s")
    row("Learning time", f"{metrics['update_seconds']:.3f}s")
    for label, key in (
        ("Mean value loss", "value_loss"),
        ("Mean surrogate loss", "policy_loss"),
        ("Mean entropy", "entropy"),
        ("Mean encoder loss", "encoder_loss"),
        ("Mean KL divergence", "kl"),
        ("Clip fraction", "clip_fraction"),
    ):
        row(label, f"{metrics[key]:.4f}")
    row("Learning rate", f"{metrics['learning_rate']:.6f}")
    row("Mean step reward", f"{metrics['mean_step_reward']:.5f}")
    if "episode/return" in metrics:
        row("Mean reward", f"{metrics['episode/return']:.2f}")
        row("Mean episode length", f"{metrics['episode/seconds'] / policy_dt:.2f} steps")
        row("Mean episode time", f"{metrics['episode/seconds']:.2f}s")
    else:
        row("Mean reward", "n/a (no completed episodes)")
        row("Mean episode length", "n/a (no completed episodes)")
    row("Mean action std", f"{metrics['action_std']:.2f}")
    row("PPO updates", f"{metrics['optimizer_updates']:.0f}")
    row("Encoder updates", f"{metrics['encoder_updates']:.0f}")
    for prefix in ("reward/", "evaluation/"):
        for key, value in metrics.items():
            if key.startswith(prefix):
                row(key, f"{value:.4f}")
    lines.append("-" * 80)
    row("Iteration time", f"{iteration_seconds:.2f}s")
    row("Time elapsed", _duration(elapsed_seconds))
    remaining = max(0, target_iteration - iteration)
    eta = elapsed_seconds / max(1, completed_iterations) * remaining
    row("ETA", _duration(eta))
    return "\n".join(lines)
