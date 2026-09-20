"""Present PE02 metrics in the same TensorBoard groups as WE11/RSL-RL."""

from collections.abc import Mapping

_TAGS = {
    "policy_loss": "Loss/surrogate",
    "value_loss": "Loss/value",
    "entropy": "Loss/entropy",
    "kl": "Loss/kl",
    "clip_fraction": "Loss/clip_fraction",
    "encoder_loss": "Loss/encoder",
    "learning_rate": "Loss/learning_rate",
    "action_std": "Policy/mean_std",
    "collection_seconds": "Perf/collection_time",
    "update_seconds": "Perf/learning_time",
    "samples_per_second": "Perf/total_fps",
    "iteration": "Train/iteration",
    "samples": "Train/total_timesteps",
    "mean_step_reward": "Train/mean_step_reward",
    "optimizer_updates": "Train/ppo_updates",
    "encoder_updates": "Train/encoder_updates",
    "episode/return": "Train/mean_reward",
    "episode/seconds": "Train/mean_episode_time",
}


def tensorboard_metrics(metrics: Mapping[str, float], *, policy_dt: float) -> dict[str, float]:
    """Keep values intact; additionally express episode length in policy steps.

    PE02 reward/* remains the weighted per-policy-step contribution, unlike
    WE11's pre-dt reward logging. JSONL and console metrics retain their names.
    """
    result = {}
    for key, value in metrics.items():
        if key.startswith("evaluation/"):
            tag = "Eval/" + key.removeprefix("evaluation/")
        else:
            tag = _TAGS.get(key, key if "/" in key else f"Diagnostics/{key}")
        result[tag] = value
    if "episode/seconds" in metrics:
        result["Train/mean_episode_length"] = metrics["episode/seconds"] / policy_dt
    return result
