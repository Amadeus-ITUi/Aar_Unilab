"""Episode-aware foot diagnostics; never count frames after termination."""

import numpy as np


class GaitMetrics:
    def __init__(self, dt):
        self.dt = dt
        self.samples = {}
        self.previous_contact = None
        self.liftoffs = np.zeros(2)

    def update(self, behavior, live):
        contact = behavior["foot_contact"]
        if self.previous_contact is not None:
            self.liftoffs += (self.previous_contact[live] & ~contact[live]).sum(0)
        self.previous_contact = contact.copy()
        values = {
            "contact": contact,
            "swing": behavior["swing"],
            "height": behavior["foot_clearance"],
            "target": behavior["target_clearance"],
            "speed": np.linalg.norm(behavior["foot_reference_velocity"], axis=-1),
        }
        values.update({f"reward/{k}": v for k, v in behavior["reward_terms"].items()})
        values.update({f"foot_reward/{k}": v for k, v in behavior["foot_reward_terms"].items()})
        for key, value in values.items():
            self.samples.setdefault(key, []).append(value[live].copy())

    def result(self):
        data = {k: np.concatenate(v) for k, v in self.samples.items()}
        if not data or not len(data["contact"]):
            return {}
        result = {"gait/double_support_fraction": float(data["contact"].all(1).mean())}
        for i, side in enumerate(("left", "right")):
            swing = data["swing"][:, i]
            contact = data["contact"][:, i]
            high = swing & (data["target"][:, i] > 0.02)
            error = np.abs(data["height"][:, i] - data["target"][:, i])[swing]
            metrics = {
                "contact_duty_fraction": float(contact.mean()),
                "swing_loaded_fraction": float((contact & swing).sum() / max(1, swing.sum())),
                "stance_speed_mps": float(data["speed"][~swing, i].sum() / max(1, (~swing).sum())),
                "liftoffs_per_second": float(self.liftoffs[i] / (len(contact) * self.dt)),
                "swing_samples": float(swing.sum()),
                "mid_swing_samples": float(high.sum()),
                "mid_swing_drag_fraction": float(
                    ((data["height"][:, i] < 0.005) & high).sum() / max(1, high.sum())
                ),
            }
            for percentile in (50, 90, 95):
                metrics[f"clearance_error_p{percentile}_m"] = (
                    float(np.percentile(error, percentile)) if len(error) else 0.0
                )
            result.update({f"gait/{side}/{k}": v for k, v in metrics.items()})
            for key, values in data.items():
                if key.startswith("foot_reward/"):
                    result[f"gait/{side}/reward/{key.removeprefix('foot_reward/')}"] = float(
                        values[:, i].mean()
                    )
        for key, values in data.items():
            if key.startswith("reward/"):
                result[key] = float(values.mean())
        return result
