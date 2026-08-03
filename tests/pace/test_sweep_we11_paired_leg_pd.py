from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PACE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pace"
sys.path.insert(0, str(PACE_DIR))

from sweep_we11_paired_leg_pd import (  # noqa: E402
    GROUP_BASELINE,
    BodeReference,
    SearchRange,
    group_search_range,
    parse_args,
    reduce_paired_metrics,
    selected_groups,
    validate_rk4_model,
    validate_search_range,
)


class ParserAndRangeTests(unittest.TestCase):
    def test_defaults_cover_group_a_acquisition_gains(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["sweep_we11_paired_leg_pd.py", "--pace-params", "params.json"],
        ):
            args = parse_args()
        for group, baseline in GROUP_BASELINE.items():
            validate_search_range(group, group_search_range(args, group), baseline)
        self.assertEqual(args.workers, 16)
        self.assertEqual(args.score_metric, "bode-complex")

    def test_calf_default_range_contains_eight_and_point_eight(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["sweep_we11_paired_leg_pd.py", "--pace-params", "params.json"],
        ):
            args = parse_args()
        search = group_search_range(args, "calf")
        self.assertLessEqual(search.kp_min, 8.0)
        self.assertGreaterEqual(search.kp_max, 8.0)
        self.assertLessEqual(search.kd_min, 0.8)
        self.assertGreaterEqual(search.kd_max, 0.8)

    def test_rejects_range_excluding_acquisition_baseline(self) -> None:
        search = SearchRange(1.0, 3.0, 0.1, 0.03, 0.09, 0.01)
        with self.assertRaisesRegex(ValueError, "acquisition baseline"):
            validate_search_range("thigh", search, GROUP_BASELINE["thigh"])

    def test_group_selection_deduplicates(self) -> None:
        self.assertEqual(selected_groups(["calf", "thigh", "calf"]), ("calf", "thigh"))


class PairedReductionTests(unittest.TestCase):
    def test_metrics_are_equal_weighted_across_sides(self) -> None:
        class FakeReference:
            def __init__(self, offset: float) -> None:
                self.offset = offset

            def metrics(self, response: np.ndarray) -> dict[str, float]:
                base = float(np.mean(response)) + self.offset
                return {
                    "bode_magnitude_mse_db2": base,
                    "bode_magnitude_rmse_db": base + 1.0,
                    "bode_phase_mse_deg2": base + 2.0,
                    "bode_phase_rmse_deg": base + 3.0,
                    "bode_complex_score": base + 4.0,
                    "time_rmse": base + 5.0,
                    "time_normalized_mse": base + 6.0,
                }

        response = np.column_stack((np.ones(8), np.full(8, 3.0)))
        metrics = reduce_paired_metrics(
            (FakeReference(0.0), FakeReference(2.0)),  # type: ignore[arg-type]
            response,
        )
        self.assertEqual(metrics["left_bode_magnitude_mse_db2"], 1.0)
        self.assertEqual(metrics["right_bode_magnitude_mse_db2"], 5.0)
        self.assertEqual(metrics["bode_magnitude_mse_db2"], 3.0)

    def test_rejects_single_trace_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            reduce_paired_metrics(
                (object(), object()),  # type: ignore[arg-type]
                np.zeros(10),
            )


class IntegratorValidationTests(unittest.TestCase):
    def test_accepts_exact_rk4_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.xml"
            path.write_text('<mujoco><option integrator="RK4"/></mujoco>', encoding="utf-8")
            import hashlib

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            validate_rk4_model(path, digest)

    def test_rejects_implicit_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.xml"
            path.write_text('<mujoco><option integrator="implicit"/></mujoco>', encoding="utf-8")
            import hashlib

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "RK4"):
                validate_rk4_model(path, digest)


if __name__ == "__main__":
    unittest.main()
