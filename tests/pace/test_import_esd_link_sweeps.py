from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pace"
sys.path.insert(0, str(PACE_DIR))

from import_esd_link_sweeps import (  # noqa: E402
    CHIRP_DURATION_S,
    CHIRP_END_HZ,
    CHIRP_START_HZ,
    SPECS,
    import_source,
    parse_filename,
    read_numeric_columns,
    required_columns,
    select_formal_segment,
    unwrap_wheel,
)
from mujoco_dr002_common import DEFAULT_ANGLES, load_chirp_data  # noqa: E402


class EsdLinkSweepImportTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, *, pair: str = "14", kp: float = 2.0, kd: float = 0.1) -> None:
        spec = SPECS[pair]
        numeric = sorted(required_columns(spec))
        fieldnames = numeric + [f"p{motor_id}_temperature" for motor_id in range(1, 7)]
        formal_time = np.arange(0.0, CHIRP_DURATION_S, 1.0 / 150.0)
        post_time = np.arange(CHIRP_DURATION_S, CHIRP_DURATION_S + 0.31, 1.0 / 150.0)
        time = np.concatenate((formal_time, post_time))
        phase = 2.0 * np.pi * (
            CHIRP_START_HZ * formal_time
            + 0.5
            * (CHIRP_END_HZ - CHIRP_START_HZ)
            / CHIRP_DURATION_S
            * np.square(formal_time)
        )
        formal_command = spec.center + spec.amplitude * np.sin(phase)
        command = np.concatenate((formal_command, np.full(post_time.size, spec.center)))
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for index, timestamp in enumerate(time):
                row: dict[str, object] = {
                    "host_monotonic_ns": int(round((100.0 + timestamp) * 1.0e9)),
                    "device_sample_time_us": int(round(timestamp * 1.0e6)),
                    "state_sample_seq": index + (1 if index >= 100 else 0),
                    "source_state_sample_seq": index,
                    "last_applied_command_seq": index,
                    "command_status_flags": 0,
                }
                for motor_id in range(1, 7):
                    active = motor_id in spec.motor_ids
                    target = command[index] if active else 0.0
                    row.update(
                        {
                            f"p{motor_id}_cmd_q": target
                            if spec.target_type == "position"
                            else 0.0,
                            f"p{motor_id}_cmd_dq": target
                            if spec.target_type == "velocity"
                            else 0.0,
                            f"p{motor_id}_kp": kp if active else 1.0,
                            f"p{motor_id}_kd": kd if active else 0.1,
                            f"p{motor_id}_q": target - 0.01 if active else 0.0,
                            f"p{motor_id}_dq": 0.2 if active else 0.0,
                            f"p{motor_id}_tau": 0.3 if active else 0.0,
                            f"p{motor_id}_temperature": "unavailable",
                        }
                    )
                writer.writerow(row)

    def test_filename_parser_recognizes_current_gain_tokens(self) -> None:
        spec, kp, kd = parse_filename(
            Path("motor36_20260910_203210_kp0_kd0p2.csv")
        )
        self.assertEqual(spec.pair, "36")
        self.assertEqual((kp, kd), (0.0, 0.2))

    def test_numeric_reader_ignores_unavailable_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motor14_20260910_000000_kp2_kd0p1.csv"
            self._write_csv(path)
            data = read_numeric_columns(path, required_columns(SPECS["14"]))
        self.assertIn("p1_q", data)
        self.assertNotIn("p1_temperature", data)

    def test_missing_and_nonfinite_numeric_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["a"])
                writer.writeheader()
                for _ in range(101):
                    writer.writerow({"a": "nan"})
            with self.assertRaises(KeyError):
                read_numeric_columns(path, {"a", "b"})
            with self.assertRaisesRegex(ValueError, "not finite"):
                read_numeric_columns(path, {"a"})

    def test_formal_segment_excludes_post_hold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motor14_20260910_000000_kp2_kd0p1.csv"
            self._write_csv(path)
            data = read_numeric_columns(path, required_columns(SPECS["14"]))
            time, selected, metadata = select_formal_segment(data, SPECS["14"])
        self.assertGreater(time[-1], 39.9)
        self.assertLess(time[-1], 40.01)
        self.assertLess(selected.stop, len(data["host_monotonic_ns"]))
        self.assertEqual(
            metadata["selection"],
            "40_seconds_immediately_before_final_stable_command_center_hold",
        )

    def test_import_maps_policy_coordinates_and_resamples_to_200hz(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "motor14_20260910_000000_kp2_kd0p1.csv"
            self._write_csv(path)
            out = root / "out"
            out.mkdir()
            source, quality = import_source(path, out)
            payload = load_chirp_data(out / source["pt"])
        np.testing.assert_allclose(np.diff(payload["time"]), 0.005, atol=1.0e-12)
        self.assertEqual(source["sample_rate_hz"], 200.0)
        self.assertGreater(source["chirp_validation"]["waveform_correlation"], 0.99)
        self.assertEqual(quality["state_sample_sequence_missing_count"], 1)
        self.assertAlmostEqual(
            payload["des_dof_pos"][0, 0], DEFAULT_ANGLES[0] + SPECS["14"].center
        )
        self.assertAlmostEqual(
            payload["dof_pos"][0, 0], DEFAULT_ANGLES[0] + SPECS["14"].center - 0.01
        )

    def test_wheel_unwrap_uses_esd_link_period_and_starts_at_zero(self) -> None:
        wrapped = np.asarray([6.20, 6.27, -6.27, -6.10])
        actual = unwrap_wheel(wrapped)
        np.testing.assert_allclose(actual, [0.0, 0.07, 0.09, 0.26], atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
