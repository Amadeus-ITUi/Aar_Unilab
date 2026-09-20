from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pace"
sys.path.insert(0, str(PACE_DIR))

from import_real_sweep_csv_for_pd_fit import (  # noqa: E402
    JULY30_PD_GROUPS,
    JULY30_POLICY_SWEEP_CENTER,
    JULY30_RAW_ENCODER_CENTER,
    JULY30_RAW_TO_POLICY_SIGN,
    build_coordinate_contract,
    import_july30_paired_directory,
    import_one,
    parse_paired_csv_name,
    resample_uniform,
    resampling_metadata,
    sha256_file,
    slice_after_last_timestamp_reset,
)
from mujoco_dr002_common import DEFAULT_ANGLES, load_chirp_data  # noqa: E402


class RealSweepResamplingTests(unittest.TestCase):
    def test_command_uses_zero_order_hold_while_sensors_remain_linear(self) -> None:
        time = np.asarray([0.0, 0.005, 0.0125])
        command = np.asarray([0.0, 1.0, -1.0])
        measured_position = 10.0 * time
        measured_velocity = 20.0 * time

        uniform_time, traces = resample_uniform(
            time,
            {"cmd": command, "act": measured_position, "vel": measured_velocity},
            0.0025,
            zero_order_hold_traces={"cmd"},
        )

        np.testing.assert_allclose(uniform_time, [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125])
        np.testing.assert_array_equal(traces["cmd"], [0.0, 0.0, 1.0, 1.0, 1.0, -1.0])
        np.testing.assert_allclose(traces["act"], 10.0 * uniform_time)
        np.testing.assert_allclose(traces["vel"], 20.0 * uniform_time)

    def test_manifest_metadata_names_each_resampling_method(self) -> None:
        metadata = resampling_metadata(0.0025)

        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["target_dt_s"], 0.0025)
        self.assertEqual(metadata["target_sample_rate_hz"], 400.0)
        self.assertEqual(
            metadata["methods"],
            {
                "command_position": "zero_order_hold_previous_sample",
                "measured_position": "linear_interpolation",
                "measured_velocity": "linear_interpolation",
            },
        )

    def test_disabled_resampling_is_explicit_in_metadata(self) -> None:
        metadata = resampling_metadata(0.0)

        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["target_sample_rate_hz"], 0.0)
        self.assertEqual(set(metadata["methods"].values()), {"none_original_samples"})

    def test_import_save_load_preserves_exact_200hz_controller_ticks(self) -> None:
        # Five seconds is already long enough for float32 timestamps to cross
        # exact 200 Hz query ticks hundreds of times.  Exercise the complete
        # CSV -> resample -> torch.save -> torch.load path to prevent that
        # regression from returning.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sweep_motor_4_right_thigh_joint_20260725_010203_001.csv"
            source_time = np.arange(0.0, 5.0 + 0.5 * 0.005, 0.005, dtype=np.float64)
            with source.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "motor_4_cmd_pos",
                        "motor_4_act_pos",
                        "motor_4_act_vel",
                    ]
                )
                for index, timestamp in enumerate(source_time):
                    command = 0.001 * index
                    writer.writerow([timestamp, command, command - 0.01, 0.2])

            controller_time = np.arange(source_time.size, dtype=np.float64) / 200.0
            resampled_time, _ = resample_uniform(
                source_time,
                {"cmd": source_time},
                0.005,
                zero_order_hold_traces={"cmd"},
            )
            pre_save_indices = (
                np.searchsorted(
                    resampled_time,
                    controller_time,
                    side="right",
                )
                - 1
            )
            np.testing.assert_array_equal(
                pre_save_indices,
                np.arange(source_time.size, dtype=np.int64),
            )

            out_dir = root / "imported"
            out_dir.mkdir()
            source_meta = import_one(
                source,
                out_dir,
                case="time_precision_regression",
                default_angles=DEFAULT_ANGLES.copy(),
                align_initial=False,
                alignment_window_s=0.5,
                resample_dt=0.005,
                strict_single_motor=True,
            )

            pt_path = out_dir / source_meta["pt"]
            import torch

            stored = torch.load(pt_path, map_location="cpu")
            self.assertEqual(stored["time"].dtype, torch.float64)
            loaded = load_chirp_data(pt_path)
            expected_time = np.arange(source_time.size, dtype=np.float64) / 200.0

            self.assertEqual(loaded["time"].dtype, np.dtype(np.float64))
            np.testing.assert_array_equal(loaded["time"], expected_time)

            selected_indices = (
                np.searchsorted(
                    loaded["time"],
                    controller_time,
                    side="right",
                )
                - 1
            )
            np.testing.assert_array_equal(
                selected_indices,
                np.arange(expected_time.size, dtype=np.int64),
            )

            self.assertEqual(source_meta["source_csv_sha256"], sha256_file(source))
            self.assertEqual(source_meta["pt_sha256"], sha256_file(pt_path))
            self.assertEqual(
                source_meta["truth_csv_sha256"],
                sha256_file(out_dir / source_meta["truth_csv"]),
            )
            self.assertEqual(
                source_meta["serialization_dtypes"],
                {"time": "float64", "signals": "float32"},
            )


class July30PairedSweepImportTests(unittest.TestCase):
    MUJOCO_CENTER = np.asarray(
        [1.19980, -2.25662, 0.0, 1.19980, -2.25662, 0.0],
        dtype=np.float64,
    )
    MUJOCO_SIGN = np.ones(6, dtype=np.float64)
    RUNTIME_DEFAULT = np.asarray(
        [-0.920, 1.07, 0.0, -0.920, 1.07, 0.0],
        dtype=np.float64,
    )

    @staticmethod
    def _write_paired_csv(
        path: Path,
        *,
        motor_ids: tuple[int, int],
        joint_ids: tuple[int, int],
        amplitude: float,
    ) -> dict[int, float]:
        headers = ["timestamp"]
        for motor_id in motor_ids:
            headers.extend(
                [
                    f"motor_{motor_id}_target_raw",
                    f"motor_{motor_id}_pos_raw",
                    f"motor_{motor_id}_vel_raw",
                ]
            )
        formal_timestamps = [0.0, 0.004, 0.011]
        policy_deltas = [0.0, amplitude, -amplitude]
        raw_position_bias = {
            motor_ids[0]: 0.03,
            motor_ids[1]: 0.04,
        }
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for timestamp in (0.0, 0.010):
                row: dict[str, float] = {"timestamp": timestamp}
                for motor_id, joint_id in zip(motor_ids, joint_ids, strict=True):
                    row[f"motor_{motor_id}_target_raw"] = 99.0
                    row[f"motor_{motor_id}_pos_raw"] = 98.0
                    row[f"motor_{motor_id}_vel_raw"] = 97.0
                writer.writerow(row)
            for sample_index, (timestamp, policy_delta) in enumerate(
                zip(formal_timestamps, policy_deltas, strict=True)
            ):
                row = {"timestamp": timestamp}
                for motor_id, joint_id in zip(motor_ids, joint_ids, strict=True):
                    raw_command = (
                        JULY30_RAW_ENCODER_CENTER[joint_id]
                        + JULY30_RAW_TO_POLICY_SIGN[joint_id] * policy_delta
                    )
                    row[f"motor_{motor_id}_target_raw"] = raw_command
                    row[f"motor_{motor_id}_pos_raw"] = raw_command + raw_position_bias[motor_id]
                    row[f"motor_{motor_id}_vel_raw"] = float(sample_index * 2)
                writer.writerow(row)
        return raw_position_bias

    def _write_complete_fixture(self, root: Path) -> dict[int, float]:
        biases: dict[int, float] = {}
        files = (
            ("motor14_20260730_163922_kp2_kd0p1.csv", (1, 4), (0, 3), 0.20),
            ("motor25_20260730_163219_kp8_kd0p8.csv", (2, 5), (1, 4), 0.25),
            ("motor14_20260730_163436_kp4_kd0p2.csv", (1, 4), (0, 3), 0.20),
            ("motor25_20260730_163050_kp4_kd0p2.csv", (2, 5), (1, 4), 0.25),
        )
        for filename, motor_ids, joint_ids, amplitude in files:
            biases.update(
                self._write_paired_csv(
                    root / filename,
                    motor_ids=motor_ids,
                    joint_ids=joint_ids,
                    amplitude=amplitude,
                )
            )
        return biases

    def test_name_parser_recognizes_pair_and_pd(self) -> None:
        meta = parse_paired_csv_name(Path("motor25_20260730_163219_kp8_kd0p8.csv"))

        self.assertEqual(meta["pair"], "25")
        self.assertEqual(meta["active_joint_ids"], [1, 4])
        self.assertEqual(meta["kp"], 8.0)
        self.assertEqual(meta["kd"], 0.8)
        self.assertEqual(meta["expected_amplitude_rad"], 0.25)

    def test_formal_segment_starts_at_final_timestamp_rollback(self) -> None:
        data = {
            "timestamp": np.asarray([0.0, 0.1, 0.0, 0.1, 0.0, 0.005]),
            "signal": np.asarray([10.0, 11.0, 20.0, 21.0, 30.0, 31.0]),
        }

        time, segment, meta = slice_after_last_timestamp_reset(data)

        np.testing.assert_array_equal(time, [0.0, 0.005])
        np.testing.assert_array_equal(segment["signal"], [30.0, 31.0])
        self.assertEqual(meta["reset_row_indices_zero_based"], [2, 4])
        self.assertEqual(meta["selected_start_row_zero_based"], 4)

    def test_coordinate_contract_is_fail_closed_and_records_all_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "mujoco-qpos-at-policy-center"):
            build_coordinate_contract(None, self.MUJOCO_SIGN, self.RUNTIME_DEFAULT)
        with self.assertRaisesRegex(ValueError, "runtime-joint-default"):
            build_coordinate_contract(self.MUJOCO_CENTER, self.MUJOCO_SIGN, None)
        with self.assertRaisesRegex(ValueError, "exactly \\+1 or -1"):
            build_coordinate_contract(
                self.MUJOCO_CENTER,
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                self.RUNTIME_DEFAULT,
            )

        contract = build_coordinate_contract(
            self.MUJOCO_CENTER,
            self.MUJOCO_SIGN,
            self.RUNTIME_DEFAULT,
        )
        self.assertEqual(
            set(contract["coordinate_layers"]),
            {"raw_encoder", "policy_named", "runtime_topic", "mujoco_absolute_qpos"},
        )
        np.testing.assert_allclose(
            contract["runtime_topic_center_rad"],
            JULY30_POLICY_SWEEP_CENTER - self.RUNTIME_DEFAULT,
        )
        self.assertEqual(len(contract["sha256"]), 64)

    def test_imports_two_independent_pd_groups_with_both_sides_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            biases = self._write_complete_fixture(source_dir)

            index = import_july30_paired_directory(
                csv_dir=source_dir,
                out_dir=output_dir,
                case="july30_regression",
                mujoco_qpos_at_policy_center=self.MUJOCO_CENTER,
                policy_delta_to_mujoco_sign=self.MUJOCO_SIGN,
                runtime_joint_default=self.RUNTIME_DEFAULT,
            )

            self.assertEqual(index["sample_rate_hz"], 200.0)
            self.assertFalse(index["align_initial"])
            self.assertEqual(
                [item["name"] for item in index["groups"]],
                [spec["name"] for spec in JULY30_PD_GROUPS],
            )
            self.assertTrue((output_dir / "paired_group_index.json").is_file())
            for group_entry in index["groups"]:
                group_dir = output_dir / group_entry["directory"]
                manifest_path = group_dir / "chirp_source_manifest.json"
                self.assertEqual(
                    group_entry["manifest_sha256"],
                    sha256_file(manifest_path),
                )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertFalse(manifest["align_initial"])
                self.assertEqual(manifest["alignment_window_s"], 0.0)
                self.assertEqual(manifest["sample_rate_hz"], 200.0)
                self.assertEqual(
                    manifest["coordinate_contract"]["sha256"],
                    index["coordinate_contract"]["sha256"],
                )
                self.assertEqual(len(manifest["sources"]), 2)
                self.assertEqual(
                    [source["active_joint_ids"] for source in manifest["sources"]],
                    [[0, 3], [1, 4]],
                )
                for source in manifest["sources"]:
                    self.assertTrue(source["amplitude_validation"]["passed"])
                    self.assertFalse(source["initial_dc_alignment"]["enabled"])
                    self.assertEqual(
                        source["source_csv_sha256"],
                        sha256_file(Path(source["source_csv"])),
                    )
                    pt_path = group_dir / source["pt"]
                    self.assertEqual(source["pt_sha256"], sha256_file(pt_path))
                    payload = load_chirp_data(pt_path)
                    np.testing.assert_array_equal(payload["time"], [0.0, 0.005, 0.01])
                    active_joint_ids = source["active_joint_ids"]
                    # ZOH keeps the +amplitude command from t=0.004 at t=0.005.
                    expected_amplitude = source["amplitude_validation"]["expected_amplitude_rad"]
                    np.testing.assert_allclose(
                        payload["des_dof_pos"][1, active_joint_ids],
                        self.MUJOCO_CENTER[active_joint_ids] + expected_amplitude,
                        atol=1.0e-6,
                    )
                    # The two independent measured biases survive; no initial
                    # 0.5-second DC alignment is allowed in paired mode.
                    expected_initial_positions = []
                    for motor_id, joint_id in zip(
                        source["motor_ids"],
                        active_joint_ids,
                        strict=True,
                    ):
                        expected_initial_positions.append(
                            self.MUJOCO_CENTER[joint_id]
                            + JULY30_RAW_TO_POLICY_SIGN[joint_id] * biases[motor_id]
                        )
                    np.testing.assert_allclose(
                        payload["dof_pos"][0, active_joint_ids],
                        expected_initial_positions,
                        atol=1.0e-6,
                    )


if __name__ == "__main__":
    unittest.main()
