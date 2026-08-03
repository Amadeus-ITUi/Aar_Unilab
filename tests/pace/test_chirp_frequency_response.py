from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PACE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pace"
sys.path.insert(0, str(PACE_DIR))

from chirp_frequency_response import (  # noqa: E402
    SharedCommandDelayBuffer,
    bode_error_components,
    coerce_shared_delay_steps,
    command_psd_relative_db,
    delayed_command_index,
    delayed_trace,
    estimate_chirp_frequency_response,
    linear_chirp_time_window,
    reference_excitation_mask,
    transfer_bode,
    transfer_magnitude_db,
)


class CommandDelayTests(unittest.TestCase):
    def test_delay_changes_command_index_only(self) -> None:
        self.assertEqual([delayed_command_index(step, 2) for step in range(6)], [0, 0, 0, 1, 2, 3])

    def test_fractional_command_delay_holds_first_sample(self) -> None:
        time = np.asarray([0.0, 0.002, 0.004, 0.006])
        command = np.asarray([0.0, 2.0, 4.0, 6.0])
        delayed = delayed_trace(time, command, 1.0)
        np.testing.assert_allclose(delayed, [0.0, 1.0, 3.0, 5.0])

    def test_shared_fifo_delays_complete_six_joint_frames(self) -> None:
        frames = np.arange(5 * 6 * 2, dtype=np.float64).reshape(5, 6, 2)
        for delay, indices in (
            (0, [0, 1, 2, 3, 4]),
            (1, [0, 0, 1, 2, 3]),
            (3, [0, 0, 0, 0, 1]),
        ):
            fifo = SharedCommandDelayBuffer(delay)
            delayed = np.stack([fifo.push(frame) for frame in frames])
            np.testing.assert_array_equal(delayed, frames[indices])

    def test_shared_fifo_reset_holds_the_new_first_frame(self) -> None:
        fifo = SharedCommandDelayBuffer(2)
        old_frame = np.full((6, 2), 7.0)
        new_frame = np.full((6, 2), -3.0)
        fifo.push(old_frame)
        fifo.reset()

        np.testing.assert_array_equal(fifo.push(new_frame), new_frame)

    def test_shared_delay_rejects_per_joint_or_fractional_values(self) -> None:
        self.assertEqual(coerce_shared_delay_steps(4.0), 4)
        with self.assertRaisesRegex(ValueError, "one scalar shared by all joints"):
            coerce_shared_delay_steps([4, 6, 0, 4, 6, 0])
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            coerce_shared_delay_steps(1.5)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            coerce_shared_delay_steps(-1)


class ChirpFrequencyResponseTests(unittest.TestCase):
    def test_linear_chirp_window_selects_only_requested_excitation_times(self) -> None:
        time = np.arange(0.0, 40.0 + 0.005, 0.005)

        window = linear_chirp_time_window(
            time,
            chirp_start_hz=0.1,
            chirp_end_hz=5.0,
            chirp_duration_s=40.0,
            fmin=0.5,
            fmax=4.5,
            fit_start=0.5,
            fit_end_margin=0.25,
        )

        self.assertAlmostEqual(window.theoretical_time_range_s[0], 3.26530612244898)
        self.assertAlmostEqual(window.theoretical_time_range_s[1], 35.91836734693877)
        self.assertEqual(window.sampled_time_range_s, (3.27, 35.915))
        self.assertGreaterEqual(window.sampled_frequency_range_hz[0], 0.5)
        self.assertLessEqual(window.sampled_frequency_range_hz[1], 4.5)
        self.assertFalse(np.any(window.mask[time < window.theoretical_time_range_s[0]]))
        self.assertFalse(np.any(window.mask[time > window.theoretical_time_range_s[1]]))
        self.assertEqual(window.summary()["mode"], "strict_linear_chirp_time_window")

    def test_linear_chirp_window_rejects_truncated_record_before_band_end(self) -> None:
        time = np.arange(0.0, 20.0, 0.005)

        with self.assertRaisesRegex(ValueError, "do not cover the complete"):
            linear_chirp_time_window(
                time,
                chirp_start_hz=0.1,
                chirp_end_hz=5.0,
                chirp_duration_s=40.0,
                fmin=0.5,
                fmax=4.5,
            )

    def test_default_welch_window_is_sample_rate_invariant(self) -> None:
        def sampled_response(sample_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            time = np.arange(0.0, 40.0, 1.0 / sample_hz)
            frequency_slope = (5.0 - 0.1) / 40.0
            phase = 2.0 * np.pi * (0.1 * time + 0.5 * frequency_slope * np.square(time))
            gain = 0.35 + 0.45 * time / 40.0
            phase_lag = 0.15 + 0.20 * time / 40.0
            command = np.sin(phase) + 0.03 * np.sin(2.0 * phase)
            response = gain * np.sin(phase - phase_lag) + 0.04 * np.sin(2.0 * phase + 0.2)
            return time, command, response

        time_400, command_400, response_400 = sampled_response(400.0)
        time_200, command_200, response_200 = sampled_response(200.0)
        frequency_400, magnitude_400, phase_400 = transfer_bode(command_400, response_400, time_400)
        frequency_200, magnitude_200, phase_200 = transfer_bode(command_200, response_200, time_200)
        band_400 = (frequency_400 >= 0.1) & (frequency_400 <= 5.0)
        band_200 = (frequency_200 >= 0.1) & (frequency_200 <= 5.0)

        np.testing.assert_allclose(frequency_400[band_400], frequency_200[band_200], atol=1.0e-12)
        np.testing.assert_allclose(magnitude_400[band_400], magnitude_200[band_200], atol=1.0e-8)
        phase_error = phase_200[band_200] - phase_400[band_400]
        phase_error = (phase_error + 180.0) % 360.0 - 180.0
        self.assertLess(float(np.sqrt(np.mean(np.square(phase_error)))), 1.0e-6)
        self.assertAlmostEqual(float(frequency_400[1]), 1.0 / 10.24, places=12)
        self.assertAlmostEqual(float(frequency_200[1]), 1.0 / 10.24, places=12)

    def test_explicit_sample_limit_retains_legacy_override(self) -> None:
        sample_hz = 200.0
        time = np.arange(0.0, 40.0, 1.0 / sample_hz)
        command = np.sin(2.0 * np.pi * time)

        frequency, _, _ = transfer_bode(
            command,
            command,
            time,
            maximum_segment_samples=4096,
        )
        magnitude_frequency, _ = transfer_magnitude_db(
            command,
            command,
            time,
            maximum_segment_samples=4096,
        )

        self.assertAlmostEqual(float(frequency[1]), sample_hz / 4096.0, places=12)
        np.testing.assert_array_equal(magnitude_frequency, frequency)

    def test_transfer_bode_uses_standard_negative_delay_phase(self) -> None:
        sample_hz = 1000.0
        time = np.arange(0.0, 20.0, 1.0 / sample_hz)
        test_frequency_hz = 2.0
        delay_s = 0.020
        command = np.sin(2.0 * np.pi * test_frequency_hz * time)
        response = np.sin(2.0 * np.pi * test_frequency_hz * (time - delay_s))

        frequency, _, phase_deg = transfer_bode(command, response, time)
        index = int(np.argmin(np.abs(frequency - test_frequency_hz)))
        expected_phase_deg = -360.0 * frequency[index] * delay_s

        self.assertLess(phase_deg[index], 0.0)
        self.assertAlmostEqual(float(phase_deg[index]), float(expected_phase_deg), delta=0.5)

    def test_command_delay_is_identified_by_phase_not_magnitude(self) -> None:
        sample_hz = 400.0
        time = np.arange(0.0, 40.0, 1.0 / sample_hz)
        frequency_slope = (5.0 - 0.1) / 40.0
        phase = 2.0 * np.pi * (0.1 * time + 0.5 * frequency_slope * np.square(time))
        command = np.sin(phase)
        delayed = np.interp(
            time - 0.020, time, command, left=float(command[0]), right=float(command[-1])
        )

        magnitude_mse, phase_mse = bode_error_components(command, delayed, command, time, 0.1, 5.0)

        self.assertLess(magnitude_mse, 0.01)
        self.assertGreater(phase_mse, 100.0)

    def test_fixed_reference_excitation_mask_excludes_weak_bins_inside_requested_band(self) -> None:
        sample_hz = 200.0
        time = np.arange(0.0, 40.0, 1.0 / sample_hz)
        strong_frequency_hz = 20.0 * sample_hz / 2048.0
        weak_frequency_hz = 41.0 * sample_hz / 2048.0
        strong = np.sin(2.0 * np.pi * strong_frequency_hz * time)
        weak = 1.0e-3 * np.sin(2.0 * np.pi * weak_frequency_hz * time)
        command = strong + weak
        frequency, _, _ = transfer_bode(command, command, time)

        valid, relative_db = reference_excitation_mask(
            command,
            time,
            frequency,
            0.5,
            4.5,
            command_psd_threshold_db=-20.0,
        )
        psd_frequency, direct_relative_db = command_psd_relative_db(command, time)
        np.testing.assert_allclose(frequency, psd_frequency, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(relative_db, direct_relative_db, rtol=0.0, atol=0.0)

        strong_index = int(np.argmin(np.abs(frequency - strong_frequency_hz)))
        weak_index = int(np.argmin(np.abs(frequency - weak_frequency_hz)))
        self.assertTrue(valid[strong_index])
        self.assertFalse(valid[weak_index])
        self.assertTrue(np.all(frequency[valid] >= 0.5))
        self.assertTrue(np.all(frequency[valid] <= 4.5))

    def test_low_excitation_bin_does_not_dominate_masked_bode_score(self) -> None:
        sample_hz = 200.0
        time = np.arange(0.0, 40.0, 1.0 / sample_hz)
        strong_frequency_hz = 20.0 * sample_hz / 2048.0
        weak_frequency_hz = 41.0 * sample_hz / 2048.0
        strong = np.sin(2.0 * np.pi * strong_frequency_hz * time)
        weak = 1.0e-3 * np.sin(2.0 * np.pi * weak_frequency_hz * time)
        command = strong + weak
        truth = strong + weak
        simulated = strong + 0.2 * np.sin(2.0 * np.pi * weak_frequency_hz * time)

        unmasked_magnitude_mse, _ = bode_error_components(command, simulated, truth, time, 0.5, 4.5)
        masked_magnitude_mse, _ = bode_error_components(
            command,
            simulated,
            truth,
            time,
            0.5,
            4.5,
            command_psd_threshold_db=-20.0,
        )

        self.assertGreater(unmasked_magnitude_mse, 10.0)
        self.assertLess(masked_magnitude_mse, unmasked_magnitude_mse * 1.0e-3)

    def test_known_gain_and_phase_are_recovered_and_endpoints_masked(self) -> None:
        sample_hz = 400.0
        time = np.arange(0.0, 40.0, 1.0 / sample_hz)
        log_rate = np.log(5.0 / 0.1) / 40.0
        frequency = 0.1 * np.exp(log_rate * time)
        phase = 2.0 * np.pi * 0.1 * np.expm1(log_rate * time) / log_rate
        command = np.sin(phase)
        response = 0.5 * np.sin(phase - np.deg2rad(30.0))

        result = estimate_chirp_frequency_response(time, command, response, frequency)

        self.assertGreater(np.count_nonzero(result.valid), 30)
        self.assertFalse(result.valid[0])
        self.assertFalse(result.valid[-1])
        self.assertAlmostEqual(
            float(np.median(result.magnitude_db[result.valid])), -6.0206, places=3
        )
        self.assertAlmostEqual(float(np.median(result.phase_deg[result.valid])), -30.0, places=2)


if __name__ == "__main__":
    unittest.main()
