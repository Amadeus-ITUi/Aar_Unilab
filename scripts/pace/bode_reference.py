"""Reusable Bode reference and scoring helpers for PACE/PD fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from chirp_frequency_response import (
    linear_chirp_time_window,
    reference_excitation_mask,
    reference_excitation_summary,
    transfer_bode,
)


@dataclass(frozen=True)
class BodeReference:
    """Frozen real-system Bode reference and validity mask."""

    time: np.ndarray
    command: np.ndarray
    truth: np.ndarray
    fit_mask: np.ndarray
    frequency: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray
    frequency_mask: np.ndarray
    command_psd_relative_db: np.ndarray
    fmin: float
    fmax: float
    command_psd_threshold_db: float
    phase_weight: float
    fit_window_summary: dict[str, Any]

    @classmethod
    def create(
        cls,
        time: np.ndarray,
        command: np.ndarray,
        truth: np.ndarray,
        *,
        fit_start: float,
        fit_end_margin: float,
        fmin: float,
        fmax: float,
        phase_weight: float,
        command_psd_threshold_db: float,
        chirp_source_frequency_range_hz: tuple[float, float],
        chirp_duration_s: float,
    ) -> "BodeReference":
        time = np.asarray(time, dtype=np.float64)
        command = np.asarray(command, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        if time.ndim != 1 or command.shape != time.shape or truth.shape != time.shape:
            raise ValueError("time, command, and truth must be equal-length one-dimensional arrays")
        chirp_start_hz, chirp_end_hz = chirp_source_frequency_range_hz
        time_window = linear_chirp_time_window(
            time,
            chirp_start_hz=chirp_start_hz,
            chirp_end_hz=chirp_end_hz,
            chirp_duration_s=chirp_duration_s,
            fmin=fmin,
            fmax=fmax,
            fit_start=fit_start,
            fit_end_margin=fit_end_margin,
        )
        fit_mask = time_window.mask
        if np.count_nonzero(fit_mask) < 16:
            raise ValueError("Bode fit window contains fewer than 16 samples")
        frequency, magnitude, phase = transfer_bode(
            command[fit_mask], truth[fit_mask], time[fit_mask]
        )
        frequency_mask, command_psd_relative_db = reference_excitation_mask(
            command[fit_mask],
            time[fit_mask],
            frequency,
            fmin,
            fmax,
            command_psd_threshold_db=command_psd_threshold_db,
        )
        if not np.any(frequency_mask):
            raise ValueError(f"Bode reference has no valid excited bins in [{fmin:g}, {fmax:g}] Hz")
        return cls(
            time=time,
            command=command,
            truth=truth,
            fit_mask=fit_mask,
            frequency=frequency,
            magnitude_db=magnitude,
            phase_deg=phase,
            frequency_mask=frequency_mask,
            command_psd_relative_db=command_psd_relative_db,
            fmin=float(fmin),
            fmax=float(fmax),
            command_psd_threshold_db=float(command_psd_threshold_db),
            phase_weight=float(phase_weight),
            fit_window_summary=time_window.summary(),
        )

    def validity_summary(self) -> dict[str, Any]:
        summary = reference_excitation_summary(
            self.frequency,
            self.frequency_mask,
            self.command_psd_relative_db,
            self.fmin,
            self.fmax,
            self.command_psd_threshold_db,
        )
        summary["time_window"] = self.fit_window_summary
        return summary

    def metrics(self, response: np.ndarray) -> dict[str, float]:
        response = np.asarray(response, dtype=np.float64)
        if response.shape != self.time.shape:
            raise ValueError("response shape does not match the Bode reference")
        sim_frequency, sim_magnitude, sim_phase = transfer_bode(
            self.command[self.fit_mask],
            response[self.fit_mask],
            self.time[self.fit_mask],
        )
        if sim_frequency.size == 0:
            raise ValueError("simulated Bode response is empty")
        frequencies = self.frequency[self.frequency_mask]
        magnitude_error = (
            np.interp(frequencies, sim_frequency, sim_magnitude)
            - self.magnitude_db[self.frequency_mask]
        )
        unwrapped_sim_phase = np.unwrap(np.deg2rad(sim_phase))
        interpolated_sim_phase = np.rad2deg(
            np.interp(frequencies, sim_frequency, unwrapped_sim_phase)
        )
        phase_error = interpolated_sim_phase - self.phase_deg[self.frequency_mask]
        phase_error = (phase_error + 180.0) % 360.0 - 180.0
        magnitude_mse = float(np.mean(np.square(magnitude_error)))
        phase_mse = float(np.mean(np.square(phase_error)))
        error = response[self.fit_mask] - self.truth[self.fit_mask]
        scale = max(float(np.std(self.truth[self.fit_mask])), 1.0e-3)
        return {
            "bode_magnitude_mse_db2": magnitude_mse,
            "bode_magnitude_rmse_db": math.sqrt(magnitude_mse),
            "bode_phase_mse_deg2": phase_mse,
            "bode_phase_rmse_deg": math.sqrt(phase_mse),
            "bode_complex_score": magnitude_mse + self.phase_weight * phase_mse,
            "time_rmse": float(np.sqrt(np.mean(np.square(error)))),
            "time_normalized_mse": float(np.mean(np.square(error / scale))),
        }
