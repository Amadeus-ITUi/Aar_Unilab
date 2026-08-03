#!/usr/bin/env python3
"""Frequency-local response estimates for monotonic sine chirps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid

DEFAULT_WELCH_SEGMENT_DURATION_S = 10.24
DEFAULT_COMMAND_PSD_THRESHOLD_DB = -20.0


def delayed_trace(time: np.ndarray, values: np.ndarray, delay_ms: float) -> np.ndarray:
    """Delay a sampled scalar trace by ``delay_ms`` with endpoint holding."""

    sample_time = np.asarray(time, dtype=np.float64)
    trace = np.asarray(values, dtype=np.float64)
    if sample_time.ndim != 1 or trace.shape != sample_time.shape:
        raise ValueError("time and values must be matching one-dimensional arrays")
    if sample_time.size < 2 or np.any(np.diff(sample_time) <= 0.0):
        raise ValueError("time must contain at least two strictly increasing samples")
    delay_s = float(delay_ms) * 1.0e-3
    if not np.isfinite(delay_s) or delay_s < 0.0:
        raise ValueError("delay_ms must be finite and non-negative")
    return np.interp(
        sample_time - delay_s,
        sample_time,
        trace,
        left=float(trace[0]),
        right=float(trace[-1]),
    )


@dataclass(frozen=True)
class LinearChirpTimeWindow:
    mask: np.ndarray
    source_frequency_range_hz: tuple[float, float]
    requested_frequency_range_hz: tuple[float, float]
    chirp_duration_s: float
    theoretical_time_range_s: tuple[float, float]
    sampled_time_range_s: tuple[float, float]
    sampled_frequency_range_hz: tuple[float, float]
    fit_window_margins_s: tuple[float, float]

    @property
    def sample_count(self) -> int:
        """Number of samples selected by this chirp window."""

        return int(np.count_nonzero(self.mask))

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "strict_linear_chirp_time_window",
            "source_frequency_range_hz": list(self.source_frequency_range_hz),
            "requested_frequency_range_hz": list(self.requested_frequency_range_hz),
            "chirp_duration_s": self.chirp_duration_s,
            "theoretical_time_range_s": list(self.theoretical_time_range_s),
            "sampled_time_range_s": list(self.sampled_time_range_s),
            "sampled_frequency_range_hz": list(self.sampled_frequency_range_hz),
            "fit_window_margins_s": list(self.fit_window_margins_s),
            "sample_count": self.sample_count,
        }


def linear_chirp_time_window(
    time: np.ndarray,
    *,
    chirp_start_hz: float,
    chirp_end_hz: float,
    chirp_duration_s: float,
    fmin: float,
    fmax: float,
    fit_start: float = 0.0,
    fit_end_margin: float = 0.0,
) -> LinearChirpTimeWindow:
    """Select samples whose linear-chirp excitation lies in ``[fmin, fmax]``."""

    t = np.asarray(time, dtype=np.float64).reshape(-1)
    if t.size < 2 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        raise ValueError("time must contain at least two finite strictly increasing samples")

    values = np.asarray(
        [
            chirp_start_hz,
            chirp_end_hz,
            chirp_duration_s,
            fmin,
            fmax,
            fit_start,
            fit_end_margin,
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(values)):
        raise ValueError("linear chirp window parameters must be finite")
    if chirp_start_hz <= 0.0 or chirp_end_hz <= 0.0 or chirp_start_hz == chirp_end_hz:
        raise ValueError("chirp start/end frequencies must be positive and different")
    if chirp_duration_s <= 0.0:
        raise ValueError("chirp duration must be positive")
    if fmin <= 0.0 or fmax <= fmin:
        raise ValueError("requested frequency range must be positive and increasing")
    if fit_start < 0.0 or fit_end_margin < 0.0:
        raise ValueError("fit-window margins must be non-negative")

    source_low = min(float(chirp_start_hz), float(chirp_end_hz))
    source_high = max(float(chirp_start_hz), float(chirp_end_hz))
    tolerance = 1.0e-12 * max(1.0, source_high)
    if fmin < source_low - tolerance or fmax > source_high + tolerance:
        raise ValueError(
            f"requested [{fmin:g}, {fmax:g}] Hz window lies outside the "
            f"[{source_low:g}, {source_high:g}] Hz source chirp"
        )

    def frequency_time(frequency_hz: float) -> float:
        fraction = (frequency_hz - chirp_start_hz) / (chirp_end_hz - chirp_start_hz)
        return float(t[0] + fraction * chirp_duration_s)

    band_times = sorted((frequency_time(fmin), frequency_time(fmax)))
    theoretical_start, theoretical_end = band_times
    sample_tolerance = 0.5 * float(np.median(np.diff(t))) + 1.0e-12
    if theoretical_start < t[0] - sample_tolerance or theoretical_end > t[-1] + sample_tolerance:
        raise ValueError(
            f"time samples [{t[0]:g}, {t[-1]:g}] s do not cover the complete "
            f"[{fmin:g}, {fmax:g}] Hz chirp window "
            f"[{theoretical_start:g}, {theoretical_end:g}] s"
        )

    effective_start = max(theoretical_start, float(t[0] + fit_start))
    effective_end = min(theoretical_end, float(t[-1] - fit_end_margin))
    mask = (t >= effective_start) & (t <= effective_end)
    if not np.any(mask):
        raise ValueError("linear chirp fit window contains no samples after applying margins")

    selected_time = t[mask]
    selected_fraction = (selected_time - t[0]) / chirp_duration_s
    selected_frequency = chirp_start_hz + (chirp_end_hz - chirp_start_hz) * selected_fraction
    return LinearChirpTimeWindow(
        mask=mask,
        source_frequency_range_hz=(float(chirp_start_hz), float(chirp_end_hz)),
        requested_frequency_range_hz=(float(fmin), float(fmax)),
        chirp_duration_s=float(chirp_duration_s),
        theoretical_time_range_s=(theoretical_start, theoretical_end),
        sampled_time_range_s=(float(selected_time[0]), float(selected_time[-1])),
        sampled_frequency_range_hz=(
            float(min(selected_frequency[0], selected_frequency[-1])),
            float(max(selected_frequency[0], selected_frequency[-1])),
        ),
        fit_window_margins_s=(float(fit_start), float(fit_end_margin)),
    )


@dataclass(frozen=True)
class ChirpFrequencyResponse:
    frequency_hz: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray
    input_amplitude: np.ndarray
    response_amplitude: np.ndarray
    input_fit_r2: np.ndarray
    response_fit_r2: np.ndarray
    window_cycles: np.ndarray
    relative_bandwidth: np.ndarray
    valid: np.ndarray

    @property
    def valid_frequency_range(self) -> tuple[float, float] | None:
        frequencies = self.frequency_hz[self.valid]
        if frequencies.size == 0:
            return None
        return float(frequencies[0]), float(frequencies[-1])


def coerce_shared_delay_steps(value: Any, name: str = "delay_steps") -> int:
    """Validate one controller-tick delay shared by a complete command frame."""

    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be one scalar shared by all joints, got shape {array.shape}")
    try:
        scalar = float(array.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative integer, got {value!r}") from exc
    rounded = round(scalar)
    if (
        not np.isfinite(scalar)
        or scalar < 0.0
        or not np.isclose(scalar, rounded, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(f"{name} must be a finite non-negative integer, got {value!r}")
    return int(rounded)


class SharedCommandDelayBuffer:
    """Delay an entire multi-joint command frame with one shared FIFO.

    The first frame fills the FIFO, matching ``command[max(0, tick-delay)]``.
    A reset removes all history so separate sweep sources cannot leak commands
    into one another.
    """

    def __init__(self, delay_steps: Any):
        self.delay_steps = coerce_shared_delay_steps(delay_steps)
        self._history: deque[np.ndarray] = deque(maxlen=self.delay_steps + 1)
        self._frame_shape: tuple[int, ...] | None = None

    def reset(self) -> None:
        self._history.clear()
        self._frame_shape = None

    def push(self, command_frame: Any) -> np.ndarray:
        frame = np.asarray(command_frame, dtype=np.float64)
        if frame.ndim == 0:
            raise ValueError("command_frame must contain a complete multi-joint frame")
        if self._frame_shape is None:
            self._frame_shape = frame.shape
            self._history.extend(frame.copy() for _ in range(self.delay_steps + 1))
        else:
            if frame.shape != self._frame_shape:
                raise ValueError(
                    f"command_frame shape changed after FIFO initialization: "
                    f"{frame.shape} != {self._frame_shape}"
                )
            self._history.append(frame.copy())
        return self._history[0].copy()


def delayed_command_index(step: int, delay_steps: Any) -> int:
    """Return the source index for a controller-input command FIFO.

    The FIFO is initialized with the first available command.  Plant state is
    always current; only desired position/velocity is delayed.
    """

    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if isinstance(delay_steps, (int, np.integer)):
        delay = int(delay_steps)
        if delay < 0:
            raise ValueError(f"delay_steps must be non-negative, got {delay_steps}")
    else:
        delay = coerce_shared_delay_steps(delay_steps)
    return max(0, step - delay)


def transfer_bode(
    command: np.ndarray,
    response: np.ndarray,
    time: np.ndarray,
    *,
    maximum_segment_samples: int | None = None,
    segment_duration_s: float = DEFAULT_WELCH_SEGMENT_DURATION_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate command-to-response Bode magnitude/phase with Welch/CSD.

    This matches the Bode-magnitude score used by the existing MuJoCo PD
    sweeps.  It is intentionally separate from the chirp-local estimator below:
    the Welch estimate is smooth enough for parameter optimization, while the
    chirp-local estimate remains useful for the final phase/quality report when
    a reliable instantaneous-frequency trace is available.  The default Welch
    segment spans 10.24 seconds at every sample rate.  Passing
    ``maximum_segment_samples`` explicitly retains the legacy sample-count
    override.
    """
    from scipy import signal

    n = min(np.asarray(command).size, np.asarray(response).size, np.asarray(time).size)
    if n < 16:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    u = np.asarray(command, dtype=np.float64).reshape(-1)[:n]
    y = np.asarray(response, dtype=np.float64).reshape(-1)[:n]
    t = np.asarray(time, dtype=np.float64).reshape(-1)[:n]
    dt = float(np.median(np.diff(t)))
    if dt <= 0.0 or np.any(~np.isfinite(u)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(t)):
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    u = signal.detrend(u, type="constant")
    y = signal.detrend(y, type="constant")
    fs = 1.0 / dt
    if maximum_segment_samples is None:
        if not np.isfinite(segment_duration_s) or segment_duration_s <= 0.0:
            raise ValueError("segment_duration_s must be finite and positive")
        segment_samples = int(round(float(segment_duration_s) * fs))
    else:
        segment_samples = int(maximum_segment_samples)
    nperseg = min(segment_samples, n)
    if nperseg < 16 or float(np.max(np.abs(u))) < 1.0e-9:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    noverlap = nperseg // 2
    # SciPy defines CSD(x, y) as conj(X) * Y.  Therefore CSD(command,
    # response) / PSD(command) is the conventional output/input transfer H.
    frequency, command_response_csd = signal.csd(
        u, y, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
    )
    _, command_psd = signal.welch(u, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    transfer = command_response_csd / np.maximum(command_psd, 1.0e-18)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(transfer), 1.0e-12))
    phase_deg = np.rad2deg(np.angle(transfer))
    return (
        frequency.astype(np.float64),
        magnitude_db.astype(np.float64),
        phase_deg.astype(np.float64),
    )


def transfer_magnitude_db(
    command: np.ndarray,
    response: np.ndarray,
    time: np.ndarray,
    *,
    maximum_segment_samples: int | None = None,
    segment_duration_s: float = DEFAULT_WELCH_SEGMENT_DURATION_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate command-to-response transfer magnitude with Welch/CSD."""
    frequency, magnitude_db, _ = transfer_bode(
        command,
        response,
        time,
        maximum_segment_samples=maximum_segment_samples,
        segment_duration_s=segment_duration_s,
    )
    return frequency, magnitude_db


def command_psd_relative_db(
    command: np.ndarray,
    time: np.ndarray,
    *,
    maximum_segment_samples: int | None = None,
    segment_duration_s: float = DEFAULT_WELCH_SEGMENT_DURATION_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Welch command PSD in dB relative to its spectral peak.

    This uses the same detrending, Hann window, segment duration, and overlap
    as :func:`transfer_bode`, so its frequency grid can define a fixed
    reference-side excitation mask for Bode scoring.
    """
    from scipy import signal

    n = min(np.asarray(command).size, np.asarray(time).size)
    if n < 16:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    u = np.asarray(command, dtype=np.float64).reshape(-1)[:n]
    t = np.asarray(time, dtype=np.float64).reshape(-1)[:n]
    dt = float(np.median(np.diff(t)))
    if dt <= 0.0 or np.any(~np.isfinite(u)) or np.any(~np.isfinite(t)):
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    u = signal.detrend(u, type="constant")
    fs = 1.0 / dt
    if maximum_segment_samples is None:
        if not np.isfinite(segment_duration_s) or segment_duration_s <= 0.0:
            raise ValueError("segment_duration_s must be finite and positive")
        segment_samples = int(round(float(segment_duration_s) * fs))
    else:
        segment_samples = int(maximum_segment_samples)
    nperseg = min(segment_samples, n)
    if nperseg < 16 or float(np.max(np.abs(u))) < 1.0e-9:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    frequency, command_psd = signal.welch(
        u,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    peak = float(np.max(command_psd))
    if not np.isfinite(peak) or peak <= 0.0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    relative_db = 10.0 * np.log10(np.maximum(command_psd, np.finfo(np.float64).tiny) / peak)
    return frequency.astype(np.float64), relative_db.astype(np.float64)


def reference_excitation_mask(
    command: np.ndarray,
    time: np.ndarray,
    reference_frequency: np.ndarray,
    fmin: float,
    fmax: float,
    *,
    command_psd_threshold_db: float | None = None,
    maximum_segment_samples: int | None = None,
    segment_duration_s: float = DEFAULT_WELCH_SEGMENT_DURATION_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a candidate-independent Bode validity mask.

    Frequencies must first lie in the requested closed interval.  When
    ``command_psd_threshold_db`` is provided, they must also have reference
    command power at least that many dB relative to the command PSD peak.
    Only the measured command determines this mask; simulated candidates can
    never add or remove scored bins.
    """
    frequency = np.asarray(reference_frequency, dtype=np.float64).reshape(-1)
    lower = float(fmin)
    upper = float(fmax)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
        raise ValueError("Bode frequency range must be finite, positive, and increasing")
    threshold = None if command_psd_threshold_db is None else float(command_psd_threshold_db)
    if threshold is not None and (not np.isfinite(threshold) or threshold > 0.0):
        raise ValueError("command PSD threshold must be finite and no greater than 0 dB")

    band_mask = (frequency >= lower) & (frequency <= upper)
    psd_frequency, relative_psd_db = command_psd_relative_db(
        command,
        time,
        maximum_segment_samples=maximum_segment_samples,
        segment_duration_s=segment_duration_s,
    )
    interpolated_relative_db = np.full(frequency.shape, -np.inf, dtype=np.float64)
    if psd_frequency.size:
        interpolated_relative_db = np.interp(
            frequency,
            psd_frequency,
            relative_psd_db,
            left=-np.inf,
            right=-np.inf,
        )
    if threshold is None:
        return band_mask, interpolated_relative_db
    return band_mask & (interpolated_relative_db >= threshold), interpolated_relative_db


def reference_excitation_summary(
    reference_frequency: np.ndarray,
    frequency_mask: np.ndarray,
    command_psd_relative_db_values: np.ndarray,
    fmin: float,
    fmax: float,
    command_psd_threshold_db: float | None,
) -> dict[str, Any]:
    """Return JSON-ready provenance for a fixed Bode excitation mask."""
    frequency = np.asarray(reference_frequency, dtype=np.float64).reshape(-1)
    valid = np.asarray(frequency_mask, dtype=bool).reshape(-1)
    relative_db = np.asarray(command_psd_relative_db_values, dtype=np.float64).reshape(-1)
    if valid.shape != frequency.shape or relative_db.shape != frequency.shape:
        raise ValueError("reference frequency, mask, and command PSD arrays must have equal shape")
    band = (frequency >= float(fmin)) & (frequency <= float(fmax))
    valid_frequencies = frequency[valid]
    valid_relative_db = relative_db[valid]
    finite_relative_db = valid_relative_db[np.isfinite(valid_relative_db)]
    return {
        "requested_frequency_range_hz": [float(fmin), float(fmax)],
        "command_psd_threshold_db_relative_to_peak": (
            None if command_psd_threshold_db is None else float(command_psd_threshold_db)
        ),
        "band_frequency_count": int(np.count_nonzero(band)),
        "valid_frequency_count": int(np.count_nonzero(valid)),
        "rejected_low_excitation_count": int(np.count_nonzero(band & ~valid)),
        "valid_frequency_range_hz": (
            None
            if valid_frequencies.size == 0
            else [float(valid_frequencies[0]), float(valid_frequencies[-1])]
        ),
        "valid_command_psd_relative_db_range": (
            None
            if finite_relative_db.size == 0
            else [float(np.min(finite_relative_db)), float(np.max(finite_relative_db))]
        ),
    }


def bode_magnitude_mse(
    command: np.ndarray,
    simulated: np.ndarray,
    truth: np.ndarray,
    time: np.ndarray,
    fmin: float,
    fmax: float,
    *,
    command_psd_threshold_db: float | None = None,
) -> float:
    """Return mean squared real/simulated Bode-magnitude error in dB squared."""
    real_frequency, real_magnitude = transfer_magnitude_db(command, truth, time)
    sim_frequency, sim_magnitude = transfer_magnitude_db(command, simulated, time)
    if real_frequency.size == 0 or sim_frequency.size == 0:
        return float("nan")
    mask, _ = reference_excitation_mask(
        command,
        time,
        real_frequency,
        fmin,
        fmax,
        command_psd_threshold_db=command_psd_threshold_db,
    )
    if not np.any(mask):
        return float("nan")
    interpolated_sim = np.interp(real_frequency[mask], sim_frequency, sim_magnitude)
    difference = interpolated_sim - real_magnitude[mask]
    return float(np.mean(np.square(difference)))


def bode_error_components(
    command: np.ndarray,
    simulated: np.ndarray,
    truth: np.ndarray,
    time: np.ndarray,
    fmin: float,
    fmax: float,
    *,
    command_psd_threshold_db: float | None = None,
) -> tuple[float, float]:
    """Return magnitude and circular phase MSE as ``(dB^2, deg^2)``."""
    real_frequency, real_magnitude, real_phase = transfer_bode(command, truth, time)
    sim_frequency, sim_magnitude, sim_phase = transfer_bode(command, simulated, time)
    if real_frequency.size == 0 or sim_frequency.size == 0:
        return float("nan"), float("nan")
    mask, _ = reference_excitation_mask(
        command,
        time,
        real_frequency,
        fmin,
        fmax,
        command_psd_threshold_db=command_psd_threshold_db,
    )
    if not np.any(mask):
        return float("nan"), float("nan")
    frequencies = real_frequency[mask]
    magnitude_error = np.interp(frequencies, sim_frequency, sim_magnitude) - real_magnitude[mask]
    unwrapped_sim_phase = np.unwrap(np.deg2rad(sim_phase))
    interpolated_sim_phase = np.rad2deg(np.interp(frequencies, sim_frequency, unwrapped_sim_phase))
    phase_error = interpolated_sim_phase - real_phase[mask]
    phase_error = (phase_error + 180.0) % 360.0 - 180.0
    return float(np.mean(np.square(magnitude_error))), float(np.mean(np.square(phase_error)))


def reconstruct_chirp_phase(
    time: np.ndarray,
    command: np.ndarray,
    instantaneous_frequency_hz: np.ndarray,
) -> np.ndarray:
    """Integrate instantaneous frequency and align phase to the command."""

    t = np.asarray(time, dtype=np.float64).reshape(-1)
    u = np.asarray(command, dtype=np.float64).reshape(-1)
    frequency = np.asarray(instantaneous_frequency_hz, dtype=np.float64).reshape(-1)
    if t.size < 8 or u.size != t.size or frequency.size != t.size:
        raise ValueError(
            "time, command, and instantaneous frequency must have the same length >= 8"
        )
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(u)) or np.any(~np.isfinite(frequency)):
        raise ValueError("chirp inputs must be finite")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time must be strictly increasing")
    if np.any(frequency <= 0.0):
        raise ValueError("instantaneous frequency must stay positive")

    base_phase = cumulative_trapezoid(2.0 * np.pi * frequency, t, initial=0.0)
    centered_time = (t - np.mean(t)) / max(float(np.ptp(t)), np.finfo(np.float64).eps)
    design = np.column_stack(
        (np.sin(base_phase), np.cos(base_phase), np.ones_like(t), centered_time)
    )
    coefficients = np.linalg.lstsq(design, u, rcond=None)[0]
    phase_offset = np.arctan2(coefficients[1], coefficients[0])
    return base_phase + phase_offset


def _weighted_phasor_fit(
    phase: np.ndarray,
    time: np.ndarray,
    values: np.ndarray,
) -> tuple[complex, float, float]:
    count = phase.size
    if count < 8:
        return complex(np.nan, np.nan), float("nan"), float("nan")
    weights = np.hanning(count)
    if not np.any(weights > 0.0):
        weights = np.ones(count, dtype=np.float64)
    local_time = time - np.mean(time)
    local_time /= max(float(np.ptp(time)), np.finfo(np.float64).eps)
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones(count), local_time))
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_values = values * sqrt_weights
    coefficients = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
    prediction = design @ coefficients
    mean = float(np.average(values, weights=weights))
    residual_energy = float(np.sum(weights * np.square(values - prediction)))
    total_energy = float(np.sum(weights * np.square(values - mean)))
    r_squared = 1.0 - residual_energy / max(total_energy, np.finfo(np.float64).eps)
    phasor = complex(float(coefficients[1]), -float(coefficients[0]))
    return phasor, abs(phasor), r_squared


def estimate_chirp_frequency_response(
    time: np.ndarray,
    command: np.ndarray,
    response: np.ndarray,
    instantaneous_frequency_hz: np.ndarray,
    *,
    frequency_points: int = 80,
    window_cycles: float = 4.0,
    minimum_cycles: float = 2.0,
    minimum_side_cycles: float = 0.5,
    maximum_relative_bandwidth: float = 0.75,
    minimum_input_fit_r2: float = 0.95,
    minimum_response_fit_r2: float = 0.5,
) -> ChirpFrequencyResponse:
    """Estimate a Bode response by fitting the local chirp phasor.

    A single chirp is non-stationary, so treating it as many stationary Welch
    segments smears the sweep and creates convincing-looking endpoint values.
    This estimator follows the known instantaneous phase, fits a complex
    fundamental in a cycle-local window, and marks points invalid when the
    window is one-sided, too broad, or poorly explained by the fundamental.
    """

    t = np.asarray(time, dtype=np.float64).reshape(-1)
    u = np.asarray(command, dtype=np.float64).reshape(-1)
    y = np.asarray(response, dtype=np.float64).reshape(-1)
    frequency = np.asarray(instantaneous_frequency_hz, dtype=np.float64).reshape(-1)
    if y.size != t.size:
        raise ValueError("response must have the same length as time")
    phase = reconstruct_chirp_phase(t, u, frequency)
    if frequency_points < 3:
        raise ValueError("frequency_points must be at least 3")
    if window_cycles <= 0.0 or minimum_cycles <= 0.0 or minimum_side_cycles < 0.0:
        raise ValueError("cycle-window settings must be positive")

    increasing = frequency[-1] >= frequency[0]
    if not increasing:
        t = t[::-1]
        u = u[::-1]
        y = y[::-1]
        frequency = frequency[::-1]
        phase = -phase[::-1]
    if np.any(np.diff(frequency) < -1.0e-9):
        raise ValueError("instantaneous frequency must be monotonic")

    f_min = float(np.min(frequency))
    f_max = float(np.max(frequency))
    target_frequency = np.geomspace(f_min, f_max, frequency_points)
    magnitude_db = np.full(frequency_points, np.nan, dtype=np.float64)
    phase_deg = np.full_like(magnitude_db, np.nan)
    input_amplitude = np.full_like(magnitude_db, np.nan)
    response_amplitude = np.full_like(magnitude_db, np.nan)
    input_fit_r2 = np.full_like(magnitude_db, np.nan)
    response_fit_r2 = np.full_like(magnitude_db, np.nan)
    actual_cycles = np.zeros_like(magnitude_db)
    relative_bandwidth = np.full_like(magnitude_db, np.inf)
    enough_each_side = np.zeros(frequency_points, dtype=bool)

    half_phase_width = np.pi * window_cycles
    for index, target in enumerate(target_frequency):
        center_time = float(np.interp(target, frequency, t))
        center_phase = float(np.interp(center_time, t, phase))
        mask = np.abs(phase - center_phase) <= half_phase_width
        indices = np.flatnonzero(mask)
        if indices.size < 8:
            continue
        first = int(indices[0])
        last = int(indices[-1])
        window_phase = phase[first : last + 1]
        window_time = t[first : last + 1]
        window_command = u[first : last + 1]
        window_response = y[first : last + 1]

        before_cycles = max(0.0, (center_phase - float(window_phase[0])) / (2.0 * np.pi))
        after_cycles = max(0.0, (float(window_phase[-1]) - center_phase) / (2.0 * np.pi))
        actual_cycles[index] = before_cycles + after_cycles
        enough_each_side[index] = (
            before_cycles >= minimum_side_cycles and after_cycles >= minimum_side_cycles
        )
        low_frequency = float(np.min(frequency[first : last + 1]))
        high_frequency = float(np.max(frequency[first : last + 1]))
        relative_bandwidth[index] = (high_frequency - low_frequency) / max(float(target), 1.0e-12)

        input_phasor, input_amplitude[index], input_fit_r2[index] = _weighted_phasor_fit(
            window_phase, window_time, window_command
        )
        response_phasor, response_amplitude[index], response_fit_r2[index] = _weighted_phasor_fit(
            window_phase, window_time, window_response
        )
        if abs(input_phasor) <= 1.0e-12:
            continue
        transfer = response_phasor / input_phasor
        magnitude_db[index] = 20.0 * np.log10(max(abs(transfer), 1.0e-12))
        phase_deg[index] = np.angle(transfer)

    finite_phase = np.isfinite(phase_deg)
    if np.any(finite_phase):
        phase_deg[finite_phase] = np.rad2deg(np.unwrap(phase_deg[finite_phase]))
    amplitude_floor = max(float(np.nanmax(input_amplitude)) * 1.0e-3, 1.0e-12)
    valid = (
        np.isfinite(magnitude_db)
        & np.isfinite(phase_deg)
        & (actual_cycles >= minimum_cycles)
        & enough_each_side
        & (relative_bandwidth <= maximum_relative_bandwidth)
        & (input_amplitude >= amplitude_floor)
        & (input_fit_r2 >= minimum_input_fit_r2)
        & (response_fit_r2 >= minimum_response_fit_r2)
    )
    return ChirpFrequencyResponse(
        frequency_hz=target_frequency,
        magnitude_db=magnitude_db,
        phase_deg=phase_deg,
        input_amplitude=input_amplitude,
        response_amplitude=response_amplitude,
        input_fit_r2=input_fit_r2,
        response_fit_r2=response_fit_r2,
        window_cycles=actual_cycles,
        relative_bandwidth=relative_bandwidth,
        valid=valid,
    )
