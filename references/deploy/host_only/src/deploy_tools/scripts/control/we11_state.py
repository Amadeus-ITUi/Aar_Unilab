#!/usr/bin/env python3
"""Pure state/input helpers for the WE11 deployment supervisor."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from enum import Enum, auto


class We11State(Enum):
    BOOT = auto()
    CAN_CHECK = auto()
    IMU_CHECK = auto()
    DEVICE_CHECK = auto()
    WING_POSITIONING = auto()
    WAIT_STANDBY = auto()
    STANDBY_ENTERING = auto()
    POLICY_ENTERING = auto()
    STANDBY = auto()
    POLICY = auto()
    FAULT = auto()
    STOPPED = auto()


class StartupMode(Enum):
    STANDBY = auto()
    DIRECT_POLICY = auto()


def toggle_startup_mode(mode: StartupMode) -> StartupMode:
    return (
        StartupMode.DIRECT_POLICY
        if mode == StartupMode.STANDBY
        else StartupMode.STANDBY
    )


_FORWARD_TRANSITIONS = {
    We11State.BOOT: {We11State.CAN_CHECK},
    We11State.CAN_CHECK: {We11State.IMU_CHECK},
    We11State.IMU_CHECK: {We11State.DEVICE_CHECK},
    We11State.DEVICE_CHECK: {We11State.WING_POSITIONING},
    We11State.WING_POSITIONING: {We11State.WAIT_STANDBY},
    We11State.WAIT_STANDBY: {
        We11State.STANDBY_ENTERING,
        We11State.POLICY_ENTERING,
    },
    We11State.STANDBY_ENTERING: {We11State.STANDBY},
    We11State.POLICY_ENTERING: {We11State.POLICY},
    We11State.STANDBY: {We11State.POLICY, We11State.WAIT_STANDBY},
    We11State.POLICY: {We11State.STANDBY, We11State.WAIT_STANDBY},
    We11State.FAULT: set(),
    We11State.STOPPED: set(),
}


def can_transition(current: We11State, target: We11State) -> bool:
    """Return whether a supervisor state transition is legal."""
    if current == target:
        return current in {We11State.BOOT, We11State.STANDBY, We11State.POLICY}
    if target == We11State.FAULT:
        return current not in {We11State.FAULT, We11State.STOPPED}
    if target == We11State.STOPPED:
        return current not in {We11State.FAULT, We11State.STOPPED}
    return target in _FORWARD_TRANSITIONS[current]


class ButtonTracker:
    """Thread-safe rising edges plus a debounced D-pad-up+A stop chord."""

    def __init__(self, exit_hold_seconds: float = 0.25) -> None:
        self._lock = threading.Lock()
        self._last_buttons: tuple[int, ...] = ()
        self._edges: deque[int] = deque()
        self._combo_since: float | None = None
        self._exit_requested = False
        self._exit_hold_seconds = exit_hold_seconds

    def update(
        self,
        buttons: tuple[int, ...],
        axes: tuple[float, ...],
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            for index, value in enumerate(buttons):
                previous = self._last_buttons[index] if index < len(self._last_buttons) else 0
                if value == 1 and previous == 0:
                    self._edges.append(index)
            self._last_buttons = buttons
            dpad_up = len(axes) > 7 and math.isfinite(axes[7]) and axes[7] >= 0.5
            a_pressed = len(buttons) > 0 and buttons[0] == 1
            if dpad_up and a_pressed:
                if self._combo_since is None:
                    self._combo_since = now
                elif now - self._combo_since >= self._exit_hold_seconds:
                    self._exit_requested = True
            else:
                self._combo_since = None

    def consume(self, button_index: int) -> bool:
        with self._lock:
            for index, value in enumerate(self._edges):
                if value == button_index:
                    del self._edges[index]
                    return True
            return False

    def clear(self, button_index: int | None = None) -> None:
        with self._lock:
            if button_index is None:
                self._edges.clear()
            else:
                self._edges = deque(value for value in self._edges if value != button_index)

    @property
    def exit_requested(self) -> bool:
        with self._lock:
            return self._exit_requested


def sample_rate_ok(
    timestamps: list[float],
    *,
    minimum_hz: float = 100.0,
    maximum_gap: float = 0.2,
) -> tuple[bool, float, float]:
    """Return pass, observed rate, and largest adjacent timestamp gap."""
    if len(timestamps) < 2:
        return False, 0.0, math.inf
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0.0:
        return False, 0.0, math.inf
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    rate = (len(timestamps) - 1) / elapsed
    largest_gap = max(gaps, default=math.inf)
    return rate > minimum_hz and largest_gap <= maximum_gap, rate, largest_gap
