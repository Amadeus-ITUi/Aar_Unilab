#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CONTROL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "control"
sys.path.insert(0, str(CONTROL_DIR))

from we11_state import (  # noqa: E402
    ButtonTracker,
    StartupMode,
    We11State,
    can_transition,
    sample_rate_ok,
    toggle_startup_mode,
)


class ButtonTrackerTest(unittest.TestCase):
    def test_rising_edge_only_fires_once_until_release(self) -> None:
        tracker = ButtonTracker()
        tracker.update((1,), (0.0,) * 8, 0.0)
        tracker.update((1,), (0.0,) * 8, 0.1)
        self.assertTrue(tracker.consume(0))
        self.assertFalse(tracker.consume(0))
        tracker.update((0,), (0.0,) * 8, 0.2)
        tracker.update((1,), (0.0,) * 8, 0.3)
        self.assertTrue(tracker.consume(0))

    def test_exit_chord_requires_continuous_hold(self) -> None:
        tracker = ButtonTracker(exit_hold_seconds=0.25)
        axes = (0.0,) * 7 + (1.0,)
        tracker.update((1,), axes, 1.0)
        tracker.update((1,), axes, 1.2)
        self.assertFalse(tracker.exit_requested)
        tracker.update((1,), axes, 1.26)
        self.assertTrue(tracker.exit_requested)


class RateTest(unittest.TestCase):
    def test_strictly_above_100_hz(self) -> None:
        passed, rate, gap = sample_rate_ok([i / 185.0 for i in range(186)])
        self.assertTrue(passed)
        self.assertGreater(rate, 100.0)
        self.assertLess(gap, 0.2)

    def test_exactly_100_hz_is_rejected(self) -> None:
        passed, _, _ = sample_rate_ok([i / 100.0 for i in range(101)])
        self.assertFalse(passed)

    def test_large_gap_is_rejected(self) -> None:
        stamps = [i / 185.0 for i in range(80)] + [0.7 + i / 185.0 for i in range(120)]
        passed, _, gap = sample_rate_ok(stamps)
        self.assertFalse(passed)
        self.assertGreater(gap, 0.2)


class StateTransitionTest(unittest.TestCase):
    def test_complete_success_path_is_legal(self) -> None:
        path = [
            We11State.BOOT,
            We11State.CAN_CHECK,
            We11State.IMU_CHECK,
            We11State.DEVICE_CHECK,
            We11State.WING_POSITIONING,
            We11State.WAIT_STANDBY,
            We11State.STANDBY_ENTERING,
            We11State.STANDBY,
            We11State.POLICY,
            We11State.STANDBY,
            We11State.STOPPED,
        ]
        for current, target in zip(path, path[1:]):
            self.assertTrue(can_transition(current, target), (current, target))

    def test_direct_policy_start_path_is_legal(self) -> None:
        path = [
            We11State.WAIT_STANDBY,
            We11State.POLICY_ENTERING,
            We11State.POLICY,
        ]
        for current, target in zip(path, path[1:]):
            self.assertTrue(can_transition(current, target), (current, target))

    def test_standby_and_policy_can_soft_disarm_back_to_wait(self) -> None:
        self.assertTrue(can_transition(We11State.STANDBY, We11State.WAIT_STANDBY))
        self.assertTrue(can_transition(We11State.POLICY, We11State.WAIT_STANDBY))
        self.assertTrue(
            can_transition(We11State.WAIT_STANDBY, We11State.STANDBY_ENTERING)
        )

    def test_skipping_automatic_positioning_or_y_gate_is_illegal(self) -> None:
        self.assertFalse(can_transition(We11State.DEVICE_CHECK, We11State.WAIT_STANDBY))
        self.assertFalse(can_transition(We11State.WING_POSITIONING, We11State.STANDBY))
        self.assertFalse(can_transition(We11State.WAIT_STANDBY, We11State.POLICY))

    def test_fault_is_latched(self) -> None:
        for state in We11State:
            if state not in {We11State.FAULT, We11State.STOPPED}:
                self.assertTrue(can_transition(state, We11State.FAULT))
        self.assertFalse(can_transition(We11State.FAULT, We11State.BOOT))
        self.assertFalse(can_transition(We11State.FAULT, We11State.STOPPED))


class StartupModeTest(unittest.TestCase):
    def test_x_toggle_is_reversible(self) -> None:
        mode = toggle_startup_mode(StartupMode.STANDBY)
        self.assertEqual(mode, StartupMode.DIRECT_POLICY)
        self.assertEqual(toggle_startup_mode(mode), StartupMode.STANDBY)


if __name__ == "__main__":
    unittest.main()
