#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

CONTROL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "control"
sys.path.insert(0, str(CONTROL_DIR))

from we11_gamepad_launcher import (  # noqa: E402
    JS_EVENT,
    JS_EVENT_AXIS,
    JS_EVENT_BUTTON,
    JS_EVENT_INIT,
    IDLE_CONNECTED_COLOR,
    IDLE_DISCONNECTED_COLOR,
    RGB_HANDOFF_READY_FD_ENV,
    RGB_HANDOFF_RELEASE_FD_ENV,
    STARTING_COLOR,
    GamepadLifecycleLauncher,
    HoldChord,
    LinuxJoystick,
)


class FakeLed:
    def __init__(self, *, failed: bool = False) -> None:
        self.failed = failed
        self.closed = False
        self.breathe_calls: list[tuple[tuple[int, int, int], float, float]] = []

    def raise_if_failed(self) -> None:
        if self.failed:
            raise RuntimeError("fake RGB failure")

    def breathe(
        self, color: tuple[int, int, int], *, brightness: float, period: float
    ) -> None:
        self.breathe_calls.append((color, brightness, period))

    def close(self) -> None:
        self.closed = True


class HoldChordTest(unittest.TestCase):
    def make_active(self, chord: HoldChord) -> None:
        chord.update_axis(7, -32767)
        chord.update_button(chord.button_index, 1)

    def test_requires_continuous_hold(self) -> None:
        chord = HoldChord(button_index=2, hold_seconds=0.25)
        self.make_active(chord)
        self.assertFalse(chord.poll(1.0))
        self.assertFalse(chord.poll(1.24))
        self.assertTrue(chord.poll(1.25))

    def test_does_not_repeat_until_released(self) -> None:
        chord = HoldChord(button_index=0, hold_seconds=0.25)
        self.make_active(chord)
        chord.poll(1.0)
        self.assertTrue(chord.poll(1.25))
        self.assertFalse(chord.poll(2.0))
        chord.update_button(0, 0)
        self.assertFalse(chord.poll(2.1))
        chord.update_button(0, 1)
        self.assertFalse(chord.poll(3.0))
        self.assertTrue(chord.poll(3.25))

    def test_interrupted_hold_resets_timer(self) -> None:
        chord = HoldChord(button_index=2, hold_seconds=0.25)
        self.make_active(chord)
        chord.poll(1.0)
        chord.update_axis(7, 0)
        chord.poll(1.2)
        chord.update_axis(7, -32767)
        self.assertFalse(chord.poll(1.3))
        self.assertFalse(chord.poll(1.54))
        self.assertTrue(chord.poll(1.55))

    def test_a_and_x_are_distinct(self) -> None:
        start = HoldChord(button_index=2, hold_seconds=0.25)
        stop = HoldChord(button_index=0, hold_seconds=0.25)
        for chord in (start, stop):
            chord.update_axis(7, -32767)
            chord.update_button(0, 1)
            chord.update_button(2, 0)
        start.poll(1.0)
        stop.poll(1.0)
        self.assertFalse(start.poll(1.3))
        self.assertTrue(stop.poll(1.3))

    def test_linux_joy_events_feed_both_chords(self) -> None:
        start = HoldChord(button_index=2)
        stop = HoldChord(button_index=0)
        joystick = LinuxJoystick(Path("/not-opened"), (start, stop))
        joystick.buffer.extend(
            JS_EVENT.pack(1, -32767, JS_EVENT_AXIS | JS_EVENT_INIT, 7)
            + JS_EVENT.pack(2, 1, JS_EVENT_BUTTON, 2)
        )
        joystick._consume()
        self.assertTrue(start.active)
        self.assertFalse(stop.active)
        self.assertEqual(joystick.buffer, bytearray())

    def test_dpad_down_does_not_activate(self) -> None:
        chord = HoldChord(button_index=2)
        chord.update_axis(7, 32767)
        chord.update_button(2, 1)
        self.assertFalse(chord.active)


class IdleRgbTest(unittest.TestCase):
    def make_launcher(self, leds: list[FakeLed]) -> GamepadLifecycleLauncher:
        return GamepadLifecycleLauncher(
            joy_device=Path("/not-opened"),
            hold_seconds=0.25,
            led_factory=lambda: leds.append(FakeLed()) or leds[-1],
        )

    def test_idle_color_tracks_connection(self) -> None:
        leds: list[FakeLed] = []
        launcher = self.make_launcher(leds)
        launcher._open_idle_led()
        self.assertEqual(leds[0].breathe_calls[-1], (IDLE_DISCONNECTED_COLOR, 0.5, 2.0))

        launcher.joystick.fd = 123
        launcher._sync_idle_led()
        self.assertEqual(leds[0].breathe_calls[-1], (IDLE_CONNECTED_COLOR, 0.5, 2.0))
        launcher.joystick.fd = None

    def test_starting_breathe_remains_until_child_requests_handoff(self) -> None:
        leds: list[FakeLed] = []
        launcher = self.make_launcher(leds)
        launcher._open_idle_led()
        fake_child = object()
        with (
            patch("we11_gamepad_launcher.subprocess.Popen", return_value=fake_child) as popen,
            patch.object(launcher, "_wait_for_child_led_ready", return_value=True),
            patch("we11_gamepad_launcher.os.write", return_value=1),
        ):
            launcher._start_robot()
        self.assertEqual(leds[0].breathe_calls[-1], (STARTING_COLOR, 0.5, 2.0))
        self.assertTrue(leds[0].closed)
        self.assertIsNone(launcher.led)
        self.assertIs(launcher.child, fake_child)
        popen_environment = popen.call_args.kwargs["env"]
        self.assertIn(RGB_HANDOFF_READY_FD_ENV, popen_environment)
        self.assertIn(RGB_HANDOFF_RELEASE_FD_ENV, popen_environment)
        self.assertEqual(len(popen.call_args.kwargs["pass_fds"]), 2)

        launcher.child = None
        launcher._open_idle_led()
        self.assertEqual(len(leds), 2)
        self.assertFalse(leds[1].closed)

    def test_child_launch_failure_restores_idle_breathe(self) -> None:
        leds: list[FakeLed] = []
        launcher = self.make_launcher(leds)
        launcher._open_idle_led()
        with (
            patch(
                "we11_gamepad_launcher.subprocess.Popen",
                side_effect=OSError("fake launch failure"),
            ),
            self.assertRaisesRegex(OSError, "fake launch failure"),
        ):
            launcher._start_robot()
        self.assertIs(launcher.led, leds[0])
        self.assertFalse(leds[0].closed)
        self.assertEqual(
            leds[0].breathe_calls[-1], (IDLE_DISCONNECTED_COLOR, 0.5, 2.0)
        )

    def test_rgb_failure_is_fail_closed(self) -> None:
        launcher = GamepadLifecycleLauncher(
            joy_device=Path("/not-opened"),
            hold_seconds=0.25,
            led_factory=lambda: FakeLed(failed=True),
        )
        with self.assertRaisesRegex(RuntimeError, "fake RGB failure"):
            launcher._open_idle_led()


if __name__ == "__main__":
    unittest.main()
