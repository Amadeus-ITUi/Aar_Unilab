#!/usr/bin/env python3

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

HARDWARE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "hardware"
sys.path.insert(0, str(HARDWARE_DIR))

from rgb_led import RGBLed  # noqa: E402


class FakeRGBDevice:
    def __init__(self, **_kwargs: object) -> None:
        self.history: list[tuple[float, float, float]] = []
        self.closed = False
        self.fail_writes = False

    @property
    def color(self) -> tuple[float, float, float]:
        return self.history[-1] if self.history else (0.0, 0.0, 0.0)

    @color.setter
    def color(self, value: tuple[float, float, float]) -> None:
        if self.fail_writes:
            raise OSError("fake GPIO write failure")
        self.history.append(tuple(value))

    def close(self) -> None:
        self.closed = True


class FakePinFactory:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class RGBLedTest(unittest.TestCase):
    def make_led(self) -> tuple[RGBLed, FakeRGBDevice]:
        device = FakeRGBDevice()
        return RGBLed(device_factory=lambda **_kwargs: device), device

    def test_solid_scales_conventional_rgb(self) -> None:
        led, device = self.make_led()
        led.solid((255, 128, 0), brightness=0.5)
        self.assertAlmostEqual(device.color[0], 0.5)
        self.assertAlmostEqual(device.color[1], 128 * 0.5 / 255)
        self.assertEqual(device.color[2], 0.0)
        led.close()
        self.assertTrue(device.closed)

    def test_close_releases_owned_pin_factory(self) -> None:
        device = FakeRGBDevice()
        factory = FakePinFactory()
        led = RGBLed(
            device_factory=lambda **_kwargs: device,
            pin_factory_factory=lambda: factory,
        )

        led.close()

        self.assertTrue(device.closed)
        self.assertTrue(factory.closed)

    def test_new_animation_cancels_previous_worker(self) -> None:
        led, device = self.make_led()
        led.breathe((255, 255, 255), period=0.08, frame_interval=0.005)
        time.sleep(0.03)
        led.blink((255, 0, 0), interval=0.01, count=1)
        time.sleep(0.04)
        self.assertEqual(device.color, (0.0, 0.0, 0.0))
        self.assertTrue(led.healthy)
        led.close()

    def test_fault_pattern_starts_and_stops(self) -> None:
        led, _device = self.make_led()
        led.fault_pattern((0, 0, 255), 3)
        time.sleep(0.02)
        led.stop_animation()
        self.assertTrue(led.healthy)
        led.close()

    def test_fault_pattern_uses_color_header_and_exact_detail_count(self) -> None:
        led, _device = self.make_led()
        captured: list[object] = []
        led.pattern = lambda steps, repeat=True: captured.extend(  # type: ignore[method-assign]
            (list(steps), repeat)
        )
        led.fault_pattern((0, 0, 255), 3, brightness=0.5)
        steps = captured[0]
        self.assertEqual(steps[0], ((255, 0, 0), 1.0, 0.5))
        self.assertEqual(steps[1], ((0, 0, 255), 1.0, 0.5))
        self.assertEqual(sum(step[0] == (255, 255, 255) for step in steps), 3)
        self.assertEqual(steps[-1], (None, 2.0, 0.0))
        self.assertTrue(captured[1])
        led.close()

    def test_invalid_fault_code_is_rejected(self) -> None:
        led, _device = self.make_led()
        with self.assertRaises(ValueError):
            led.fault_pattern((0, 0, 255), 0)
        led.close()

    def test_async_gpio_failure_is_reported(self) -> None:
        led, device = self.make_led()
        led.breathe((255, 255, 255), period=0.08, frame_interval=0.005)
        time.sleep(0.01)
        device.fail_writes = True
        time.sleep(0.03)
        self.assertFalse(led.healthy)
        with self.assertRaisesRegex(RuntimeError, "fake GPIO write failure"):
            led.raise_if_failed()
        device.fail_writes = False
        led.close()


if __name__ == "__main__":
    unittest.main()
