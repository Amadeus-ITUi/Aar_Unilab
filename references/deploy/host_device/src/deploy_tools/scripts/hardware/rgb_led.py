#!/usr/bin/env python3
"""Reusable, single-owner driver for a four-pin PWM RGB LED."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from threading import Event, Lock, Thread, current_thread
from time import monotonic
from typing import Protocol, TypeAlias

RGB: TypeAlias = tuple[int, int, int]
PatternStep: TypeAlias = tuple[RGB | None, float, float]


class RGBDevice(Protocol):
    color: tuple[float, float, float]

    def close(self) -> None: ...


class PinFactory(Protocol):
    def close(self) -> None: ...


class RGBLed:
    """Drive an RGB LED using BCM pin numbers with one animation owner."""

    def __init__(
        self,
        red_pin: int = 22,
        green_pin: int = 17,
        blue_pin: int = 27,
        *,
        common_anode: bool = True,
        device_factory: Callable[..., RGBDevice] | None = None,
        pin_factory_factory: Callable[[], PinFactory] | None = None,
    ) -> None:
        if len({red_pin, green_pin, blue_pin}) != 3:
            raise ValueError("red_pin, green_pin and blue_pin must be different")
        if device_factory is None:
            try:
                from gpiozero import RGBLED
                from gpiozero.pins.lgpio import LGPIOFactory
            except ImportError as exc:
                raise RuntimeError(
                    "gpiozero is required: sudo apt install python3-gpiozero"
                ) from exc
            device_factory = RGBLED
            if pin_factory_factory is None:
                pin_factory_factory = LGPIOFactory

        # gpiozero's process-global LGPIOFactory keeps its gpiochip handle open
        # after RGBLED.close(). That lets the same process reuse the pins but
        # prevents the supervisor process from acquiring them during handoff.
        # Give each RGB owner a private factory and explicitly close it below.
        self._pin_factory = pin_factory_factory() if pin_factory_factory else None
        device_options: dict[str, object] = {
            "red": red_pin,
            "green": green_pin,
            "blue": blue_pin,
            "active_high": not common_anode,
            "initial_value": (0.0, 0.0, 0.0),
            "pwm": True,
        }
        if self._pin_factory is not None:
            device_options["pin_factory"] = self._pin_factory
        try:
            self._led = device_factory(**device_options)
        except BaseException:
            if self._pin_factory is not None:
                self._pin_factory.close()
            raise
        self._lock = Lock()
        self._animation_stop = Event()
        self._animation_thread: Thread | None = None
        self._closed = False
        self._last_error: BaseException | None = None

    @staticmethod
    def _validate_color(color: RGB) -> RGB:
        if len(color) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
            for value in color
        ):
            raise ValueError("color must be three integers in the range 0..255")
        return color

    @staticmethod
    def _validate_brightness(brightness: float) -> float:
        value = float(brightness)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("brightness must be finite and in the range 0.0..1.0")
        return value

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("RGBLed is closed")
        self.raise_if_failed()

    def _write(self, color: RGB | None, brightness: float = 1.0) -> None:
        with self._lock:
            if self._closed:
                return
            if color is None:
                value = (0.0, 0.0, 0.0)
            else:
                scale = brightness / 255.0
                value = tuple(channel * scale for channel in color)
            self._led.color = value

    def _start_animation(self, name: str, worker: Callable[[], None]) -> None:
        self._check_open()
        self.stop_animation()
        self._last_error = None
        self._animation_stop.clear()

        def guarded_worker() -> None:
            try:
                worker()
            except BaseException as exc:
                self._last_error = exc
                self._animation_stop.set()
                try:
                    self._write(None)
                except BaseException:
                    pass

        self._animation_thread = Thread(
            target=guarded_worker, name=f"rgb-led-{name}", daemon=True
        )
        self._animation_thread.start()

    @property
    def healthy(self) -> bool:
        return not self._closed and self._last_error is None

    def raise_if_failed(self) -> None:
        if self._last_error is not None:
            raise RuntimeError(
                f"RGB LED animation failed: {self._last_error}"
            ) from self._last_error

    def solid(self, color: RGB, brightness: float = 1.0) -> None:
        self._check_open()
        color = self._validate_color(color)
        level = self._validate_brightness(brightness)
        self.stop_animation()
        self._write(color, level)

    def blink(
        self,
        color: RGB,
        *,
        brightness: float = 1.0,
        interval: float = 0.5,
        count: int | None = None,
    ) -> None:
        color = self._validate_color(color)
        level = self._validate_brightness(brightness)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("interval must be finite and greater than 0")
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
        ):
            raise ValueError("count must be a positive integer or None")

        def worker() -> None:
            flashes = 0
            while not self._animation_stop.is_set() and (count is None or flashes < count):
                self._write(color, level)
                if self._animation_stop.wait(interval):
                    break
                self._write(None)
                flashes += 1
                if self._animation_stop.wait(interval):
                    break
            self._write(None)

        self._start_animation("blink", worker)

    def breathe(
        self,
        color: RGB,
        *,
        brightness: float = 0.5,
        period: float = 2.0,
        minimum: float = 0.04,
        frame_interval: float = 0.02,
    ) -> None:
        color = self._validate_color(color)
        maximum = self._validate_brightness(brightness)
        minimum = self._validate_brightness(minimum)
        if minimum > maximum:
            raise ValueError("minimum brightness must not exceed brightness")
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("period must be finite and greater than 0")
        if not math.isfinite(frame_interval) or frame_interval <= 0.0:
            raise ValueError("frame_interval must be finite and greater than 0")

        def worker() -> None:
            started = monotonic()
            while not self._animation_stop.is_set():
                phase = ((monotonic() - started) % period) / period
                wave = 0.5 - 0.5 * math.cos(phase * 2.0 * math.pi)
                self._write(color, minimum + (maximum - minimum) * wave)
                self._animation_stop.wait(frame_interval)
            self._write(None)

        self._start_animation("breathe", worker)

    def pattern(self, steps: Iterable[PatternStep], *, repeat: bool = True) -> None:
        validated: list[PatternStep] = []
        for color, duration, brightness in steps:
            checked_color = None if color is None else self._validate_color(color)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError("pattern durations must be finite and greater than 0")
            validated.append(
                (checked_color, float(duration), self._validate_brightness(brightness))
            )
        if not validated:
            raise ValueError("pattern must contain at least one step")

        def worker() -> None:
            while not self._animation_stop.is_set():
                for color, duration, brightness in validated:
                    self._write(color, brightness)
                    if self._animation_stop.wait(duration):
                        self._write(None)
                        return
                if not repeat:
                    break
            self._write(None)

        self._start_animation("pattern", worker)

    def fault_pattern(
        self,
        category_color: RGB,
        detail_code: int,
        *,
        brightness: float = 0.5,
    ) -> None:
        category_color = self._validate_color(category_color)
        if (
            isinstance(detail_code, bool)
            or not isinstance(detail_code, int)
            or not 1 <= detail_code <= 9
        ):
            raise ValueError("detail_code must be an integer from 1 to 9")
        level = self._validate_brightness(brightness)
        steps: list[PatternStep] = [
            ((255, 0, 0), 1.0, level),
            (category_color, 1.0, level),
        ]
        for index in range(detail_code):
            steps.append(((255, 255, 255), 0.2, level))
            if index + 1 < detail_code:
                steps.append((None, 0.2, 0.0))
        steps.append((None, 2.0, 0.0))
        self.pattern(steps, repeat=True)

    def stop_animation(self, *, keep_color: RGB | None = None, brightness: float = 1.0) -> None:
        thread = self._animation_thread
        if thread is not None and thread.is_alive():
            self._animation_stop.set()
            if thread is not current_thread():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    raise RuntimeError("RGB LED animation thread did not stop")
        self._animation_thread = None
        self._write(keep_color, self._validate_brightness(brightness))

    def stop_blink(self, *, keep_on: bool = False) -> None:
        self.stop_animation(keep_color=(255, 255, 255) if keep_on else None)

    def set_color(self, color: RGB, brightness: float | None = None) -> None:
        self.solid(color, 1.0 if brightness is None else brightness)

    def set_brightness(self, brightness: float) -> None:
        self.solid((255, 255, 255), brightness)

    def off(self) -> None:
        self._check_open()
        self.stop_animation()

    def close(self) -> None:
        if self._closed:
            return
        self.stop_animation()
        with self._lock:
            try:
                self._led.close()
            finally:
                if self._pin_factory is not None:
                    self._pin_factory.close()
                self._closed = True

    def __enter__(self) -> "RGBLed":
        self._check_open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
