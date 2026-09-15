#!/usr/bin/env python3
"""Minimal usage example for the RGBLed hardware interface."""

from __future__ import annotations

import signal
from time import sleep

from rgb_led import RGBLed


COLORS = (
    ("红", (255, 0, 0)),
    ("橙", (255, 64, 0)),
    ("蓝", (0, 0, 255)),
    ("绿", (0, 255, 0)),
    ("青", (0, 255, 255)),
    ("靛", (74, 0, 130)),
    ("紫", (148, 0, 211)),
)


def main() -> None:
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    with RGBLed(red_pin=22, green_pin=17, blue_pin=27, common_anode=True) as led:
        while running:
            for name, color in COLORS:
                if not running:
                    break
                print(name, flush=True)
                led.solid(color, brightness=0.5)
                sleep(0.8)


if __name__ == "__main__":
    main()
