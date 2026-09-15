#!/usr/bin/env python3
"""Keep a gamepad-controlled start/stop boundary around start_robot.sh."""

from __future__ import annotations

import argparse
import errno
import os
import select
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
HARDWARE_DIR = SCRIPT_DIR.parent / "hardware"
if str(HARDWARE_DIR) not in sys.path:
    sys.path.insert(0, str(HARDWARE_DIR))

from rgb_led import RGBLed  # noqa: E402


JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT = struct.Struct("<IhBB")
IDLE_DISCONNECTED_COLOR = (255, 64, 0)
IDLE_CONNECTED_COLOR = (0, 255, 0)
STARTING_COLOR = (255, 255, 255)
RGB_HANDOFF_READY_FD_ENV = "WE11_RGB_HANDOFF_READY_FD"
RGB_HANDOFF_RELEASE_FD_ENV = "WE11_RGB_HANDOFF_RELEASE_FD"
RGB_HANDOFF_TIMEOUT_SECONDS = 15.0


def find_deploy_root() -> Path:
    configured = os.environ.get("WE11_DEPLOY_ROOT")
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "start_robot.sh").is_file() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("cannot locate Deploy root; set WE11_DEPLOY_ROOT")


class HoldChord:
    """One-shot D-pad-up+button hold detector that rearms after release."""

    def __init__(
        self,
        *,
        button_index: int,
        hold_seconds: float = 0.25,
        axis_index: int = 7,
        axis_threshold: int = 16384,
    ) -> None:
        self.hold_seconds = hold_seconds
        self.button_index = button_index
        self.axis_index = axis_index
        self.axis_threshold = axis_threshold
        self.buttons: dict[int, int] = {}
        self.axes: dict[int, int] = {}
        self.combo_since: float | None = None
        self.armed = True

    @property
    def active(self) -> bool:
        return (
            self.buttons.get(self.button_index, 0) == 1
            and self.axes.get(self.axis_index, 0) <= -self.axis_threshold
        )

    def update_button(self, number: int, value: int) -> None:
        self.buttons[number] = value

    def update_axis(self, number: int, value: int) -> None:
        self.axes[number] = value

    def reset(self) -> None:
        self.buttons.clear()
        self.axes.clear()
        self.combo_since = None
        self.armed = True

    def poll(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if not self.active:
            self.combo_since = None
            self.armed = True
            return False
        if not self.armed:
            return False
        if self.combo_since is None:
            self.combo_since = now
            return False
        if now - self.combo_since < self.hold_seconds:
            return False
        self.combo_since = None
        self.armed = False
        return True


class LinuxJoystick:
    """Reconnectable reader for the Linux joydev event interface."""

    def __init__(self, path: Path, chords: tuple[HoldChord, ...]) -> None:
        self.path = path
        self.chords = chords
        self.fd: int | None = None
        self.buffer = bytearray()
        self._waiting_logged = False

    @property
    def connected(self) -> bool:
        return self.fd is not None

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.buffer.clear()

    def _open(self) -> bool:
        if self.fd is not None:
            return True
        try:
            self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno not in {errno.ENOENT, errno.ENODEV, errno.EACCES}:
                raise
            if not self._waiting_logged:
                print(f"[GAMEPAD] waiting for readable {self.path}", flush=True)
                self._waiting_logged = True
            return False
        self._waiting_logged = False
        self.buffer.clear()
        print(f"[GAMEPAD] connected: {self.path}", flush=True)
        return True

    def _disconnect(self, reason: BaseException | str) -> None:
        print(f"[GAMEPAD] disconnected: {reason}", file=sys.stderr, flush=True)
        self.close()
        for chord in self.chords:
            chord.reset()

    def _consume(self) -> None:
        complete = len(self.buffer) - (len(self.buffer) % JS_EVENT.size)
        for offset in range(0, complete, JS_EVENT.size):
            _stamp, value, event_type, number = JS_EVENT.unpack_from(self.buffer, offset)
            event_type &= ~JS_EVENT_INIT
            for chord in self.chords:
                if event_type == JS_EVENT_BUTTON:
                    chord.update_button(number, value)
                elif event_type == JS_EVENT_AXIS:
                    chord.update_axis(number, value)
        if complete:
            del self.buffer[:complete]

    def read(self, timeout: float = 0.05) -> None:
        if not self._open():
            time.sleep(timeout)
            return
        assert self.fd is not None
        try:
            readable, _, _ = select.select([self.fd], [], [], timeout)
            if readable:
                data = os.read(self.fd, JS_EVENT.size * 64)
                if not data:
                    self._disconnect("end of device stream")
                else:
                    self.buffer.extend(data)
                    self._consume()
        except OSError as exc:
            if exc.errno not in {errno.EAGAIN, errno.EINTR}:
                self._disconnect(exc)


class GamepadLifecycleLauncher:
    def __init__(
        self,
        *,
        joy_device: Path,
        hold_seconds: float,
        led_factory: Callable[[], RGBLed] = RGBLed,
    ) -> None:
        self.root = find_deploy_root()
        self.stop_requested = False
        self.child: subprocess.Popen[bytes] | None = None
        self.start_chord = HoldChord(button_index=2, hold_seconds=hold_seconds)
        self.stop_chord = HoldChord(button_index=0, hold_seconds=hold_seconds)
        self.joystick = LinuxJoystick(joy_device, (self.start_chord, self.stop_chord))
        self._led_factory = led_factory
        self.led: RGBLed | None = None
        self._idle_led_connected: bool | None = None

    def _signal(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True

    def _open_idle_led(self) -> None:
        if self.led is None:
            self.led = self._led_factory()
            self._idle_led_connected = None
        self._sync_idle_led()

    def _sync_idle_led(self) -> None:
        if self.led is None:
            return
        self.led.raise_if_failed()
        connected = self.joystick.connected
        if connected == self._idle_led_connected:
            return
        color = IDLE_CONNECTED_COLOR if connected else IDLE_DISCONNECTED_COLOR
        self.led.breathe(color, brightness=0.5, period=2.0)
        self._idle_led_connected = connected
        state = "connected; ready for D-pad-up+X" if connected else "disconnected"
        print(f"[RGB] idle gamepad state: {state}", flush=True)

    def _close_idle_led(self) -> None:
        if self.led is not None:
            self.led.close()
            self.led = None
        self._idle_led_connected = None

    def _restore_idle_led(self) -> None:
        self._idle_led_connected = None
        self._open_idle_led()

    def _wait_for_child_led_ready(self, ready_fd: int) -> bool:
        deadline = time.monotonic() + RGB_HANDOFF_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.child is None or self.child.poll() is not None:
                return False
            readable, _, _ = select.select([ready_fd], [], [], 0.05)
            if not readable:
                continue
            return os.read(ready_fd, 1) == b"R"
        return False

    def _start_robot(self) -> None:
        command = [str(self.root / "start_robot.sh")]
        print(f"[LIFECYCLE] start requested: {' '.join(command)}", flush=True)
        assert self.led is not None
        self.led.breathe(STARTING_COLOR, brightness=0.5, period=2.0)
        print(
            "[RGB] WE11 starting: white breathing until supervisor handoff",
            flush=True,
        )
        ready_read_fd, ready_write_fd = os.pipe()
        release_read_fd, release_write_fd = os.pipe()
        child_environment = os.environ.copy()
        child_environment[RGB_HANDOFF_READY_FD_ENV] = str(ready_write_fd)
        child_environment[RGB_HANDOFF_RELEASE_FD_ENV] = str(release_read_fd)
        try:
            self.child = subprocess.Popen(
                command,
                cwd=self.root,
                start_new_session=True,
                env=child_environment,
                pass_fds=(ready_write_fd, release_read_fd),
            )
            os.close(ready_write_fd)
            ready_write_fd = -1
            os.close(release_read_fd)
            release_read_fd = -1
            if not self._wait_for_child_led_ready(ready_read_fd):
                raise RuntimeError(
                    "supervisor did not request RGB handoff within 15 seconds"
                )
            self._close_idle_led()
            os.write(release_write_fd, b"R")
            print("[RGB] GPIO ownership handed to WE11 supervisor", flush=True)
        except BaseException:
            self._restore_idle_led()
            raise
        finally:
            for fd in (
                ready_read_fd,
                ready_write_fd,
                release_read_fd,
                release_write_fd,
            ):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    def _signal_child(self, signum: signal.Signals) -> None:
        if self.child is None or self.child.poll() is not None:
            return
        try:
            os.killpg(self.child.pid, signum)
        except ProcessLookupError:
            pass

    def _stop_robot(self) -> None:
        if self.child is None or self.child.poll() is not None:
            return
        print("[LIFECYCLE] stop requested; sending SIGINT for safe cleanup", flush=True)
        self._signal_child(signal.SIGINT)
        try:
            self.child.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            print("[LIFECYCLE] cleanup timeout; sending SIGTERM", file=sys.stderr, flush=True)
        self._signal_child(signal.SIGTERM)
        try:
            self.child.wait(timeout=8)
            return
        except subprocess.TimeoutExpired:
            print("[LIFECYCLE] termination timeout; sending SIGKILL", file=sys.stderr, flush=True)
        self._signal_child(signal.SIGKILL)
        self.child.wait(timeout=3)

    def _wait_for_start_release(self) -> bool:
        print("[GAMEPAD] release D-pad-up+X to confirm start", flush=True)
        while not self.stop_requested:
            self.joystick.read()
            self.start_chord.poll()
            self.stop_chord.poll()
            if not self.start_chord.active:
                return True
        return False

    def run(self) -> int:
        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)
        print(
            "[LIFECYCLE] stopped; hold D-pad-up+X for "
            f"{self.start_chord.hold_seconds:.2f}s to start WE11",
            flush=True,
        )
        try:
            self._open_idle_led()
            while not self.stop_requested:
                self.joystick.read()
                if self.child is None:
                    self._open_idle_led()
                    self.stop_chord.poll()
                    if not self.start_chord.poll():
                        continue
                    print("[GAMEPAD] start chord accepted", flush=True)
                    if not self._wait_for_start_release():
                        break
                    self._start_robot()
                    continue

                code = self.child.poll()
                if code is not None:
                    print(
                        f"[LIFECYCLE] WE11 exited with code {code}; "
                        "hold D-pad-up+X to start again",
                        flush=True,
                    )
                    self.child = None
                    self._open_idle_led()
                    continue

                self.start_chord.poll()
                if self.stop_chord.poll():
                    print("[GAMEPAD] stop chord accepted", flush=True)
                    self._stop_robot()
        finally:
            self._stop_robot()
            self.joystick.close()
            self._close_idle_led()
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start WE11 with D-pad-up+X and safely stop it with D-pad-up+A."
    )
    parser.add_argument(
        "--joy-dev",
        default=os.environ.get("START_ROBOT_JOY_DEV", "/dev/input/js0"),
    )
    parser.add_argument("--hold-seconds", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hold_seconds <= 0.0:
        raise SystemExit("--hold-seconds must be positive")
    return GamepadLifecycleLauncher(
        joy_device=Path(args.joy_dev), hold_seconds=args.hold_seconds
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
