#!/usr/bin/env python3
"""Pseudo-terminal integration test for the event-driven ESD-Link bridge.

The fake lower controller implements only frozen schema-1 messages needed by
this test.  It never opens the real robot serial device.
"""

import binascii
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import tempfile
import threading
import time

import pytest


MAGIC = 0xA5
WIRE_VERSION = 1
SESSION_ID = 0x13572468
FINGERPRINT = 0x4CA27910
SAFE_RAW_POSITIONS = {
    1: 0.92020,
    2: 0.98338,
    3: 0.0,
    4: -0.92020,
    5: -0.98338,
    6: 0.0,
    7: -0.7,
    8: 0.8,
}


def cobs_encode(raw: bytes) -> bytes:
    encoded = bytearray(b"\x00")
    code_index = 0
    code = 1
    for value in raw:
        if value == 0:
            encoded[code_index] = code
            code_index = len(encoded)
            encoded.append(0)
            code = 1
        else:
            encoded.append(value)
            code += 1
            if code == 0xFF:
                encoded[code_index] = code
                code_index = len(encoded)
                encoded.append(0)
                code = 1
    encoded[code_index] = code
    return bytes(encoded)


def cobs_decode(encoded: bytes) -> bytes:
    raw = bytearray()
    offset = 0
    while offset < len(encoded):
        code = encoded[offset]
        if code == 0:
            raise ValueError("zero byte inside COBS payload")
        offset += 1
        count = code - 1
        if offset + count > len(encoded):
            raise ValueError("truncated COBS payload")
        raw.extend(encoded[offset:offset + count])
        offset += count
        if code != 0xFF and offset < len(encoded):
            raw.append(0)
    return bytes(raw)


def make_wire(kind: int, service: int, message: int, sequence: int, payload: bytes) -> bytes:
    header = struct.pack("<BBBBBBHI", MAGIC, WIRE_VERSION, kind, 0, service, message,
                         len(payload), sequence)
    raw = header + payload
    raw += struct.pack("<I", binascii.crc32(raw) & 0xFFFFFFFF)
    return b"\x00" + cobs_encode(raw) + b"\x00"


def parse_wire(encoded: bytes):
    raw = cobs_decode(encoded)
    if len(raw) < 16:
        raise ValueError("short frame")
    magic, version, kind, flags, service, message, payload_len, sequence = struct.unpack(
        "<BBBBBBHI", raw[:12])
    if magic != MAGIC or version != WIRE_VERSION or flags != 0:
        raise ValueError("invalid header")
    if len(raw) != 16 + payload_len:
        raise ValueError("invalid payload length")
    expected_crc, = struct.unpack("<I", raw[-4:])
    if expected_crc != (binascii.crc32(raw[:-4]) & 0xFFFFFFFF):
        raise ValueError("invalid CRC")
    return kind, service, message, sequence, raw[12:-4]


def robot_state_payload(sample_sequence: int, control_state: int = 0,
                        raw_positions=None) -> bytes:
    prefix = struct.pack(
        "<IHIIIBBB", SESSION_ID, 1, (sample_sequence * 4000) & 0xFFFFFFFF,
        sample_sequence & 0xFFFFFFFF, 0, 0, control_state, 3)
    imu = struct.pack("<B10f", 0x07, 0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0,
                      0.0, 0.0, 9.81)
    kinematics = bytearray(struct.pack("<B", 8))
    efforts = bytearray(struct.pack("<B", 8))
    for port_id in range(1, 9):
        position = (raw_positions or SAFE_RAW_POSITIONS)[port_id]
        kinematics.extend(struct.pack("<BBff", port_id, 0x07, position, 0.0))
        efforts.extend(struct.pack("<Bf", port_id, 0.0))
    return (prefix + struct.pack("<BH", 1, len(imu)) + imu
            + struct.pack("<BH", 2, len(kinematics)) + bytes(kinematics)
            + struct.pack("<BH", 3, len(efforts)) + bytes(efforts))


def device_status_payload(control_state: int) -> bytes:
    return struct.pack(
        "<IIBIHIIIHII", SESSION_ID, 0x01020304, control_state, 0, 0,
        11, 2, 3, 4, 0x1FE, 0)


class FakeLower:
    def __init__(self, start_sequence: int, disable_delay_seconds: float = 0.0):
        self.master_fd, self.slave_fd = pty.openpty()
        os.set_blocking(self.master_fd, False)
        self.slave_path = os.ttyname(self.slave_fd)
        self.sample_sequence = start_sequence
        self.wire_sequence = start_sequence
        self.session_open = threading.Event()
        self.send_states = threading.Event()
        self.send_states.set()
        self.session_open_count = 0
        self.stop_event = threading.Event()
        self.enable_requests = []
        self.actuator_commands = 0
        self.actuator_command_ports = []
        self.control_state = 2
        self.disable_delay_seconds = disable_delay_seconds
        self.disable_complete_at = None
        self.raw_positions = dict(SAFE_RAW_POSITIONS)
        self._rx = bytearray()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        self._thread.join(timeout=2.0)
        for fd in (self.master_fd, self.slave_fd):
            if fd < 0:
                continue
            try:
                os.close(fd)
            except OSError:
                pass
        self.master_fd = -1
        self.slave_fd = -1

    def disconnect(self):
        self.stop()

    def _write(self, wire: bytes):
        offset = 0
        while offset < len(wire) and not self.stop_event.is_set():
            try:
                offset += os.write(self.master_fd, wire[offset:])
            except BlockingIOError:
                select.select([], [self.master_fd], [], 0.01)

    def _session_response(self, host_nonce: int):
        payload = struct.pack(
            "<HIIIBHIHIHIB", 0, host_nonce, 0x01020304, SESSION_ID, 1, 1,
            0x1FE, 1, FINGERPRINT, 250, 0, 0)
        self._write(make_wire(1, 1, 1, self.wire_sequence, payload))
        self.wire_sequence += 1
        self._write(make_wire(
            2, 3, 2, self.wire_sequence,
            device_status_payload(self.control_state)))
        self.wire_sequence += 1
        self.session_open_count += 1
        self.session_open.set()

    def _transaction_response(self, service: int, message: int, transaction_id: int):
        payload = struct.pack(
            "<HIIIBHIIIII", 0, SESSION_ID, SESSION_ID, transaction_id, 0, 0,
            FINGERPRINT, 0x1FE, 0x1FE, 0x1FE, 0)
        self._write(make_wire(1, service, message, self.wire_sequence, payload))
        self.wire_sequence += 1

    def _handle_host_frame(self, encoded: bytes):
        try:
            kind, service, message, _, payload = parse_wire(encoded)
        except ValueError:
            return
        if kind == 0 and service == 1 and message == 1 and len(payload) == 17:
            host_nonce, = struct.unpack("<I", payload[:4])
            self._session_response(host_nonce)
        elif kind == 0 and service == 2 and message == 2 and len(payload) >= 9:
            transaction_id, = struct.unpack("<I", payload[4:8])
            enable = bool(payload[8])
            self.enable_requests.append(enable)
            if enable:
                self.control_state = 3
                self.disable_complete_at = None
            elif self.disable_delay_seconds > 0.0:
                self.control_state = 5
                self.disable_complete_at = time.monotonic() + self.disable_delay_seconds
            else:
                self.control_state = 2
            self._transaction_response(service, message, transaction_id)
        elif kind == 0 and service == 4 and message == 1 and len(payload) >= 10:
            transaction_id, = struct.unpack("<I", payload[4:8])
            self._transaction_response(service, message, transaction_id)
        elif kind == 2 and service == 2 and message == 1:
            self.actuator_commands += 1
            if len(payload) == 15 + 8 * 21 and payload[14] == 8:
                ports = {}
                offset = 15
                for _ in range(8):
                    port_id, position, velocity, kp, kd, effort = struct.unpack_from(
                        "<Bfffff", payload, offset)
                    ports[port_id] = (position, velocity, kp, kd, effort)
                    offset += 21
                self.actuator_command_ports.append(ports)

    def _read_host(self):
        try:
            chunk = os.read(self.master_fd, 4096)
        except BlockingIOError:
            return
        if not chunk:
            return
        self._rx.extend(chunk)
        while True:
            try:
                first = self._rx.index(0)
            except ValueError:
                self._rx.clear()
                return
            if first:
                del self._rx[:first]
            try:
                end = self._rx.index(0, 1)
            except ValueError:
                return
            encoded = bytes(self._rx[1:end])
            del self._rx[:end + 1]
            if encoded:
                self._handle_host_frame(encoded)

    def _run(self):
        next_state = time.monotonic()
        while not self.stop_event.is_set():
            if (self.disable_complete_at is not None
                    and time.monotonic() >= self.disable_complete_at):
                self.control_state = 2
                self.disable_complete_at = None
            readable, _, _ = select.select([self.master_fd], [], [], 0.001)
            if readable:
                self._read_host()
            now = time.monotonic()
            if self.session_open.is_set() and self.send_states.is_set() and now >= next_state:
                payload = robot_state_payload(
                    self.sample_sequence, self.control_state, self.raw_positions)
                self._write(make_wire(2, 3, 1, self.wire_sequence, payload))
                self.sample_sequence = (self.sample_sequence + 1) & 0xFFFFFFFF
                self.wire_sequence = (self.wire_sequence + 1) & 0xFFFFFFFF
                next_state += 0.004
                if next_state < now - 0.020:
                    next_state = now + 0.004
            elif not self.send_states.is_set():
                next_state = now + 0.004


def bridge_executable() -> Path:
    configured = os.environ.get("ESD_LINK_BRIDGE_TEST_EXECUTABLE")
    if configured:
        return Path(configured)
    deploy_root = Path(__file__).resolve().parents[3]
    return deploy_root / "build" / "esd_link_bridge" / "esd_link_bridge_node"


def spin_until(node, predicate, timeout: float) -> bool:
    import rclpy
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return True
    return False


def test_250hz_reconnect_and_lower_watchdog_owned_command_flow():
    # Imports follow ROS_DOMAIN_ID selection so this test stays isolated from a
    # running robot graph.
    os.environ["ROS_DOMAIN_ID"] = str(180 + (os.getpid() % 20))
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    import rclpy
    from esd_link_msgs.msg import LinkStatus, LowerState, MaintenanceCommand, PolicyCommand
    from esd_link_msgs.srv import SetControlMode, SetLowerZeros
    from motors.srv import SetMasterID, SetOperationalState
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    executable = bridge_executable()
    assert executable.is_file(), f"bridge executable not found: {executable}"
    lower1 = FakeLower(start_sequence=1000)
    lower2 = None
    process = None
    rclpy.init()
    node = rclpy.create_node("esd_link_bridge_pty_test")
    state_sequences = []
    states = []
    statuses = []

    def save_state(msg):
        state_sequences.append(msg.state_sample_seq)
        states.append(msg)
    node.create_subscription(
        LowerState, "/lower/state", save_state,
        QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT))
    node.create_subscription(
        LinkStatus, "/lower/link_status", lambda msg: statuses.append(msg),
        QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                   durability=DurabilityPolicy.TRANSIENT_LOCAL,
                   history=HistoryPolicy.KEEP_LAST))
    mode_client = node.create_client(SetControlMode, "/lower/set_control_mode")
    maintenance_pub = node.create_publisher(
        MaintenanceCommand, "/maintenance/commands", 1)
    policy_pub = node.create_publisher(
        PolicyCommand, "/policy/stamped_commands", 1)
    zero_client = node.create_client(SetLowerZeros, "/lower/set_zeros")
    operational_client = node.create_client(
        SetOperationalState, "/motors/set_operational_state")
    master_id_client = node.create_client(SetMasterID, "/set_master_id")

    try:
        with tempfile.TemporaryDirectory(prefix="esd-link-pty-") as temp_dir:
            device_link = Path(temp_dir) / "esd-link"
            identity_record = Path(temp_dir) / "identity.txt"
            os.symlink(lower1.slave_path, device_link)
            lower1.start()
            process = subprocess.Popen(
                [str(executable), "--ros-args", "-p", f"serial_device:={device_link}",
                 "-p", "realtime_priority:=0", "-p", "cpu_affinity:=-1",
                 "-p", f"identity_record_path:={identity_record}",
                 "-p", "standby_ramp_seconds:=0.08"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            assert spin_until(node, lambda: lower1.session_open.is_set(), 4.0)
            # DDS discovery and best-effort delivery can legitimately cover
            # intermediate states on a loaded Raspberry Pi.  A short run of
            # monotonic snapshots proves publication; LinkStatus below checks
            # the bridge-side source rate independently of ROS delivery count.
            assert spin_until(
                node, lambda: node.count_publishers("/lower/state") == 1, 8.0)
            state_sequences.clear()
            states.clear()
            assert spin_until(node, lambda: len(state_sequences) >= 40, 3.0)
            assert all(b > a for a, b in zip(state_sequences, state_sequences[1:]))
            assert states[-1].position_rad[6] == pytest.approx(0.7, abs=1e-5)
            assert states[-1].position_rad[7] == pytest.approx(-0.8, abs=1e-5)
            assert spin_until(
                node, lambda: any(240.0 <= status.state_rate_hz <= 260.0 for status in statuses),
                4.0)
            assert spin_until(
                node,
                lambda: any(
                    status.device_valid_commands == 11
                    and status.device_invalid_frames == 2
                    and status.device_rejected_commands == 3
                    and status.device_command_age_ms == 4
                    for status in statuses),
                2.5)
            assert lower1.enable_requests == []
            assert lower1.actuator_commands == 0

            assert operational_client.wait_for_service(timeout_sec=2.0)
            invalid_operational = SetOperationalState.Request()
            invalid_operational.target_state = 255
            invalid_future = operational_client.call_async(invalid_operational)
            assert spin_until(node, invalid_future.done, 2.0)
            assert not invalid_future.result().success
            assert invalid_future.result().actual_state == SetOperationalState.Request.DISARMED
            assert lower1.enable_requests == []

            assert master_id_client.wait_for_service(timeout_sec=2.0)
            master_id_request = SetMasterID.Request()
            master_id_request.motor_id = 1
            master_id_request.master_id = 13
            master_id_future = master_id_client.call_async(master_id_request)
            assert spin_until(node, master_id_future.done, 2.0)
            assert not master_id_future.result().success
            assert master_id_future.result().message == "unsupported by frozen ESD-Link firmware"
            assert lower1.enable_requests == []

            assert zero_client.wait_for_service(timeout_sec=2.0)
            zero_request = SetLowerZeros.Request()
            zero_request.persist = True
            zero_request.port_ids = [1, 2, 3, 4, 5, 6]
            zero_request.assigned_position_rad = [0.0] * 6
            zero_future = zero_client.call_async(zero_request)
            assert spin_until(node, zero_future.done, 3.0)
            assert zero_future.result().success
            assert spin_until(node, lambda: lower1.session_open_count >= 2, 2.0)
            assert "config_fingerprint=0x4ca27910" in identity_record.read_text()
            assert lower1.enable_requests == []

            before_reconnect = len(state_sequences)
            lower1.disconnect()
            # Normal DISABLED transitions deliberately exceed the 40 ms
            # maintenance deadline.  The bridge must suppress stale-command
            # faults while an explicit full-device disable is in progress.
            lower2 = FakeLower(start_sequence=10000, disable_delay_seconds=0.12)
            device_link.unlink()
            os.symlink(lower2.slave_path, device_link)
            lower2.start()
            assert spin_until(node, lambda: lower2.session_open.is_set(), 4.0)
            assert spin_until(
                node, lambda: len(state_sequences) >= before_reconnect + 40
                and state_sequences[-1] >= 10000, 2.0)
            assert lower2.enable_requests == []
            assert lower2.actuator_commands == 0

            # Direct ground start must only enable the lower controller.  It
            # must not route through STANDBY or inject a current/default-pose
            # command before the first genuine PolicyCommand arrives.
            assert operational_client.wait_for_service(timeout_sec=2.0)
            direct_request = SetOperationalState.Request()
            direct_request.target_state = direct_request.ACTIVE
            direct_future = operational_client.call_async(direct_request)
            assert spin_until(node, direct_future.done, 2.0)
            assert direct_future.result().success
            assert direct_future.result().actual_state == direct_request.ACTIVE
            assert lower2.enable_requests == [True]
            no_command_deadline = time.monotonic() + 0.10
            while time.monotonic() < no_command_deadline:
                rclpy.spin_once(node, timeout_sec=0.01)
            assert lower2.actuator_commands == 0

            assert spin_until(node, lambda: policy_pub.get_subscription_count() == 1, 2.0)
            first_policy = PolicyCommand()
            first_policy.source_state_sample_seq = states[-1].state_sample_seq
            first_policy.action = [0.20, -0.20, 5.0, -0.10, 0.10, -5.0]
            policy_pub.publish(first_policy)
            assert spin_until(node, lambda: lower2.actuator_commands == 1, 1.0)
            first_ports = lower2.actuator_command_ports[0]
            assert first_ports[1][0] == pytest.approx(0.72020, abs=2e-4)
            assert first_ports[2][0] == pytest.approx(0.78338, abs=2e-4)
            assert first_ports[3][1] == pytest.approx(-5.0, abs=2e-4)
            assert first_ports[4][0] == pytest.approx(-1.02020, abs=2e-4)
            assert first_ports[5][0] == pytest.approx(-1.08338, abs=2e-4)
            assert first_ports[6][1] == pytest.approx(-5.0, abs=2e-4)

            direct_request.target_state = direct_request.DISARMED
            direct_future = operational_client.call_async(direct_request)
            assert spin_until(node, direct_future.done, 3.0)
            assert direct_future.result().success
            assert lower2.enable_requests == [True, False]

            assert mode_client.wait_for_service(timeout_sec=2.0)
            request = SetControlMode.Request()
            request.target_mode = request.STANDBY
            standby_started = time.monotonic()
            future = mode_client.call_async(request)
            assert spin_until(node, future.done, 3.0)
            assert future.result().success
            assert time.monotonic() - standby_started >= 0.06
            assert lower2.actuator_commands >= 10
            expected_standby = {
                1: 0.92020, 2: 0.98338, 3: 0.0,
                4: -0.92020, 5: -0.98338, 6: 0.0,
                7: -0.7, 8: 0.8,
            }
            assert spin_until(
                node,
                lambda: lower2.actuator_command_ports
                and all(
                    lower2.actuator_command_ports[-1][port_id][0]
                    == pytest.approx(expected_position, abs=2e-4)
                    for port_id, expected_position in expected_standby.items()),
                1.0)
            # Supervisor handoff intentionally enables inference just before it
            # selects bridge POLICY, and stops it just after returning to
            # STANDBY. These boundary samples are expected, not rejections.
            assert spin_until(node, lambda: policy_pub.get_subscription_count() == 1, 2.0)
            status_count = len(statuses)
            rejection_baseline = statuses[-1].rejected_commands
            priming_policy = PolicyCommand()
            priming_policy.source_state_sample_seq = states[-1].state_sample_seq
            priming_policy.action = [0.0] * 6
            for _ in range(5):
                policy_pub.publish(priming_policy)
                rclpy.spin_once(node, timeout_sec=0.01)
            assert spin_until(node, lambda: len(statuses) > status_count, 1.5)
            assert statuses[-1].rejected_commands == rejection_baseline
            final_standby = lower2.actuator_command_ports[-1]
            for port_id, expected_position in expected_standby.items():
                assert final_standby[port_id][0] == pytest.approx(
                    expected_position, abs=2e-4)

            # Match the legacy motors_node POLICY contract: each finite
            # position target is clipped independently, and one out-of-range
            # joint must not discard the complete P1-P8 command. Wheel targets
            # retain their separate velocity contract.
            request.target_mode = request.POLICY
            future = mode_client.call_async(request)
            assert spin_until(node, future.done, 2.0)
            assert future.result().success
            actuator_count = lower2.actuator_commands
            rejection_baseline = statuses[-1].rejected_commands
            clipped_policy = PolicyCommand()
            clipped_policy.source_state_sample_seq = states[-1].state_sample_seq
            clipped_policy.action = [0.90, -0.95, 12.0, 0.874, 0.30, -12.0]
            policy_pub.publish(clipped_policy)
            assert spin_until(
                node,
                lambda: lower2.actuator_command_ports
                and lower2.actuator_command_ports[-1][1][0]
                == pytest.approx(0.08020, abs=2e-4)
                and lower2.actuator_command_ports[-1][4][0]
                == pytest.approx(-0.08020, abs=2e-4),
                1.0)
            assert lower2.actuator_commands > actuator_count
            clipped_ports = lower2.actuator_command_ports[-1]
            assert clipped_ports[1][0] == pytest.approx(0.08020, abs=2e-4)
            assert clipped_ports[2][0] == pytest.approx(0.09338, abs=2e-4)
            assert clipped_ports[3][1] == pytest.approx(-12.0, abs=2e-4)
            assert clipped_ports[4][0] == pytest.approx(-0.08020, abs=2e-4)
            assert clipped_ports[5][0] == pytest.approx(-1.20338, abs=2e-4)
            assert clipped_ports[6][1] == pytest.approx(-12.0, abs=2e-4)
            status_count = len(statuses)
            assert spin_until(node, lambda: len(statuses) > status_count, 1.5)
            assert statuses[-1].rejected_commands == rejection_baseline

            request.target_mode = request.DISABLED
            disable_count = lower2.enable_requests.count(False)
            future = mode_client.call_async(request)
            assert spin_until(node, future.done, 3.0)
            assert future.result().success
            assert lower2.enable_requests.count(False) == disable_count + 1

            # Gravity may leave a disabled joint between the software command
            # boundary and its mechanical stop.  SWEEP must still enter with
            # an exact current-position hold and allow only inward recovery.
            lower2.raw_positions[2] = 1.28338  # P2 policy position +0.3000 rad.
            assert spin_until(
                node, lambda: states and states[-1].position_rad[1] > 0.29, 1.0)
            enable_count = lower2.enable_requests.count(True)
            request.target_mode = request.SWEEP
            future = mode_client.call_async(request)
            assert spin_until(node, future.done, 3.0)
            assert future.result().success
            assert lower2.enable_requests.count(True) == enable_count + 1
            assert spin_until(
                node,
                lambda: lower2.actuator_command_ports
                and lower2.actuator_command_ports[-1][2][0]
                == pytest.approx(1.28338, abs=1e-5),
                1.0)

            maintenance = MaintenanceCommand()
            maintenance.mode = MaintenanceCommand.SWEEP
            maintenance.port_id = list(range(1, 9))
            maintenance.position_rad = [0.0, 0.28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            maintenance.velocity_rad_s = [0.0] * 8
            maintenance.kp = [1.0] * 8
            maintenance.kd = [0.1] * 8
            maintenance.effort_nm = [0.0] * 8
            for _ in range(4):
                maintenance.source_state_sample_seq = states[-1].state_sample_seq
                maintenance_pub.publish(maintenance)
                rclpy.spin_once(node, timeout_sec=0.005)
            assert spin_until(
                node,
                lambda: lower2.actuator_command_ports
                and lower2.actuator_command_ports[-1][2][0]
                == pytest.approx(1.26338, abs=1e-5),
                1.0)

            # A target farther toward the mechanical stop than the position at
            # mode entry is dropped locally. One malformed publisher sample no
            # longer turns a healthy lower controller into a global host fault.
            # Let the last valid SWEEP command clear the PTY before taking the
            # baseline; it may arrive just after the ROS service response.
            settle_end = time.monotonic() + 0.10
            while time.monotonic() < settle_end:
                rclpy.spin_once(node, timeout_sec=0.01)
            disable_count = lower2.enable_requests.count(False)
            actuator_count = lower2.actuator_commands
            maintenance.position_rad[1] = 0.32
            previous_sequence = states[-1].state_sample_seq
            assert spin_until(
                node, lambda: states[-1].state_sample_seq > previous_sequence, 1.0)
            maintenance.source_state_sample_seq = states[-1].state_sample_seq
            maintenance_pub.publish(maintenance)
            end = time.monotonic() + 0.15
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.01)
            assert lower2.enable_requests.count(False) == disable_count
            assert lower2.actuator_commands == actuator_count

            request.target_mode = request.DISABLED
            future = mode_client.call_async(request)
            assert spin_until(node, future.done, 3.0)
            assert future.result().success
            lower2.raw_positions[2] = SAFE_RAW_POSITIONS[2]
            assert spin_until(
                node, lambda: states and abs(states[-1].position_rad[1]) < 0.01, 1.0)

            request.target_mode = request.SWEEP
            future = mode_client.call_async(request)
            assert spin_until(node, future.done, 3.0)
            assert future.result().success

            maintenance = MaintenanceCommand()
            maintenance.mode = MaintenanceCommand.SWEEP
            maintenance.port_id = list(range(1, 9))
            # Exercise targets outside the old +/-pi/2 limit but inside the
            # disabled-bench measured ranges.  Both must be sign-flipped on TX.
            maintenance.position_rad = [0.0] * 6 + [1.8, -1.9]
            maintenance.velocity_rad_s = [0.0] * 8
            maintenance.kp = [1.0] * 8
            maintenance.kd = [0.1] * 8
            maintenance.effort_nm = [0.0] * 8
            for _ in range(4):
                maintenance.source_state_sample_seq = states[-1].state_sample_seq
                maintenance_pub.publish(maintenance)
                rclpy.spin_once(node, timeout_sec=0.005)
            assert spin_until(
                node,
                lambda: lower2.actuator_command_ports
                and lower2.actuator_command_ports[-1][7][0] == pytest.approx(-1.8, abs=1e-5)
                and lower2.actuator_command_ports[-1][8][0] == pytest.approx(1.9, abs=1e-5),
                1.0)
            settle_end = time.monotonic() + 0.10
            while time.monotonic() < settle_end:
                rclpy.spin_once(node, timeout_sec=0.01)
            actuator_count = lower2.actuator_commands
            maintenance_pub.publish(maintenance)
            end = time.monotonic() + 0.10
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.01)
            assert lower2.actuator_commands == actuator_count

            # The host policy contract accepts at most 35 rad/s on P3/P6.
            # The wheel speed observed in the real-policy dry run must pass.
            maintenance.velocity_rad_s[2] = 18.6
            previous_sequence = states[-1].state_sample_seq
            assert spin_until(
                node, lambda: states[-1].state_sample_seq > previous_sequence, 1.0)
            maintenance.source_state_sample_seq = states[-1].state_sample_seq
            maintenance_pub.publish(maintenance)
            assert spin_until(
                node, lambda: lower2.actuator_commands == actuator_count + 1, 1.0)
            actuator_count = lower2.actuator_commands

            # A legacy direct-CAN wheel target above that capability must be
            # dropped by the bridge and never forwarded to the device.
            maintenance.velocity_rad_s[2] = 35.01
            previous_sequence = states[-1].state_sample_seq
            assert spin_until(
                node, lambda: states[-1].state_sample_seq > previous_sequence, 1.0)
            maintenance.source_state_sample_seq = states[-1].state_sample_seq
            maintenance_pub.publish(maintenance)
            end = time.monotonic() + 0.10
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.01)
            assert lower2.actuator_commands == actuator_count
            maintenance.velocity_rad_s[2] = 0.0

            request.target_mode = request.DISABLED
            disable_count = lower2.enable_requests.count(False)
            future = mode_client.call_async(request)
            assert spin_until(node, future.done, 3.0)
            assert future.result().success
            assert lower2.enable_requests.count(False) == disable_count + 1

            request.target_mode = request.SWEEP
            actuator_before_enable = lower2.actuator_commands
            future = mode_client.call_async(request)
            assert spin_until(node, future.done, 3.0)
            assert future.result().success
            assert spin_until(node, lambda: True in lower2.enable_requests, 1.0)
            assert spin_until(
                node, lambda: lower2.actuator_commands > actuator_before_enable, 1.0)
            disable_count = lower2.enable_requests.count(False)
            # No MaintenanceCommand is sent. The bridge sends no heartbeat and
            # does not synthesize a host-side timeout fault; the frozen lower's
            # 200 ms watchdog is the sole runtime command deadline.
            actuator_count = lower2.actuator_commands
            end = time.monotonic() + 0.15
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.01)
            assert lower2.actuator_commands == actuator_count
            assert lower2.enable_requests.count(False) == disable_count
            assert not any(status.link_state == status.FAULT for status in statuses[-2:])

            request.target_mode = request.DISABLED
            future = mode_client.call_async(request)
            assert spin_until(node, future.done, 3.0)
            assert future.result().success
            disable_count = lower2.enable_requests.count(False)
            request.target_mode = request.SWEEP
            actuator_before_enable = lower2.actuator_commands
            future = mode_client.call_async(request)
            assert spin_until(node, future.done, 3.0)
            assert future.result().success, (
                future.result().message,
                lower2.control_state,
                states[-1].control_state,
                statuses[-1].device_control_state,
                statuses[-1].fault_flags,
                statuses[-1].offline_port_mask,
            )
            assert spin_until(
                node, lambda: lower2.actuator_commands > actuator_before_enable, 1.0)
            lower2.send_states.clear()
            # A silent-but-connected fake lower is not interpreted by the host
            # as a second watchdog. With no new state-derived command, TX also
            # remains silent and the real lower would own the 50 ms transition.
            actuator_count = lower2.actuator_commands
            end = time.monotonic() + 0.15
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.01)
            assert lower2.enable_requests.count(False) == disable_count
            assert lower2.actuator_commands == actuator_count
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if process is not None and process.returncode not in (None, 0, -signal.SIGINT):
            output = process.stdout.read() if process.stdout else ""
            pytest.fail(f"bridge exited with {process.returncode}:\n{output}")
        lower1.stop()
        if lower2 is not None:
            lower2.stop()
        node.destroy_node()
        rclpy.shutdown()
