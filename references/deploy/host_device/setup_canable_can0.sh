#!/usr/bin/env bash
set -u

# Create SocketCAN can0 from a CANable/CANable2 CDC-ACM device.
# Intended for udev RUN, but can also be run manually:
#   sudo ./setup_canable_can0.sh /dev/ttyACM1

TTY="${1:-${SLCAN_TTY:-/dev/ttyACM0}}"
CAN_IFACE="${CAN_IFACE:-can0}"
SLCAN_SPEED="${SLCAN_SPEED:-s8}"  # s8 = 1 Mbps
TXQUEUELEN="${CAN_TXQUEUELEN:-1000}"
LOG_FILE="${CANABLE_CAN0_LOG:-/tmp/canable_can0_udev.log}"

PATH=/usr/sbin:/usr/bin:/sbin:/bin

log() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG_FILE"
}

log "start tty=${TTY} iface=${CAN_IFACE} speed=${SLCAN_SPEED}"

if [ ! -e "$TTY" ]; then
    log "missing tty ${TTY}"
    exit 1
fi

modprobe can >>"$LOG_FILE" 2>&1 || true
modprobe can_raw >>"$LOG_FILE" 2>&1 || true
modprobe slcan >>"$LOG_FILE" 2>&1 || true

ip link set "$CAN_IFACE" down >>"$LOG_FILE" 2>&1 || true
ip link delete "$CAN_IFACE" >>"$LOG_FILE" 2>&1 || true
pkill -f "slcand.*${CAN_IFACE}" >>"$LOG_FILE" 2>&1 || true

slcand -o -c "-${SLCAN_SPEED}" "$TTY" "$CAN_IFACE" >>"$LOG_FILE" 2>&1
if [ "$?" -ne 0 ]; then
    log "slcand failed"
    exit 1
fi

for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if ip link show "$CAN_IFACE" >>"$LOG_FILE" 2>&1; then
        ip link set "$CAN_IFACE" txqueuelen "$TXQUEUELEN" >>"$LOG_FILE" 2>&1 || true
        ip link set "$CAN_IFACE" up >>"$LOG_FILE" 2>&1
        log "ready ${CAN_IFACE}"
        exit 0
    fi
    sleep 0.1
done

log "timeout waiting for ${CAN_IFACE}"
exit 1
