#!/usr/bin/env bash
set -euo pipefail

CONFIG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd -- "${CONFIG_ROOT}/../.." && pwd)"
ENABLE_SERVICE=false

if [[ "${1:-}" == "--enable" ]]; then
  ENABLE_SERVICE=true
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--enable]" >&2
  exit 2
fi

echo "This installs WE11 host templates using sudo; no password is stored."
echo "Target repository: ${DEPLOY_ROOT}"

if [[ ! -r "${DEPLOY_ROOT}/install/setup.bash" ]]; then
  echo "Workspace is not built. Autostart never builds; run ./start_robot.sh --build first." >&2
  exit 1
fi
if [[ ! -x "${DEPLOY_ROOT}/src/deploy_tools/scripts/control/we11_gamepad_launcher.py" ]]; then
  echo "Gamepad lifecycle launcher is not executable." >&2
  exit 1
fi

sudo install -m 0644 "${CONFIG_ROOT}/99-auto-up-devs.rules" /etc/udev/rules.d/99-auto-up-devs.rules
sudo install -m 0644 "${CONFIG_ROOT}/canable-can0@.service" /etc/systemd/system/canable-can0@.service
sudo install -m 0644 "${CONFIG_ROOT}/we11-deploy.service" /etc/systemd/system/we11-deploy.service
sudo install -m 0644 "${CONFIG_ROOT}/we11-xpad.conf" /etc/modules-load.d/we11-xpad.conf
sudo install -m 0755 "${DEPLOY_ROOT}/setup_canable_can0.sh" /usr/local/sbin/setup_canable_can0.sh

sudo udevadm control --reload-rules
sudo systemctl daemon-reload

if [[ "${ENABLE_SERVICE}" == true ]]; then
  sudo systemctl enable we11-deploy.service
  echo "we11-deploy.service enabled; it will start at the next boot and was not started now."
  echo "Orange breathing means no gamepad; green breathing means ready."
  echo "At the next boot, hold D-pad-up+X to start WE11."
else
  echo "Templates installed, but we11-deploy.service was not enabled or started."
  echo "After real-hardware acceptance: sudo systemctl enable --now we11-deploy.service"
fi
