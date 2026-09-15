# Host system configuration

This directory stores host-level configuration templates that are not ROS
package resources:

- `99-auto-up-devs.rules`: udev rule for the configured CANable2 serial number.
- `canable-can0@.service`: systemd template used by that udev rule.
- `we11-deploy.service`: gamepad lifecycle service template. It runs as `esd`,
  never builds the workspace, starts WE11 only after a held D-pad-up+X chord,
  and safely stops it after a held D-pad-up+A chord. Only the lightweight
  launcher is restarted on failure; the robot child never restarts itself.
  While the child is stopped, orange breathing means the gamepad is missing
  and green breathing means it is connected and ready for D-pad-up+X.
- `wait-we11-devices.sh`: retained as a manual boot-device diagnostic helper;
  the lifecycle service no longer uses it, so missing CAN/IMU hardware can be
  reported through the normal supervisor fault LED after a gamepad start.
- `we11-xpad.conf`: loads the Xbox controller kernel driver at boot.
- `install-we11-host.sh`: explicit host installer. Without `--enable`, it does
  not enable or start the deployment service.

Moving these templates does not install or activate them. Existing deployed
copies under `/etc` are unaffected.

Repository-root compatibility symlinks preserve the previous source paths, so
existing `sudo install ... 99-auto-up-devs.rules` and service installation
commands continue to work.

The repository never stores or pipes a sudo password. Build once in a
maintenance terminal, then install and enable boot autostart:

```bash
./start_robot.sh --build
./config/system/install-we11-host.sh --enable
```

`--enable` does not start the robot immediately. It enables the gamepad launcher
for the next boot. Holding D-pad-up+X then releasing it invokes `start_robot.sh`
without `--build`; `WE11_AUTOSTART=1` also makes `start_robot.sh --build` fail if
a future unit edit accidentally adds it.
Systemd standard input is explicitly set to `null`; all operator gates use the
Xbox `/joy` topic rather than a keyboard, mouse, or terminal.

The lifecycle launcher owns the RGB GPIO while idle. After the start chord it
switches to white breathing, launches the child, and keeps that animation until
the supervisor requests ownership through a two-way pipe handshake. It then
releases GPIO and acknowledges the supervisor, which immediately continues the
white boot breathing pattern. Stop `we11-deploy.service` before invoking
`./start_robot.sh` directly, otherwise the script intentionally refuses to run.
