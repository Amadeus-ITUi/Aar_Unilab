# Experiments

Hardware experiments that are not part of the normal deployment launch chain
live here. They keep their original behavior and require the same operator
safety checks as before the workspace reorganization.

Run the ESD-Link wing sweep through its compatibility path:

```bash
python3 experiments/wing/sweep_motor78_200hz.py
```

The active implementation sends a full P1-P8 `MaintenanceCommand`, requires
all ports and the IMU online, and uses three gamepad confirmations. It never
claims to enable P7/P8 independently. The former direct-SocketCAN implementation
is retained only at `experiments/wing/legacy/sweep_motor78_socketcan_200hz.py`.
