# Operational scripts

Repository-level operator workflows that combine configuration, logging, and
multiple executables live here. ROS package executables remain under `src/`.

- `sweeps/`: guided paired-motor sweep acquisition workflows and their YAML files.

This repository owns hardware acquisition only. Offline CSV import, coordinate
conversion, PACE fitting, Kp/Kd search, and Bode reports live in
`scripts/pace/` on the `Walking_Eagle-Unilab_final` branch of the UniLab
repository. The handoff boundary is the raw CSV plus its SHA-256, not a Python
source dependency between repositories.
