# Deploy reference snapshots

Imported 2026-09-15 as source-only working-tree snapshots.

| Snapshot | Source | Git provenance | State |
|---|---|---|---|
| `host_only` | `/ssd/Pheonix/DeployHostOnly/Deploy` | not a Git worktree | current files |
| `host_device` | `/ssd/Pheonix/DeployHostDevice/Deploy` | `0d5a910d8678d243c736972e77dd49be3d52790c`, branch `main` | dirty working tree included |

Filtering excludes nested `.git`, `build`, `install`, `log/logs`, Python/test
caches, bag/video captures, sweep result datasets and compiled libraries.
Source, configuration, scripts, documents, tests and required small policy
models are retained. These directories are never training or Sim2Sim build
dependencies; board-side Deploy remains independently developed.

Exact source status and selected-tree hashes are in
`baselines/migration/source_before.json`.
