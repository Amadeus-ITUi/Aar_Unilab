#!/usr/bin/env bash
# Uses the supported SSD environment; no Conda activation or conda init required.
set -euo pipefail

train_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
train_python=/ssd/conda/envs/aar_unilab/bin/python
if [[ ! -x "$train_python" ]]; then
  echo "SSD environment missing; run bash tools/install_environment.sh" >&2
  exit 1
fi
cd "$train_root"
# -E/-s isolate the lightweight launcher itself; its child gets YAML environment settings.
exec "$train_python" -E -s -m unilab.catalog.launch "$@"
