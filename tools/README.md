# Local environment

All mutable dependencies are kept outside this repository on the SSD:

```bash
bash tools/install_environment.sh
source tools/activate_environment.sh
bash tools/rebind_environment.sh --prefix /ssd/conda/envs/aar_unilab
bash tools/validate_installation.sh --all
python tools/compare_source_snapshots.py \
  baselines/migration/source_before.json baselines/migration/source_after.json
```

The scripts never run `conda init` and never change a user-level Conda
configuration. Re-run `rebind_environment.sh` after moving or cloning the
repository.
