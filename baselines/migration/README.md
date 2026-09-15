# Migration safety baseline

`source_before.json` and `source_after.json` are generated with
`tools/snapshot_sources.py`. The digest covers source/config/document files and
intentionally excludes logs, builds, environments and Git internals. Equality
of each source's Git status and selected-tree digest is the no-impact gate.
