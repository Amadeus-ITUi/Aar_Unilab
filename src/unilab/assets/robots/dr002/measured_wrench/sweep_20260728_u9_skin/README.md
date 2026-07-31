# U9 2026-07-28 with-skin measured wrench

This directory contains the phase-aligned 1/2/3 Hz six-axis wrench assets used
by the WE9 MuJoCo training task.

The immutable raw recordings are:

- `010_1hz_20cyc.csv`
- `020_2hz_20cyc.csv`
- `030_3hz_20cyc.csv`

They were read from `/home/esd_wch/下载/u9/7.28带蒙皮`. Their complete SHA-256
lineage is recorded in `manifest.json`; the raw files are not copied or edited.

## Preprocessing

`scripts/data/prepare_u9_skin_wrench.py`:

1. validates and deduplicates force frames by `force_frame_count`;
2. uses `force_t_abs_s` as the wrench clock;
3. averages the steady `[2/f, 19/f)` interval across 17 cycles without
   subtracting DC;
4. aligns latest `m1_pos_deg` to the retained old motor7 wing-observation
   phase;
5. writes `time,Fx,Fy,Fz,Mx,My,Mz` at 1 kHz over 10 seconds and copies the
   first wrench exactly to the 10-second endpoint.

The source-time advances are:

| Frequency | Phase advance | Time advance |
|---:|---:|---:|
| 1 Hz | 193.276089517 deg | 536.878026435 ms |
| 2 Hz | 200.852380249 deg | 278.961639235 ms |
| 3 Hz | 163.734797622 deg | 151.606294094 ms |

Positive advance means
`aligned(t) = unaligned((t + advance) modulo one cycle)`.

## Runtime interpretation

The output preserves the force-sensor-local channel convention expected by
WE9. The raw logger does not independently declare the sensor mounting, so
unchanged mounting/sign conventions are an explicit assumption.

WE9 currently zeros raw sensor `Fy`, rotates force and measured moment into the
base frame, and adds `(push_site - randomized COM) x F` only to the physical
application torque. The critic receives the rotated measured moment without
that transport term. One reset-sampled scalar in `[0.5, 1.5]` still scales all
six raw channels together for the episode.

The latest `m1` column has no CAN ID. Its positive angle envelope is a
high-confidence match to the old motor7 channel, but the identity cannot be
proven from these CSVs alone.

## Difference from the previous WE9 wrench

Like-for-like comparison uses the same 17-cycle average and excludes the
duplicated 10-second endpoint. `Fz` and `My` are the dominant, trusted sensor
channels:

| Frequency | Fz H1 amplitude | My H1 amplitude | Fz RMS | My RMS |
|---:|---:|---:|---:|---:|
| 1 Hz | +6.68% | +4.97% | +7.16% | +9.53% |
| 2 Hz | -12.78% | -21.46% | -14.09% | -26.90% |
| 3 Hz | -33.35% | -37.27% | -33.47% | -37.07% |

After the phase alignment above, new-minus-old H1 phase is
`+0.569/+7.068/+7.191 deg` for Fz and
`-0.127/+0.252/-0.589 deg` for My at 1/2/3 Hz.

## Rebuild

```bash
cd /home/esd_wch/lsaac_lab_ws/Walking_Eagle-UniLab
UV_CACHE_DIR=/tmp/uv-cache-we9-wrench \
  uv run python scripts/data/prepare_u9_skin_wrench.py \
  --source-dir '/home/esd_wch/下载/u9/7.28带蒙皮'
```
