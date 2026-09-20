# WE11 PACE offline fitting

本目录只负责离线数据导入、MuJoCo PACE 参数拟合、Kp/Kd 搜索和报告生成。
真机 CAN 控制、手柄确认、扫频命令和原始 CSV 采集属于 Deploy 仓库
`Walking_Eagle-Deploy_final` 分支，不在此目录复制硬件代码。

## 职责边界

| 阶段 | 所属仓库 | 输出 |
| --- | --- | --- |
| 真机扫频采集 | Deploy | 原始 CSV、`SHA256SUMS` |
| 坐标转换与重采样 | UniLab | `chirp_data.pt`、manifest、转换后 CSV |
| PACE 动力学拟合 | UniLab | `pace_best_params.json`、候选表、Bode 报告 |
| Kp/Kd 拟合 | UniLab | `best_kp_kd.json`、候选表、回放曲线 |
| 训练参数采用 | UniLab | 经人工复核后的 WE11 PACE JSON、training MJCF 和 task PD/delay |

UniLab 工具不 import Deploy 源码。两仓库之间的唯一数据接口是原始 CSV 及其
SHA-256；`--csv-dir` 可以指向 Deploy checkout，也可以指向校验后复制的数据目录。

## 依赖

先完成仓库根目录的 Conda 环境安装，再安装本目录额外依赖：

```bash
python -m pip install -r scripts/pace/requirements.txt
```

所有命令都从 UniLab 仓库根目录运行。

## 2026-09-10 ESD-Link sweep flow

Current ESD-Link CSV files are imported with a dedicated entry point.  The raw
CSV files stay in `../temp`; UniLab writes only derived artifacts, reports, and
candidate parameters.

```bash
python -u scripts/pace/import_esd_link_sweeps.py \
  --csv-dir ../temp \
  --out-root outputs/we11_esd_link_sweep_import \
  --run-name current_sweeps_20260910
```

The importer reads `host_monotonic_ns` as the main clock, keeps device time and
sequence counters for quality reports, accepts `unavailable` temperature fields,
selects the final 40 s formal chirp, converts policy coordinates to MuJoCo
coordinates, unwraps wheel position, and resamples every source to 200 Hz.  It
creates these fit-compatible truth directories:

- `thigh_kp2_kd0p1__calf_kp8_kd0p8`: Group A, primary leg PACE fit.
- `thigh_kp4_kd0p2__calf_kp4_kd0p2`: Group B, cross-gain validation only.
- `wheel_kd0p05`, `wheel_kd0p1`, `wheel_kd0p2`: independent wheel fits.

The ESD-Link coordinate contract is:

- `q_mujoco = [0.8, -1.6, 0, 0.8, -1.6, 0] + p_q`.
- `qd_mujoco = p_dq`.
- Position commands use the same MuJoCo default-angle offset.
- Wheel positions are unwrapped across `[-6.28, 6.28]`, then made relative to
  the first formal sample.

Leg PACE uses Group A and validates on Group B:

```bash
TRUTH_RUN_DIR=outputs/we11_esd_link_sweep_import/<run>/thigh_kp2_kd0p1__calf_kp8_kd0p8

python -u scripts/pace/fit_mujoco_pace_params.py \
  --truth-run-dir "$TRUTH_RUN_DIR" \
  --out-root outputs/we11_pace_fit \
  --run-name esd_groupA_full \
  --model src/unilab/assets/robots/dr002/we11/we11.xml \
  --sim-hz 400 \
  --control-hz 200 \
  --require-rk4 \
  --delay-semantics command \
  --delay-values 0,1,2,3,4,5,6,7,8,9,10,11,12 \
  --fixture-mode high-impedance \
  --enable-gravity \
  --initial-joint-qpos 0.8 -1.6 0 0.8 -1.6 0 \
  --initial-joint-qvel 0 0 0 0 0 0 \
  --kp 2 8 0 2 8 0 \
  --kd 0.1 0.8 0.05 0.1 0.8 0.05 \
  --time-score-freq-range 0.5 4.5 \
  --bode-freq-range 0.5 4.5 \
  --population-size 64 \
  --max-generations 40 \
  --epsilon 0.01 \
  --workers 32

python -u scripts/pace/validate_mujoco_pace_params.py \
  --pace-params outputs/we11_pace_fit/<fit-run>/pace_best_params.json \
  --truth-run-dir outputs/we11_esd_link_sweep_import/<run>/thigh_kp4_kd0p2__calf_kp4_kd0p2 \
  --out-root outputs/we11_pace_validation \
  --run-name esd_groupB_validation \
  --model src/unilab/assets/robots/dr002/we11/we11.xml
```

After Group B passes, freeze the PACE artifact and search symmetric leg gains:

```bash
python -u scripts/pace/sweep_we11_paired_leg_pd.py \
  --pace-params outputs/we11_pace_fit/<fit-run>/pace_best_params.json \
  --truth-run-dir "$TRUTH_RUN_DIR" \
  --model src/unilab/assets/robots/dr002/we11/we11.xml \
  --groups thigh calf \
  --out-root outputs/we11_paired_kp_kd_fit \
  --run-name esd_groupA_full \
  --workers 32
```

Fit each wheel Kd directory independently.  Do not merge the three Kd sweeps
into one optimizer target.

```bash
WHEEL_RUN=outputs/we11_esd_link_sweep_import/<run>

for KD_DIR in wheel_kd0p05 wheel_kd0p1 wheel_kd0p2; do
  python -u scripts/pace/fit_mujoco_pace_params.py \
    --truth-run-dir "$WHEEL_RUN/$KD_DIR" \
    --out-root outputs/we11_wheel_pace_fit \
    --run-name "esd_${KD_DIR}_full" \
    --model src/unilab/assets/robots/dr002/we11/we11.xml \
    --control-mode mixed \
    --source-joints active \
    --fit-joints active \
    --sim-hz 400 \
    --control-hz 200 \
    --require-rk4 \
    --delay-semantics command \
    --delay-values 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --fixture-mode high-impedance \
    --enable-gravity \
    --initial-joint-qpos 0.8 -1.6 0 0.8 -1.6 0 \
    --initial-joint-qvel 0 0 0 0 0 0 \
    --kp 1 1 0 1 1 0 \
    --kd 0.1 0.1 0.1 0.1 0.1 0.1 \
    --time-score-freq-range 0.5 4.5 \
    --bode-freq-range 0.5 4.5 \
    --population-size 64 \
    --max-generations 40 \
    --epsilon 0.01 \
    --workers 32
done

python -u scripts/pace/summarize_esd_link_wheel_fits.py \
  outputs/we11_wheel_pace_fit/<kd005-run> \
  outputs/we11_wheel_pace_fit/<kd01-run> \
  outputs/we11_wheel_pace_fit/<kd02-run> \
  --out-dir outputs/we11_wheel_pace_fit/<summary-run>
```

This flow uses the source manifest's per-source fixture gains.  For example,
thigh sweeps hold the non-swept calf joints at the measured `20/1`, calf sweeps
hold thighs at `40/2`, and wheel sweeps hold leg joints at `1/0.1`.  The old
uniform `1/0.1` fixture assumption is not valid for these ESD-Link files.

The scripts only generate reports and candidate JSONs.  They do not modify
`we11_pace_params.json` or training task/control configs.

## Legacy 2026-07-30 CSV flow

The remaining sections describe the earlier July 2026 sweep files and are kept
for reproducibility of archived results.

## 1. 校验 Deploy 原始数据

在 Deploy 仓库根目录运行：

```bash
cd sweep_results_20260730
sha256sum -c SHA256SUMS
cd ..
```

不要改写原始 CSV。正式段从最后一次 `timestamp` 回退处开始，导入器会检查该
合同并在 manifest 中记录源 CSV 的 SHA-256。

## 2. 导入 1/4 与 2/5 腿部数据

下面的坐标合同来自 2026-07-30 新扫频零点。若硬件零点、运行默认角或 MJCF
关节方向发生变化，必须重新测量这些参数，不能沿用本例。

```bash
python -u scripts/pace/import_real_sweep_csv_for_pd_fit.py \
  --input-format july30-paired \
  --csv-dir ../Walking_Eagle-Deploy_final/sweep_results_20260730 \
  --out-root outputs/we11_real_sweep_import \
  --run-name we11_july30_paired_pd_groups \
  --mujoco-qpos-at-policy-center 1.1998 -2.25662 0 1.1998 -2.25662 0 \
  --policy-delta-to-mujoco-sign 1 1 1 1 1 1 \
  --runtime-joint-default -0.92 1.07 0 -0.92 1.07 0
```

导入结果会生成两组 paired truth：

- Group A：大腿 `Kp=2,Kd=0.1`；小腿 `Kp=8,Kd=0.8`。
- Group B：大腿和小腿均为 `Kp=4,Kd=0.2`。

## 3. 拟合腿部 PACE 参数

将 `TRUTH_RUN_DIR` 指向第 2 步生成的 Group A 目录：

```bash
TRUTH_RUN_DIR=outputs/we11_real_sweep_import/<run>/thigh_kp2_kd0p1__calf_kp8_kd0p8

python -u scripts/pace/fit_mujoco_pace_params.py \
  --truth-run-dir "$TRUTH_RUN_DIR" \
  --out-root outputs/we11_pace_fit \
  --run-name we11_groupA_05_45_delay4_rk4 \
  --model src/unilab/assets/robots/dr002/we11/we11.xml \
  --sim-hz 400 \
  --control-hz 200 \
  --require-rk4 \
  --delay-semantics command \
  --delay-values 4 \
  --fixture-mode high-impedance \
  --fixture-hold-kp 1 1 0 1 1 0 \
  --fixture-hold-kd 0.1 0.1 0 0.1 0.1 0 \
  --enable-gravity \
  --initial-joint-qpos 1.1998 -2.25662 0 1.1998 -2.25662 0 \
  --initial-joint-qvel 0 0 0 0 0 0 \
  --kp 2 8 0 2 8 0 \
  --kd 0.1 0.8 0.05 0.1 0.8 0.05 \
  --time-score-freq-range 0.5 4.5 \
  --bode-freq-range 0.5 4.5 \
  --workers 16
```

核心结果是 `<fit-run>/pace_best_params.json`。该文件同时记录模型 SHA、原始数据
manifest、频率窗口、控制频率、delay 语义和优化结果，不能只抄最终三个数字。

## 4. 在冻结的 PACE 参数上拟合 Kp/Kd

```bash
python -u scripts/pace/sweep_we11_paired_leg_pd.py \
  --pace-params outputs/we11_pace_fit/<fit-run>/pace_best_params.json \
  --truth-run-dir "$TRUTH_RUN_DIR" \
  --model src/unilab/assets/robots/dr002/we11/we11.xml \
  --groups thigh calf \
  --out-root outputs/we11_paired_kp_kd_fit \
  --run-name we11_groupA_paired_pd_rk4 \
  --workers 16
```

该步骤只搜索对称的腿部 Kp/Kd，不重新优化 PACE 动力学，也不改变共享 command
delay。模型 SHA 必须与 PACE artifact 一致，否则脚本拒绝运行。

## 5. 导入并分别拟合轮子三组 Kd 数据

```bash
python -u scripts/pace/import_we11_wheel_sweeps.py \
  --csv-dir ../Walking_Eagle-Deploy_final/sweep_results_20260730 \
  --out-root outputs/we11_real_wheel_sweep_import \
  --run-name we11_wheel_kd005_01_02_velocity_pm5
```

导入器为 `Kd=0.05/0.1/0.2` 各生成一个独立目录，每个目录包含左右轮两个 source。
按用户要求独立辨识时，应对三个目录分别调用
`fit_mujoco_pace_params.py`，不要把三个 Kd 混成一组共享参数：

```bash
WHEEL_RUN=outputs/we11_real_wheel_sweep_import/<run>

for KD_DIR in wheel_kd0p05 wheel_kd0p1 wheel_kd0p2; do
  python -u scripts/pace/fit_mujoco_pace_params.py \
    --truth-run-dir "$WHEEL_RUN/$KD_DIR" \
    --out-root outputs/we11_wheel_pace_fit \
    --run-name "we11_${KD_DIR}_independent_rk4" \
    --model src/unilab/assets/robots/dr002/we11/we11.xml \
    --control-mode mixed \
    --source-joints active \
    --fit-joints active \
    --sim-hz 400 \
    --control-hz 200 \
    --require-rk4 \
    --delay-semantics command \
    --delay-values 4 \
    --fixture-mode equality \
    --enable-gravity \
    --time-score-freq-range 0.5 4.5 \
    --bode-freq-range 0.5 4.5 \
    --workers 16
done
```

轮子使用 `mixed` 速度 D-only 控制。`fixture-mode equality` 会释放当前 source
轮子，并把其余五个关节逐物理步精确投影回初始状态；这才符合台架采集时的
“其他关节 fix”语义。

## 6. 采用拟合结果

1. 先检查时域曲线、Bode 幅相、边界命中、力矩饱和和左右一致性。
2. PACE 的 armature/damping/frictionloss 写入
   `src/unilab/assets/robots/dr002/we11/we11_pace_params.json`，由训练专用 MJCF 注入；
   基础 URDF/MJCF 的关节动力学保持为零。
3. Kp/Kd 和 command-delay 写入 WE11 task/control config，不烘焙到 URDF。
4. 修改后运行 WE11 资产/训练合同测试，再开始新训练。

## 文件说明

- `import_real_sweep_csv_for_pd_fit.py`：腿部 CSV 坐标转换、正式段切分和 200 Hz 重采样。
- `import_we11_wheel_sweeps.py`：左右轮 ±5 rad/s 数据导入，三组 Kd 独立归档。
- `fit_mujoco_pace_params.py`：CMA-ES PACE 参数拟合与报告。
- `sweep_we11_paired_leg_pd.py`：冻结 PACE 后的双侧等权 Kp/Kd 搜索。
- `run_mujoco_dr002_sim.py`：离线 hold/sine/chirp/replay 调试。
- `chirp_frequency_response.py`、`bode_reference.py`：频率窗、激励有效性和 Bode 指标。
- `mujoco_dr002_common.py`：数据、模型、参数和 delay 的公共合同。
