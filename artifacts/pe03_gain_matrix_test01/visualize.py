"""Reproduce this experiment's figures without changing source measurements."""

import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fontManager.addfont(FONT)
plt.rcParams.update(
    {
        "font.family": FontProperties(fname=FONT).get_name(),
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.unicode_minus": False,
        "figure.facecolor": "#f6f8fc",
        "savefig.facecolor": "#f6f8fc",
    }
)
COLORS = {"A": "#2563eb", "B": "#ea580c", "C": "#059669"}
NAMES = {"A": "A 基准", "B": "B 提高 Kp", "C": "C 提高 Kd"}
JOINTS = {"hip": "髋 hip", "thigh": "大腿 thigh", "calf": "小腿 calf"}
PORTS = {1: "R_hip_", 2: "R_thigh_", 3: "R_calf_", 4: "L_hip_", 5: "L_thigh_", 6: "L_calf_"}
RUNS = []
METRICS = []
QUALITY = []


def channel(data, port, field):
    return data[f"p{port}_{PORTS[port]}_{field}"]


def weighted_mean(value, time):
    return float(np.trapezoid(value, time) / (time[-1] - time[0]))


for path in sorted(ROOT.glob("*.csv")):
    config = yaml.safe_load(path.with_suffix(".yaml").read_text())
    data = np.genfromtxt(path, delimiter=",", names=True)
    assert len(data) == config["rows"] and config["completed"]
    time = (data["host_monotonic_ns"] - data["host_monotonic_ns"][0]) * 1e-9
    assert np.all(np.diff(time) > 0)
    target = config["targets"][0]
    q = channel(data, target["port_id"], "cmd_q")
    center = q[-1]
    end = time[np.flatnonzero(np.abs(np.diff(q)) > 1e-7)[-1] + 1]
    duration = config["duration_sec"]
    f0, f1 = config["start_frequency_hz"], config["end_frequency_hz"]
    slope = (f1 - f0) / duration
    fit_mask = (time > end - duration + 5) & (time < end - 0.15)

    def fit_cost(start):
        u = time[fit_mask] - start
        prediction = center + target["phase_sign"] * target["amplitude"] * np.sin(
            2 * np.pi * (f0 * u + 0.5 * slope * u**2)
        )
        return np.mean((q[fit_mask] - prediction) ** 2)

    grid = np.linspace(end - duration - 1, end - duration + 1, 1001)
    best = grid[np.argmin([fit_cost(s) for s in grid])]
    fit = minimize_scalar(fit_cost, bounds=(best - 0.005, best + 0.005), method="bounded")
    start = float(fit.x)
    fit_rmse = float(np.sqrt(fit.fun))
    assert abs(end - start - duration) < 0.05
    assert fit_rmse < 0.025 * target["amplitude"]
    relative = time - start
    mask = (relative >= 0) & (relative < duration)
    letter = path.stem.split("_")[1]
    run = dict(
        path=path,
        config=config,
        data=data,
        time=time,
        relative=relative,
        start=start,
        end=start + duration,
        mask=mask,
        letter=letter,
    )
    RUNS.append(run)
    device_dt = np.diff(data["device_sample_time_us"]) * 1e-6
    assert np.all(device_dt > 0), "Device clock discontinuity needs explicit handling"
    flags, counts = np.unique(data["command_status_flags"].astype(int), return_counts=True)
    quality = dict(
        run=path.stem,
        rows=len(data),
        recorded_hz=(len(data) - 1) / time[-1],
        duration_s=float(time[-1]),
        sweep_start_host_s=start,
        sweep_end_host_s=start + duration,
        command_chirp_fit_rmse_rad=fit_rmse,
        device_interval_median_ms=float(np.median(device_dt) * 1000),
        device_interval_max_ms=float(np.max(device_dt) * 1000),
        device_vs_host_elapsed_difference_ms=float((np.sum(device_dt) - time[-1]) * 1000),
        flags_counts={str(v): int(n) for v, n in zip(flags, counts)},
        formal_command_mean_by_joint={
            name: float(np.mean(channel(data, port, "cmd_q")[mask])) for port, name in PORTS.items()
        },
        csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        yaml_sha256=hashlib.sha256(path.with_suffix(".yaml").read_bytes()).hexdigest(),
    )
    QUALITY.append(quality)
    for target in config["targets"]:
        port = target["port_id"]
        t = time[mask]
        error = channel(data, port, "q")[mask] - channel(data, port, "cmd_q")[mask]
        assert np.all(np.isfinite(error))
        for field, expected in [("kp", config["kp"]), ("kd", config["kd"])]:
            assert np.allclose(channel(data, port, field)[mask], expected)
        METRICS.append(
            dict(
                run=path.stem,
                group=letter,
                joint_type=config["group_id"],
                joint=target["joint_name"],
                port=port,
                kp=config["kp"],
                kd=config["kd"],
                amplitude_rad=target["amplitude"],
                center_rad=float(channel(data, port, "cmd_q")[-1]),
                rmse_rad=np.sqrt(weighted_mean(error**2, t)),
                bias_rad=weighted_mean(error, t),
                peak_abs_error_rad=float(np.max(np.abs(error))),
                peak_abs_velocity_rad_s=float(np.max(np.abs(channel(data, port, "dq")[mask]))),
                peak_abs_reported_effort_nm=float(np.max(np.abs(channel(data, port, "tau")[mask]))),
            )
        )


with (OUT / "metrics.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(METRICS[0]))
    writer.writeheader()
    writer.writerows(METRICS)
(OUT / "data_quality.json").write_text(json.dumps(QUALITY, ensure_ascii=False, indent=2))

PDF = PdfPages(OUT / "pe03_gain_matrix_report.pdf")
FIGURES = []


def save(fig, name, caption):
    fig.savefig(OUT / f"{name}.png", dpi=165, bbox_inches="tight")
    PDF.savefig(fig, bbox_inches="tight")
    FIGURES.append((name, caption))
    plt.close(fig)


fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True, sharey="row", layout="constrained")
for run in RUNS:
    c, d, m = run["config"], run["data"], run["mask"]
    ax = axes[list(JOINTS).index(c["group_id"]), "ABC".index(run["letter"])]
    for i, target in enumerate(c["targets"]):
        p, sign = target["port_id"], target["phase_sign"]
        center = channel(d, p, "cmd_q")[-1]
        x = run["relative"][m]
        if i == 0:
            ax.plot(
                x,
                sign * (channel(d, p, "cmd_q")[m] - center),
                color="#9ca3af",
                lw=0.9,
                label="目标",
            )
        ax.plot(
            x,
            sign * (channel(d, p, "q")[m] - center),
            lw=0.8,
            color=["#2563eb", "#dc2626"][i],
            alpha=0.85,
            label=["左侧实际", "右侧实际"][i],
        )
    ax.set_title(
        f"{JOINTS[c['group_id']]} · {NAMES[run['letter']]}\nKp={c['kp']:g}  Kd={c['kd']:g}  A={target['amplitude']:.2f} rad"
    )
    ax.set_xlabel("正式扫频时间 / s")
    ax.set_ylabel("沿激励方向的中心偏移 / rad")
    ax.legend(fontsize=8, loc="upper right", ncol=3)
fig.suptitle(
    "PE03 三组 PD 扫频：目标与左右实际位置\n仅绘制正式 40 秒；左右按 phase_sign 镜像对齐，保留跟踪偏差",
    fontsize=17,
)
save(
    fig,
    "01_tracking_overview",
    "正式扫频总览：灰色为记录目标，蓝色为左侧，红色为右侧。左右已镜像对齐。",
)

fig, axes = plt.subplots(3, 3, figsize=(16, 11), layout="constrained")
fields = [
    ("rmse_rad", "跟踪 RMSE / rad"),
    ("bias_rad", "平均偏差（实际−目标）/ rad"),
    ("peak_abs_reported_effort_nm", "反馈力矩绝对峰值 / N·m"),
]
for row, joint in enumerate(JOINTS):
    for col, (field, label) in enumerate(fields):
        ax = axes[row, col]
        for side, offset, hatch in [("L", -0.19, None), ("R", 0.19, "//")]:
            values = [
                next(
                    v[field]
                    for v in METRICS
                    if v["joint_type"] == joint and v["group"] == g and v["joint"].startswith(side)
                )
                for g in "ABC"
            ]
            bars = ax.bar(
                np.arange(3) + offset,
                values,
                width=0.36,
                color=list(COLORS.values()),
                hatch=hatch,
                alpha=0.85,
                edgecolor="white",
                label="左侧" if side == "L" else "右侧",
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set_xticks(range(3), [NAMES[g] for g in "ABC"])
        ax.set_ylabel(label)
        ax.set_title(JOINTS[joint])
        ax.margins(y=0.2)
        ax.axhline(0, lw=0.5, color="#64748b")
        ax.legend(fontsize=8)
fig.suptitle(
    "三组 PD 的记录值对比（正式扫频段）\nRMSE / 偏差按时间加权；未补偿命令应用延迟；力矩是设备反馈值",
    fontsize=16,
)
save(
    fig,
    "02_metrics_comparison",
    "误差与反馈力矩对比。实色柱为左侧，斜线柱为右侧；偏差保留原关节坐标符号。",
)

fig, axes = plt.subplots(3, 2, figsize=(15, 11), layout="constrained", sharex=True, sharey="row")
for run in RUNS:
    c, d = run["config"], run["data"]
    u = run["relative"]
    freq = (
        c["start_frequency_hz"]
        + (c["end_frequency_hz"] - c["start_frequency_hz"]) * u / c["duration_sec"]
    )
    for side_index, target in enumerate(c["targets"]):
        p = target["port_id"]
        error = channel(d, p, "q") - channel(d, p, "cmd_q")
        centers, values = [], []
        for lo in np.arange(0.1, 5.0, 0.35):
            hi = min(lo + 0.35, 5.0)
            m = run["mask"] & (freq >= lo) & (freq < hi)
            if np.count_nonzero(m) > 5:
                centers.append((lo + hi) / 2)
                values.append(np.sqrt(weighted_mean(error[m] ** 2, run["time"][m])))
        ax = axes[list(JOINTS).index(c["group_id"]), side_index]
        ax.plot(
            centers, values, "o-", ms=4, color=COLORS[run["letter"]], label=NAMES[run["letter"]]
        )
        ax.set_title(f"{JOINTS[c['group_id']]} · {'左侧' if side_index == 0 else '右侧'}")
        ax.set_xlabel("名义激励频率 / Hz")
        ax.set_ylabel("分段跟踪 RMSE / rad")
        ax.legend()
fig.suptitle(
    "跟踪误差随扫频进程变化\n横轴由记录目标拟合出的线性 chirp 时间映射；这是分段误差，不是频率响应/带宽辨识",
    fontsize=15,
)
save(
    fig,
    "03_error_by_frequency",
    "按名义频率分段比较误差；每段约 0.35 Hz。低频周期少，不能据此判定精确带宽。",
)

for run in RUNS:
    c, d, t = run["config"], run["data"], run["time"]
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True, layout="constrained")
    for col, target in enumerate(c["targets"]):
        p = target["port_id"]
        cmd, actual = channel(d, p, "cmd_q"), channel(d, p, "q")
        axes[0, col].plot(t, cmd, color="#94a3b8", lw=1.0, label="记录目标")
        axes[0, col].plot(t, actual, color=COLORS[run["letter"]], lw=0.85, label="实际位置")
        axes[0, col].set_title(f"{target['joint_name']} / P{p}")
        axes[0, col].legend(loc="upper right")
        axes[1, col].plot(t, actual - cmd, color="#7c3aed", lw=0.8)
        axes[2, col].plot(t, channel(d, p, "dq"), color="#0891b2", lw=0.8)
        axes[3, col].plot(t, channel(d, p, "tau"), color="#d97706", lw=0.8)
        for row, ylabel in enumerate(
            ["位置 / rad", "实际−目标 / rad", "反馈速度 / rad/s", "反馈力矩 / N·m"]
        ):
            ax = axes[row, col]
            ax.axvspan(0, run["start"], color="#cbd5e1", alpha=0.35)
            ax.axvspan(run["end"], t[-1], color="#cbd5e1", alpha=0.35)
            ax.axvline(run["start"], color="#64748b", ls="--", lw=0.8)
            ax.axvline(run["end"], color="#64748b", ls="--", lw=0.8)
            ax.set_ylabel(ylabel)
        axes[3, col].set_xlabel("记录开始后的主机单调时间 / s")
    fig.suptitle(
        f"{run['path'].stem}  |  Kp={c['kp']:g}, Kd={c['kd']:g}\n完整记录；灰色区域为正式扫频之外的准备/保持段；左右保留原关节坐标",
        fontsize=16,
    )
    save(fig, run["path"].stem + "_detail", f"{run['path'].stem}：完整位置、误差、速度和反馈力矩。")
PDF.close()

notes = [
    "9 次试验、18 条被测关节曲线；每次左右同类关节同时激励，并非逐轴独立试验。",
    "三组实际幅度一致：hip ±0.35 rad、thigh ±0.22 rad、calf ±0.41 rad；非被测轴保持增益为 Kp=100、Kd=5。",
    "CSV 无试验阶段字段。以末次目标变化定位结束附近，再将记录目标拟合为 YAML 指定的 0.1–5 Hz、40 秒线性 chirp；拟合质量与切段时间保存于 data_quality.json。",
    "正式段左右中心由记录目标末值读取，不假定为机器人默认站姿。同一关节类型的三组非被测轴目标一致；calf 试验中两侧 thigh 目标均为 0 rad。不同关节类型的台架姿态不同；每组只有一次记录，不据此确定差异的统计显著性。",
    "记录平均约 157–159 行/秒，不等同于电机控制频率。状态序号跳变只能说明日志未覆盖所有源状态，不能单凭它认定 USB 丢包。",
    "记录目标缺少与实际发送 wire sequence 的逐条映射；误差为同一 CSV 行的实际−记录目标，未补偿应用延迟，不能据此分离纯机械响应与通信延迟。",
    "command_status_flags 包含 0、1、4；未核实本次固件的位定义，按原值统计，不将 4 擅自解释为饱和或故障。温度均 unavailable。",
    "反馈力矩不是独立力矩传感器的校准值；本报告不判定限矩、不拟合惯量，也不据单次 RMSE 宣布某组适合部署。",
]
table = "<table><thead><tr><th>试验</th><th>关节</th><th>Kp / Kd</th><th>中心 / 幅度(rad)</th><th>RMSE(rad)</th><th>偏差(rad)</th><th>速度峰值(rad/s)</th><th>反馈力矩峰值(N·m)</th></tr></thead><tbody>"
for v in METRICS:
    table += f"<tr><td>{v['run']}</td><td>{v['joint']}</td><td>{v['kp']} / {v['kd']}</td><td>{v['center_rad']:.3f} / {v['amplitude_rad']:.2f}</td><td>{v['rmse_rad']:.4f}</td><td>{v['bias_rad']:.4f}</td><td>{v['peak_abs_velocity_rad_s']:.3f}</td><td>{v['peak_abs_reported_effort_nm']:.3f}</td></tr>"
table += "</tbody></table>"
sections = []
for index, (name, caption) in enumerate(FIGURES):
    encoded = base64.b64encode((OUT / f"{name}.png").read_bytes()).decode()
    body = f'<p>{html.escape(caption)}</p><img loading="lazy" src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">'
    sections.append(
        f"<section>{body}</section>"
        if index < 3
        else f"<details><summary>{html.escape(caption)}</summary>{body}</details>"
    )
document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PE03 扫频可视化</title><style>
body{font-family:system-ui,sans-serif;background:#f6f8fc;color:#172033;margin:0 auto;padding:32px;max-width:1550px;line-height:1.7}h1{font-size:30px}section,details,.card{background:white;padding:20px;border-radius:12px;margin:20px 0;border:1px solid #e2e8f0}img{width:100%;height:auto}summary{cursor:pointer;font-weight:600}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:8px;border-bottom:1px solid #e2e8f0}th{background:#eff6ff}.scroll{overflow:auto}li{margin:6px 0}</style>
<h1>PE03 / PE05 实机扫频数据可视化</h1><p>pe03_gain_matrix_test01 · A 基准 / B 提高 Kp / C 提高 Kd · 9 次试验</p>"""
document += (
    '<div class="card"><b>阅读与分析范围</b><ul>'
    + "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    + "</ul></div>"
)
document += (
    "".join(sections[:3])
    + '<div class="card scroll"><h2>正式段数值汇总</h2>'
    + table
    + "</div>"
    + "".join(sections[3:])
)
document += "</html>"
(OUT / "index.html").write_text(document)
print(
    f"Created {len(FIGURES)} PNG figures, PDF, self-contained HTML, metrics CSV and quality JSON in {OUT}"
)
for v in METRICS:
    print(
        v["group"],
        v["joint"],
        "RMSE",
        round(v["rmse_rad"], 4),
        "bias",
        round(v["bias_rad"], 4),
        "effort peak",
        round(v["peak_abs_reported_effort_nm"], 3),
    )
