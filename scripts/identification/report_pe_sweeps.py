"""Validate and visualize the offline PE sweep candidate; no hardware access."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from fit_pe_sweeps import BASE, GROUPS, JOINTS, ROOT, Trial, digest, metrics, native_library
from matplotlib.backends.backend_pdf import PdfPages


def audit_warnings(data, out, quality):
    logs = sorted((out / "provenance/upper/scripts/sweeps/logs/launcher").glob("*.log"))
    result = []
    for q, log in zip(quality, logs, strict=True):
        raw = np.genfromtxt(data / (q["run"] + ".csv"), delimiter=",", names=True)
        host = (raw["host_monotonic_ns"] - raw["host_monotonic_ns"][0]) * 1e-9
        warnings = []
        for line in log.read_text().splitlines():
            if "Command Arbiter rejected" not in line:
                continue
            match = re.search(r"\[WARN\] \[([0-9.]+)\]", line)
            if match:
                epoch = float(match.group(1))
                nearest = np.argmin(abs(raw["wall_time"] - epoch))
                warnings.append(
                    {
                        "wall_time": epoch,
                        "formal_time_s": float(host[nearest] - q["sweep_start_host_s"]),
                        "nearest_record_wall_delta_s": float(raw["wall_time"][nearest] - epoch),
                    }
                )
        result.append(
            {
                "run": q["run"],
                "log_sha256": digest(log),
                "warnings": warnings,
                "interpretation": "Throttled warnings; counts are lower bounds. Rejection reason is composite, not proven stale-source only.",
            }
        )
    (out / "rejection_audit.json").write_text(json.dumps(result, indent=2))
    return result


def sensitivity(trials, par, out):
    result = {}
    # This is a fixed-parameter sensitivity study, not refitting or tuning on C.
    for trial in trials:
        ref = trial.replay(par, contacts=False)
        cases = {
            "dt_1ms_control_2ms": trial.replay(par, dt=0.001, control_dt=0.002, contacts=False),
            "dt_2p5ms_control_2p5ms": trial.replay(
                par, dt=0.0025, control_dt=0.0025, contacts=False
            ),
            "net_torque_clip": trial.replay(par, legacy=True, contacts=False),
        }
        for shift in (-0.004, 0.004):
            changed = par.copy()
            changed[9:] = np.maximum(0, changed[9:] + shift)
            cases[f"delay_shift_{shift:+.3f}s"] = trial.replay(changed, contacts=False)
        current = BASE.copy()
        current[9:] = 0.02
        cases["pe03_current_timing_and_delay_approximation"] = trial.replay(
            current, dt=0.0025, control_dt=0.0025, legacy=True, contacts=False
        )
        result[trial.name] = {}
        for key, value in cases.items():
            result[trial.name][key] = metrics(trial, value)
            result[trial.name][key]["selected_change_from_fit_rms_rad"] = float(
                np.sqrt(np.mean(((value[:, :6] - ref[:, :6])[trial.mask][:, trial.selected]) ** 2))
            )
        print("SENSITIVITY", trial.name, flush=True)
    (out / "sensitivity.json").write_text(json.dumps(result, indent=2))
    # Local derivative columns use meaningful perturbations, not formal CIs.
    local = {}
    for group in range(3):
        train = [t for t in trials[:6] if t.group == group]
        columns = []
        steps = [max(par[group] * 0.1, 0.001), 0.02, 0.02, 0.004]
        for index, step in zip([group, 3 + group, 6 + group, 9 + group], steps, strict=True):
            minus, plus = par.copy(), par.copy()
            minus[index] = max(0 if index >= 3 else 1e-5, minus[index] - step)
            plus[index] += step
            col = []
            for trial in train:
                low = trial.replay(minus, contacts=False)[:, :6]
                high = trial.replay(plus, contacts=False)[:, :6]
                col.extend(((high - low)[trial.mask][:, trial.selected] / trial.amplitude).ravel())
            columns.append(col)
        matrix = np.array(columns).T
        norms = np.linalg.norm(matrix, axis=0)
        normalized = matrix / np.maximum(norms, 1e-12)
        sv = np.linalg.svd(normalized, compute_uv=False)
        local[GROUPS[group]] = {
            "parameter_order": ["armature", "damping", "friction", "effective_delay"],
            "perturbations": steps,
            "column_rms_normalized_amplitude": np.sqrt(np.mean(matrix**2, axis=0)).tolist(),
            "normalized_column_cosines": (normalized.T @ normalized).tolist(),
            "normalized_singular_values": sv.tolist(),
            "condition_number": float(sv[0] / max(sv[-1], 1e-12)),
            "meaning": "Local one-at-a-time sensitivity; not confidence intervals. Boundary estimates need special care.",
        }
    (out / "local_identifiability.json").write_text(json.dumps(local, indent=2))
    return result, local


def harmonic_response(time, command, observed, predicted, joint):
    frequencies = np.arange(0.8, 4.81, 0.25)
    phase = 2 * np.pi * (0.1 * time + 0.5 * (4.9 / 40) * time**2)
    output = []
    for freq in frequencies:
        center = (freq - 0.1) / (4.9 / 40)
        mask = abs(time - center) < 1.5
        x = time[mask] - center
        design = np.column_stack([np.sin(phase[mask]), np.cos(phase[mask]), np.ones(mask.sum()), x])
        coeff = np.linalg.lstsq(
            design,
            np.column_stack([command[mask, joint], observed[mask, joint], predicted[mask, joint]]),
            rcond=None,
        )[0]
        phasor = coeff[0] + 1j * coeff[1]
        output.append(phasor[1:] / phasor[0])
    return frequencies, np.array(output)


def build_report(data, out, quality, scores, candidate, sens, local, warnings):
    plt.rcParams.update(
        {"font.family": "Noto Sans CJK SC", "font.size": 10, "axes.unicode_minus": False}
    )
    image_sections = []
    pdf = PdfPages(out / "replay_comparisons.pdf")

    def save(fig, name, caption):
        fig.savefig(out / name, dpi=150, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode((out / name).read_bytes()).decode()
        image_sections.append(
            f'<figure><img src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True, sharey=True)
    for group, ax in enumerate(axes):
        qs = quality[group::3]
        baseline = [scores[q["run"]]["baseline"]["selected_q_nrmse_amplitude"] for q in qs]
        fitted = [scores[q["run"]]["fit"]["selected_q_nrmse_amplitude"] for q in qs]
        ax.bar(
            np.arange(3) - 0.18, baseline, 0.36, label="辨识前参数，同一回放时序", color="#a9b5c7"
        )
        ax.bar(np.arange(3) + 0.18, fitted, 0.36, label="辨识后", color="#177e89")
        ax.axhline(0.15, linestyle="--", color="#bb5544", label="暂定 15% 验收线")
        ax.set_xticks(range(3), ["A 拟合", "B 拟合", "C 留出验证"])
        ax.set_title(GROUPS[group])
        ax.set_ylabel("位置回放 RMSE / 目标单边幅值")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    save(
        fig,
        "01_validation.png",
        "误差是仿真位置与实测位置之差，不是电机对目标的跟踪误差。C 组未用于优化。",
    )
    rows = []
    gates = []
    velocity_checks = []
    warning_exclusion = []
    for q in quality:
        name = q["run"]
        group = GROUPS.index(name.split("_")[-1])
        sel = [group, group + 3]
        bundle = np.load(out / (name + "_fit.npz"))
        baseline = np.load(out / (name + "_baseline.npz"))
        time = bundle["time"]
        observed = bundle["q"]
        predicted = bundle["replay"][:, :6]
        clean = (time >= 0.2) & (time < 40)
        events = next(w["warnings"] for w in warnings if w["run"] == name)
        for event in events:
            clean &= abs(time - event["formal_time_s"]) > 0.2
        warning_exclusion.append(
            {
                "run": name,
                "exclusion_half_window_s": 0.2,
                "remaining_samples": int(clean.sum()),
                "q_rmse_rad_without_warning_windows": np.sqrt(
                    np.mean((predicted - observed)[clean] ** 2, axis=0)
                ).tolist(),
                "dq_rmse_rad_s_without_warning_windows": np.sqrt(
                    np.mean((bundle["replay"][:, 6:12] - bundle["dq"])[clean] ** 2, axis=0)
                ).tolist(),
                "note": "Score sensitivity only; not a refit. Throttled logs cannot locate every rejection.",
            }
        )
        raw = np.genfromtxt(data / (name + ".csv"), delimiter=",", names=True)
        rt = (raw["host_monotonic_ns"] - raw["host_monotonic_ns"][0]) * 1e-9 - q[
            "sweep_start_host_s"
        ]
        command = np.column_stack(
            [
                np.interp(time, rt, raw[f"p{port}_{joint}_cmd_q"])
                for port, joint in zip([4, 5, 6, 1, 2, 3], JOINTS, strict=True)
            ]
        )
        fig, axes = plt.subplots(3, 2, figsize=(15, 9), constrained_layout=True)
        for column, joint in enumerate(sel):
            ax = axes[0, column]
            mask = (time >= 0) & (time <= 40)
            ax.plot(time[mask], command[mask, joint], color="#c8cdd3", lw=0.8, label="记录目标")
            ax.plot(time[mask], observed[mask, joint], color="#172b4d", lw=0.9, label="实测")
            ax.plot(
                time[mask],
                predicted[mask, joint],
                color="#f08b35",
                lw=0.8,
                alpha=0.8,
                label="辨识模型",
            )
            ax.set_title(f"{name} · {JOINTS[joint]}")
            ax.set_ylabel("位置 rad")
            ax.set_xlabel("扫频时间 s")
            ax.legend(fontsize=8, ncol=3)
            ax = axes[1, column]
            ax.plot(
                time[mask],
                (baseline["replay"][:, :6] - observed)[mask, joint],
                lw=0.7,
                color="#a9b5c7",
                label="辨识前",
            )
            ax.plot(
                time[mask],
                (predicted - observed)[mask, joint],
                lw=0.7,
                color="#177e89",
                label="辨识后",
            )
            ax.set_ylabel("仿真 − 实测 rad")
            ax.set_xlabel("扫频时间 s")
            ax.legend(fontsize=8)
            frequencies, response = harmonic_response(time, command, observed, predicted, joint)
            ax = axes[2, column]
            ax.plot(frequencies, 20 * np.log10(abs(response[:, 0])), "o-", label="实测幅频")
            ax.plot(frequencies, 20 * np.log10(abs(response[:, 1])), "s--", label="模型幅频")
            ax.set_xlabel("局部 chirp 频率 Hz")
            ax.set_ylabel("目标 → 位置增益 dB")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
            before = scores[name]["baseline"]["q_rmse_rad"][joint]
            after = scores[name]["fit"]["q_rmse_rad"][joint]
            rows.append(
                {
                    "run": name,
                    "split": "validation" if name.startswith(("07", "08", "09")) else "fit",
                    "joint": JOINTS[joint],
                    "baseline_q_rmse_rad": before,
                    "fit_q_rmse_rad": after,
                    "reduction_percent": 100 * (1 - after / before),
                    "bias_rad": scores[name]["fit"]["q_bias_rad"][joint],
                    "dq_rmse_rad_s": scores[name]["fit"]["dq_rmse_rad_s"][joint],
                }
            )
        save(
            fig,
            name + "_comparison.png",
            "上：完整正式段回放；中：模型残差；下：3 秒滑窗谐波回归的局部增益，仅作非平稳 chirp 诊断，不等同于严格稳态频响。评分段为开始后 0.2–40 秒。",
        )
        if name.startswith(("07", "08", "09")):
            s = scores[name]
            scoring = (time >= 0.2) & (time < 40)
            measured_velocity = bundle["dq"][scoring][:, sel]
            velocity_scale = np.sqrt(
                np.mean((measured_velocity - measured_velocity.mean(axis=0)) ** 2)
            )
            velocity_error = np.sqrt(
                np.mean((bundle["replay"][scoring, 6:12][:, sel] - measured_velocity) ** 2)
            )
            held = [j for j in range(6) if j not in sel]
            velocity_checks.append(
                {
                    "run": name,
                    "rms_error_rad_s": float(velocity_error),
                    "measured_velocity_std_rad_s": float(velocity_scale),
                    "relative_rmse": float(velocity_error / max(velocity_scale, 1e-12)),
                }
            )
            gates.append(
                {
                    "run": name,
                    "q_nrmse_below_15pct": s["fit"]["selected_q_nrmse_amplitude"] <= 0.15,
                    "both_bias_below_0p03rad": bool(
                        np.max(abs(np.array(s["fit"]["q_bias_rad"])[sel])) <= 0.03
                    ),
                    "improves_by_30pct": s["fit"]["selected_q_nrmse_amplitude"]
                    <= 0.7 * s["baseline"]["selected_q_nrmse_amplitude"],
                    "no_contact": s["fit"]["max_contacts"] == 0,
                    "held_joint_rmse_below_0p03rad": bool(
                        np.max(np.array(s["fit"]["q_rmse_rad"])[held]) <= 0.03
                    ),
                    "velocity_relative_rmse_below_25pct": bool(
                        velocity_error / max(velocity_scale, 1e-12) <= 0.25
                    ),
                }
            )
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), constrained_layout=True, sharey="row")
    for group, q in enumerate(quality[6:]):
        bundle = np.load(out / (q["run"] + "_fit.npz"))
        time = bundle["time"]
        zoom = (time >= 32) & (time < 40)
        for side, joint in enumerate([group, group + 3]):
            ax = axes[group, side]
            ax.plot(
                time[zoom], bundle["dq"][zoom, joint], label="实测编码器 dq", lw=1, color="#172b4d"
            )
            ax.plot(
                time[zoom],
                bundle["replay"][zoom, 6 + joint],
                label="辨识模型 qvel",
                lw=1,
                color="#f08b35",
            )
            ax.set_title(f"C 组 · {JOINTS[joint]} · 高频段速度对照")
            ax.set_xlabel("扫频时间 s")
            ax.set_ylabel("角速度 rad/s")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(alpha=0.2)
    save(
        fig,
        "11_C_velocity.png",
        "C 组 32–40 秒速度放大图，未用于拟合。位置拟合较好不代表速度残差已经足够小；验收速度 RMS 使用完整 0.2–40 秒。",
    )
    pdf.close()
    (out / "warning_exclusion_sensitivity.json").write_text(json.dumps(warning_exclusion, indent=2))
    with (out / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = all(all(v for k, v in g.items() if k != "run") for g in gates)
    assessment = {
        "provisional_kinematic_gate_passed": passed,
        "gates": gates,
        "velocity_details": velocity_checks,
        "promote_training": False,
        "note": "Passing these provisional sweep gates does not validate loaded locomotion or timing identifiability. Boundary and timing sensitivities must also be reviewed.",
    }
    boundary_notes = []
    for key, threshold in (
        ("passive_damping_nm_s_rad", 0.001),
        ("coulomb_frictionloss_nm", 0.001),
    ):
        for group, value in zip(GROUPS, candidate[key], strict=True):
            if value <= threshold:
                boundary_notes.append(
                    {
                        "group": group,
                        "parameter": key,
                        "value": value,
                        "interpretation": "Near the nonnegative boundary; does not establish exact physical zero.",
                    }
                )
    candidate["sweep_kinematic_gate_passed"] = passed
    candidate["boundary_estimates"] = boundary_notes
    candidate["pe05_usage"] = (
        "Prior only: matching current assets, no PE05 measurements in this dataset."
    )
    (out / "candidate.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False))
    (out / "assessment.json").write_text(json.dumps(assessment, indent=2))
    parameter_rows = "".join(
        f"<tr><td>{g}</td><td>{candidate['armature_total_kg_m2'][i]:.6g}</td><td>{candidate['passive_damping_nm_s_rad'][i]:.6g}</td><td>{candidate['coulomb_frictionloss_nm'][i]:.6g}</td><td>{candidate['effective_delay_s'][i] * 1000:.3f}</td></tr>"
        for i, g in enumerate(GROUPS)
    )
    metric_rows = "".join(
        f"<tr><td>{r['run']}</td><td>{r['joint']}</td><td>{r['baseline_q_rmse_rad']:.4f}</td><td>{r['fit_q_rmse_rad']:.4f}</td><td>{r['reduction_percent']:.1f}%</td><td>{r['bias_rad']:.4f}</td></tr>"
        for r in rows
    )
    timing_rows = "".join(
        f"<tr><td>{name}</td><td>{entry['dt_1ms_control_2ms']['selected_change_from_fit_rms_rad']:.4f}</td><td>{entry['dt_2p5ms_control_2p5ms']['selected_change_from_fit_rms_rad']:.4f}</td><td>{entry['delay_shift_+0.004s']['selected_change_from_fit_rms_rad']:.4f}</td></tr>"
        for name, entry in sens.items()
        if name.startswith(("07", "08", "09"))
    )
    gate_rows = "".join(
        f"<tr><td>{g['run']}</td><td>{100 * scores[g['run']]['fit']['selected_q_nrmse_amplitude']:.2f}%</td>"
        f"<td>{v['rms_error_rad_s']:.3f}</td><td>{100 * v['relative_rmse']:.2f}%</td>"
        f"<td>{'通过' if all(value for key, value in g.items() if key != 'run') else '未全部通过'}</td></tr>"
        for g, v in zip(gates, velocity_checks, strict=True)
    )
    text = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>PE03 现有扫频辨识</title>
<style>body{{max-width:1200px;margin:35px auto;padding:0 25px;font-family:system-ui,sans-serif;line-height:1.7;color:#172b4d}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}figure{{margin:32px 0}}.note{{padding:18px;background:#eef5f6;border-left:4px solid #177e89}}code{{background:#f0f2f4;padding:2px 5px}}</style>
<h1>PE03 扫频辨识 · 现有 9 次数据</h1><p>数据目录：{html.escape(str(data))}。未补测，未改远端代码，未运行实机动作。A/B 共 6 次拟合；C 共 3 次整段留出。采集数据与原可视化哈希逐项核对通过。</p>
<div class="note">本轮生成候选执行器参数与离线验证结果。暂定运动学验收线：C 组位置 NRMSE ≤ 15%，左右偏差 ≤ 0.03 rad，较原参数改善 ≥ 30%，保持关节位置 RMS ≤ 0.03 rad，速度 RMS / 实测速度标准差 ≤ 25%，无模型碰撞。此次结果：<b>{"通过" if passed else "未全部通过"}</b>。训练配置未自动覆盖。</div>
<table><tr><th>C 组</th><th>位置 NRMSE</th><th>速度 RMS 误差 rad/s</th><th>相对速度误差</th><th>暂定验收</th></tr>{gate_rows}</table>
<h2>数据和模型约定</h2><p>固定竖直机身、足端悬空，左右同类型关节同步扫频，其余关节 Kp/Kd=100/5。calf 扫频时 thigh 目标为 0。使用记录的六关节五元组命令与完整准备过程；不另作左右符号翻转。CAD 质量、连杆惯量、编码器零位和力矩倍率固定，只辨识每类关节共用的 armature、被动阻尼、摩擦和等效延迟。2 ms 物理步长、2 ms 固件增益缩放周期，电机内部 PD 用连续理想响应近似。</p>
<p>优化使用有界 Powell 搜索，按关节组进行两轮细化；每次都回放完整六关节模型。损失是 A/B 被扫关节的位置误差平方，按目标单边幅值归一化，评分采样 50 Hz、正式段 0.2–40 秒；速度不参与拟合，单独用于验证。它是条件性局部最优候选，不是唯一物理参数的证明。实际 CSV 中位间隔约 6 ms，记录频率约 158 Hz，不能把它当成请求的 200 Hz 或下位机的 500 Hz。</p>
<p>固件以 /ssd/esd-slave/Wzh/esd-h7-slave_V2.1、提交 4b2ce17 为依据。力矩缩放为 min(1,限值/(|P|+|D|+|FF|))。CSV 同行目标是刚发布的命令，不是电机已执行的命令；device_sample_time_us 来自 IMU 更新时刻。每次试验均发现命令仲裁拒绝警告，节流日志不能还原拒绝总数和被拒命令。等效延迟同时吸收采集、传输、驱动和模型误差，不能当作纯通信延迟。</p>
<p>为加速优化，仅优化阶段跳过接触求解；全部正式评分使用完整接触模型复核。固定根部后显式排除原本由自动父子过滤排除的机身–hip 直接相邻碰撞，其他自碰撞与关节限位保留。现有姿态稀疏审计无其他接触；每次完整回放另记录接触计数。</p>
<h2>候选参数</h2><table><tr><th>关节组</th><th>总 armature kg·m²</th><th>被动阻尼 N·m·s/rad</th><th>库仑摩擦 N·m</th><th>等效延迟 ms</th></tr>{parameter_rows}</table>
<p>armature 是在 CAD 连杆惯量之外的关节侧等效附加惯量，替换 XML 原 armature，不能重复相加。这里的被动阻尼不是 Kd；Kp/Kd 是每次试验的已知输入，未作为未知量拟合。摩擦采用 MuJoCo frictionloss。近零或触边估计只说明这批数据未把该项分离出来，不代表物理量确实为零。</p>
{image_sections[0]}
<h2>逐关节验证</h2><table><tr><th>试验</th><th>关节</th><th>原模型 RMSE rad</th><th>候选 RMSE rad</th><th>改善</th><th>平均偏差 rad</th></tr>{metric_rows}</table>
<h2>时序敏感性</h2><p>下表是固定候选参数后，相对 2 ms 回放的轨迹变化 RMS。1 ms 测试保持 2 ms 固件更新；2.5 ms 测试同步改为 400 Hz。此处只作敏感性分析，没有用 C 组重新调参。</p><table><tr><th>C 组</th><th>1 ms 步长 rad</th><th>400 Hz 回放 rad</th><th>延迟增加 4 ms rad</th></tr>{timing_rows}</table>
<h2>边界与训练接入判断</h2><p>参数局部可辨识性在 local_identifiability.json 中；参数扰动列的相关性反映惯量、阻尼、摩擦与延迟的混淆。它不是统计置信区间。这批同步双侧、单姿态、0.1–5 Hz 数据不能独立校准力矩常数、全部连杆惯性、全姿态耦合、足地接触或带载行走。</p>
<p>PE03 与 PE05 当前资产的质量、惯量、关节轴、限位、armature 和 damping 数值一致（pe05_asset_comparison.json），因此候选可作为 PE05 的参数先验；这不构成 PE05 实机辨识通过。训练当前为 400 Hz，PE03 延迟配置 20 ms，而本轮是 500 Hz 回放下的等效延迟；不能把候选延迟机械叠加到现有延迟配置。训练中的速度观测定义、动作缩放、限位和固件控制语义仍需一起核对。</p>
<p>当前 PE03/PE05 配置 dof_vel_use_pos_diff=true，训练的 PD 与观测会使用位置差分速度，而部署读编码器 dq。本轮回放使用 MuJoCo qvel；400 Hz 敏感性对比只改变步长、更新周期和指定延迟，不等同于完整训练后端。速度定义差异已确认，但本次未修改生产后端或策略。</p>
<p>不补测条件下，优先使用本报告中通过跨增益验证、对时序扰动稳定的参数作为离线训练候选；对触边和强耦合参数保留不确定范围。此报告不据此更改部署 Kp/Kd，也不宣称仿真与实机行走已一致。</p>
<h2>回放图</h2>{"".join(image_sections[1:])}
<h2>可复现文件</h2><p>candidate.yaml、checkpoint.json、audit.json、metrics.csv、metrics.json、sensitivity.json、local_identifiability.json、assessment.json、rejection_audit.json、fixture.mjb 及 provenance/。运行脚本位于 scripts/identification/，fit.log 保留优化轨迹。原始 CSV/YAML 未修改。</p></html>"""
    (out / "index.html").write_text(text)
    return assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/pe03_gain_matrix_test01")
    parser.add_argument("--reuse-sensitivity", action="store_true")
    args = parser.parse_args()
    out = args.data / "identification"
    quality = json.loads((args.data / "visualizations/data_quality.json").read_text())
    scores = json.loads((out / "metrics.json").read_text())
    candidate = yaml.safe_load((out / "candidate.yaml").read_text())
    par = np.array(json.loads((out / "checkpoint.json").read_text())["parameters"])
    warnings = audit_warnings(args.data, out, quality)
    if args.reuse_sensitivity:
        sens = json.loads((out / "sensitivity.json").read_text())
        local = json.loads((out / "local_identifiability.json").read_text())
    else:
        dll = native_library(out)
        trials = [Trial(args.data / (q["run"] + ".csv"), q, dll, out) for q in quality]
        sens, local = sensitivity(trials, par, out)
        for trial in trials:
            trial.close()
    result = build_report(args.data, out, quality, scores, candidate, sens, local, warnings)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
