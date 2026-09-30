"""Create an offline, standalone visual report for the six-joint PE03 fit."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scripts.identification.fit_pe_sweeps import GROUPS, JOINTS, ROOT, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/pe03_gain_matrix_test01/identification/independent_lr",
    )
    out = parser.parse_args().out
    report = json.loads((out / "comparison.json").read_text())
    par = np.array(report["independent_parameters"]).reshape(4, 6)
    tied = np.array(report["tied_parameters"]).reshape(4, 6)
    plt.rcParams.update(
        {"font.family": "Noto Sans CJK SC", "font.size": 10, "axes.unicode_minus": False}
    )
    rows = []
    for run in report["held_out_runs"]:
        group = GROUPS.index(run.rsplit("_", 1)[1])
        score = report["metrics"][run]
        for joint in (group, group + 3):
            old, new = score["tied"], score["independent"]
            rows.append(
                dict(
                    joint=JOINTS[joint],
                    armature_kg_m2=par[0, joint],
                    damping_nm_s_rad=par[1, joint],
                    frictionloss_nm=par[2, joint],
                    delay_ms=1000 * par[3, joint],
                    tied_q_rmse_rad=old["q_rmse_rad"][joint],
                    independent_q_rmse_rad=new["q_rmse_rad"][joint],
                    q_rmse_improvement_percent=100
                    * (1 - new["q_rmse_rad"][joint] / old["q_rmse_rad"][joint]),
                    tied_dq_rmse_rad_s=old["dq_rmse_rad_s"][joint],
                    independent_dq_rmse_rad_s=new["dq_rmse_rad_s"][joint],
                    independent_q_bias_rad=new["q_bias_rad"][joint],
                )
            )
    with (out / "joint_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures = []
    with PdfPages(out / "independent_lr_report.pdf") as pdf:

        def save(fig, name, caption):
            fig.tight_layout()
            fig.savefig(out / (name + ".png"), dpi=160, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            figures.append((name, caption))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
        for block, (ax, title, unit) in enumerate(
            zip(
                axes,
                ("关节附加惯量（armature）", "被动黏性阻尼", "库仑摩擦"),
                ("kg·m²", "N·m·s/rad", "N·m"),
                strict=True,
            )
        ):
            x = np.arange(3)
            ax.bar(x - 0.18, par[block, :3], 0.35, label="左侧独立辨识", color="#167d9a")
            ax.bar(x + 0.18, par[block, 3:], 0.35, label="右侧独立辨识", color="#e58943")
            ax.scatter(
                x, tied[block, :3], marker="x", color="black", label="此前左右共用", zorder=4
            )
            ax.set(xticks=x, xticklabels=GROUPS, title=title, ylabel=unit)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8)
        save(
            fig,
            "parameters",
            "左右参数分别估计；近零阻尼或摩擦是边界拟合结果，不能解释为实物没有损耗。",
        )

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        x = np.arange(6)
        for ax, key, title in zip(
            axes,
            ("q_rmse_rad", "dq_rmse_rad_s"),
            ("C 组位置 RMSE（rad，越小越好）", "C 组速度 RMSE（rad/s，越小越好）"),
            strict=True,
        ):
            ax.bar(
                x - 0.18,
                [r["tied_" + key] for r in rows],
                0.35,
                label="左右共用参数",
                color="#8695a7",
            )
            ax.bar(
                x + 0.18,
                [r["independent_" + key] for r in rows],
                0.35,
                label="左右独立参数",
                color="#167d9a",
            )
            ax.set(xticks=x, xticklabels=[r["joint"] for r in rows], title=title)
            ax.grid(axis="y", alpha=0.2)
            ax.legend()
        save(fig, "validation", "C 组未参与本次拟合；位置与速度都检查，防止只改善拟合目标。")

        for run in report["held_out_runs"]:
            data = np.load(out / (run + ".npz"))
            group = GROUPS.index(run.rsplit("_", 1)[1])
            mask = data["mask"]
            t = data["time"]
            fig, axes = plt.subplots(3, 2, figsize=(14, 9))
            for column, joint in enumerate((group, group + 3)):
                for row, zoom in ((0, False), (1, True)):
                    selected = mask & (t > t[mask][-1] - 2) if zoom else mask
                    ax = axes[row, column]
                    ax.plot(
                        t[selected],
                        data["measured"][selected, joint],
                        label="实测",
                        color="black",
                        lw=1.2,
                    )
                    ax.plot(
                        t[selected],
                        data["tied"][selected, joint],
                        label="左右共用参数",
                        color="#8695a7",
                        lw=1,
                        alpha=0.8,
                    )
                    ax.plot(
                        t[selected],
                        data["independent"][selected, joint],
                        label="左右独立参数",
                        color="#167d9a",
                        lw=1,
                    )
                    ax.set(
                        title=f"{JOINTS[joint]} — {'末尾 2 秒' if zoom else '完整扫频'}",
                        ylabel="q（rad）",
                    )
                    ax.grid(alpha=0.2)
                    if row == 0:
                        ax.legend(fontsize=8, ncol=3)
                selected = mask & (t > t[mask][-1] - 2)
                ax = axes[2, column]
                ax.plot(
                    t[selected],
                    data["measured_dq"][selected, joint],
                    label="实测",
                    color="black",
                    lw=1.2,
                )
                ax.plot(
                    t[selected],
                    data["tied"][selected, 6 + joint],
                    label="左右共用参数",
                    color="#8695a7",
                    lw=1,
                )
                ax.plot(
                    t[selected],
                    data["independent"][selected, 6 + joint],
                    label="左右独立参数",
                    color="#167d9a",
                    lw=1,
                )
                ax.set(ylabel="dq（rad/s）", xlabel="扫频时间（s）", title="速度 — 末尾 2 秒")
                ax.grid(alpha=0.2)
            fig.suptitle(f"C 组 {GROUPS[group]}：实测与两种模型", fontsize=14)
            save(
                fig,
                "validation_" + GROUPS[group],
                f"{run}：保持日志中的关节坐标，左右没有再次翻转符号。",
            )

        profile = np.array(report["delay_profile_conditional_on_frozen_physical_parameters"])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(profile[:, 0] * 1000, profile[:, 1], "o-", color="#167d9a", ms=4)
        ax.axvline(
            report["fixed_shared_delay_s"] * 1000, color="#e58943", ls="--", label="本次固定：24 ms"
        )
        ax.set(
            xlabel="统一有效延迟（ms）",
            ylabel="A/B 归一化位置均方误差",
            title="冻结六关节物理参数后的统一延迟检查",
        )
        ax.grid(alpha=0.2)
        ax.legend()
        save(
            fig, "delay_profile", "这是固定物理参数的条件扫描，不能当作延迟置信区间或实测通信延迟。"
        )

    parameter_rows = "".join(
        f"<tr><td>{r['joint']}</td><td>{r['armature_kg_m2']:.7f}</td>"
        f"<td>{r['damping_nm_s_rad']:.7g}</td><td>{r['frictionloss_nm']:.7g}</td>"
        f"<td>{r['delay_ms']:.0f}</td></tr>"
        for r in rows
    )
    validation_rows = "".join(
        f"<tr><td>{r['joint']}</td><td>{r['tied_q_rmse_rad']:.5f}</td>"
        f"<td>{r['independent_q_rmse_rad']:.5f}</td><td>{r['q_rmse_improvement_percent']:+.2f}%</td>"
        f"<td>{r['tied_dq_rmse_rad_s']:.4f} → {r['independent_dq_rmse_rad_s']:.4f}</td></tr>"
        for r in rows
    )
    differences = []
    for group in range(3):
        differences.append(
            dict(
                group=GROUPS[group],
                armature_symmetric_relative_difference_percent=float(
                    200
                    * abs(par[0, group] - par[0, group + 3])
                    / (par[0, group] + par[0, group + 3])
                ),
                damping_absolute_difference=float(abs(par[1, group] - par[1, group + 3])),
                friction_absolute_difference=float(abs(par[2, group] - par[2, group + 3])),
            )
        )
    final_pass = max(h["pass_index"] for h in report["history"])
    convergence = [h for h in report["history"] if h["pass_index"] == final_pass]
    summary = dict(
        joints=rows,
        left_right_differences=differences,
        train_loss_improvement_percent=100
        * (1 - report["independent_train_loss"] / report["tied_train_loss"]),
        final_pass_converged=all(h["converged"] for h in convergence),
        conditional_best_shared_delay_ms=float(profile[np.argmin(profile[:, 1]), 0] * 1000),
    )
    for model in ("tied", "independent"):
        summary[model + "_validation_q_nrmse"] = float(
            np.sqrt(
                np.mean(
                    [
                        report["metrics"][run][model]["selected_q_nrmse_amplitude"] ** 2
                        for run in report["held_out_runs"]
                    ]
                )
            )
        )
    summary["validation_q_nrmse_improvement_percent"] = 100 * (
        1 - summary["independent_validation_q_nrmse"] / summary["tied_validation_q_nrmse"]
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    figures_html = "".join(
        f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,'
        + base64.b64encode((out / (name + ".png")).read_bytes()).decode()
        + f'"><figcaption>{html.escape(caption)}</figcaption></figure>'
        for name, caption in figures
    )
    converged_text = (
        "六个关节均收敛"
        if summary["final_pass_converged"]
        else "部分优化块触及搜索预算；详见 comparison.json"
    )
    difference_rows = "".join(
        f"<tr><td>{d['group']}</td>"
        f"<td>{d['armature_symmetric_relative_difference_percent']:.2f}%</td>"
        f"<td>{d['damping_absolute_difference']:.6g}</td>"
        f"<td>{d['friction_absolute_difference']:.6g}</td></tr>"
        for d in differences
    )
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>PE03 左右独立辨识</title><style>
body{{max-width:1200px;margin:35px auto;padding:0 24px;font-family:system-ui,sans-serif;line-height:1.7;color:#172b4d}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{width:100%}}figure{{margin:35px 0}}.note{{padding:18px;background:#eef5f6;border-left:4px solid #167d9a}}
</style><h1>PE03 左右独立辨识</h1>
<p>数据：pe03_gain_matrix_test01。使用现有 A/B 组拟合，C 组仅验证；无补测，无实机控制，未恢复训练。</p>
<div class="note">六个关节分别估计附加惯量、被动阻尼和库仑摩擦；统一有效延迟固定为上次的 24 ms。
这些是基于当前 CAD 模型及控制近似的有效参数。结果保存为候选，尚未替换训练配置。</div>
<p>A/B 归一化位置均方误差：{report["tied_train_loss"]:.8f} → {report["independent_train_loss"]:.8f}，
下降 {summary["train_loss_improvement_percent"]:.2f}%。最后一轮：{converged_text}。
物理参数冻结后，在 0–60 ms、间隔 2 ms 的扫描中，条件最优统一延迟为 {summary["conditional_best_shared_delay_ms"]:.0f} ms。</p>
<h2>每个关节的候选参数</h2>
<table><tr><th>关节</th><th>armature（kg·m²）</th><th>被动阻尼（N·m·s/rad）</th><th>摩擦（N·m）</th><th>有效延迟（ms）</th></tr>{parameter_rows}</table>
<p>armature 是加入关节自由度的惯量，刚体 CAD 惯量仍保留；并非整条腿的总惯量。
被动阻尼不是控制器 Kd，本次也没有重新估计或调整 Kp/Kd。
接近零或等于零的拟合值表示当前数据及模型不支持更大的正值，不能证明实际损耗为零。</p>
<h2>同组左右差异</h2><table><tr><th>组</th><th>惯量相对差异</th><th>阻尼绝对差异</th><th>摩擦绝对差异</th></tr>{difference_rows}</table>
<p>相对差异定义为 |左−右| / 左右均值；阻尼和摩擦可能接近零，因此使用带原单位的绝对差异，避免夸大的百分比。</p>
<h2>C 组验证</h2><table><tr><th>关节</th><th>共用参数位置 RMSE</th><th>独立参数位置 RMSE</th><th>位置改善</th><th>速度 RMSE（旧 → 新）</th></tr>{validation_rows}</table>
<p>三个关节组按各自激励幅值归一化后等权汇总，C 组位置 NRMSE 为
{summary["tied_validation_q_nrmse"]:.6f} → {summary["independent_validation_q_nrmse"]:.6f}，
改善 {summary["validation_q_nrmse_improvement_percent"]:+.2f}%。</p>
<p>位置单位 rad，速度单位 rad/s；负改善表示变差。更多参数降低 A/B 误差是预期现象，是否有用应同时看未参与拟合的 C 组。
C 组此前已用于评价原模型，因此它是本次优化未使用的验证集，不是从未查看过的全新测试集。</p>
{figures_html}<h2>解释边界与可复现性</h2>
<p>左右同组同步激励，无法分离所有耦合、传感器与伺服效应；日志缺少电机执行时间戳和反馈样本年龄。
左右参数差异可能吸收这些误差，不能全部解释为制造差异。该验证只覆盖悬空台架扫频，不等同于落地步态验证。</p>
<p>完整数值见 comparison.json、summary.json、joint_summary.csv；候选为 candidate.yaml。
每组实测和模型序列保存为 NPZ，图片及独立 PDF 可直接分享。
搜索使用有界 Powell 坐标迭代，最后一轮收敛不等于找到全局最优；参数扰动结果不是统计置信区间。</p>
<p>运行：<code>python -m scripts.identification.fit_independent_lr</code>，随后
<code>python -m scripts.identification.report_independent_lr</code>。源码快照及 SHA256 见 provenance 与 artifact_manifest.json。</p></html>"""
    (out / "index.html").write_text(content)
    provenance = out / "provenance"
    provenance.mkdir(exist_ok=True)
    for name in (
        "fit_independent_lr.py",
        "report_independent_lr.py",
        "fit_pe_sweeps.py",
        "pe_replay.cpp",
    ):
        shutil.copyfile(Path(__file__).with_name(name), provenance / name)
    manifest = {
        str(p.relative_to(out)): digest(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "artifact_manifest.json"
    }
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
