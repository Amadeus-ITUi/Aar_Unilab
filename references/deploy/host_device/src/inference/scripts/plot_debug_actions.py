#!/usr/bin/env python3
"""
Plot debug actions from inference_debug_csv_node output.

Input CSV format:
  time, obs_0..obs_29, act_0..act_5

This script generates a PDF with:
  - Page 1: all 6 actions vs time (one figure, 6 curves)
  - Page 2: per-action subplots vs time (6 rows)
"""

import sys
import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_debug_csv(csv_file):
    times = []
    obs = []   # shape: [N, 30]
    acts = []  # shape: [N, 6]

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if fields is None:
            raise RuntimeError("CSV 无表头，无法解析")

        # 检查必要字段
        if "time" not in fields:
            raise RuntimeError("CSV 缺少列 'time'")

        obs_cols = [c for c in fields if c.startswith("obs_")]
        act_cols = [c for c in fields if c.startswith("act_")]
        if len(obs_cols) == 0:
            raise RuntimeError("CSV 中没有任何 'obs_*' 列")
        if len(act_cols) == 0:
            raise RuntimeError("CSV 中没有任何 'act_*' 列")

        obs_cols_sorted = sorted(obs_cols, key=lambda x: int(x.split("_")[1]))
        act_cols_sorted = sorted(act_cols, key=lambda x: int(x.split("_")[1]))

        for row in reader:
            try:
                t = float(row["time"])
            except Exception:
                continue
            times.append(t)

            obs_row = [float(row[c]) for c in obs_cols_sorted]
            act_row = [float(row[c]) for c in act_cols_sorted]
            obs.append(obs_row)
            acts.append(act_row)

    if len(times) == 0:
        raise RuntimeError("CSV 中没有有效数据行")

    return np.array(times), np.array(obs), np.array(acts)


def plot_debug_actions(csv_file, output_pdf=None):
    t, obs, acts = load_debug_csv(csv_file)

    if output_pdf is None:
        base = os.path.splitext(csv_file)[0]
        output_pdf = base + "_actions.pdf"

    with PdfPages(output_pdf) as pdf:
        # 样式
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except Exception:
            try:
                plt.style.use("seaborn-darkgrid")
            except Exception:
                plt.style.use("default")

        # Page 1: 所有 action 在一张图上
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        num_acts = acts.shape[1]
        colors = plt.cm.tab10(np.linspace(0, 1, max(num_acts, 3)))
        for i in range(num_acts):
            ax1.plot(t, acts[:, i], label=f"act_{i}", color=colors[i % len(colors)], linewidth=1.2)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Action value")
        ax1.set_title("Actions vs Time (all joints)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", fontsize=9)
        ax1.set_xlim(t[0], t[-1])
        plt.tight_layout()
        pdf.savefig(fig1, bbox_inches="tight", dpi=150)
        plt.close(fig1)

        # Page 2: 每个 action 单独一行子图
        fig2, axes = plt.subplots(num_acts, 1, figsize=(12, 2.5 * num_acts), sharex=True)
        if num_acts == 1:
            axes = [axes]
        for i in range(num_acts):
            ax = axes[i]
            ax.plot(t, acts[:, i], color=colors[i % len(colors)], linewidth=1.2)
            ax.set_ylabel(f"act_{i}")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        fig2.suptitle("Actions vs Time (per joint)", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        pdf.savefig(fig2, bbox_inches="tight", dpi=150)
        plt.close(fig2)

    print(f"Plots saved to: {output_pdf}")
    return output_pdf


def main():
    parser = argparse.ArgumentParser(description="Plot debug actions from CSV")
    parser.add_argument("csv_file", type=str, help="Input CSV file path (time, obs_*, act_*)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output PDF file path (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: file not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    try:
        out = plot_debug_actions(args.csv_file, args.output)
        print(f"Success: plots saved to {out}")
        sys.exit(0)
    except Exception as e:
        print(f"Error: plotting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

