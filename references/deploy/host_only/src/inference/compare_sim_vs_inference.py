#!/usr/bin/env python3
"""
!!!! WARNING - STALE TOOL !!!!
本工具按 26 维旧观测契约解释 CSV 列（含 4 维 command 含 JumpHeight、无 wing_angle、
无 wing_vel），与当前 lab_inference_node 使用的 29 维 obs (含 wing_angle 2 维和
wing_vel 2 维，command 3 维 [vx, yaw, height]) 完全不兼容。使用它对比新契约
下的 CSV 会把多个通道标错。使用前必须按 29 维契约重写列名映射与图例。

对比 sim.csv 与 inference 观测 CSV 的观测数据。
支持通过命令行传入两个 CSV 路径，或使用默认路径。
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_compare_data(sim_file=None, inference_file=None, output_pdf=None):
    # 项目根目录：优先使用环境变量 DEPLOY_CPP_ROOT，否则使用当前脚本所在目录的上级（deploy_cpp）
    project_root = os.environ.get("DEPLOY_CPP_ROOT")
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]  # .../deploy_cpp/inference -> deploy_cpp
    else:
        project_root = Path(project_root)

    plots_dir = project_root / "plots" / "inference"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 未传入时使用默认路径
    sim_file = Path(sim_file) if sim_file else (plots_dir / "sim.csv")
    inference_file = Path(inference_file) if inference_file else (plots_dir / "inference_obs_20260120_144950.csv")
    output_pdf = Path(output_pdf) if output_pdf else (plots_dir / "sim_vs_inference_comparison.pdf")

    # 读取数据
    print(f"Reading sim.csv from {sim_file} ...")
    df_sim = pd.read_csv(sim_file)
    print(f"sim.csv shape: {df_sim.shape}")

    print(f"Reading inference_obs CSV from {inference_file} ...")
    df_inference = pd.read_csv(inference_file)
    print(f"inference CSV shape: {df_inference.shape}")

    # 统一时间列名（sim 可能为 'time'，inference 为 'timestamp'）
    if 'time' in df_sim.columns and 'timestamp' not in df_sim.columns:
        df_sim = df_sim.rename(columns={'time': 'timestamp'})
    if 'time' in df_inference.columns and 'timestamp' not in df_inference.columns:
        df_inference = df_inference.rename(columns={'time': 'timestamp'})

    # 限制数据为前10秒
    time_limit = 10.0
    n_sim_before = len(df_sim)
    n_inf_before = len(df_inference)
    if 'timestamp' in df_sim.columns:
        df_sim = df_sim[df_sim['timestamp'] <= time_limit]
    if 'timestamp' in df_inference.columns:
        df_inference = df_inference[df_inference['timestamp'] <= time_limit]
    n_sim_after = len(df_sim)
    n_inf_after = len(df_inference)
    if n_sim_before > n_sim_after or n_inf_before > n_inf_after:
        print("[数据裁切] 已按时间截断为前 {:.1f} 秒：sim {} -> {} 行，inference {} -> {} 行".format(
            time_limit, n_sim_before, n_sim_after, n_inf_before, n_inf_after))

    # 自动发现所有观测列（只选择前若干个以避免图表过多）
    obs_columns = [col for col in df_sim.columns if col.startswith('obs_')]
    obs_columns = obs_columns[:36]  # 限制为前12个观测项

    print(f"Plotting {len(obs_columns)} observation columns: {obs_columns}")

    # 创建子图 - 每个观测项独占一行
    n_rows = len(obs_columns)
    n_cols = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    fig.suptitle('sim.csv vs inference_obs_20260120_134340.csv\nObservation Data Comparison (First 10 seconds)',
                fontsize=14, fontweight='bold', y=0.995)

    # 确保 axes 是一维数组
    if n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 观测维度名称（按“策略输入顺序”解释，已适配 26 维 obs + 3 维 latent）：
    #  - 0-25: observations
    #       0-2  : Ang Vel X/Y/Z
    #       3-5  : Grav Proj X/Y/Z
    #       6-9  : Cmd VX/Yaw/Height/JumpHeight
    #       10-13: Dof Pos 0-3
    #       14-19: Dof Vel 0-5
    #       20-25: Last Act 0-5
    #  - 26-28: estimated_velocity_（latent）
    obs_names = [
        # 0-2: 机身角速度（来自 IMU，经缩放）
        'Ang Vel X', 'Ang Vel Y', 'Ang Vel Z',  # 0-2
        # 3-5: 重力投影到机体坐标系
        'Grav Proj X', 'Grav Proj Y', 'Grav Proj Z',  # 3-5
        # 6-9: 命令（期望线速度 / 期望转速 / 期望高度 / 跳跃高度）
        'Cmd VX', 'Cmd Yaw', 'Cmd H', 'Cmd Jump H',  # 6-9
        # 10-13: 关节位置（4 个自由度）
        'Dof Pos 0', 'Dof Pos 1', 'Dof Pos 2', 'Dof Pos 3',  # 10-13
        # 14-19: 关节速度（6 个自由度）
        'Dof Vel 0', 'Dof Vel 1', 'Dof Vel 2', 'Dof Vel 3', 'Dof Vel 4', 'Dof Vel 5',  # 14-19
        # 20-25: 上一次动作（raw action，6 维）
        'Last Act 0', 'Last Act 1', 'Last Act 2', 'Last Act 3', 'Last Act 4', 'Last Act 5',  # 20-25
        # 26-28: latent / estimated_velocity_
        'Est Vel X', 'Est Vel Y', 'Est Vel Z',  # 26-28
        # 留空占位，防止索引越界（如果 sim.csv 里有更多列）
        'obs_30', 'obs_31', 'obs_32', 'obs_33', 'obs_34', 'obs_35'  # 30-35
    ]

    # 对每个观测项进行对比
    for idx, obs_col in enumerate(obs_columns):
        ax = axes[idx]

        # 获取观测项的索引和名称
        obs_num = int(obs_col.split('_')[1])
        if obs_num < len(obs_names):
            title = obs_names[obs_num]
        else:
            title = f'obs_{obs_num}'

        # 绘制 sim.csv 数据
        if obs_col in df_sim.columns:
            ax.plot(df_sim['timestamp'], df_sim[obs_col],
                   label='sim.csv', color='blue', linewidth=2, alpha=0.8)

        # 绘制 inference 数据
        if obs_col in df_inference.columns:
            ax.plot(df_inference['timestamp'], df_inference[obs_col],
                   label='inference_obs', color='red', linewidth=2, alpha=0.8, linestyle='--')

        ax.set_title(f'{obs_col}: {title}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图形
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_pdf}")

    # 显示图形
    plt.show()

    # 打印统计信息
    print("\n=== Data Statistics Comparison ===")
    for obs_col in obs_columns[:6]:  # 显示前6个观测项的统计信息
        if obs_col in df_sim.columns and obs_col in df_inference.columns:
            sim_stats = df_sim[obs_col].describe()
            inf_stats = df_inference[obs_col].describe()

            print(f"\n{obs_col} Statistics:")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="对比 sim 与 inference 观测 CSV，并生成对比图。"
    )
    parser.add_argument(
        "sim_file",
        nargs="?",
        default=None,
        help="sim 观测 CSV 路径（可选，默认: plots/inference/sim.csv）",
    )
    parser.add_argument(
        "inference_file",
        nargs="?",
        default=None,
        help="inference 观测 CSV 路径（可选，默认: plots/inference/inference_obs_*.csv）",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出 PDF 路径（可选）",
    )
    args = parser.parse_args()
    load_and_compare_data(
        sim_file=args.sim_file,
        inference_file=args.inference_file,
        output_pdf=args.output,
    )
