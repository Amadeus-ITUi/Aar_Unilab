#!/usr/bin/env python3
"""
观测数据绘图脚本
电机1,2,4,5的 action（raw）与关节位置画在同一张图；电机3,6的 action（raw）与关节速度画在同一张图。
画图使用的 action 为 raw（策略观测空间，与反馈给策略的 last_action 一致）。
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def plot_obs_data(csv_file, pdf_file):
    """绘制观测数据到PDF，action 使用 raw（obs_20..25，26维obs布局，含4维命令）"""
    print(f"正在读取数据: {csv_file}")
    
    # 读取CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return
    
    # 检查必需的列（29维 obs：26维观测 + 3维latent）
    required_columns = ['timestamp'] + [f'obs_{i}' for i in range(29)]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ CSV缺少必需的列: {missing_columns}")
        return
    
    print("✅ 画图使用 Action (raw)，即 obs_20..25（与反馈给策略的 last_action 一致，26维obs布局）")
    print(f"✅ 成功读取 {len(df)} 帧数据")
    
    # 创建PDF
    with PdfPages(pdf_file) as pdf:
        # 电机索引对应关系：
        # 索引0 -> 电机1 (left_thigh)
        # 索引1 -> 电机2 (left_calf)
        # 索引2 -> 电机3 (left_foot, 轮子)
        # 索引3 -> 电机4 (right_thigh)
        # 索引4 -> 电机5 (right_calf)
        # 索引5 -> 电机6 (right_foot, 轮子)
        
        # 观测数据索引（26维obs布局）：
        # obs_0-2  : Ang Vel X/Y/Z
        # obs_3-5  : Grav Proj X/Y/Z
        # obs_6-9  : Cmd VX/Yaw/Height/JumpHeight
        # obs_10-13: Dof Pos 0-3 (腿部4个关节位置，已物理删除轮子位置)
        # obs_14-19: Dof Vel 0-5 (6个关节速度，含轮子)
        # obs_20-25: Last Act 0-5 (上次动作，6个关节的 raw action)
        
        # 第一页：电机1,2,4,5的action与关节位置
        # 索引与 C++ 一致：
        #  - Dof Pos: 0..3 -> 电机1,2,4,5 位置 -> obs_10..13
        #  - Last Act: 0..5 -> 电机1,2,3,4,5,6 动作 -> obs_20..25
        # 电机1 (索引0): obs_10 (位置) + obs_20 (action)
        # 电机2 (索引1): obs_11 (位置) + obs_21 (action)
        # 电机4 (索引3): obs_12 (位置) + obs_23 (action)
        # 电机5 (索引4): obs_13 (位置) + obs_24 (action)
        fig1, axes1 = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig1.suptitle('Motor Action vs Position (Motors 1,2,4,5)', fontsize=16, fontweight='bold')
        
        motor_info = [
            (0, 'Motor 1 (Left Thigh)', 10, 20),   # 电机索引, 名称, 位置obs索引, action obs索引
            (1, 'Motor 2 (Left Calf)', 11, 21),
            (3, 'Motor 4 (Right Thigh)', 12, 23),
            (4, 'Motor 5 (Right Calf)', 13, 24)
        ]
        
        for idx, (motor_idx, motor_name, pos_obs_idx, act_obs_idx) in enumerate(motor_info):
            ax = axes1[idx]
            t = df['timestamp'].values
            
            # 位置 (rad) 与 action (raw，策略观测空间)
            pos_data = df[f'obs_{pos_obs_idx}'].values
            act_data = df[f'obs_{act_obs_idx}'].values
            
            line1 = ax.plot(t, pos_data, linewidth=1.2, color='tab:blue', label='Position (rad)')
            line2 = ax.plot(t, act_data, linewidth=1.2, color='tab:red', label='Action (raw)', linestyle='--')
            ax.set_ylabel('Position (rad) / Action (raw)', fontsize=10)
            ax.set_title(motor_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linewidth=0.5)
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right', fontsize=9)
            
            # 添加统计信息
            pos_mean = pos_data.mean()
            pos_std = pos_data.std()
            act_mean = act_data.mean()
            act_std = act_data.std()
            stats_text = f'Pos: μ={pos_mean:.3f} σ={pos_std:.3f}\nAct: μ={act_mean:.3f} σ={act_std:.3f}'
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # 只在最后一个子图显示x轴标签
            if idx == len(motor_info) - 1:
                ax.set_xlabel('Time (s)', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig1, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # 第二页：电机3,6的action与关节速度
        # 速度索引（Dof Vel 0-5 -> obs_14..19）：
        #  - 电机3 (索引2): Dof Vel 2 -> obs_16
        #  - 电机6 (索引5): Dof Vel 5 -> obs_19
        # 动作索引（Last Act 0-5 -> obs_20..25）：
        #  - 电机3 (索引2): Last Act 2 -> obs_22
        #  - 电机6 (索引5): Last Act 5 -> obs_25
        fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig2.suptitle('Motor Action vs Velocity (Motors 3,6 - Wheels)', fontsize=16, fontweight='bold')
        
        wheel_info = [
            (2, 'Motor 3 (Left Foot - Wheel)', 16, 22),  # 电机索引, 名称, 速度obs索引, action obs索引
            (5, 'Motor 6 (Right Foot - Wheel)', 19, 25)
        ]
        
        for idx, (motor_idx, motor_name, vel_obs_idx, act_obs_idx) in enumerate(wheel_info):
            ax = axes2[idx]
            t = df['timestamp'].values
            
            # 速度（obs 缩放值）与 raw action 同一 y 轴
            vel_data = df[f'obs_{vel_obs_idx}'].values
            act_data = df[f'obs_{act_obs_idx}'].values  # raw action（与反馈给策略的 last_action 一致）
            line1 = ax.plot(t, vel_data, linewidth=1.2, color='tab:green', label='Velocity (obs)')
            line2 = ax.plot(t, act_data, linewidth=1.2, color='tab:red', label='Action (raw)', linestyle='--')
            ax.set_ylabel('Velocity (obs) | Action (raw)', fontsize=10)
            ax.set_title(motor_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linewidth=0.5)
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right', fontsize=9)
            
            # 添加统计信息
            vel_mean = vel_data.mean()
            vel_std = vel_data.std()
            act_mean = act_data.mean()
            act_std = act_data.std()
            stats_text = f'Vel: μ={vel_mean:.3f} σ={vel_std:.3f}\nAct: μ={act_mean:.3f} σ={act_std:.3f}'
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            if idx == len(wheel_info) - 1:
                ax.set_xlabel('Time (s)', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        # 第三页：其他观测数据（不包括已绘制的电机位置、速度和动作）
        # 已绘制的索引：9,10,12,13 (位置), 17,20 (速度), 21,22,23,24,25,26 (动作)
        # 需要绘制的其他索引：0-8, 11, 14-16, 18-19（去掉已删除的高维 obs_28, obs_29）
        other_obs_indices = list(range(9)) + [11] + [14, 15, 16] + [18, 19]
        
        if len(other_obs_indices) > 0:
            fig3, axes3 = plt.subplots(len(other_obs_indices), 1, figsize=(12, 2 * len(other_obs_indices)), sharex=True)
            if len(other_obs_indices) == 1:
                axes3 = [axes3]  # 确保是列表
            fig3.suptitle('Other Observation Data', fontsize=16, fontweight='bold', y=0.995)
            
            obs_name_map = {
                0: 'Est Vel X', 1: 'Est Vel Y', 2: 'Est Vel Z',
                3: 'Ang Vel X', 4: 'Ang Vel Y', 5: 'Ang Vel Z',
                6: 'Grav Proj X', 7: 'Grav Proj Y', 8: 'Grav Proj Z',
                11: 'Dof Pos 3',  # 电机4位置（已在第一页绘制，但保留用于完整性）
                14: 'Dof Pos 5',  # 电机6位置
                15: 'Dof Vel 0', 16: 'Dof Vel 1',  # 电机1,2速度
                18: 'Dof Vel 3', 19: 'Dof Vel 4',  # 电机4,5速度
            }
            
            for plot_idx, obs_idx in enumerate(other_obs_indices):
                ax = axes3[plot_idx]
                col_name = f'obs_{obs_idx}'
                
                # 绘制数据
                ax.plot(df['timestamp'].values, df[col_name].values, linewidth=0.8, color='tab:blue')
                
                # 设置标签和标题
                obs_name = obs_name_map.get(obs_idx, f'Obs {obs_idx}')
                ax.set_ylabel(obs_name, fontsize=9)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                
                # 添加统计信息
                mean_val = df[col_name].mean()
                std_val = df[col_name].std()
                min_val = df[col_name].min()
                max_val = df[col_name].max()
                
                stats_text = f'μ={mean_val:.3f} σ={std_val:.3f}\nmin={min_val:.3f} max={max_val:.3f}'
                ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                       fontsize=7, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                
                # 只在最后一个子图显示x轴标签
                if plot_idx == len(other_obs_indices) - 1:
                    ax.set_xlabel('Time (s)', fontsize=10)
            
            plt.tight_layout(rect=[0, 0, 1, 0.995])
            pdf.savefig(fig3, dpi=150, bbox_inches='tight')
            plt.close(fig3)
        
        print(f"✅ 绘图完成: {pdf_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python3 plot_obs_data.py <input.csv> <output.pdf>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    pdf_file = sys.argv[2]
    
    plot_obs_data(csv_file, pdf_file)
