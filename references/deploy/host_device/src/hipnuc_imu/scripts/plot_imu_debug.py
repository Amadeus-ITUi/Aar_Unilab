#!/usr/bin/env python3
"""
IMU Debug Data Plotter
绘制IMU调试数据（角速度和重力投影）
每个数据一张图，每张图独占一行，保存为PDF
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def plot_imu_data(csv_file, pdf_file):
    """
    读取CSV数据并绘制图表
    
    Args:
        csv_file: 输入CSV文件路径
        pdf_file: 输出PDF文件路径
    """
    # 读取数据
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False
    
    # 检查必要的列
    required_columns = ['timestamp', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                       'gravity_x', 'gravity_y', 'gravity_z']
    optional_columns = ['quat_w', 'quat_x', 'quat_y', 'quat_z']
    raw_columns = ['ang_vel_x_raw', 'ang_vel_y_raw', 'ang_vel_z_raw',
                   'gravity_x_raw', 'gravity_y_raw', 'gravity_z_raw']
    
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in CSV file")
            return False
    
    # 检查是否有四元数数据
    has_quaternion = all(col in df.columns for col in optional_columns)
    # 检查是否有原始数据
    has_raw_data = all(col in df.columns for col in raw_columns)
    
    # 创建PDF
    with PdfPages(pdf_file) as pdf:
        # 设置全局字体大小
        plt.rcParams.update({'font.size': 10})
        
        # 数据列表：(列名, 原始列名, y轴标签, 标题, 颜色)
        data_to_plot = [
            ('ang_vel_x', 'ang_vel_x_raw', 'Angular Velocity X (rad/s)', 'Angular Velocity X', 'tab:blue'),
            ('ang_vel_y', 'ang_vel_y_raw', 'Angular Velocity Y (rad/s)', 'Angular Velocity Y', 'tab:orange'),
            ('ang_vel_z', 'ang_vel_z_raw', 'Angular Velocity Z (rad/s)', 'Angular Velocity Z', 'tab:green'),
            ('gravity_x', 'gravity_x_raw', 'Projected Gravity X', 'Projected Gravity X', 'tab:red'),
            ('gravity_y', 'gravity_y_raw', 'Projected Gravity Y', 'Projected Gravity Y', 'tab:purple'),
            ('gravity_z', 'gravity_z_raw', 'Projected Gravity Z', 'Projected Gravity Z', 'tab:brown'),
        ]
        
        # 四元数数据（无原始值对比）
        quat_data_to_plot = []
        if has_quaternion:
            quat_data_to_plot = [
                ('quat_w', 'Quaternion W', 'Quaternion W Component', 'tab:cyan'),
                ('quat_x', 'Quaternion X', 'Quaternion X Component', 'tab:pink'),
                ('quat_y', 'Quaternion Y', 'Quaternion Y Component', 'tab:olive'),
                ('quat_z', 'Quaternion Z', 'Quaternion Z Component', 'tab:gray'),
            ]
        
        # 为每个数据创建一张图
        for col_name, raw_col_name, ylabel, title, color in data_to_plot:
            fig, ax = plt.subplots(figsize=(10, 3))
            
            # 绘制滤波后的数据
            ax.plot(df['timestamp'].values, df[col_name].values, 
                   label=f'{col_name} (filtered)', 
                   color=color, linewidth=1.5, alpha=0.9)
            
            # 如果有原始数据，叠加绘制
            if has_raw_data and raw_col_name in df.columns:
                ax.plot(df['timestamp'].values, df[raw_col_name].values, 
                       label=f'{raw_col_name} (raw)', 
                       color=color, linewidth=0.5, alpha=0.4, linestyle=':')
            
            # 设置标签和标题
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # 添加统计信息
            mean_val = df[col_name].mean()
            std_val = df[col_name].std()
            min_val = df[col_name].min()
            max_val = df[col_name].max()
            
            stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5), fontsize=8)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # 为四元数数据创建单独的图（无滤波对比）
        for col_name, ylabel, title, color in quat_data_to_plot:
            fig, ax = plt.subplots(figsize=(10, 3))
            
            # 绘制四元数数据
            ax.plot(df['timestamp'].values, df[col_name].values, 
                   label=col_name, color=color, linewidth=1.0, alpha=0.8)
            
            # 设置标签和标题
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # 添加统计信息
            mean_val = df[col_name].mean()
            std_val = df[col_name].std()
            min_val = df[col_name].min()
            max_val = df[col_name].max()
            
            stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5), fontsize=8)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # 添加综合图（可选）- 角速度
        fig, axes = plt.subplots(3, 1, figsize=(10, 9))
        fig.suptitle('Angular Velocity (All Axes)', fontsize=14, fontweight='bold')
        
        for i, (col_name, raw_col_name, ylabel, _, color) in enumerate(data_to_plot[:3]):
            axes[i].plot(df['timestamp'].values, df[col_name].values, 
                        color=color, linewidth=1.5, alpha=0.9, label='filtered')
            if has_raw_data and raw_col_name in df.columns:
                axes[i].plot(df['timestamp'].values, df[raw_col_name].values, 
                            color=color, linewidth=0.5, alpha=0.4, linestyle=':', label='raw')
            axes[i].set_ylabel(ylabel)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right')
            
            if i == 2:
                axes[i].set_xlabel('Time (s)')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # 添加综合图（可选）- 重力投影
        fig, axes = plt.subplots(3, 1, figsize=(10, 9))
        fig.suptitle('Projected Gravity (All Axes)', fontsize=14, fontweight='bold')
        
        for i, (col_name, raw_col_name, ylabel, _, color) in enumerate(data_to_plot[3:6]):
            axes[i].plot(df['timestamp'].values, df[col_name].values, 
                        color=color, linewidth=1.5, alpha=0.9, label='filtered')
            if has_raw_data and raw_col_name in df.columns:
                axes[i].plot(df['timestamp'].values, df[raw_col_name].values, 
                            color=color, linewidth=0.5, alpha=0.4, linestyle=':', label='raw')
            axes[i].set_ylabel(ylabel)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right')
            
            if i == 2:
                axes[i].set_xlabel('Time (s)')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # 添加综合图（可选）- 四元数（如果有）
        if has_quaternion:
            fig, axes = plt.subplots(4, 1, figsize=(10, 12))
            fig.suptitle('Quaternion (All Components)', fontsize=14, fontweight='bold')
            
            quat_data = [
                ('quat_w', 'Quaternion W', 'tab:cyan'),
                ('quat_x', 'Quaternion X', 'tab:pink'),
                ('quat_y', 'Quaternion Y', 'tab:olive'),
                ('quat_z', 'Quaternion Z', 'tab:gray'),
            ]
            
            for i, (col_name, ylabel, color) in enumerate(quat_data):
                axes[i].plot(df['timestamp'].values, df[col_name].values, color=color, linewidth=1.0)
                axes[i].set_ylabel(ylabel)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend([col_name], loc='upper right')
                axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
                
                if i == 3:
                    axes[i].set_xlabel('Time (s)')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # 添加元数据页
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # 统计信息
        stats_info = "IMU Debug Data Statistics\n" + "="*50 + "\n\n"
        stats_info += f"Total Frames: {len(df)}\n"
        stats_info += f"Duration: {df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]:.2f} seconds\n"
        stats_info += f"Sample Rate: {len(df) / (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]):.1f} Hz\n\n"
        
        stats_info += "Angular Velocity Statistics:\n"
        for col in ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']:
            stats_info += f"  {col}:\n"
            stats_info += f"    Mean: {df[col].mean():.6f} rad/s\n"
            stats_info += f"    Std:  {df[col].std():.6f} rad/s\n"
            stats_info += f"    Range: [{df[col].min():.6f}, {df[col].max():.6f}]\n"
        
        stats_info += "\nProjected Gravity Statistics:\n"
        for col in ['gravity_x', 'gravity_y', 'gravity_z']:
            stats_info += f"  {col}:\n"
            stats_info += f"    Mean: {df[col].mean():.6f}\n"
            stats_info += f"    Std:  {df[col].std():.6f}\n"
            stats_info += f"    Range: [{df[col].min():.6f}, {df[col].max():.6f}]\n"
        
        ax.text(0.1, 0.9, stats_info, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"Plots saved successfully to {pdf_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 plot_imu_debug.py <input_csv> <output_pdf>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    pdf_file = sys.argv[2]
    
    success = plot_imu_data(csv_file, pdf_file)
    sys.exit(0 if success else 1)
