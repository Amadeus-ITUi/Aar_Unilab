#!/usr/bin/env python3
"""
Sweep frequency test data plotting script
Reads data from CSV file and generates PDF plots
"""

import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse


def load_data(csv_file):
    """Load data from CSV file"""
    timestamps = []
    motor_data = {}  # {motor_id: {'cmd_pos': [], 'act_pos': [], 'act_vel': [], 'act_torque': []}}
    motor_ids = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        # Extract motor IDs
        headers = reader.fieldnames
        for header in headers:
            if header.startswith('motor_') and header.endswith('_cmd_pos'):
                motor_id = int(header.split('_')[1])
                if motor_id not in motor_ids:
                    motor_ids.append(motor_id)
                    motor_data[motor_id] = {
                        'cmd_pos': [],
                        'act_pos': [],
                        'act_vel': [],
                        'act_torque': []
                    }
        
        motor_ids.sort()  # Sort
        
        # Read data
        for row in reader:
            timestamps.append(float(row['timestamp']))
            for motor_id in motor_ids:
                motor_data[motor_id]['cmd_pos'].append(
                    float(row[f'motor_{motor_id}_cmd_pos']))
                motor_data[motor_id]['act_pos'].append(
                    float(row[f'motor_{motor_id}_act_pos']))
                motor_data[motor_id]['act_vel'].append(
                    float(row[f'motor_{motor_id}_act_vel']))
                motor_data[motor_id]['act_torque'].append(
                    float(row[f'motor_{motor_id}_act_torque']))
    
    return np.array(timestamps), motor_data, motor_ids


def plot_sweep_frequency(csv_file, output_pdf=None):
    """Plot sweep frequency test data"""
    # Load data
    timestamps, motor_data, motor_ids = load_data(csv_file)
    
    # If output filename not specified, auto-generate
    if output_pdf is None:
        base_name = os.path.splitext(csv_file)[0]
        output_pdf = base_name + '_plot.pdf'
    
    # Create PDF file
    with PdfPages(output_pdf) as pdf:
        # Set plot style (try multiple styles for compatibility)
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('default')  # Use default style if others unavailable
        fig_width = 12  # inches
        fig_height = 8  # height per row (inches)
        
        # First page: Position comparison (6 rows, 1 column)
        fig1, axes1 = plt.subplots(len(motor_ids), 1, figsize=(fig_width, fig_height * len(motor_ids)))
        if len(motor_ids) == 1:
            axes1 = [axes1]  # Ensure it's a list
        
        for idx, motor_id in enumerate(motor_ids):
            ax = axes1[idx]
            cmd_pos = np.array(motor_data[motor_id]['cmd_pos'])
            act_pos = np.array(motor_data[motor_id]['act_pos'])
            
            ax.plot(timestamps, cmd_pos, label='Command Position', linewidth=1.5, alpha=0.8, color='blue')
            ax.plot(timestamps, act_pos, label='Actual Position', linewidth=1.5, alpha=0.8, color='red')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Position (rad)', fontsize=10)
            ax.set_title(f'Motor {motor_id} - Position Comparison', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([timestamps[0], timestamps[-1]])
        
        plt.tight_layout()
        pdf.savefig(fig1, bbox_inches='tight', dpi=150)
        plt.close(fig1)
        
        # Second page: Velocity and torque (6 rows, 1 column, each motor one row with velocity and torque on same plot)
        fig2, axes2 = plt.subplots(len(motor_ids), 1, figsize=(fig_width, fig_height * len(motor_ids)))
        if len(motor_ids) == 1:
            axes2 = [axes2]  # Ensure it's a list
        
        for idx, motor_id in enumerate(motor_ids):
            ax = axes2[idx]
            act_vel = np.array(motor_data[motor_id]['act_vel'])
            act_torque = np.array(motor_data[motor_id]['act_torque'])
            
            # Use dual y-axis to display velocity and torque
            ax1 = ax
            ax2 = ax.twinx()
            
            # Plot velocity (left y-axis)
            line1 = ax1.plot(timestamps, act_vel, label='Actual Velocity', linewidth=1.5, alpha=0.8, color='green')
            ax1.set_xlabel('Time (s)', fontsize=10)
            ax1.set_ylabel('Velocity (rad/s)', fontsize=10, color='green')
            ax1.tick_params(axis='y', labelcolor='green')
            
            # Plot torque (right y-axis)
            line2 = ax2.plot(timestamps, act_torque, label='Actual Torque', linewidth=1.5, alpha=0.8, color='orange')
            ax2.set_ylabel('Torque (Nm)', fontsize=10, color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            ax1.set_title(f'Motor {motor_id} - Velocity and Torque', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([timestamps[0], timestamps[-1]])
            
            # Merge legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig2, bbox_inches='tight', dpi=150)
        plt.close(fig2)
    
    print(f"Plots saved to: {output_pdf}")
    return output_pdf


def main():
    parser = argparse.ArgumentParser(description='Plot sweep frequency test data')
    parser.add_argument('csv_file', type=str, help='Input CSV file path')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output PDF file path (optional)')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        output_pdf = plot_sweep_frequency(args.csv_file, args.output)
        print(f"Success: Plots saved to {output_pdf}")
        sys.exit(0)
    except Exception as e:
        print(f"Error: Plotting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

