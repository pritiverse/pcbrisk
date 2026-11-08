"""plot_cost_curves.py - Generates cost curves versus escape-to-review cost ratio R and identifies crossover points."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

def generate_cost_curves():
    # Load cost evaluation summary
    df = pd.read_csv('05_results/cost_sensitive_summary.csv')
    
    # We examine R in range [10, 2000]
    R_vals = np.linspace(1, 2000, 500)
    
    # Extract reviews and escapes per 1k boards
    # Real alpha=0.05
    pooled_row = df[df.method == 'pooled'].iloc[0]
    worst_row = df[df.method == 'worst-board'].iloc[0]
    shift_row = df[df.method == 'shift-inflated'].iloc[0]
    median_row = df[df.method == 'board-median'].iloc[0]
    
    # Baseline A0 and A (pooled)
    cost_a0_pooled = pooled_row['load_A0'] * 1000 + R_vals * pooled_row['esc_1k_A0']
    cost_a_pooled = pooled_row['load_A'] * 1000 + R_vals * pooled_row['esc_1k_A']
    
    # Policy B across calibrators
    cost_b_pooled = pooled_row['load_B'] * 1000 + R_vals * pooled_row['esc_1k_B']
    cost_b_shift = shift_row['load_B'] * 1000 + R_vals * shift_row['esc_1k_B']
    cost_b_worst = worst_row['load_B'] * 1000 + R_vals * worst_row['esc_1k_B']
    cost_b_median = median_row['load_B'] * 1000 + R_vals * median_row['esc_1k_B']
    
    # Also baseline A0 worst-board
    cost_a0_worst = worst_row['load_A0'] * 1000 + R_vals * worst_row['esc_1k_A0']
    cost_a_worst = worst_row['load_A'] * 1000 + R_vals * worst_row['esc_1k_A']

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Subplot 1: Main Triage Comparison (Pooled & Shift-Inflated)
    ax1.plot(R_vals, cost_a0_pooled, 'k--', label='Baseline A0 (Raw Conf, Pooled)', linewidth=2.0)
    ax1.plot(R_vals, cost_a_pooled, 'b-.', label='Policy A (Image-Only, Pooled)', linewidth=2.0)
    ax1.plot(R_vals, cost_b_pooled, 'r-', label='Policy B +Graph (Pooled - Aggregate Pareto)', linewidth=2.5)
    ax1.plot(R_vals, cost_b_shift, 'g-', label='Policy B +Graph (Shift-Inflated)', linewidth=2.0)
    ax1.plot(R_vals, cost_b_median, 'm:', label='Policy B +Graph (Board-Median)', linewidth=2.0)
    
    ax1.set_title('Expected Inspection Cost vs Escape Penalty Ratio (Pooled Calibration)', fontsize=12, fontweight='bold')
    ax1.set_xlabel(r'Cost Ratio $R = C_{escape} / C_{review}$', fontsize=11)
    ax1.set_ylabel('Normalized Cost per 1,000 Boards (Review Units)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=9)
    
    # Subplot 2: The Worst-Board Trade-Off & Crossover Analysis
    ax2.plot(R_vals, cost_a0_worst, 'k--', label='Baseline A0 (Raw Conf, Worst-Board)', linewidth=2.0)
    ax2.plot(R_vals, cost_a_worst, 'b-.', label='Policy A (Image-Only, Worst-Board)', linewidth=2.0)
    ax2.plot(R_vals, cost_b_worst, 'crimson', label='Policy B +Graph (Worst-Board)', linewidth=2.5)
    
    # Crossover point: cost_b_worst == cost_a0_worst
    # delta_load = (load_B - load_A0)*1000 = (8089.56 - 12099.78) = -4010.22
    # delta_esc = (esc_B - esc_A0) = (83.56 - 80.22) = +3.333
    # R* = 4010.22 / 3.333 = 1203.07
    r_cross_a0 = (worst_row['load_A0'] - worst_row['load_B']) * 1000 / (worst_row['esc_1k_B'] - worst_row['esc_1k_A0'])
    cost_cross_a0 = worst_row['load_B'] * 1000 + r_cross_a0 * worst_row['esc_1k_B']
    
    r_cross_a = (worst_row['load_A'] - worst_row['load_B']) * 1000 / (worst_row['esc_1k_B'] - worst_row['esc_1k_A'])
    cost_cross_a = worst_row['load_B'] * 1000 + r_cross_a * worst_row['esc_1k_B']

    ax2.scatter([r_cross_a0], [cost_cross_a0], color='black', s=80, zorder=5)
    ax2.annotate(f'Crossover vs A0\n$R^* \\approx {int(r_cross_a0)}$', 
                 xy=(r_cross_a0, cost_cross_a0), xytext=(r_cross_a0 - 450, cost_cross_a0 + 15000),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=9, fontweight='bold')
                 
    ax2.scatter([r_cross_a], [cost_cross_a], color='blue', s=80, zorder=5)
    ax2.annotate(f'Crossover vs A\n$R^* \\approx {int(r_cross_a)}$', 
                 xy=(r_cross_a, cost_cross_a), xytext=(r_cross_a - 400, cost_cross_a - 15000),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=6),
                 fontsize=9, fontweight='bold')

    ax2.axvspan(1, r_cross_a0, alpha=0.1, color='green', label=r'Policy B Preferred ($R < 1,203$)')
    ax2.axvspan(r_cross_a0, 2000, alpha=0.1, color='red', label=r'A0 Preferred (Ultra-High Escape Penalty)')
    
    ax2.set_title('Worst-Board Calibration: Review Savings vs Escape Penalty Crossover', fontsize=12, fontweight='bold')
    ax2.set_xlabel(r'Cost Ratio $R = C_{escape} / C_{review}$', fontsize=11)
    ax2.set_ylabel('Normalized Cost per 1,000 Boards (Review Units)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    os.makedirs('05_results', exist_ok=True)
    plot_path = '05_results/cost_curve_analysis.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Cost curve plot saved to {plot_path}")
    print(f"Crossover vs A0: R* = {r_cross_a0:.2f}")
    print(f"Crossover vs A:  R* = {r_cross_a:.2f}")

if __name__ == '__main__':
    generate_cost_curves()
