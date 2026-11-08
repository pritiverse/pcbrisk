"""cost_sensitive_eval.py - Computes human-in-the-loop cost evaluation across all calibrators and seeds."""
import pandas as pd
import numpy as np
import json
import os

def compute_cost_evaluation():
    dfs = [pd.read_csv(f'05_results/lobo07_real_s{s}.csv') for s in [42, 43, 44]]
    df = pd.concat(dfs, ignore_index=True)

    results = []
    
    # Analyze alpha = 0.05
    for method in ['pooled', 'shift-inflated', 'worst-board', 'board-median']:
        sub = df[(df.alpha == 0.05) & (df.method == method)]
        
        # Summary per policy
        p_stats = {}
        for pol in ['A0 raw conf', 'A image-only', 'B +graph (template)']:
            psub = sub[sub.policy == pol]
            n_img = psub.n_img.sum()
            n_gt = psub.n_gt.sum()
            flagged = psub.flagged.sum()
            n_miss = psub.n_miss.sum()
            
            load = flagged / n_img
            miss_rate = n_miss / n_gt
            rev_1k = (flagged / n_img) * 1000
            esc_1k = (n_miss / n_img) * 1000
            
            p_stats[pol] = {
                'load': load,
                'miss_rate': miss_rate,
                'reviews_per_1k': rev_1k,
                'escapes_per_1k': esc_1k
            }
        
        b = p_stats['B +graph (template)']
        a0 = p_stats['A0 raw conf']
        a = p_stats['A image-only']
        
        # Delta vs A0 (raw confidence)
        d_load_a0 = b['load'] - a0['load']
        rel_load_a0 = (b['load'] - a0['load']) / a0['load']
        d_esc_a0 = b['escapes_per_1k'] - a0['escapes_per_1k']
        
        # Delta vs A (image-only)
        d_load_a = b['load'] - a['load']
        rel_load_a = (b['load'] - a['load']) / a['load']
        d_esc_a = b['escapes_per_1k'] - a['escapes_per_1k']
        
        # Break-even ratios R = C_escape / C_review
        # Delta Cost = C_r * d_load_boards + C_e * d_esc_boards = 0 => R* = -d_load_boards / d_esc_boards
        if d_esc_a0 <= 0:
            breakeven_a0 = "Strict Dominance (Fewer reviews & Fewer escapes)"
        else:
            breakeven_a0 = float((-d_load_a0 * 1000) / d_esc_a0)
            
        if d_esc_a <= 0:
            breakeven_a = "Strict Dominance (Fewer reviews & Fewer escapes)"
        else:
            breakeven_a = float((-d_load_a * 1000) / d_esc_a)
            
        # Cost savings for standard industrial scenarios:
        # Scenario 1: Electronics assembly line (C_review = $0.10 / box, C_escape = $25 / escaped defect => R = 250)
        # Scenario 2: High-reliability / Automotive PCB (C_review = $0.20 / box, C_escape = $100 / escaped defect => R = 500)
        cost_scenarios = {}
        for r_ratio in [50, 100, 250, 500, 1000]:
            cost_a0 = a0['reviews_per_1k'] + r_ratio * a0['escapes_per_1k']
            cost_a = a['reviews_per_1k'] + r_ratio * a['escapes_per_1k']
            cost_b = b['reviews_per_1k'] + r_ratio * b['escapes_per_1k']
            
            savings_vs_a0 = (cost_a0 - cost_b) / cost_a0 * 100
            savings_vs_a = (cost_a - cost_b) / cost_a * 100
            cost_scenarios[f"R_{r_ratio}"] = {
                "ratio": r_ratio,
                "cost_a0": float(cost_a0),
                "cost_a": float(cost_a),
                "cost_b": float(cost_b),
                "pct_savings_vs_a0": float(savings_vs_a0),
                "pct_savings_vs_a": float(savings_vs_a)
            }
            
        results.append({
            'method': method,
            'alpha': 0.05,
            'policy_A0': a0,
            'policy_A': a,
            'policy_B': b,
            'd_load_vs_A0': float(d_load_a0),
            'rel_load_pct_vs_A0': float(rel_load_a0 * 100),
            'd_load_vs_A': float(d_load_a),
            'rel_load_pct_vs_A': float(rel_load_a * 100),
            'breakeven_vs_A0': breakeven_a0,
            'breakeven_vs_A': breakeven_a,
            'cost_scenarios': cost_scenarios
        })

    with open('05_results/cost_sensitive_eval.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Make summary DataFrame
    rows = []
    for r in results:
        rows.append({
            'method': r['method'],
            'alpha': r['alpha'],
            'load_A0': r['policy_A0']['load'],
            'load_A': r['policy_A']['load'],
            'load_B': r['policy_B']['load'],
            'rel_load_vs_A0_%': r['rel_load_pct_vs_A0'],
            'rel_load_vs_A_%': r['rel_load_pct_vs_A'],
            'esc_1k_A0': r['policy_A0']['escapes_per_1k'],
            'esc_1k_A': r['policy_A']['escapes_per_1k'],
            'esc_1k_B': r['policy_B']['escapes_per_1k'],
            'breakeven_vs_A0': r['breakeven_vs_A0'],
            'breakeven_vs_A': r['breakeven_vs_A'],
            'savings_R250_vs_A0_%': r['cost_scenarios']['R_250']['pct_savings_vs_a0'],
            'savings_R250_vs_A_%': r['cost_scenarios']['R_250']['pct_savings_vs_a']
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv('05_results/cost_sensitive_summary.csv', index=False)
    print("Cost-sensitive evaluation complete!")
    print(df_out.to_string())

if __name__ == '__main__':
    compute_cost_evaluation()
