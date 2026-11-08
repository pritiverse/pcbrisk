"""eval_mondrian_calibration.py - Compares global split-conformal calibration with group-conditional (Mondrian) conformal calibration on critical opens (cls 0) vs critical shorts (cls 1)."""
import pandas as pd
import numpy as np
import json
import os

def run_mondrian_analysis():
    # Load engine validation data where ground truth defects and graph detections are matched
    df = pd.read_csv('05_results/engine_validation.csv')
    crit = df[df.cls.isin([0, 1])].copy()
    
    # Construct a continuous score proxy: missing pixels for opens, excess pixels for shorts
    # Real conformal scores combine confidence + graph evidence.
    # In Policy B, score for open is driven by missing pixels & g_open; for short by excess pixels & g_short.
    crit['score'] = np.where(crit.cls == 0, 
                             np.log1p(crit.missing) * 0.5 + crit.g_open.astype(int) * 2.0,
                             np.log1p(crit.excess) * 0.5 + crit.g_short.astype(int) * 2.0)
    
    alpha = 0.05
    
    # 1. Global Pooled Conformal Threshold
    scores_all = crit.score.values
    k_global = int(np.floor(alpha * (len(scores_all) + 1)))
    tau_global = np.sort(scores_all)[max(0, k_global - 1)]
    
    # Measure group-specific miss rates under global threshold
    miss_global_open = (crit[crit.cls == 0].score < tau_global).mean()
    miss_global_short = (crit[crit.cls == 1].score < tau_global).mean()
    miss_global_overall = (crit.score < tau_global).mean()
    
    # 2. Mondrian / Group-Conditional Conformal Calibration
    # Calibrate separate tau for opens and shorts
    scores_open = crit[crit.cls == 0].score.values
    scores_short = crit[crit.cls == 1].score.values
    
    k_open = int(np.floor(alpha * (len(scores_open) + 1)))
    k_short = int(np.floor(alpha * (len(scores_short) + 1)))
    
    tau_mondrian_open = np.sort(scores_open)[max(0, k_open - 1)]
    tau_mondrian_short = np.sort(scores_short)[max(0, k_short - 1)]
    
    miss_mondrian_open = (crit[crit.cls == 0].score < tau_mondrian_open).mean()
    miss_mondrian_short = (crit[crit.cls == 1].score < tau_mondrian_short).mean()
    miss_mondrian_overall = ((crit.cls == 0) & (crit.score < tau_mondrian_open) | 
                             (crit.cls == 1) & (crit.score < tau_mondrian_short)).mean()
    
    results = {
        "alpha_target": alpha,
        "sample_counts": {
            "opens_class_0": int(len(scores_open)),
            "shorts_class_1": int(len(scores_short)),
            "total_critical": int(len(crit))
        },
        "global_calibration": {
            "tau_global": float(tau_global),
            "overall_miss_rate": float(miss_global_overall),
            "open_miss_rate": float(miss_global_open),
            "short_miss_rate": float(miss_global_short),
            "group_disparity": float(abs(miss_global_open - miss_global_short))
        },
        "mondrian_group_conditional_calibration": {
            "tau_open": float(tau_mondrian_open),
            "tau_short": float(tau_mondrian_short),
            "overall_miss_rate": float(miss_mondrian_overall),
            "open_miss_rate": float(miss_mondrian_open),
            "short_miss_rate": float(miss_mondrian_short),
            "group_disparity": float(abs(miss_mondrian_open - miss_mondrian_short))
        }
    }
    
    os.makedirs('05_results', exist_ok=True)
    with open('05_results/mondrian_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Mondrian Calibration Evaluation Complete:")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    run_mondrian_analysis()
