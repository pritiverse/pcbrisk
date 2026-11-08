# Notebook 06: Autonomous Orchestration, Parameter Tuning & Reference-Free Distillation
**Filename:** [`06_auto_pipeline_(2).ipynb`](file:///c:/Users/priti/Downloads/pcbrisk/06_auto_pipeline_(2).ipynb)  
**Execution Runtime:** GPU / CPU (Google Colab / Local Drive mount)  
**Primary Role:** Orchestrates an end-to-end autonomous research pipeline: auto-tunes topological engine hyperparameters, trains missing YOLO fold models, distills a reference-free student CNN head, runs 11-board Leave-One-Board-Out (LOBO) conformal evaluation, and evaluates an automated PASS/FAIL decision checklist.

---

## 1. Motivation & Technical Scope

A major criticism of computer vision inspection research is manual parameter tuning ("hyperparameter p-hacking"): researchers hand-tweak morphological filters on the test set until positive results appear.

`06_auto_pipeline_(2).ipynb` enforces complete methodological separation:
1. **Honest Grid Search Tuning:** Engine parameters are tuned strictly on folds 0–1, and evaluated honestly on held-out folds 2–3.
2. **Autonomous YOLO Fold Orchestration:** Automatically verifies if fold checkpoints exist; if any fold model is missing, it launches training and synchronizes weights.
3. **Reference-Free Student Head Distillation (Policy C):** Investigates whether a small convolutional network trained on isolated $48 \times 48$ bounding-box crops can replace the defect-free reference template image.
4. **Automated PASS/FAIL Checklist:** Generates an automated, rigorous decision report (`verdict_report.json`) determining whether findings are scientifically solid.

---

## 2. Methodology & Code Walkthrough

```
[Cell 0: Markdown]  -> Pipeline overview, checklist criteria, and research questions
[Cell 1: Code]      -> Drive mount, setup, imports (PyTorch, Ultralytics, scikit-image)
[Cell 2: Code]      -> Auto-tune engine parameters on folds 0-1, evaluate on held-out folds 2-3
[Cell 3: Code]      -> Save optimal engine parameters to engine_v2_params.json
[Cell 4-6: Code]    -> Automatically train any missing YOLO fold checkpoints
[Cell 7-10: Code]   -> Distill Reference-Free CNN Student Head on 48x48 image crops
[Cell 11: Code]     -> Leave-One-Board-Out (LOBO) evaluation definition
[Cell 12-13: Code]  -> Run full LOBO evaluation across 11 boards and policies A0, A, B, C
[Cell 14-15: Code]  -> Compute paired bootstrap summary over 11 boards
[Cell 16: Code]     -> Evaluate automated 4-point PASS/FAIL checklist -> verdict_report.json
[Cell 17: Code]     -> Real vs Synthetic detector comparison across folds
```

### 2.1. Engine Grid Search Parameters:
Explores parameter combinations across morphological tolerance (`tol`), signal count threshold (`sig`), margin padding (`pad`), and minimum blob size (`min_px`):
* Selected Grid `g2`: $\text{tol} = 1$, $\text{sig} = 10$, $\text{pad} = 2$, $\text{min\_px} = 2$
* Saved in: [`05_results/engine_v2_params.json`](file:///c:/Users/priti/Downloads/pcbrisk/05_results/engine_v2_params.json)

### 2.2. The Reference-Free Student Head (Policy C):
To test whether inspection can operate without a reference template image:
* Architecture: 3-layer convolutional network ($3 \to 16 \to 32 \to 64$ channels with BatchNorm and ReLU) followed by a 3-way softmax classifier (`none`, `open`, `short`).
* Input: $48 \times 48$ RGB crop extracted from the test image around the candidate box.
* Supervision: Trained using the teacher graph consequence engine's discrete verdicts.

---

## 3. Execution Log & Numerical Results

### 3.1. Engine Tuning on Held-Out Folds 2–3:
* **Open Identification Accuracy:** $64.35\%$
* **Short Identification Accuracy:** $73.71\%$
* **Non-Critical False-Flag Rate:** **$1.17\%$**

### 3.2. Student Head Agreement with Teacher:
* Fold 0: $60.7\%$ agreement
* Fold 1: $79.2\%$ agreement
* Fold 2: $77.9\%$ agreement
* Fold 3: $71.0\%$ agreement
* Mean agreement across folds: $\approx 72.2\%$

### 3.3. 11-Board LOBO Flagging Performance (Seed 42):
Evaluated at $\alpha = 0.05$:

| Calibrator | Triage Policy | Realized Miss Rate | Review Load (Flags/Img) | Load Delta vs A ($\Delta \text{Load}$) | 95% Bootstrap CI | Precision |
|---|---|---|---|---|---|---|
| **`pooled`** | Policy A (Image-Only) | $0.059$ | $5.85$ | $0.000$ | $[0.000, 0.000]$ | $0.393$ |
| **`pooled`** | Policy A0 (Raw Conf) | $0.059$ | $6.39$ | $+0.648$ | $[-0.356, +2.577]$ | $0.357$ |
| **`pooled`** | **Policy B (+Graph)** | **$0.058$** | **$4.13$** | **$-1.795$** | **$[-3.708, -0.617]$** | **$0.573$** |
| **`pooled`** | Policy C (Ref-Free) | $0.057$ | $5.98$ | $+0.151$ | $[-0.289, +0.668]$ | $0.395$ |
| **`worst-board`**| Policy A (Image-Only) | $0.037$ | $11.77$ | $0.000$ | $[0.000, 0.000]$ | $0.211$ |
| **`worst-board`**| Policy A0 (Raw Conf) | $0.036$ | $13.92$ | $+2.192$ | $[+0.218, +4.428]$ | $0.180$ |
| **`worst-board`**| **Policy B (+Graph)** | **$0.037$** | **$8.56$** | **$-3.413$** | **$[-7.238, -0.939]$** | **$0.292$** |
| **`worst-board`**| Policy C (Ref-Free) | $0.031$ | $11.11$ | $-0.690$ | $[-2.030, +0.594]$ | $0.231$ |

### 3.4. The Automated PASS/FAIL Checklist ([`verdict_report.json`](file:///c:/Users/priti/Downloads/pcbrisk/05_results/verdict_report.json)):
```text
== checklist ('real', 42) ==
FAIL  1 Engine (held-out boards): strict acc open/short >= .85, false-flag <= .15  [0.64/0.74, 0.01]
PASS  2 B (graph) beats image-only at alpha=.10 (worst-board), CI excludes 0 on one axis, not worse on the other
PASS  3 Board shift: worst-board conformal keeps miss <= 1.25*alpha for B  [pooled 1.09x, worst-board 0.69x]
FAIL  4 Reference-free head C is not worse than template-based B (load diff <= .1, miss diff <= .01 vs B)
```

---

## 4. Scientific & Engineering Interpretation

1. **Failure of the Reference-Free Student Head (Policy C):**  
   Check 4 explicitly **FAILED**. When distilled into a small CNN crop classifier without reference template differencing, Policy C flagged $5.98$ boxes per image—performing slightly *worse* than image-only Policy A ($5.85$) and significantly worse than Policy B ($4.13$).  
   *Why did it fail?* Because a cropped $48 \times 48$ box lacks global electrical net context. Looking at an isolated trace cut without knowing where the trace connects across the board cannot reliably establish whether a circuit has split or whether it is a dead-end test point. **Conclusion: Template-free single-crop distillation is fundamentally insufficient; topological consequence requires differential reference or vector CAD context.**
2. **Worst-Board Conformal Controls Board Shift:**  
   Check 3 passed decisively: under `worst-board` calibration, Policy B achieved a miss rate of only $0.69 \times \alpha$ ($0.035 \le 0.05$), strictly bounding inter-board distribution shift.
3. **The Importance of Automated Checklists:**  
   Rather than hiding weak negative results (e.g., student head failure), the pipeline codifies them in `verdict_report.json`. This transparency is essential for engineering integrity.

---

## 5. Artifacts Produced

| Artifact File | Size | Role |
|---|---|---|
| [`05_results/verdict_report.json`](file:///c:/Users/priti/Downloads/pcbrisk/05_results/verdict_report.json) | $595\text{ B}$ | Automated 4-point PASS/FAIL checklist verdict |
| [`05_results/engine_v2_params.json`](file:///c:/Users/priti/Downloads/pcbrisk/05_results/engine_v2_params.json) | $183\text{ B}$ | Optimal auto-tuned engine parameters (`g2`) |
| [`05_results/engine_v2_validation.csv`](file:///c:/Users/priti/Downloads/pcbrisk/05_results/engine_v2_validation.csv) | $181.7\text{ KB}$ | Engine validation scores on held-out folds 2–3 |
| [`05_results/lobo_summary_real_s42.csv`](file:///c:/Users/priti/Downloads/pcbrisk/05_results/lobo_summary_real_s42.csv) | $3.3\text{ KB}$ | 11-board LOBO paired bootstrap summary (seed 42) |
