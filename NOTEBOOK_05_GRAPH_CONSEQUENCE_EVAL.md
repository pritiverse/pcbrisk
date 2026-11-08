# Notebook 05: Consequence Engine Validation & Pilot Split-Conformal Evaluation
**Filename:** [`05_graph_consequence_eval (1).ipynb`](file:///c:/Users/priti/Downloads/pcbrisk/05_graph_consequence_eval%20(1).ipynb)  
**Execution Runtime:** CPU / GPU (Google Colab / Local Drive mount)  
**Primary Role:** Validates the differential topological consequence engine against ground-truth defect labels (Part A) and conducts the pilot split-conformal triage experiment across 4 cross-validation folds (Part B) to test whether graph features reduce inspector workload at bounded miss rates $\alpha$.

---

## 1. Motivation & Scientific Hypotheses

Standard object detectors output bounding boxes with continuous confidence scores $s \in [0, 1]$. In production AOI, factory managers set a low threshold $\tau$ to avoid missing critical functional circuit defects. However, this causes immense false-alarm fatigue: inspectors must visually verify dozens of cosmetic defects to find rare opens/shorts.

`05_graph_consequence_eval (1).ipynb` tests two foundational scientific hypotheses:
1. **Hypothesis 1 (Topological Engine Accuracy):** Can a lightweight image-differential graph engine (Otsu binarization + morphological skeletonization + local connected components) distinguish between functional circuit defects (opens/shorts) and harmless cosmetic blemishes (mousebites, spurs, pin-holes)?
2. **Hypothesis 2 (Conformal Workload Reduction):** At a mathematically bounded critical-defect miss rate $\le \alpha$ (e.g. $\alpha = 0.05$), does incorporating topological consequence features (Policy B) reduce the number of flagged boxes per image compared to detector confidence alone (Policy A0) or image-only features (Policy A)?

---

## 2. Methodology & Code Walkthrough

```
[Cell 0: Markdown]  -> Methodological scope (Part A engine accuracy, Part B conformal load)
[Cell 1: Code]      -> Drive mount, setup, seed fixing (seed=42)
[Cell 2: Code]      -> Define Engine v1 (global Otsu, dilation, connected components)
[Cell 3: Code]      -> Part A: Evaluate engine against 10,013 ground-truth defect annotations
[Cell 4: Code]      -> Compute accuracy on opens, shorts, and false-alarm rate on cosmetic defects
[Cell 5-7: Code]    -> Cache and visual confirmation of differential masks
[Cell 8: Code]      -> Out-of-fold inference on 4 CV folds (23,180 predicted boxes)
[Cell 9: Code]      -> Extract graph features for all 23,180 candidate boxes
[Cell 10: Code]     -> Match predicted boxes to critical ground-truth defects (IoU >= 0.5)
[Cell 11: Code]     -> Fit Logistic Regression scoring models and run split-conformal calibration
[Cell 12: Code]     -> Aggregate results table across folds and alphas (0.02, 0.05, 0.10)
[Cell 13: Markdown] -> Scientific interpretation guide: exchangeability limits and future steps
```

### 2.1. Feature Engineering & Triage Policies:
For each predicted candidate box $i$:
* **Baseline Image Features (Policy A):**
  - $\text{conf}$: Raw detector confidence
  - $\text{crit}$: Indicator for open or short ($\text{cls} \in \{0, 1\}$)
  - $\text{conf\_crit} = \text{conf} \times \text{crit}$
  - $\text{logarea} = \log((x_2 - x_1) \cdot (y_2 - y_1))$
* **Topological Consequence Features (Policy B):**
  - All Policy A features $+$
  - $g_{\text{open}}, g_{\text{short}}$: Binary indicator from local consequence engine
  - $g_{\text{agree}}$: Indicator that detector class matches graph consequence ($(\text{cls}=0 \land g_{\text{open}}) \lor (\text{cls}=1 \land g_{\text{short}})$)
  - $g_{\text{cross}}$: Cross-consequence indicator ($(\text{cls}=0 \land g_{\text{short}}) \lor (\text{cls}=1 \land g_{\text{open}})$)
  - $\log(1 + \text{missing\_px}), \log(1 + \text{excess\_px})$
  - $\text{conf\_agree} = \text{conf} \times g_{\text{agree}}$

### 2.2. Split-Conformal Calibration Formulation:
For calibration fold $\mathcal{D}_{\text{cal}}$, the non-conformity threshold $\tau$ at target miss rate $\alpha$ is computed as the empirical quantile:
$$k = \lfloor \alpha \cdot (|\mathcal{D}_{\text{cal}}| + 1) \rfloor, \quad \tau = \text{Sort}(s_{\text{cal}})[k - 1]$$
A test box is flagged for human review if $s_{\text{test}} \ge \tau$.

---

## 3. Execution Log & Numerical Results

### Part A: Engine Accuracy on Ground-Truth Defects:
* **Open Defect Identification Accuracy:** **$64.35\%$**
* **Short Defect Identification Accuracy:** **$73.71\%$**
* **Non-Critical False-Flag Rate:** **$1.17\%$**  
  *(Crucial finding: Cosmetic pin-holes, mousebites, and spurs were correctly dismissed as non-functional $98.83\%$ of the time).*

### Part B: Pilot 4-Fold Conformal Triage Results:
Evaluated on $23,180$ candidate boxes ($3,448$ critical GT defects):

| Target Miss $\alpha$ | Triage Policy | Realized Miss Rate | Flagged Boxes / Image | Load Std Dev | Precision |
|---|---|---|---|---|---|
| **$\alpha = 0.02$** | Policy A0 (Raw Conf) | $0.042$ | $9.82$ | $7.90$ | $0.429$ |
| **$\alpha = 0.02$** | Policy A (Image-Only) | $0.043$ | $10.03$ | $7.78$ | $0.421$ |
| **$\alpha = 0.02$** | **Policy B (+Graph)** | **$0.039$** | **$9.79$** | **$7.87$** | **$0.417$** |
| **$\alpha = 0.05$** | Policy A0 (Raw Conf) | $0.087$ | $5.79$ | $4.00$ | $0.571$ |
| **$\alpha = 0.05$** | Policy A (Image-Only) | $0.090$ | $4.19$ | $1.98$ | $0.640$ |
| **$\alpha = 0.05$** | **Policy B (+Graph)** | **$0.070$** | **$3.94$** | **$1.70$** | **$0.677$** |
| **$\alpha = 0.10$** | Policy A0 (Raw Conf) | $0.178$ | $2.19$ | $0.53$ | $0.884$ |
| **$\alpha = 0.10$** | Policy A (Image-Only) | $0.178$ | $2.20$ | $0.51$ | $0.884$ |
| **$\alpha = 0.10$** | **Policy B (+Graph)** | **$0.137$** | **$2.19$** | **$0.37$** | **$0.911$** |

### Policy B vs Policy A Delta ($\Delta \text{Load} = \text{Load}_B - \text{Load}_A$):
* **$\alpha = 0.05$:** $\Delta \text{Load} = -0.251 \pm 0.826$ boxes/image (Policy B flags fewer in 2 of 4 folds; reduces mean load from $5.79$ under A0 to $3.94$ under B).
* **$\alpha = 0.10$:** $\Delta \text{Load} = -0.006 \pm 0.240$ boxes/image (detector confidence alone suffices; graph features add marginal delta).

---

## 4. Scientific & Engineering Interpretation

1. **Topological Engine as an Asymmetric Filter:**  
   The consequence engine exhibits extreme precision in identifying non-critical defects: its false-flag rate on cosmetic defects is only $1.17\%$. Even though open/short accuracy is moderate ($64\%/74\%$, constrained by raster resolution), its ability to safely ignore cosmetic mousebites and pin-holes eliminates the overwhelming majority of false alarms.
2. **Realized Miss Rate vs. Nominal Alpha:**  
   Notice that the realized miss rate at $\alpha=0.05$ landed at $0.070$ for Policy B (and $0.090$ for Policy A). Why did it exceed $0.05$?  
   *Because standard split-conformal guarantees assume exchangeability between calibration and test samples.* When boards have distinct trace thicknesses and lighting, exchangeability is violated across folds.
3. **The Catalyst for Notebook 07 (LOBO):**  
   The large variance across 4 folds ($\text{std} = 0.826$ at $\alpha=0.05$) revealed that 4 folds were too coarse: each fold contained a heterogeneous mixture of 2 to 3 boards. This observation directly motivated Notebook 07's transition to **11-board Leave-One-Board-Out (LOBO)** cross-fitting and board-shift-inflated calibration.

---

## 5. Artifacts Produced

| Artifact File | Size | Role |
|---|---|---|
| [`05_results/flagging_results_real_s42.csv`](file:///c:/Users/priti/Downloads/pcbrisk/05_results/flagging_results_real_s42.csv) | $2.9\text{ KB}$ | Fold-level conformal triage metrics across policies and alphas |
| [`05_results/engine_validation.csv`](file:///c:/Users/priti/Downloads/pcbrisk/05_results/engine_validation.csv) | $338.7\text{ KB}$ | Ground-truth defect validation ledger ($10,013$ instances) |
| [`05_results/oof_preds_real_s42.csv`](file:///c:/Users/priti/Downloads/pcbrisk/05_results/oof_preds_real_s42.csv) | $1.64\text{ MB}$ | Out-of-fold predicted bounding boxes ($23,180$ boxes) |
