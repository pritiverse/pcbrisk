# Notebook 02: Dataset Restructuring, Board-Level Partitioning & YOLO Packaging
**Filename:** [`02_prepare_split (1).ipynb`](file:///c:/Users/priti/Downloads/pcbrisk/02_prepare_split%20(1).ipynb)  
**Execution Runtime:** CPU (Google Colab / Local Drive mount)  
**Primary Role:** Transforms raw DeepPCB annotations into normalized YOLO format, fixes the v1 image-level data leakage bug by establishing a strict board-level partition, generates 4 cross-validation folds, and packages assets into an immutable tarball.

---

## 1. Motivation & The v1 Data Leakage Bug Fix

### The Fatal Flaw in DeepPCB v1 Official Splits:
The canonical DeepPCB dataset distributed by Tang et al. partitions images **randomly across train and test sets without regard to board identity**. Because images within the same board group share identical copper trace layout, solder mask coloration, and etching thickness, an image-level split leaks all 11 board topologies into both the training and test sets. A vision model can simply memorize board-specific trace artifacts rather than learning generalizable defect representations.

### The v2 Architectural Fix:
1. **Archive Deprecated v1 Artifacts:** Automatically relocates all v1 files (`prepared_v1.tar`, `split_manifest_v1.csv`, `synth_v1.tar`) into `99_archive/wrong_split_v1/`.
2. **Strict Board-Level Partitioning:** DeepPCB's 11 independent manufacturing groups are partitioned at the **board level**:
   - **Train (6 Boards):** `group12000`, `group12100`, `group13000`, `group20085`, `group77000`, `group90100` ($882$ images).
   - **Validation (2 Boards):** `group12300`, `group92000` ($218$ images).
   - **Test (3 Boards):** `group00041`, `group44000`, `group50600` ($400$ images).
3. **Leave-Boards-Out Cross-Validation (Folds 0–3):** Maps boards to 4 cross-validation folds for out-of-fold risk calibration.
4. **Zero Board Overlap Assertion:** Enforces zero board intersection:
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

---

## 2. Methodology & Code Walkthrough

```
[Cell 0: Markdown]  -> Explanation of v1 bug and board-level partitioning rationale
[Cell 1: Code]      -> Drive mount, path setup, random seed fixing (seed=42)
[Cell 2: Code]      -> Archive corrupted v1 artifacts into 99_archive/wrong_split_v1/
[Cell 3: Code]      -> Enforce board permutation with RNG(42), assign train/val/test & folds
[Cell 4: Code]      -> Parse raw bounding boxes, convert to normalized YOLO format, build manifest
[Cell 5: Code]      -> Hard validation checks: defect class distribution, zero board leakage
[Cell 6: Code]      -> Export class_map.yaml, split_manifest_v2.csv, package prepared_v2.tar
[Cell 7: Markdown]  -> String dtype warning: enforce leading zeros in image IDs
```

### YOLO Bounding Box Normalization:
Raw DeepPCB annotations provide coordinates as $[x_1, y_1, x_2, y_2, \text{class}]$. Notebook 02 converts these to normalized YOLO format ($640 \times 640$ frame):
$$x_{\text{center}} = \frac{x_1 + x_2}{2 \times 640}, \quad y_{\text{center}} = \frac{y_1 + y_2}{2 \times 640}, \quad w = \frac{x_2 - x_1}{640}, \quad h = \frac{y_2 - y_1}{640}$$
with classes zero-indexed ($c \in \{0, 1, 2, 3, 4, 5\}$).

---

## 3. Execution Log & Numerical Results

### Board Partition Summary:
```text
train ['group12000', 'group12100', 'group13000', 'group20085', 'group77000', 'group90100']
val   ['group12300', 'group92000']
test  ['group00041', 'group44000', 'group50600']
```

### Image and Board Distribution across Splits:
```text
       images  boards
split                
test      400       3
train     882       6
val       218       2
```

### Defect Class Distribution across Splits:
| Split | Open (c0) | Short (c1) | Mousebite (c2) | Spur (c3) | Copper (c4) | Pin-hole (c5) | Total Defects |
|---|---|---|---|---|---|---|---|
| **Train** | $1,118$ | $929$ | $1,159$ | $966$ | $889$ | $871$ | $5,932$ |
| **Val** | $275$ | $175$ | $215$ | $203$ | $184$ | $208$ | $1,260$ |
| **Test** | $549$ | $402$ | $591$ | $456$ | $401$ | $422$ | $2,821$ |
| **Total** | **$1,942$** | **$1,506$** | **$1,965$** | **$1,625$** | **$1,474$** | **$1,501$** | **$10,013$** |

### Cross-Validation Fold Distribution:
* **Fold 0:** 3 boards (`group00041`, `group13000`, `group77000`)
* **Fold 1:** 3 boards (`group44000`, `group20085`, `group90100`)
* **Fold 2:** 3 boards (`group50600`, `group12000`, `group92000`)
* **Fold 3:** 2 boards (`group12300`, `group12100`)

### Manifest Cryptographic Hash:
```text
manifest hash: 1f5bffd29c0d... (SHA-256)
```

---

## 4. Scientific & Engineering Interpretation

1. **Guaranteed Generalization to Unseen Boards:**  
   Because no board in the test set shares any physical layout, trace routing, or lighting properties with the training boards, the resulting model performance reflects genuine industrial transfer across manufacturing lots.
2. **Defect Representation Balance:**  
   As verified by the assertion `assert (tab > 0).all().all()`, every split contains a healthy representation of all 6 defect classes, ensuring that evaluation metrics are not skewed by zero-instance defect categories.
3. **Data Integrity & Numeric String Truncation Gotcha:**  
   Cell 7 highlights an insidious Python gotcha: standard `pd.read_csv()` parses numeric string IDs like `00041000` into integer `41000`, destroying file path matching. Notebook 02 enforces `dtype={'image_id': str}` throughout, with an automated round-trip integrity assertion.

---

## 5. Artifacts Produced

| Artifact File | Size | Role |
|---|---|---|
| [`01_prepared/prepared_v2.tar`](file:///c:/Users/priti/Downloads/pcbrisk/01_prepared/prepared_v2.tar) | $52.5\text{ MB}$ | YOLO-formatted images and label text files |
| [`01_prepared/split_manifest_v2.csv`](file:///c:/Users/priti/Downloads/pcbrisk/01_prepared/split_manifest_v2.csv) | $62.3\text{ KB}$ | Full metadata: image_id, source_board, split, fold, defect counts |
| [`01_prepared/class_map.yaml`](file:///c:/Users/priti/Downloads/pcbrisk/01_prepared/class_map.yaml) | $79\text{ B}$ | Class ID mapping (0: open, 1: short, ..., 5: pin-hole) |
| [`01_prepared/manifest_v2.sha256`](file:///c:/Users/priti/Downloads/pcbrisk/01_prepared/manifest_v2.sha256) | $64\text{ B}$ | SHA-256 integrity hash (`1f5bffd29c0d...`) |
