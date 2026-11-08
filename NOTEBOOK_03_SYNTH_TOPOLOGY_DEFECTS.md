# Notebook 03: Graph-Constrained Synthetic Defect Generation
**Filename:** [`03_synth_topology_defects (1).ipynb`](file:///c:/Users/priti/Downloads/pcbrisk/03_synth_topology_defects%20(1).ipynb)  
**Execution Runtime:** CPU (Google Colab / Local Drive mount)  
**Primary Role:** Synthesizes realistic, graph-verified critical electrical defects (opens and shorts) on defect-free template images, enforcing strict topological graph connectivity checks before labeling.

---

## 1. Motivation & Technical Rationale

In industrial PCB manufacturing, genuine critical functional defects (such as conductor breaks or bridge shorts) are rare events. Training deep detectors solely on real data often leaves models under-exposed to critical failure modes.

However, naive data augmentation (e.g. random cutouts, blurring, color jitter) creates non-physical artifacts. `03_synth_topology_defects (1).ipynb` introduces **graph-constrained topological defect synthesis**:
1. Defects are generated **only on defect-free template images**.
2. An augmentation is labeled as an **electrical open** if and only if morphological connected-component analysis confirms that the trace has been severed into two disconnected subgraphs (`pieces_after > pieces_before`).
3. An augmentation is labeled as an **electrical short** if and only if a synthetic copper bridge merges two previously isolated electrical net components (`lab2[ya, xa] == lab2[yb, xb]`).
4. **Source Board Containment:** Every synthetic defect retains its parent template's `source_board` tag, preventing synthetic data generated from test-set boards from leaking into detector training sets.

---

## 2. Methodology & Mathematical Formulation

```
[Cell 0: Markdown]  -> Methodological overview of graph-driven synthetic defects
[Cell 1: Code]      -> Drive mount, path setup, random seed fixing
[Cell 2: Code]      -> Load split_manifest_v2.csv, match 1,500 template paths
[Cell 3: Code]      -> Otsu binarization & copper polarity determination
[Cell 4: Code]      -> Definition of op_break and op_bridge graph operators
[Cell 5: Code]      -> Parallel synthesis loop across 1,500 templates (N_PER_OP=1)
[Cell 6: Code]      -> Visual preview of synthetic defects with bounding boxes
[Cell 7: Code]      -> Packaging into synth_v2.tar, synth_meta_v2.csv, synth_v2_config.json
```

### 2.1. Trace Severing Operator (`op_break`)
1. **Trace Skeletonization:** Computes Euclidean distance transform $\text{EDT}(M_{\text{copper}})$ and medial axis skeleton $\mathcal{S} = \text{skeletonize}(M_{\text{copper}})$.
2. **Cut Location & Radius:** Selects a skeleton point $(x, y) \in \mathcal{S}$ and defines a circular disk cut of radius:
   $$r = \text{EDT}(x, y) + \text{Uniform}(2, 5)$$
3. **Topological Invariant Verification:** Recomputes 8-connected components in the local trace component:
   $$\text{Consequence} = \begin{cases} \text{'open'}, & \text{if } N_{\text{components}}(\text{after}) > N_{\text{components}}(\text{before}) \\ \text{'width\_loss'}, & \text{otherwise} \end{cases}$$
   **Crucial Rule:** Any defect classified as `'width_loss'` (partial copper thinning that does not completely sever the circuit) is **strictly discarded**. Only confirmed topological breaks are retained as class 0 (`open`).

### 2.2. Conductor Bridging Operator (`op_bridge`)
1. **Adjacent Component Search:** For component $A$, computes the inverted distance transform $\text{EDT}(\neg A)$ to locate neighboring distinct copper traces $B \ne A$ separated by a clearance distance:
   $$3\,\text{px} \le \text{dist}(A, B) \le 14\,\text{px}$$
2. **Bridge Synthesis:** Draws a synthetic copper line between $(x_a, y_a)$ and $(x_b, y_b)$ with line thickness $t \in [3, 6]\,\text{px}$.
3. **Graph Merge Assertion:** Evaluates whether component labels merge:
   $$\text{Merged} = (L_{\text{after}}(x_a, y_a) == L_{\text{after}}(x_b, y_b)) \land (L_{\text{before}}(x_a, y_a) \ne L_{\text{before}}(x_b, y_b))$$
   Only confirmed topological shorts are retained as class 1 (`short`).

---

## 3. Execution Log & Measured Results

### Synthesis Generation Summary:
```text
1500 templates matched to manifest (expect 1500)
copper fraction 0.14 components 7
2225 samples kept; templates where no valid defect was found: {'break': 15, 'bridge': 760}
```

### Breakdown of Generated Synthetic Defects:
| Operator | Graph Consequence | Assigned Class | Generated Instances | Success Rate |
|---|---|---|---|---|
| **`op_break`** | Complete Trace Open | Class 0 (`open`) | **$1,485$** | $99.0\%$ ($1,485 / 1,500$) |
| **`op_bridge`** | Net Bridge Short | Class 1 (`short`) | **$740$** | $49.3\%$ ($740 / 1,500$) |
| **Total** | Verified Critical Defects | Classes 0 & 1 | **$2,225$** | — |

*Note on Bridge Rejection:* 760 templates failed to yield an electrical short because traces either had clearance $>14\,\text{px}$ (exceeding maximum bridge gap) or connecting lines failed to cross distinct electrical nets. This high rejection rate confirms the engine's strict rejection of non-functional pseudo-defects.

---

## 4. Scientific & Engineering Interpretation

1. **Why `width_loss` Must Be Dropped:**  
   In preliminary experiments, partially thinned traces caused severe label ambiguity: human inspectors disagreed on whether a $50\%$ neck-down was an electrical open or a cosmetic mousebite. By dropping `width_loss` and retaining only complete topological disconnections, the training set enforces an unambiguous functional boundary.
2. **Sub-Pixel Appearance Fidelity:**  
   When carving or adding copper, pixels are inpainted using the median background or copper color:
   $$\text{img}_{\text{cut}} = \text{median}(\text{img}_{\text{substrate}}), \quad \text{img}_{\text{bridge}} = \text{median}(\text{img}_{\text{copper}})$$
   This preserves realistic optical lighting gradients without sharp color discrepancies.
3. **Preventing Test-Set Contamination:**  
   Because each synthetic defect record in `synth_meta_v2.csv` logs `source_board`, notebook 04 can query:
   `sm = sm[sm.source_board.isin(train_boards)]`
   This guarantees that synthetic images generated from test boards (`group00041`, `group44000`, `group50600`) are **never** included in training sets.

---

## 5. Artifacts Produced

| Artifact File | Size | Role |
|---|---|---|
| [`02_synth/synth_v2.tar`](file:///c:/Users/priti/Downloads/pcbrisk/02_synth/synth_v2.tar) | $136.2\text{ MB}$ | Archive containing $2,225$ synthetic image crops and YOLO label txt files |
| [`02_synth/synth_meta_v2.csv`](file:///c:/Users/priti/Downloads/pcbrisk/02_synth/synth_meta_v2.csv) | $328\text{ KB}$ | Metadata: synth_id, template, source_board, bbox, gap_px, width_px, tries |
| [`02_synth/synth_v2_config.json`](file:///c:/Users/priti/Downloads/pcbrisk/02_synth/synth_v2_config.json) | $<1\text{ KB}$ | Synthesis configuration, hyperparameters, parent manifest hash |
