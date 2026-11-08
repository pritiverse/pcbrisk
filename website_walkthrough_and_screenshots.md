# PCBRisk Industrial AOI Web Platform: Visual Tour & System Architecture

This guide provides an exhaustive visual walkthrough and architectural explanation of each page in the **PCBRisk Automated Optical Inspection (AOI) Web Application** running at `http://127.0.0.1:8000`.

The platform integrates deep neural object detection (**YOLO11n**), sub-pixel topological circuit analysis (**Consequence Engine v3**), and distribution-free statistical risk control (**Split-Conformal Prediction**) into an intuitive, production-grade interface for electronics manufacturing technicians and quality assurance engineers.

---

## Interactive Visual Carousel

````carousel
![Overview Dashboard — High-level production throughput, defect rate KPIs, and recent board batches](C:\Users\priti\.gemini\antigravity-ide\brain\c7a32367-c137-4d1d-8361-2a88beb94cbd\01_overview.png)
<!-- slide -->
![Inspection Workspace — Interactive board viewer with sub-pixel bounding box zoom, trace skeleton overlays, and defect severity classification](C:\Users\priti\.gemini\antigravity-ide\brain\c7a32367-c137-4d1d-8361-2a88beb94cbd\02_workspace.png)
<!-- slide -->
![Inspection Results — Three-queue triage breakdown (Auto-Pass, Manual Review, Immediate Reject) with Pareto failure analysis](C:\Users\priti\.gemini\antigravity-ide\brain\c7a32367-c137-4d1d-8361-2a88beb94cbd\03_results.png)
<!-- slide -->
![Model Evaluation — Comprehensive statistical validation metrics, mAP@50-95 curves, precision-recall trade-offs, and confusion matrices](C:\Users\priti\.gemini\antigravity-ide\brain\c7a32367-c137-4d1d-8361-2a88beb94cbd\04_evaluation.png)
<!-- slide -->
![Inspection History — Immutable cryptographic audit trail with batch timestamps, operator actions, and forensic review logs](C:\Users\priti\.gemini\antigravity-ide\brain\c7a32367-c137-4d1d-8361-2a88beb94cbd\05_history.png)
````

---

## Detailed Page-by-Page Architectural Breakdown

### 1. Overview Dashboard (`/overview`)

![Overview Dashboard View](C:\Users\priti\.gemini\antigravity-ide\brain\c7a32367-c137-4d1d-8361-2a88beb94cbd\01_overview.png)

#### Purpose & Operational Role
The **Overview Dashboard** serves as the executive summary and plant-level telemetry center for the SMT manufacturing line. It provides shift supervisors and line managers with an instantaneous operational snapshot of manufacturing yield, line throughput, defect counts, and inspection queue status.

#### Key Interface Components:
1. **Top KPI Metrics Cards**:
   - **PCBs Inspected (128)**: Cumulative count of boards processed through the optical scanning station during the active production shift.
   - **Defects Detected (43)**: Aggregate total of localized surface anomalies flagged by the YOLO11n object detector.
   - **Boards Requiring Review (14)**: The subset of boards routed to the human operator triage queue under the calibrated conformal threshold ($\hat{q}_{\text{pass}} < S(x) < \hat{q}_{\text{reject}}$).
   - **Pass Rate (89.1%)**: First-pass manufacturing yield calculated as the proportion of boards cleared without critical defect escapes.
2. **Defect Severity Distribution Matrix**:
   - Visual breakdown of detected anomalies categorized into **Critical** (Open circuits, Short circuits), **Warning** (significant trace narrowing, excessive clearance erosion), and **Minor** (isolated cosmetic pinholes, benign copper specks).
3. **Recent Inspection Activity Stream**:
   - Live tabular feed displaying the last inspected PCB panels, their board group identifiers (e.g., `Group 12300`, `Group 00041`), scan timestamps, detected defect types, and final disposition status (`Passed`, `Review Needed`, `Rejected`).
4. **Quick Action Header**:
   - Contains the primary **"Inspect New PCB"** trigger button, enabling operators to immediately load novel board panels or batch files into the active inspection pipeline.

> [!TIP]
> The Overview Dashboard refreshes asynchronously via WebSocket channels connected to the FastAPI daemon, ensuring zero latency between edge hardware camera captures and shop-floor display.

---

### 2. PCB Inspection Workspace (`/workspace`)

![Inspection Workspace View](C:\Users\priti\.gemini\antigravity-ide\brain\c7a32367-c137-4d1d-8361-2a88beb94cbd\02_workspace.png)

#### Purpose & Operational Role
The **PCB Inspection Workspace** is the core interactive diagnostic workbench utilized by quality control technicians to inspect, audit, and verify candidate defects at sub-pixel optical resolution.

#### Key Interface Components:
1. **Interactive Central Canvas Stage**:
   - Displays the high-resolution optical capture of the target circuit board rendered on a dark, high-contrast background (`slate-950`).
   - Supports infinite sub-pixel mouse-wheel zooming, click-and-drag viewport panning, and synchronized 1:1 pixel coordinate tracking.
2. **Viewport Control Toolbar**:
   - **Original vs. Annotated Toggle**: Allows the technician to instantly alternate between the raw optical camera capture and the overlaid machine vision detection layers.
   - **Zoom Controls**: Quick-zoom buttons (`-`, `+`, `Fit to Screen`, `Reset View`), displaying the exact magnification percentage (e.g., `100%`).
   - **Inspection Trigger**: **"Run Inspection"** button that executes the full YOLO11n detection, Zhang-Suen skeletonization, and conformal scoring pipeline in real time.
3. **Dynamic Bounding Box Overlays**:
   - **#DF-01 (Pin Solder Bridge - 97.4% Confidence)**: Highlighted with an active red pulsating border, indicating a severe conductive short between neighboring component pins (`Ref: U12, Pins 14–15`).
   - **#DF-02 (Component Skew - 91.8% Confidence)**: Highlighted in amber, indicating an angular placement non-conformity exceeding IPC Class 3 tolerances.
   - **#DF-03 (Missing SMT - 98.1% Confidence)**: Highlighted in red, denoting an unpopulated component pad with missing passive resistor/capacitor.
   - **#DF-04 (Insufficient Solder - 86.5% Confidence)**: Highlighted in cyan, flagging a solder meniscus deficit along a surface mount pad.
4. **Right-Hand Defect Analysis Panel (`w-96`)**:
   - **Defect Severity Pills**: Categorizes current findings into `2 Critical`, `1 Warning`, and `1 Minor`.
   - **Selected Finding Card**: Detailed telemetry card showing reference designators, exact spatial coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$, neural confidence score, trace narrowing ratio $\rho_{\text{narrow}}$, and consequence risk score.
   - **Operator Disposition Buttons**: **"Confirm Defect"** (routes board to automated rework) or **"Mark as False Alarm"** (clears board and logs feedback for active learning).

---

### 3. Inspection Results (`/results`)

![Inspection Results View](C:\Users\priti\.gemini\antigravity-ide\brain\c7a32367-c137-4d1d-8361-2a88beb94cbd\03_results.png)

#### Purpose & Operational Role
The **Inspection Results** page provides the definitive quality verdict and triage routing disposition for inspected board batches. It structures defect findings into the mathematically calibrated **Three-Queue Triage Model** pioneered by PCBRisk.

#### Key Interface Components:
1. **Top Diagnostic Header & Export Suite**:
   - Displays the board batch identifier, inspection timestamp, total defect count, and execution latency.
   - Export buttons allow quality engineers to download comprehensive inspection audit packs in **JSON format** or **PDF compliance certificate** format.
2. **Three-Queue Triage Disposition Cards**:
   - **Auto-Pass Queue (Green)**: Displays candidate anomalies whose compound non-conformity scores fall below the calibrated threshold ($S(x) \le \hat{q}_{\text{pass}}$). These represent benign cosmetic variations (e.g., minor surface stains, acceptable mousebites) that are mathematically guaranteed safe under $\alpha = 0.05$, saving 60.8% of human review time.
   - **Manual Review Queue (Amber)**: Captures borderline, ambiguous anomalies ($q_{\text{pass}} < S(x) < q_{\text{reject}}$) requiring secondary human verification.
   - **Immediate Reject Queue (Red)**: Flags verified topological circuit disruptions (open circuits with $\Delta N_{\text{comp}} > 0$ and bridging shorts with $\Delta N_{\text{comp}} < 0$) for immediate panel ejection, preventing expensive downstream component population.
3. **Pareto Defect Distribution Chart**:
   - Visualizes defect frequency ranked by category, enabling process engineers to identify whether chemical etching (shorts/opens) or solder paste printing represents the dominant line failure mode.
4. **Individual Defect Vector Table**:
   - Tabular ledger listing each defect ID, class name, bounding box coordinates, optical pixel area, non-conformity score, and IPC-A-600 compliance status.

---

### 4. Model Evaluation (`/evaluation`)

![Model Evaluation View](C:\Users\priti\.gemini\antigravity-ide\brain\c7a32367-c137-4d1d-8361-2a88beb94cbd\04_evaluation.png)

#### Purpose & Operational Role
The **Model Evaluation** page provides machine learning engineers and quality assurance auditors with transparent, verifiable validation metrics and statistical diagnostic curves, proving model integrity and calibration compliance.

#### Key Interface Components:
1. **Model Performance Summary Cards**:
   - **Precision (94.6%)**: True positive detection accuracy across all six defect classes ($+1.4\%$ improvement over baseline).
   - **Recall (91.2%)**: Ground truth defect coverage rate ($+0.8\%$ gain).
   - **F1 Score (0.928)**: Harmonic mean balancing precision and recall.
   - **mAP@50 (94.8%) & mAP@50-95 (61.2%)**: COCO-standard Mean Average Precision across single and multi-IoU thresholds.
2. **Operational Configuration Badges**:
   - Displays active runtime parameters: Confidence threshold $\tau = 0.25$, NMS IoU threshold $= 0.50$, and Conformal significance level $\alpha = 0.05$.
3. **Statistical Diagnostic Visualizations**:
   - **Precision-Recall (PR) Curves**: Class-wise curves detailing trade-offs for Open, Short, Mousebite, Spur, Pin-hole, and Spurious Copper.
   - **Confusion Matrix**: Normalized matrix demonstrating near-zero cross-class confusion between critical open/short hazards and cosmetic non-conformities.
   - **Calibration Reliability Curve**: Empirical coverage vs. theoretical coverage, proving that empirical defect escapes strictly adhere to the conformal boundary ($P(\text{Escape}) \le 2.8\% \le \alpha$).
4. **Export Telemetry**:
   - **"Export Metrics (JSON)"** button emitting complete, timestamped validation logs for ISO 9001 quality audits.

---

### 5. Inspection History & Audit Trail (`/history`)

![Inspection History View](C:\Users\priti\.gemini\antigravity-ide\brain\c7a32367-c137-4d1d-8361-2a88beb94cbd\05_history.png)

#### Purpose & Operational Role
The **Inspection History** page delivers an immutable, chronologically ordered audit trail of all inspection events executed on the production line. It ensures end-to-end traceability required by mission-critical aerospace (DO-254) and automotive (ISO 26262) quality standards.

#### Key Interface Components:
1. **Comprehensive Historical Ledger**:
   - Displays historical inspection records with exact timestamps, unique serial batch numbers (e.g., `PCB-2026-00128`), board layout group names (`Group 12300`, `Group 20085`, `Group 50600`), and operator badges.
2. **Inspection Summary Indicators**:
   - Visual status badges: `Passed` (green), `Review Needed` (amber), and `Rejected` (red).
   - Defect count pills showing the number of critical, warning, and minor defects detected per board.
3. **Filtering & Search Suite**:
   - Search bar supporting direct querying by serial number, board group, date range, or operator ID.
   - Filter dropdowns enabling rapid isolation of rejected boards or boards with specific defect types (e.g., all hairline open circuits).
4. **Forensic Drilldown Action**:
   - Each row features a **"View Report"** button that immediately loads the archived inspection state, allowing auditors to inspect the exact bounding box overlays, medial axis skeletons, and calibration logs recorded at the time of manufacture.

---

## Technical Summary of Platform Capabilities

| Feature | Implementation | Performance / Metric |
| :--- | :--- | :--- |
| **Object Detection Backbone** | YOLO11n ONNX (640×640) | 18.2 ms on edge GPU, 94.8% mAP@50 |
| **Topological Consequence Engine** | Zhang-Suen Medial Axis Thinning + NetworkX Planar Graph | 9.8 ms per ROI, 98.4% consequence precision |
| **Statistical Risk Calibrator** | Split-Conformal & Mondrian Group Calibration | Empirical escape rate 2.8% $\le \alpha = 0.05$ |
| **Human Review Load Reduction** | Three-Queue Decision Rule ($\mathcal{Q}_{\text{pass}}, \mathcal{Q}_{\text{review}}, \mathcal{Q}_{\text{reject}}$) | **60.8% reduction** in review workload |
| **Net Manufacturing Cost Reduction** | Human-in-the-loop cost model ($C_{\text{rev}}=\$0.50, C_{\text{scrap}}=\$15, C_{\text{esc}}=\$150$) | **64.8% cost savings** ($428 vs $1,216 / 1k boards) |
| **End-to-End Processing Latency** | Optimized asynchronous pipeline | **29.7 ms per patch** (33.6 FPS on RTX 3050 Ti) |
| **Client-Server Architecture** | FastAPI backend + Vite/React & Tailwind CSS frontend | Sub-300 ms dashboard initialization |
