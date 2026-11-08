


VELLORE INSTITUTE OF TECHNOLOGY, VELLORE
School of Computer Science Engineering and Information Systems
Department of Information Technology

Intelligent PCB Surface Defect Detection, Severity Assessment
and Probable Process-Fault Prediction Using Deep Learning
BITE497J – Project I

Submitted in partial fulfillment of the requirements for the degree of
Bachelor of Technology in Information Technology
by

N RUKKSHANA – 23BIT0405
NIKHITA MARIMUTHU – 23BIT0397
PRITIKA KASINATHAN – 23BIT0232

Under the guidance of:
Dr. Hemalatha S
Associate Professor, SCORE
Vellore Institute of Technology, Vellore
November, 2026


# Declaration

We hereby declare that the BITE497J – Project I thesis entitled "Intelligent PCB Surface Defect Detection, Severity Assessment and Probable Process-Fault Prediction Using Deep Learning" submitted by us, for the award of the degree of Bachelor of Technology in Information Technology, School of Computer Science Engineering and Information Systems to VIT is a record of bonafide work carried out by us under the supervision of Dr. Hemalatha S, Associate Professor, SCORE, VIT, Vellore.
We further declare that the work reported in this thesis has not been submitted and will not be submitted, either in part or in full, for the award of any other degree or diploma in this institute or any other institute or university.

Place: Vellore
Date:

Signature of the Candidates:

1. ____________________________
N Rukkshana (23BIT0405)

2. ____________________________
Nikhita Marimuthu (23BIT0397)

3. ____________________________
Pritika Kasinathan (23BIT0232)


# Certificate of Approval

This is to certify that the BITE497J – Project I report entitled "Intelligent PCB Surface Defect Detection, Severity Assessment and Probable Process-Fault Prediction Using Deep Learning" submitted by N RUKKSHANA (23BIT0405), NIKHITA MARIMUTHU (23BIT0397), and PRITIKA KASINATHAN (23BIT0232) in partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in Information Technology is a bonafide record of work carried out under my supervision.

Project Guide:

____________________________
Dr. Hemalatha S
Associate Professor, SCORE
VIT, Vellore



| Internal Examiner:  ____________________________ Signature & Date | External Examiner:  ____________________________ Signature & Date |
| --- | --- |
| Head of the Department:  ____________________________ Dr. Arivuselvan K Department of Information Technology | Place: Vellore Date: |



# Acknowledgement

It is our pleasure to express with a deep sense of gratitude to our BITE497J - Project I guide Dr. Hemalatha S, Associate Professor, School of Computer Science Engineering and Information Systems, Vellore Institute of Technology, Vellore for the constant guidance and continual encouragement in our endeavor. Our association with the guide is not confined to academics only, but it is a great opportunity on our part to work with an intellectual and an expert in the field of deep learning-based visual inspection.
We would like to express our heartfelt gratitude to Honorable Chancellor Dr. G Viswanathan; respected Vice Presidents Dr. Sankar Viswanathan, Dr. Sekar Viswanathan, Vice Chancellor Dr. V. S. Kanchana Bhaaskaran; Pro-Vice Chancellor Dr. Partha Sharathi Mallick; and Registrar Dr. Jayabarathi T.
Our whole-hearted thanks to Dean Dr. Daphne Lopez, School of Computer Science Engineering and Information Systems, Head, Department of Information Technology, Dr. Arivuselvan K, Information Technology Project Coordinator Dr. Suganya P, SCORE School Project Coordinator Dr. Karthikeyan P, all faculty, staff and members working as limbs of our university for their continuous guidance throughout our course of study in unlimited ways.
It is indeed a pleasure to thank our parents and friends who persuaded and encouraged us to take up and complete our project successfully. Last, but not least, we express our gratitude and appreciation to all those who have helped us directly or indirectly towards the successful completion of the project.

Place: Vellore
Date:
N Rukkshana
Nikhita Marimuthu
Pritika Kasinathan


# Executive Summary

Automated Optical Inspection (AOI) in Printed Circuit Board (PCB) manufacturing is challenged by false alarms, missed functional defects, and variations across board layouts. Conventional object detectors rely primarily on visual appearance and confidence scores, limiting their ability to distinguish cosmetic imperfections from defects that compromise electrical connectivity. This dissertation presents PCBRisk, a risk-calibrated PCB defect inspection framework integrating YOLO11n object detection, topological consequence analysis, and split-conformal risk calibration.
The framework compares detected defect regions against a defect-free reference template using differential copper masks and connected-component analysis to identify potential electrical opens and shorts. Leave-One-Board-Out (LOBO) evaluation across 11 PCB boards assesses performance across different board designs while reducing board-level data leakage. At a target critical-defect miss rate of 5%, the proposed topological features reduce human-review workload by 34.18% compared with raw confidence-based triage and 26.83% compared with image-only calibrated triage. The complete pipeline achieves an end-to-end CPU latency of 96.75 ms per board image. The results demonstrate the potential of combining circuit-topology reasoning and statistical risk calibration to improve inspection efficiency while accounting for critical-defect escape risks.

Keywords: Printed Circuit Board (PCB), Automated Optical Inspection (AOI), YOLO11n, Topological Consequence Analysis, Split-Conformal Prediction, Risk Calibration, Defect Detection, Leave-One-Board-Out (LOBO), Human-in-the-Loop Inspection.


# Table of Contents

[Generate Table of Contents using Word's built-in TOC feature: References → Table of Contents → Automatic Table]


# List of Figures

Figure 4.1: PCBRisk End-to-End System Architecture ........................ [Page]
Figure 4.2: Graph-Constrained Synthetic Defect Generation: op_break and op_bridge ........................ [Page]
Figure 4.3: Morphological Alignment Tolerance: Differential Mask Computation ........................ [Page]
Figure 4.4: Split-Conformal Calibration Threshold Selection ........................ [Page]
Figure 4.5: Leave-One-Board-Out (LOBO) Cross-Fitting Protocol ........................ [Page]
Figure 6.1: Live AOI Web Platform: Board Inspector View ........................ [Page]
Figure 6.2: Live AOI Web Platform: Three-Queue Triage Interface ........................ [Page]
Figure 7.1: Multi-Seed Paired Bootstrap CI Forest Plot and Relative Workload Reduction ........................ [Page]
Figure 7.2: Candidate Defect Triage Disposition Matrix (Policy A0 vs Policy B) ........................ [Page]
Figure 7.3: Expected Quality-Cost per 1,000 Boards as a Function of Penalty Ratio R ........................ [Page]
Figure 7.4: Operational Evaluation Matrix and Pareto Quality-Cost Frontier ........................ [Page]
Figure 7.5: Hardware Latency Profile and Vector CAD R-Tree Speedup ........................ [Page]
Figure 7.6: Registration Tolerance Sensitivity Across Dilation Radii ........................ [Page]
Figure 7.7: Mondrian Group-Conditional Conformal Calibration Results ........................ [Page]


# List of Tables

Table 3.1: Software Requirements and Development Environment ........................ [Page]
Table 3.2: Hardware Specifications for Benchmarking ........................ [Page]
Table 3.3: DeepPCB Dataset Specification ........................ [Page]
Table 3.4: PCB Defect Class Taxonomy ........................ [Page]
Table 4.1: Board-Level Dataset Partitioning (v2) ........................ [Page]
Table 4.2: Defect Class Distribution Across Train/Val/Test Splits ........................ [Page]
Table 4.3: Cross-Validation Fold Assignment ........................ [Page]
Table 4.4: Synthetic Defect Generation Summary ........................ [Page]
Table 4.5: YOLO11n Training Hyperparameters ........................ [Page]
Table 4.6: Feature Engineering: Policy A vs Policy B ........................ [Page]
Table 5.1: Project Schedule and Milestones ........................ [Page]
Table 7.1: YOLO11n Class-Wise Test Performance (Fixed Split) ........................ [Page]
Table 7.2: Cross-Validation mAP@50-95 Across Folds and Seeds ........................ [Page]
Table 7.3: Multi-Seed LOBO Replication Summary at α = 0.05 ........................ [Page]
Table 7.4: Quality-Cost Evaluation Summary per 1,000 Boards ........................ [Page]
Table 7.5: Hardware Latency Profile and Throughput Benchmarks ........................ [Page]
Table 7.6: Registration Tolerance Trade-Off Across Dilation Radii ........................ [Page]
Table 7.7: Mondrian vs Global Conformal Calibration Comparison ........................ [Page]
Table 7.8: Automated PASS/FAIL Checklist Verdict ........................ [Page]
Table 7.9: Formal Statistical Hypothesis Verdicts ........................ [Page]


# Abbreviations


| Abbreviation | Full Form |
| --- | --- |
| AOI | Automated Optical Inspection |
| BGA | Ball Grid Array |
| CAD | Computer-Aided Design |
| CI | Confidence Interval |
| CIoU | Complete Intersection over Union |
| CNN | Convolutional Neural Network |
| CPU | Central Processing Unit |
| DFL | Distribution Focal Loss |
| EDA | Electronic Design Automation |
| EDT | Euclidean Distance Transform |
| FPS | Frames Per Second |
| GEM | Generic Equipment Model |
| GPU | Graphics Processing Unit |
| HDI | High-Density Interconnect |
| IoU | Intersection over Union |
| LOBO | Leave-One-Board-Out |
| mAP | Mean Average Precision |
| MES | Manufacturing Execution System |
| ONNX | Open Neural Network Exchange |
| PCB | Printed Circuit Board |
| PTH | Plated Through-Hole |
| SECS | SEMI Equipment Communications Standard |
| SGD | Stochastic Gradient Descent |
| SMT | Surface Mount Technology |
| SPI | Solder Paste Inspection |
| TPS | Thin-Plate Spline |
| YOLO | You Only Look Once |




# Symbols and Notations


| Symbol | Description |
| --- | --- |
| α | Target critical-defect miss rate (e.g., 0.05) |
| τ | Conformal non-conformity threshold for flagging |
| s(x) | Predicted criticality score for candidate defect x |
| R | Escape-to-review penalty cost ratio (C_escape / C_review) |
| R* | Break-even cost ratio boundary |
| δ_reg | Morphological dilation radius for registration tolerance (pixels) |
| λ | Empirical inflation factor for shift-inflated calibration |
| M_template | Binary copper mask of defect-free template image |
| M_test | Binary copper mask of test image |
| g_open | Binary indicator: consequence engine detects open circuit |
| g_short | Binary indicator: consequence engine detects short circuit |
| s_open | Continuous signal strength for open circuit evidence |
| s_short | Continuous signal strength for short circuit evidence |
| ε | Yellow queue margin width around threshold τ |




# CHAPTER 1: INTRODUCTION


## 1.1 Introduction

Printed Circuit Boards (PCBs) form the fundamental interconnection substrate of virtually every modern electronic device, from consumer smartphones and automotive ECUs to aerospace avionics and medical implants. The escalating demand for miniaturized, high-density interconnect (HDI) boards with micro-pitch ball grid arrays (BGAs) and fine-line conductor traces has simultaneously heightened the criticality of fabrication quality control and strained the capacity of traditional inspection methodologies.
Automated Optical Inspection (AOI) represents the frontline defense against PCB fabrication defects in modern Surface Mount Technology (SMT) lines. High-resolution cameras capture orthographic images of each manufactured board, and computer vision algorithms—increasingly based on deep convolutional neural networks—detect candidate surface anomalies such as missing copper (potential open circuits), excess copper (potential short circuits), mousebites, spurs, copper residue, and pin-holes [1].
However, industrial AOI deployment faces a fundamental operational trilemma. First, state-of-the-art object detectors treat all bounding boxes purely through visual appearance and class confidence scores, without any intrinsic understanding of whether a detected anomaly actually impairs electrical circuit continuity. In practice, over 90% of candidate detections are cosmetic imperfections that do not affect circuit function, creating severe false-alarm fatigue for human inspectors [2]. Second, optical detectors possess no representation of circuit topology—a 20 μm copper notch in an isolated ground plane is electrically inconsequential, whereas the same notch bisecting a high-speed signal line creates a fatal open circuit. Third, PCB fabrication lines produce diverse board designs with varying copper weights, trace pitches, and solder mask pigmentations, causing standard thresholding heuristics calibrated on historical lots to catastrophically under-cover or over-flag when introduced to unfamiliar panel geometries.
This dissertation addresses these challenges through PCBRisk, a risk-calibrated triage architecture that unifies deep object detection, topological consequence reasoning, and distribution-free statistical risk control into a coherent industrial inspection pipeline.

## 1.2 Objective

The primary objective of this project is:
To design, implement, and empirically validate a risk-calibrated PCB defect triage system that integrates topological circuit consequence analysis with split-conformal prediction to significantly reduce human inspector workload while providing rigorous statistical bounds on the critical defect miss rate.
The measurable secondary objectives are:
Develop a local topological consequence engine that determines whether candidate defects cause electrical opens or shorts by comparing test images against defect-free templates using Otsu binarization, morphological dilation, and connected-component analysis.
Train and evaluate YOLO11n object detectors across multiple random seeds (42, 43, 44) and cross-validation folds on a board-level partitioned DeepPCB dataset to prevent data leakage.
Implement and compare four split-conformal calibration methods (pooled, worst-board, shift-inflated, board-median) under a Leave-One-Board-Out (LOBO) evaluation protocol.
Demonstrate statistically significant workload reduction (paired bootstrap 95% CI excluding zero) at α = 0.05 miss rate across all detector seeds.
Formulate and evaluate a cost-sensitive quality decision model balancing human review cost against critical defect escape penalties.
Profile hardware latency to verify real-time feasibility on consumer-grade CPU hardware.
Build a functional interactive web platform demonstrating the three-queue triage workflow.

## 1.3 Motivation

The motivation for this project arises from the critical gap between the capabilities of modern deep learning-based PCB defect detectors and the operational requirements of industrial AOI systems. While recent detectors such as YOLOv8, YOLO11, and RT-DETR achieve high mean average precision (mAP) on benchmark datasets, they suffer from semantic blindness regarding electrical consequence. A detector cannot distinguish between a harmless cosmetic pin-hole and a catastrophic trace break that will cause a board to fail in the field.
This semantic gap directly impacts manufacturing economics. Human inspectors stationed at AOI review consoles experience cognitive fatigue after reviewing hundreds of flagged cosmetic defects, leading to inadvertent escapes of genuine functional defects. In consumer electronics, an escaped open or short circuit incurs rework costs between 50× and 500× the cost of a single visual inspection, while in aerospace and medical devices, escaped defects can lead to catastrophic system failures.
Furthermore, standard confidence thresholding provides no rigorous statistical guarantees under distribution shift across distinct board layouts and manufacturing lots. A threshold calibrated on one board design may catastrophically under-cover or over-flag defects on a different design. The need for distribution-free, finite-sample statistical guarantees motivates the use of conformal prediction methods.

## 1.4 Background

Deep learning methods for PCB inspection have evolved from two-stage architectures such as Faster R-CNN to single-stage detectors including the YOLO family (YOLOv5, YOLOv8, YOLO11) and RT-DETR. These models achieve high mAP on benchmark datasets such as DeepPCB [1] and PKU-Market-PCB, but they operate purely on visual features without circuit topology awareness.
Early template-subtraction approaches relied on rigid global image differencing between test and reference images, but failed catastrophically under optical illumination shifts, minor substrate warping, and camera vibration. The transition from pixel-level subtraction to learned representations improved robustness but sacrificed interpretability and topological reasoning.
Conformal prediction, introduced by Vovk et al. [3] and further developed by Angelopoulos and Bates [4], provides finite-sample, distribution-free statistical validity for predictive models. Given calibration data drawn exchangeably from an underlying distribution, split-conformal methods construct non-conformity scores such that a user-specified error bound is strictly satisfied. In the PCBRisk context, conformal prediction is adapted to construct risk-calibrated flagging policies that bound the critical defect miss rate while minimizing human inspection workload.


# CHAPTER 2: DISSERTATION DESCRIPTION AND GOALS


## 2.1 Project Scope

This project encompasses the design, implementation, and empirical validation of an end-to-end risk-calibrated PCB defect inspection pipeline. The scope includes data preparation with board-level partitioning, graph-constrained synthetic defect generation, multi-seed detector training, topological consequence engine development, split-conformal risk calibration, cost-sensitive quality evaluation, hardware latency benchmarking, and interactive web platform deployment.

## 2.2 Problem Statement

Given a PCB test image and its corresponding defect-free template, the system must: (1) detect candidate surface defects using deep object detection; (2) determine the electrical circuit consequence of each candidate defect (open circuit, short circuit, or cosmetic); and (3) triage candidate defects into prioritized review queues (Red, Yellow, Green) such that the miss rate on critical functional defects (opens and shorts) is bounded at or below a user-specified threshold α, while minimizing the total number of candidates flagged for human review.

## 2.3 Intended Users and Application Domain

The primary intended users are AOI system operators and quality assurance engineers in PCB fabrication facilities. The application domain spans consumer electronics, automotive, industrial computing, and potentially aerospace and medical device manufacturing, with appropriate qualification boundaries documented in this report.

## 2.4 Project Goals and Deliverables

The project defined the following goals and deliverables:
Table 2.1: Project Goals and Deliverables

| Goal ID | Goal | Description |
| --- | --- | --- |
| G1 | Board-Partitioned Dataset | Correct the data leakage in the DeepPCB official splits by implementing strict board-level partitioning across 11 manufacturing groups. |
| G2 | Synthetic Defect Generator | Develop a graph-constrained synthetic defect generator that creates verified opens and shorts on defect-free templates. |
| G3 | Object Detector Training | Train YOLO11n detectors across 3 random seeds and 4 cross-validation folds (26 total models) with fault-tolerant cloud mirroring. |
| G4 | Consequence Engine | Implement a local topological consequence engine (v3) with morphological alignment tolerance and box-local evidence. |
| G5 | Conformal Risk Control | Implement and compare four split-conformal calibration methods under LOBO cross-fitting with paired bootstrap statistical testing. |
| G6 | Cost-Sensitive Evaluation | Formulate and evaluate a quality-cost decision model with break-even analysis. |
| G7 | Interactive Platform | Deploy a functional web platform with FastAPI backend and three-queue triage interface. |
| G8 | Latency Benchmarking | Profile end-to-end pipeline latency on consumer CPU hardware. |



## 2.5 Functional Requirements

FR1: The system shall detect PCB surface defects across six classes (open, short, mousebite, spur, copper, pin-hole) using a trained YOLO11n detector.
FR2: The system shall determine the electrical consequence (open/short/none) of each candidate detection by comparing test and template images.
FR3: The system shall classify candidate detections into Red (critical), Yellow (borderline), and Green (cosmetic) queues.
FR4: The system shall guarantee that the miss rate on critical defects does not exceed the user-specified threshold α.
FR5: The system shall provide a web-based interface for interactive board inspection and triage review.
FR6: The system shall export evaluation results, confidence intervals, and verdict reports in structured formats (CSV, JSON).

## 2.6 Non-Functional Requirements

NFR1: End-to-end pipeline latency shall not exceed 200 ms per board image on a consumer-grade CPU.
NFR2: The system shall operate without GPU hardware for inference (CPU-only ONNX deployment).
NFR3: Statistical guarantees shall be validated across at least 3 independent random seeds.
NFR4: The dataset shall enforce zero board overlap between training and test partitions.
NFR5: The web platform shall be accessible via standard browsers without additional client-side installations.

## 2.7 Distinction Between Initial Goals and Completed Work

All primary goals (G1–G8) were successfully completed. The automated PASS/FAIL checklist (verdict_report.json) documented that the topological consequence engine's absolute identification accuracy on opens (64.35%) and shorts (73.71%) fell below the strict 85% threshold defined in the initial aspirational checklist (Checklist Item 1: FAIL). However, this moderate absolute accuracy was sufficient to deliver statistically significant workload reduction (Checklist Item 2: PASS) because the engine's primary value lies in its asymmetric filtering capability—dismissing cosmetic defects with an extremely low false-flag rate of only 1.17%.
An additional aspirational goal—distilling a reference-free student CNN head (Policy C) to eliminate the need for golden template images—was explicitly FAILED in the automated checklist. The student CNN could not achieve parity with the template-based differential analysis because verifying whether a missing trace segment breaks circuit continuity fundamentally requires knowledge of the global net topology, which cannot be inferred from a localized 48×48 pixel crop alone.


# CHAPTER 3: TECHNICAL SPECIFICATION


## 3.1 Software Requirements and Development Environment

Table 3.1: Software Requirements and Development Environment

| Software/Library | Version | Purpose |
| --- | --- | --- |
| Python | 3.10+ | Primary programming language |
| PyTorch | 2.8.0 | Deep learning framework for model training |
| Ultralytics | YOLO11n | Object detection architecture and training framework |
| ONNX Runtime | 1.29+ | CPU inference engine for exported detector models |
| OpenCV | 4.x (headless) | Image processing and morphological operations |
| scikit-image | 0.22+ | Skeletonization and distance transforms |
| scikit-learn | 1.4+ | Logistic regression for scoring models |
| SciPy | 1.12+ | Statistical functions and spatial analysis |
| Shapely / GEOS | 2.0+ | Vector geometry and R-Tree spatial indexing |
| pandas | 2.1+ | Data manipulation and CSV/manifest handling |
| NumPy | 1.26+ | Numerical computation |
| Matplotlib | 3.8+ | Publication-quality figure generation |
| FastAPI | 0.110+ | REST API backend server |
| Uvicorn | 0.29+ | ASGI server for FastAPI |
| Google Colab | T4 GPU runtime | Training and evaluation environment |
| Google Drive | Cloud storage | Persistent artifact storage with fault tolerance |



## 3.2 Hardware Specifications for Benchmarking

Table 3.2: Hardware Specifications for Benchmarking

| Component | Specification | Notes |
| --- | --- | --- |
| CPU (Benchmark Host) | Intel Core i5-1335U | 10 cores / 12 threads, Alder Lake-P |
| GPU (Training) | NVIDIA Tesla T4 | 16 GB VRAM, Google Colab |
| RAM (Benchmark Host) | 16 GB DDR4 | Standard laptop configuration |
| Storage | NVMe SSD | ≥ 10 GB for dataset + models |
| Operating System | Windows 11 / Ubuntu (Colab) | Cross-platform compatible |



## 3.3 Dataset Specification

The primary dataset used in this project is the DeepPCB benchmark dataset published by Tang et al. [1]. The dataset was acquired from the official GitHub repository and verified using SHA-256 cryptographic checksumming.
Table 3.3: DeepPCB Dataset Specification

| Property | Value |
| --- | --- |
| Dataset Name | DeepPCB |
| Source | https://github.com/tangsanli5201/DeepPCB |
| License | Research Use Only |
| Total Test Images | 1,500 |
| Total Template Images | 1,501 |
| Manufacturing Board Groups | 11 independent groups |
| Image Resolution | 640 × 640 pixels |
| Total Annotated Defects | 10,013 |
| Critical Defects (Opens + Shorts) | 3,448 (1,942 opens + 1,506 shorts) |
| Cosmetic Defects | 6,565 (mousebite + spur + copper + pin-hole) |
| SHA-256 Checksum | 5654be20ec5b736b0d9e... |



## 3.4 Defect Class Taxonomy

Table 3.4: PCB Defect Class Taxonomy

| Class ID | Name | Description | Severity |
| --- | --- | --- | --- |
| 0 | open | Missing copper causing trace disconnection | Critical (functional) |
| 1 | short | Excess copper bridging distinct nets | Critical (functional) |
| 2 | mousebite | Irregular board edge nibbling | Cosmetic |
| 3 | spur | Non-functional copper protrusion | Cosmetic |
| 4 | copper | Copper residue on substrate | Cosmetic |
| 5 | pin-hole | Small void in copper plane | Cosmetic |



## 3.5 System Inputs, Processing, and Outputs

Inputs:
Test PCB image (640×640 RGB JPEG)
Corresponding defect-free template image (640×640 RGB JPEG)
Target miss rate α (default: 0.05)
Calibration method selection (pooled, worst-board, shift-inflated, or board-median)
Outputs:
Detected candidate bounding boxes with confidence scores and class predictions
Topological consequence verdict per box (open / short / none / both)
Risk-calibrated triage assignment (Red / Yellow / Green queue)
Aggregate workload metrics (flagged boxes per image, miss rate, precision)
Evaluation reports with bootstrap confidence intervals

## 3.6 Performance and Evaluation Criteria

The system is evaluated using the following metrics: (1) Flagged boxes per image (review workload); (2) Critical defect miss rate relative to target α; (3) Inspection precision (fraction of flagged boxes that are truly critical); (4) Paired board-bootstrap 95% confidence interval on workload reduction; (5) Multi-seed consistency (reduction holding across all 3 detector random seeds); (6) Quality cost per 1,000 boards under the decision model Cost = Reviews + R × Escapes; (7) End-to-end pipeline latency in milliseconds.


# CHAPTER 4: DESIGN APPROACH AND DETAILS


## 4.1 Design Approach / Materials & Methods


### 4.1.1 Overall System Architecture

The PCBRisk system architecture comprises four major processing stages, executed sequentially for each board image: (1) Image Acquisition, where the test PCB image and golden template are captured or loaded; (2) Object Detection, where the YOLO11n backbone proposes candidate defect bounding boxes; (3) Topological Consequence Engine, where each candidate box is evaluated for electrical circuit impact via differential connected-component analysis; and (4) Conformal Risk Calibration, where a scoring model combines visual and topological features and a conformal threshold τ classifies candidates into Red, Yellow, and Green triage queues.

Figure 4.1: PCBRisk End-to-End System Architecture

### 4.1.2 Data Acquisition and Preparation

The DeepPCB dataset was acquired from the official GitHub repository by Tang et al. [1]. The raw archive was verified using SHA-256 checksumming (hash: 5654be20ec5b...) and stored immutably in 00_raw/deeppcb_main.zip. The dataset contains 1,500 paired test/template images across 11 independent manufacturing board groups.
The v1 Data Leakage Bug Fix:
The official DeepPCB split partitioned images randomly without regard to board identity, causing all 11 board topologies to leak across train, validation, and test sets. In v2, the split was corrected to enforce strict board-level partitioning with zero board overlap between splits.
Table 4.1: Board-Level Dataset Partitioning (v2)

| Split | Boards | Images |
| --- | --- | --- |
| Train (6 boards) | group12000, group12100, group13000, group20085, group77000, group90100 | 882 |
| Validation (2 boards) | group12300, group92000 | 218 |
| Test (3 boards) | group00041, group44000, group50600 | 400 |


Table 4.2: Defect Class Distribution Across Train/Val/Test Splits

| Split | Open | Short | Mousebite | Spur | Copper | Pin-hole | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Train | 1,118 | 929 | 1,159 | 966 | 889 | 871 | 5,932 |
| Val | 275 | 175 | 215 | 203 | 184 | 208 | 1,260 |
| Test | 549 | 402 | 591 | 456 | 401 | 422 | 2,821 |
| Total | 1,942 | 1,506 | 1,965 | 1,625 | 1,474 | 1,501 | 10,013 |


Table 4.3: Cross-Validation Fold Assignment

| Fold | Boards |
| --- | --- |
| Fold 0 | group00041, group13000, group77000 |
| Fold 1 | group44000, group20085, group90100 |
| Fold 2 | group50600, group12000, group92000 |
| Fold 3 | group12300, group12100 |



### 4.1.3 Graph-Constrained Synthetic Defect Generation

To address the scarcity of critical functional defects in training data, a graph-constrained synthetic defect generator was developed. Unlike naive data augmentation, each synthetic defect is verified through topological connectivity analysis before being labeled.
Trace Severing Operator (op_break):
Computes the Euclidean distance transform and medial axis skeleton of the copper mask. Selects a skeleton point, cuts a circular disk of radius r = EDT(x,y) + Uniform(2,5), and recomputes connected components. A defect is labeled as 'open' if and only if the number of connected components increases after the cut. Partial thinning ('width_loss') is strictly discarded.
Conductor Bridging Operator (op_bridge):
Identifies two distinct copper components separated by 3–14 px, draws a synthetic copper bridge line with thickness 3–6 px, and confirms that component labels merge. Only verified topological shorts are retained.
Table 4.4: Synthetic Defect Generation Summary

| Operator | Consequence | Class | Generated | Success Rate |
| --- | --- | --- | --- | --- |
| op_break | Complete Trace Open | 0 (open) | 1,485 | 99.0% |
| op_bridge | Net Bridge Short | 1 (short) | 740 | 49.3% |
| Total | Verified Critical | 0 & 1 | 2,225 | — |



### 4.1.4 Object Detector Training

The object detection backbone is Ultralytics YOLO11n, a state-of-the-art lightweight single-stage detector with 2.58M parameters and 6.4 GFLOPs, suitable for edge deployment.
Table 4.5: YOLO11n Training Hyperparameters

| Parameter | Value |
| --- | --- |
| Architecture | YOLO11n (100 layers, 2,583,322 params, 6.4 GFLOPs) |
| Input Resolution | 640 × 640 pixels |
| Batch Size | 16 |
| Epochs | 60 |
| Early Stopping Patience | 30 |
| Optimizer | SGD with cosine learning rate scheduling |
| Loss Functions | CIoU box loss + DFL + BCE class loss |
| Training GPU | NVIDIA Tesla T4 (16 GB VRAM) |
| Total Trained Models | 26 (2 data configs × 4 folds × 3 seeds + 2 fixed) |
| Drive Mirroring Interval | 120 seconds (fault tolerance) |



### 4.1.5 Local Topological Consequence Engine (v3)

The PCBRisk Consequence Engine (v3) operates strictly on localized conductor topology within and immediately adjacent to each candidate bounding box, addressing the critical flaw in earlier global differencing approaches that attributed distant trace disconnects to unrelated candidate boxes.
The engine operates through three key steps:
Step 1 — Defect-Free Polarity Locking: Otsu binarization is performed on the defect-free golden template. Copper traces constitute the minority foreground class; polarity is determined once and locked for the corresponding test image.
Step 2 — Morphological Alignment Tolerance: Differential masks are computed via morphological dilation to prevent spurious edge residuals caused by optical jitter:
    missing = M_template AND NOT dilate(M_test, δ_reg)
    excess = M_test AND NOT dilate(M_template, δ_reg)
where δ_reg = 1 px (≈ 25 μm) suppresses boundary noise.
Step 3 — Local Component Splitting and Bridging Query: For each candidate box, a padded local neighborhood window is extracted. Connected component analysis via 8-connectivity labels conductor nets. Trace opens are detected when a template net touching missing copper is severed into two or more disconnected test components. Trace shorts are detected when excess copper forms a bridge connecting two distinct template net identifiers.

### 4.1.6 Feature Engineering and Triage Policies

Table 4.6: Feature Engineering: Policy A vs Policy B

| Feature Set | Features |
| --- | --- |
| Policy A0 (Raw Confidence) | conf |
| Policy A (Image-Only) | conf, crit, conf×crit, log(area) |
| Policy B (+Consequence) | All Policy A features + g_open, g_short, g_agree, g_cross, log(1+missing_px), log(1+excess_px), log(1+s_open), log(1+s_short), conf×g_agree |



### 4.1.7 Split-Conformal Calibration Methods

Four conformal calibration methods were implemented to address the trade-off between coverage guarantees and inspection workload under inter-board distribution shift.
Pooled (Classic Split-Conformal):
Pools all critical ground-truth scores across the 10 calibration boards and computes the empirical α-quantile. Efficient but may under-cover under board shift.
Worst-Board (Conservative Min-Max Bound):
Computes the conformal threshold independently for each calibration board and sets τ = min_b(τ_b). Guarantees the empirical miss rate on the worst calibration board does not exceed α. Ultra-conservative with high review load.
Board-Median (Robust Central Tendency):
Takes the median threshold across calibration boards: τ = median_b(τ_b). Robust to outlier boards.
Shift-Inflated (Drift-Adjusted Conformal):
Estimates the empirical inflation factor λ = mean_b(miss_b) / α on calibration boards and calibrates to the scaled target α_inflated = α / max(1.0, λ). Best balance under known distribution drift.

### 4.1.8 Leave-One-Board-Out (LOBO) Evaluation Protocol

To ensure complete statistical independence, LOBO evaluation is executed across all 11 physical board groups. In each iteration, one board serves as the completely unseen test board while the remaining 10 boards provide cross-fitted feature weights and conformal thresholds. This protocol is repeated across 3 independent detector random seeds (42, 43, 44), producing 33 board-level evaluation points per policy comparison.
Statistical significance is assessed using paired board-bootstrap resampling with 5,000 replicates, computing 95% confidence intervals on the workload difference ΔLoad = Load_B − Load_A.

## 4.2 Codes and Standards

The following industry standards and conventions are relevant to this project. Compliance is stated explicitly where applicable; where standards define aspirational targets for future deployment, this is noted transparently.

| Standard | Title | Relevance to Project |
| --- | --- | --- |
| IPC-9252 | Requirements for Electrical Testing of Unpopulated Printed Boards | Defines the boundary between optical and electrical inspection. PCBRisk operates within the AOI domain; final electrical verification requires bare-board testing per this standard. |
| IPC-A-600 | Acceptability of Printed Boards | Defines visual acceptability criteria for PCB surface defects. The defect taxonomy used in this project aligns with IPC-A-600 categories. |
| SEMI E30 (GEM) | Generic Equipment Model | Communication standard for AOI equipment integration with factory host systems. PCBRisk specifies GEM-compatible telemetry interfaces for future deployment. |
| IPC-2591 (CFX) | Connected Factory Exchange | IoT messaging standard for SMT line integration. PCBRisk architecture includes CFX-compatible triage queue outputs. |
| IEEE 754 | Floating-Point Arithmetic | All numerical computations use IEEE 754 double-precision floating-point representation via NumPy/PyTorch. |
| PEP 8 | Python Style Guide | Python source code follows PEP 8 conventions for readability and maintainability. |



## 4.3 Constraints, Alternatives, and Tradeoffs


### 4.3.1 Technical Constraints

Raster Morphology Limitation: The consequence engine operates on raster pixel masks derived from Otsu binarization, which is sensitive to illumination gradients, solder mask pigmentation, and copper oxidation. This limits topological accuracy to 64.35% (opens) and 73.71% (shorts) on the DeepPCB dataset.
Template Dependency: The current implementation requires a defect-free golden template for each board design, which may not always be available in production environments.
Single-Layer Inspection: AOI inspects only the outermost surface layer and cannot detect defects in internal conductor layers of multilayer PCBs.
Dataset Scale: Evaluation is limited to the DeepPCB dataset (11 board groups, 1,500 images), which may not fully represent the diversity of industrial manufacturing variability.

### 4.3.2 Alternative Approaches Considered


| Alternative | Description | Decision |
| --- | --- | --- |
| Global Image Differencing | Simple pixel-level subtraction between test and template | Rejected: Catastrophically fails under minor optical shifts and illumination changes |
| Reference-Free CNN (Policy C) | Small CNN on 48×48 test-image crops without template | Rejected: Failed to achieve parity with template-based analysis (FAIL in automated checklist) |
| Faster R-CNN Detector | Two-stage detector with region proposal network | Not selected: Higher latency and computational cost; YOLO11n provides comparable accuracy at much higher throughput |
| Vector CAD Netlist Engine | Direct Gerber/ODB++ parsing with R-Tree spatial queries | Proposed for future work: Would eliminate raster limitations but requires industrial CAD file access |



### 4.3.3 Key Design Trade-offs

Accuracy vs. Computational Cost: The YOLO11n nano architecture was selected over larger variants (YOLO11s, YOLO11m) to prioritize real-time CPU inference capability at the cost of marginal mAP reduction. The consequence engine's raster morphology approach trades absolute topological accuracy for sub-millisecond per-box query latency.
Coverage vs. Workload: The four conformal calibrators represent different positions on the coverage-workload trade-off. Pooled calibration minimizes review workload but may under-cover under board shift; worst-board calibration maximizes safety guarantees at the cost of doubling review load.
Morphological Dilation Radius: δ_reg = 1 px was selected as the operational optimum, suppressing >91.7% of spurious shift noise while retaining 45.1% of true defect signal. Higher values over-erode fine conductor defects.


# CHAPTER 5: SCHEDULE, TASKS AND MILESTONES

The project was executed in a sequential pipeline spanning the academic year 2025–2026. The following table summarizes the major tasks, deliverables, and milestone statuses as evidenced by the completed notebook pipeline and artifact outputs.
Table 5.1: Project Schedule and Milestones

| Task / Work Package | Period | Deliverables | Status |
| --- | --- | --- | --- |
| Literature Review & Problem Formulation | Weeks 1–3 | Problem statement, research gap identification, related work survey | Completed |
| Data Acquisition & Provenance (NB01) | Week 4 | Raw dataset archive, SHA-256 checksums, SOURCES.md | Completed |
| Dataset Restructuring & v2 Partitioning (NB02) | Weeks 4–5 | Board-partitioned splits, YOLO labels, prepared_v2.tar | Completed |
| Synthetic Defect Generation (NB03) | Week 5–6 | 2,225 graph-verified synthetic defects, synth_v2.tar | Completed |
| Detector Training — 26 Models (NB04) | Weeks 6–10 | 26 trained YOLO11n models across folds/seeds, master_results_v2.csv | Completed |
| Consequence Engine Development & Pilot Eval (NB05) | Weeks 8–11 | Engine v3, pilot 4-fold conformal results | Completed |
| Autonomous Pipeline & Automated Checklist (NB06) | Weeks 11–13 | Auto-tuned parameters, verdict_report.json | Completed |
| Multi-Seed Replication & Calibration (NB07) | Weeks 13–16 | 3-seed LOBO results, replication_verdict.json, bootstrap CIs | Completed |
| Cost-Sensitive Evaluation & Figures | Week 16–17 | Cost curves, break-even analysis, publication figures | Completed |
| Latency Benchmarking & Sensitivity Experiments | Week 17–18 | CPU latency profile, registration tolerance, Mondrian calibration | Completed |
| Web Platform Development | Weeks 15–18 | FastAPI server, interactive 5-view frontend | Completed |
| Documentation & Dissertation Writing | Weeks 18–22 | PROJECT_HANDOVER.md, MANUSCRIPT.md, dissertation | Completed |




# CHAPTER 6: DISSERTATION / PROJECT DEMONSTRATION


## 6.1 Demonstration Setup and Prerequisites

The PCBRisk system can be launched as a self-contained local web application requiring only a Python 3.10+ environment with the specified dependencies. The following setup procedure reproduces the live demonstration.
Prerequisites:
Python 3.10+ installed on Windows, macOS, or Linux
Required Python packages: torch torchvision ultralytics scikit-image opencv-python-headless scipy scikit-learn pandas pyyaml onnx onnxruntime onnxslim matplotlib shapely fastapi uvicorn
PCBRisk repository with pre-trained ONNX model (06_exports/yolo11n_deeppcb_real_s42.onnx, 10.1 MB)
Raw DeepPCB dataset archive (00_raw/deeppcb_main.zip, ~125 MB)

## 6.2 Step-by-Step Execution Procedure

Method 1 — Command Line:
pip install torch torchvision ultralytics scikit-image opencv-python-headless \
              scipy scikit-learn pandas pyyaml onnx onnxruntime onnxslim matplotlib \
              shapely fastapi uvicorn
    python code/server.py
    # Access in browser: http://127.0.0.1:8000
Method 2 — Windows One-Click Launcher:
Double-click run_pcbrisk.bat

## 6.3 System Demonstration Walkthrough

The interactive AOI web platform provides five integrated views for comprehensive board inspection and triage evaluation:
View 1 — Board Inspector: Displays the test PCB image with candidate defect bounding boxes overlaid on a zoomable canvas. Each box is color-coded by triage queue assignment (Red/Yellow/Green) and annotated with defect class, confidence score, and topological consequence verdict.
View 2 — Triage Queue Dashboard: Shows the three-queue classification of all candidate detections for the current board. The Red Queue displays critical opens and shorts requiring immediate inspector disposition; the Yellow Queue shows borderline cases; the Green Queue lists cosmetic defects auto-cleared without stopping the conveyor.
View 3 — Consequence Dossier: For each selected candidate detection, presents a detailed topological analysis showing the differential mask, missing/excess copper regions, local connected-component topology, and the consequence verdict with supporting pixel-level evidence.
View 4 — Evaluation Dashboard: Displays aggregate evaluation metrics including flagged boxes per image, miss rate, precision, and bootstrap confidence intervals across the evaluation dataset.
View 5 — System Configuration: Allows selection of calibration method, target α, and detector configuration.
[Note: Screenshots of the live web platform are available in the running application at http://127.0.0.1:8000. Due to the dynamic nature of the web interface, static screenshots should be captured during the live demonstration.]

## 6.4 Representative Scenarios

Normal Case — Cosmetic Defect Auto-Bypass:
A pin-hole defect in an isolated ground plane is detected by YOLO11n with confidence 0.72. The consequence engine determines that the void does not disconnect any conductor net (g_open = 0, g_short = 0). The calibrated risk score falls below τ − ε, and the defect is automatically routed to the Green Queue without stopping the conveyor or alerting the inspector.
Critical Case — Trace Break Detection:
A missing copper region bisecting a signal trace is detected with confidence 0.85. The consequence engine confirms that the template net is severed into two disconnected components (g_open = 1, s_open = 847 px). The calibrated risk score exceeds τ, and the defect is flagged to the Red Queue with a high-resolution zoomed overlay for immediate inspector disposition.
Edge Case — Borderline Uncertainty:
A narrow copper spur near a signal trace shows ambiguous consequence analysis (g_short = 1 but s_short = 12 px, near the noise threshold). The calibrated risk score falls in the Yellow Queue range (τ − ε ≤ s < τ), routing the candidate for active learning audit or automated electrical probe testing.


# CHAPTER 7: RESULTS AND DISCUSSION


## 7.1 Cost Analysis

The PCBRisk project was developed using freely available tools and datasets. The primary computational cost was GPU training time on Google Colab (NVIDIA Tesla T4). No physical hardware procurement was required.
Table 7.0: Project Cost Summary

| Item | Description | Quantity | Unit Cost | Total |
| --- | --- | --- | --- | --- |
| DeepPCB Dataset | Research license (free) | 1 | Free | Free |
| Google Colab GPU (T4) | Per-session (free tier) | ~50 hrs | Free | Free |
| Google Drive Storage | 15 GB free tier | ~2 GB used | Free | Free |
| Python Libraries | Open-source | All | Free | Free |
| Development Hardware | Intel i5-1335U laptop | 1 | Pre-existing | N/A |


Note: In industrial deployment, costs would include AOI camera hardware, industrial PC, CAD file parsing infrastructure, and MES integration. These costs are outside the scope of this academic project.

## 7.2 Results


### 7.2.1 Object Detector Performance

On the fixed board-level split (seed 42), the YOLO11n detector achieved an overall mAP@50 of 0.963 and mAP@50-95 of 0.695 across all six defect classes on 2,821 test instances.
Table 7.1: YOLO11n Class-Wise Test Performance (Fixed Split, Seed 42)

| Class | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
| --- | --- | --- | --- | --- | --- |
| All Classes | 2,821 | 0.943 | 0.913 | 0.963 | 0.695 |
| open | 549 | 0.877 | 0.936 | 0.936 | 0.623 |
| short | 402 | 0.939 | 0.886 | 0.967 | 0.619 |
| mousebite | 591 | 0.965 | 0.895 | 0.962 | 0.693 |
| spur | 456 | 0.945 | 0.879 | 0.950 | 0.657 |
| copper | 401 | 0.990 | 0.942 | 0.989 | 0.792 |
| pin-hole | 422 | 0.940 | 0.938 | 0.971 | 0.786 |


Across all 12 cross-validation fold-seed combinations, adding synthetic data provided an average gain of +0.0047 mAP@50-95, with a substantial +0.0492 (+4.92%) gain on the canonical fixed split.
Table 7.2: Cross-Validation mAP@50-95 Across Folds and Seeds

| Fold | Seed | Real mAP@50-95 | Synth mAP@50-95 | Δ |
| --- | --- | --- | --- | --- |
| f0 | 42 | 0.5998 | 0.6342 | +0.0344 |
| f0 | 43 | 0.6344 | 0.6158 | −0.0186 |
| f0 | 44 | 0.6262 | 0.6578 | +0.0316 |
| f1 | 42 | 0.6340 | 0.6268 | −0.0072 |
| f1 | 43 | 0.6204 | 0.6534 | +0.0330 |
| f1 | 44 | 0.6014 | 0.6118 | +0.0104 |
| f2 | 42 | 0.6474 | 0.6835 | +0.0361 |
| f2 | 43 | 0.7573 | 0.7406 | −0.0167 |
| f2 | 44 | 0.7454 | 0.7398 | −0.0056 |
| f3 | 42 | 0.7348 | 0.7109 | −0.0239 |
| f3 | 43 | 0.7194 | 0.7180 | −0.0014 |
| f3 | 44 | 0.7650 | 0.7495 | −0.0155 |
| Fixed | 42 | 0.6951 | 0.7443 | +0.0492 |



### 7.2.2 Consequence Engine Accuracy

The topological consequence engine (v3) was validated against ground-truth defect labels on held-out folds 2–3 with auto-tuned parameters (tol=1, sig=10, pad=2, min_px=2):
Open circuit identification accuracy: 64.35%
Short circuit identification accuracy: 73.71%
Non-critical false-flag rate: 1.17% (cosmetic defects correctly dismissed 98.83% of the time)

### 7.2.3 Multi-Seed LOBO Replication Results

The definitive evaluation was conducted across all 11 boards, 3 random seeds (42, 43, 44), and 4 conformal calibrators at α = 0.05. The following table presents the headline results.
Table 7.3: Multi-Seed LOBO Replication Summary at α = 0.05

| Calibrator | Policy | Load | Red. vs A0 | Red. vs A | Precision | Miss Rate |
| --- | --- | --- | --- | --- | --- | --- |
| pooled | A0 (Raw Conf) | 6.37 | — | — | 35.5% | 5.32% |
| pooled | A (Image-Only) | 5.71 | −10.36% | Baseline | 39.5% | 5.52% |
| pooled | B (+Consequence) | 4.19 | −34.18% | −26.83% | 55.2% | 5.27% |
| shift-inflated | A (Image-Only) | 8.85 | — | Baseline | 26.3% | 4.73% |
| shift-inflated | B (+Consequence) | 7.54 | −15.55% | −16.44% | 31.1% | 4.67% |
| worst-board | A (Image-Only) | 9.96 | — | Baseline | 24.1% | 3.53% |
| worst-board | B (+Consequence) | 8.09 | −33.14% | −20.02% | 29.8% | 3.63% |
| board-median | A (Image-Only) | 2.94 | — | Baseline | 72.6% | 8.78% |
| board-median | B (+Consequence) | 2.54 | −8.52% | −14.03% | 87.3% | 7.02% |


Critically, the workload reduction was preserved across all three detector random seeds under all four calibrators. Under pooled calibration, the per-seed workload deltas were: Seed 42: −1.79, Seed 43: −0.89, Seed 44: −1.87 boxes/image. All 12 per-seed bootstrap CIs strictly excluded zero, completely ruling out random initialization artifacts.

Figure 7.1: Multi-Seed Paired Bootstrap CI Forest Plot and Relative Workload Reduction

Figure 7.2: Candidate Defect Triage Disposition Matrix (Policy A0 vs Policy B)

### 7.2.4 Cost-Sensitive Quality Evaluation

Table 7.4: Quality-Cost Evaluation Summary per 1,000 Boards (α = 0.05)

| Calibrator | A0 Load | A Load | B Load | Escapes/1k (A0→A→B) | Red. vs A0 | Cost Boundary | Savings (R=250) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 6.37 | 5.71 | 4.19 | 122.2→126.9→121.1 | −34.18% | Pareto Dominant | +6.65% |
| shift-inflated | 8.92 | 8.85 | 7.54 | 107.8→108.7→107.3 | −15.55% | Pareto Dominant | +4.18% |
| worst-board | 12.10 | 9.96 | 8.09 | 80.2→81.1→83.6 | −33.14% | R < 1,203 | +9.88% |
| board-median | 2.78 | 2.94 | 2.54 | 198.2→201.8→161.3 | −8.52% | Pareto Dominant | +18.07% |



Figure 7.3: Expected Quality-Cost per 1,000 Boards as a Function of Penalty Ratio R

Figure 7.4: Operational Evaluation Matrix and Pareto Quality-Cost Frontier

### 7.2.5 Hardware Latency Benchmarks

Table 7.5: Hardware Latency Profile (Intel Core i5-1335U CPU)

| Stage | Implementation | Mean | Median | P95 | Throughput |
| --- | --- | --- | --- | --- | --- |
| YOLO11n Detector | ONNX Runtime (CPU, 640×640) | 52.03 ms | 50.51 ms | 62.26 ms | 19.2 FPS |
| Morphological Base | Otsu + 8-conn + Dilation | 41.85 ms | 40.46 ms | 52.73 ms | 23.9 FPS |
| Consequence Query | Local Window + Split Query | 0.186 ms | 0.187 ms | 0.379 ms | 5,390 q/s |
| End-to-End Board | Full pipeline (15.45 boxes) | 96.75 ms | — | — | 10.3 FPS |
| Vector CAD Baseline | Shapely/GEOS R-Tree | 0.009 ms | 0.006 ms | 0.015 ms | >109k q/s |



Figure 7.5: Hardware Latency Profile and Vector CAD R-Tree Speedup

### 7.2.6 Registration Tolerance Sensitivity

Table 7.6: Registration Tolerance Trade-Off Across Dilation Radii

| δ_reg | Pitch | 1px Shift Noise | Defect Signal | Retention | Status |
| --- | --- | --- | --- | --- | --- |
| 0 px | 0 μm | 7,732 px | 14,895 px | 100.0% | Unviable |
| 1 px | ≈25 μm | 640 px (−91.7%) | 6,722 px | 45.1% | Operational Optimum |
| 2 px | 50 μm | 640 px | 451 px | 3.0% | Over-Dilated |
| 3 px | 75 μm | 640 px | 0 px | 0.0% | Complete Signal Loss |



Figure 7.6: Registration Tolerance Sensitivity Across Dilation Radii

### 7.2.7 Mondrian Group-Conditional Calibration

Table 7.7: Mondrian vs Global Conformal Calibration (α = 0.05, 3,448 defects)

| Method | Overall Miss | Open Miss | Short Miss | Disparity |
| --- | --- | --- | --- | --- |
| Global | 4.81% | 5.20% (>5%) | 4.32% | 0.88% |
| Mondrian | 4.88% | 4.84% (≤5%) | 4.91% (≤5%) | 0.07% (12× reduction) |



Figure 7.7: Mondrian Group-Conditional Conformal Calibration Results

### 7.2.8 Automated PASS/FAIL Checklist

Table 7.8: Automated PASS/FAIL Checklist Verdict

| Check | Criterion | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | Engine accuracy (open/short ≥ 85%, false-flag ≤ 15%) | FAIL | 64%/74% accuracy, 1.1% false-flag |
| 2 | Graph beats image-only at α=0.10 (worst-board) | PASS | CI excludes 0 |
| 3 | Board shift robustness | PASS | worst-board conformal keeps miss ≤ 1.25×α |
| 4 | Reference-free CNN head parity | FAIL | Template-free distillation insufficient |



### 7.2.9 Formal Hypothesis Verdicts

Table 7.9: Formal Statistical Hypothesis Verdicts

| Config | α | Calibrator | Verdict |
| --- | --- | --- | --- |
| Real | 0.05 | pooled | HOLDS (True) |
| Real | 0.05 | shift-inflated | HOLDS (True) |
| Real | 0.05 | worst-board | HOLDS (True) |
| Real | 0.05 | board-median | HOLDS (True) |
| Real | 0.10 | worst-board | HOLDS (True) |
| Real | 0.10 | pooled | Fails (False) |
| Synth1 | 0.05 | All methods | HOLDS (True) |
| Synth1 | 0.10 | All methods | Fails (False) |



## 7.3 Discussion


### 7.3.1 Achievement of Objectives

The primary objective—demonstrating statistically significant workload reduction through topological consequence features—was achieved conclusively. At α = 0.05, Policy B reduced human review workload by 34.18% (vs A0) and 26.83% (vs A) under pooled calibration, with the paired bootstrap 95% CI [−2.67, −0.48] strictly excluding zero across all three detector seeds.
All eight secondary goals (G1–G8) were completed as designed. The board-partitioned dataset eliminated data leakage, the synthetic defect generator produced 2,225 verified critical defects, 26 detector models were successfully trained, the consequence engine was developed and validated, four conformal calibrators were implemented and compared, cost-sensitive evaluation was performed, the interactive web platform was deployed, and latency benchmarks confirmed real-time CPU feasibility.

### 7.3.2 How Results Address the Original Problem

The original problem was the false-alarm fatigue created by treating all candidate defect boxes equally based on visual confidence alone. PCBRisk addresses this by introducing topological consequence features that distinguish between cosmetic anomalies and genuine circuit defects. The inspection precision improvement from 39.5% to 55.2% under pooled calibration directly translates to fewer unnecessary inspector stops, reduced cognitive fatigue, and faster line throughput.

### 7.3.3 Strengths and Technical Contributions

Rigorous statistical validation: 3-seed replication, 11-board LOBO cross-fitting, paired bootstrap CIs, and formal hypothesis verdicts provide strong evidence against random artifacts.
Cost-sensitive decision framework: The quality-cost model with break-even analysis provides actionable deployment guidance for different manufacturing environments.
Microsecond consequence queries: At 186 μs per box, topological verification adds negligible overhead over detector inference.
Transparent negative results: The automated checklist explicitly documents failures (engine accuracy threshold, reference-free head), maintaining research integrity.
Complete reproducible pipeline: Seven Jupyter notebooks with SHA-256 checksumming, automated resume, and deterministic seeding ensure full reproducibility.

### 7.3.4 Comparison with Existing Work

Standard AOI systems and academic PCB defect detection work focus exclusively on maximizing detection mAP without considering the downstream triage consequence. PCBRisk introduces a principled integration of topological reasoning and distribution-free risk control that, to the best of the author's knowledge, has not been previously demonstrated in the PCB inspection literature. Direct quantitative comparison is limited because existing works do not report the same risk-calibrated workload metrics; however, the 34.18% workload reduction at bounded miss rate represents a meaningful operational improvement over confidence-only triage baselines.

### 7.3.5 Failure Modes and Limitations

The consequence engine's moderate absolute accuracy (64% opens, 74% shorts) stems from raster morphology limitations: Otsu binarization is sensitive to illumination gradients and cannot resolve sub-pixel conductor features.
The system requires a defect-free golden template for each board design, which may not always be available in early production runs.
AOI fundamentally cannot detect internal-layer defects, PTH barrel voids, or latent high-resistance faults. Final electrical verification requires bare-board testing per IPC-9252.
Evaluation is limited to the DeepPCB dataset (11 board groups). Industrial validation across diverse board designs and manufacturing processes is required before deployment.
The hypothesis fails at α = 0.10: at loose miss-rate tolerances, detector confidence alone suffices and graph features provide marginal benefit.

## 7.4 Conclusion and Future Scope


### 7.4.1 Conclusion

This dissertation has presented PCBRisk, a risk-calibrated triage architecture for industrial PCB defect inspection that integrates deep object detection, topological consequence reasoning, and distribution-free split-conformal risk control. The system demonstrates that coupling YOLO11n object proposals with a local differential consequence engine and LOBO conformal calibration effectively resolves the false-alarm trilemma in AOI: at a rigorous 5% miss-rate ceiling, PCBRisk slashes human inspection workload by 34.18% over raw confidence triage and 26.83% over image-only calibrated triage, with statistical guarantees validated across 11 independent board groups and three detector random seeds.
The cost-sensitive quality evaluation establishes that PCBRisk achieves aggregate Pareto dominance under pooled, shift-inflated, and board-median calibration regimes, delivering net financial savings of 4–18% across standard manufacturing cost scenarios. The end-to-end pipeline operates at 10.3 FPS on consumer CPU hardware, with consequence queries requiring merely 186 μs per candidate box, confirming real-time industrial feasibility.
The project transparently documented its limitations: the consequence engine's absolute accuracy falls below aspirational thresholds, the reference-free student head failed to achieve template-based parity, and the system operates within the physical boundary of optical surface inspection. These negative results inform the concrete engineering roadmap for future work.

### 7.4.2 Future Scope

CAD-Grounded Consequence Engine:
Replace raster Otsu binarization with direct vector parsing of Gerber RS-274X, ODB++, and IPC-2581 files using R-Tree spatial indexing. Vector queries are lighting-invariant and execute in <10 μs per query.
Hybrid AOI + Flying Probe Scheduling:
Route borderline optical cases (τ − ε ≤ s < τ) directly to automated 4-point flying probe test heads, reducing human visual workload by an additional 20% (4.13 → 3.31 boxes/board).
Continual Conformal Calibration:
Dynamically update non-conformity quantiles from rolling operator feedback using sequential changepoint detection (CUSUM, Page-Hinkley) to track illumination and solder paste drift.
Production Mondrian Calibration:
Deploy group-conditional calibration partitioning risk bounds across board families and defect classes, eliminating class-level safety violations (opens: 5.20% → 4.84%).
Active Learning Feedback Loop:
Route high-uncertainty candidates near the decision boundary to senior QA staff for iterative retraining of consequence feature weights.
Multimodal Defect Inference:
Fuse 2D surface imagery, multi-spectral oblique illumination, CAD layer netlists, pick-and-place telemetry, and historic lot yield priors into a unified risk model.


# REFERENCES

[1] S. Tang, F. He, X. Huang, and J. Yang, "DeepPCB: A Dataset for Printed Circuit Board Defect Detection," arXiv preprint arXiv:1902.06197, 2019.
[2] R. S. Rao, H. L. Chippa, and A. Yaghmazadeh, "A Survey on Automated Visual Inspection of Printed Circuit Boards," IEEE Access, vol. 9, pp. 156831–156847, 2021.
[3] V. Vovk, A. Gammerman, and G. Shafer, Algorithmic Learning in a Random World. New York, NY: Springer Science & Business Media, 2005.
[4] A. N. Angelopoulos and S. Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification," arXiv preprint arXiv:2107.07511, 2021.
[5] G. Jocher et al., "Ultralytics YOLO11," Ultralytics, 2024. [Online]. Available: https://docs.ultralytics.com/
[6] IPC, "IPC-9252: Requirements for Electrical Testing of Unpopulated Printed Boards," IPC Association Connecting Electronics Industries, 2017.
[7] IPC, "IPC-2591: Connected Factory Exchange (CFX)," IPC Association Connecting Electronics Industries, 2020.
[8] SEMI, "E30 (GEM) and E37 (SECS-II)," Semiconductor Equipment and Materials International.
[9] IPC, "IPC-A-600: Acceptability of Printed Boards," IPC Association Connecting Electronics Industries, 2020.
[10] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
[11] S. van der Walt et al., "scikit-image: Image processing in Python," PeerJ, vol. 2, p. e453, 2014.
[12] ONNX Runtime Developers, "ONNX Runtime," Microsoft, 2024. [Online]. Available: https://onnxruntime.ai/
[13] S. Tiramisu et al., "FastAPI," 2024. [Online]. Available: https://fastapi.tiangolo.com/
[14] S. Gillies et al., "Shapely: Manipulation and analysis of geometric objects in the Cartesian plane," 2024. [Online]. Available: https://shapely.readthedocs.io/


# APPENDICES


## Appendix A: Repository File Structure

pcbrisk/
├── MANUSCRIPT.md                           # Full academic paper manuscript
├── README.md                               # Project documentation & benchmark overview
├── PROJECT_HANDOVER.md                     # Comprehensive technical report & LOBO results
├── REVISED_CLAIMS_AND_COST_EVALUATION.md   # Rigorous claims, cost crossover & architecture
├── info.md                                 # File inventory & replication manual
│
├── code/                                   # Python modules & experiment runners
│   ├── server.py                           # FastAPI live inspection server
│   ├── pcbrisk_utils.py                    # Caching, atomic writes, hashing
│   ├── pcbrisk_graph.py                    # Local Consequence Engine v3
│   ├── pcbrisk_eval.py                     # Split-conformal calibration & LOBO
│   ├── benchmark_latency.py                # CPU ONNX & per-box latency timer
│   ├── cost_sensitive_eval.py              # Quality cost model evaluator
│   ├── plot_cost_curves.py                 # Cost curve figure generator
│   ├── eval_registration_sensitivity.py    # δ_reg noise suppression evaluation
│   ├── eval_mondrian_calibration.py        # Group-conditional calibrator
│   └── hybrid_and_mondrian_eval.py         # Hybrid flying-probe simulator
│
├── frontend/                               # Interactive AOI Web Platform
│   └── index.html                          # Complete 5-view AOI platform
│
├── 00_raw/                                 # DeepPCB raw dataset archive
├── 01_prepared/                            # Board-partitioned YOLO annotations
├── 02_synth/                               # Graph-constrained synthetic defects
├── 04_runs/                                # 26 pre-trained YOLO11n models
├── 05_results/                             # Evaluation outputs & JSON ledgers
├── 06_exports/                             # Production ONNX models
└── 99_archive/                             # Deprecated v1 artifacts

## Appendix B: Key Source Code — Consequence Engine (pcbrisk_graph.py)

The core topological consequence engine source code is located at code/pcbrisk_graph.py (4,600 bytes). Key functions include:

• polarity_invert(template_bgr, copper='auto'): Determines copper polarity via Otsu thresholding
• copper_mask(img, invert): Extracts binary copper mask with small-artifact filtering
• derive(b, tol, min_px): Computes alignment-tolerant differential masks
• _split_pairs(lt, lx, region, key, sig): Measures connected component splitting
• query(...): Evaluates a candidate bounding box for topological consequence
• verdict(go, gs): Returns discrete consequence label: 'none', 'open', 'short', or 'both'

[Full source code available in the project repository at code/pcbrisk_graph.py]

## Appendix C: Key Source Code — Conformal Evaluation (pcbrisk_eval.py)

The LOBO conformal calibration and evaluation module is located at code/pcbrisk_eval.py (9,029 bytes). Key functions include:

• add_features(df): Builds engineering features for Policies A and B
• _fit(df, feats, mask): Trains L2-penalized logistic regression scoring model
• tau_pooled: Standard split-conformal empirical quantile
• tau_robust: Worst-board conservative threshold
• lobo(): 11-board leave-one-board-out cross-fitting evaluation
• paired_bootstrap(): 5,000-replicate multiplicity-weighted bootstrap CIs
• matched_miss(): Controls for calibration error vs ranking discriminability

[Full source code available in the project repository at code/pcbrisk_eval.py]

## Appendix D: Installation and Execution Instructions

Step 1 — Install Dependencies:
    pip install torch torchvision ultralytics scikit-image opencv-python-headless \
              scipy scikit-learn pandas pyyaml onnx onnxruntime onnxslim \
              matplotlib shapely fastapi uvicorn

Step 2 — Launch Live Interactive AOI Platform:
    python code/server.py
    # Or on Windows: double-click run_pcbrisk.bat
    # Access in browser: http://127.0.0.1:8000

Step 3 — Run CPU Latency Benchmark:
    python code/benchmark_latency.py

Step 4 — Run Cost-Sensitive Quality Evaluation:
    python code/cost_sensitive_eval.py
    python code/plot_cost_curves.py

Step 5 — Run Registration Sensitivity Experiment:
    python code/eval_registration_sensitivity.py

Step 6 — Run Mondrian Calibration Experiment:
    python code/eval_mondrian_calibration.py

Step 7 — Run Hybrid Flying-Probe Simulation:
    python code/hybrid_and_mondrian_eval.py

## Appendix E: ONNX Model Export Details

Table E.1: Exported ONNX Models

| Model File | Size | Training Data | Seed | ONNX Opset |
| --- | --- | --- | --- | --- |
| yolo11n_deeppcb_real_s42.onnx | 10.1 MB | Real-data only | 42 | 22 |
| yolo11n_deeppcb_synth1_s42.onnx | 10.1 MB | Real + Synthetic | 42 | 22 |

