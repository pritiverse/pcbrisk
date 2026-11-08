# Project Progress Log: PCBRisk

**Repository:** `pritiverse/pcbrisk`  
**Initial Baseline Date:** November 2026 / October 2026  
**Objective:** Push and establish initial clean Git tracking for the PCBRisk industrial research framework on GitHub (`https://github.com/pritiverse/pcbrisk`).

---

## Current Phase & Task Checklist

- [x] Initial Repository Audit & Remote Verification (Gate A)
  - Confirmed directory: `C:\Users\priti\Downloads\pcbrisk`
  - Confirmed remote repository is reachable, empty, and authenticated (`pritiverse`)
  - Identified file size blocker: `02_synth/synth_v2.tar` (130 MB > 100 MB GitHub limit)
- [x] Initialize Git repository on branch `main`
- [x] Configure `.gitignore` to protect against >100 MB limits and exclude large training runs & tarballs
- [x] Create project progress tracking log (`docs/PROJECT_PROGRESS.md`)
- [x] Review untracked files & verify total payload size (49.25 MB across 120 files)
- [x] Stage verified source and results files (Gate F approval granted)
- [x] Create initial commit on `main` (Commit `f830e0e`)
- [x] Push to `origin main` (Successfully pushed and verified up-to-date with `origin/main`)

---

## Files Inspected & Managed

### Tracked & Prepared for Commit
- **Core Engine & Server:** `code/*.py` (FastAPI backend, topological consequence engine, conformal evaluators)
- **Frontend Platform:** `frontend/*` (Interactive AOI dashboard, HTML templates, screen captures)
- **Documentation & Papers:** `README.md`, `MANUSCRIPT.md`, `PROJECT_HANDOVER.md`, `REVISED_CLAIMS_AND_COST_EVALUATION.md`, `TEAM_WORK_SPLIT.md`, `NOTEBOOK_*.md`
- **Jupyter Notebooks:** `01_datasets_to_drive (1).ipynb` through `07_replication_and_calibration (1).ipynb`
- **Evaluation Artifacts:** `05_results/*` (Metrics, LOBO calibrations, confusion matrices, bootstrap forest plots, publication figures)
- **Model Exports:** `06_exports/*.onnx` (YOLO11n ONNX CPU inference models, 10.6 MB each)
- **Automation Scripts:** `run_pcbrisk.bat`, `insert_screenshots.py`, `update_page_numbers.py`

### Excluded by `.gitignore` (Size & Artifact Management)
- `00_raw/deeppcb_main.zip` (86.8 MB)
- `01_prepared/prepared_v2.tar` (50.0 MB)
- `02_synth/synth_v2.tar` (130.2 MB — exceeds GitHub 100MB limit)
- `04_runs/` (307.9 MB — training runs & PyTorch `.pt` checkpoints)
- `99_archive/` (89.0 MB — legacy v1 splits)
- `__pycache__/` and Python bytecode
- `*ORIGINAL_BACKUP.docx`

---

## Verification & Validation Commands

- `git branch --show-current`: Verified `main`
- `git remote -v`: Verified `origin https://github.com/pritiverse/pcbrisk.git`
- `git check-ignore -v 02_synth/synth_v2.tar`: Confirmed oversized archives are properly ignored
- `git status -u`: Confirmed all proposed files for staging are <15 MB

