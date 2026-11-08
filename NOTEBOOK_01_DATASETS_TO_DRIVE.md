# Notebook 01: Dataset Acquisition, Cryptographic Checksumming & Raw Asset Storage
**Filename:** [`01_datasets_to_drive (1).ipynb`](file:///c:/Users/priti/Downloads/pcbrisk/01_datasets_to_drive%20(1).ipynb)  
**Execution Runtime:** CPU (Google Colab / Local Drive mount)  
**Primary Role:** Downloads and archives the canonical DeepPCB dataset, computes cryptographic SHA-256 checksums, validates image-template pair counts across 11 manufacturing groups, maintains research licensing provenance in `00_raw/SOURCES.md`, and specifies boundaries for auxiliary CAD Gerber and IEEE DataPort assets.

---

## 1. Motivation & Data Provenance

Automated Optical Inspection (AOI) research often suffers from irreproducible data preparation, missing provenance, and accidental license violations. 

`01_datasets_to_drive (1).ipynb` establishes an immutable, single-source-of-truth ingestion foundation for the entire **PCBRisk** research pipeline:
1. **Automated Ingestion:** Directly clones the canonical **DeepPCB** dataset repository published by Tang et al. (2019) from GitHub.
2. **Cryptographic Checksumming:** Hashes all compressed raw archives with **SHA-256** into an append-only registry (`00_raw/checksums.txt`) to guarantee bit-level repeatability across workstations and Colab runtimes.
3. **Idempotent Single-Fetch Logic:** Checks existing Drive/storage locations before initiating network requests, avoiding redundant downloads and bandwidth consumption.
4. **License & Provenance Documentation:** Automatically records the acquisition URL, fetch timestamp, citation, and license terms (`00_raw/SOURCES.md`). The DeepPCB dataset is released under a **Research Use Only** license; this boundary is strictly enforced before any patent or production deployment.
5. **Auxiliary Data Ingestion Hooks:** Provides structured procedures for manual uploads of IEEE DataPort Gerber files and secret-safe Kaggle/HuggingFace API token management.

---

## 2. Methodology & Code Walkthrough

```
[Cell 0: Markdown]  -> Runtime specification (CPU), execution guidelines, dataset overview table
[Cell 1: Code]      -> Google Drive mount, environment setup, importing pcbrisk_utils, seed fixing (seed=42)
[Cell 2: Code]      -> Clone DeepPCB from GitHub, strip .git directory, compress to deeppcb_main.zip, write SOURCES.md
[Cell 3: Code]      -> Unzip archive to local scratch (/content/raw), count test/template pairs and board groups
[Cell 4: Markdown]  -> Guidance on manual Gerber samples / IEEE DataPort dataset ingestion
[Cell 5: Code]      -> Compute and append SHA-256 hashes to 00_raw/checksums.txt
[Cell 6: Markdown]  -> Colab Secrets security guidelines for Kaggle / Hugging Face API keys
[Cell 7: Code]      -> Optional Kaggle CLI dataset fetch using userdata.get()
[Cell 8: Code]      -> Optional Hugging Face snapshot download using userdata.get('HF_TOKEN')
```

### Detailed Ingestion Steps:
1. **GitHub Shallow Clone:**
   ```bash
   git clone --depth 1 https://github.com/tangsanli5201/DeepPCB $RAW/DeepPCB
   ```
2. **Clean Tarball Packaging:**
   Packages `DeepPCB/` into `deeppcb_main.zip` while strictly excluding Git version control metadata (`-x "DeepPCB/.git/*"`), reducing artifact size and filesystem overhead.
3. **Cryptographic Verification (`append_checksum`):**
   Computes the SHA-256 hex digest of the zip file and records it in `00_raw/checksums.txt`.
4. **Pair Integrity Count:**
   Glob-matches all `*_test.jpg` and `*_temp.jpg` files under `PCBData/group*/*` to assert that all 11 board groups are fully present.

---

## 3. Execution Log & Measured Results

### Verified Raw Dataset Statistics:
```text
Mounted at /content/drive
already in Drive: /content/drive/MyDrive/pcbrisk/00_raw/deeppcb_main.zip
1500 test imgs, 1501 template imgs
11 boards (groups)
```

*(Note: The $1,500$ test images and $1,501$ template images reflect the official DeepPCB repository distribution, which contains one stray template image in `group12300` that is safely matched and handled in downstream splitting).*

### Identified Board Groups (11 Manufacturing Groups):
* `group00041`
* `group12000`
* `group12100`
* `group12300`
* `group13000`
* `group20085`
* `group44000`
* `group50600`
* `group77000`
* `group90100`
* `group92000`

### Cryptographic Checksum Registry ([`00_raw/checksums.txt`](file:///c:/Users/priti/Downloads/pcbrisk/00_raw/checksums.txt)):
```text
5654be20ec5b84c10e7b234ce2fce78b0c046251dbca513b9b6c4255256af3ca  deeppcb_main.zip
```

---

## 4. Engineering Considerations & Security Protocols

1. **Strict Credential Isolation (Zero Hardcoded Keys):**  
   Notebook 01 explicitly enforces Google Colab Secrets management:
   ```python
   from google.colab import userdata
   os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
   os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
   ```
   Hardcoding secret tokens or API keys into notebook source cells is strictly prohibited to avoid token leaks in public repository mirrors.
2. **Idempotency & Cost Optimization:**  
   Colab runtimes and cloud storage mounts can be interrupted. The execution checks `if not os.path.exists(zip_drive):` before running network operations, preventing corrupted overwrites and unnecessary re-downloads.
3. **Research-Only Licensing Boundary:**  
   The record in [`00_raw/SOURCES.md`](file:///c:/Users/priti/Downloads/pcbrisk/00_raw/SOURCES.md) enforces that the DeepPCB benchmark dataset is restricted to academic and research evaluation. Synthetic and physical testing extensions must adhere to industrial IP guidelines.

---

## 5. Artifacts Produced

| Artifact File | Size | Role |
|---|---|---|
| [`00_raw/deeppcb_main.zip`](file:///c:/Users/priti/Downloads/pcbrisk/00_raw/deeppcb_main.zip) | $\approx 125\text{ MB}$ | Immutable compressed archive of canonical DeepPCB images and defect bounding-box annotations |
| [`00_raw/checksums.txt`](file:///c:/Users/priti/Downloads/pcbrisk/00_raw/checksums.txt) | $172\text{ B}$ | Append-only SHA-256 cryptographic checksum ledger for data integrity verification |
| [`00_raw/SOURCES.md`](file:///c:/Users/priti/Downloads/pcbrisk/00_raw/SOURCES.md) | $154\text{ B}$ | Data provenance ledger specifying repository origin, fetch dates, citation, and licensing constraints |

---

## 6. Downstream Pipeline Integration

The raw assets ingested by Notebook 01 serve as the direct input to:
- [`02_prepare_split (1).ipynb`](file:///c:/Users/priti/Downloads/pcbrisk/02_prepare_split%20(1).ipynb) (detailed in [`NOTEBOOK_02_PREPARE_SPLIT.md`](file:///c:/Users/priti/Downloads/pcbrisk/NOTEBOOK_02_PREPARE_SPLIT.md)), which resolves the fatal v1 image-level data leakage bug by constructing an 11-board group-partitioned train/val/test split and packaging normalized YOLO format files into [`01_prepared/prepared_v2.tar`](file:///c:/Users/priti/Downloads/pcbrisk/01_prepared/prepared_v2.tar).
- The live inspection server ([`code/server.py`](file:///c:/Users/priti/Downloads/pcbrisk/code/server.py)), which dynamically streams board samples and golden templates from `00_raw/deeppcb_main.zip` to the interactive AOI frontend.
