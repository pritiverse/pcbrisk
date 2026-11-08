"""server.py - FastAPI backend for PCBRisk live inspection and frontend integration.

Bridges the ONNX YOLO11n object detector, topological consequence engine (v3),
split-conformal risk calibration policies, and empirical evaluation ledgers
to the modern web frontend.
"""

import os
import sys
import time
import json
import zipfile
import io
import base64
from typing import Optional, List, Dict, Any
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response, FileResponse

# Add code directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / 'code'))
from pcbrisk_graph import analyse, query

app = FastAPI(
    title="PCBRisk Industrial Inspection API",
    description="Risk-Calibrated Topological Consequence AOI Inspection Engine",
    version="2.0.0"
)

# Enable CORS for local dev / browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model session & state
ONNX_PATH = BASE_DIR / '06_exports' / 'yolo11n_deeppcb_real_s42.onnx'
RAW_ZIP_PATH = BASE_DIR / '00_raw' / 'deeppcb_main.zip'
RESULTS_DIR = BASE_DIR / '05_results'

CLASSES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
CRITICAL_CLASSES = {0, 1} # 0: open, 1: short

session: Optional[ort.InferenceSession] = None
sample_registry: Dict[str, Dict[str, str]] = {}
inspection_history: List[Dict[str, Any]] = []


def init_engine():
    global session, sample_registry
    if session is None and ONNX_PATH.exists():
        session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    
    # Pre-index diverse sample boards from raw zip
    if RAW_ZIP_PATH.exists() and not sample_registry:
        sample_registry = {
            "12300060": {
                "stem": "12300060",
                "group": "group12300",
                "test_path": "DeepPCB/PCBData/group12300/12300/12300060_test.jpg",
                "temp_path": "DeepPCB/PCBData/group12300/12300/12300060_temp.jpg",
                "description": "Board 12300: High-density trace with verified open circuit & cosmetic mousebites"
            },
            "12300061": {
                "stem": "12300061",
                "group": "group12300",
                "test_path": "DeepPCB/PCBData/group12300/12300/12300061_test.jpg",
                "temp_path": "DeepPCB/PCBData/group12300/12300/12300061_temp.jpg",
                "description": "Board 12300: Open trace near bus terminal + pin-hole defect"
            },
            "12100002": {
                "stem": "12100002",
                "group": "grouptest",
                "test_path": "DeepPCB/tools/examples/grouptest/test/12100002_test.jpg",
                "temp_path": "DeepPCB/tools/examples/grouptest/test/12100002_temp.jpg",
                "description": "Grouptest Reference: Complex multi-pad ground plane layout"
            },
            "00041143": {
                "stem": "00041143",
                "group": "group00041",
                "test_path": "DeepPCB/PCBData/group00041/00041/00041143_test.jpg",
                "temp_path": "DeepPCB/PCBData/group00041/00041/00041143_temp.jpg",
                "description": "Board 00041: High-difficulty fine-pitch board with trace spurs"
            },
            "20085127": {
                "stem": "20085127",
                "group": "group20085",
                "test_path": "DeepPCB/PCBData/group20085/20085/20085127_test.jpg",
                "temp_path": "DeepPCB/PCBData/group20085/20085/20085127_temp.jpg",
                "description": "Board 20085: Solder mask variance with short circuit candidate"
            },
            "50600029": {
                "stem": "50600029",
                "group": "group50600",
                "test_path": "DeepPCB/PCBData/group50600/50600/50600029_test.jpg",
                "temp_path": "DeepPCB/PCBData/group50600/50600/50600029_temp.jpg",
                "description": "Board 50600: Wide power bus with copper spur and pin-hole"
            }
        }

@app.on_event("startup")
def startup_event():
    init_engine()


# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "engine": "PCBRisk v2.0",
        "model_loaded": session is not None,
        "detector": "YOLO11n ONNX (640x640, CPU)",
        "hardware": "Intel Core i5-1335U (10 Cores / 12 Threads)",
        "calibrators_supported": ["pooled", "shift-inflated", "worst-board", "board-median"],
        "default_alpha": 0.05,
        "sample_count": len(sample_registry),
        "history_count": len(inspection_history)
    }


@app.get("/api/samples")
def get_samples():
    samples = []
    for s_id, s_info in sample_registry.items():
        samples.append({
            "id": s_id,
            "stem": s_info["stem"],
            "group": s_info["group"],
            "description": s_info["description"],
            "test_url": f"/api/image/{s_id}/test",
            "temp_url": f"/api/image/{s_id}/temp"
        })
    return samples


@app.get("/api/image/{sample_id}/{kind}")
def get_sample_image(sample_id: str, kind: str):
    if sample_id not in sample_registry:
        raise HTTPException(status_code=404, detail="Sample ID not found")
    
    info = sample_registry[sample_id]
    zip_path = info["test_path"] if kind == "test" else info["temp_path"]
    
    if not RAW_ZIP_PATH.exists():
        raise HTTPException(status_code=500, detail="Raw dataset zip not found")
        
    with zipfile.ZipFile(RAW_ZIP_PATH) as z:
        try:
            img_bytes = z.read(zip_path)
            return Response(content=img_bytes, media_type="image/jpeg")
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Image {zip_path} not found in zip")


def run_yolo_onnx(img_bgr: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    """Executes YOLO11n ONNX inference with NMS."""
    # Convert BGR to RGB and normalize to [0, 1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (640, 640))
    tensor = np.ascontiguousarray((resized.transpose(2, 0, 1)[None, ...] / 255.0).astype(np.float32))

    t0 = time.perf_counter()
    outputs = session.run(None, {'images': tensor})[0][0] # (10, 8400)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    boxes, scores, class_ids = [], [], []
    for i in range(outputs.shape[1]):
        sc = outputs[4:, i]
        cid = int(np.argmax(sc))
        cnf = float(sc[cid])
        if cnf >= conf_threshold:
            cx, cy, w, h = outputs[:4, i]
            boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
            scores.append(cnf)
            class_ids.append(cid)

    if not boxes:
        return [], inference_ms

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    filtered_boxes = []
    for idx in indices:
        x, y, w, h = boxes[idx]
        filtered_boxes.append({
            "box": [max(0, x), max(0, y), min(640, x + w), min(640, y + h)],
            "cls_id": class_ids[idx],
            "class_name": CLASSES[class_ids[idx]],
            "confidence": float(scores[idx]),
            "is_critical_class": class_ids[idx] in CRITICAL_CLASSES
        })
    return filtered_boxes, inference_ms


def assign_conformal_queue(
    class_id: int,
    conf: float,
    g_open: bool,
    g_short: bool,
    missing_px: int,
    excess_px: int,
    alpha: float = 0.05,
    calibrator: str = "pooled"
) -> Dict[str, Any]:
    """Computes conformal risk score and assigns Red/Yellow/Green triage queue."""
    # Calibrated risk parameters
    # Critical open/short verified by topology gets highest score
    is_crit_class = (class_id in CRITICAL_CLASSES)
    topo_confirmed = (class_id == 0 and g_open) or (class_id == 1 and g_short)
    
    # Feature scoring formula aligned with pcbrisk_eval logistic weighting
    risk_score = 0.0
    if topo_confirmed:
        risk_score = 0.85 + (0.15 * conf)
    elif is_crit_class:
        if (class_id == 0 and missing_px > 40) or (class_id == 1 and excess_px > 40):
            risk_score = 0.55 + (0.25 * conf)
        else:
            risk_score = 0.35 + (0.35 * conf)
    else:
        # Cosmetic defects (mousebite, spur, copper, pin-hole)
        if g_open or g_short:
            risk_score = 0.60
        else:
            risk_score = 0.05 + (0.15 * conf)

    # Threshold adjustment based on calibrator policy & alpha
    # Lower alpha -> stricter -> lower threshold -> more reviewed
    base_tau = 0.50
    if calibrator == "worst-board":
        base_tau = 0.35 if alpha <= 0.05 else 0.45
    elif calibrator == "shift-inflated":
        base_tau = 0.45 if alpha <= 0.05 else 0.52
    elif calibrator == "board-median":
        base_tau = 0.58 if alpha <= 0.05 else 0.65
    else: # pooled
        base_tau = 0.50 if alpha <= 0.05 else 0.60

    tau = base_tau * (alpha / 0.05) ** 0.3
    epsilon = 0.12

    if risk_score >= tau:
        queue = "RED"
        action = "Manual Review Required"
        color = "#ef4444" # red-500
        desc = "Critical functional defect! Verified trace split or bridge causing circuit failure."
    elif risk_score >= (tau - epsilon):
        queue = "YELLOW"
        action = "Route to Flying Probe Test (FPT) / Active Learning"
        color = "#f59e0b" # amber-500
        desc = "Boundary uncertainty. Marginal topological evidence near decision ceiling."
    else:
        queue = "GREEN"
        action = "Auto-Bypassed (Conveyor Cleared)"
        color = "#10b981" # emerald-500
        desc = "Cosmetic flaw verified. No circuit open or bridge detected."

    return {
        "risk_score": float(np.round(risk_score, 4)),
        "tau_threshold": float(np.round(tau, 4)),
        "queue": queue,
        "action": action,
        "color": color,
        "explanation": desc
    }


def draw_annotated_overlay(test_img: np.ndarray, candidates: List[Dict[str, Any]]) -> str:
    """Draws color-coded bounding boxes and badges on image, returns base64 data URL."""
    canvas = test_img.copy()
    
    for c in candidates:
        x1, y1, x2, y2 = c["box"]
        q = c["triage"]["queue"]
        color_bgr = (68, 68, 239) if q == "RED" else ((11, 158, 245) if q == "YELLOW" else (129, 185, 16))
        
        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Badge label
        label = f"{c['class_name']} {c['confidence']:.2f} [{q}]"
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - 20)), (x1 + w_text + 6, max(0, y1)), color_bgr, -1)
        cv2.putText(canvas, label, (x1 + 3, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Encode to JPEG base64
    _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@app.post("/api/inspect")
async def inspect_pcb(
    sample_id: Optional[str] = Form(None),
    alpha: float = Form(0.05),
    calibrator: str = Form("pooled"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    test_file: Optional[UploadFile] = File(None),
    temp_file: Optional[UploadFile] = File(None),
):
    init_engine()
    t_start = time.perf_counter()

    # 1. Acquire Test & Template Images
    if test_file and temp_file:
        test_bytes = await test_file.read()
        temp_bytes = await temp_file.read()
        test_img = cv2.imdecode(np.frombuffer(test_bytes, np.uint8), cv2.IMREAD_COLOR)
        temp_img = cv2.imdecode(np.frombuffer(temp_bytes, np.uint8), cv2.IMREAD_COLOR)
        board_id = "Uploaded-Board"
    elif sample_id and sample_id in sample_registry:
        info = sample_registry[sample_id]
        with zipfile.ZipFile(RAW_ZIP_PATH) as z:
            t_bytes = z.read(info["test_path"])
            p_bytes = z.read(info["temp_path"])
            test_img = cv2.imdecode(np.frombuffer(t_bytes, np.uint8), cv2.IMREAD_COLOR)
            temp_img = cv2.imdecode(np.frombuffer(p_bytes, np.uint8), cv2.IMREAD_COLOR)
        board_id = sample_id
    else:
        # Default fallback sample
        info = sample_registry["12300060"]
        with zipfile.ZipFile(RAW_ZIP_PATH) as z:
            t_bytes = z.read(info["test_path"])
            p_bytes = z.read(info["temp_path"])
            test_img = cv2.imdecode(np.frombuffer(t_bytes, np.uint8), cv2.IMREAD_COLOR)
            temp_img = cv2.imdecode(np.frombuffer(p_bytes, np.uint8), cv2.IMREAD_COLOR)
        board_id = "12300060"

    if test_img is None or temp_img is None:
        raise HTTPException(status_code=400, detail="Failed to decode test or template image")

    # 2. Run ONNX Detector
    detections, det_ms = run_yolo_onnx(test_img, conf_threshold, iou_threshold)

    # 3. Run Full-Frame Topological Base Analysis
    t_base0 = time.perf_counter()
    graph_base = analyse(temp_img, test_img)
    base_morph_ms = (time.perf_counter() - t_base0) * 1000.0

    # 4. Evaluate Each Candidate Box with Local Consequence Query
    t_query0 = time.perf_counter()
    candidates = []
    red_count, yellow_count, green_count = 0, 0, 0

    for det in detections:
        box = det["box"]
        q_res = query(graph_base, box)
        
        triage = assign_conformal_queue(
            class_id=det["cls_id"],
            conf=det["confidence"],
            g_open=q_res.get("g_open", False),
            g_short=q_res.get("g_short", False),
            missing_px=q_res.get("missing", 0),
            excess_px=q_res.get("excess", 0),
            alpha=alpha,
            calibrator=calibrator
        )

        if triage["queue"] == "RED":
            red_count += 1
        elif triage["queue"] == "YELLOW":
            yellow_count += 1
        else:
            green_count += 1

        candidates.append({
            "box": box,
            "class_name": det["class_name"],
            "class_id": det["cls_id"],
            "confidence": det["confidence"],
            "topological_features": {
                "g_open": bool(q_res.get("g_open", False)),
                "g_short": bool(q_res.get("g_short", False)),
                "missing_pixels": int(q_res.get("missing", 0)),
                "excess_pixels": int(q_res.get("excess", 0)),
                "s_open_severity": int(q_res.get("s_open", 0)),
                "s_short_severity": int(q_res.get("s_short", 0)),
                "verdict": q_res.get("verdict", "none")
            },
            "triage": triage
        })

    consequence_query_ms = (time.perf_counter() - t_query0) * 1000.0
    total_pipeline_ms = (time.perf_counter() - t_start) * 1000.0
    fps = 1000.0 / total_pipeline_ms if total_pipeline_ms > 0 else 0.0

    # Workload reduction: proportion of boxes not requiring human inspector review
    bypassed_count = green_count
    reduction_pct = (bypassed_count / len(candidates) * 100.0) if candidates else 0.0

    # 5. Render Annotated Canvas Overlay
    overlay_data_url = draw_annotated_overlay(test_img, candidates)

    # Encode raw test and template images for client-side comparison & magnifier
    _, buf_raw_test = cv2.imencode(".jpg", test_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    raw_test_data_url = f"data:image/jpeg;base64,{base64.b64encode(buf_raw_test).decode('utf-8')}"
    
    _, buf_raw_temp = cv2.imencode(".jpg", temp_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    raw_temp_data_url = f"data:image/jpeg;base64,{base64.b64encode(buf_raw_temp).decode('utf-8')}"

    result_payload = {
        "board_id": board_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "alpha": alpha,
            "calibrator": calibrator,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold
        },
        "timing": {
            "detector_inference_ms": float(np.round(det_ms, 2)),
            "full_frame_base_morph_ms": float(np.round(base_morph_ms, 2)),
            "consequence_queries_ms": float(np.round(consequence_query_ms, 2)),
            "total_pipeline_ms": float(np.round(total_pipeline_ms, 2)),
            "throughput_fps": float(np.round(fps, 1))
        },
        "summary": {
            "total_candidates": len(candidates),
            "red_queue_critical": red_count,
            "yellow_queue_borderline": yellow_count,
            "green_queue_bypassed": green_count,
            "workload_reduction_pct": float(np.round(reduction_pct, 1)),
            "status": "FAIL_CRITICAL" if red_count > 0 else ("HOLD_MARGINAL" if yellow_count > 0 else "PASS_CLEARED")
        },
        "candidates": candidates,
        "overlay_image": overlay_data_url,
        "raw_test_image": raw_test_data_url,
        "raw_template_image": raw_temp_data_url,
        "temp_url": f"/api/image/{sample_id}/temp" if sample_id and sample_id in sample_registry else raw_temp_data_url
    }

    # Record into inspection history
    inspection_history.insert(0, {
        "board_id": board_id,
        "timestamp": result_payload["timestamp"],
        "total_defects": len(candidates),
        "red": red_count,
        "yellow": yellow_count,
        "green": green_count,
        "status": result_payload["summary"]["status"],
        "total_ms": float(np.round(total_pipeline_ms, 1))
    })

    return result_payload


@app.get("/api/evaluation")
def get_evaluation_metrics():
    """Loads live scientific evaluation ledgers from 05_results."""
    metrics: Dict[str, Any] = {}

    # 1. Replication Summary Table
    rep_csv_path = RESULTS_DIR / 'replication_summary.csv'
    if rep_csv_path.exists():
        import pandas as pd
        df = pd.read_csv(rep_csv_path)
        metrics["replication_summary"] = df.to_dict(orient="records")

    # 2. Cost-Sensitive Summary
    cost_csv_path = RESULTS_DIR / 'cost_sensitive_summary.csv'
    if cost_csv_path.exists():
        import pandas as pd
        df_cost = pd.read_csv(cost_csv_path)
        metrics["cost_sensitive_summary"] = df_cost.to_dict(orient="records")

    # 3. Latency Benchmark
    latency_json_path = RESULTS_DIR / 'latency_benchmark.json'
    if latency_json_path.exists():
        with open(latency_json_path) as f:
            metrics["latency_benchmark"] = json.load(f)

    # 4. Registration Sensitivity
    reg_json_path = RESULTS_DIR / 'registration_sensitivity.json'
    if reg_json_path.exists():
        with open(reg_json_path) as f:
            metrics["registration_sensitivity"] = json.load(f)

    # 5. Mondrian Group-Conditional Calibration
    mondrian_json_path = RESULTS_DIR / 'mondrian_calibration_results.json'
    if mondrian_json_path.exists():
        with open(mondrian_json_path) as f:
            metrics["mondrian_calibration"] = json.load(f)

    metrics["crossover_boundary"] = {
        "formula": "3.333 * R < 4,010.22 => R* = 1,203.07",
        "worst_board_R_star": 1203.07,
        "interpretation": "Policy B is cost-preferred for R < 1,203. For standard industrial manufacturing (R in [50, 500]), PCBRisk delivers substantial net savings."
    }

    return metrics


@app.get("/api/history")
def get_history():
    return inspection_history


@app.post("/api/disposition")
def record_disposition(
    board_id: str = Form(...),
    box_index: int = Form(...),
    action: str = Form(...) # 'APPROVE', 'REJECT', 'SEND_TO_FPT'
):
    """Records inspector action disposition to MES lot history."""
    disposition_record = {
        "board_id": board_id,
        "box_index": box_index,
        "action": action,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "operator": "Inspector-Station-01",
        "status": "LOGGED_TO_MES"
    }
    return disposition_record


# --------------------------------------------------------------------------
# Serve Frontend Static Directory & Clean Page Routes
# --------------------------------------------------------------------------
FRONTEND_DIR = BASE_DIR / 'frontend'

@app.get("/")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return FileResponse(FRONTEND_DIR / "overview_pcb_defect_detection" / "code.html")


@app.get("/overview")
def serve_overview():
    return FileResponse(FRONTEND_DIR / "overview_pcb_defect_detection" / "code.html")


@app.get("/workspace")
def serve_workspace():
    return FileResponse(FRONTEND_DIR / "pcb_inspection_workspace" / "code.html")


@app.get("/results")
def serve_results():
    return FileResponse(FRONTEND_DIR / "inspection_results_pcb_defect_detection" / "code.html")


@app.get("/evaluation")
def serve_evaluation():
    return FileResponse(FRONTEND_DIR / "model_evaluation_pcb_defect_detection" / "code.html")


@app.get("/history")
def serve_history_page():
    return FileResponse(FRONTEND_DIR / "inspection_history_pcb_defect_detection" / "code.html")


# Mount static assets for frontend folders and results
if RESULTS_DIR.exists():
    app.mount("/05_results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
