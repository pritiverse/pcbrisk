"""benchmark_latency.py - Measures latency for YOLO ONNX detector and Graph Consequence Engine."""
import time
import zipfile
import json
import numpy as np
import cv2
import onnxruntime as ort
import sys
import os

# Add code dir to sys.path
sys.path.insert(0, os.path.dirname(__file__))
from pcbrisk_graph import analyse, query

def run_benchmark():
    # 1. Load sample images from raw zip
    with zipfile.ZipFile('00_raw/deeppcb_main.zip') as z:
        temp_bytes = z.read('DeepPCB/tools/examples/grouptest/test/12100002_temp.jpg')
        test_bytes = z.read('DeepPCB/tools/examples/grouptest/test/12100002_test.jpg')
    
    temp_img = cv2.imdecode(np.frombuffer(temp_bytes, np.uint8), cv2.IMREAD_COLOR)
    test_img = cv2.imdecode(np.frombuffer(test_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 2. Benchmark ONNX YOLO11n on CPU
    onnx_path = '06_exports/yolo11n_deeppcb_real_s42.onnx'
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Prepare dummy input (1, 3, 640, 640) float32 in [0, 1]
    input_tensor = np.ascontiguousarray((cv2.resize(test_img, (640, 640)).transpose(2, 0, 1)[None, ...] / 255.0).astype(np.float32))

    # Warmup ONNX
    for _ in range(10):
        _ = session.run(None, {input_name: input_tensor})

    # Benchmark ONNX
    n_onnx_runs = 50
    onnx_times = []
    for _ in range(n_onnx_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: input_tensor})
        t1 = time.perf_counter()
        onnx_times.append((t1 - t0) * 1000.0) # ms

    # 3. Benchmark Graph Engine Base Analysis (Full Image: Otsu, Connected Components, Differential Dilation)
    # Warmup
    for _ in range(5):
        _ = analyse(temp_img, test_img)

    n_graph_base_runs = 50
    graph_base_times = []
    for _ in range(n_graph_base_runs):
        t0 = time.perf_counter()
        a = analyse(temp_img, test_img)
        t1 = time.perf_counter()
        graph_base_times.append((t1 - t0) * 1000.0)

    # Precompute maps once for box queries
    a = analyse(temp_img, test_img)

    # 4. Benchmark Query per candidate box
    # Generate realistic candidate boxes (e.g., 20x20 to 60x60 around center/edges)
    rng = np.random.default_rng(42)
    sample_boxes = []
    for _ in range(100):
        cx, cy = rng.integers(50, 590, 2)
        w, h = rng.integers(15, 65, 2)
        sample_boxes.append([cx - w//2, cy - h//2, cx + w//2, cy + h//2])

    # Warmup queries
    for b in sample_boxes[:10]:
        _ = query(a, b)

    n_box_runs = 500
    box_times = []
    for i in range(n_box_runs):
        b = sample_boxes[i % len(sample_boxes)]
        t0 = time.perf_counter()
        res = query(a, b)
        t1 = time.perf_counter()
        box_times.append((t1 - t0) * 1000.0) # ms

    # Calculate statistics
    results = {
        "hardware": "CPU (Intel Core i5-1335U, 10-core / 12-thread)",
        "onnx_detector": {
            "model": "YOLO11n (640x640, ONNX opset 22/slimming)",
            "mean_ms": float(np.mean(onnx_times)),
            "std_ms": float(np.std(onnx_times)),
            "median_ms": float(np.median(onnx_times)),
            "p95_ms": float(np.percentile(onnx_times, 95)),
            "throughput_fps": float(1000.0 / np.mean(onnx_times))
        },
        "graph_engine_image_base": {
            "operation": "Otsu binarization + 8-conn net labeling + differential morphological dilation (640x640)",
            "mean_ms": float(np.mean(graph_base_times)),
            "std_ms": float(np.std(graph_base_times)),
            "median_ms": float(np.median(graph_base_times)),
            "p95_ms": float(np.percentile(graph_base_times, 95))
        },
        "graph_engine_per_candidate_box": {
            "operation": "Local neighborhood windowing + connected net pair splitting (_split_pairs)",
            "mean_ms": float(np.mean(box_times)),
            "std_ms": float(np.std(box_times)),
            "median_ms": float(np.median(box_times)),
            "p95_ms": float(np.percentile(box_times, 95)),
            "p99_ms": float(np.percentile(box_times, 99))
        },
        "typical_board_pipeline": {
            "mean_boxes_per_board": 15.45,
            "projected_total_pipeline_ms": float(np.mean(onnx_times) + np.mean(graph_base_times) + 15.45 * np.mean(box_times)),
            "projected_fps": float(1000.0 / (np.mean(onnx_times) + np.mean(graph_base_times) + 15.45 * np.mean(box_times)))
        }
    }

    os.makedirs('05_results', exist_ok=True)
    with open('05_results/latency_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Latency Benchmark Completed!")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    run_benchmark()
