"""eval_registration_sensitivity.py - Evaluates the effect of tolerance parameter (tol) on alignment error suppression and defect signal preservation."""
import zipfile
import cv2
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pcbrisk_graph import base, derive

def run_experiment():
    with zipfile.ZipFile('00_raw/deeppcb_main.zip') as z:
        temp_bytes = z.read('DeepPCB/tools/examples/grouptest/test/12100002_temp.jpg')
        test_bytes = z.read('DeepPCB/tools/examples/grouptest/test/12100002_test.jpg')

    temp_img = cv2.imdecode(np.frombuffer(temp_bytes, np.uint8), cv2.IMREAD_COLOR)
    test_img = cv2.imdecode(np.frombuffer(test_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 1. Base alignment (ground truth image pair with natural defect)
    b_natural = base(temp_img, test_img)
    
    # 2. Perfect alignment on defect-free template (test = temp)
    b_defect_free = base(temp_img, temp_img)
    
    # 3. Defect-free template with artificial 1-pixel and 2-pixel registration shift
    # Shift temp_img by (dx=1, dy=0) and (dx=1, dy=1)
    M1 = np.float32([[1, 0, 1], [0, 1, 0]])
    M2 = np.float32([[1, 0, 1], [0, 1, 1]])
    temp_shifted_1px = cv2.warpAffine(temp_img, M1, (640, 640))
    temp_shifted_2px = cv2.warpAffine(temp_img, M2, (640, 640))
    
    b_shift_1px = base(temp_img, temp_shifted_1px)
    b_shift_2px = base(temp_img, temp_shifted_2px)

    results = {}
    tolerances = [0, 1, 2, 3]

    for tol in tolerances:
        # Measure spurious noise on 1px shifted defect-free board
        d_s1 = derive(b_shift_1px, tol=tol)
        spurious_1px = int(d_s1['miss'].sum() + d_s1['exc'].sum())
        
        # Measure spurious noise on 2px shifted defect-free board
        d_s2 = derive(b_shift_2px, tol=tol)
        spurious_2px = int(d_s2['miss'].sum() + d_s2['exc'].sum())
        
        # Measure real defect signal retention on the natural defect board
        d_nat = derive(b_natural, tol=tol)
        real_defect_px = int(d_nat['miss'].sum() + d_nat['exc'].sum())
        
        results[f"tol_{tol}"] = {
            "tolerance_pixels": tol,
            "approx_physical_um": tol * 25.0, # at 25 um/pixel
            "spurious_noise_1px_shift_px": spurious_1px,
            "spurious_noise_2px_shift_px": spurious_2px,
            "natural_defect_signal_retained_px": real_defect_px,
            "relative_defect_signal_retention": float(real_defect_px / (derive(b_natural, tol=0)['miss'].sum() + derive(b_natural, tol=0)['exc'].sum()))
        }

    os.makedirs('05_results', exist_ok=True)
    with open('05_results/registration_sensitivity.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Registration sensitivity analysis completed:")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    run_experiment()
