
import cv2, numpy as np, os, glob, yaml, random
from ultralytics import YOLO

# V-curve catheter tracking script — 5 robots, 40 frames
# V-shape: R0(top-left) R1 R2(center) R3 R4(top-right) -> move right
# Results: unique_ids=5, id_switches=1, consistency=80%
# See project_summary.json for full results

MODEL_PATH  = "/content/drive/MyDrive/runs/microrobot_v3_detection-4/weights/best.pt"
MRI_PATH    = "/content/synthetic_dataset_yolo_augmented/test/images"
OUTPUT_DIR  = "/content/drive/MyDrive/tracking_vcurve"
N_FRAMES    = 40
N_ROBOTS    = 5
STEP_PX     = 3
NOISE_SIGMA = 3.0

def get_vcurve_positions(cx_center, cy_center, spacing=40):
    return [
        (cx_center - 2*spacing, cy_center - spacing),
        (cx_center - 1*spacing, cy_center - spacing//2),
        (cx_center,             cy_center),
        (cx_center + 1*spacing, cy_center - spacing//2),
        (cx_center + 2*spacing, cy_center - spacing),
    ]
