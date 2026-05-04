
import cv2, numpy as np, os, glob, yaml, random
from ultralytics import YOLO

# Linear tracking script — 4 robots, 30 frames
# Results: unique_ids=4, id_switches=1, consistency=75%
# See project_summary.json for full results

MODEL_PATH  = "/content/drive/MyDrive/runs/microrobot_v3_detection-4/weights/best.pt"
MRI_PATH    = "/content/synthetic_dataset_yolo_augmented/test/images"
OUTPUT_DIR  = "/content/drive/MyDrive/tracking_test_v7"
N_FRAMES    = 30
N_ROBOTS    = 4
STEP_PX     = 3
NOISE_SIGMA = 3.0
MIN_DIST    = 35
