
# V-Curve Tracking v2 — Distance Constrained + Spline Curve
# Results: unique_ids=5, id_switches=2, consistency=60%, false_pos=0.10
# Scenario: 5 robots in V-curve catheter formation, moving right, 40 frames

import cv2, numpy as np, os, glob, imageio, yaml
from ultralytics import YOLO
from scipy.interpolate import splprep, splev

OUTPUT_DIR  = "/content/drive/MyDrive/tracking_vcurve_v2"
N_FRAMES    = 40
N_ROBOTS    = 5
STEP_PX     = 3
ROBOT_DIST  = 40

def get_vcurve_positions(cx_center, cy_center, dist=40):
    a    = 1.0 / (2 * dist)
    xs   = np.linspace(cx_center - 2*dist, cx_center + 2*dist, 500)
    ys   = a * (xs - cx_center)**2 + cy_center
    dx   = np.diff(xs); dy = np.diff(ys)
    arc  = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    targets = np.linspace(0, arc[-1], N_ROBOTS)
    return [(int(xs[np.clip(np.searchsorted(arc,t),0,len(xs)-1)]),
             int(ys[np.clip(np.searchsorted(arc,t),0,len(xs)-1)])) for t in targets]

def enforce_distance_constraint(positions, target_dist=40, iterations=5):
    pts = [list(p) for p in positions]
    for _ in range(iterations):
        for i in range(1, len(pts)):
            dx = pts[i][0]-pts[i-1][0]; dy = pts[i][1]-pts[i-1][1]
            d  = np.sqrt(dx**2+dy**2)+1e-6
            if abs(d-target_dist) > 1:
                f  = target_dist/d
                mx = (pts[i][0]+pts[i-1][0])/2; my = (pts[i][1]+pts[i-1][1])/2
                pts[i-1] = [int(mx-dx*f/2), int(my-dy*f/2)]
                pts[i]   = [int(mx+dx*f/2), int(my+dy*f/2)]
    return [(p[0],p[1]) for p in pts]

def fit_smooth_curve(points, n_pts=200):
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    try:
        tck,_ = splprep([xs,ys], s=0, k=min(3,len(points)-1))
        u = np.linspace(0,1,n_pts)
        xf,yf = splev(u,tck)
        return xf.astype(int), yf.astype(int)
    except:
        return xs.astype(int), ys.astype(int)
