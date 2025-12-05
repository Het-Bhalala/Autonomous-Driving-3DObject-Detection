## 3D Object Detection on KITTI & nuScenes using MMDetection3D

I performed 3D object detection on **two datasets (KITTI + nuScenes)** using **multiple models**, saved visualizations (`.png`, `.ply`, `.json`), generated Open3D screenshots, and produced a combined demo video.  
All experiments were executed on my local machine using MMDetection3D with custom modifications.

---

# 1. Environment Setup

### System
- **OS:** Windows 11  
- **GPU:** NVIDIA GeForce MX350  
- **Python:** 3.10  
- **CUDA:** 11.8 (via PyTorch wheels)

### 1.1 Create virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
````

### 1.2 Install ML and 3D dependencies

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
mim install "mmengine==0.10.7" "mmcv==2.1.0" "mmdet==3.2.0" "mmdet3d==1.4.0"

pip install open3d opencv-python tqdm numpy==1.26.4
```

Verify installation:

```bash
python -c "import mmengine, mmcv, mmdet, mmdet3d; \
print(mmengine.__version__, mmcv.__version__, mmdet.__version__, mmdet3d.__version__)"
```

---

# 2. Pretrained Models Used

All checkpoints were downloaded using:

```bash
# KITTI models
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest checkpoints/kitti_pointpillars
mim download mmdet3d --config 3dssd_4x4_kitti-3d-car --dest checkpoints/3dssd
mim download mmdet3d --config second_hv_secfpn_8xb6-80e_kitti-3d-3class --dest checkpoints/kitti_second

# nuScenes models
mim download mmdet3d --config pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d --dest checkpoints/nuscenes_pointpillars
mim download mmdet3d --config centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d --dest checkpoints/nuscenes_centerpoint
```

Each folder contains a config `.py` file and a `.pth` checkpoint (excluded from GitHub).

---

# 3. Dataset Structure (Not Included in Repo)

The datasets are stored locally and excluded from GitHub.

## 3.1 KITTI mini-subset structure

```
data/kitti/training/
    velodyne/
        000123.bin
    image_2/
        000123.png
    calib/
        000123.txt  
```

## 3.2 nuScenes demo sample

```
data/nuscenes_demo/
    lidar/
        n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin
    images/
        CAM_FRONT, CAM_BACK, etc.
```

# 4. Main Inference Script

All detections were generated using my modified script:

```
mmdet3d_inference2.py
```

Additions include:

* **Preset-based argument filling** (less typing per experiment)
* **Headless mode** for non-GUI inference
* Automatic saving of:

  * `*_2d_vis.png`
  * `*_points.ply`
  * `*_pred_bboxes.ply`
  * `*_pred_labels.ply`
  * `*_predictions.json`
* Support for KITTI and single-file nuScenes inference

Modifications are clearly marked with comments such as:

# 5. Automation Script

I created a helper:

```
run_all_experiments.py
```

Running:

```bash
python run_all_experiments.py
```

executes all five experiments and logs timing information to:

```
results/experiment_timings.csv
```

# 6. Comparison & Analysis

To compare performance across datasets and models, I evaluated five detectors on two datasets (KITTI and nuScenes).  
Metrics extracted automatically using `run_all_experiments.py` and `compare_results_to_csv.py` include:

- **Latency** (seconds per frame)  
- **FPS** (1 / latency)  
- **Number of detections**  
- **Average score** (mean confidence of predicted bounding boxes)  

These metrics provide a realistic picture of computational cost and detection behavior without needing full mAP evaluation.

### 6.1 Quantitative Model Comparison

| Dataset   | Model         | Latency (s) | FPS     | # Detections | Avg Score |
|-----------|---------------|-------------|---------|--------------|-----------|
| KITTI     | PointPillars  | **124.27**  | 0.0080  | 10           | 0.6074    |
| KITTI     | 3DSSD         | **14.35**   | 0.0697  | 100          | 0.0536    |
| KITTI     | SECOND        | **12.23**   | 0.0817  | 0            | –         |
| nuScenes  | PointPillars  | **12.71**   | 0.0786  | 365          | 0.1266    |
| nuScenes  | CenterPoint   | **25.40**   | 0.0393  | 264          | 0.2443    |

### 6.2 Key Observations

1. **3DSSD is dramatically faster than KITTI PointPillars**, running almost **9× faster** while producing many more detections.  
   - It achieves the best speed–accuracy balance for KITTI on a low-power GPU like the MX350.

2. **PointPillars excels in confidence but not speed (KITTI)**.  
   - Although slow (124s per frame), it gives the **highest average detection score (0.6074)**, indicating stable detections but heavy computation.

3. **SECOND produces no valid detections** on the chosen frame.  
   - This result aligns with earlier warnings about **weight–architecture mismatches** in mmdet3d 1.4.0.  
   - SECOND is included as a *negative example* of model compatibility issues.

4. **CenterPoint outperforms PointPillars on nuScenes in quality**, despite being slower.  
   - PointPillars: 365 detections, avg score 0.1266  
   - CenterPoint: 264 detections, avg score 0.2443  
   - Fewer detections but **much higher-quality predictions**.

5. **nuScenes produces far more detections than KITTI**, due to:  
   - 360° LiDAR coverage  
   - Dense urban scenes  
   - Multi-class prediction  
   - Better model–dataset alignment (CenterPoint is designed for nuScenes)

### 6.3 Conclusion of Analysis

- **Best KITTI model:** 3DSSD (speed + high detection count)  
- **Best nuScenes model:** CenterPoint (high-confidence detections)  
- **Most consistent cross-dataset model:** PointPillars  
- **Poor performance:** SECOND (compatibility issue, zero detections)  

Latency, FPS, and detection density clearly show how model architecture interacts with dataset characteristics, especially under GPU limitations.

---

# 7. Visualization with Open3D

To render and save `.png` screenshots reliably on Windows, I used:

```
scripts/open3d_save_view.py
```


Screenshots used in the report:

* `results/screenshots/kitti_pointpillars_000123.png`
* `results/screenshots/kitti_3dssd_000123.png`
* `results/screenshots/kitti_second_000123.png`
* `results/screenshots/nuscenes_pointpillars_open3d.png`
* `results/screenshots/nuscenes_centerpoint_open3d.png`

---

# 8. Demo Video

I created a combined demo video using:

```
make_demo_video.py
```

Running:

```bash
python make_demo_video.py
```

produces:

```
results/demo_all_experiments.mp4
```

The video displays:

1. KITTI – PointPillars
2. KITTI – 3DSSD
3. KITTI – SECOND
4. nuScenes – PointPillars (Open3D screenshot)
5. nuScenes – CenterPoint (Open3D screenshot)

Each frame is shown for ~2.5 seconds.

---

# 9. Repository Structure (Submitted Version)

```
homework-2/
  README.md
  report.md
  mmdet3d_inference2.py
  run_all_experiments.py
  make_demo_video.py
  scripts/
    open3d_view_saved_ply.py
    open3d_save_view.py
  checkpoints/
    <config .py files only, no .pth>
  results/
    demo_all_experiments.mp4
    screenshots/
      *.png
  outputs/
    <optional small PNG/PLY/JSON examples>
  .gitignore
```

Datasets and `.venv/` are excluded.

---

# 10. Notes & Limitations

* SECOND’s pretrained checkpoint produced shape mismatch warnings, yielding weaker detections.
* Runtime varies substantially due to GPU limitations (MX350).
* Qualitative comparison focuses on *relative* model behavior across identical frames.

---

