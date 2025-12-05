## 3D Object Detection on KITTI & nuScenes using MMDetection3D

This repository contains my full implementation of HW2. I performed 3D object detection on **two datasets (KITTI + nuScenes)** using **multiple models**, saved visualizations (`.png`, `.ply`, `.json`), generated Open3D screenshots, and produced a combined demo video.  
All experiments were executed on my local machine using MMDetection3D with custom modifications.

---

# 1. Environment Setup

### System
- **OS:** Windows 10  
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
        000008.bin
        000123.bin
    image_2/
        000008.png
        000123.png
    calib/
        000008.txt
        000123.txt   # reused 000008 calibration for consistent projection
    label_2/
        000008.txt   # optional GT
```

## 3.2 nuScenes demo sample

```
data/nuscenes_demo/
    lidar/
        n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin
    images/
        CAM_FRONT, CAM_BACK, etc.
```

---

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

```python
# ---- HW2 Custom Preset ----
```

---

# 5. Inference Commands Used

## 5.1 KITTI – PointPillars (frame: 000123)

```bash
python mmdet3d_inference2.py ^
  --dataset kitti ^
  --input-path data/kitti/training ^
  --frame-number 000123 ^
  --model pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car ^
  --checkpoint checkpoints/kitti_pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth ^
  --out-dir outputs/kitti_pointpillars ^
  --device cuda:0 ^
  --headless ^
  --score-thr 0.3
```

## 5.2 KITTI – 3DSSD

```bash
python mmdet3d_inference2.py ^
  --dataset kitti ^
  --input-path data/kitti/training ^
  --frame-number 000123 ^
  --model 3dssd_4x4_kitti-3d-car ^
  --checkpoint checkpoints/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth ^
  --out-dir outputs/kitti_3dssd ^
  --device cuda:0 ^
  --headless ^
  --score-thr 0.6
```

## 5.3 KITTI – SECOND (weaker baseline)

```bash
python mmdet3d_inference2.py ^
  --dataset kitti ^
  --input-path data/kitti/training ^
  --frame-number 000123 ^
  --model second_hv_secfpn_8xb6-80e_kitti-3d-3class ^
  --checkpoint checkpoints/kitti_second/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth ^
  --out-dir outputs/kitti_second ^
  --device cuda:0 ^
  --headless ^
  --score-thr 0.05
```

## 5.4 nuScenes – PointPillars

```bash
python mmdet3d_inference2.py ^
  --dataset any ^
  --input-path data/nuscenes_demo/lidar/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin ^
  --model pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d ^
  --checkpoint checkpoints/nuscenes_pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth ^
  --out-dir outputs/nuscenes_pointpillars ^
  --device cuda:0 ^
  --headless ^
  --score-thr 0.2
```

## 5.5 nuScenes – CenterPoint

```bash
python mmdet3d_inference2.py ^
  --dataset any ^
  --input-path data/nuscenes_demo/lidar/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin ^
  --model centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d ^
  --checkpoint checkpoints/nuscenes_centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic-20e_nus-3d_20220810_030004-9061688e.pth ^
  --out-dir outputs/nuscenes_centerpoint ^
  --device cuda:0 ^
  --headless ^
  --score-thr 0.25
```

---

# 6. Automation Script

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

### Final runtimes:

| Experiment              | Runtime (sec) |
| ----------------------- | ------------- |
| KITTI – PointPillars    | **124.27**    |
| KITTI – 3DSSD           | **14.35**     |
| KITTI – SECOND          | **12.23**     |
| nuScenes – PointPillars | **12.71**     |
| nuScenes – CenterPoint  | **25.40**     |

---

# 7. Visualization with Open3D

To render and save `.png` screenshots reliably on Windows, I used:

```
scripts/open3d_save_view.py
```

Example:

```bash
python scripts/open3d_save_view.py ^
  --dir outputs/nuscenes_pointpillars ^
  --basename n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd ^
  --width 1600 --height 1200 ^
  --save-path results/screenshots/nuscenes_pointpillars_open3d.png
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

* For KITTI frame `000123`, calibration was adapted from `000008` to enable 2D projection.
* SECOND’s pretrained checkpoint produced shape mismatch warnings, yielding weaker detections.
* Runtime varies substantially due to GPU limitations (MX350).
* Qualitative comparison focuses on *relative* model behavior across identical frames.

---

If you want, I can now also create a **final 1–2 page report.md** to fully complete your submission.
```
