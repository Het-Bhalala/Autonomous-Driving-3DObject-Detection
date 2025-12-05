"""
run_all_experiments.py  (HW2 – Automated Evaluation Script)
-----------------------------------------------------------

This script runs all 3D detection experiments required for CMPE 249 HW2.
It automates the full evaluation pipeline:

1. Execute mmdet3d_inference2.py for each model–dataset pair.
2. Measure wall-clock runtime for each experiment.
3. Record failures (if any) and print clean summaries.
4. Save results into results/experiment_timings.csv for analysis.

This ensures consistent, repeatable evaluation across models.
The script is intentionally heavily commented for clarity and grading.
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import os
import time
import csv
import subprocess
from pathlib import Path
import sys  # ensures we use the SAME Python (venv) that runs this script


# ----------------------------------------------------------------------
# GLOBAL SETTINGS
# ----------------------------------------------------------------------

# Use the currently active Python interpreter (important on Windows!)
PYTHON = sys.executable

# This is the main inference script used for each experiment
MMDET3D_SCRIPT = "mmdet3d_inference2.py"

# Where to write timing summary CSV
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIMINGS_CSV = RESULTS_DIR / "experiment_timings.csv"


# ----------------------------------------------------------------------
# Helper: Construct a CLI command for subprocess.run()
# ----------------------------------------------------------------------
def build_cmd(args_dict):
    """
    Build a subprocess command:

        python mmdet3d_inference2.py --arg1 val1 --arg2 val2 ...

    Boolean flags (e.g., headless=True) are included only when True.
    Empty/None arguments are skipped entirely.
    """
    cmd = [PYTHON, MMDET3D_SCRIPT]  # Always use venv Python

    for key, value in args_dict.items():
        if value is None or value == "":
            continue  # skip unused flags

        flag = f"--{key}"

        # Boolean flag (e.g., --headless)
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue

        # Regular flag with value
        cmd.append(flag)
        cmd.append(str(value))

    return cmd


# ----------------------------------------------------------------------
# Helper: Run a single experiment and measure runtime
# ----------------------------------------------------------------------
def run_experiment(name, args_dict):
    """
    Runs one HW2 experiment by:
      - Building the CLI command
      - Calling mmdet3d_inference2.py
      - Measuring runtime
      - Capturing stdout/stderr
      - Returning a summary dict

    Returns dict:
      {
        "name": experiment name,
        "success": True/False,
        "seconds": runtime,
        "error": "" or message
      }
    """

    cmd = build_cmd(args_dict)

    print("\n" + "=" * 80)
    print(f"Running experiment: {name}")
    print("Command:")
    print("  " + " ".join(cmd))
    print("=" * 80)

    start = time.perf_counter()

    try:
        # capture_output=True → prevents terminal spam
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as e:
        end = time.perf_counter()
        print(f"[{name}] FAILED to launch: {e}")
        return {
            "name": name,
            "success": False,
            "seconds": end - start,
            "error": str(e),
        }

    end = time.perf_counter()
    elapsed = end - start

    # Check if inference succeeded
    if result.returncode == 0:
        print(f"[{name}] finished successfully in {elapsed:.2f} seconds.")
        return {
            "name": name,
            "success": True,
            "seconds": elapsed,
            "error": "",
        }

    # On failure, print stderr/stdout for debugging
    print(f"[{name}] FAILED with return code {result.returncode} in {elapsed:.2f} seconds.")
    print("---- STDOUT ----")
    print(result.stdout)
    print("---- STDERR ----")
    print(result.stderr)

    return {
        "name": name,
        "success": False,
        "seconds": elapsed,
        "error": f"returncode={result.returncode}",
    }


# ----------------------------------------------------------------------
# MAIN: Define and run all HW2 experiments
# ----------------------------------------------------------------------
def main():
    """
    Defines all five experiments used for HW2:

    KITTI:
        1. PointPillars
        2. 3DSSD
        3. SECOND

    nuScenes:
        4. PointPillars
        5. CenterPoint

    Each experiment specifies:
      - dataset
      - input-path
      - frame-number (KITTI)
      - model + checkpoint
      - output folder
      - GPU device
      - score threshold
      - headless rendering
    """

    experiments = [

        # --------------------------------------------------------------
        # 1) KITTI – PointPillars (frame 000123)
        # --------------------------------------------------------------
        {
            "name": "kitti_pointpillars_000123",
            "args": {
                "dataset": "kitti",
                "input-path": "data/kitti/training",
                "frame-number": "000123",
                "model": "pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car",
                "checkpoint": "checkpoints/kitti_pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth",
                "out-dir": "outputs/kitti_pointpillars",
                "device": "cuda:0",
                "headless": True,
                "score-thr": 0.3,
            },
        },

        # --------------------------------------------------------------
        # 2) KITTI – 3DSSD
        # --------------------------------------------------------------
        {
            "name": "kitti_3dssd_000123",
            "args": {
                "dataset": "kitti",
                "input-path": "data/kitti/training",
                "frame-number": "000123",
                "model": "3dssd_4x4_kitti-3d-car",
                "checkpoint": "checkpoints/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth",
                "out-dir": "outputs/kitti_3dssd",
                "device": "cuda:0",
                "headless": True,
                "score-thr": 0.6,
            },
        },

        # --------------------------------------------------------------
        # 3) KITTI – SECOND (weaker baseline)
        # --------------------------------------------------------------
        {
            "name": "kitti_second_000123",
            "args": {
                "dataset": "kitti",
                "input-path": "data/kitti/training",
                "frame-number": "000123",
                "model": "second_hv_secfpn_8xb6-80e_kitti-3d-3class",
                "checkpoint": "checkpoints/kitti_second/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth",
                "out-dir": "outputs/kitti_second",
                "device": "cuda:0",
                "headless": True,
                "score-thr": 0.05,
            },
        },

        # --------------------------------------------------------------
        # 4) nuScenes – PointPillars (single lidar sample)
        # --------------------------------------------------------------
        {
            "name": "nuscenes_pointpillars_demo",
            "args": {
                "dataset": "any",
                "input-path": "data/nuscenes_demo/lidar/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin",
                "model": "pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d",
                "checkpoint": "checkpoints/nuscenes_pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth",
                "out-dir": "outputs/nuscenes_pointpillars",
                "device": "cuda:0",
                "headless": True,
                "score-thr": 0.2,
            },
        },

        # --------------------------------------------------------------
        # 5) nuScenes – CenterPoint
        # --------------------------------------------------------------
        {
            "name": "nuscenes_centerpoint_demo",
            "args": {
                "dataset": "any",
                "input-path": "data/nuscenes_demo/lidar/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin",
                "model": "centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d",
                "checkpoint": "checkpoints/nuscenes_centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic-20e_nus-3d_20220810_030004-9061688e.pth",
                "out-dir": "outputs/nuscenes_centerpoint",
                "device": "cuda:0",
                "headless": True,
                "score-thr": 0.25,
            },
        },
    ]

    # ------------------------------------------------------------------
    # Run each experiment and collect results
    # ------------------------------------------------------------------
    results = []
    for exp in experiments:
        res = run_experiment(exp["name"], exp["args"])
        results.append(res)

    # ------------------------------------------------------------------
    # Write timing summary CSV
    # ------------------------------------------------------------------
    fieldnames = ["name", "success", "seconds", "error"]
    with TIMINGS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # ------------------------------------------------------------------
    # Print summary to console
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Summary of HW2 Experiments")
    print("=" * 80)

    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"{r['name']:30s}  {status:4s}  {r['seconds']:.2f} s  {r['error']}")

    print(f"\nTiming CSV written to: {TIMINGS_CSV.resolve()}")
    print("Next step: run  python compare_results_to_csv.py  to compute metrics.\n")


# ----------------------------------------------------------------------
# Program Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
