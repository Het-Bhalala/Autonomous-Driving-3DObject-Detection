import os
import time
import csv
import subprocess
from pathlib import Path
import sys  # NEW: to get the current interpreter path

"""
run_all_experiments.py

Small helper script to run all the HW2 experiments in one go.

Assumptions:
- This file lives in the same folder as mmdet3d_inference2.py
- Virtualenv is already activated when you run:  python run_all_experiments.py
- Data + checkpoints are laid out as described in README.md
"""

# Use the *same* Python interpreter that is running this script (i.e., the venv one)
PYTHON = sys.executable

# Change this if your entry script has a different filename
MMDET3D_SCRIPT = "mmdet3d_inference2.py"

# Where to save timing summary
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIMINGS_CSV = RESULTS_DIR / "experiment_timings.csv"


def build_cmd(args_dict):
    """
    Build a `python mmdet3d_inference2.py ...` command from a dictionary of args.

    Keys with value None or "" are skipped.
    """
    cmd = [PYTHON, MMDET3D_SCRIPT]  # use venv python, not bare "python"
    for k, v in args_dict.items():
        if v is None or v == "":
            continue
        flag = f"--{k}"
        # boolean flags (like headless) can be passed as just the flag
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
            # if False, skip entirely
            continue
        cmd.append(flag)
        cmd.append(str(v))
    return cmd


def run_experiment(name, args_dict):
    cmd = build_cmd(args_dict)
    print("\n" + "=" * 80)
    print(f"Running experiment: {name}")
    print("Command:")
    print("  " + " ".join(cmd))
    print("=" * 80)

    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as e:
        t1 = time.perf_counter()
        print(f"[{name}] FAILED to launch: {e}")
        return {
            "name": name,
            "success": False,
            "seconds": t1 - t0,
            "error": str(e),
        }

    t1 = time.perf_counter()
    elapsed = t1 - t0

    if result.returncode == 0:
        print(f"[{name}] finished successfully in {elapsed:.2f} seconds.")
        return {
            "name": name,
            "success": True,
            "seconds": elapsed,
            "error": "",
        }
    else:
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


def main():
    experiments = [
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
        {
            "name": "kitti_second_000123",
            "args": {
                "dataset": "kitti",
                "input-path": "data/kitti/training",
                "frame-number": "000123",
                "model": "second_hv_secfpn_8xb6-80e_kitti-3d-3class",
                # adjust filename if your .pth hash suffix differs
                "checkpoint": "checkpoints/kitti_second/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth",
                "out-dir": "outputs/kitti_second",
                "device": "cuda:0",
                "headless": True,
                "score-thr": 0.05,
            },
        },
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

    results = []
    for exp in experiments:
        res = run_experiment(exp["name"], exp["args"])
        results.append(res)

    fieldnames = ["name", "success", "seconds", "error"]
    with TIMINGS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"{r['name']:30s}  {status:4s}  {r['seconds']:.2f} s  {r['error']}")

    print(f"\nTiming CSV written to: {TIMINGS_CSV.resolve()}")


if __name__ == "__main__":
    main()
