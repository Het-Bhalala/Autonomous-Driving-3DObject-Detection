import os
import csv
import json
from statistics import mean

"""
compare_results_to_csv.py

Reads:
  - results/experiment_timings.csv   (from run_all_experiments.py)
  - *_predictions.json files under outputs/

Computes per-experiment metrics:
  - latency (seconds per frame)
  - FPS (1 / latency)
  - number of detections
  - average detection score

Writes:
  - results/metrics_summary.csv

Also prints a Markdown table you can paste into report.md.
"""

# --------------------------------------------------------------------
# 1. Configure experiment metadata
# --------------------------------------------------------------------

EXPERIMENTS = {
    "kitti_pointpillars_000123": {
        "dataset": "KITTI",
        "model": "PointPillars",
        "pred_json": os.path.join("outputs", "kitti_pointpillars", "000123_predictions.json"),
    },
    "kitti_3dssd_000123": {
        "dataset": "KITTI",
        "model": "3DSSD",
        "pred_json": os.path.join("outputs", "kitti_3dssd", "000123_predictions.json"),
    },
    "kitti_second_000123": {
        "dataset": "KITTI",
        "model": "SECOND",
        "pred_json": os.path.join("outputs", "kitti_second", "000123_predictions.json"),
    },
    "nuscenes_pointpillars_demo": {
        "dataset": "nuScenes",
        "model": "PointPillars",
        "pred_json": os.path.join(
            "outputs",
            "nuscenes_pointpillars",
            "n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd_predictions.json",
        ),
    },
    "nuscenes_centerpoint_demo": {
        "dataset": "nuScenes",
        "model": "CenterPoint",
        "pred_json": os.path.join(
            "outputs",
            "nuscenes_centerpoint",
            "n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd_predictions.json",
        ),
    },
}

TIMINGS_CSV = os.path.join("results", "experiment_timings.csv")
OUT_CSV = os.path.join("results", "metrics_summary.csv")


# --------------------------------------------------------------------
# 2. Helpers
# --------------------------------------------------------------------

def load_timings(csv_path):
    """Return dict: experiment_name -> time_seconds."""
    timings = {}
    if not os.path.exists(csv_path):
        print(f"[WARN] Timing CSV not found: {csv_path}")
        return timings

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("experiment") or row.get("name") or row.get("exp_name")
            t_str = row.get("time_sec") or row.get("seconds") or row.get("runtime")
            if name and t_str:
                try:
                    timings[name] = float(t_str)
                except ValueError:
                    print(f"[WARN] Could not parse time value '{t_str}' for experiment {name}")
    return timings


def extract_scores_from_json(pred_json_path):
    """
    Try to extract detection scores from the JSON file.
    Handles a few common MMDet3D formats.

    Returns:
        num_dets (int), avg_score (float or None)
    """
    if not os.path.exists(pred_json_path):
        print(f"[WARN] Prediction file not found: {pred_json_path}")
        return 0, None

    with open(pred_json_path, "r") as f:
        data = json.load(f)

    scores = []

    # Case 1: list of data samples
    if isinstance(data, list) and len(data) > 0:
        sample = data[0]
        if isinstance(sample, dict):
            if "pred_instances_3d" in sample:
                inst = sample["pred_instances_3d"]
                if isinstance(inst, dict) and "scores_3d" in inst:
                    scores = inst["scores_3d"]
            elif "scores_3d" in sample:
                scores = sample["scores_3d"]

    # Case 2: dict with top-level fields
    elif isinstance(data, dict):
        if "pred_instances_3d" in data and isinstance(data["pred_instances_3d"], dict):
            inst = data["pred_instances_3d"]
            if "scores_3d" in inst:
                scores = inst["scores_3d"]
        elif "scores_3d" in data:
            scores = data["scores_3d"]

    # Flatten numeric scores
    flat_scores = []
    for s in scores:
        try:
            flat_scores.append(float(s))
        except (TypeError, ValueError):
            continue

    num_dets = len(flat_scores)
    avg_score = mean(flat_scores) if flat_scores else None
    return num_dets, avg_score


# --------------------------------------------------------------------
# 3. Main
# --------------------------------------------------------------------

def main():
    os.makedirs("results", exist_ok=True)

    timings = load_timings(TIMINGS_CSV)

    rows = []
    for exp_name, info in EXPERIMENTS.items():
        dataset = info["dataset"]
        model = info["model"]
        pred_json = info["pred_json"]

        time_sec = timings.get(exp_name)
        if time_sec is None:
            print(f"[WARN] No timing found for {exp_name}, setting time_sec=None")
        fps = (1.0 / time_sec) if time_sec and time_sec > 0 else None

        num_dets, avg_score = extract_scores_from_json(pred_json)

        row = {
            "experiment": exp_name,
            "dataset": dataset,
            "model": model,
            "time_sec": time_sec if time_sec is not None else "",
            "fps": fps if fps is not None else "",
            "num_dets": num_dets,
            "avg_score": avg_score if avg_score is not None else "",
        }
        rows.append(row)

    # Save to CSV
    fieldnames = ["experiment", "dataset", "model", "time_sec", "fps", "num_dets", "avg_score"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Metrics CSV written to: {OUT_CSV}")

    # Also print as a Markdown table
    print("\n=== Markdown Table for report.md ===\n")
    print("| Dataset | Model | Time / frame (s) | FPS | #Detections | Avg Score |")
    print("|---------|-------|------------------|-----|------------:|----------:|")

    for r in rows:
        time_str = f"{float(r['time_sec']):.2f}" if r["time_sec"] != "" else "-"
        fps_str = f"{float(r['fps']):.2f}" if r["fps"] != "" else "-"
        avg_str = f"{float(r['avg_score']):.3f}" if r["avg_score"] != "" else "-"

        print(
            f"| {r['dataset']} | {r['model']} | {time_str} | {fps_str} | {r['num_dets']:>10} | {avg_str} |"
        )

    print("\nDone. You can open metrics_summary.csv in Excel and/or paste the table above into report.md.\n")


if __name__ == "__main__":
    main()
