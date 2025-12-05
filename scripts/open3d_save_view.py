import open3d as o3d
import os
import argparse

"""
Reliable screenshot script for Windows.
Loads *.ply files and saves PNG using an ABSOLUTE path.
"""

def load_if_exists(path):
    if os.path.exists(path):
        return o3d.io.read_point_cloud(path) if path.endswith(".ply") else None
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--basename", type=str, required=True)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()

    base = os.path.join(args.dir, args.basename)

    pcd = load_if_exists(base + "_points.ply")
    axes = load_if_exists(base + "_axes.ply")
    bbox = load_if_exists(base + "_pred_bboxes.ply")

    geometries = []
    if pcd: geometries.append(pcd)
    if axes: geometries.append(axes)
    if bbox: geometries.append(bbox)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=args.width, height=args.height, visible=False)
    for g in geometries:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()

    # Convert save-path to ABSOLUTE
    save_abs_path = os.path.abspath(args.save_path)
    os.makedirs(os.path.dirname(save_abs_path), exist_ok=True)

    print("[INFO] Saving screenshot to:", save_abs_path)
    vis.capture_screen_image(save_abs_path)
    vis.destroy_window()
    print("[INFO] Screenshot saved successfully.")

if __name__ == "__main__":
    main()
