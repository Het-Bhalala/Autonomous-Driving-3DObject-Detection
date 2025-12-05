import cv2
import os

"""
make_demo_video.py

Creates a single demo video that shows all HW2 experiments:

1. KITTI - PointPillars (000123_2d_vis.png)
2. KITTI - 3DSSD (000123_2d_vis.png)
3. KITTI - SECOND (000123_2d_vis.png)
4. nuScenes - PointPillars (Open3D screenshot)
5. nuScenes - CenterPoint (Open3D screenshot)

Output: results/demo_all_experiments.mp4
"""

frames = [
    "outputs/kitti_pointpillars/000123_2d_vis.png",
    "outputs/kitti_3dssd/000123_2d_vis.png",
    "outputs/kitti_second/000123_2d_vis.png",
    "results/screenshots/nuscenes_pointpillars_open3d.png",
    "results/screenshots/nuscenes_centerpoint_open3d.png",
]

# Keep only existing files
frames = [f for f in frames if os.path.exists(f)]
if not frames:
    raise RuntimeError("No valid frames found. Check paths and generate images first.")

print("Using frames:")
for f in frames:
    print("  ✓", f)

first_img = cv2.imread(frames[0])
if first_img is None:
    raise RuntimeError(f"Failed to load first frame: {frames[0]}")

height, width = first_img.shape[:2]
fps = 4
seconds_per_image = 2.5
repeat_frames = int(fps * seconds_per_image)

os.makedirs("results", exist_ok=True)
out_path = "results/demo_all_experiments.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

for img_path in frames:
    img = cv2.imread(img_path)
    if img is None:
        print("Warning: skipping unreadable frame:", img_path)
        continue

    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))

    for _ in range(repeat_frames):
        video.write(img)

video.release()
print("\nDemo video created successfully:")
print(os.path.abspath(out_path))
