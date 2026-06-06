# RTMPose Multi-View Triangulation Validation

This repository validates a multi-view human pose pipeline:

1. Run RTMPose on synchronized camera videos.
2. Triangulate 2D keypoints with DLT.
3. Compute MPJPE against mocap 3D ground truth.
4. Export summary metrics and skeleton overlay videos for reporting.

The code was built for Blender-rendered validation videos from
`blender_scripting`, but the script also accepts explicit video and prediction
paths.

## Repository Layout

```text
.
├── src/validate_rtmpose_triangulation.py
├── tests/test_geometry.py
├── environment.yml
├── pyproject.toml
└── README.md
```

Generated outputs are written under `output/` and are ignored by git.

## External Data

The default run expects the companion `blender_scripting` project at:

```text
/home/haoyucen/Documents/blender_scripting
```

Default input videos:

```text
/home/haoyucen/Documents/blender_scripting/output/render/rendered_videos/val_3cam_00.mp4
/home/haoyucen/Documents/blender_scripting/output/render/rendered_videos/val_3cam_01.mp4
/home/haoyucen/Documents/blender_scripting/output/render/rendered_videos/val_3cam_02.mp4
```

Each video must have a same-name sidecar JSON file, for example
`val_3cam_00.json`. The sidecar supplies camera intrinsics, extrinsics,
resolution, and render FPS. Mocap ground truth is loaded from:

```text
<blender-root>/data/mocap/<trial>.json
```

Use `--blender-root` or repeated `--video CAMERA_ID PATH` arguments to point at
different data.

## Environment

Use Python 3.10. The most reliable setup on this machine is to install PyTorch
from the official CUDA wheel first, then install OpenMMLab packages.

```bash
conda env create -f environment.yml
conda activate rtmose_vali

python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
python -m pip install --no-build-isolation "chumpy==0.70"
mim install "mmpose>=1.3.0"

python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
python - <<'PY'
import torch
from mmpose.apis import MMPoseInferencer

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("mmpose inferencer:", type(MMPoseInferencer(pose2d="human", device="cpu")).__name__)
PY
```

Notes:

- `numpy<2` is required for compatibility with this OpenMMLab stack.
- `mmcv==2.1.0` is pinned because newer `mmcv` versions can conflict with
  `mmdet 3.x` and `mmpose 1.x`.
- If `mim install mmengine` fails because importing `torch` fails, use
  `python -m pip install mmengine` instead.

## Run

Smoke test:

```bash
conda activate rtmose_vali
python src/validate_rtmpose_triangulation.py \
  --max-frames 5 \
  --output output/val_3cam_smoke
```

Full default validation:

```bash
python src/validate_rtmpose_triangulation.py \
  --output output/val_3cam
```

By default the script uses:

```text
--rtmpose-model human
--bbox-mode foreground
--score-threshold 0.3
--min-views 2
```

`--bbox-mode foreground` estimates a tight person bounding box from the clean
Blender foreground. This works better than sending the whole image to a top-down
pose estimator. For real videos with complex backgrounds, use
`--bbox-mode detector` or provide precomputed prediction JSONs.

Useful options:

```text
--bbox-mode {foreground,detector,whole_image}
--foreground-threshold 20
--bbox-padding 0.15
--foreground-min-area 500
--score-threshold 0.3
--overlay-score-threshold 0.1
--no-overlay
--no-overlay-gt
--force-inference
```

If cached prediction JSONs were generated with a different bbox mode or key
settings, the script reruns RTMPose automatically unless those predictions are
explicitly supplied with `--prediction`.

## Outputs

Example output directory:

```text
output/val_3cam/
├── predictions/camera0_rtmpose.json
├── predictions/camera1_rtmpose.json
├── predictions/camera2_rtmpose.json
├── overlays/camera0_skeleton_overlay.mp4
├── overlays/camera1_skeleton_overlay.mp4
├── overlays/camera2_skeleton_overlay.mp4
├── cameras.json
├── triangulated_3d.json
├── metrics.json
├── summary.json
└── summary.txt
```

`summary.txt` contains report-ready metrics, for example:

```text
Mean MPJPE: 46.30 mm
Median MPJPE: 33.51 mm
P90 MPJPE: 83.09 mm
2D observations above threshold: 1829 / 2088 (2088 possible)
Point counts (total / above threshold / DLT attempted / triangulated): 696 / 696 / 695 / 695
```

MPJPE is computed between DLT-triangulated 3D COCO body joints and mocap
ground-truth joint centers after downsampling mocap to the video FPS.

Point-count definitions:

- `total`: number of frame-joint targets, usually `frames * joints`.
- `above threshold`: targets with at least one camera observation above
  `--score-threshold`.
- `DLT attempted`: targets with at least `--min-views` valid camera
  observations.
- `triangulated`: targets that successfully produced a 3D point.

Overlay colors:

- Green: RTMPose 2D prediction.
- Red: projected mocap 3D ground truth.

## Tests

```bash
pytest -q
```

## Prepare For GitHub

The generated `output/` directory, caches, local editor settings, videos, and
model checkpoints are ignored by `.gitignore`.

To upload this project to a new GitHub repository:

```bash
cd /home/haoyucen/Documents/rtmose_vali
git init
git add README.md environment.yml pyproject.toml .gitignore src tests
git commit -m "Initial RTMPose triangulation validation workflow"
git branch -M main
git remote add origin git@github.com:<USER>/<REPO>.git
git push -u origin main
```

Replace `<USER>/<REPO>` with the actual GitHub repository path.
