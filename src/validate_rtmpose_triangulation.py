#!/usr/bin/env python3
"""RTMPose -> DLT triangulation -> MPJPE validation for the 3-camera render."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ORIGINAL_MOCAP_FREQUENCY = 200.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER_ROOT = Path("/home/haoyucen/Documents/blender_scripting")
DEFAULT_TRIAL = "p08_bird_correct"
DEFAULT_VIDEO_NAMES = ["val_3cam_init_00.mp4", "val_3cam_init_01.mp4", "val_3cam_init_02.mp4"]

COCO17_JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO17_TO_MOCAP = {
    "left_shoulder": "LSJC",
    "right_shoulder": "RSJC",
    "left_elbow": "LEJC",
    "right_elbow": "REJC",
    "left_wrist": "LWJC",
    "right_wrist": "RWJC",
    "left_hip": "LHJC",
    "right_hip": "RHJC",
    "left_knee": "LKJC",
    "right_knee": "RKJC",
    "left_ankle": "LAJC",
    "right_ankle": "RAJC",
}

COCO17_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

PRED_COLOR = (50, 220, 50)
GT_COLOR = (30, 90, 255)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=True)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_camera_path(values: list[list[str]] | None) -> dict[int, Path]:
    parsed = {}
    for camera_id_raw, path_raw in values or []:
        camera_id = int(camera_id_raw)
        if camera_id in parsed:
            raise ValueError(f"Duplicate camera id: {camera_id}")
        parsed[camera_id] = Path(path_raw).expanduser().resolve()
    return parsed


def default_video_paths(blender_root: Path) -> dict[int, Path]:
    video_dir = blender_root / "output" / "render" / "rendered_videos"
    return {idx: video_dir / name for idx, name in enumerate(DEFAULT_VIDEO_NAMES)}


def read_video_metadata(video_paths: dict[int, Path]) -> tuple[tuple[int, int], float | None]:
    resolutions = {}
    frame_rates = {}
    for camera_id, video_path in video_paths.items():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video for camera {camera_id}: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video resolution for camera {camera_id}: {video_path}")
        resolutions[camera_id] = (width, height)
        if fps > 0:
            frame_rates[camera_id] = fps

    unique_resolutions = set(resolutions.values())
    if len(unique_resolutions) != 1:
        raise ValueError(f"All videos must share one resolution, got {resolutions}")

    unique_fps = {round(fps, 6) for fps in frame_rates.values()}
    if len(unique_fps) > 1:
        raise ValueError(f"All videos must share one FPS, got {frame_rates}")
    fps = next(iter(unique_fps)) if unique_fps else None
    return next(iter(unique_resolutions)), fps


def load_render_sidecars(video_paths: dict[int, Path]) -> dict[int, dict[str, Any]]:
    sidecars = {}
    for camera_id, video_path in video_paths.items():
        sidecar_path = video_path.with_suffix(".json")
        if not sidecar_path.exists():
            raise FileNotFoundError(f"Missing camera sidecar JSON: {sidecar_path}")
        data = read_json(sidecar_path)
        if "camera" not in data or "intrinsics" not in data["camera"]:
            raise ValueError(f"Sidecar JSON does not contain camera intrinsics: {sidecar_path}")
        sidecars[camera_id] = data
    return sidecars


def sidecar_resolution(sidecars: dict[int, dict[str, Any]]) -> tuple[int, int] | None:
    values = {
        tuple(data.get("camera", {}).get("resolution_px", []))
        for data in sidecars.values()
        if data.get("camera", {}).get("resolution_px")
    }
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"Sidecar resolutions do not match: {values}")
    return next(iter(values))


def sidecar_fps(sidecars: dict[int, dict[str, Any]]) -> float | None:
    values = {
        float(data.get("render", {}).get("fps"))
        for data in sidecars.values()
        if data.get("render", {}).get("fps") is not None
    }
    if not values:
        return None
    rounded = {round(fps, 6) for fps in values}
    if len(rounded) != 1:
        raise ValueError(f"Sidecar FPS values do not match: {values}")
    return next(iter(rounded))


def euler_xyz_degrees_to_matrix(rotation_deg: list[float]) -> np.ndarray:
    rx, ry, rz = [math.radians(float(v)) for v in rotation_deg]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_mat = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_mat = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_mat = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_mat @ ry_mat @ rx_mat


def intrinsic_matrix(camera_parameters: dict[str, Any], resolution: tuple[int, int]) -> np.ndarray:
    width, height = resolution
    focal_mm = float(camera_parameters["focal_length"])
    sensor_width_mm = float(camera_parameters.get("sensor_width_mm", 36.0))
    sensor_height_mm = float(camera_parameters.get("sensor_height_mm", 24.0))
    pixel_aspect_x = float(camera_parameters.get("pixel_aspect_x", 1.0))
    pixel_aspect_y = float(camera_parameters.get("pixel_aspect_y", 1.0))
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    sensor_fit = str(camera_parameters.get("sensor_fit", "AUTO")).upper()

    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if width >= height * pixel_aspect_ratio else "VERTICAL"

    if sensor_fit == "VERTICAL":
        sensor_size_mm = sensor_height_mm
        view_fac_px = height * pixel_aspect_ratio
    else:
        sensor_size_mm = sensor_width_mm
        view_fac_px = width

    fx = focal_mm / sensor_size_mm * view_fac_px
    fy = fx / pixel_aspect_ratio
    cx = width * 0.5 - float(camera_parameters.get("shift_x", 0.0)) * view_fac_px
    cy = height * 0.5 + float(camera_parameters.get("shift_y", 0.0)) * view_fac_px / pixel_aspect_ratio
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def camera_parameters_from_sidecar(sidecar: dict[str, Any]) -> dict[str, Any]:
    camera = sidecar["camera"]
    intrinsics = camera["intrinsics"]
    return {
        "focal_length": intrinsics["focal_length_mm"],
        "sensor_width_mm": intrinsics.get("sensor_width_mm", 36.0),
        "sensor_height_mm": intrinsics.get("sensor_height_mm", 24.0),
        "pixel_aspect_y": intrinsics.get("pixel_aspect_y", 1.0),
        "position": camera["location_m"],
        "orientation": camera["rotation_deg"],
    }


def projection_from_camera_parameters(
    camera_parameters: dict[str, Any],
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_matrix = intrinsic_matrix(camera_parameters, resolution)
    position = np.asarray(camera_parameters["position"], dtype=float)
    rotation_world_from_camera = euler_xyz_degrees_to_matrix(camera_parameters["orientation"])
    blender_to_cv = np.diag([1.0, -1.0, -1.0])
    rotation_world_to_camera = blender_to_cv @ rotation_world_from_camera.T
    translation_world_to_camera = -rotation_world_to_camera @ position
    projection = k_matrix @ np.column_stack([rotation_world_to_camera, translation_world_to_camera])
    return k_matrix, rotation_world_to_camera, translation_world_to_camera, projection


def build_cameras(
    sidecars: dict[int, dict[str, Any]],
    resolution: tuple[int, int],
) -> tuple[dict[int, dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    cameras = {}
    projections = []
    rotations = []
    translations = []
    for camera_id in sorted(sidecars):
        params = camera_parameters_from_sidecar(sidecars[camera_id])
        k_matrix, rotation, translation, projection = projection_from_camera_parameters(
            params, resolution
        )
        cameras[camera_id] = {
            "parameters": params,
            "resolution": list(resolution),
            "K": k_matrix.tolist(),
            "P": projection.tolist(),
        }
        projections.append(projection)
        rotations.append(rotation)
        translations.append(translation)
    return cameras, np.stack(projections), np.stack(rotations), np.stack(translations)


def project_point(projection: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    homog = projection @ np.append(point, 1.0)
    if abs(float(homog[-1])) < 1e-10:
        return None
    pixel = homog[:2] / homog[-1]
    return pixel if np.all(np.isfinite(pixel)) else None


def triangulate_point_linear(
    projections: np.ndarray,
    observations: list[tuple[int, np.ndarray]],
) -> np.ndarray | None:
    if len(observations) < 2:
        return None
    rows = []
    for camera_matrix_idx, pixel in observations:
        u, v = np.asarray(pixel, dtype=float)
        projection = projections[camera_matrix_idx]
        rows.append(u * projection[2] - projection[0])
        rows.append(v * projection[2] - projection[1])
    design = np.asarray(rows, dtype=float)
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    homog = vh[-1]
    if abs(float(homog[-1])) < 1e-10:
        return None
    point = homog[:3] / homog[-1]
    return point if np.all(np.isfinite(point)) else None


def load_prediction_frames(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, list):
        return data
    if "frames" in data:
        return data["frames"]
    if len(data) == 1:
        value = next(iter(data.values()))
        if isinstance(value, list):
            return value
    raise ValueError(f"Unsupported prediction JSON format: {path}")


def metadata_value_matches(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return actual == expected


def prediction_metadata_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        return False
    return all(metadata_value_matches(metadata.get(key), value) for key, value in expected.items())


def expected_prediction_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if args.bbox_mode == "foreground":
        return {
            "pose2d": args.rtmpose_model,
            "bbox_mode": "foreground",
            "foreground_threshold": args.foreground_threshold,
            "bbox_padding": args.bbox_padding,
            "foreground_min_area": args.foreground_min_area,
            "kpt_thr": args.kpt_thr,
        }
    return {
        "pose2d": args.rtmpose_model,
        "bbox_mode": args.bbox_mode,
        "det_model": "whole_image" if args.bbox_mode == "whole_image" else args.det_model,
        "det_cat_id": args.det_cat_id,
        "bbox_thr": args.bbox_thr,
        "kpt_thr": args.kpt_thr,
    }


def flatten_instances(predictions: Any) -> list[dict[str, Any]]:
    if predictions is None:
        return []
    if isinstance(predictions, dict):
        return [predictions]
    if not isinstance(predictions, list):
        return []
    if predictions and isinstance(predictions[0], list):
        return [item for group in predictions for item in group]
    return [item for item in predictions if isinstance(item, dict)]


def select_primary_instance(instances: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not instances:
        return None

    def instance_score(instance: dict[str, Any]) -> float:
        if "bbox_score" in instance:
            values = np.asarray(instance["bbox_score"], dtype=float).reshape(-1)
            return float(values[0]) if values.size else 0.0
        scores = np.asarray(
            instance.get("keypoint_scores", instance.get("keypoint_score", [])),
            dtype=float,
        ).reshape(-1)
        return float(np.mean(scores)) if scores.size else 0.0

    return max(instances, key=instance_score)


def instance_to_landmarks(instance: dict[str, Any] | None) -> dict[str, list[float]]:
    if instance is None:
        return {}
    keypoints = np.asarray(instance.get("keypoints", []), dtype=float)
    scores = np.asarray(instance.get("keypoint_scores", instance.get("keypoint_score", [])), dtype=float)
    landmarks = {}
    for idx, name in enumerate(COCO17_JOINT_NAMES):
        if idx >= len(keypoints):
            continue
        point = keypoints[idx]
        if len(point) < 2:
            continue
        score = float(scores[idx]) if idx < len(scores) else 1.0
        landmarks[name] = [float(point[0]), float(point[1]), score]
    return landmarks


def pose_sample_to_landmarks(sample: Any | None) -> dict[str, list[float]]:
    if sample is None:
        return {}
    pred_instances = sample.pred_instances
    if hasattr(pred_instances, "cpu"):
        pred_instances = pred_instances.cpu().numpy()
    keypoints = np.asarray(pred_instances.keypoints, dtype=float)
    scores = np.asarray(
        pred_instances.keypoint_scores
        if "keypoint_scores" in pred_instances
        else pred_instances.keypoints_visible,
        dtype=float,
    )
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if scores.ndim == 2:
        scores = scores[0]
    return instance_to_landmarks({"keypoints": keypoints, "keypoint_scores": scores})


def expand_and_clip_bbox(
    bbox: np.ndarray,
    frame_shape: tuple[int, int, int],
    padding_ratio: float,
) -> list[float]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio
    return [
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(width), x2 + pad_x),
        min(float(height), y2 + pad_y),
    ]


def foreground_bbox_from_frame(
    frame: np.ndarray,
    *,
    threshold: int,
    padding_ratio: float,
    min_area: int,
) -> list[float] | None:
    height, width = frame.shape[:2]
    border = max(4, min(20, height // 20, width // 20))
    border_pixels = np.concatenate(
        [
            frame[:border].reshape(-1, 3),
            frame[-border:].reshape(-1, 3),
            frame[:, :border].reshape(-1, 3),
            frame[:, -border:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border_pixels, axis=0)
    diff = np.max(np.abs(frame.astype(np.int16) - background.astype(np.int16)), axis=2)
    mask = (diff > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0 or int(np.max(areas)) < min_area:
        return None

    keep_threshold = max(float(min_area), float(np.max(areas)) * 0.05)
    keep_labels = np.where(areas >= keep_threshold)[0] + 1
    kept_mask = np.isin(labels, keep_labels)
    ys, xs = np.where(kept_mask)
    if len(xs) == 0:
        return None

    bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
    return expand_and_clip_bbox(bbox, frame.shape, padding_ratio)


def run_rtmpose(
    video_path: Path,
    output_path: Path,
    model: str,
    device: str,
    max_frames: int | None,
    *,
    det_model: str | None,
    det_weights: str | None,
    det_cat_id: int,
    bbox_thr: float,
    kpt_thr: float,
    bbox_mode: str,
):
    try:
        from mmpose.apis import MMPoseInferencer
    except Exception as exc:
        raise RuntimeError(
            "RTMPose inference requires MMPose. Install the conda environment from README.md, "
            "or provide existing predictions with --prediction."
        ) from exc

    inferencer_kwargs = {"pose2d": model, "device": device}
    if det_model:
        inferencer_kwargs["det_model"] = det_model
        inferencer_kwargs["det_cat_ids"] = [det_cat_id]
    if det_weights:
        inferencer_kwargs["det_weights"] = det_weights
    inferencer = MMPoseInferencer(**inferencer_kwargs)
    frames = []
    for frame_idx, result in enumerate(
        inferencer(
            str(video_path),
            show=False,
            return_vis=False,
            bbox_thr=bbox_thr,
            kpt_thr=kpt_thr,
        )
    ):
        if max_frames is not None and frame_idx >= max_frames:
            break
        instances = flatten_instances(result.get("predictions"))
        primary = select_primary_instance(instances)
        frames.append({"frame": frame_idx, "landmarks": instance_to_landmarks(primary)})

    payload = {
        "metadata": {
            "estimator": "rtmpose",
            "pose2d": model,
            "det_model": det_model,
            "bbox_mode": bbox_mode,
            "det_cat_id": det_cat_id,
            "bbox_thr": bbox_thr,
            "kpt_thr": kpt_thr,
            "device": device,
            "source": str(video_path),
            "joint_names": COCO17_JOINT_NAMES,
        },
        "frames": frames,
    }
    write_json(output_path, payload)
    return frames


def run_rtmpose_with_foreground_bboxes(
    video_path: Path,
    output_path: Path,
    model: str,
    device: str,
    max_frames: int | None,
    *,
    foreground_threshold: int,
    bbox_padding: float,
    foreground_min_area: int,
    kpt_thr: float,
):
    try:
        from mmpose.apis import MMPoseInferencer, inference_topdown
    except Exception as exc:
        raise RuntimeError(
            "Foreground-bbox RTMPose inference requires MMPose. Install the conda "
            "environment from README.md, or provide existing predictions with --prediction."
        ) from exc

    inferencer = MMPoseInferencer(pose2d=model, det_model="whole_image", device=device)
    pose_model = inferencer.inferencer.model
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        bbox = foreground_bbox_from_frame(
            frame,
            threshold=foreground_threshold,
            padding_ratio=bbox_padding,
            min_area=foreground_min_area,
        )
        bbox_source = "foreground"
        if bbox is None:
            height, width = frame.shape[:2]
            bbox = [0.0, 0.0, float(width), float(height)]
            bbox_source = "whole_image_fallback"

        samples = inference_topdown(
            pose_model,
            frame,
            bboxes=np.asarray([bbox], dtype=np.float32),
            bbox_format="xyxy",
        )
        frames.append(
            {
                "frame": frame_idx,
                "bbox": bbox,
                "bbox_source": bbox_source,
                "landmarks": pose_sample_to_landmarks(samples[0] if samples else None),
            }
        )
        frame_idx += 1

    cap.release()
    payload = {
        "metadata": {
            "estimator": "rtmpose",
            "pose2d": model,
            "det_model": "whole_image",
            "bbox_mode": "foreground",
            "foreground_threshold": foreground_threshold,
            "bbox_padding": bbox_padding,
            "foreground_min_area": foreground_min_area,
            "kpt_thr": kpt_thr,
            "device": device,
            "source": str(video_path),
            "joint_names": COCO17_JOINT_NAMES,
        },
        "frames": frames,
    }
    write_json(output_path, payload)
    return frames


def downsample_trial(mocap_frames: list[dict[str, Any]], target_fps: float) -> list[dict[str, Any]]:
    target_indices = np.arange(0, len(mocap_frames), ORIGINAL_MOCAP_FREQUENCY / float(target_fps))
    downsampled = []
    for out_idx, source_idx in enumerate(target_indices):
        low = int(np.floor(source_idx))
        high = int(np.ceil(source_idx))
        if high >= len(mocap_frames):
            break
        alpha = float(source_idx - low)
        if alpha <= 1e-12:
            landmarks = mocap_frames[low]["landmarks"]
        else:
            landmarks = {}
            low_landmarks = mocap_frames[low]["landmarks"]
            high_landmarks = mocap_frames[high]["landmarks"]
            for name, low_value in low_landmarks.items():
                high_value = high_landmarks.get(name)
                if high_value is None:
                    continue
                low_arr = np.asarray(low_value, dtype=float)
                high_arr = np.asarray(high_value, dtype=float)
                if low_arr.shape != high_arr.shape:
                    continue
                landmarks[name] = ((1.0 - alpha) * low_arr + alpha * high_arr).tolist()
        downsampled.append({"frame": out_idx, "source_frame": float(source_idx), "landmarks": landmarks})
    return downsampled


def load_mocap_gt(blender_root: Path, trial: str, target_fps: float | None, max_frames: int | None):
    mocap_path = blender_root / "data" / "mocap" / f"{trial}.json"
    data = read_json(mocap_path)
    if trial not in data:
        raise KeyError(f"Trial {trial} not found in {mocap_path}")
    frames = data[trial]
    if target_fps is not None:
        frames = downsample_trial(frames, target_fps)
    if max_frames is not None:
        frames = frames[:max_frames]

    gt_frames = []
    for idx, frame in enumerate(frames):
        raw = frame.get("landmarks", {})
        landmarks = {}
        for coco_name, mocap_name in COCO17_TO_MOCAP.items():
            value = raw.get(mocap_name)
            if value is None:
                continue
            arr = np.asarray(value[:3], dtype=float)
            if np.all(np.isfinite(arr)):
                landmarks[coco_name] = [float(arr[0]), float(arr[1]), float(arr[2])]
        gt_frames.append({"frame": idx, "landmarks": landmarks})
    return gt_frames, list(COCO17_TO_MOCAP.keys())


def triangulate_sequence(
    predictions_by_camera: dict[int, list[dict[str, Any]]],
    projections: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    joint_names: list[str],
    min_views: int,
    score_threshold: float,
    require_cheirality: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    camera_ids = sorted(predictions_by_camera)
    frame_count = min(len(predictions_by_camera[cid]) for cid in camera_ids)
    total_points = frame_count * len(joint_names)
    stats = {
        "frames": frame_count,
        "camera_count": len(camera_ids),
        "joint_count": len(joint_names),
        "total_points": total_points,
        "total_2d_observations": total_points * len(camera_ids),
        "available_2d_observations": 0,
        "thresholded_2d_observations": 0,
        "thresholded_points": 0,
        "dlt_attempted_points": 0,
        "attempted_points": 0,
        "triangulated_points": 0,
        "skipped_low_view_points": 0,
        "skipped_dlt_fail_points": 0,
        "skipped_cheirality_points": 0,
    }
    output_frames = []
    for frame_idx in range(frame_count):
        landmarks_3d = {}
        metadata = {}
        for joint_name in joint_names:
            observations = []
            scores = []
            for matrix_idx, camera_id in enumerate(camera_ids):
                landmarks_2d = predictions_by_camera[camera_id][frame_idx].get("landmarks", {})
                value = landmarks_2d.get(joint_name)
                if value is None or len(value) < 2:
                    continue
                stats["available_2d_observations"] += 1
                score = float(value[2]) if len(value) >= 3 and value[2] is not None else 1.0
                if score < score_threshold:
                    continue
                stats["thresholded_2d_observations"] += 1
                observations.append((matrix_idx, np.asarray(value[:2], dtype=float)))
                scores.append(score)

            if not observations:
                continue
            stats["thresholded_points"] += 1
            stats["attempted_points"] += 1
            if len(observations) < min_views:
                stats["skipped_low_view_points"] += 1
                continue

            stats["dlt_attempted_points"] += 1
            point = triangulate_point_linear(projections, observations)
            if point is None:
                stats["skipped_dlt_fail_points"] += 1
                continue

            if require_cheirality:
                depths = []
                for matrix_idx, _pixel in observations:
                    point_camera = rotations[matrix_idx] @ point + translations[matrix_idx]
                    depths.append(float(point_camera[2]))
                if not np.all(np.asarray(depths) > 1e-6):
                    stats["skipped_cheirality_points"] += 1
                    continue

            stats["triangulated_points"] += 1
            landmarks_3d[joint_name] = [float(point[0]), float(point[1]), float(point[2])]
            metadata[joint_name] = {
                "views": len(observations),
                "mean_score": float(np.mean(scores)) if scores else math.nan,
            }
        output_frames.append({"frame": frame_idx, "landmarks": landmarks_3d, "metadata": metadata})
    return output_frames, stats


def compute_mpjpe(predicted_frames, gt_frames, joint_names: list[str]) -> dict[str, Any]:
    errors = []
    per_joint = defaultdict(list)
    per_frame = []
    frame_count = min(len(predicted_frames), len(gt_frames))
    for frame_idx in range(frame_count):
        frame_errors = []
        pred_landmarks = predicted_frames[frame_idx].get("landmarks", {})
        gt_landmarks = gt_frames[frame_idx].get("landmarks", {})
        for joint_name in joint_names:
            pred = pred_landmarks.get(joint_name)
            gt = gt_landmarks.get(joint_name)
            if pred is None or gt is None:
                continue
            pred_arr = np.asarray(pred[:3], dtype=float)
            gt_arr = np.asarray(gt[:3], dtype=float)
            if not np.all(np.isfinite(pred_arr)) or not np.all(np.isfinite(gt_arr)):
                continue
            err_mm = float(np.linalg.norm(pred_arr - gt_arr) * 1000.0)
            errors.append(err_mm)
            per_joint[joint_name].append(err_mm)
            frame_errors.append(err_mm)
        per_frame.append(
            {
                "frame": frame_idx,
                "joint_count": len(frame_errors),
                "mpjpe_mm": float(np.mean(frame_errors)) if frame_errors else math.nan,
            }
        )

    arr = np.asarray(errors, dtype=float)
    return {
        "frames_compared": frame_count,
        "joint_observations": int(len(arr)),
        "mpjpe_mm": float(np.mean(arr)) if len(arr) else math.nan,
        "median_mm": float(np.median(arr)) if len(arr) else math.nan,
        "p90_mm": float(np.percentile(arr, 90)) if len(arr) else math.nan,
        "max_mm": float(np.max(arr)) if len(arr) else math.nan,
        "per_joint": {
            name: {
                "count": len(values),
                "mpjpe_mm": float(np.mean(values)) if values else math.nan,
                "median_mm": float(np.median(values)) if values else math.nan,
            }
            for name, values in per_joint.items()
        },
        "per_frame": per_frame,
    }


def make_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    mpjpe = payload["mpjpe"]
    triangulation = payload["triangulation"]
    dlt_attempted_points = triangulation.get(
        "dlt_attempted_points",
        triangulation["triangulated_points"]
        + triangulation["skipped_dlt_fail_points"]
        + triangulation["skipped_cheirality_points"],
    )
    return {
        "trial": payload["trial"],
        "camera_ids": payload["camera_ids"],
        "score_threshold": payload.get("score_threshold"),
        "min_views": payload.get("min_views"),
        "mean_mpjpe_mm": mpjpe["mpjpe_mm"],
        "median_mpjpe_mm": mpjpe["median_mm"],
        "p90_mpjpe_mm": mpjpe["p90_mm"],
        "max_mpjpe_mm": mpjpe["max_mm"],
        "frames_compared": mpjpe["frames_compared"],
        "joint_observations": mpjpe["joint_observations"],
        "total_points": triangulation.get("total_points", triangulation["attempted_points"]),
        "total_2d_observations": triangulation.get("total_2d_observations"),
        "available_2d_observations": triangulation.get("available_2d_observations"),
        "thresholded_2d_observations": triangulation.get("thresholded_2d_observations"),
        "thresholded_points": triangulation.get(
            "thresholded_points", triangulation["attempted_points"]
        ),
        "dlt_attempted_points": dlt_attempted_points,
        "triangulated_points": triangulation["triangulated_points"],
        "attempted_points": triangulation["attempted_points"],
        "skipped_low_view_points": triangulation["skipped_low_view_points"],
        "skipped_dlt_fail_points": triangulation["skipped_dlt_fail_points"],
        "skipped_cheirality_points": triangulation["skipped_cheirality_points"],
    }


def write_summary_files(output_dir: Path, payload: dict[str, Any]) -> None:
    summary = make_summary_payload(payload)
    write_json(output_dir / "summary.json", summary)
    lines = [
        f"Trial: {summary['trial']}",
        f"Cameras: {summary['camera_ids']}",
        f"Mean MPJPE: {summary['mean_mpjpe_mm']:.2f} mm",
        f"Median MPJPE: {summary['median_mpjpe_mm']:.2f} mm",
        f"P90 MPJPE: {summary['p90_mpjpe_mm']:.2f} mm",
        f"Max MPJPE: {summary['max_mpjpe_mm']:.2f} mm",
        f"Frames compared: {summary['frames_compared']}",
        f"Joint observations: {summary['joint_observations']}",
        f"Score threshold: {summary['score_threshold']}",
        (
            "2D observations above threshold: "
            f"{summary['thresholded_2d_observations']} / "
            f"{summary['available_2d_observations']} "
            f"({summary['total_2d_observations']} possible)"
        ),
        (
            "Point counts (total / above threshold / DLT attempted / triangulated): "
            f"{summary['total_points']} / {summary['thresholded_points']} / "
            f"{summary['dlt_attempted_points']} / {summary['triangulated_points']}"
        ),
        (
            "Triangulated points: "
            f"{summary['triangulated_points']} / {summary['thresholded_points']}"
        ),
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_skeleton(
    frame: np.ndarray,
    landmarks: dict[str, list[float]],
    *,
    color: tuple[int, int, int],
    label_prefix: str,
    score_threshold: float = 0.0,
) -> None:
    points = {}
    for name, value in landmarks.items():
        if value is None or len(value) < 2:
            continue
        score = float(value[2]) if len(value) >= 3 and value[2] is not None else 1.0
        if score < score_threshold:
            continue
        x, y = int(round(float(value[0]))), int(round(float(value[1])))
        points[name] = (x, y)

    for start, end in COCO17_EDGES:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], color, 3, lineType=cv2.LINE_AA)
    for x, y in points.values():
        cv2.circle(frame, (x, y), 5, color, -1, lineType=cv2.LINE_AA)

    cv2.putText(
        frame,
        label_prefix,
        (16, 36 if color == PRED_COLOR else 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def projected_gt_for_camera(
    gt_frames: list[dict[str, Any]],
    projection: np.ndarray,
    joint_names: list[str],
) -> list[dict[str, Any]]:
    projected = []
    for frame_idx, frame in enumerate(gt_frames):
        landmarks = {}
        for joint_name in joint_names:
            value = frame.get("landmarks", {}).get(joint_name)
            if value is None:
                continue
            uv = project_point(projection, np.asarray(value[:3], dtype=float))
            if uv is not None:
                landmarks[joint_name] = [float(uv[0]), float(uv[1]), 1.0]
        projected.append({"frame": frame_idx, "landmarks": landmarks})
    return projected


def create_overlay_video(
    video_path: Path,
    prediction_frames: list[dict[str, Any]],
    gt_projected_frames: list[dict[str, Any]] | None,
    output_path: Path,
    *,
    score_threshold: float,
    max_frames: int | None = None,
) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open overlay source video: {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        print(f"Could not open overlay writer: {output_path}")
        return False

    frame_idx = 0
    pred_len = len(prediction_frames)
    gt_len = len(gt_projected_frames) if gt_projected_frames is not None else 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        if frame_idx < pred_len:
            draw_skeleton(
                frame,
                prediction_frames[frame_idx].get("landmarks", {}),
                color=PRED_COLOR,
                label_prefix="RTMPose",
                score_threshold=score_threshold,
            )
        if gt_projected_frames is not None and frame_idx < gt_len:
            draw_skeleton(
                frame,
                gt_projected_frames[frame_idx].get("landmarks", {}),
                color=GT_COLOR,
                label_prefix="GT projection",
            )

        cv2.putText(
            frame,
            f"frame {frame_idx}",
            (16, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return frame_idx > 0


def create_overlay_videos(
    video_paths: dict[int, Path],
    predictions_by_camera: dict[int, list[dict[str, Any]]],
    gt_frames: list[dict[str, Any]],
    projections: np.ndarray,
    joint_names: list[str],
    output_dir: Path,
    *,
    score_threshold: float,
    max_frames: int | None,
    include_gt: bool,
) -> dict[str, str]:
    overlay_dir = output_dir / "overlays"
    outputs = {}
    for matrix_idx, camera_id in enumerate(sorted(video_paths)):
        gt_projected = (
            projected_gt_for_camera(gt_frames, projections[matrix_idx], joint_names)
            if include_gt
            else None
        )
        overlay_path = overlay_dir / f"camera{camera_id}_skeleton_overlay.mp4"
        ok = create_overlay_video(
            video_paths[camera_id],
            predictions_by_camera[camera_id],
            gt_projected,
            overlay_path,
            score_threshold=score_threshold,
            max_frames=max_frames,
        )
        if ok:
            outputs[str(camera_id)] = str(overlay_path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default=DEFAULT_TRIAL)
    parser.add_argument("--blender-root", type=Path, default=DEFAULT_BLENDER_ROOT)
    parser.add_argument("--video", nargs=2, action="append", metavar=("CAMERA_ID", "PATH"))
    parser.add_argument("--prediction", nargs=2, action="append", metavar=("CAMERA_ID", "PATH"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "val_3cam")
    parser.add_argument(
        "--rtmpose-model",
        default="human",
        help="MMPose pose2d alias/config/path. Official RTMPose human alias: 'human'.",
    )
    parser.add_argument(
        "--det-model",
        default=None,
        help="Optional MMPose/MMDetection detector alias/config/path used by --bbox-mode detector.",
    )
    parser.add_argument("--det-weights", default=None)
    parser.add_argument("--det-cat-id", type=int, default=0, help="COCO person category id.")
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument(
        "--bbox-mode",
        choices=("detector", "whole_image", "foreground"),
        default="foreground",
        help=(
            "How to provide the top-down person bbox: official detector, whole image, "
            "or a tight foreground bbox estimated from the rendered frame."
        ),
    )
    parser.add_argument("--foreground-threshold", type=int, default=20)
    parser.add_argument("--bbox-padding", type=float, default=0.15)
    parser.add_argument("--foreground-min-area", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument(
        "--overlay-score-threshold",
        type=float,
        default=None,
        help="Keypoint score threshold used only for overlay videos. Defaults to --score-threshold.",
    )
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--mocap-fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--no-cheirality", action="store_true")
    parser.add_argument("--no-overlay", action="store_true", help="Skip skeleton overlay videos.")
    parser.add_argument(
        "--no-overlay-gt",
        action="store_true",
        help="Only draw RTMPose predictions, not projected 3D ground truth.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    blender_root = args.blender_root.expanduser().resolve()
    video_paths = parse_camera_path(args.video) if args.video else default_video_paths(blender_root)
    prediction_paths = parse_camera_path(args.prediction)
    camera_ids = sorted(set(video_paths) | set(prediction_paths))
    if len(camera_ids) < 2:
        raise SystemExit("At least two cameras are required.")

    args.output.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    predictions_by_camera = {}
    expected_metadata = expected_prediction_metadata(args)
    for camera_id in camera_ids:
        pred_path = prediction_paths.get(camera_id, prediction_dir / f"camera{camera_id}_rtmpose.json")
        cache_matches = camera_id in prediction_paths or prediction_metadata_matches(
            pred_path, expected_metadata
        )
        if camera_id in video_paths and (
            args.force_inference or not pred_path.exists() or not cache_matches
        ):
            if pred_path.exists() and not args.force_inference and not cache_matches:
                print(f"Cached prediction metadata differs from requested mode; rerunning: {pred_path}")
            print(f"Running RTMPose camera {camera_id}: {video_paths[camera_id]}")
            if args.bbox_mode == "foreground":
                run_rtmpose_with_foreground_bboxes(
                    video_paths[camera_id],
                    pred_path,
                    args.rtmpose_model,
                    args.device,
                    args.max_frames,
                    foreground_threshold=args.foreground_threshold,
                    bbox_padding=args.bbox_padding,
                    foreground_min_area=args.foreground_min_area,
                    kpt_thr=args.kpt_thr,
                )
            else:
                det_model = "whole_image" if args.bbox_mode == "whole_image" else args.det_model
                run_rtmpose(
                    video_paths[camera_id],
                    pred_path,
                    args.rtmpose_model,
                    args.device,
                    args.max_frames,
                    det_model=det_model,
                    det_weights=args.det_weights,
                    det_cat_id=args.det_cat_id,
                    bbox_thr=args.bbox_thr,
                    kpt_thr=args.kpt_thr,
                    bbox_mode=args.bbox_mode,
                )
        if not pred_path.exists():
            raise SystemExit(f"Missing prediction for camera {camera_id}: {pred_path}")
        frames = load_prediction_frames(pred_path)
        if args.max_frames is not None:
            frames = frames[: args.max_frames]
        predictions_by_camera[camera_id] = frames

    video_resolution, video_fps = read_video_metadata(video_paths)
    sidecars = load_render_sidecars(video_paths)
    resolution = sidecar_resolution(sidecars) or video_resolution
    camera_payload, projections, rotations, translations = build_cameras(sidecars, resolution)

    mocap_fps = args.mocap_fps if args.mocap_fps is not None else sidecar_fps(sidecars) or video_fps
    gt_frames, joint_names = load_mocap_gt(blender_root, args.trial, mocap_fps, args.max_frames)

    triangulated_frames, triangulation_stats = triangulate_sequence(
        predictions_by_camera,
        projections,
        rotations,
        translations,
        joint_names,
        min_views=args.min_views,
        score_threshold=args.score_threshold,
        require_cheirality=not args.no_cheirality,
    )
    metrics = compute_mpjpe(triangulated_frames, gt_frames, joint_names)

    write_json(args.output / "cameras.json", camera_payload)
    write_json(
        args.output / "triangulated_3d.json",
        {
            args.trial: triangulated_frames,
            "metadata": {
                "trial": args.trial,
                "camera_ids": camera_ids,
                "joint_names": joint_names,
                "mocap_fps": mocap_fps,
                "score_threshold": args.score_threshold,
                "min_views": args.min_views,
            },
        },
    )
    payload = {
        "trial": args.trial,
        "camera_ids": camera_ids,
        "videos": {str(k): str(v) for k, v in video_paths.items()},
        "score_threshold": args.score_threshold,
        "min_views": args.min_views,
        "triangulation": triangulation_stats,
        "mpjpe": metrics,
    }
    if not args.no_overlay:
        overlay_score_threshold = (
            args.overlay_score_threshold
            if args.overlay_score_threshold is not None
            else args.score_threshold
        )
        payload["overlays"] = create_overlay_videos(
            video_paths,
            predictions_by_camera,
            gt_frames,
            projections,
            joint_names,
            args.output,
            score_threshold=overlay_score_threshold,
            max_frames=args.max_frames,
            include_gt=not args.no_overlay_gt,
        )
    write_json(args.output / "metrics.json", payload)
    write_summary_files(args.output, payload)
    print(json.dumps(make_summary_payload(payload), indent=2, allow_nan=True))
    print(f"Wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
