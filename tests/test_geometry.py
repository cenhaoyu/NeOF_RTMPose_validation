import math

import numpy as np

from src.validate_rtmpose_triangulation import (
    foreground_bbox_from_frame,
    intrinsic_matrix,
    project_point,
    projection_from_camera_parameters,
    triangulate_point_linear,
)


def test_intrinsic_matrix_uses_pixel_aspect_y():
    params = {
        "focal_length": 26.488948,
        "sensor_width_mm": 36.0,
        "sensor_height_mm": 24.0,
        "pixel_aspect_y": 1.185185,
    }

    k_matrix = intrinsic_matrix(params, (1920, 1080))

    fx = 26.488948 * 1920.0 / 36.0
    assert math.isclose(k_matrix[0, 0], fx)
    assert math.isclose(k_matrix[1, 1], fx / 1.185185)


def test_projection_matches_blender_negative_z_forward():
    params = {
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0],
        "focal_length": 50.0,
        "sensor_width_mm": 50.0,
        "sensor_height_mm": 50.0,
        "pixel_aspect_y": 1.0,
    }

    _, _, _, projection = projection_from_camera_parameters(params, (1000, 1000))

    assert np.allclose(project_point(projection, np.array([0.0, 0.0, -5.0])), [500.0, 500.0])
    assert np.allclose(project_point(projection, np.array([1.0, 0.0, -5.0])), [700.0, 500.0])
    assert np.allclose(project_point(projection, np.array([0.0, 1.0, -5.0])), [500.0, 300.0])


def test_dlt_reconstructs_two_view_point():
    k_matrix = np.array([[800.0, 0.0, 500.0], [0.0, 800.0, 500.0], [0.0, 0.0, 1.0]])
    p0 = k_matrix @ np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    p1 = k_matrix @ np.array(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    point = np.array([0.25, -0.1, 4.0])
    observations = [(0, project_point(p0, point)), (1, project_point(p1, point))]

    reconstructed = triangulate_point_linear(np.stack([p0, p1]), observations)

    assert np.allclose(reconstructed, point)


def test_foreground_bbox_uses_border_background_and_padding():
    frame = np.full((100, 120, 3), 206, dtype=np.uint8)
    frame[30:70, 40:80] = [80, 90, 100]

    bbox = foreground_bbox_from_frame(
        frame,
        threshold=20,
        padding_ratio=0.1,
        min_area=20,
    )

    assert bbox == [36.0, 26.0, 84.0, 74.0]
