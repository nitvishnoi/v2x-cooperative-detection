import numpy as np


def remove_points_outside_range(points: np.ndarray, min_max_xyz: np.ndarray) -> np.ndarray:
    assert min_max_xyz.shape == (6,), f"{min_max_xyz.shape} != (6,)"
    mask_in_range = np.logical_and(points[:, :3] >= min_max_xyz[:3], 
                                   points[:, :3] < min_max_xyz[3:]).all(axis=1)
    assert np.any(mask_in_range), "expect at least 1 point in range"
    return points[mask_in_range]
