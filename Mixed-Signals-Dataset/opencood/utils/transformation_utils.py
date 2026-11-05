# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Transformation utils
"""

import numpy as np
from typing import Union, List


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix


def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous

def x1_to_x2_np(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : np.ndarray
        (4, 4) The pose of x1 under world coordinates.
    x2 : np.ndarray
        (4, 4) The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.
    """
    assert isinstance(x1, np.ndarray), f"{type(x1)} != np.ndarray"
    assert isinstance(x2, np.ndarray), f"{type(x2)} != np.ndarray"
    assert x1.shape == (4, 4), f"{x1.shape} != (4, 4)"
    assert x2.shape == (4, 4), f"{x2.shape} != (4, 4)"
    x2_se3_x1 = np.linalg.inv(x2) @ x1
    return x2_se3_x1


def make_se3(translation: Union[List[float], np.ndarray], yaw: float = None, rotation_matrix: np.ndarray = None, 
             quaternion: Union[List[float], np.ndarray] = None):
    if yaw is not None:
        assert quaternion is None
        assert rotation_matrix is None
        rotation_matrix = make_rotation_around_z(yaw)
    elif quaternion is not None:
        assert yaw is None
        assert rotation_matrix is None
        assert isinstance(quaternion, list) or isinstance(quaternion, np.ndarray), f"{type(quaternion)} is not supported"
        rotation_matrix = Quaternion(quaternion).rotation_matrix
    else:
        assert rotation_matrix is not None
        
    
    out = np.zeros((4, 4))
    out[-1, -1] = 1.0

    out[:3, :3] = rotation_matrix

    if not isinstance(translation, np.ndarray):
        translation = np.array(translation)
    out[:3, -1] = translation.reshape(3)

    return out


def rotation_matrix_to_yaw(rot: np.ndarray) -> float:
    return np.arctan2(rot[1, 0], rot[0, 0])


def make_rotation_around_z(yaw: float) -> np.ndarray:
    cos, sin = np.cos(yaw), np.sin(yaw)
    out = np.array([
        [cos, -sin, 0.],
        [sin, cos, 0.],
        [0., 0., 1.]
    ])
    return out


def apply_se3_(se3_tf: np.ndarray, 
               points_: np.ndarray = None, 
               boxes_: np.ndarray = None, boxes_has_velocity: bool = False, 
               vector_: np.ndarray = None) -> None:
    """
    Inplace function

    Args:
        se3_tf: (4, 4) - homogeneous transformation
        points_: (N, 3 + C) - x, y, z, [others]
        boxes_: (N, 7 + 2 + C) - x, y, z, dx, dy, dz, yaw, [vx, vy], [others]
        boxes_has_velocity: make boxes_velocity explicit
        vector_: (N, 2 [+ 1]) - x, y, [z]
    """
    if points_ is not None:
        assert points_.shape[1] >= 3, f"points_.shape: {points_.shape}"
        points_[:, :3] = points_[:, :3] @  se3_tf[:3, :3].T + se3_tf[:3, -1]

    if boxes_ is not None:
        assert boxes_.shape[1] >= 7, f"boxes_.shape: {boxes_.shape}"
        boxes_[:, :3] = boxes_[:, :3] @  se3_tf[:3, :3].T + se3_tf[:3, -1]
        boxes_[:, 6] += rotation_matrix_to_yaw(se3_tf[:3, :3])
        boxes_[:, 6] = np.arctan2(np.sin(boxes_[:, 6]), np.cos(boxes_[:, 6]))
        if boxes_has_velocity:
            boxes_velo = np.pad(boxes_[:, 7: 9], pad_width=[(0, 0), (0, 1)], constant_values=0.0)  # (N, 3) - vx, vy, vz
            boxes_velo = boxes_velo @ se3_tf[:3, :3].T
            boxes_[:, 7: 9] = boxes_velo[:, :2]

    if vector_ is not None:
        if vector_.shape[1] == 2:
            vector = np.pad(vector_, pad_width=[(0, 0), (0, 1)], constant_values=0.)
            vector_[:, :2] = (vector @ se3_tf[:3, :3].T)[:, :2]
        else:
            assert vector_.shape[1] == 3, f"vector_.shape: {vector_.shape}"
            vector_[:, :3] = vector_ @ se3_tf[:3, :3].T

    return