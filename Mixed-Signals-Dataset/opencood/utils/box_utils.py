# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Bounding box related utility functions
"""
import sys
from typing import List, Union

import numpy as np

import torch
import torch.nn.functional as F
import opencood.utils.common_utils as common_utils
from opencood.utils.transformation_utils import x1_to_x2, make_se3
from opencood.pcdet_utils.iou3d_nms import iou3d_nms_cuda


def corner_to_center(corner3d, order='lwh'):
    """
    Convert 8 corners to x, y, z, dx, dy, dz, yaw.

    Parameters
    ----------
    corner3d : np.ndarray
        (N, 8, 3)

    order : str
        'lwh' or 'hwl'

    Returns
    -------
    box3d : np.ndarray
        (N, 7)
    """
    assert corner3d.ndim == 3
    batch_size = corner3d.shape[0]

    xyz = np.mean(corner3d[:, [0, 3, 5, 6], :], axis=1)
    h = abs(np.mean(corner3d[:, 4:, 2] - corner3d[:, :4, 2], axis=1,
                    keepdims=True))
    l = (np.sqrt(np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 5, [0, 1]] - corner3d[:, 6, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    w = (np.sqrt(
        np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2, axis=1,
               keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 5, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 6, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    theta = (np.arctan2(corner3d[:, 1, 1] - corner3d[:, 2, 1],
                        corner3d[:, 1, 0] - corner3d[:, 2, 0]) +
             np.arctan2(corner3d[:, 0, 1] - corner3d[:, 3, 1],
                        corner3d[:, 0, 0] - corner3d[:, 3, 0]) +
             np.arctan2(corner3d[:, 5, 1] - corner3d[:, 6, 1],
                        corner3d[:, 5, 0] - corner3d[:, 6, 0]) +
             np.arctan2(corner3d[:, 4, 1] - corner3d[:, 7, 1],
                        corner3d[:, 4, 0] - corner3d[:, 7, 0]))[:,
            np.newaxis] / 4

    if order == 'lwh':
        return np.concatenate([xyz, l, w, h, theta], axis=1).reshape(
            batch_size, 7)
    elif order == 'hwl':
        return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(
            batch_size, 7)
    else:
        sys.exit('Unknown order')


def boxes_to_corners2d(boxes3d, order):
    """
      0 -------- 1
      |          |
      |          |
      |          |
      3 -------- 2
    Parameters
    __________
    boxes3d: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners2d: np.ndarray or torch.Tensor
        (N, 4, 3), the 4 corners of the bounding box.

    """
    corners3d = boxes_to_corners_3d(boxes3d, order)
    corners2d = corners3d[:, :4, :]
    return corners2d


def boxes2d_to_corners2d(boxes2d, order="lwh"):
    """
      0 -------- 1
      |          |
      |          |
      |          |
      3 -------- 2
    Parameters
    __________
    boxes2d: np.ndarray or torch.Tensor
        (..., 5) [x, y, dx, dy, heading], (x, y) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners2d: np.ndarray or torch.Tensor
        (..., 4, 2), the 4 corners of the bounding box.

    """
    assert order == "lwh", \
        "boxes2d_to_corners_2d only supports lwh order for now."
    boxes2d, is_numpy = common_utils.check_numpy_to_torch(boxes2d)
    template = boxes2d.new_tensor((
        [1, -1], [1, 1], [-1, 1], [-1, -1]
    )) / 2
    input_shape = boxes2d.shape
    boxes2d = boxes2d.view(-1, 5)
    corners2d = boxes2d[:, None, 2:4].repeat(1, 4, 1) * template[None, :, :]
    corners2d = common_utils.rotate_points_along_z_2d(corners2d.view(-1, 2),
                                                      boxes2d[:,
                                                      4].repeat_interleave(
                                                          4)).view(-1, 4,
                                                                   2)
    corners2d += boxes2d[:, None, 0:2]
    corners2d = corners2d.view(*(input_shape[:-1]), 4, 2)
    return corners2d


def boxes_to_corners_3d(boxes3d, order):
    """
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    Parameters
    __________
    boxes3d: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners3d: np.ndarray or torch.Tensor
        (N, 8, 3), the 8 corners of the bounding box.

    """
    # ^ z
    # |
    # |
    # | . x
    # |/
    # +-------> y

    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    boxes3d_ = boxes3d

    if order == 'hwl':
        boxes3d_ = boxes3d[:, [0, 1, 2, 5, 4, 3, 6]]

    template = boxes3d_.new_tensor((
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    )) / 2

    corners3d = boxes3d_[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3),
                                                   boxes3d_[:, 6]).view(-1, 8,
                                                                        3)
    corners3d += boxes3d_[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d



def box3d_to_2d(box3d):
    """
    Convert 3D bounding box to 2D.

    Parameters
    ----------
    box3d : np.ndarray
        (n, 8, 3)

    Returns
    -------
    box2d : np.ndarray
        (n, 4, 2), project 3d to 2d.
    """
    box2d = box3d[:, :4, :2]
    return box2d


def corner2d_to_standup_box(box2d):
    """
    Find the minmaxx, minmaxy for each 2d box. (N, 4, 2) -> (N, 4)
    x1, y1, x2, y2

    Parameters
    ----------
    box2d : np.ndarray
        (n, 4, 2), four corners of the 2d bounding box.

    Returns
    -------
    standup_box2d : np.ndarray
        (n, 4)
    """
    N = box2d.shape[0]
    standup_boxes2d = np.zeros((N, 4))

    standup_boxes2d[:, 0] = np.min(box2d[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(box2d[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(box2d[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(box2d[:, :, 1], axis=1)

    return standup_boxes2d


def corner_to_standup_box_torch(box_corner):
    """
    Find the minmax x and y for each bounding box.

    Parameters
    ----------
    box_corner : torch.Tensor
        Shape: (N, 8, 3) or (N, 4)

    Returns
    -------
    standup_box2d : torch.Tensor
        (n, 4)
    """
    N = box_corner.shape[0]
    standup_boxes2d = torch.zeros((N, 4))

    standup_boxes2d = standup_boxes2d.to(box_corner.device)

    standup_boxes2d[:, 0] = torch.min(box_corner[:, :, 0], dim=1).values
    standup_boxes2d[:, 1] = torch.min(box_corner[:, :, 1], dim=1).values
    standup_boxes2d[:, 2] = torch.max(box_corner[:, :, 0], dim=1).values
    standup_boxes2d[:, 3] = torch.max(box_corner[:, :, 1], dim=1).values

    return standup_boxes2d


def project_box3d(box3d, transformation_matrix):
    """
    Project the 3d bounding box to another coordinate system based on the
    transfomration matrix.

    Parameters
    ----------
    box3d : torch.Tensor or np.ndarray
        3D bounding box, (N, 8, 3)

    transformation_matrix : torch.Tensor or np.ndarray
        Transformation matrix, (4, 4)

    Returns
    -------
    projected_box3d : torch.Tensor
        The projected bounding box, (N, 8, 3)
    """
    assert transformation_matrix.shape == (4, 4)
    box3d, is_numpy = \
        common_utils.check_numpy_to_torch(box3d)
    transformation_matrix, _ = \
        common_utils.check_numpy_to_torch(transformation_matrix)

    # (N, 3, 8)
    box3d_corner = box3d.transpose(1, 2)
    # (N, 1, 8)
    torch_ones = torch.ones((box3d_corner.shape[0], 1, 8))
    torch_ones = torch_ones.to(box3d_corner.device)
    # (N, 4, 8)
    box3d_corner = torch.cat((box3d_corner, torch_ones),
                             dim=1)
    # (N, 4, 8)
    projected_box3d = torch.matmul(transformation_matrix,
                                   box3d_corner)
    # (N, 8, 3)
    projected_box3d = projected_box3d[:, :3, :].transpose(1, 2)

    return projected_box3d if not is_numpy else projected_box3d.numpy()


def project_points_by_matrix_torch(points, transformation_matrix):
    """
    Project the points to another coordinate system based on the
    transfomration matrix.

    Parameters
    ----------
    points : torch.Tensor
        3D points, (N, 3)

    transformation_matrix : torch.Tensor
        Transformation matrix, (4, 4)

    Returns
    -------
    projected_points : torch.Tensor
        The projected points, (N, 3)
    """
    # convert to homogeneous  coordinates via padding 1 at the last dimension.
    # (N, 4)
    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1)
    # (N, 4)
    projected_points = torch.einsum("ik, jk->ij", points_homogeneous,
                                    transformation_matrix)
    return projected_points[:, :3]


def get_mask_for_boxes_within_range_torch(boxes):
    """
    Generate mask to remove the bounding boxes
    outside the range.

    Parameters
    ----------
    boxes : torch.Tensor
        Groundtruth bbx, shape: N,8,3 or N,4,2
    Returns
    -------
    mask: torch.Tensor
        The mask for bounding box -- True means the
        bbx is within the range and False means the
        bbx is outside the range.
    """
    from opencood.data_utils.datasets import GT_RANGE

    # mask out the gt bounding box out fixed range (-140, -40, -3, 140, 40 1)
    device = boxes.device
    boundary_lower_range = \
        torch.Tensor(GT_RANGE[:2]).reshape(1, 1, -1).to(device)
    boundary_higher_range = \
        torch.Tensor(GT_RANGE[3:5]).reshape(1, 1, -1).to(device)

    mask = torch.all(
        torch.all(boxes[:, :, :2] >= boundary_lower_range,
                  dim=-1) & \
        torch.all(boxes[:, :, :2] <= boundary_higher_range,
                  dim=-1), dim=-1)

    return mask


def mask_boxes_outside_range_numpy(boxes, limit_range, order,
                                   min_num_corners=8, return_mask=False):
    """
    Parameters
    ----------
    boxes: np.ndarray
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    limit_range: list
        [minx, miny, minz, maxx, maxy, maxz]

    min_num_corners: int
        The required minimum number of corners to be considered as in range.

    order : str
        'lwh' or 'hwl'

    return_mask : bool
        Whether return the mask.

    Returns
    -------
    boxes: np.ndarray
        The filtered boxes.
    """
    assert boxes.shape[1] == 8 or boxes.shape[1] == 7

    new_boxes = boxes.copy()
    if boxes.shape[1] == 7:
        new_boxes = boxes_to_corners_3d(new_boxes, order)

    mask = ((new_boxes[:, :, :2] >= limit_range[0:2]) &
            (new_boxes[:, :, :2] <= limit_range[3:5])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    if return_mask:
        return boxes[mask], mask
    return boxes[mask]


def create_bbx(extent):
    """
    Create bounding box with 8 corners under obstacle vehicle reference.

    Parameters
    ----------
    extent : list
        Width, height, length of the bbx.

    Returns
    -------
    bbx : np.array
        The bounding box with 8 corners, shape: (8, 3)
    """

    bbx = np.array([[extent[0], -extent[1], -extent[2]],
                    [extent[0], extent[1], -extent[2]],
                    [-extent[0], extent[1], -extent[2]],
                    [-extent[0], -extent[1], -extent[2]],
                    [extent[0], -extent[1], extent[2]],
                    [extent[0], extent[1], extent[2]],
                    [-extent[0], extent[1], extent[2]],
                    [-extent[0], -extent[1], extent[2]]])

    return bbx


def project_world_objects(object_dict,
                          output_dict,
                          lidar_pose,
                          lidar_range,
                          order):
    """
    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_dict : dict
        The dictionary contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].

    lidar_range : list
         [minx, miny, minz, maxx, maxy, maxz]

    order : str
        'lwh' or 'hwl'
    """
    for object_id, object_content in object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']

        object_pose = [location[0] + center[0],
                       location[1] + center[1],
                       location[2] + center[2],
                       rotation[0], rotation[1], rotation[2]]
        object2lidar = x1_to_x2(object_pose, lidar_pose)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to world coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        bbx_lidar = mask_boxes_outside_range_numpy(bbx_lidar,
                                                   lidar_range,
                                                   order)

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: bbx_lidar})


def get_points_in_rotated_box(p, box_corner):
    """
    Get points within a rotated bounding box (2D version).

    Parameters
    ----------
    p : numpy.array
        Points to be tested with shape (N, 2).
    box_corner : numpy.array
        Corners of bounding box with shape (4, 2).

    Returns
    -------
    p_in_box : numpy.array
        Points within the box.

    """
    edge1 = box_corner[1, :] - box_corner[0, :]
    edge2 = box_corner[3, :] - box_corner[0, :]
    p_rel = p - box_corner[0, :].reshape(1, -1)

    l1 = get_projection_length_for_vector_projection(p_rel, edge1)
    l2 = get_projection_length_for_vector_projection(p_rel, edge2)
    # A point is within the box, if and only after projecting the
    # point onto the two edges s.t. p_rel = [edge1, edge2] @ [l1, l2]^T,
    # we have 0<=l1<=1 and 0<=l2<=1.
    mask = np.logical_and(l1 >= 0, l1 <= 1)
    mask = np.logical_and(mask, l2 >= 0)
    mask = np.logical_and(mask, l2 <= 1)
    p_in_box = p[mask, :]
    return p_in_box


def get_points_in_rotated_box_3d(p, box_corner):
    """
    Get points within a rotated bounding box (3D version).

    Parameters
    ----------
    p : numpy.array
        Points to be tested with shape (N, 3).
    box_corner : numpy.array
        Corners of bounding box with shape (8, 3).

    Returns
    -------
    p_in_box : numpy.array
        Points within the box.

    """
    edge1 = box_corner[1, :] - box_corner[0, :]
    edge2 = box_corner[3, :] - box_corner[0, :]
    edge3 = box_corner[4, :] - box_corner[0, :]

    p_rel = p - box_corner[0, :].reshape(1, -1)

    l1 = get_projection_length_for_vector_projection(p_rel, edge1)
    l2 = get_projection_length_for_vector_projection(p_rel, edge2)
    l3 = get_projection_length_for_vector_projection(p_rel, edge3)
    # A point is within the box, if and only after projecting the
    # point onto the two edges s.t. p_rel = [edge1, edge2] @ [l1, l2]^T,
    # we have 0<=l1<=1 and 0<=l2<=1.
    mask1 = np.logical_and(l1 >= 0, l1 <= 1)
    mask2 = np.logical_and(l2 >= 0, l2 <= 1)
    mask3 = np.logical_and(l3 >= 0, l3 <= 1)

    mask = np.logical_and(mask1, mask2)
    mask = np.logical_and(mask, mask3)
    p_in_box = p[mask, :]

    return p_in_box


def get_projection_length_for_vector_projection(a, b):
    """
    Get projection length for the Vector projection of a onto b s.t.
    a_projected = length * b. (2D version) See
    https://en.wikipedia.org/wiki/Vector_projection#Vector_projection_2
    for more details.

    Parameters
    ----------
    a : numpy.array
        The vectors to be projected with shape (N, 2).

    b : numpy.array
        The vector that is projected onto with shape (2).

    Returns
    -------
    length : numpy.array
        The length of projected a with respect to b.
    """
    assert np.sum(b ** 2, axis=-1) > 1e-6
    length = a.dot(b) / np.sum(b ** 2, axis=-1)
    return length


def nms_rotated(boxes, scores, threshold):
    """Performs rorated non-maximum suppression and returns indices of kept
    boxes.

    Parameters
    ----------
    boxes : torch.tensor
        The location preds with shape (N, 4, 2).

    scores : torch.tensor
        The predicted confidence score with shape (N,)

    threshold: float
        IoU threshold to use for filtering.

    Returns
    -------
        An array of index
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    boxes = boxes.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()

    polygons = common_utils.convert_format(boxes)

    top = 1000
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1][:top]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = common_utils.compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)

def nms_rotated_gpu(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> List[int]:
    """
    Parameters:
    ----------
    boxes :
        (N, 7)

    scores :
        (N,)

    Return:
    ------
    chosen_indices :
        (N,)
    """
    assert boxes.is_cuda
    assert scores.is_cuda

    sorted_indices = torch.argsort(scores, descending=True)
    
    # kept only 1000 boxes before compute iou
    top = 300
    sorted_indices = sorted_indices[:top]

    _temp_boxes = boxes[sorted_indices].contiguous()
    iou = torch.tensor(
        torch.zeros((_temp_boxes.shape[0], _temp_boxes.shape[0])), 
        dtype=torch.float32, 
        device='cuda'
    )
    iou3d_nms_cuda.boxes_overlap_bev_gpu(
        _temp_boxes, _temp_boxes, 
        iou
    )

    # zero out diag to because a box has 1.0-iou with itself
    diag = np.arange(iou.shape[0])
    iou[diag, diag] = 0

    chosen_indices = list()
    sorted_indices = sorted_indices.cpu().numpy()
    mask_suppressed = _temp_boxes.new_zeros(sorted_indices.shape[0]).bool()
    for ii, idx in enumerate(sorted_indices):
        if mask_suppressed[ii]:
            # this box is already suppressed -> nothing more to do
            continue
        
        chosen_indices.append(idx)
        
        # suppress boxes that have high iou with this box
        mask_large_iou = iou[ii] > threshold
        mask_suppressed[mask_large_iou] = True
    
    return chosen_indices



def nms_pytorch(boxes: torch.tensor, thresh_iou: float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.

    Parameters
    ----------
    boxes : torch.tensor
        The location preds along with the class predscores,
         Shape: [num_boxes,5].
    thresh_iou : float
        (float) The overlap thresh for suppressing unnecessary boxes.
    Returns
    -------
        A list of index
    """

    # we extract coordinates for every
    # prediction box present in P
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # we extract the confidence scores as well
    scores = boxes[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:

        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(idx.numpy().item()
                    if not idx.is_cuda else idx.cpu().detach().numpy().item())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]

        # find the IoU of every prediction in P with S
        IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]

    return keep


def remove_large_pred_bbx(bbx_3d):
    """
    Remove large bounding box.

    Parameters
    ----------
    bbx_3d : torch.Tensor
        Predcited 3d bounding box, shape:(N,8,3)

    Returns
    -------
    index : torch.Tensor
        The keep index.
    """
    bbx_x_max = torch.max(bbx_3d[:, :, 0], dim=1)[0]
    bbx_x_min = torch.min(bbx_3d[:, :, 0], dim=1)[0]
    x_len = bbx_x_max - bbx_x_min

    bbx_y_max = torch.max(bbx_3d[:, :, 1], dim=1)[0]
    bbx_y_min = torch.min(bbx_3d[:, :, 1], dim=1)[0]
    y_len = bbx_y_max - bbx_y_min

    bbx_z_max = torch.max(bbx_3d[:, :, 2], dim=1)[0]
    bbx_z_min = torch.min(bbx_3d[:, :, 2], dim=1)[0]
    z_len = bbx_z_max - bbx_z_min

    index = torch.logical_and(x_len <= 6, y_len <= 6)
    index = torch.logical_and(index, z_len)

    return index


def remove_bbx_abnormal_z(bbx_3d):
    """
    Remove bounding box that has negative z axis.

    Parameters
    ----------
    bbx_3d : torch.Tensor
        Predcited 3d bounding box, shape:(N,8,3)

    Returns
    -------
    index : torch.Tensor
        The keep index.
    """
    bbx_z_min = torch.min(bbx_3d[:, :, 2], dim=1)[0]
    bbx_z_max = torch.max(bbx_3d[:, :, 2], dim=1)[0]
    index = torch.logical_and(bbx_z_min >= -3, bbx_z_max <= 1)

    return index


def project_points_by_matrix_torch(points, transformation_matrix):
    """
    Project the points to another coordinate system based on the
    transformation matrix.

    Parameters
    ----------
    points : torch.Tensor
        3D points, (N, 3)
    transformation_matrix : torch.Tensor
        Transformation matrix, (4, 4)
    Returns
    -------
    projected_points : torch.Tensor
        The projected points, (N, 3)
    """
    points, is_numpy = \
        common_utils.check_numpy_to_torch(points)
    transformation_matrix, _ = \
        common_utils.check_numpy_to_torch(transformation_matrix)

    # convert to homogeneous coordinates via padding 1 at the last dimension.
    # (N, 4)
    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1)
    # (N, 4)
    projected_points = torch.einsum("ik, jk->ij", points_homogeneous,
                                    transformation_matrix)

    return projected_points[:, :3] if not is_numpy \
        else projected_points[:, :3].numpy()


def box_encode(
        boxes,
        anchors,
        encode_angle_to_vector=False,
        encode_angle_with_residual=False,
        smooth_dim=False,
        norm_velo=False
):
    """box encode for VoxelNet
        Args:
            boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r.
            anchors ([N, 7] Tensor): anchors.
    """

    box_ndim = anchors.shape[-1]

    if box_ndim == 7:
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
    else:
        xa, ya, za, wa, la, ha, vxa, vya, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, vxg, vyg, rg = torch.split(boxes, 1, dim=-1)

    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha

    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)

    ret = [xt, yt, zt, wt, lt, ht]

    if box_ndim > 7:
        if norm_velo:
            vxt = (vxg - vxa) / diagonal
            vyt = (vyg - vya) / diagonal
        else:
            vxt = vxg - vxa
            vyt = vyg - vya
        ret.extend([vxt, vyt])

    if encode_angle_to_vector:
        rgx = torch.cos(rg)
        rgy = torch.sin(rg)
        if encode_angle_with_residual:
            rax = torch.cos(ra)
            ray = torch.sin(ra)
            rtx = rgx - rax
            rty = rgy - ray
            ret.extend([rtx, rty])
        else:
            ret.extend([rgx, rgy])
    else:
        rt = rg - ra
        ret.append(rt)

    return torch.cat(ret, dim=-1)


def box_decode(
        box_encodings,
        anchors,
        encode_angle_to_vector=False,
        encode_angle_with_residual=False,
        bin_loss=False,
        smooth_dim=False,
        norm_velo=False,
):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    box_ndim = anchors.shape[-1]

    if box_ndim == 9:  # False
        xa, ya, za, wa, la, ha, vxa, vya, ra = torch.split(anchors, 1, dim=-1)
        if encode_angle_to_vector:
            xt, yt, zt, wt, lt, ht, vxt, vyt, rtx, rty = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, wt, lt, ht, vxt, vyt, rt = torch.split(box_encodings, 1, dim=-1)

    elif box_ndim == 7:
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        if encode_angle_to_vector:  # False
            xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    ret = [xg, yg, zg]

    if smooth_dim:  # False
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:
        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
    ret.extend([wg, lg, hg])

    if encode_angle_to_vector:  # False
        if encode_angle_with_residual:
            rax = torch.cos(ra)
            ray = torch.sin(ra)
            rgx = rtx + rax
            rgy = rty + ray
            rg = torch.atan2(rgy, rgx)
        else:
            rg = torch.atan2(rty, rtx)
    else:
        rg = rt + ra

    if box_ndim > 7:  # False
        if norm_velo:
            vxg = vxt * diagonal + vxa
            vyg = vyt * diagonal + vya
        else:
            vxg = vxt + vxa
            vyg = vyt + vya
        ret.extend([vxg, vyg])

    ret.append(rg)

    return torch.cat(ret, dim=-1)


def bbox_lwh_to_hwl(bbx_: Union[np.ndarray, torch.Tensor]) -> None:
    """
    Organize a set of bboxes in the form (x, y, z, l, w, h, yaw) (i.e., OpenPCDet format) to
    (x, y, z, h, w, l, yaw)
    """
    if isinstance(bbx_, np.ndarray):
        tmp = np.copy(bbx_[:, 3])
    else:
        tmp = bbx_.new_zeros(bbx_.shape[0])
        tmp = tmp + bbx_[:, 3]
    bbx_[:, 3] = bbx_[:, 5]
    bbx_[:, 5] = tmp
    return


def bbox_hwl_to_lwh(bbx_: Union[np.ndarray, torch.Tensor]) -> None:
    """
    Organize a set of bboxes in the form (x, y, z, h, w, l, yaw) (i.e., OpenPCDet format) to
    (x, y, z, l, w, h, yaw)
    """
    if isinstance(bbx_, np.ndarray):
        tmp = np.copy(bbx_[:, 3])
    else:
        tmp = bbx_.new_zeros(bbx_.shape[0])
        tmp = tmp + bbx_[:, 3]
    bbx_[:, 3] = bbx_[:, 5]
    bbx_[:, 5] = tmp
    return



def find_non_empty_boxes(boxes: np.ndarray, points: np.ndarray, min_num_points: int = 5, 
                         return_num_points_in_boxes: bool = False,
                         order: str = 'lwh') -> np.ndarray:
    """
    Find boxes that contain at least a certain number of points

    Parameters
    ----------
    boxes:
        (M, 7) - x, y, z, l, w, h, yaw

    points:
        (N, 4) - x, y, z, intensity | in the same frame as boxes

    min_num_points:
        minimum number of points a box should have to be considered not empty

    Return
    ------
    mask_non_empty:
        (M,) - True for non empty boxes
    """
    eps = 0.05
    if order == 'hwl':
        bbox_hwl_to_lwh(boxes)

    assert len(boxes.shape) == 2, f"len({boxes.shape}) != 2"
    assert len(points.shape) == 2, f"len({points.shape}) != 2"

    points_inds = np.arange(points.shape[0])
    box_idxs_of_pts = -np.ones(points.shape[0], dtype=int)  # (N,)
    num_points_in_boxes = np.zeros(boxes.shape[0], dtype=int)  # (M,)

    for box_idx in range(boxes.shape[0]):
        mask_remain_points = box_idxs_of_pts < 0  # (N,)
        if not np.any(mask_remain_points):
            continue

        # transform points to box coordinate    
        map_se3_box = make_se3(translation=boxes[box_idx, :3], yaw=boxes[box_idx, 6])
        box_se3_map = np.linalg.inv(map_se3_box)
        # apply_se3_(box_se3_map, points[mask_remain_points])  # NOTE: this fnc doesn't work here 
        # NOTE: (points coord don't change)
        # NOTE: so using the formula below instead
        points[mask_remain_points, :3] = points[mask_remain_points, :3] @  box_se3_map[:3, :3].T + box_se3_map[:3, -1]

        # compare with box size
        mask_in_box = np.abs(points[mask_remain_points, :3] / boxes[box_idx, 3: 6]) < (0.5 + eps)  # (n, 3)
        mask_in_box = np.all(mask_in_box, axis=1)  # (n,)

        # count number points in this box
        num_points_in_boxes[box_idx] = np.sum(mask_in_box.astype(int))

        # mark points in boxes to exclude them in subsequent calculation
        active_pts_inds = points_inds[mask_remain_points]  # (n,)
        active_pts_in_box_inds = active_pts_inds[mask_in_box]
        box_idxs_of_pts[active_pts_in_box_inds] = box_idx

        # re-transform points to map frame for the next iter
        # apply_se3_(map_se3_box, points[mask_remain_points])  # NOTE: this fnc doesn't work here 
        # NOTE: (points coord don't change)
        # NOTE: so using the formula below instead
        points[mask_remain_points, :3] = points[mask_remain_points, :3] @  map_se3_box[:3, :3].T + map_se3_box[:3, -1]

    if order == 'hwl':
        bbox_lwh_to_hwl(boxes)
    
    if return_num_points_in_boxes:
        return num_points_in_boxes
    else:
        return num_points_in_boxes >= min_num_points


def corners_top_face_to_boxes3d(corners: torch.Tensor) -> torch.Tensor:
    """
    
    Parameters
    ----------
    corners :
        (N, 4, 2)

    Return
    ------
    boxes :
        (N, 7)
    """
    centers = torch.mean(corners[:, [0, 2]], dim=1)  # (N, 3)
    cz = corners.new_ones(corners.shape[0])  # (N,)
    heights = corners.new_ones(corners.shape[0], device=corners.device)  # (N,)
    lengths = torch.norm(corners[:, 0] - corners[:, 3], dim=-1)  # (N,)
    widths = torch.norm(corners[:, 0] - corners[:, 1], dim=-1)  # (N,)
    heading = corners[:, 0] - corners[:, 3]  # (N, 3)
    yaw = torch.arctan2(heading[:, 1], heading[:, 0])
    boxes = torch.cat((centers, 
                       cz.unsqueeze(-1),
                       lengths.unsqueeze(-1), 
                       widths.unsqueeze(-1), 
                       heights.unsqueeze(-1), 
                       yaw.unsqueeze(-1)), 
                       dim=1)
    return boxes


def corners_3d_to_boxes(corners: torch.Tensor, return_tensor: bool = False):
    """
    
    Parameters
    ----------
    corners :
        (N, 8, 3)

    Return
    ------
    boxes :
        (N, 7)
    """
    centers = torch.mean(corners[:, [2, 4]], dim=1)  # (N, 3)
    heights = torch.norm(corners[:, 4] - corners[:, 0], dim=-1)  # (N,)
    lengths = torch.norm(corners[:, 0] - corners[:, 3], dim=-1)  # (N,)
    widths = torch.norm(corners[:, 0] - corners[:, 1], dim=-1)  # (N,)
    heading = corners[:, 0] - corners[:, 3]  # (N, 3)
    yaw = torch.atan2(heading[:, 1], heading[:, 0])
    boxes = torch.cat((centers, 
                       lengths.unsqueeze(-1), 
                       widths.unsqueeze(-1), 
                       heights.unsqueeze(-1), 
                       yaw.unsqueeze(-1)), 
                       dim=1)
    if return_tensor:
        return boxes
    else:
        return boxes.cpu().numpy()


def get_mask_center_in_range(boxes: Union[np.ndarray, torch.Tensor], coord_range: list) \
    -> Union[np.ndarray, torch.Tensor]:
    
    assert isinstance(coord_range, list), f"{type(coord_range)} != list"
     
    if isinstance(boxes, np.ndarray):
        _range = np.array(coord_range)

        mask_in_range = np.logical_and(
            boxes[:, :2] >= _range[:2], boxes[:, :2] <= _range[3: 5]
        ).all(axis=1)

    elif isinstance(boxes, torch.Tensor):
        _range = boxes.new_zeros(len(coord_range))
        for i, val in enumerate(coord_range):
            _range[i] += val

        mask_in_range = torch.logical_and(
            boxes[:, :2] >= _range[:2], boxes[:, :2] <= _range[3: 5]
        ).all(dim=1)

    else:
        raise ValueError(f"{type(boxes)} is unknown")

    return mask_in_range
