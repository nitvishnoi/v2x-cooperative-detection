import pickle
from tqdm import tqdm

import numpy as np
import torch
from einops import rearrange

from opencood.tools.type_def import *
from opencood.tools.point_cloud_tools import remove_points_outside_range
from opencood.hypes_yaml.yaml_utils import msig_load_yaml
from opencood.data_utils.datasets import MSigEarlyFusion
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


def torch_unique_return_indices(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


def remove_repetitive_boxes(boxes: torch.Tensor) -> torch.Tensor:
    assert boxes.shape[1] == 9, f"{boxes.shape} != (N, 9), boxes must have track_id"
    unq_track_id, indices = torch_unique_return_indices(boxes[:, -1].long())
    if indices.shape[0] == boxes.shape[0]:
        return boxes
    else:
        return boxes[indices]


def get_points_to_boxes_correspondance(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Args:
        points: (N, 3+C) - x, y, z, C channels
        boxes: (M, 7+2) - box-7, cls_idx, track_idx
    
    Returns:
        pts2box_corr: (N, M) - bool
            pts2box_corr[i, j] = True if points[i] is inside boxes[j]
    """
    box_index_of_points = points_in_boxes_gpu(
        points[:, :3].unsqueeze(0).contiguous(),
        boxes[:, :7].unsqueeze(0).contiguous()
    )
    box_index_of_points = rearrange(box_index_of_points, '1 N -> N 1')

    boxes_indices = torch.arange(boxes.shape[0]).long().cuda()
    boxes_indices = rearrange(boxes_indices, 'M -> 1 M')

    pts2box_corr = box_index_of_points == boxes_indices  # (N, M)
    return pts2box_corr


def process_boxes_based_on_points(points: np.ndarray, 
                                  boxes: np.ndarray, 
                                  num_agents: int,
                                  num_min_points_in_boxes: int) -> Tuple[np.ndarray]:
    """
    Args:
        points: (N, 3+2) - x, y, z, i, agent_indicator
        boxes: (M, 7+2) - box-7, cls_idx, track_idx
    
    Returns:
        boxes: with empty boxes removed
        visibility: (M, num_agents) - bool
            visibility[j, k] = True if boxes[j] contains a point of agent_indicator := np.arange(num_agent)[k]
    """
    assert isinstance(points, np.ndarray), f"expect np.ndarray, got {type(points)}"
    assert isinstance(boxes, np.ndarray), f"expect np.ndarray, got {type(boxes)}"
    assert points.shape[1] == 5, f"{points.shape[1]} != 5, expect points to have agent indicator"
    assert boxes.shape[1] == 9, f"{boxes.shape} != (N, 9), boxes must have track_id"

    points = torch.from_numpy(points).float().cuda()
    boxes = torch.from_numpy(boxes).float().cuda()

    boxes = remove_repetitive_boxes(boxes)

    pts2box_corr = get_points_to_boxes_correspondance(points, boxes)  # (N, M)

    # === filter empty boxes
    num_points_of_boxes = pts2box_corr.long().sum(dim=0)  # (M,)
    mask_kept = num_points_of_boxes > num_min_points_in_boxes
    boxes = boxes[mask_kept].cpu().numpy()  # (m, 7+2)

    # === compute visibility (i.e., a box is visible to which agent)
    pts2box_corr = pts2box_corr[:, mask_kept].bool().cpu().numpy()  # (N, m)

    visibility = np.zeros((boxes.shape[0], num_agents))
    agent_id_of_points = points[:, -1].long().cpu().numpy()
    for box_idx in range(boxes.shape[0]):
        agent_id_of_points_in_box = agent_id_of_points[pts2box_corr[:, box_idx]]
        unq_agent_indicator = np.unique(agent_id_of_points_in_box)
        visibility[box_idx, unq_agent_indicator] = True

    return boxes, visibility


def make_partition_info(params: dict) -> None:
    dataset = MSigEarlyFusion(params, 
                              flag_return_only_early_pc_and_gt=True)
    flag_train = params['flag_train']

    gt_infos = dict()  # output

    for idx_in_dataset in tqdm(range(len(dataset)), desc=f'preprocess MSig, flag_train: {flag_train}'):
        data = dataset[idx_in_dataset]
        agg_points, gt_boxes, seq_idx, idx_in_seq = \
            data['agg_points'], data['gt_boxes'], data['seq_idx'], data['idx_in_seq']
        
        # NOTE: agg_points & gt_boxes are in ego frame
        # agg_points: (N, 5) - x, y, z, i, agent_indicator
        # gt_boxes: (M, 7+2) - box-7, category_id, track_id)

        sample_token = dataset.make_sample_token(seq_idx, idx_in_seq)
        if sample_token in gt_infos:
            continue  # don't process a data sample twice

        agg_points = remove_points_outside_range(agg_points, dataset.make_gt_info_range)

        gt_boxes, gt_visibility = process_boxes_based_on_points(agg_points, 
                                                                gt_boxes, 
                                                                dataset.num_agents, 
                                                                dataset.num_min_points_in_boxes)
        
        gt_infos[sample_token] = {
            'sample_token': sample_token,
            'gt_boxes_in_top': gt_boxes[:, :7].astype(np.float32),
            'gt_class': gt_boxes[:, -2].astype(np.int32),
            'gt_track_id': gt_boxes[:, -1].astype(np.int32),
            'gt_visibility': gt_visibility,
        }

    name_file_gt_info = 'train_gt_infos.pkl' if flag_train else 'test_gt_infos.pkl'
    data_root = params['data_root']
    path_gt_info = Path(data_root) / name_file_gt_info
    print(f"flag_train {flag_train}, save gt infos to {path_gt_info}")
    with open(str(path_gt_info), 'wb') as f:
        pickle.dump(gt_infos, f)
    return


def main():
    params = msig_load_yaml('opencood/hypes_yaml/msig_preprocessing.yaml')

    params['flag_train'] = True
    make_partition_info(params)
    
    params['flag_train'] = False
    make_partition_info(params)
   

if __name__ == '__main__':
    main()
