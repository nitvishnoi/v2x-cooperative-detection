import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import rearrange
import lovely_tensors as lt
lt.monkey_patch()

from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils
from opencood.utils.my_visualization import PointsPainter
from opencood.utils.freezed_rsu_fusion_utils import extract_rsu_cav_sparse_processed_lidar


def main():
    path_hypes = 'opencood/hypes_yaml/point_pillar_intermediate_multi.yaml'
    hypes = yaml_utils.load_yaml(path_hypes)
    dataset = build_dataset(hypes, visualize=True, train=True)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=4, 
                            collate_fn=dataset.collate_batch_train)
    iter_dataloader = iter(dataloader)

    batch = None
    for _ in range(1):
        batch = next(iter_dataloader)

    processed_lidar = batch['ego']['processed_lidar']
    print('=====================================')
    for k, v in processed_lidar.items():
        print(f"{k} | {v}")
    print('=====================================')

    print('---------------------')
    print('record_len: ', batch['ego']['record_len'].cpu().numpy())
    print('unique voxel_coords[:, 0]: ', torch.unique(processed_lidar['voxel_coords'][:, 0]).cpu().numpy())
    print('---------------------')

    rsu_dict, cavs_dict, merge_top_dome_indices, cav_indices = extract_rsu_cav_sparse_processed_lidar(batch['ego'])

    print('---------------------')
    print('merge_top_dome_indices: ', merge_top_dome_indices.cpu().numpy())
    print('---------------------')

    rsu_voxel_features = rsu_dict['voxel_features']
    rsu_voxel_coords = rsu_dict['voxel_coords']

    cav_voxel_features = cavs_dict['voxel_features']
    cav_voxel_coords = cavs_dict['voxel_coords']


    # ======================================================

    # extract voxel of rsu
    # --------------------
    chosen_batch_idx = 2
    
    unq_rsu_batch_idx = torch.unique(rsu_voxel_coords[:, 0].long())
    this_data_rsu_indices = unq_rsu_batch_idx[2 * chosen_batch_idx: 2 *(chosen_batch_idx + 1)]

    this_data_rsu_mask = rsu_voxel_coords[:, 0].long().unsqueeze(1) == this_data_rsu_indices.unsqueeze(0)
    this_data_rsu_mask = torch.any(this_data_rsu_mask, dim=1)

    this_data_voxel_features = rsu_voxel_features[this_data_rsu_mask]  # (N_this, num_pts_1vox, 4)
    this_data_rsu_points = rearrange(this_data_voxel_features[..., :3], 'N_vox N_pts C -> (N_vox N_pts) C')

    # extract voxel of cavs
    # --------------------
    num_cavs = batch['ego']['record_len'] - 2
    num_cavs = torch.cumsum(num_cavs, dim=0)

    unq_cavs_batch_idx = torch.unique(cav_voxel_coords[:, 0].long())

    if chosen_batch_idx == 0:
        this_data_cavs_indices = unq_cavs_batch_idx[:num_cavs[0]]
    else:
        this_data_cavs_indices = unq_cavs_batch_idx[num_cavs[chosen_batch_idx - 1]: num_cavs[chosen_batch_idx]]

    this_data_cav_mask = cav_voxel_coords[:, 0].long().unsqueeze(1) == this_data_cavs_indices.unsqueeze(0)
    this_data_cav_mask = torch.any(this_data_cav_mask, dim=1)

    this_data_cav_voxel_features = cav_voxel_features[this_data_cav_mask]  # (N_this, num_pts_1vox, 4)
    this_data_cav_points = rearrange(this_data_cav_voxel_features[..., :3], 'N_vox N_pts C -> (N_vox N_pts) C')


    # =========================================================
    gt_boxes_in_ego = batch['metadata']['gt_boxes_in_ego']
    gt_class = batch['metadata']['gt_class']
    gt_batch_counts = batch['metadata']['gt_batch_counts'].long()

    gt_boxes_in_ego = gt_boxes_in_ego[
        gt_batch_counts[max(0, chosen_batch_idx - 1)] : gt_batch_counts[chosen_batch_idx]
    ]
    gt_class = gt_class[
        gt_batch_counts[max(0, chosen_batch_idx - 1)] : gt_batch_counts[chosen_batch_idx]
    ]
    if hypes['postprocess']['order'] == 'hwl':
        box_utils.bbox_lwh_to_hwl(gt_boxes_in_ego)

    rsu_points = this_data_rsu_points.cpu().numpy()
    cav_points = this_data_cav_points.cpu().numpy()
    points = np.concatenate([rsu_points, cav_points])

    painter = PointsPainter(points, gt_boxes_in_ego)
    color_points = np.concatenate([
        np.zeros((rsu_points.shape[0], 3)) + 0.44,
        np.zeros((cav_points.shape[0], 3)) + np.array([0, 0, 1])
    ])
    color_class = np.eye(3)
    color_gt = color_class[gt_class.long().numpy() - 1]
    painter.show(color_points, color_gt)





if __name__ == '__main__':
    main()
