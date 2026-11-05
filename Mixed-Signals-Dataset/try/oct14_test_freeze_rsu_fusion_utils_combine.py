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
from opencood.utils.freezed_rsu_fusion_utils import extract_rsu_cav_sparse_processed_lidar, \
    combine_rsu_and_cav_dense_features


def simple_vfe(batch_dict, voxel_x, voxel_y, voxel_z, x_offset, y_offset, z_offset, 
               grid_size) -> torch.Tensor:
    voxel_features, voxel_num_points, coords = \
        batch_dict['voxel_features'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
    
    points_mean = \
        voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.float().view(-1, 1, 1)
    
    f_cluster = voxel_features[:, :, :3] - points_mean

    f_center = torch.zeros_like(voxel_features[:, :, :3])
    f_center[:, :, 0] = voxel_features[:, :, 0] - (
            coords[:, 3].to(voxel_features.dtype).unsqueeze(
                1) * voxel_x + x_offset)
    
    f_center[:, :, 1] = voxel_features[:, :, 1] - (
            coords[:, 2].to(voxel_features.dtype).unsqueeze(
                1) * voxel_y + y_offset)
    
    f_center[:, :, 2] = voxel_features[:, :, 2] - (
            coords[:, 1].to(voxel_features.dtype).unsqueeze(
                1) * voxel_z + z_offset)
    
    pillar_features = torch.cat([voxel_features, f_cluster, f_center], dim=-1)  # (..., 10)
    pillar_features = torch.mean(pillar_features, dim=1)

    # -----------------------
    # -----------------------
    nx, ny, nz = grid_size
    num_bev_features = 10
    batch_spatial_features = []
    batch_size = coords[:, 0].max().int().item() + 1

    for batch_idx in range(batch_size):
        spatial_feature = torch.zeros(
            num_bev_features,
            nz * nx * ny,
            dtype=pillar_features.dtype,
            device=pillar_features.device)

        batch_mask = coords[:, 0] == batch_idx
        this_coords = coords[batch_mask, :]

        indices = this_coords[:, 1] + \
                    this_coords[:, 2] * nx + \
                    this_coords[:, 3]
        indices = indices.type(torch.long)

        pillars = pillar_features[batch_mask, :]
        pillars = pillars.t()
        spatial_feature[:, indices] = pillars
        batch_spatial_features.append(spatial_feature)

    batch_spatial_features = \
        torch.stack(batch_spatial_features, 0)
    batch_spatial_features = \
        batch_spatial_features.view(batch_size, num_bev_features *
                                    nz, ny, nx)
    return batch_spatial_features


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

    # ---------------------------
    voxel_size = hypes['preprocess']['args']['voxel_size']
    cav_lidar_range = hypes['preprocess']['cav_lidar_range']
    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)

    x_offset = voxel_size[0] / 2 + cav_lidar_range[0]
    y_offset = voxel_size[1] / 2 + cav_lidar_range[1]
    z_offset = voxel_size[2] / 2 + cav_lidar_range[2]
    # ---------------------------

    # ---------------------------
    # make BEV image of mixed rsu & cav lidar
    target_bev_img = simple_vfe(batch['ego']['processed_lidar'], 
                                voxel_size[0], voxel_size[1], voxel_size[2],
                                x_offset, y_offset, z_offset, 
                                grid_size)
    print(f"target_bev_img: {target_bev_img}")
    # ---------------------------

    rsu_dict, cavs_dict, merge_top_dome_indices, cav_indices = extract_rsu_cav_sparse_processed_lidar(batch['ego'])
    rsu_bev_img = simple_vfe(rsu_dict, 
                             voxel_size[0], voxel_size[1], voxel_size[2],
                             x_offset, y_offset, z_offset, 
                             grid_size)
    print('rsu_bev_img: ', rsu_bev_img)
    
    cav_bev_img = simple_vfe(cavs_dict, 
                             voxel_size[0], voxel_size[1], voxel_size[2],
                             x_offset, y_offset, z_offset, 
                             grid_size)
    print('cav_bev_img: ', cav_bev_img)
    
    reconstruct_bev_img = combine_rsu_and_cav_dense_features(rsu_bev_img, 
                                                             cav_bev_img, 
                                                             merge_top_dome_indices, 
                                                             cav_indices)
    print(f"reconstruct_bev_img: {reconstruct_bev_img}")

    assert torch.allclose(target_bev_img, reconstruct_bev_img)
    print('test passed')


if __name__ == '__main__':
    main()
