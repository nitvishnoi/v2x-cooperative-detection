from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from pprint import pprint

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone

class RSUPointPillarBackBone(nn.Module):
    def __init__(self, rsu_cfg: dict, 
                 save_downsample_feat_map: bool = False, 
                 save_spatial_feature_2d: bool = False):
        super().__init__()
        self.pillar_vfe = PillarVFE(rsu_cfg['pillar_vfe'],
                                    num_point_features=rsu_cfg['num_point_features'],
                                    voxel_size=rsu_cfg['voxel_size'],
                                    point_cloud_range=rsu_cfg['lidar_range'])
            
        self.scatter = PointPillarScatter(rsu_cfg['point_pillar_scatter'])

        self.backbone = BaseBEVBackbone(rsu_cfg['base_bev_backbone'], 64, 
                                        True,
                                        save_downsample_feat_map, 
                                        save_spatial_feature_2d)

        openpcdet_load_weights(self, rsu_cfg['path_ckpt'])

        for param in self.parameters():
            param.requires_grad_ = False

    def forward(self, batch_dict):
        self.eval()
        with torch.no_grad():
            batch_dict = self.pillar_vfe(batch_dict)
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)
        return batch_dict


def find_idx_top_dome_cavs_in_batch(record_len: torch.Tensor) -> Tuple[List[int]]:
    """
                top0, dome0, cav1_0, cav2_0 | top1, dome1, cav1_1 | top2, dome2
    record_len: 4                           | 3                   |   2
    index:      0      1     x       x      | 4      5      x     |   7    8
    """
    assert isinstance(record_len, torch.Tensor), f"type record_len: {type(record_len)}"

    _record_len = record_len.cpu().numpy()
    batch_size = _record_len.shape[0]

    top_indices = np.zeros(batch_size, dtype=int)
    top_indices[1:] = np.cumsum(_record_len)[:-1].astype(int)

    dome_indices = top_indices + 1

    top_indices = top_indices.tolist()
    dome_indices = dome_indices.tolist()

    cav_indices = set(np.arange(int(_record_len.sum())).tolist()) \
                    - set(top_indices) - set(dome_indices)
    cav_indices = list(cav_indices)
    cav_indices.sort()

    return top_indices, dome_indices, cav_indices


def extract_rsu_cav_sparse_processed_lidar(data_dict: dict) -> Dict[str, torch.Tensor]:
    """
    to avoid significant changes in dataset class
    rsu's processed lidar & cavs' processed lidar are mixed inside voxel_features
    the order of mixing is as following
    batch_0 : top_0, dome_0, cav_0_0, cav_1_0, cav_2_0
    batch_1 : top_1, dome_1, cav_0_1
    batch_2 : top_2, dome_2, cav_0_2, cav_1_2

    This function extracts RSUs' processed lidar & CAVs' processed lidar from the mix

    Returns:
    -------
    rsu_dict :

    cavs_dict :
    """
    voxel_features = data_dict['processed_lidar']['voxel_features']
    voxel_coords = data_dict['processed_lidar']['voxel_coords']
    voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

    device = voxel_features.device
    
    top_indices, dome_indices, cav_indices = \
        find_idx_top_dome_cavs_in_batch(data_dict['record_len'])
    
    merge_top_dome_indices = voxel_features.new_zeros(len(top_indices) * 2).long()
    merge_top_dome_indices[::2] = torch.tensor(top_indices).long().to(device)  # even
    merge_top_dome_indices[1::2] = torch.tensor(dome_indices).long().to(device)  # odd

    mask_voxels_of_top_dome = rearrange(voxel_coords[:, 0].long(), 'N_vox_tot -> N_vox_tot 1') \
        == rearrange(merge_top_dome_indices, 'ids -> 1 ids')  # (N_vox_tot, ids)
    mask_voxels_of_top_dome = torch.any(mask_voxels_of_top_dome, dim=1)  

    rsu_dict = {
        'voxel_features': voxel_features[mask_voxels_of_top_dome],
        'voxel_coords': voxel_coords[mask_voxels_of_top_dome],
        'voxel_num_points': voxel_num_points[mask_voxels_of_top_dome],
    }

    make_batch_index_contiguous(rsu_dict)

    # ============
    mask_voxels_of_cavs = torch.logical_not(mask_voxels_of_top_dome)
    cavs_dict = {
        'voxel_features': voxel_features[mask_voxels_of_cavs],
        'voxel_coords': voxel_coords[mask_voxels_of_cavs],
        'voxel_num_points': voxel_num_points[mask_voxels_of_cavs],
    }

    make_batch_index_contiguous(cavs_dict)

    # convert cav_indices from list to Tensor
    cav_indices = torch.tensor(cav_indices).long().to(merge_top_dome_indices.device)
    return rsu_dict, cavs_dict, merge_top_dome_indices, cav_indices


def combine_rsu_and_cav_dense_features(rsu_features: torch.Tensor, 
                                 cav_features: torch.Tensor, 
                                 top_dome_indices: torch.Tensor, 
                                 cav_indices: torch.Tensor) -> torch.Tensor:
    """
    Inverse function of extract_rsu_cav_processed_lidar
    """
    output = rsu_features.new_zeros(
        top_dome_indices.shape[0] + cav_indices.shape[0], *rsu_features.shape[1:],
    )

    output[top_dome_indices] = output[top_dome_indices] + rsu_features  # order: top0 - dome0 - top1 - dome1
    
    output[cav_indices] = output[cav_indices] + cav_features
    
    return output


def openpcdet_load_weights(model, path_ckpt, strict: bool = False):
    loaded_state_dict = torch.load( path_ckpt, map_location='cpu')

    model_state_dict = model.state_dict()

    new_model_state_dict = {}
    unloaded_params = list()
    for name, param in model_state_dict.items():
        if name in loaded_state_dict and loaded_state_dict[name].shape == param.shape:
            new_model_state_dict[name] = loaded_state_dict[name]
        # else:
        #     unloaded_params.append(name)
        elif name not in loaded_state_dict:
            unloaded_params.append(name)
            print(f"not found {name}")
        else:
            unloaded_params.append(name)
            print(f"shape mismatch {name}: {param.shape} != {loaded_state_dict[name].shape}")
    
    if len(unloaded_params) > 0:
        print('unloaded params')
        pprint(unloaded_params)
        if strict:
            raise ValueError(f"there are {len(unloaded_params)} unloaded params")

    model_state_dict.update(new_model_state_dict)
    model.load_state_dict(model_state_dict)
    
    print(f"loaded weights for {len(new_model_state_dict)} / {len(model_state_dict)}")
    return


def make_batch_index_contiguous(processed_lidar_: Dict[str, torch.Tensor]):
    batch_idx = processed_lidar_['voxel_coords'][:, 0].long()
    
    device = processed_lidar_['voxel_coords'].device
    
    sorted_ids = torch.argsort(batch_idx)
    for k in processed_lidar_.keys():
        processed_lidar_[k] = processed_lidar_[k][sorted_ids]
    
    unq, inv = torch.unique(processed_lidar_['voxel_coords'][:, 0].long(), return_inverse=True)
    
    processed_lidar_['voxel_coords'][:, 0] = torch.arange(unq.shape[0]).to(device)[inv]
    return
