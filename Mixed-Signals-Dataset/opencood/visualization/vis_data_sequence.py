# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
import argparse
from torch.utils.data import DataLoader

import numpy as np
import torch

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils


def vis_parser():
    parser = argparse.ArgumentParser(description="data visualization")
    parser.add_argument('--color_mode', type=str, default="intensity",
                        help='lidar color rendering mode, e.g. intensity,'
                             'z-value or constant.')
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    opt = parser.parse_args()
    return opt


sensor_to_rgb_map = {'003': np.array([204, 102, 153]), 
                     '004': np.array([255, 153, 51]),
                     'laser': np.array([255, 255, 153]), 
                     'top': np.array([0, 102, 255]), 
                     'dome': np.array([0, 204, 102])}
idx_to_name = ('top', 'dome', '003', '004', 'laser')

def collate_batch_for_visualization(batch):
    """
    Customized collate function for pytorch dataloader during training
    for late fusion dataset.

    Parameters
    ----------
    batch : dict

    Returns
    -------
    batch : dict
        Reformatted batch.
    """
    assert len(batch) == 1, f"{len(batch)} != 1"

    # during training, we only care about ego.
    output_dict = {'ego': {}}

    object_bbx_center = []
    object_bbx_mask = []
    projected_lidar_list = []

    # get ground truth
    ego_dict = batch[0]['ego']
    object_bbx_center.append(ego_dict['object_bbx_center'])
    object_bbx_mask.append(ego_dict['object_bbx_mask'])

    # convert to numpy, (B, max_num, 7)
    object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
    object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
    output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                'object_bbx_mask': object_bbx_mask})

    # get aggregate point cloud
    origin_lidar = ego_dict['origin_lidar']
    output_dict['ego'].update({'origin_lidar': origin_lidar})
    
    # get points color
    if origin_lidar.shape[1] == 5:
        points_agent_idx = origin_lidar[:, 4].astype(int)
        print('unq agent index: ', np.unique(points_agent_idx))
        points_color = np.zeros((origin_lidar.shape[0], 3))
        for idx_agent, agent_name in enumerate(idx_to_name):
            mask_agent = points_agent_idx == idx_agent
            if np.any(mask_agent):
                points_color[mask_agent] += sensor_to_rgb_map[agent_name]    
    else:
        points_color = None
    
    output_dict['ego'].update({'points_color': points_color})

    return output_dict


if __name__ == '__main__':
    opt = vis_parser()
    params = load_yaml(opt.hypes_yaml)

    opencda_dataset = build_dataset(params, visualize=True, train=False)
    data_loader = DataLoader(opencda_dataset, batch_size=1, num_workers=0,
                             collate_fn=collate_batch_for_visualization,
                             shuffle=False,
                             pin_memory=False)
        
    vis_utils.visualize_sequence_dataloader(data_loader,
                                            params['postprocess']['order'],
                                            color_mode=opt.color_mode)
