import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import rearrange
import lovely_tensors as lt
lt.monkey_patch()

from opencood.data_utils.datasets import build_dataset
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.utils.my_visualization import PointsPainter
from opencood.utils import box_utils


def voir_batch():
    path_hypes = 'opencood/hypes_yaml/point_pillar_intermediate_multi.yaml'
    hypes = yaml_utils.load_yaml(path_hypes)
    dataset = build_dataset(hypes, visualize=True, train=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, 
                            collate_fn=dataset.collate_batch_train)
    iter_dataloader = iter(dataloader)

    batch = None
    for _ in range(1):
        batch = next(iter_dataloader)

    processed_lidar = batch['ego']['processed_lidar']
    for k, v in processed_lidar.items():
        print(f"{k} | {v}")

    print(f"batch['ego']['record_len']: {batch['ego']['record_len'].numpy()}")

    voxel_coords = batch['ego']['processed_lidar']['voxel_coords']
    unq_batch_idx = torch.unique(voxel_coords[:, 0].long())
    print(f"unq_batch_idx: {unq_batch_idx.numpy()}")



if __name__ == '__main__':
    voir_batch()
