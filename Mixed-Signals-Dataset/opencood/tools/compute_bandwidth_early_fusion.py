from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.common_utils import measure_tensor_size_in_bytes


def main():
    hypes_yaml = 'opencood/hypes_yaml/point_pillar_early_multi.yaml'
    hypes = yaml_utils.load_yaml(hypes_yaml)
    dataset = build_dataset(hypes, visualize=True, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, 
                            collate_fn=dataset.collate_batch_test)

    total_size = 0
    total_num_agents = 0

    for data in tqdm(dataloader, total=len(dataset)):
        points = data['metadata']['points']
        num_cavs = torch.unique(points[:, -1].long()).shape[0]
        total_num_agents += num_cavs

        processed_lidar = data['ego']['processed_lidar']
        total_size += measure_tensor_size_in_bytes(processed_lidar['voxel_features'])
        total_size += measure_tensor_size_in_bytes(processed_lidar['voxel_coords'])
        total_size += measure_tensor_size_in_bytes(processed_lidar['voxel_num_points'])
    # ---

    avg_bandwidth = float(total_size)  / (1e3 * float(total_num_agents))
    print(f'EVAL INFO | avg bandwidth: {avg_bandwidth} (KB)')


if __name__ == '__main__':
    main()
