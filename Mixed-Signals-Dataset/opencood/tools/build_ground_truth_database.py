from pathlib import Path
from tqdm import tqdm
import pickle

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from opencood.data_utils.datasets import build_dataset
import opencood.hypes_yaml.yaml_utils as yaml_utils

from opencood.utils.box_utils import corners_3d_to_boxes
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


def main():
    root = Path('opencood/hypes_yaml')
    hypes_yaml = root / 'point_pillar_intermediate_multi.yaml'
    hypes = yaml_utils.load_yaml(hypes_yaml)
    hypes['data_augment'] = list()  # turn off data aug
    # if 'debug_model' in hypes:
    #     hypes['debug_model'] = True  # to use the entire training set
    hypes['make_gt_database'] = True

    dataset = build_dataset(hypes, visualize=True, train=True)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=4,
                             collate_fn=dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False)
    
    # root_gt_database = Path('data') / 'usyd_gt_database'
    root_gt_database = Path('/share/nikola/v2x_acfr') / 'usyd_gt_database'
    root_gt_database.mkdir(exist_ok=True)

    name_classes = ('car', 'bike', 'pedestrian')
    dir_classes = [root_gt_database / f"{cls_name}_database" for cls_name in name_classes]
    for dir_cls in dir_classes:
        dir_cls.mkdir(exist_ok=True)

    database_infos = dict([(cls_name, list()) for cls_name in name_classes])

    threshold_min_points = 50


    # TODO: for each sequence, avoid saving multiple boxes at 1 loc

    point_cloud_range = np.array([-70.4, -70.4, -6, 70.4, 70.4, 1])
    vox_res = np.array([0.1, 0.1, 7])
    bev_size = np.floor((point_cloud_range[3: 5] - point_cloud_range[:2]) / vox_res[:2]).astype(int)

    num_classes = 3  # ['car', 'bike', 'ped']
    max_num_samples = 10  # for each class, store no more than "max_num_samples" object per quantized location
    bev = torch.zeros(num_classes, bev_size[0], bev_size[1]).long()  # (3, W, H) 
    # NOTE: bev is to count how many samples of each class at each quantized location


    for data in tqdm(data_loader, total=len(dataset)):
        
        # NOTE: points & gt_boxes are in TOP frame

        points = data['ego']['origin_lidar']  # Tensor (N, 4+1) - x, y, z, intensity, agent_indicator
        assert points.shape[1] == 5, f"{points.shape[1]} != 5"
        points_agent_indicator = points[:, 4].long()

        gt_box_tensor, gt_class = dataset.post_processor.generate_gt_bbx(data)
        # NOTE: gt_class go from 1

        gt_boxes = corners_3d_to_boxes(gt_box_tensor, return_tensor=True)  # Tensor (M, 7)

        # find points in boxes
        box_idxs_of_pts = points_in_boxes_gpu(
            points[:, :3].unsqueeze(0).float().cuda(),
            gt_boxes[:, :7].unsqueeze(0).float().cuda()
        ).squeeze(0)  # (N,)
        
        unq_box_ids, counts = torch.unique(box_idxs_of_pts.long(), return_counts=True)
        unq_box_ids = unq_box_ids[counts >= threshold_min_points]

        unq_box_ids = unq_box_ids.cpu().numpy()
        box_idxs_of_pts = box_idxs_of_pts.cpu().numpy()

        for box_idx in unq_box_ids:
            # check if this box is eligible to store
            # i.e.: see if the location of this box (specified by its center
            # has reached max_num_samples for its class
            box_cls = gt_class[box_idx] - 1  # NOTE: gt_class go from 1
            box_coord_bev = torch.floor((gt_boxes[box_idx, :2] - point_cloud_range[:2]) / vox_res[:2]).long()
            flag_valid = bev[box_cls, box_coord_bev[0], box_coord_bev[1]] <= max_num_samples

            if not flag_valid:
                # skip this box
                continue

            mask_pts_in_box = box_idxs_of_pts == box_idx
            indicators = points_agent_indicator[mask_pts_in_box]           
            box_cls_name = name_classes[box_cls]

            # save box
            _filename = '%s_%010d.npy' % (box_cls_name, len(database_infos[box_cls_name]))
            box_info  = {
                'box_in_frame': 'top',
                'points_in_frame': 'top',
                'order': 'lwh',
                'agent_indices': np.unique(indicators),
                'box': gt_boxes[box_idx].numpy(),
                'cls_idx': gt_class[box_idx],  # NOTE: gt_class go from 1
                'path_points': str(dir_classes[box_cls] / _filename),
                'num_points': mask_pts_in_box.sum(),
            }

            database_infos[box_cls_name].append(box_info)

            # save points of this box
            pts = points[mask_pts_in_box].numpy()  # (n, 5)
            np.save(box_info['path_points'], pts)

            # increase counter of this class at the box's location
            bev[box_cls, box_coord_bev[0], box_coord_bev[1]] += 1
    
    with open(root_gt_database / 'database_infos.pkl', 'wb') as f:
        pickle.dump(database_infos, f)

  
if __name__ == '__main__':
    main()

