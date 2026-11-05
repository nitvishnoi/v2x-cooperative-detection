import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import cm

from opencood.data_utils.datasets import build_dataset
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.utils.my_visualization import PointsPainter
from opencood.utils import box_utils


def visualize_train():
    hypes_yaml = 'opencood/hypes_yaml/point_pillar_laly.yaml'
    hypes = yaml_utils.load_yaml(hypes_yaml)
    dataset = build_dataset(hypes, visualize=True, train=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, 
                            collate_fn=dataset.collate_batch_train)
    iter_dataloader = iter(dataloader)

    batch = None
    for _ in range(2):
        batch = next(iter_dataloader)

    chosen_batch_idx = 1
    
    points = batch['ego']['points']
    gt_boxes = batch['metadata']['gt_boxes_in_ego']
    gt_class = batch['metadata']['gt_class']

    if hypes['postprocess']['order'] == 'hwl':
        box_utils.bbox_hwl_to_lwh(gt_boxes)

    num_lidar_points = batch['metadata']['num_lidar_points']
    batch_gt_count = batch['metadata']['batch_gt_count']
    others_lidar = batch['metadata']['others_lidar']

    # =============================
    points = points[points[:, 0].long() == chosen_batch_idx, 1:]
    assert points.shape[1] == dataset.num_point_features

    others_lidar = others_lidar[others_lidar[:, 0].long() == chosen_batch_idx, 1:]

    if chosen_batch_idx == 0:
        gt_boxes = gt_boxes[:batch_gt_count[0]]
        gt_class = gt_class[:batch_gt_count[0]]
    else:
        gt_boxes = gt_boxes[batch_gt_count[chosen_batch_idx-1]: batch_gt_count[chosen_batch_idx]]
        gt_class = gt_class[batch_gt_count[chosen_batch_idx-1]: batch_gt_count[chosen_batch_idx]]

    points = points.numpy()
    gt_boxes = gt_boxes.numpy()
    gt_class = gt_class.numpy()
    num_lidar_points = int(num_lidar_points[chosen_batch_idx])
    others_lidar = others_lidar.numpy()

    print('points.shape: ', points.shape)
    print('num_lidar_points: ', num_lidar_points)

    colors_points = np.zeros((points.shape[0], 3)) + 0.44
    colors_cls = np.eye(3)
    colors_gt = colors_cls[gt_class - 1]
    colors_agent = cm.rainbow(np.linspace(0, 1, 5))[:, :3]
    colors_other_lidar = colors_agent[others_lidar[:, -1].astype(int)]

    painter = PointsPainter(
        np.concatenate([points[:, :3], others_lidar[:, :3]]), 
        gt_boxes)
    painter.show(np.concatenate([colors_points, colors_other_lidar]), 
                 colors_gt, special_points=points[num_lidar_points:, :3])


def visualize_test():
    hypes_yaml = 'opencood/hypes_yaml/point_pillar_laly.yaml'
    hypes = yaml_utils.load_yaml(hypes_yaml)
    dataset = build_dataset(hypes, visualize=True, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, 
                            collate_fn=dataset.collate_batch_test)
    iter_dataloader = iter(dataloader)

    batch = None
    for _ in range(1):
        batch = next(iter_dataloader)

    chosen_batch_idx = 0
    
    points = batch['ego']['points']
    gt_boxes = batch['metadata']['gt_boxes_in_ego']
    gt_class = batch['metadata']['gt_class']

    # if hypes['postprocess']['order'] == 'hwl':
    #     box_utils.bbox_hwl_to_lwh(gt_boxes)

    num_lidar_points = batch['metadata']['num_lidar_points']

    # =============================
    points = points[points[:, 0].long() == chosen_batch_idx, 1:]
    assert points.shape[1] == dataset.num_point_features

    points = points.numpy()
    gt_boxes = gt_boxes.numpy()
    gt_class = gt_class.numpy()
    num_lidar_points = int(num_lidar_points[chosen_batch_idx])

    colors_points = np.zeros((points.shape[0], 3)) + 0.44
    colors_cls = np.eye(3)
    colors_gt = colors_cls[gt_class - 1]

    painter = PointsPainter(points, gt_boxes)
    painter.show(colors_points, colors_gt, special_points=points[num_lidar_points:, :3])


if __name__ == '__main__':
    visualize_train()
