import numpy as np
import torch
from torch.utils.data import DataLoader
from opencood.data_utils.datasets import build_dataset
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.utils.my_visualization import PointsPainter
from opencood.utils import box_utils
from matplotlib import cm


def main():
    hypes_yaml = 'opencood/hypes_yaml/point_pillar_v2xvit_multi.yaml'
    hypes = yaml_utils.load_yaml(hypes_yaml)
    dataset = build_dataset(hypes, visualize=True, train=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, 
                            collate_fn=dataset.collate_batch_train)
    iter_dataloader = iter(dataloader)

    batch = None
    for _ in range(2):
        batch = next(iter_dataloader)

    chosen_batch_idx = 1
    
    gt_boxes = batch['metadata']['gt_boxes_in_ego']
    gt_class = batch['metadata']['gt_class']
    gt_batch_counts = batch['metadata']['gt_batch_counts']

    if hypes['postprocess']['order'] == 'hwl':
        box_utils.bbox_hwl_to_lwh(gt_boxes)

    if chosen_batch_idx == 0:
        gt_boxes = gt_boxes[:gt_batch_counts[0]]
    else:
        gt_boxes = gt_boxes[
            gt_batch_counts[chosen_batch_idx - 1]: gt_batch_counts[chosen_batch_idx]]

    points = batch['metadata']['points']
    points = points[points[:, 0].long() == chosen_batch_idx]
    print('points.shape: ', points.shape)

    unq_agent_index = torch.unique(points[:, -1].long())
    print('unq_agent_index: ', unq_agent_index)


    agent_colors = cm.rainbow(np.linspace(0, 1, 5))[:, :3]
    points_color = agent_colors[points[:, -1].long().numpy()]

    painter = PointsPainter(
        points[:, 1:].numpy(),
        gt_boxes.numpy(),
    )

    painter.show(points_color)


if __name__ == '__main__':
    main()
    