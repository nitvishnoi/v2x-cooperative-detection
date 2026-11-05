import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from matplotlib import cm

from opencood.data_utils.datasets import build_dataset
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.utils.my_visualization import PointsPainter
from opencood.utils import box_utils


def visualize_train():
    path_hypes = 'opencood/hypes_yaml/point_pillar_late_multi.yaml'
    hypes = yaml_utils.load_yaml(path_hypes)
    dataset = build_dataset(hypes, visualize=True, train=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, 
                            collate_fn=dataset.collate_batch_train)
    iter_dataloader = iter(dataloader)

    batch = None
    for _ in range(1):
        batch = next(iter_dataloader)

    chosen_batch_idx = 1
    
    points = batch['ego']['points']
    gt_boxes_in_agent = batch['ego']['gt_boxes_in_agent']
    gt_class = batch['ego']['gt_class']
    gt_batch_counts = batch['ego']['gt_batch_counts'].long()
    
    anchor_box = dataset.post_processor.generate_anchor_box()
    anchor_box = torch.from_numpy(anchor_box).float()
    print(f'anchor_box: {anchor_box.shape}')
    anchor_box = rearrange(anchor_box, 'h w n_cls num_anchor C -> h w (num_anchor n_cls) C')
    
    label_dict = batch['ego']['label_dict']
    mask_pos_anchors = label_dict['pos_equal_one'][chosen_batch_idx] > 0  # (h, w, num_anchor, n_cls)
    mask_pos_anchors = rearrange(mask_pos_anchors, 'h w num_anchor n_cls -> h w (num_anchor n_cls)')

    pos_anchor_box = anchor_box[mask_pos_anchors]  # (n_pos, 7)
    print(f'pos_anchor_box: {pos_anchor_box}')
    box_utils.bbox_hwl_to_lwh(pos_anchor_box)


    # ==============================================================
    points = points[points[:, 0].long() == chosen_batch_idx, 1:]
    gt_boxes_in_agent = gt_boxes_in_agent[
        gt_batch_counts[max(0, chosen_batch_idx - 1)] : gt_batch_counts[chosen_batch_idx]
    ]
    gt_class = gt_class[
        gt_batch_counts[max(0, chosen_batch_idx - 1)] : gt_batch_counts[chosen_batch_idx]
    ]
    if hypes['postprocess']['order'] == 'hwl':
        box_utils.bbox_lwh_to_hwl(gt_boxes_in_agent)

    painter = PointsPainter(points, np.concatenate([gt_boxes_in_agent, pos_anchor_box]))
    color_points = np.zeros((points.shape[0], 3)) + 0.44
    color_class = np.eye(3)
    color_gt = color_class[gt_class.long().numpy() - 1]
    color_anchor = np.zeros((pos_anchor_box.shape[0], 3)) + np.array([0, 1, 1])
    painter.show(color_points, np.concatenate([color_gt, color_anchor]))


def visualize_test():
    path_hypes = 'opencood/hypes_yaml/pixor_late_fusion.yaml'
    hypes = yaml_utils.load_yaml(path_hypes)
    dataset = build_dataset(hypes, visualize=True, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, 
                            collate_fn=dataset.collate_batch_test)
    iter_dataloader = iter(dataloader)

    batch = None
    for _ in range(1):
        batch = next(iter_dataloader)

    points_in_top = batch['metadata']['points'].numpy()
    gt_boxes_in_top = batch['metadata']['gt_boxes_in_top'].numpy()
    gt_class = batch['metadata']['gt_class'].numpy().astype(int)
    gt_visibility = batch['metadata']['gt_visibility'].numpy()

    color_agents = cm.rainbow(np.linspace(0, 1, dataset.max_num_agents))[:, :3]
    color_class = np.eye(3)

    color_points = color_agents[points_in_top[:, -1].astype(int)]
    color_gt = color_class[gt_class - 1]
    painter = PointsPainter(points_in_top, gt_boxes_in_top)
    painter.show(color_points, color_gt)

    print('showing gt visible to top only')
    top_indicator = dataset.usyd.convert_agent_name_to_agent_indicator('top')
    mask_visible_to_top = gt_visibility[:, top_indicator]
    color_points = color_points * \
        (points_in_top[:, -1].astype(int) == top_indicator).astype(float).reshape(-1, 1)
    painter = PointsPainter(points_in_top, gt_boxes_in_top[mask_visible_to_top])
    painter.show(color_points, color_gt[mask_visible_to_top])




if __name__ == '__main__':
    print('=======================')
    print('showing points and labels of 1 single agent')
    # visualize_train()
    print('=======================')

    print('=======================')
    print('showing points and labels of every agent')
    visualize_test()
    print('=======================')

