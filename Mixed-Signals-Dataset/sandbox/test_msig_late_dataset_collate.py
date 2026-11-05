import numpy as np
from torch.utils.data import DataLoader
import lovely_tensors as lt
lt.monkey_patch()

from opencood.hypes_yaml.yaml_utils import msig_load_yaml
from opencood.data_utils.datasets import build_dataset

from mixedsignals.utils.o3d_viz_utils import PointPainter
from mixedsignals.utils.mixed_signals_utils import AGENT_COLOR


def main():
    path_hypes = "opencood/hypes_yaml/msig_pillar_late.yaml"
    hypes = msig_load_yaml(path_hypes)
    dataset = build_dataset(hypes)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=1,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=dataset.collate_batch_test)
    iter_dataloader = iter(dataloader)
    for _ in range(2):
        data = next(iter_dataloader)

    metadata = data['metadata']
    points = metadata['points'].numpy()
    gt_boxes_in_ego = metadata['gt_boxes_in_ego']
    gt_class = metadata['gt_class']

    print(f'points: {points.shape}')
    print(f'gt_boxes_in_ego: {gt_boxes_in_ego.shape}')
    print(f'gt_class: {gt_class.shape}')

    painter = PointPainter()
    
    pts_indicators = points[:, -1].astype(int)
    unq_indicators = np.unique(pts_indicators)
    colors_points = np.zeros((points.shape[0], 3))
    for ind in unq_indicators:
        mask_agent = pts_indicators == ind
        agent_name = dataset.convert_agent_indicator_to_name(ind)
        colors_points[mask_agent] = AGENT_COLOR[agent_name]
    painter.add_pointclouds_(points, colors_points)
    
    colors_classes = np.eye(3)
    colors_gt = colors_classes[gt_class.numpy() - 1]
    painter.add_boxes_(gt_boxes_in_ego.numpy(), colors_gt)
    painter.show()


if __name__ == '__main__':
    main()
