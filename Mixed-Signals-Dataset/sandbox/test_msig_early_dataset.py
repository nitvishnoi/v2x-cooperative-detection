from pprint import pprint
import numpy as np

from opencood.hypes_yaml.yaml_utils import msig_load_yaml
from opencood.data_utils.datasets import build_dataset

from mixedsignals.utils.o3d_viz_utils import PointPainter


def main():
    path_hypes = "opencood/hypes_yaml/msig_pillar_early.yaml"
    hypes = msig_load_yaml(path_hypes)
    dataset = build_dataset(hypes)
    data = dataset[293]

    metadata = data['metadata']
    points = metadata['points']
    gt_boxes_in_ego = metadata['gt_boxes_in_ego']
    gt_class = metadata['gt_class']

    painter = PointPainter()
    painter.add_pointclouds_(points)
    colors_classes = np.eye(3)
    colors_gt = colors_classes[gt_class - 1]
    painter.add_boxes_(gt_boxes_in_ego, colors_gt)
    painter.show()


if __name__ == '__main__':
    main()
