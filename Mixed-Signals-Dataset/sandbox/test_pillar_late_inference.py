import numpy as np
import torch
from torch.utils.data import DataLoader
import lovely_tensors as lt
lt.monkey_patch()
from pprint import pprint

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import msig_load_yaml
from opencood.tools import train_utils, inference_utils

from mixedsignals.utils.o3d_viz_utils import PointPainter


def main_inference():
    path_hypes = "opencood/hypes_yaml/msig_pillar_late.yaml"
    path_ckpt = "ckpt/pointpillar_early_fusion.pth"
    hypes = msig_load_yaml(path_hypes)
    device = torch.device('cuda')

    dataset = build_dataset(hypes)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=1,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=dataset.collate_batch_test)

    model = train_utils.create_model(hypes)
    train_utils.openpcdet_load_weights(model, path_ckpt, strict=True)
    model.eval().cuda()

    iter_dataloader = iter(dataloader)
    for _ in range(2):
        batch_data = next(iter_dataloader)
    batch_data = train_utils.to_device(batch_data, device)

    pprint(batch_data)

    pred_boxes, pred_scores, pred_class, gt_boxes, gt_class = \
        inference_utils.inference_late_fusion_slow(batch_data, model, dataset)
    
    print(f"pred_boxes: {pred_boxes}")
    print(f"pred_scores: {pred_scores}")
    print(f"pred_class: {pred_class}")

    print(f"gt_boxes: {gt_boxes}")
    print(f"gt_class: {gt_class}")

    output = {
        'points': batch_data['metadata']['points'].cpu(),
        'pred_boxes': pred_boxes.cpu(),
        'pred_scores': pred_scores.cpu(),
        'pred_class': pred_class.cpu(),
        'gt_boxes': gt_boxes.cpu(),
        'gt_class': gt_class.cpu(),
    }

    torch.save(output, 'sandbox/output_test_pillar_late_inference.pth')
    print('output is saved to output_test_pillar_late_inference.pth')


def main_visualize():
    output = torch.load('sandbox/output_test_pillar_late_inference.pth')

    points = output['points']
    gt_boxes = output['gt_boxes'].numpy()
    gt_class = output['gt_class'].numpy()
    pred_boxes = output['pred_boxes'].numpy()
    pred_class = output['pred_class'].numpy()


    painter = PointPainter()
    painter.add_pointclouds_(points)
    painter.add_boxes_(gt_boxes, np.array([0.5, 0.5, 0.]))
    colors_classes = np.eye(3)
    colors_pred = colors_classes[pred_class - 1]
    painter.add_boxes_(pred_boxes, colors_pred)
    painter.show()


if __name__ == '__main__':
    main_inference()
    main_visualize()

   