"""
Vehicle TP/FP/FN@0.5 (and @0.3/@0.7 for context) decomposition across
TOP only, TOP+DOME, and TOP+DOME+all-CAVs, on the same DEV frames -
to see whether RSU-only degradation is just "less information" (TOP+DOME
already close to full RSU-only capability) or DOME specifically pulls its
weight (TOP+DOME notably better than TOP alone).

Usage:
    .venv/bin/python msig_compat/vehicle_tp_fp_fn_by_agents.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Mixed-Signals-Dataset"))

from msig_compat.dataset_patch import apply_all_patches

apply_all_patches()

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou_bev
from msig_compat.export_submission import drop_agents_with_no_points_in_range

ASSOC_IOU = 0.1
THRESHOLDS = (0.3, 0.5, 0.7)

CONFIGS = {
    'TOP only': {'top'},
    'TOP+DOME': {'top', 'dome'},
    'TOP+DOME+all CAVs': None,  # None = keep everything naturally present
}


def fresh_copy(d):
    return {k: {'points': v['points'].copy(), 'ego_se3_agent': v['ego_se3_agent']} for k, v in d.items()}


def run_inference(dataset, model, device, dict_agents_pts_poses):
    ego_part, agg_points = dataset._get_projected_voxel_grid_of_every_agent(dict_agents_pts_poses)
    data = dataset.init_output_of_getitem()
    data['ego'].update(ego_part)
    data['ego']['anchor_box'] = dataset._anchor_box
    data['metadata']['points'] = agg_points
    batch = dataset.collate_batch_test([data])
    batch = train_utils.to_device(batch, device)
    with torch.no_grad():
        output_dict = {'ego': model(batch['ego'])}
    pb, ps, pc, _, _ = dataset.post_process(batch, output_dict)
    return pb, ps, pc


def match_counts(pred_boxes, pred_scores, gt_boxes, acc):
    acc['n_gt'] += gt_boxes.shape[0]
    acc['n_pred'] += pred_boxes.shape[0]
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return

    order = torch.argsort(pred_scores, descending=True)
    iou = boxes_iou_bev(pred_boxes, gt_boxes).cpu().numpy()
    matched_gt = np.zeros(gt_boxes.shape[0], dtype=bool)

    for i in order.cpu().numpy():
        row = iou[i].copy()
        row[matched_gt] = -1
        j = int(np.argmax(row))
        best_iou = row[j]
        if best_iou < ASSOC_IOU:
            continue
        matched_gt[j] = True
        for thr in THRESHOLDS:
            if best_iou >= thr:
                acc['tp'][thr] += 1


def new_acc():
    return {'n_gt': 0, 'n_pred': 0, 'tp': {t: 0 for t in THRESHOLDS}}


def report(name, acc):
    n_gt, n_pred = acc['n_gt'], acc['n_pred']
    print(f"\n{name}  (GT={n_gt}, predicted={n_pred})")
    for thr in THRESHOLDS:
        tp = acc['tp'][thr]
        fp = n_pred - tp
        fn = n_gt - tp
        prec = tp / (tp + fp) if (tp + fp) else float('nan')
        rec = tp / n_gt if n_gt else float('nan')
        print(f"  IoU>={thr}: TP={tp:>6} FP={fp:>6} FN={fn:>6}  precision={prec:.4f}  recall={rec:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--limit', type=int, default=None)
    opt = ap.parse_args()

    hypes = yaml_utils.load_yaml(str(Path(opt.model_dir) / 'config.yaml'))
    hypes['flag_train'] = False
    dataset = build_dataset(hypes, visualize=False, train=False)

    model = train_utils.create_model(hypes)
    model.cuda()
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()
    device = torch.device('cuda')

    n_total = min(len(dataset), opt.limit) if opt.limit else len(dataset)
    accs = {name: new_acc() for name in CONFIGS}

    for idx in tqdm(range(n_total)):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
        present = set(agents_full.keys())

        gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        gt_boxes, gt_class = dataset.preprocess_gt_boxes(
            gt_info_dict['gt_boxes_in_top'], gt_info_dict['gt_class'], gt_info_dict['gt_visibility'],
            [dataset.convert_agent_name_to_indicator(a) for a in agents_full.keys()],
        )
        gt_veh = torch.tensor(gt_boxes[gt_class == 1], dtype=torch.float32, device=device)

        for cfg_name, keep_set in CONFIGS.items():
            keep = keep_set if keep_set is not None else present
            subset = fresh_copy({k: v for k, v in agents_full.items() if k in keep})
            subset = drop_agents_with_no_points_in_range(subset, dataset._cav_lidar_range)
            if 'top' not in subset:
                continue
            pb, ps, pc = run_inference(dataset, model, device, subset)
            m = pc == 1
            match_counts(pb[m], ps[m], gt_veh, accs[cfg_name])

    for cfg_name in CONFIGS:
        report(cfg_name, accs[cfg_name])


if __name__ == '__main__':
    main()
