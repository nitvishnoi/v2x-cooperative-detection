"""
Density-conditioned FP analysis on DEV synthetic RSU-only predictions
(the only setting with both ground truth AND an input condition matching
FINAL). Per-frame: predicted cars, GT cars, TP/FP/FN@0.5. Bucketed by
predicted-car-count per frame, to see whether precision degrades as
predicted density rises - the mechanism that would make FINAL's
8.06 cars/frame and near-zero neighbor-correlation make sense.

Usage:
    .venv/bin/python msig_compat/density_conditioned_fp_analysis.py \
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

ASSOC_THR = 0.5
BUCKETS = [(0, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 10**9)]
BUCKET_LABELS = ['0-4', '5-6', '7-8', '9-10', '11-12', '13+']


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


def tp_at_thr(pred_boxes, pred_scores, gt_boxes, thr):
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return 0
    order = torch.argsort(pred_scores, descending=True)
    iou = boxes_iou_bev(pred_boxes, gt_boxes).cpu().numpy()
    matched_gt = np.zeros(gt_boxes.shape[0], dtype=bool)
    tp = 0
    for i in order.cpu().numpy():
        row = iou[i].copy()
        row[matched_gt] = -1
        j = int(np.argmax(row))
        if row[j] >= thr:
            matched_gt[j] = True
            tp += 1
    return tp


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

    n_dev = min(len(dataset), opt.limit) if opt.limit else len(dataset)
    rows = []  # (n_pred, n_gt, tp, fp, fn)

    for idx in tqdm(range(n_dev)):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
        rsu_only = fresh_copy({k: v for k, v in agents_full.items() if k in ('top', 'dome')})

        gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        gt_boxes, gt_class = dataset.preprocess_gt_boxes(
            gt_info_dict['gt_boxes_in_top'], gt_info_dict['gt_class'], gt_info_dict['gt_visibility'],
            [dataset.convert_agent_name_to_indicator(a) for a in rsu_only.keys()],
        )
        gt_veh = torch.tensor(gt_boxes[gt_class == 1], dtype=torch.float32, device=device)

        pb, ps, pc = run_inference(dataset, model, device, rsu_only)
        m = pc == 1
        pred_veh, pred_score = pb[m], ps[m]

        n_pred = pred_veh.shape[0]
        n_gt = gt_veh.shape[0]
        tp = tp_at_thr(pred_veh, pred_score, gt_veh, ASSOC_THR)
        fp = n_pred - tp
        fn = n_gt - tp
        rows.append((n_pred, n_gt, tp, fp, fn))

    rows = np.array(rows)  # (N, 5): n_pred, n_gt, tp, fp, fn

    print(f"\n{'Pred cars/frame':<17} {'Frames':>7} {'GT cars/frame':>14} {'FP/frame':>10} {'FN/frame':>10} {'Precision':>10}")
    for (lo, hi), label in zip(BUCKETS, BUCKET_LABELS):
        mask = np.logical_and(rows[:, 0] >= lo, rows[:, 0] <= hi)
        n = mask.sum()
        if n == 0:
            print(f"{label:<17} {0:>7}")
            continue
        sub = rows[mask]
        gt_mean = sub[:, 1].mean()
        fp_mean = sub[:, 3].mean()
        fn_mean = sub[:, 4].mean()
        tp_sum = sub[:, 2].sum()
        fp_sum = sub[:, 3].sum()
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else float('nan')
        print(f"{label:<17} {n:>7} {gt_mean:>14.2f} {fp_mean:>10.2f} {fn_mean:>10.2f} {precision:>10.4f}")

    print(f"\noverall: {rows.shape[0]} frames, mean predicted={rows[:,0].mean():.2f}, mean GT={rows[:,1].mean():.2f}")
    overall_prec = rows[:,2].sum() / (rows[:,2].sum() + rows[:,3].sum())
    print(f"overall precision@0.5 = {overall_prec:.4f}")


if __name__ == '__main__':
    main()
