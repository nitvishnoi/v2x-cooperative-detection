"""
Per-vehicle causal attribution: incrementally restore CAVs onto the
TOP+DOME baseline (TOP+DOME -> +laser -> +laser+003 -> +laser+003+004),
and for every GT vehicle on DEV, find the FIRST step where it flips from
FN to TP@0.5. Attributes that recovery to whichever CAV addition caused
it, then buckets by range to see whether laser/003/004 recover
near/medium/far vehicles differently.

Usage:
    .venv/bin/python msig_compat/cav_recovery_attribution.py \
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

TP_THR = 0.5
REQUIRED_CAVS = {'laser', '003', '004'}
STEPS = ['TOP+DOME', '+laser', '+003', '+004']
STEP_KEEP = [
    {'top', 'dome'},
    {'top', 'dome', 'laser'},
    {'top', 'dome', 'laser', '003'},
    {'top', 'dome', 'laser', '003', '004'},
]
RANGE_BUCKETS = [(0, 20), (20, 35), (35, 10**9)]
RANGE_LABELS = ['near (<20m)', 'medium (20-35m)', 'far (>35m)']


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


def per_gt_status(pred_boxes, pred_scores, gt_boxes):
    """For each GT box, is it matched at IoU>=TP_THR by some prediction
    (greedy, score order), and with what IoU/confidence."""
    n_gt = gt_boxes.shape[0]
    matched = np.zeros(n_gt, dtype=bool)
    iou_val = np.zeros(n_gt, dtype=np.float32)
    score_val = np.zeros(n_gt, dtype=np.float32)
    if pred_boxes.shape[0] == 0 or n_gt == 0:
        return matched, iou_val, score_val

    order = torch.argsort(pred_scores, descending=True)
    iou = boxes_iou_bev(pred_boxes, gt_boxes).cpu().numpy()
    pred_scores_np = pred_scores.cpu().numpy()
    claimed = np.zeros(n_gt, dtype=bool)
    for i in order.cpu().numpy():
        row = iou[i].copy()
        row[claimed] = -1
        j = int(np.argmax(row))
        if row[j] >= TP_THR:
            claimed[j] = True
            matched[j] = True
            iou_val[j] = row[j]
            score_val[j] = pred_scores_np[i]
    return matched, iou_val, score_val


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

    # recovered_by[step_name] -> list of (range, iou, score) for GT vehicles
    # first matched AT this step (were FN before it)
    recovered_by = {s: [] for s in STEPS[1:]}
    already_tp_at_base = []  # ranges of vehicles already TP at TOP+DOME
    never_recovered = []     # ranges of vehicles still FN after all CAVs

    for idx in tqdm(range(n_dev)):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
        present = set(agents_full.keys())
        if not REQUIRED_CAVS.issubset(present):
            continue  # keep the incremental sequence clean - only frames with all 3 CAVs

        gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        gt_boxes, gt_class = dataset.preprocess_gt_boxes(
            gt_info_dict['gt_boxes_in_top'], gt_info_dict['gt_class'], gt_info_dict['gt_visibility'],
            [dataset.convert_agent_name_to_indicator(a) for a in agents_full.keys()],
        )
        gt_veh = gt_boxes[gt_class == 1]
        if gt_veh.shape[0] == 0:
            continue
        gt_range = np.linalg.norm(gt_veh[:, :2], axis=1)
        gt_veh_t = torch.tensor(gt_veh, dtype=torch.float32, device=device)

        prev_matched = None
        for step_i, (step_name, keep) in enumerate(zip(STEPS, STEP_KEEP)):
            if not keep.issubset(present) and step_i > 0:
                # this CAV isn't present in this frame - skip forward without
                # claiming a recovery for it, still track TOP+DOME baseline
                if step_i == 0:
                    pass
                continue
            subset = fresh_copy({k: v for k, v in agents_full.items() if k in keep})
            subset = drop_agents_with_no_points_in_range(subset, dataset._cav_lidar_range)
            if 'top' not in subset:
                continue

            pb, ps, pc = run_inference(dataset, model, device, subset)
            m = pc == 1
            matched, iou_val, score_val = per_gt_status(pb[m], ps[m], gt_veh_t)

            if step_i == 0:
                already_tp_at_base.extend(gt_range[matched].tolist())
                prev_matched = matched
            else:
                newly = np.logical_and(matched, ~prev_matched)
                recovered_by[step_name].extend(gt_range[newly].tolist())
                prev_matched = np.logical_or(prev_matched, matched)

        never_recovered.extend(gt_range[~prev_matched].tolist())

    print(f"\nAlready TP at TOP+DOME baseline: {len(already_tp_at_base)} vehicles")
    print(f"Never recovered (FN through all CAVs): {len(never_recovered)} vehicles")

    print(f"\n{'CAV addition':<12} {'Recovered':>10} | {'near<20m':>10} {'20-35m':>10} {'>35m':>10}")
    for step_name in STEPS[1:]:
        ranges = np.array(recovered_by[step_name])
        n = len(ranges)
        if n == 0:
            print(f"{step_name:<12} {0:>10}")
            continue
        counts = []
        for lo, hi in RANGE_BUCKETS:
            counts.append(int(np.logical_and(ranges >= lo, ranges < hi).sum()))
        print(f"{step_name:<12} {n:>10} | {counts[0]:>10} {counts[1]:>10} {counts[2]:>10}")
        print(f"{'':<12} {'':>10} | {100*counts[0]/n:>9.1f}% {100*counts[1]/n:>9.1f}% {100*counts[2]/n:>9.1f}%")

    print(f"\n--- range distribution (mean/median) per CAV's recovered vehicles ---")
    for step_name in STEPS[1:]:
        ranges = np.array(recovered_by[step_name])
        if len(ranges) == 0:
            continue
        print(f"{step_name}: mean={ranges.mean():.2f}m  median={np.median(ranges):.2f}m")

    base_ranges = np.array(already_tp_at_base)
    never_ranges = np.array(never_recovered)
    print(f"\nalready-TP-at-baseline range: mean={base_ranges.mean():.2f}m  median={np.median(base_ranges):.2f}m")
    if len(never_ranges):
        print(f"never-recovered range: mean={never_ranges.mean():.2f}m  median={np.median(never_ranges):.2f}m")


if __name__ == '__main__':
    main()
