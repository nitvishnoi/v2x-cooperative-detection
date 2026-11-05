"""
Vehicle box-matching residual decomposition on DEV. For every predicted
vehicle box, greedily match to the best available (unmatched) GT vehicle
box by BEV IoU (loose association threshold, 0.1 - "roughly the same
object", not a TP/FP call). For matched pairs, compute the 7-parameter
residual (dx, dy, dz, dL, dW, dH, dyaw) and IoU. Reports the aggregate
picture plus the RSU-only vs CAV-present split, using the SAME frames
for both variants (mutation-safe fresh copies) so the comparison is
apples-to-apples.

Usage:
    .venv/bin/python msig_compat/vehicle_residual_analysis.py \
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

ASSOC_IOU = 0.1  # "same object" association threshold, not a TP call


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


def match_and_residuals(pred_boxes, pred_scores, gt_boxes, acc):
    """pred_boxes/gt_boxes: (N,7) cx,cy,cz,L,W,H,yaw. Greedy match by score
    order, best-IoU-available GT, association threshold ASSOC_IOU."""
    acc['n_gt'] += gt_boxes.shape[0]
    acc['n_pred'] += pred_boxes.shape[0]
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return

    order = torch.argsort(pred_scores, descending=True)
    iou = boxes_iou_bev(pred_boxes, gt_boxes).cpu().numpy()  # (N_pred, N_gt)
    pred_np = pred_boxes.cpu().numpy()
    gt_np = gt_boxes.cpu().numpy()

    matched_gt = np.zeros(gt_boxes.shape[0], dtype=bool)
    for i in order.cpu().numpy():
        row = iou[i].copy()
        row[matched_gt] = -1
        j = int(np.argmax(row))
        best_iou = row[j]
        if best_iou < ASSOC_IOU:
            continue
        matched_gt[j] = True

        dx = pred_np[i, 0] - gt_np[j, 0]
        dy = pred_np[i, 1] - gt_np[j, 1]
        dz = pred_np[i, 2] - gt_np[j, 2]
        dL = pred_np[i, 3] - gt_np[j, 3]
        dW = pred_np[i, 4] - gt_np[j, 4]
        dH = pred_np[i, 5] - gt_np[j, 5]
        dyaw = (pred_np[i, 6] - gt_np[j, 6] + np.pi) % (2 * np.pi) - np.pi

        acc['matched'] += 1
        acc['dx'].append(dx)
        acc['dy'].append(dy)
        acc['dz'].append(dz)
        acc['dL'].append(dL)
        acc['dW'].append(dW)
        acc['dH'].append(dH)
        acc['dyaw_deg'].append(np.degrees(dyaw))
        acc['iou'].append(best_iou)
        acc['gt_yaw_deg'].append(np.degrees(gt_np[j, 6]))
        acc['pred_yaw_deg'].append(np.degrees(pred_np[i, 6]))


def new_acc():
    return {
        'n_gt': 0, 'n_pred': 0, 'matched': 0,
        'dx': [], 'dy': [], 'dz': [], 'dL': [], 'dW': [], 'dH': [], 'dyaw_deg': [],
        'iou': [], 'gt_yaw_deg': [], 'pred_yaw_deg': [],
    }


def report(name, acc):
    n_gt = acc['n_gt']
    matched = acc['matched']
    print(f"\n=== {name} ===")
    print(f"GT vehicles: {n_gt}  |  predicted vehicles: {acc['n_pred']}  |  matched (IoU>={ASSOC_IOU}): {matched}")
    if n_gt:
        print(f"matched detection rate (matched/GT): {matched/n_gt:.4f}")
    if matched == 0:
        return

    dx = np.abs(acc['dx']); dy = np.abs(acc['dy']); dyaw = np.abs(acc['dyaw_deg'])
    iou = np.array(acc['iou'])

    # signed bias - is there a stable directional offset worth a zero-training correction?
    raw_dx = np.array(acc['dx']); raw_dy = np.array(acc['dy']); raw_dz = np.array(acc['dz'])
    raw_dL = np.array(acc['dL']); raw_dW = np.array(acc['dW']); raw_dH = np.array(acc['dH'])
    # only meaningful on the "correctly oriented" subset - a 180-flip subset would
    # cancel/corrupt a signed yaw average, so also report yaw bias restricted to |dyaw|<45
    small_yaw_mask = np.abs(np.array(acc['dyaw_deg'])) < 45
    print(f"\n--- signed bias (mean [std]) ---")
    for label, arr in (('dx', raw_dx), ('dy', raw_dy), ('dz', raw_dz),
                       ('dL', raw_dL), ('dW', raw_dW), ('dH', raw_dH)):
        print(f"  {label}: mean={arr.mean():+.4f}  median={np.median(arr):+.4f}  std={arr.std():.4f}")
    dyaw_small = np.array(acc['dyaw_deg'])[small_yaw_mask]
    print(f"  dyaw (|dyaw|<45 subset, n={small_yaw_mask.sum()}/{len(small_yaw_mask)}): "
          f"mean={dyaw_small.mean():+.4f}  median={np.median(dyaw_small):+.4f}  std={dyaw_small.std():.4f}")
    flip_frac = (np.abs(np.array(acc['dyaw_deg'])) > 135).mean()
    print(f"  fraction with |dyaw| > 135 deg (near-180 flip): {flip_frac:.4f}")

    print(f"\n{'Error':<8} {'median':>10} {'P75':>10} {'P90':>10}")
    for label, arr in (('|dx| m', dx), ('|dy| m', dy), ('|dz| m', np.abs(acc['dz'])),
                       ('dL m', np.array(acc['dL'])), ('dW m', np.array(acc['dW'])),
                       ('dH m', np.array(acc['dH'])), ('|dyaw|deg', dyaw)):
        print(f"{label:<8} {np.median(arr):>10.4f} {np.percentile(arr,75):>10.4f} {np.percentile(arr,90):>10.4f}")

    print(f"\nmean |dx| = {dx.mean():.4f}   mean |dy| = {dy.mean():.4f}   mean |dyaw| = {dyaw.mean():.4f} deg")
    print(f"median IoU = {np.median(iou):.4f}")
    print(f"IoU distribution: p10={np.percentile(iou,10):.4f} p25={np.percentile(iou,25):.4f} "
          f"p50={np.percentile(iou,50):.4f} p75={np.percentile(iou,75):.4f} p90={np.percentile(iou,90):.4f}")
    for thr in (0.3, 0.5, 0.7):
        frac_of_matched = (iou >= thr).mean()
        frac_of_gt = (iou >= thr).sum() / n_gt if n_gt else float('nan')
        print(f"  IoU>={thr}: {frac_of_matched:.4f} of matched pairs, {frac_of_gt:.4f} of all GT")


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

    acc_all = new_acc()
    acc_rsu = new_acc()
    acc_cav = new_acc()

    for idx in tqdm(range(n_total)):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)

        gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        gt_boxes, gt_class = dataset.preprocess_gt_boxes(
            gt_info_dict['gt_boxes_in_top'], gt_info_dict['gt_class'], gt_info_dict['gt_visibility'],
            [dataset.convert_agent_name_to_indicator(a) for a in agents_full.keys()],
        )
        gt_veh = torch.tensor(gt_boxes[gt_class == 1], dtype=torch.float32, device=device)

        pb, ps, pc = run_inference(dataset, model, device, fresh_copy(agents_full))
        m = pc == 1
        match_and_residuals(pb[m], ps[m], gt_veh, acc_all)

        rsu_only = fresh_copy({k: v for k, v in agents_full.items() if k in ('top', 'dome')})
        pb2, ps2, pc2 = run_inference(dataset, model, device, rsu_only)
        m2 = pc2 == 1
        match_and_residuals(pb2[m2], ps2[m2], gt_veh, acc_rsu)

        has_cav = any(a in ('laser', '003', '004') for a in agents_full.keys())
        if has_cav:
            match_and_residuals(pb[m], ps[m], gt_veh, acc_cav)

    report('DEV all (natural composition)', acc_all)
    report('DEV RSU-only (CAVs artificially dropped)', acc_rsu)
    report('DEV CAV-present frames (natural composition)', acc_cav)


if __name__ == '__main__':
    main()
