"""
Fusion-strategy baselines for the released epoch-55 checkpoint, all scored on
the DEV split against ONE fixed ground-truth set: every box that is inside the
detection range and visible to at least one node present in the frame (the
same ground truth the reference evaluation uses). Only the detector's input
changes between methods, so differences are attributable to the fusion
strategy rather than to an easier or harder ground truth.

Methods (same trained weights, no retraining):
  no_fusion_top     single node: the roadside unit's top LiDAR only
  rsu_intermediate  intermediate fusion of the two roadside LiDARs (top, dome),
                    connected vehicles ignored
  early_fusion      raw point clouds of all present nodes merged into one
                    cloud in the top-LiDAR frame and processed as one node
  late_fusion       the detector run on each present node separately; the
                    per-node boxes are pooled and merged with the same
                    class-wise rotated NMS (IoU 0.15) as the reference
  intermediate      reference: per-node BEV features fused by attention

The script also reports how many frames contain at least one connected
vehicle within communication range (on DEV: all 1,791), a 95% block-bootstrap
confidence interval for each mAP (30-frame blocks of consecutive frames), and
a paired interval for each method's difference to intermediate fusion.

Note: the weights were trained for intermediate fusion, so the other methods
are inference-time strategy comparisons, not independently trained models.

Usage:
    .venv/bin/python msig_compat/fusion_baselines.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model [--limit N]
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
from opencood.utils import box_utils

from msig_compat.bootstrap_utils import FrameRecorder, CLS_IDS, IOU_THRESHOLDS
from msig_compat.export_submission import drop_agents_with_no_points_in_range

METHODS = ('no_fusion_top', 'rsu_intermediate', 'early_fusion', 'late_fusion', 'intermediate')


def copy_nodes(nodes, keep=None):
    """_get_projected_voxel_grid_of_every_agent transforms points in place,
    so every method gets its own copy."""
    return {k: {'points': v['points'].copy(), 'ego_se3_agent': v['ego_se3_agent'].copy()}
            for k, v in nodes.items() if keep is None or k in keep}


def infer(dataset, model, device, nodes):
    ego_part, agg_points = dataset._get_projected_voxel_grid_of_every_agent(nodes)
    data = dataset.init_output_of_getitem()
    data['ego'].update(ego_part)
    data['ego']['anchor_box'] = dataset._anchor_box
    data['metadata']['points'] = agg_points
    data['metadata']['gt_boxes_in_ego'] = np.zeros((0, 7))
    data['metadata']['gt_class'] = np.zeros((0,), dtype=int)
    batch = train_utils.to_device(dataset.collate_batch_test([data]), device)
    with torch.no_grad():
        output_dict = {'ego': model(batch['ego'])}
    pb, ps, pc = dataset.post_processor.post_process(batch, output_dict)
    return pb.to(device), ps.to(device), pc.to(device)


def early_fusion_input(nodes):
    """Merge every node's points into the top-LiDAR frame as a single node."""
    merged = []
    for v in nodes.values():
        pts = v['points'].copy()
        pts[:, :3] = pts[:, :3] @ v['ego_se3_agent'][:3, :3].T + v['ego_se3_agent'][:3, 3]
        merged.append(pts)
    return {'top': {'points': np.concatenate(merged), 'ego_se3_agent': np.eye(4)}}


def late_fusion(dataset, model, device, nodes, nms_thresh):
    boxes, scores, classes = [], [], []
    for name in nodes:
        pb, ps, pc = infer(dataset, model, device, copy_nodes(nodes, {name}))
        boxes.append(pb.reshape(-1, 7)); scores.append(ps.reshape(-1)); classes.append(pc.reshape(-1))
    pb, ps, pc = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
    out_b, out_s, out_c = [], [], []
    for cls_id, thr in zip(CLS_IDS, nms_thresh):
        m = pc == cls_id
        if m.sum() == 0:
            continue
        b, s = pb[m].contiguous(), ps[m].contiguous()
        keep = box_utils.nms_rotated_gpu(b, s, thr)
        out_b.append(b[keep]); out_s.append(s[keep]); out_c.append(torch.full_like(s[keep], cls_id).long())
    if not out_b:
        return torch.zeros(0, 7, device=device), torch.zeros(0, device=device), torch.zeros(0, dtype=torch.long, device=device)
    return torch.cat(out_b), torch.cat(out_s), torch.cat(out_c)


def iterate_dev(dataset, device, limit=None, stride=1):
    """Yield (seq_idx, nodes, gt_boxes, gt_class) for every stride-th DEV frame,
    with the fixed ground truth used by the reference evaluation."""
    total = min(len(dataset), limit) if limit else len(dataset)
    for idx in tqdm(range(0, total, stride)):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        nodes = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
        gt_info = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        gt_boxes, gt_class = dataset.preprocess_gt_boxes(
            gt_info['gt_boxes_in_top'], gt_info['gt_class'], gt_info['gt_visibility'],
            [dataset.convert_agent_name_to_indicator(a) for a in nodes])
        yield (seq_idx, nodes,
               torch.tensor(gt_boxes, dtype=torch.float32, device=device),
               torch.tensor(gt_class, dtype=torch.long, device=device))


def load_model_and_data(model_dir):
    hypes = yaml_utils.load_yaml(str(Path(model_dir) / 'config.yaml'))
    hypes['flag_train'] = False
    dataset = build_dataset(hypes, visualize=False, train=False)
    model = train_utils.create_model(hypes)
    model.cuda()
    _, model = train_utils.load_saved_model(model_dir, model)
    model.eval()
    return hypes, dataset, model, torch.device('cuda')


def report(rec, methods, reference, n_boot, block):
    """Per-class AP table, block-bootstrap 95% CI of mAP, and paired CI of the
    difference to the reference method."""
    names = {1: 'car', 2: 'bike', 3: 'ped'}
    for m in methods:
        met = rec.metrics(m)
        cols = ' | '.join(f'{names[c]}@{t} {met[(c, t)]:.4f}' for c in CLS_IDS for t in IOU_THRESHOLDS)
        line = f'{m:>17} | {cols} | mAP {met["mAP"]:.4f}'
        if n_boot:
            lo, hi = rec.ci(m, n_boot=n_boot, block=block)
            line += f' [95% CI {lo:.4f}, {hi:.4f}]'
            if m != reference:
                dlo, dhi = rec.paired_ci(m, reference, n_boot=n_boot, block=block)
                line += f' | diff vs {reference} {met["mAP"] - rec.metrics(reference)["mAP"]:+.4f} [{dlo:+.4f}, {dhi:+.4f}]'
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--n_boot', type=int, default=1000, help='bootstrap resamples (0 = no CI)')
    ap.add_argument('--block', type=int, default=30, help='bootstrap block length in frames')
    ap.add_argument('--stride', type=int, default=1, help='evaluate every stride-th DEV frame (default all)')
    opt = ap.parse_args()

    hypes, dataset, model, device = load_model_and_data(opt.model_dir)
    nms_thresh = hypes['postprocess']['nms_thresh']
    rec = FrameRecorder(METHODS)
    n_with_cav = 0

    for seq_idx, nodes, gt_boxes_t, gt_class_t in iterate_dev(dataset, device, opt.limit, opt.stride):
        usable = drop_agents_with_no_points_in_range(copy_nodes(nodes), dataset._cav_lidar_range)
        n_with_cav += any(n not in dataset.name_rsu for n in nodes)
        preds = {
            'intermediate': infer(dataset, model, device, copy_nodes(nodes)),
            'rsu_intermediate': infer(dataset, model, device, copy_nodes(usable, set(dataset.name_rsu))),
            'no_fusion_top': infer(dataset, model, device, copy_nodes(usable, {'top'})),
            'early_fusion': infer(dataset, model, device, early_fusion_input(usable)),
            'late_fusion': late_fusion(dataset, model, device, usable, nms_thresh),
        }
        rec.add(seq_idx, gt_boxes_t, gt_class_t, preds)

    print(f'\nEvaluated {len(rec.frames)} frames ({n_with_cav} with at least one CAV in range); '
          f'CIs: block bootstrap, {opt.block}-frame blocks, {opt.n_boot} resamples, stride {opt.stride}')
    report(rec, METHODS, 'intermediate', opt.n_boot, opt.block)


if __name__ == '__main__':
    main()
