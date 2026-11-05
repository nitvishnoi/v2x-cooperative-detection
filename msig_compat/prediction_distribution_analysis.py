"""
Prediction-distribution analysis, beyond median confidence: full score
percentiles (P50/P90/P95/P99) and predictions-per-frame histograms, for
DEV (natural agent composition), the artificial RSU-only DEV ablation
(same frames, CAVs dropped), and FINAL (real competition test frames).

Usage:
    .venv/bin/python msig_compat/prediction_distribution_analysis.py \
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

from msig_compat.export_submission import (
    load_frame, drop_agents_with_no_points_in_range, build_batch,
)

CLS_NAMES = {1: 'car', 2: 'bike', 3: 'pedestrian'}
BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10**9)]
BUCKET_LABELS = ['0', '1-2', '3-5', '6-10', '11-20', '>20']


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
    return pb.cpu().numpy(), ps.cpu().numpy(), pc.cpu().numpy()


def new_acc():
    return {
        'total_per_frame': [],
        'per_class_per_frame': {c: [] for c in CLS_NAMES},
        'per_class_scores': {c: [] for c in CLS_NAMES},
    }


def accumulate(acc, pc, ps):
    acc['total_per_frame'].append(len(pc))
    for cls_id in CLS_NAMES:
        m = pc == cls_id
        acc['per_class_per_frame'][cls_id].append(int(m.sum()))
        acc['per_class_scores'][cls_id].extend(ps[m].tolist())


def bucket_counts(total_per_frame):
    counts = [0] * len(BUCKETS)
    for n in total_per_frame:
        for i, (lo, hi) in enumerate(BUCKETS):
            if lo <= n <= hi:
                counts[i] += 1
                break
    return counts


def pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--frames_dir', default=str(_PROJECT_ROOT / "data" / "mixed-signals-test-competition"))
    ap.add_argument('--dev_limit', type=int, default=None)
    ap.add_argument('--final_limit', type=int, default=None)
    opt = ap.parse_args()

    hypes = yaml_utils.load_yaml(str(Path(opt.model_dir) / 'config.yaml'))
    hypes['flag_train'] = False
    dataset = build_dataset(hypes, visualize=False, train=False)

    model = train_utils.create_model(hypes)
    model.cuda()
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()
    device = torch.device('cuda')

    n_dev = min(len(dataset), opt.dev_limit) if opt.dev_limit else len(dataset)
    acc_dev_all = new_acc()
    acc_dev_rsu = new_acc()

    for idx in tqdm(range(n_dev), desc='DEV'):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)

        pb, ps, pc = run_inference(dataset, model, device, fresh_copy(agents_full))
        accumulate(acc_dev_all, pc, ps)

        rsu_only = fresh_copy({k: v for k, v in agents_full.items() if k in ('top', 'dome')})
        pb2, ps2, pc2 = run_inference(dataset, model, device, rsu_only)
        accumulate(acc_dev_rsu, pc2, ps2)

    frame_dirs = sorted(p for p in Path(opt.frames_dir).iterdir() if p.is_dir())
    if opt.final_limit:
        frame_dirs = frame_dirs[:opt.final_limit]
    acc_final = new_acc()
    for frame_dir in tqdm(frame_dirs, desc='FINAL'):
        agents = load_frame(frame_dir)
        agents = drop_agents_with_no_points_in_range(agents, dataset._cav_lidar_range)
        if len(agents) < 2:
            accumulate(acc_final, np.array([]), np.array([]))
            continue
        batch = build_batch(dataset, agents)
        batch = train_utils.to_device(batch, device)
        with torch.no_grad():
            output_dict = {'ego': model(batch['ego'])}
        pb, ps, pc = dataset.post_process(batch, output_dict)[:3]
        pb, ps, pc = pb.cpu().numpy(), ps.cpu().numpy(), pc.cpu().numpy()
        accumulate(acc_final, pc, ps)

    print(f"\n{'Metric':<22} {'DEV all':>12} {'DEV RSU-only':>14} {'FINAL':>12}")
    print(f"{'frames':<22} {n_dev:>12} {n_dev:>14} {len(frame_dirs):>12}")
    print(f"{'predictions/frame':<22} {np.mean(acc_dev_all['total_per_frame']):>12.3f} "
          f"{np.mean(acc_dev_rsu['total_per_frame']):>14.3f} {np.mean(acc_final['total_per_frame']):>12.3f}")
    for cls_id, name in CLS_NAMES.items():
        print(f"{name+'/frame':<22} "
              f"{np.mean(acc_dev_all['per_class_per_frame'][cls_id]):>12.3f} "
              f"{np.mean(acc_dev_rsu['per_class_per_frame'][cls_id]):>14.3f} "
              f"{np.mean(acc_final['per_class_per_frame'][cls_id]):>12.3f}")

    print()
    for cls_id, name in CLS_NAMES.items():
        for p in (50, 90, 95, 99):
            d = pct(acc_dev_all['per_class_scores'][cls_id], p)
            r = pct(acc_dev_rsu['per_class_scores'][cls_id], p)
            f = pct(acc_final['per_class_scores'][cls_id], p)
            print(f"{name+f' score P{p}':<22} {d:>12.4f} {r:>14.4f} {f:>12.4f}")

    print(f"\n{'Bucket':<10} {'DEV all':>10} {'DEV RSU-only':>14} {'FINAL':>10}")
    bd = bucket_counts(acc_dev_all['total_per_frame'])
    br = bucket_counts(acc_dev_rsu['total_per_frame'])
    bf = bucket_counts(acc_final['total_per_frame'])
    for i, label in enumerate(BUCKET_LABELS):
        print(f"{label:<10} {bd[i]:>10} {br[i]:>14} {bf[i]:>10}")


if __name__ == '__main__':
    main()
