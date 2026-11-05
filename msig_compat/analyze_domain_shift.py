"""
Dev-vs-Final domain diagnostic. Compares, without needing final-test labels
(which don't exist locally - only Codabench's scorer has them):

  - Per-agent point cloud stats (n_points, xyz range, intensity) on DEV
    (held-out labeled split) vs FINAL (the real competition test frames).
  - Scene stats: agents/frame, points/frame.
  - Ground truth box-size distribution on DEV (real labels).
  - Predicted box-size, confidence, and count distribution on both DEV and
    FINAL from the submitted epoch-55 checkpoint (FINAL has no labels, so
    predictions are the only lens available there).

Usage:
    .venv/bin/python msig_compat/analyze_domain_shift.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model \
        [--dev_limit 800] [--final_limit 1158]
"""
import argparse
import json
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
    load_frame, drop_agents_with_no_points_in_range, build_batch, AGENT_ORDER,
)

CLS_NAMES = {1: 'car', 2: 'bike', 3: 'pedestrian'}


def summarize(arr, name, indent='  '):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        print(f'{indent}{name}: (no data)')
        return
    print(f'{indent}{name}: n={arr.size} mean={arr.mean():.3f} std={arr.std():.3f} '
          f'p10={np.percentile(arr,10):.3f} p50={np.percentile(arr,50):.3f} p90={np.percentile(arr,90):.3f}')


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
    return dataset.post_process(batch, output_dict)


def analyze_dev(dataset, model, device, limit):
    print('\n' + '=' * 60)
    print('=== DEV (held-out labeled split) ===')
    print('=' * 60)

    n_total = min(len(dataset), limit) if limit else len(dataset)
    agent_pts = {a: [] for a in AGENT_ORDER}
    agent_intensity = {a: [] for a in AGENT_ORDER}
    agents_per_frame = []
    points_per_frame = []

    gt_lwh = {c: [] for c in CLS_NAMES}

    pred_conf = {c: [] for c in CLS_NAMES}
    pred_lwh = {c: [] for c in CLS_NAMES}
    pred_count_per_frame = {c: [] for c in CLS_NAMES}

    for idx in tqdm(range(n_total), desc='DEV'):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)

        agents_per_frame.append(len(agents_full))
        frame_pts = 0
        for a, data in agents_full.items():
            pts = data['points']
            agent_pts[a].append(pts.shape[0])
            agent_intensity[a].extend(pts[:, 3].tolist())
            frame_pts += pts.shape[0]
        points_per_frame.append(frame_pts)

        gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        gt_boxes, gt_class = dataset.preprocess_gt_boxes(
            gt_info_dict['gt_boxes_in_top'], gt_info_dict['gt_class'], gt_info_dict['gt_visibility'],
            [dataset.convert_agent_name_to_indicator(a) for a in agents_full.keys()],
        )
        for cls_id in CLS_NAMES:
            m = gt_class == cls_id
            gt_lwh[cls_id].append(gt_boxes[m, 3:6])

        pb, ps, pc, _, _ = run_inference(dataset, model, device, agents_full)
        pb, ps, pc = pb.cpu().numpy(), ps.cpu().numpy(), pc.cpu().numpy()
        for cls_id in CLS_NAMES:
            m = pc == cls_id
            pred_count_per_frame[cls_id].append(int(m.sum()))
            pred_conf[cls_id].extend(ps[m].tolist())
            if m.sum() > 0:
                pred_lwh[cls_id].append(pb[m][:, 3:6])

    print(f'\nframes: {n_total}')
    summarize(agents_per_frame, 'agents/frame')
    summarize(points_per_frame, 'points/frame')
    for a in AGENT_ORDER:
        if agent_pts[a]:
            summarize(agent_pts[a], f'{a} points/frame (when present)')
            summarize(agent_intensity[a][:200000], f'{a} intensity (subsampled)')

    print('\n--- ground truth box size (L/W/H), per class ---')
    for cls_id, name in CLS_NAMES.items():
        arr = np.concatenate(gt_lwh[cls_id]) if gt_lwh[cls_id] else np.zeros((0, 3))
        if arr.shape[0] == 0:
            continue
        print(f'  {name}: n={arr.shape[0]}')
        for i, dim in enumerate('LWH'):
            summarize(arr[:, i], f'    {dim}', indent='')

    print('\n--- prediction stats (epoch-55 model on DEV) ---')
    for cls_id, name in CLS_NAMES.items():
        summarize(pred_count_per_frame[cls_id], f'{name} boxes/frame')
        summarize(pred_conf[cls_id], f'{name} confidence')
        arr = np.concatenate(pred_lwh[cls_id]) if pred_lwh[cls_id] else np.zeros((0, 3))
        if arr.shape[0] > 0:
            for i, dim in enumerate('LWH'):
                summarize(arr[:, i], f'{name} predicted {dim}')

    return {
        'agents_per_frame': agents_per_frame, 'points_per_frame': points_per_frame,
        'pred_conf': pred_conf, 'pred_lwh': pred_lwh, 'pred_count_per_frame': pred_count_per_frame,
    }


def analyze_final(dataset, model, device, frames_dir, limit):
    print('\n' + '=' * 60)
    print('=== FINAL (real competition test frames, no labels) ===')
    print('=' * 60)

    frame_dirs = sorted(p for p in Path(frames_dir).iterdir() if p.is_dir())
    if limit:
        frame_dirs = frame_dirs[:limit]

    agent_pts = {a: [] for a in AGENT_ORDER}
    agent_intensity = {a: [] for a in AGENT_ORDER}
    agents_per_frame = []
    points_per_frame = []

    pred_conf = {c: [] for c in CLS_NAMES}
    pred_lwh = {c: [] for c in CLS_NAMES}
    pred_count_per_frame = {c: [] for c in CLS_NAMES}

    for frame_dir in tqdm(frame_dirs, desc='FINAL'):
        agents = load_frame(frame_dir)
        agents = drop_agents_with_no_points_in_range(agents, dataset._cav_lidar_range)

        agents_per_frame.append(len(agents))
        frame_pts = 0
        for a, data in agents.items():
            pts = data['points']
            agent_pts[a].append(pts.shape[0])
            agent_intensity[a].extend(pts[:, 3].tolist())
            frame_pts += pts.shape[0]
        points_per_frame.append(frame_pts)

        if len(agents) < 2:
            for cls_id in CLS_NAMES:
                pred_count_per_frame[cls_id].append(0)
            continue

        batch = build_batch(dataset, agents)
        batch = train_utils.to_device(batch, device)
        with torch.no_grad():
            output_dict = {'ego': model(batch['ego'])}
        pb, ps, pc, _, _ = dataset.post_process(batch, output_dict)
        pb, ps, pc = pb.cpu().numpy(), ps.cpu().numpy(), pc.cpu().numpy()
        for cls_id in CLS_NAMES:
            m = pc == cls_id
            pred_count_per_frame[cls_id].append(int(m.sum()))
            pred_conf[cls_id].extend(ps[m].tolist())
            if m.sum() > 0:
                pred_lwh[cls_id].append(pb[m][:, 3:6])

    print(f'\nframes: {len(frame_dirs)}')
    summarize(agents_per_frame, 'agents/frame')
    summarize(points_per_frame, 'points/frame')
    for a in AGENT_ORDER:
        if agent_pts[a]:
            summarize(agent_pts[a], f'{a} points/frame (when present)')
            summarize(agent_intensity[a][:200000], f'{a} intensity (subsampled)')

    print('\n--- prediction stats (epoch-55 model on FINAL) ---')
    for cls_id, name in CLS_NAMES.items():
        summarize(pred_count_per_frame[cls_id], f'{name} boxes/frame')
        summarize(pred_conf[cls_id], f'{name} confidence')
        arr = np.concatenate(pred_lwh[cls_id]) if pred_lwh[cls_id] else np.zeros((0, 3))
        if arr.shape[0] > 0:
            for i, dim in enumerate('LWH'):
                summarize(arr[:, i], f'{name} predicted {dim}')

    return {
        'agents_per_frame': agents_per_frame, 'points_per_frame': points_per_frame,
        'pred_conf': pred_conf, 'pred_lwh': pred_lwh, 'pred_count_per_frame': pred_count_per_frame,
    }


def print_side_by_side(dev, final):
    print('\n' + '=' * 60)
    print('=== DEV vs FINAL side-by-side ===')
    print('=' * 60)
    print(f"\n{'metric':<28} {'DEV':>15} {'FINAL':>15}")
    d_apf = np.mean(dev['agents_per_frame'])
    f_apf = np.mean(final['agents_per_frame'])
    print(f"{'mean agents/frame':<28} {d_apf:>15.3f} {f_apf:>15.3f}")
    d_ppf = np.mean(dev['points_per_frame'])
    f_ppf = np.mean(final['points_per_frame'])
    print(f"{'mean points/frame':<28} {d_ppf:>15.1f} {f_ppf:>15.1f}")

    for cls_id, name in CLS_NAMES.items():
        d_conf = dev['pred_conf'][cls_id]
        f_conf = final['pred_conf'][cls_id]
        d_med = np.median(d_conf) if d_conf else float('nan')
        f_med = np.median(f_conf) if f_conf else float('nan')
        print(f"{name+' median confidence':<28} {d_med:>15.3f} {f_med:>15.3f}")

        d_bpf = np.mean(dev['pred_count_per_frame'][cls_id])
        f_bpf = np.mean(final['pred_count_per_frame'][cls_id])
        print(f"{name+' boxes/frame':<28} {d_bpf:>15.3f} {f_bpf:>15.3f}")

        d_lwh = np.concatenate(dev['pred_lwh'][cls_id]) if dev['pred_lwh'][cls_id] else np.zeros((0, 3))
        f_lwh = np.concatenate(final['pred_lwh'][cls_id]) if final['pred_lwh'][cls_id] else np.zeros((0, 3))
        if d_lwh.shape[0] > 0 and f_lwh.shape[0] > 0:
            for i, dim in enumerate('LWH'):
                print(f"{name+f' predicted {dim} (mean)':<28} {d_lwh[:,i].mean():>15.3f} {f_lwh[:,i].mean():>15.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--dev_limit', type=int, default=800)
    ap.add_argument('--final_limit', type=int, default=None)
    ap.add_argument('--frames_dir', default=str(_PROJECT_ROOT / "data" / "mixed-signals-test-competition"))
    opt = ap.parse_args()

    hypes = yaml_utils.load_yaml(str(Path(opt.model_dir) / 'config.yaml'))
    hypes['flag_train'] = False
    dataset = build_dataset(hypes, visualize=False, train=False)

    model = train_utils.create_model(hypes)
    model.cuda()
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()
    device = torch.device('cuda')

    dev = analyze_dev(dataset, model, device, opt.dev_limit)
    final = analyze_final(dataset, model, device, opt.frames_dir, opt.final_limit)
    print_side_by_side(dev, final)


if __name__ == '__main__':
    main()
