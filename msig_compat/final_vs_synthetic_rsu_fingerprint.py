"""
Prediction/feature fingerprint comparison:
  A - DEV naturally CAV-present (natural agent composition; DEV is 100% CAV-present)
  B - DEV artificially RSU-only (CAVs dropped from the same frames)
  C - FINAL naturally RSU-only (the ~70% subset of real test frames with no CAV
      in range at all - NOT all 1158 FINAL frames, which would dilute the
      comparison with the ~30% that do have a CAV)

Checks whether B is actually a good stand-in for C, or whether something
beyond "CAV missing" differs between the artificial ablation and the real
RSU-only regime.

Usage:
    .venv/bin/python msig_compat/final_vs_synthetic_rsu_fingerprint.py \
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
CAV_RANGE = np.array([-51.2, -51.2, -6, 51.2, 51.2, 1])
VOXEL_XY = 0.2
EDGE_MARGIN = 5.0  # meters from the +-51.2 boundary counted as "near edge"


def fresh_copy(d):
    return {k: {'points': v['points'].copy(), 'ego_se3_agent': v['ego_se3_agent']} for k, v in d.items()}


def occupied_voxels(agg_points):
    mask = np.logical_and(agg_points[:, :3] >= CAV_RANGE[:3], agg_points[:, :3] < CAV_RANGE[3:]).all(axis=1)
    pts = agg_points[mask]
    if pts.shape[0] == 0:
        return 0
    vx = np.floor((pts[:, 0] - CAV_RANGE[0]) / VOXEL_XY).astype(np.int64)
    vy = np.floor((pts[:, 1] - CAV_RANGE[1]) / VOXEL_XY).astype(np.int64)
    return len(np.unique(vx * 1000000 + vy))


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
    n_vox = occupied_voxels(agg_points)
    return pb.cpu().numpy(), ps.cpu().numpy(), pc.cpu().numpy(), agg_points.shape[0], n_vox


def new_acc():
    return {
        'points_per_frame': [], 'voxels_per_frame': [],
        'count_per_frame': {c: [] for c in CLS_NAMES},
        'car_scores': [], 'car_range': [], 'car_near_edge': 0, 'car_total': 0,
    }


def accumulate(acc, pb, ps, pc, n_pts, n_vox):
    acc['points_per_frame'].append(n_pts)
    acc['voxels_per_frame'].append(n_vox)
    for cls_id in CLS_NAMES:
        acc['count_per_frame'][cls_id].append(int((pc == cls_id).sum()))
    m = pc == 1
    car_boxes = pb[m]
    car_scores = ps[m]
    acc['car_scores'].extend(car_scores.tolist())
    if car_boxes.shape[0] > 0:
        rng = np.linalg.norm(car_boxes[:, :2], axis=1)
        acc['car_range'].extend(rng.tolist())
        near_edge = np.logical_or(
            np.abs(car_boxes[:, 0]) > (51.2 - EDGE_MARGIN),
            np.abs(car_boxes[:, 1]) > (51.2 - EDGE_MARGIN),
        )
        acc['car_near_edge'] += int(near_edge.sum())
        acc['car_total'] += car_boxes.shape[0]


def report(name, acc, n_frames):
    print(f"\n=== {name} (n={n_frames} frames) ===")
    print(f"points/frame:        mean={np.mean(acc['points_per_frame']):.1f}")
    print(f"occupied voxels:     mean={np.mean(acc['voxels_per_frame']):.1f}")
    for cls_id, cname in CLS_NAMES.items():
        print(f"{cname}/frame:          mean={np.mean(acc['count_per_frame'][cls_id]):.3f}")
    cs = np.array(acc['car_scores'])
    if cs.size:
        print(f"car confidence P50/P90/P95/P99: {np.percentile(cs,50):.4f} / {np.percentile(cs,90):.4f} / "
              f"{np.percentile(cs,95):.4f} / {np.percentile(cs,99):.4f}")
    cr = np.array(acc['car_range'])
    if cr.size:
        print(f"car range: mean={cr.mean():.2f}m  P90={np.percentile(cr,90):.2f}m  P99={np.percentile(cr,99):.2f}m")
    if acc['car_total']:
        print(f"car boxes near BEV edge (within {EDGE_MARGIN}m of +-51.2 boundary): "
              f"{acc['car_near_edge']}/{acc['car_total']} = {100*acc['car_near_edge']/acc['car_total']:.2f}%")


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
    acc_a = new_acc()
    acc_b = new_acc()

    for idx in tqdm(range(n_dev), desc='DEV'):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)

        pb, ps, pc, n_pts, n_vox = run_inference(dataset, model, device, fresh_copy(agents_full))
        accumulate(acc_a, pb, ps, pc, n_pts, n_vox)

        rsu_only = fresh_copy({k: v for k, v in agents_full.items() if k in ('top', 'dome')})
        pb2, ps2, pc2, n_pts2, n_vox2 = run_inference(dataset, model, device, rsu_only)
        accumulate(acc_b, pb2, ps2, pc2, n_pts2, n_vox2)

    frame_dirs = sorted(p for p in Path(opt.frames_dir).iterdir() if p.is_dir())
    if opt.final_limit:
        frame_dirs = frame_dirs[:opt.final_limit]
    acc_c = new_acc()
    n_final_rsu_only = 0
    for frame_dir in tqdm(frame_dirs, desc='FINAL'):
        agents = load_frame(frame_dir)
        agents = drop_agents_with_no_points_in_range(agents, dataset._cav_lidar_range)
        has_cav = any(a in ('laser', '003', '004') for a in agents.keys())
        if has_cav:
            continue  # only the naturally RSU-only subset counts as C
        if len(agents) < 2:
            continue
        n_final_rsu_only += 1

        batch = build_batch(dataset, agents)
        batch = train_utils.to_device(batch, device)
        with torch.no_grad():
            output_dict = {'ego': model(batch['ego'])}
        pb, ps, pc, _, _ = dataset.post_process(batch, output_dict)
        pb, ps, pc = pb.cpu().numpy(), ps.cpu().numpy(), pc.cpu().numpy()
        n_pts = batch['metadata']['points'].shape[0]
        n_vox = occupied_voxels(batch['metadata']['points'].cpu().numpy())
        accumulate(acc_c, pb, ps, pc, n_pts, n_vox)

    report('A - DEV naturally CAV-present', acc_a, n_dev)
    report('B - DEV artificially RSU-only', acc_b, n_dev)
    report(f'C - FINAL naturally RSU-only', acc_c, n_final_rsu_only)


if __name__ == '__main__':
    main()
