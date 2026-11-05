"""
DOME-only point cloud geometry: DEV vs FINAL. No model inference needed -
pure point cloud statistics, in ego (top) frame, matching the actual
detection range [-51.2,51.2]m x/y, [-6,1]m z, voxel_size=[0.2,0.2,7]
(a pillar-style grid - XY cells 0.2m, Z collapsed into one 7m-tall pillar,
not a 0.4m 3D voxel).

Usage:
    .venv/bin/python msig_compat/dome_geometry_comparison.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Mixed-Signals-Dataset"))

from msig_compat.dataset_patch import apply_all_patches

apply_all_patches()

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from mixedsignals.utils.geometry import apply_se3_
from msig_compat.export_submission import load_frame

CAV_RANGE = np.array([-51.2, -51.2, -6, 51.2, 51.2, 1])
VOXEL_XY = 0.2
RADIAL_BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 10**9)]
RADIAL_LABELS = ['0-10m', '10-20m', '20-30m', '30-40m', '40-50m', '>50m']


def new_acc():
    return {
        'n_points_raw': [], 'n_points_in_range': [], 'n_occupied_voxels': [],
        'x_all': [], 'y_all': [], 'z_all': [],
        'radial_counts': [[] for _ in RADIAL_BUCKETS],
    }


def process(acc, pts_ego):
    acc['n_points_raw'].append(pts_ego.shape[0])
    mask = np.logical_and(pts_ego[:, :3] >= CAV_RANGE[:3], pts_ego[:, :3] < CAV_RANGE[3:]).all(axis=1)
    in_range = pts_ego[mask]
    acc['n_points_in_range'].append(in_range.shape[0])

    if in_range.shape[0] > 0:
        vx = np.floor((in_range[:, 0] - CAV_RANGE[0]) / VOXEL_XY).astype(np.int64)
        vy = np.floor((in_range[:, 1] - CAV_RANGE[1]) / VOXEL_XY).astype(np.int64)
        voxel_ids = vx.astype(np.int64) * 100000 + vy.astype(np.int64)
        acc['n_occupied_voxels'].append(len(np.unique(voxel_ids)))
        # subsample for distribution stats to keep memory bounded
        idx = np.random.choice(in_range.shape[0], size=min(2000, in_range.shape[0]), replace=False)
        acc['x_all'].extend(in_range[idx, 0].tolist())
        acc['y_all'].extend(in_range[idx, 1].tolist())
        acc['z_all'].extend(in_range[idx, 2].tolist())

        radial = np.linalg.norm(in_range[:, :2], axis=1)
        for i, (lo, hi) in enumerate(RADIAL_BUCKETS):
            acc['radial_counts'][i].append(int(np.logical_and(radial >= lo, radial < hi).sum()))
    else:
        acc['n_occupied_voxels'].append(0)
        for i in range(len(RADIAL_BUCKETS)):
            acc['radial_counts'][i].append(0)


def summarize(name, acc, n_frames):
    print(f"\n=== {name} (DOME only, n={n_frames} frames) ===")
    print(f"DOME points/frame (raw):      mean={np.mean(acc['n_points_raw']):.1f}  median={np.median(acc['n_points_raw']):.1f}")
    print(f"DOME points/frame (in range): mean={np.mean(acc['n_points_in_range']):.1f}  median={np.median(acc['n_points_in_range']):.1f}")
    print(f"occupied voxels/frame:        mean={np.mean(acc['n_occupied_voxels']):.1f}  median={np.median(acc['n_occupied_voxels']):.1f}")
    total_cells = ((CAV_RANGE[3]-CAV_RANGE[0])/VOXEL_XY) * ((CAV_RANGE[4]-CAV_RANGE[1])/VOXEL_XY)
    print(f"occupied voxel %:             {100*np.mean(acc['n_occupied_voxels'])/total_cells:.4f}%")

    x = np.array(acc['x_all']); y = np.array(acc['y_all']); z = np.array(acc['z_all'])
    for label, arr in (('X', x), ('Y', y), ('Z', z)):
        print(f"{label}: min={arr.min():.2f} max={arr.max():.2f} mean={arr.mean():.3f} "
              f"median={np.median(arr):.3f} p10={np.percentile(arr,10):.3f} p90={np.percentile(arr,90):.3f}")

    print("radial point-density distribution (mean points/frame per band):")
    for i, label in enumerate(RADIAL_LABELS):
        print(f"  {label:<8}: {np.mean(acc['radial_counts'][i]):>10.1f}")


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

    acc_dev = new_acc()
    n_dev = min(len(dataset), opt.dev_limit) if opt.dev_limit else len(dataset)
    for idx in tqdm(range(n_dev), desc='DEV'):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
        dome = agents_full['dome']
        pts_ego = dome['points'].copy()
        apply_se3_(dome['ego_se3_agent'], points_=pts_ego)
        process(acc_dev, pts_ego)

    frame_dirs = sorted(p for p in Path(opt.frames_dir).iterdir() if p.is_dir())
    if opt.final_limit:
        frame_dirs = frame_dirs[:opt.final_limit]
    acc_final = new_acc()
    for frame_dir in tqdm(frame_dirs, desc='FINAL'):
        agents = load_frame(frame_dir)
        dome = agents['dome']
        pts_ego = dome['points'].copy()
        apply_se3_(dome['ego_se3_agent'], points_=pts_ego)
        process(acc_final, pts_ego)

    summarize('DEV', acc_dev, n_dev)
    summarize('FINAL', acc_final, len(frame_dirs))


if __name__ == '__main__':
    main()
