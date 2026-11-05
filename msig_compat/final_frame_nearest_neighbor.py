"""
For every FINAL RSU-only frame, find its nearest DEV-RSU-only frames by
actual point-cloud input shape (a coarse BEV occupancy histogram, not
metadata), then compare the model's predicted car count on the FINAL frame
against the KNOWN GT car count of its nearest DEV neighbors. DEV has labels;
FINAL doesn't, so this is the closest thing to a label-based sanity check
available for FINAL - if FINAL's predicted counts track its neighbors' GT
counts, the 8.06 cars/frame figure is plausibly just scene composition; if
FINAL consistently predicts MORE cars than its most similar labeled scenes
ever contain, that's evidence of overprediction.

Usage:
    .venv/bin/python msig_compat/final_frame_nearest_neighbor.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model \
        [--k 10]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.distance import cdist

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Mixed-Signals-Dataset"))

from msig_compat.dataset_patch import apply_all_patches

apply_all_patches()

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from mixedsignals.utils.geometry import apply_se3_
from msig_compat.export_submission import (
    load_frame, drop_agents_with_no_points_in_range, build_batch,
)

CAV_RANGE = np.array([-51.2, -51.2, -6, 51.2, 51.2, 1])
GRID_CELL = 4.0
GRID_BINS = int((CAV_RANGE[3] - CAV_RANGE[0]) / GRID_CELL)  # 26


def fresh_copy(d):
    return {k: {'points': v['points'].copy(), 'ego_se3_agent': v['ego_se3_agent']} for k, v in d.items()}


def occupancy_feature(agents_dict):
    """Fuse all agents' points into ego frame, build a normalized coarse
    BEV occupancy histogram - the compact 'actual model input' shape."""
    all_pts = []
    for a, data in agents_dict.items():
        pts = data['points'].copy()
        apply_se3_(data['ego_se3_agent'], points_=pts)
        all_pts.append(pts[:, :2])
    xy = np.concatenate(all_pts, axis=0)
    mask = np.logical_and(xy >= CAV_RANGE[:2], xy < CAV_RANGE[3:5]).all(axis=1)
    xy = xy[mask]
    hist, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1],
        bins=[GRID_BINS, GRID_BINS],
        range=[[CAV_RANGE[0], CAV_RANGE[3]], [CAV_RANGE[1], CAV_RANGE[4]]],
    )
    hist = hist.flatten()
    s = hist.sum()
    return hist / s if s > 0 else hist


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--frames_dir', default=str(_PROJECT_ROOT / "data" / "mixed-signals-test-competition"))
    ap.add_argument('--k', type=int, default=10)
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
    dev_features = []
    dev_gt_car_count = []

    print("Building DEV RSU-only feature bank...")
    for idx in tqdm(range(n_dev), desc='DEV'):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
        rsu_only = fresh_copy({k: v for k, v in agents_full.items() if k in ('top', 'dome')})

        feat = occupancy_feature(rsu_only)
        dev_features.append(feat)

        gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        gt_class = gt_info_dict['gt_class']
        dev_gt_car_count.append(int((gt_class == 1).sum()))

    dev_features = np.stack(dev_features)  # (1791, 676)
    dev_gt_car_count = np.array(dev_gt_car_count)

    frame_dirs = sorted(p for p in Path(opt.frames_dir).iterdir() if p.is_dir())
    if opt.final_limit:
        frame_dirs = frame_dirs[:opt.final_limit]
    final_features = []
    final_pred_car_count = []
    final_names = []

    print("Processing FINAL RSU-only frames...")
    for frame_dir in tqdm(frame_dirs, desc='FINAL'):
        agents = load_frame(frame_dir)
        agents = drop_agents_with_no_points_in_range(agents, dataset._cav_lidar_range)
        has_cav = any(a in ('laser', '003', '004') for a in agents.keys())
        if has_cav or len(agents) < 2:
            continue

        feat = occupancy_feature(agents)
        final_features.append(feat)
        final_names.append(frame_dir.name)

        batch = build_batch(dataset, agents)
        batch = train_utils.to_device(batch, device)
        with torch.no_grad():
            output_dict = {'ego': model(batch['ego'])}
        pb, ps, pc, _, _ = dataset.post_process(batch, output_dict)
        pc = pc.cpu().numpy()
        final_pred_car_count.append(int((pc == 1).sum()))

    final_features = np.stack(final_features)
    final_pred_car_count = np.array(final_pred_car_count)

    print(f"\nDEV RSU-only frames: {len(dev_features)}   FINAL RSU-only frames: {len(final_features)}")
    print(f"Computing pairwise distances and {opt.k}-NN...")

    dists = cdist(final_features, dev_features, metric='euclidean')
    knn_idx = np.argsort(dists, axis=1)[:, :opt.k]

    neighbor_mean_gt = np.array([dev_gt_car_count[idx].mean() for idx in knn_idx])
    neighbor_median_gt = np.array([np.median(dev_gt_car_count[idx]) for idx in knn_idx])
    neighbor_max_gt = np.array([dev_gt_car_count[idx].max() for idx in knn_idx])

    diff = final_pred_car_count - neighbor_mean_gt

    print(f"\nFINAL predicted car count: mean={final_pred_car_count.mean():.3f}  median={np.median(final_pred_car_count):.1f}")
    print(f"Nearest-DEV-neighbor mean GT car count: mean={neighbor_mean_gt.mean():.3f}  median={np.median(neighbor_mean_gt):.1f}")
    print(f"\ncorrelation(FINAL predicted count, neighbor mean GT count) = {np.corrcoef(final_pred_car_count, neighbor_mean_gt)[0,1]:.4f}")

    print(f"\ndiff = FINAL predicted - neighbor mean GT:")
    print(f"  mean={diff.mean():+.3f}  median={np.median(diff):+.3f}  std={diff.std():.3f}")
    print(f"  fraction with diff > +2 (predicts notably more than similar labeled scenes): {(diff > 2).mean():.4f}")
    print(f"  fraction with diff in [-2, +2] (consistent with scene composition): {(np.abs(diff) <= 2).mean():.4f}")
    print(f"  fraction with diff < -2 (predicts notably fewer): {(diff < -2).mean():.4f}")

    print(f"\nfraction of FINAL frames where predicted count exceeds the MAX GT count "
          f"among its {opt.k} nearest DEV neighbors: {(final_pred_car_count > neighbor_max_gt).mean():.4f}")

    # a few concrete examples
    print(f"\n--- example frames (sorted by diff, most over-predicting first) ---")
    order = np.argsort(-diff)
    for i in order[:5]:
        print(f"  {final_names[i]}: predicted={final_pred_car_count[i]}  "
              f"neighbor GT counts={sorted(dev_gt_car_count[knn_idx[i]].tolist())}")


if __name__ == '__main__':
    main()
