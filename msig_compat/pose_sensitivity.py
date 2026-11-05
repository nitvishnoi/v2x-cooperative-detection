"""
Inference-only pose-sensitivity sweep on DEV for the released epoch-55
checkpoint. Every connected vehicle's pose (its transform into the top-LiDAR
frame) is perturbed by a controlled amount before fusion; the roadside LiDARs
are left unchanged. All configurations are scored against the same ground
truth as the reference evaluation.

Configurations:
  nominal            poses as recorded (the reference, 84.51% mAP)
  trans_<d>m         each CAV shifted by d metres in a random horizontal
                     direction (drawn per CAV and frame, fixed seed)
  yaw_<a>deg         each CAV rotated by +/-a degrees (random sign) about its
                     own origin
  inverted           each CAV's transform replaced by its inverse - the DEV
                     analogue of the pose-frame change evaluated on FINAL
  cav_removed        CAVs dropped, roadside LiDARs only

The result shows how much CAV pose error costs, and how that compares with
removing the CAVs altogether.

Usage:
    .venv/bin/python msig_compat/pose_sensitivity.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model [--limit N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Mixed-Signals-Dataset"))

from msig_compat.fusion_baselines import (  # noqa: E402  (applies the compat patches)
    copy_nodes, infer, iterate_dev, load_model_and_data, report,
)
from msig_compat.bootstrap_utils import FrameRecorder  # noqa: E402
from msig_compat.export_submission import drop_agents_with_no_points_in_range  # noqa: E402

TRANS_M = (0.5, 1.0, 2.0)
YAW_DEG = (1.0, 2.0, 5.0)
CONFIGS = (['nominal'] + [f'trans_{d}m' for d in TRANS_M] + [f'yaw_{a}deg' for a in YAW_DEG]
           + ['inverted', 'cav_removed'])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


def perturb(nodes, rsu, config, rng):
    """Return a perturbed copy; RSU nodes are never changed."""
    out = copy_nodes(nodes)
    if config == 'cav_removed':
        return {k: v for k, v in out.items() if k in rsu}
    for name, v in out.items():
        if name in rsu:
            continue
        T = v['ego_se3_agent']
        if config.startswith('trans_'):
            d = float(config[len('trans_'):-1])
            phi = rng.uniform(0, 2 * np.pi)
            delta = np.eye(4)
            delta[:2, 3] = d * np.array([np.cos(phi), np.sin(phi)])
            v['ego_se3_agent'] = delta @ T
        elif config.startswith('yaw_'):
            a = np.deg2rad(float(config[len('yaw_'):-3])) * rng.choice([-1.0, 1.0])
            v['ego_se3_agent'] = T @ rot_z(a)          # rotate about the CAV's own origin
        elif config == 'inverted':
            v['ego_se3_agent'] = np.linalg.inv(T)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--block', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1, help='evaluate every stride-th DEV frame (default all)')
    opt = ap.parse_args()

    hypes, dataset, model, device = load_model_and_data(opt.model_dir)
    rsu = set(dataset.name_rsu)
    rng = np.random.default_rng(opt.seed)
    rec = FrameRecorder(CONFIGS)

    for seq_idx, nodes, gt_boxes_t, gt_class_t in iterate_dev(dataset, device, opt.limit, opt.stride):
        preds = {}
        for cfg in CONFIGS:
            p = perturb(nodes, rsu, cfg, rng)
            if cfg != 'nominal':
                p = drop_agents_with_no_points_in_range(p, dataset._cav_lidar_range)
            preds[cfg] = infer(dataset, model, device, p)
        rec.add(seq_idx, gt_boxes_t, gt_class_t, preds)

    print(f'\nEvaluated {len(rec.frames)} frames; CIs: block bootstrap, {opt.block}-frame blocks, '
          f'{opt.n_boot} resamples, stride {opt.stride}; perturbations applied to CAVs only')
    report(rec, CONFIGS, 'nominal', opt.n_boot, opt.block)


if __name__ == '__main__':
    main()
