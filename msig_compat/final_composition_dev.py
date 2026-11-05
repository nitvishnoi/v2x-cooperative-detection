"""
Evaluate the released checkpoint on DEV with the node composition of the FINAL
split.

FINAL frames ship without labels, but their inputs show which connected
vehicles (CAVs) are within communication range. This script measures the
distribution of the number of CAVs per FINAL frame, then, for every DEV frame,
draws a CAV count from that distribution (fixed seed) and keeps that many of
the frame's CAVs, chosen at random; the roadside LiDARs are always kept. It
reports the composition of both splits and the resulting DEV mAP, scored two
ways:

  fixed_gt  the ground truth of the full frame (objects visible to any node
            present in the full frame), as in the reference evaluation
  kept_gt   only objects visible to the nodes that were kept, the dataset's
            own definition for a frame with those nodes

The result separates how much of the DEV-to-FINAL gap the change in node
composition alone could explain.

Usage:
    .venv/bin/python msig_compat/final_composition_dev.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Mixed-Signals-Dataset"))

from msig_compat.fusion_baselines import copy_nodes, infer, load_model_and_data, report  # noqa: E402
from msig_compat.bootstrap_utils import FrameRecorder  # noqa: E402
from msig_compat.export_submission import load_frame, drop_agents_with_no_points_in_range, COMMUNICATION_RANGE  # noqa: E402


def final_cav_counts(frames_dir, rsu):
    counts = collections.Counter()
    for fd in sorted(Path(frames_dir).glob('f_*')):
        nodes = load_frame(fd)
        counts[sum(1 for k, v in nodes.items()
                   if k not in rsu and np.linalg.norm(v['ego_se3_agent'][:2, 3]) <= COMMUNICATION_RANGE)] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--frames_dir', default=str(_PROJECT_ROOT / 'data' / 'mixed-signals-test-competition'))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    opt = ap.parse_args()

    hypes, dataset, model, device = load_model_and_data(opt.model_dir)
    rsu = set(dataset.name_rsu)
    counts = final_cav_counts(opt.frames_dir, rsu)
    n_final = sum(counts.values())
    ks = sorted(counts)
    probs = np.array([counts[k] for k in ks], dtype=float) / n_final
    print('FINAL CAVs-in-range per frame:', {k: f'{counts[k]} ({100 * counts[k] / n_final:.1f}%)' for k in ks})

    rng = np.random.default_rng(opt.seed)
    fixed, kept = FrameRecorder(['fixed_gt']), FrameRecorder(['kept_gt'])
    dev_counts, kept_counts = collections.Counter(), collections.Counter()
    total = min(len(dataset), opt.limit) if opt.limit else len(dataset)
    for idx in tqdm(range(total)):
        seq_idx, i_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        i_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, i_in_seq)
        nodes = dataset._get_all_pointclouds_in_1sample(seq_idx, i_in_seq)
        gi = dataset.get_1sample_from_gt_infos(seq_idx, i_in_seq)

        cavs = [n for n in nodes if n not in rsu]
        dev_counts[len(cavs)] += 1
        k = min(int(rng.choice(ks, p=probs)), len(cavs))
        chosen = set(rng.choice(cavs, size=k, replace=False).tolist()) if k else set()
        kept_counts[k] += 1
        sub = drop_agents_with_no_points_in_range(copy_nodes(nodes, rsu | chosen), dataset._cav_lidar_range)
        pred = infer(dataset, model, device, sub)

        for rec, gt_nodes, name in ((fixed, nodes, 'fixed_gt'), (kept, sub, 'kept_gt')):
            b, c = dataset.preprocess_gt_boxes(gi['gt_boxes_in_top'], gi['gt_class'], gi['gt_visibility'],
                                               [dataset.convert_agent_name_to_indicator(a) for a in gt_nodes])
            rec.add(seq_idx, torch.tensor(b, dtype=torch.float32, device=device),
                    torch.tensor(c, dtype=torch.long, device=device), {name: pred})

    n = len(fixed.frames)
    print('DEV CAVs-in-range per frame:', {k: f'{dev_counts[k]} ({100 * dev_counts[k] / n:.1f}%)' for k in sorted(dev_counts)})
    print('DEV after matching to FINAL:', {k: f'{kept_counts[k]} ({100 * kept_counts[k] / n:.1f}%)' for k in sorted(kept_counts)})
    report(fixed, ['fixed_gt'], 'fixed_gt', opt.n_boot, 30)
    report(kept, ['kept_gt'], 'kept_gt', opt.n_boot, 30)


if __name__ == '__main__':
    main()
