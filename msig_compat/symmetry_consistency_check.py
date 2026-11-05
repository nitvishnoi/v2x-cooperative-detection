"""
Diagnostic only - does not touch submission.zip or export_submission.py.

Ground-truth-free sanity check on the competition's own (unlabeled) test
frames, adapted from the "Symmetry Inversion" metamorphic relation in
Nguyen, Luu & Vu, "Evaluation of Perception Robustness in Real-time
Autonomous Drivings" (IEEE T-IV, 2025): a correct detector's output should
mirror exactly when every input is mirrored. We have no labels for the real
competition test set, so ordinary accuracy checks are impossible there -
but self-consistency under a flip doesn't need labels, and it can be run
directly on the real frames rather than a labeled proxy split.

For each sampled frame: build the ego-frame input normally (source), build
it again with a scene-level flip_x applied before voxelization (follow-up,
reusing the exact flip path already in MSigIntermediateFusion's training
augmentation), run the same model on both, then flip the follow-up boxes
back and match them against the source boxes with BEV IoU (Hungarian
algorithm, cost = 1 - IoU, threshold 0.1 - same as the paper). Persistent
misses/spurious detections or large position/yaw drift between the two
indicate a real bug in the export pipeline (coordinate frames, box
conventions, etc.), not a labeling or domain-shift issue - since both runs
use the identical, real, unlabeled input, just mirrored.

Usage:
    .venv/bin/python msig_compat/symmetry_consistency_check.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model \
        --num_frames 150
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Mixed-Signals-Dataset"))

from msig_compat.dataset_patch import apply_all_patches

apply_all_patches()

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.data_utils.augmentor import augment_utils
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou_bev
from msig_compat.export_submission import load_frame, drop_agents_with_no_points_in_range


def run_inference(dataset, model, device, dict_agents_pts_poses, scene_aug):
    # _get_projected_voxel_grid_of_every_agent mutates points in place
    # (apply_se3_), so give it a fresh copy each call - the caller reuses
    # the same dict for both the source and flipped runs.
    dict_agents_pts_poses = {
        agent_name: {'points': agent_data['points'].copy(), 'ego_se3_agent': agent_data['ego_se3_agent']}
        for agent_name, agent_data in dict_agents_pts_poses.items()
    }
    ego_part, agg_points = dataset._get_projected_voxel_grid_of_every_agent(
        dict_agents_pts_poses, scene_aug)
    data = dataset.init_output_of_getitem()
    data['ego'].update(ego_part)
    data['ego']['anchor_box'] = dataset._anchor_box
    data['metadata']['points'] = agg_points
    batch = dataset.collate_batch_test([data])
    batch = train_utils.to_device(batch, device)

    with torch.no_grad():
        output_dict = {'ego': model(batch['ego'])}
    pred_boxes, pred_scores, pred_class, _, _ = dataset.post_process(batch, output_dict)
    return pred_boxes, pred_scores, pred_class


def unflip_boxes(pred_boxes):
    """random_flip_along_x is self-inverse - applying it again undoes it."""
    if pred_boxes.shape[0] == 0:
        return pred_boxes
    boxes_np = pred_boxes.cpu().numpy().copy()
    dummy_points = np.array([[100., 100., 100., 0.]])
    boxes_np, _ = augment_utils.random_flip_along_x(boxes_np, dummy_points, enable=True)
    return torch.from_numpy(boxes_np).to(pred_boxes.device)


def match_and_score(src_boxes, src_scores, src_class, flip_boxes, flip_scores, flip_class,
                    iou_thresh=0.1):
    n_src, n_flip = src_boxes.shape[0], flip_boxes.shape[0]
    if n_src == 0 and n_flip == 0:
        return {'n_matched': 0, 'n_unmatched': 0, 'pos_drift': [], 'yaw_drift': [],
                'score_drift': [], 'class_mismatch': 0}
    if n_src == 0 or n_flip == 0:
        return {'n_matched': 0, 'n_unmatched': n_src + n_flip, 'pos_drift': [],
                'yaw_drift': [], 'score_drift': [], 'class_mismatch': 0}

    iou = boxes_iou_bev(src_boxes, flip_boxes).cpu().numpy()
    cost = 1.0 - iou
    row_idx, col_idx = linear_sum_assignment(cost)

    n_matched = 0
    pos_drift, yaw_drift, score_drift = [], [], []
    class_mismatch = 0
    matched_src, matched_flip = set(), set()
    for r, c in zip(row_idx, col_idx):
        if iou[r, c] < iou_thresh:
            continue
        n_matched += 1
        matched_src.add(r)
        matched_flip.add(c)
        sb = src_boxes[r].cpu().numpy()
        fb = flip_boxes[c].cpu().numpy()
        pos_drift.append(float(np.linalg.norm(sb[:2] - fb[:2])))
        yaw_diff = abs(sb[6] - fb[6])
        yaw_diff = min(yaw_diff, 2 * np.pi - yaw_diff)
        yaw_drift.append(float(yaw_diff))
        score_drift.append(abs(float(src_scores[r]) - float(flip_scores[c])))
        if int(src_class[r]) != int(flip_class[c]):
            class_mismatch += 1

    n_unmatched = (n_src - len(matched_src)) + (n_flip - len(matched_flip))
    return {'n_matched': n_matched, 'n_unmatched': n_unmatched, 'pos_drift': pos_drift,
            'yaw_drift': yaw_drift, 'score_drift': score_drift, 'class_mismatch': class_mismatch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--frames_dir', default=str(_PROJECT_ROOT / "data" / "mixed-signals-test-competition"))
    ap.add_argument('--num_frames', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    opt = ap.parse_args()

    hypes = yaml_utils.load_yaml(str(Path(opt.model_dir) / 'config.yaml'))
    hypes['flag_train'] = False
    dataset = build_dataset(hypes, visualize=False, train=False)
    model = train_utils.create_model(hypes)
    model.cuda()
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()
    device = torch.device('cuda')

    frame_dirs = sorted(p for p in Path(opt.frames_dir).iterdir() if p.is_dir())
    rng = np.random.default_rng(opt.seed)
    sample = rng.choice(frame_dirs, size=min(opt.num_frames, len(frame_dirs)), replace=False)

    total_matched = total_unmatched = total_src = total_flip = 0
    all_pos_drift, all_yaw_drift, all_score_drift = [], [], []
    total_class_mismatch = 0

    for fd in tqdm(sample):
        dpp = load_frame(Path(fd))
        dpp = drop_agents_with_no_points_in_range(dpp, dataset._cav_lidar_range)

        src_boxes, src_scores, src_class = run_inference(dataset, model, device, dpp, scene_aug=None)
        flip_boxes, flip_scores, flip_class = run_inference(
            dataset, model, device, dpp, scene_aug={'flip_x': True, 'flip_y': False})
        flip_boxes_back = unflip_boxes(flip_boxes)

        total_src += src_boxes.shape[0]
        total_flip += flip_boxes.shape[0]

        result = match_and_score(src_boxes, src_scores, src_class,
                                 flip_boxes_back, flip_scores, flip_class)
        total_matched += result['n_matched']
        total_unmatched += result['n_unmatched']
        total_class_mismatch += result['class_mismatch']
        all_pos_drift.extend(result['pos_drift'])
        all_yaw_drift.extend(result['yaw_drift'])
        all_score_drift.extend(result['score_drift'])

    print(f'\nSymmetry consistency check over {len(sample)} real (unlabeled) competition frames')
    print(f'source detections: {total_src} | flipped detections: {total_flip}')
    print(f'matched pairs: {total_matched} | unmatched (missed/spurious): {total_unmatched}')
    if total_matched + total_unmatched > 0:
        match_rate = total_matched / (total_matched + total_unmatched / 2)
        print(f'match rate: {match_rate:.3f}  (1.0 = perfectly self-consistent)')
    if all_pos_drift:
        print(f'position drift (m): mean={np.mean(all_pos_drift):.3f} '
             f'median={np.median(all_pos_drift):.3f} p90={np.percentile(all_pos_drift, 90):.3f} '
             f'max={np.max(all_pos_drift):.3f}')
        print(f'yaw drift (rad): mean={np.mean(all_yaw_drift):.3f} '
             f'median={np.median(all_yaw_drift):.3f} max={np.max(all_yaw_drift):.3f}')
        print(f'score drift: mean={np.mean(all_score_drift):.3f} max={np.max(all_score_drift):.3f}')
        print(f'class mismatches among matched pairs: {total_class_mismatch}')


if __name__ == '__main__':
    main()
