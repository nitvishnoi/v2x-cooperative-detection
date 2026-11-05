"""
Isolated 004 test: on the exact same frame set (every frame where laser,
003, AND 004 are all present and in range), compare:

    TOP+DOME+laser+003          (004 excluded)
    TOP+DOME+laser+003+004      (004 included)

Both configs require the same three CAVs to be present, so this is a clean
matched comparison differing only in whether 004 is fed to the fusion -
unlike the broader ablation's "all CAVs" number, which mixed frames with
1, 2, or 3 CAVs present depending on what each frame actually had.

Usage:
    .venv/bin/python msig_compat/ablation_004_isolated.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model \
        [--limit 800]
"""
import argparse
import sys
from pathlib import Path

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

from msig_compat.eval_correct_bev_iou import (
    CLS_NAMES, IOU_THRESHOLDS, make_result_stat, match_class, calculate_ap,
)
from msig_compat.export_submission import drop_agents_with_no_points_in_range

REQUIRED = {'laser', '003', '004'}
CONFIGS = {
    'TOP+DOME+laser+003 (004 excluded)': {'top', 'dome', 'laser', '003'},
    'TOP+DOME+laser+003+004 (004 included)': {'top', 'dome', 'laser', '003', '004'},
}


def fresh_copy(dict_agents_pts_poses):
    """apply_se3_ mutates points in place - each config must get its own
    copy, since the same agent can appear in both configs for a frame and
    would otherwise be transformed twice."""
    return {k: {'points': v['points'].copy(), 'ego_se3_agent': v['ego_se3_agent']}
           for k, v in dict_agents_pts_poses.items()}


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

    n_total = len(dataset)
    if opt.limit:
        n_total = min(n_total, opt.limit)

    stats = {name: make_result_stat() for name in CONFIGS}
    n_frames = 0

    for idx in tqdm(range(n_total)):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)

        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
        present = set(agents_full.keys())
        if not REQUIRED.issubset(present):
            continue

        gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)

        results_this_frame = {}
        frame_ok = True
        for cfg_name, keep_set in CONFIGS.items():
            subset = fresh_copy({k: v for k, v in agents_full.items() if k in keep_set})
            subset = drop_agents_with_no_points_in_range(subset, dataset._cav_lidar_range)
            # this config's intended CAVs must all have survived the
            # zero-points-in-range filter, or the two configs stop being a
            # matched comparison (missing a CAV that was supposed to be there)
            intended_cavs = keep_set & REQUIRED
            if 'top' not in subset or not intended_cavs.issubset(set(subset.keys())):
                frame_ok = False
                break
            results_this_frame[cfg_name] = subset

        if not frame_ok:
            continue

        gt_boxes, gt_class = dataset.preprocess_gt_boxes(
            gt_info_dict['gt_boxes_in_top'],
            gt_info_dict['gt_class'],
            gt_info_dict['gt_visibility'],
            [dataset.convert_agent_name_to_indicator(a) for a in agents_full.keys()],
        )
        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
        gt_class_t = torch.tensor(gt_class, dtype=torch.long, device=device)

        n_frames += 1
        for cfg_name, subset in results_this_frame.items():
            pb, ps, pc, _, _ = run_inference(dataset, model, device, subset)
            for cls_id in CLS_NAMES:
                mp = pc == cls_id
                mg = gt_class_t == cls_id
                match_class(pb[mp].contiguous(), ps[mp].contiguous(),
                           gt_boxes_t[mg].contiguous(), stats[cfg_name][cls_id])

    print(f'\n{n_frames} frames had laser+003+004 all present and in range\n')
    for cfg_name in CONFIGS:
        stat = stats[cfg_name]
        aps = {t: {c: calculate_ap(stat[c][t]) for c in CLS_NAMES} for t in IOU_THRESHOLDS}
        print(cfg_name)
        for cls_id, cls_name in CLS_NAMES.items():
            print(f'  {cls_name:>10} | AP@0.3 {aps[0.3][cls_id]:.4f} | AP@0.5 {aps[0.5][cls_id]:.4f} | AP@0.7 {aps[0.7][cls_id]:.4f}')
        comp_map = (aps[0.5][1] + aps[0.5][2] + aps[0.3][3]) / 3
        print(f'  Competition mAP = {comp_map:.4f}\n')


if __name__ == '__main__':
    main()
