"""
Agent-contribution ablation: for the submitted epoch-55 checkpoint, measure
per-class AP on the held-out labeled split under different agent subsets -
TOP alone, TOP+DOME (RSU pair, no CAV), TOP+DOME+ one CAV at a time, and
TOP+DOME+ all CAVs present (the normal/baseline configuration) - to see
whether the model is over-relying on a particular sensor, and how much each
CAV actually contributes once it's in range.

Each CAV-specific config is only scored on frames where that CAV is actually
present and in range - a frame without "003" in range can't test the
"TOP+DOME+003" config, so frame counts differ per config and are reported
alongside the AP numbers.

Usage:
    .venv/bin/python msig_compat/agent_ablation_study.py \
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

CAV_NAMES = ('laser', '003', '004')

# name -> (required agents that must be present in the raw frame, agents to
# actually keep once required agents are confirmed present)
CONFIGS = {
    'TOP only':              (set(),                 {'top'}),
    'TOP+DOME':              (set(),                 {'top', 'dome'}),
    'TOP+DOME+laser':        ({'laser'},             {'top', 'dome', 'laser'}),
    'TOP+DOME+003':          ({'003'},               {'top', 'dome', '003'}),
    'TOP+DOME+004':          ({'004'},               {'top', 'dome', '004'}),
    'TOP+DOME+all CAVs':     (set(),                 None),  # None = keep everything available
}


def fresh_copy(dict_agents_pts_poses):
    """apply_se3_ mutates points in place - each config must get its own
    copy, since the same agent (e.g. dome) can appear in several configs
    for the same frame and would otherwise be transformed multiple times."""
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
    n_frames = {name: 0 for name in CONFIGS}

    for idx in tqdm(range(n_total)):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)

        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
        present = set(agents_full.keys())

        gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)

        for cfg_name, (required, keep) in CONFIGS.items():
            if not required.issubset(present):
                continue
            keep_set = keep if keep is not None else present
            subset = fresh_copy({k: v for k, v in agents_full.items() if k in keep_set})
            subset = drop_agents_with_no_points_in_range(subset, dataset._cav_lidar_range)
            if len(subset) < 1 or 'top' not in subset:
                continue

            gt_boxes, gt_class = dataset.preprocess_gt_boxes(
                gt_info_dict['gt_boxes_in_top'],
                gt_info_dict['gt_class'],
                gt_info_dict['gt_visibility'],
                [dataset.convert_agent_name_to_indicator(a) for a in subset.keys()],
            )
            gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
            gt_class_t = torch.tensor(gt_class, dtype=torch.long, device=device)

            pb, ps, pc, _, _ = run_inference(dataset, model, device, subset)
            n_frames[cfg_name] += 1
            for cls_id in CLS_NAMES:
                mp = pc == cls_id
                mg = gt_class_t == cls_id
                match_class(pb[mp].contiguous(), ps[mp].contiguous(),
                           gt_boxes_t[mg].contiguous(), stats[cfg_name][cls_id])

    print()
    for cfg_name in CONFIGS:
        stat = stats[cfg_name]
        aps = {t: {c: calculate_ap(stat[c][t]) for c in CLS_NAMES} for t in IOU_THRESHOLDS}
        print(f'{cfg_name}  (n={n_frames[cfg_name]} frames)')
        for cls_id, cls_name in CLS_NAMES.items():
            print(f'  {cls_name:>10} | AP@0.3 {aps[0.3][cls_id]:.4f} | AP@0.5 {aps[0.5][cls_id]:.4f} | AP@0.7 {aps[0.7][cls_id]:.4f}')
        comp_map = (aps[0.5][1] + aps[0.5][2] + aps[0.3][3]) / 3
        print(f'  Competition mAP = {comp_map:.4f}\n')


if __name__ == '__main__':
    main()
