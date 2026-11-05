"""
Diagnostic: measure how much local mAP changes if a CAV's ego_se3_agent is
built from the RAW (pre-z-height-correction) odometry pose instead of the
devkit-corrected one, while its points keep the normal -3.25 z shift -
the exact mismatch export_submission.py's CAV branch would have if
transforms.json ships the raw pose (see the CAV branch of load_frame() and
its --no_cav_z_shift flag).

This runs on the held-out labeled split, so real ground truth is available -
it measures how much this class of error can hurt accuracy, in a setting we
can verify. It does NOT tell us whether the real competition transforms.json
actually has this issue - that can only be settled by a real submission,
since the two pipelines diverge exactly at the point this script is probing
and there's no ground truth for the real test frames.

Usage:
    .venv/bin/python msig_compat/diagnose_cav_z_mismatch.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model \
        [--limit 300]
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
from mixedsignals.utils.geometry import make_se3

from msig_compat.eval_correct_bev_iou import (
    CLS_NAMES, IOU_THRESHOLDS, make_result_stat, match_class, calculate_ap,
)


def raw_map_se3_agent(dataset, seq_idx, agent_name, timestamp):
    """The CAV's odometry pose *before* SequenceOdomAgent's z-height
    correction (mixed_signals_utils.py:134-146) - what transforms.json would
    hold if it ships the uncorrected pose."""
    oa = dataset._msig.odom_generators[seq_idx][agent_name]
    idx = oa._find_indices_odom(timestamp)
    return make_se3(oa.all_position[idx], quaternion=oa.all_orientation[idx])


def build_agents_dict(dataset, seq_idx, idx_in_seq, broken: bool):
    avail_cavs = dataset._msig.return_name_cavs_in_seq(seq_idx)
    avail_agents = list(dataset._name_rsu) + avail_cavs
    out = {}
    has_cav = False
    for agent_name in avail_agents:
        agent_pc, agent_ts = dataset._msig.return_agent_point_cloud(seq_idx, agent_name, idx_in_seq)
        if agent_name in dataset._name_rsu:
            map_se3_agent = dataset._msig.return_map_se3_agent(seq_idx, agent_name, agent_ts)
        else:
            has_cav = True
            map_se3_agent = raw_map_se3_agent(dataset, seq_idx, agent_name, agent_ts) if broken \
                else dataset._msig.return_map_se3_agent(seq_idx, agent_name, agent_ts)
        ego_se3_agent = dataset.ego_se3_map @ map_se3_agent
        if dataset._is_agent_in_comm_range(ego_se3_agent[:3, -1]):
            out[agent_name] = {'points': agent_pc, 'ego_se3_agent': ego_se3_agent}
    return out, has_cav


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
    pred_boxes, pred_scores, pred_class, _, _ = dataset.post_process(batch, output_dict)
    return pred_boxes, pred_scores, pred_class


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

    stat_correct = make_result_stat()
    stat_broken = make_result_stat()
    n_cav_frames = 0
    n_frames = 0

    for idx in tqdm(range(n_total)):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)

        agents_correct, has_cav = build_agents_dict(dataset, seq_idx, idx_in_seq, broken=False)
        if len(agents_correct) < 2:
            continue
        n_frames += 1

        gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        gt_boxes, gt_class = dataset.preprocess_gt_boxes(
            gt_info_dict['gt_boxes_in_top'],
            gt_info_dict['gt_class'],
            gt_info_dict['gt_visibility'],
            [dataset.convert_agent_name_to_indicator(a) for a in agents_correct.keys()],
        )
        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
        gt_class_t = torch.tensor(gt_class, dtype=torch.long, device=device)

        pb, ps, pc = run_inference(dataset, model, device, agents_correct)
        for cls_id in CLS_NAMES:
            mp = pc == cls_id
            mg = gt_class_t == cls_id
            match_class(pb[mp].contiguous(), ps[mp].contiguous(), gt_boxes_t[mg].contiguous(), stat_correct[cls_id])

        if not has_cav:
            # no CAV in range - broken variant is identical to correct, skip the extra forward pass
            for cls_id in CLS_NAMES:
                mp = pc == cls_id
                mg = gt_class_t == cls_id
                match_class(pb[mp].contiguous(), ps[mp].contiguous(), gt_boxes_t[mg].contiguous(), stat_broken[cls_id])
            continue

        n_cav_frames += 1
        agents_broken, _ = build_agents_dict(dataset, seq_idx, idx_in_seq, broken=True)
        pb2, ps2, pc2 = run_inference(dataset, model, device, agents_broken)
        for cls_id in CLS_NAMES:
            mp = pc2 == cls_id
            mg = gt_class_t == cls_id
            match_class(pb2[mp].contiguous(), ps2[mp].contiguous(), gt_boxes_t[mg].contiguous(), stat_broken[cls_id])

    print(f'\n{n_frames} frames evaluated, {n_cav_frames} had a CAV in range (only those differ between variants)\n')

    for label, stat in (('CORRECT (devkit-consistent CAV pose)', stat_correct),
                        ('BROKEN (raw CAV pose + shifted points, simulated mismatch)', stat_broken)):
        aps = {t: {c: calculate_ap(stat[c][t]) for c in CLS_NAMES} for t in IOU_THRESHOLDS}
        print(label)
        for cls_id, cls_name in CLS_NAMES.items():
            print(f'  {cls_name:>10} | AP@0.3 {aps[0.3][cls_id]:.4f} | AP@0.5 {aps[0.5][cls_id]:.4f} | AP@0.7 {aps[0.7][cls_id]:.4f}')
        comp_map = (aps[0.5][1] + aps[0.5][2] + aps[0.3][3]) / 3
        print(f'  Competition mAP = {comp_map:.4f}\n')


if __name__ == '__main__':
    main()
