"""
Per-held-out-sequence breakdown: is the pooled 84.5% DEV mAP representative
of each of the 6 held-out sequences individually, or does it hide large
scene-to-scene variance? Also splits each sequence into RSU-only vs
CAV-present frames to separate scene difficulty from agent composition.

Usage:
    .venv/bin/python msig_compat/per_sequence_breakdown.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model
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

TEST_SEQUENCES = (31, 32, 33, 34, 35, 37)


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


def comp_map(stat):
    ap_car = calculate_ap(stat[1][0.5])
    ap_bike = calculate_ap(stat[2][0.5])
    ap_ped = calculate_ap(stat[3][0.3])
    return ap_car, ap_bike, ap_ped, (ap_car + ap_bike + ap_ped) / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
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

    per_seq_stat = {s: make_result_stat() for s in TEST_SEQUENCES}
    per_seq_cav_stat = {s: make_result_stat() for s in TEST_SEQUENCES}
    per_seq_rsu_stat = {s: make_result_stat() for s in TEST_SEQUENCES}
    per_seq_n = {s: 0 for s in TEST_SEQUENCES}
    per_seq_n_cav = {s: 0 for s in TEST_SEQUENCES}

    for idx in tqdm(range(n_total)):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)

        has_cav = any(a in ('laser', '003', '004') for a in agents_full.keys())

        gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        gt_boxes, gt_class = dataset.preprocess_gt_boxes(
            gt_info_dict['gt_boxes_in_top'], gt_info_dict['gt_class'], gt_info_dict['gt_visibility'],
            [dataset.convert_agent_name_to_indicator(a) for a in agents_full.keys()],
        )
        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
        gt_class_t = torch.tensor(gt_class, dtype=torch.long, device=device)

        pb, ps, pc, _, _ = run_inference(dataset, model, device, agents_full)

        per_seq_n[seq_idx] += 1
        if has_cav:
            per_seq_n_cav[seq_idx] += 1

        for cls_id in CLS_NAMES:
            mp = pc == cls_id
            mg = gt_class_t == cls_id
            pb_c, ps_c, gb_c = pb[mp].contiguous(), ps[mp].contiguous(), gt_boxes_t[mg].contiguous()
            match_class(pb_c, ps_c, gb_c, per_seq_stat[seq_idx][cls_id])
            if has_cav:
                match_class(pb_c, ps_c, gb_c, per_seq_cav_stat[seq_idx][cls_id])
            else:
                match_class(pb_c, ps_c, gb_c, per_seq_rsu_stat[seq_idx][cls_id])

    print(f"\n{'Seq':>5} {'Frames':>7} {'CAV%':>7} | {'Vehicle@.5':>10} {'Bike@.5':>8} {'Ped@.3':>7} | {'mAP':>7}")
    macro_maps = []
    for s in TEST_SEQUENCES:
        n = per_seq_n[s]
        cav_pct = 100.0 * per_seq_n_cav[s] / n if n else 0.0
        ap_car, ap_bike, ap_ped, m = comp_map(per_seq_stat[s])
        macro_maps.append(m)
        print(f"{s:>5} {n:>7} {cav_pct:>6.1f}% | {ap_car:>10.4f} {ap_bike:>8.4f} {ap_ped:>7.4f} | {m:>7.4f}")

    macro_avg = sum(macro_maps) / len(macro_maps)
    print(f"\nMacro average of 6 sequence mAPs: {macro_avg:.4f}")

    print(f"\n{'Seq':>5} | {'RSU-only mAP (n)':>22} | {'CAV-present mAP (n)':>22}")
    for s in TEST_SEQUENCES:
        n_rsu = per_seq_n[s] - per_seq_n_cav[s]
        n_cav = per_seq_n_cav[s]
        if n_rsu > 0:
            _, _, _, m_rsu = comp_map(per_seq_rsu_stat[s])
            rsu_str = f"{m_rsu:.4f} (n={n_rsu})"
        else:
            rsu_str = "n/a (n=0)"
        if n_cav > 0:
            _, _, _, m_cav = comp_map(per_seq_cav_stat[s])
            cav_str = f"{m_cav:.4f} (n={n_cav})"
        else:
            cav_str = "n/a (n=0)"
        print(f"{s:>5} | {rsu_str:>22} | {cav_str:>22}")


if __name__ == '__main__':
    main()
