"""
Post-processing-only sweep: run the submitted epoch-55 model ONCE per held-out
frame (the expensive part), cache the raw output_dict, and re-run only
post_process_1cls (score threshold + NMS - cheap, no backbone re-run) for a
grid of (score_threshold, nms_thresh) combos applied uniformly across all 3
classes. Doesn't touch the model or training - purely explores whether the
submitted checkpoint's post-processing config is already close to locally
optimal.

Usage:
    .venv/bin/python msig_compat/threshold_nms_sweep.py \
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

SCORE_THRESHOLDS = [0.10, 0.15, 0.20, 0.30, 0.40]  # baseline = 0.20
NMS_THRESHOLDS = [0.10, 0.15, 0.25, 0.40]           # baseline = 0.15


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

    combos = [(s, n) for s in SCORE_THRESHOLDS for n in NMS_THRESHOLDS]
    stats = {combo: make_result_stat() for combo in combos}
    n_frames = 0

    for idx in tqdm(range(n_total)):
        item = dataset[idx]
        batch = dataset.collate_batch_test([item])
        batch = train_utils.to_device(batch, device)

        with torch.no_grad():
            output_dict = {'ego': model(batch['ego'])}

        gt_boxes_t = batch['metadata']['gt_boxes_in_ego'].to(device=device, dtype=torch.float32)
        gt_class_t = batch['metadata']['gt_class'].to(device=device, dtype=torch.long)
        n_frames += 1

        for score_thresh, nms_thresh in combos:
            all_boxes, all_scores, all_class = [], [], []
            for cls_idx in (1, 2, 3):
                pb, ps = dataset.post_processor.post_process_1cls(
                    cls_idx, batch, output_dict, score_thresh, nms_thresh)
                if isinstance(pb, torch.Tensor) and pb.shape[0] > 0:
                    all_boxes.append(pb)
                    all_scores.append(ps)
                    all_class.append(torch.full((pb.shape[0],), cls_idx, dtype=torch.long, device=device))

            if all_boxes:
                pred_boxes = torch.cat(all_boxes, dim=0)
                pred_scores = torch.cat(all_scores, dim=0)
                pred_class = torch.cat(all_class, dim=0)
            else:
                pred_boxes = torch.zeros((0, 7), device=device)
                pred_scores = torch.zeros((0,), device=device)
                pred_class = torch.zeros((0,), dtype=torch.long, device=device)

            for cls_id in CLS_NAMES:
                mp = pred_class == cls_id
                mg = gt_class_t == cls_id
                match_class(pred_boxes[mp].contiguous(), pred_scores[mp].contiguous(),
                           gt_boxes_t[mg].contiguous(), stats[(score_thresh, nms_thresh)][cls_id])

    print(f'\n{n_frames} frames evaluated\n')
    print(f"{'score_thr':>10} {'nms_thr':>8} | {'car@.5':>7} {'bike@.5':>8} {'ped@.3':>7} | {'mAP':>7}")
    results = []
    for score_thresh, nms_thresh in combos:
        stat = stats[(score_thresh, nms_thresh)]
        ap_car = calculate_ap(stat[1][0.5])
        ap_bike = calculate_ap(stat[2][0.5])
        ap_ped = calculate_ap(stat[3][0.3])
        comp_map = (ap_car + ap_bike + ap_ped) / 3
        results.append((score_thresh, nms_thresh, ap_car, ap_bike, ap_ped, comp_map))
        marker = '  <-- baseline' if (score_thresh, nms_thresh) == (0.20, 0.15) else ''
        print(f"{score_thresh:>10.2f} {nms_thresh:>8.2f} | {ap_car:>7.4f} {ap_bike:>8.4f} {ap_ped:>7.4f} | {comp_map:>7.4f}{marker}")

    best = max(results, key=lambda r: r[5])
    print(f"\nBest combo: score_threshold={best[0]}, nms_thresh={best[1]}, mAP={best[5]:.4f}")


if __name__ == '__main__':
    main()
