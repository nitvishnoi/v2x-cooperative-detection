"""
Class-specific score threshold sweep, epoch-55 model, on DEV (natural
composition). Single forward pass per frame (cached), cheap post-process
sweep across a small per-class threshold grid. NMS held at baseline
(0.15/0.15/0.15) for this pass - isolates the threshold axis first.
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

CAR_THR = [0.15, 0.20, 0.25]
BIKE_THR = [0.15, 0.20]
PED_THR = [0.15, 0.20]
NMS_BASELINE = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--limit', type=int, default=300)
    opt = ap.parse_args()

    hypes = yaml_utils.load_yaml(str(Path(opt.model_dir) / 'config.yaml'))
    hypes['flag_train'] = False
    dataset = build_dataset(hypes, visualize=False, train=False)

    model = train_utils.create_model(hypes)
    model.cuda()
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()
    device = torch.device('cuda')

    combos = [(c, b, p) for c in CAR_THR for b in BIKE_THR for p in PED_THR]
    stats = {combo: make_result_stat() for combo in combos}
    n_frames = 0

    for idx in tqdm(range(opt.limit)):
        item = dataset[idx]
        batch = dataset.collate_batch_test([item])
        batch = train_utils.to_device(batch, device)
        with torch.no_grad():
            output_dict = {'ego': model(batch['ego'])}

        gt_boxes_t = batch['metadata']['gt_boxes_in_ego'].to(device=device, dtype=torch.float32)
        gt_class_t = batch['metadata']['gt_class'].to(device=device, dtype=torch.long)
        n_frames += 1

        for car_thr, bike_thr, ped_thr in combos:
            thr_map = {1: car_thr, 2: bike_thr, 3: ped_thr}
            all_boxes, all_scores, all_class = [], [], []
            for cls_id in (1, 2, 3):
                pb, ps = dataset.post_processor.post_process_1cls(
                    cls_id, batch, output_dict, thr_map[cls_id], NMS_BASELINE)
                if isinstance(pb, torch.Tensor) and pb.shape[0] > 0:
                    all_boxes.append(pb)
                    all_scores.append(ps)
                    all_class.append(torch.full((pb.shape[0],), cls_id, dtype=torch.long, device=device))
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
                           gt_boxes_t[mg].contiguous(), stats[(car_thr, bike_thr, ped_thr)][cls_id])

    print(f'\n{n_frames} frames evaluated\n')
    print(f"{'car_thr':>8} {'bike_thr':>9} {'ped_thr':>8} | {'car@.5':>7} {'bike@.5':>8} {'ped@.3':>7} | {'mAP':>7}")
    results = []
    for car_thr, bike_thr, ped_thr in combos:
        stat = stats[(car_thr, bike_thr, ped_thr)]
        ap_car = calculate_ap(stat[1][0.5])
        ap_bike = calculate_ap(stat[2][0.5])
        ap_ped = calculate_ap(stat[3][0.3])
        comp_map = (ap_car + ap_bike + ap_ped) / 3
        results.append((car_thr, bike_thr, ped_thr, ap_car, ap_bike, ap_ped, comp_map))
        marker = '  <-- baseline' if (car_thr, bike_thr, ped_thr) == (0.20, 0.20, 0.20) else ''
        print(f"{car_thr:>8.2f} {bike_thr:>9.2f} {ped_thr:>8.2f} | {ap_car:>7.4f} {ap_bike:>8.4f} {ap_ped:>7.4f} | {comp_map:>7.4f}{marker}")

    best = max(results, key=lambda r: r[6])
    print(f"\nBest combo: car={best[0]}, bike={best[1]}, ped={best[2]}, mAP={best[6]:.4f}")


if __name__ == '__main__':
    main()
