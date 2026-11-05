"""
Re-run AP evaluation with a corrected BEV IoU.

Mixed-Signals-Dataset/opencood/utils/eval_utils.py's caluclate_tp_fp_gpu calls
iou3d_nms_cuda.boxes_overlap_bev_gpu directly and compares its output against
the IoU threshold (0.3 / 0.5 / 0.7). That kernel returns the raw BEV
intersection AREA in square metres, not a normalized [0, 1] IoU - so for car
and bike (large boxes, ~10 m^2 and ~1 m^2), almost any real overlap already
exceeds 0.7 and the "AP@0.3/0.5/0.7" columns in eval.yaml come out identical
or nearly so. This script is the same TP/FP/AP procedure
(opencood.utils.eval_utils.voc_ap, unmodified) but matches with
opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils.boxes_iou_bev, which divides
by the union and returns a real IoU - confirmed empirically to land in
[0, 1] against the same predictions. This is a read-only diagnostic; it
doesn't touch the provided starter kit.

Second fix: frames with zero ground-truth boxes used to be skipped outright
(`continue` before any matching happened), so any predictions the model made
on an empty frame were never counted as false positives - a real and
significant hole, since VOC-style AP is sensitive to every FP in the corpus,
and the loop never noticed how many frames it was throwing away. Empty-GT
frames are now kept and scored normally: match_class already treats
gt_boxes.shape[0] == 0 as "everything predicted here is a false positive",
the bug was that the frame never reached match_class at all.

Usage:
    .venv/bin/python msig_compat/eval_correct_bev_iou.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model \
        [--ckpt net_best.pth]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Mixed-Signals-Dataset"))

from msig_compat.dataset_patch import apply_all_patches

apply_all_patches()

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils, inference_utils
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou_bev
from opencood.utils.eval_utils import voc_ap

CLS_NAMES = {1: 'car', 2: 'bike', 3: 'pedestrian'}
IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def make_result_stat():
    return {cls: {t: {'tp': [], 'fp': [], 'score': [], 'gt': 0} for t in IOU_THRESHOLDS}
            for cls in CLS_NAMES}


def match_class(pred_boxes, pred_scores, gt_boxes, result_stat_cls):
    for thresh in IOU_THRESHOLDS:
        result_stat_cls[thresh]['gt'] += gt_boxes.shape[0]

    if pred_boxes.shape[0] == 0:
        return

    sorted_scores, sorted_idx = torch.sort(pred_scores, descending=True)
    sorted_scores = sorted_scores.cpu().numpy().tolist()

    if gt_boxes.shape[0] == 0:
        for thresh in IOU_THRESHOLDS:
            result_stat_cls[thresh]['fp'] += [1] * pred_boxes.shape[0]
            result_stat_cls[thresh]['tp'] += [0] * pred_boxes.shape[0]
            result_stat_cls[thresh]['score'] += sorted_scores
        return

    iou = boxes_iou_bev(pred_boxes, gt_boxes)  # (N_pred, N_gt), real [0,1] IoU
    iou = iou[sorted_idx].cpu().numpy()  # score-descending order

    for thresh in IOU_THRESHOLDS:
        matched_gt = np.zeros(gt_boxes.shape[0], dtype=bool)
        tp, fp = [], []
        for row in iou:
            order = np.argsort(-row)
            hit = False
            for gt_idx in order:
                if row[gt_idx] < thresh:
                    break
                if not matched_gt[gt_idx]:
                    matched_gt[gt_idx] = True
                    hit = True
                    break
            tp.append(1 if hit else 0)
            fp.append(0 if hit else 1)
        result_stat_cls[thresh]['tp'] += tp
        result_stat_cls[thresh]['fp'] += fp
        result_stat_cls[thresh]['score'] += sorted_scores


def calculate_ap(stat):
    gt_total = stat['gt']
    if gt_total == 0 or len(stat['score']) == 0:
        return 0.0

    fp = np.array(stat['fp'])
    tp = np.array(stat['tp'])
    score = np.array(stat['score'])
    order = np.argsort(-score, kind='stable')
    fp, tp = fp[order], tp[order]

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = (tp / gt_total).tolist()
    prec = (tp / np.maximum(tp + fp, 1e-12)).tolist()
    ap, _, _ = voc_ap(rec, prec)
    return ap


def main():
    ap_parser = argparse.ArgumentParser()
    ap_parser.add_argument('--model_dir', required=True)
    ap_parser.add_argument('--ckpt', default='',
                           help="checkpoint filename inside --model_dir (or an absolute "
                                "path), e.g. net_best.pth - overrides the epoch load_saved_model "
                                "would otherwise pick")
    ap_parser.add_argument('--limit', type=int, default=None)
    ap_parser.add_argument('--force_rsu_only', action='store_true',
                           help="drop every CAV, evaluate as if only the RSU pair were "
                                "in range - a stress test for the case that dominates the "
                                "competition's own test set (see export_submission.py)")
    opt = ap_parser.parse_args()

    hypes = yaml_utils.load_yaml(str(Path(opt.model_dir) / 'config.yaml'))
    hypes['flag_train'] = False
    ds = build_dataset(hypes, visualize=False, train=False)
    if opt.force_rsu_only:
        ds.force_rsu_only = True
        print('EVAL INFO | force_rsu_only=True - every sample dropped to just the RSU pair')
    model = train_utils.create_model(hypes)
    model.cuda()
    if opt.ckpt:
        ckpt_path = Path(opt.ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(opt.model_dir) / ckpt_path
        print(f'EVAL INFO | loading explicit checkpoint: {ckpt_path}')
        state = torch.load(str(ckpt_path), map_location='cuda')
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    else:
        _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()
    device = torch.device('cuda')

    loader = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=ds.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)

    result_stat = make_result_stat()
    n_frames = 0
    n_empty_gt_frames = 0
    total = min(len(ds), opt.limit) if opt.limit else len(ds)
    for i, batch_data in enumerate(tqdm(loader, total=total)):
        if opt.limit and i >= opt.limit:
            break
        batch_data = train_utils.to_device(batch_data, device)
        if len(batch_data.keys()) == 1:
            continue
        pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
            inference_utils.inference_intermediate_fusion(batch_data, model, ds, None)
        if pred_box_tensor is None:
            pred_box_tensor = torch.zeros(0, 7, device=device)
        if pred_score is None:
            pred_score = torch.zeros(0, device=device)
        if pred_class is None:
            pred_class = torch.zeros(0, dtype=torch.long, device=device)
        if gt_box_tensor is None:
            gt_box_tensor = torch.zeros(0, 7, device=device)
        if gt_class is None:
            gt_class = torch.zeros(0, dtype=torch.long, device=device)
        n_frames += 1
        if gt_box_tensor.shape[0] == 0:
            n_empty_gt_frames += 1

        for cls_id, cls_name in CLS_NAMES.items():
            mask_pred = pred_class == cls_id
            mask_gt = gt_class == cls_id
            pb = pred_box_tensor[mask_pred]
            gb = gt_box_tensor[mask_gt]
            sc = pred_score[mask_pred]
            match_class(pb, sc, gb, result_stat[cls_id])

    print(f'\nEvaluated {n_frames} frames with corrected BEV IoU')
    print(f'Frames with zero ground-truth boxes (all predictions there count as FP): {n_empty_gt_frames}\n')
    per_class_ap = {}
    for cls_id, cls_name in CLS_NAMES.items():
        aps = {t: calculate_ap(result_stat[cls_id][t]) for t in IOU_THRESHOLDS}
        per_class_ap[cls_name] = aps
        print(f'{cls_name:>10} | AP@0.3 {aps[0.3]:.4f} | AP@0.5 {aps[0.5]:.4f} | AP@0.7 {aps[0.7]:.4f}')

    comp_map = (per_class_ap['car'][0.5] + per_class_ap['bike'][0.5] + per_class_ap['pedestrian'][0.3]) / 3
    print(f'\nCompetition mAP = mean(Vehicle AP@0.5, Bike AP@0.5, Pedestrian AP@0.3) = {comp_map:.4f}')


if __name__ == '__main__':
    main()
