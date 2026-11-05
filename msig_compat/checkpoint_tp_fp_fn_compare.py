"""
Fast TP/FP/FN@threshold breakdown across checkpoints 55/56/57, both
Full DEV and RSU-only, on the same 300-frame slice already used for the
mAP comparison. Per-class (car@0.5, bike@0.5, ped@0.3) plus vehicle
precision/recall highlighted.
"""
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
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou_bev

CLS_THR = {1: 0.5, 2: 0.5, 3: 0.3}
CLS_NAMES = {1: 'car', 2: 'bike', 3: 'ped'}
LIMIT = 300

CHECKPOINTS = [
    ('55 (baseline)', 'Mixed-Signals-Dataset/opencood/logs/final_model', None),
    ('56 (1 epoch)', 'Mixed-Signals-Dataset/opencood/logs/dropout030_finetune', 'net_epoch56.pth'),
    ('57 (2 epochs)', 'Mixed-Signals-Dataset/opencood/logs/dropout030_finetune', 'net_epoch57.pth'),
]


def load_model(model_dir, ckpt):
    hypes = yaml_utils.load_yaml(str(Path(model_dir) / 'config.yaml'))
    hypes['flag_train'] = False
    model = train_utils.create_model(hypes)
    model.cuda()
    if ckpt:
        state = torch.load(str(Path(model_dir) / ckpt), map_location='cuda')
        model.load_state_dict(state, strict=True)
    else:
        _, model = train_utils.load_saved_model(model_dir, model)
    model.eval()
    return model, hypes


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
    pb, ps, pc, _, _ = dataset.post_process(batch, output_dict)
    return pb, ps, pc


def tp_count(pred_boxes, pred_scores, gt_boxes, thr):
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return 0
    order = torch.argsort(pred_scores, descending=True)
    iou = boxes_iou_bev(pred_boxes, gt_boxes).cpu().numpy()
    matched = np.zeros(gt_boxes.shape[0], dtype=bool)
    tp = 0
    for i in order.cpu().numpy():
        row = iou[i].copy()
        row[matched] = -1
        j = int(np.argmax(row))
        if row[j] >= thr:
            matched[j] = True
            tp += 1
    return tp


def fresh_copy(d):
    return {k: {'points': v['points'].copy(), 'ego_se3_agent': v['ego_se3_agent']} for k, v in d.items()}


def main():
    dataset = None
    results = {}

    for label, model_dir, ckpt in CHECKPOINTS:
        model, hypes = load_model(model_dir, ckpt)
        if dataset is None:
            dataset = build_dataset(hypes, visualize=False, train=False)
        device = torch.device('cuda')

        for cfg_name in ('full', 'rsu_only'):
            counts = {c: {'tp': 0, 'fp': 0, 'fn': 0, 'n_pred': 0, 'n_gt': 0} for c in CLS_NAMES}

            for idx in tqdm(range(LIMIT), desc=f'{label} {cfg_name}'):
                seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
                idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
                agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
                if cfg_name == 'rsu_only':
                    subset = fresh_copy({k: v for k, v in agents_full.items() if k in ('top', 'dome')})
                else:
                    subset = fresh_copy(agents_full)

                gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
                gt_boxes, gt_class = dataset.preprocess_gt_boxes(
                    gt_info_dict['gt_boxes_in_top'], gt_info_dict['gt_class'], gt_info_dict['gt_visibility'],
                    [dataset.convert_agent_name_to_indicator(a) for a in subset.keys()],
                )
                gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
                gt_class_t = torch.tensor(gt_class, dtype=torch.long, device=device)

                pb, ps, pc = run_inference(dataset, model, device, subset)

                for cls_id, thr in CLS_THR.items():
                    mp = pc == cls_id
                    mg = gt_class_t == cls_id
                    pred_c, score_c, gt_c = pb[mp], ps[mp], gt_boxes_t[mg]
                    tp = tp_count(pred_c, score_c, gt_c, thr)
                    counts[cls_id]['tp'] += tp
                    counts[cls_id]['fp'] += pred_c.shape[0] - tp
                    counts[cls_id]['fn'] += gt_c.shape[0] - tp
                    counts[cls_id]['n_pred'] += pred_c.shape[0]
                    counts[cls_id]['n_gt'] += gt_c.shape[0]

            results[(label, cfg_name)] = counts
        del model
        torch.cuda.empty_cache()

    print(f"\n{'Checkpoint':<16} {'Config':<10} {'Class':<6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Recall':>7}")
    for label, model_dir, ckpt in CHECKPOINTS:
        for cfg_name in ('full', 'rsu_only'):
            counts = results[(label, cfg_name)]
            for cls_id, cname in CLS_NAMES.items():
                c = counts[cls_id]
                prec = c['tp'] / (c['tp'] + c['fp']) if (c['tp'] + c['fp']) else float('nan')
                rec = c['tp'] / c['n_gt'] if c['n_gt'] else float('nan')
                print(f"{label:<16} {cfg_name:<10} {cname:<6} {c['tp']:>6} {c['fp']:>6} {c['fn']:>6} {prec:>7.4f} {rec:>7.4f}")
        print()


if __name__ == '__main__':
    main()
