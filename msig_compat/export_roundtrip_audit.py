"""
Forensic audit of the submission serialization contract. On labeled DEV
frames (so we have ground truth to score against), for each frame:

  1. Run the model, get in-memory predictions (x,y,z,L,W,H,yaw,score,class).
  2. Write them out using the EXACT same formatting export_submission.py
     uses for the real submission.zip.
  3. Read the text back and re-parse into box arrays - the reverse of what
     a scorer consuming submission.zip would do.
  4. Compare before vs after: max abs diff per field. Should be ~0 (bounded
     only by the %.4f print precision), or a formatting bug is real.
  5. Run the SAME local evaluator (match_class/calculate_ap from
     eval_correct_bev_iou.py) on the round-tripped (re-parsed-from-text)
     predictions instead of the in-memory ones, and compare the resulting
     mAP against the established baseline (0.8450) - if the round trip is
     lossless, these must match almost exactly.

Usage:
    .venv/bin/python msig_compat/export_roundtrip_audit.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model \
        [--limit 500]
"""
import argparse
import sys
import zipfile
import io
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

from msig_compat.eval_correct_bev_iou import (
    CLS_NAMES, IOU_THRESHOLDS, make_result_stat, match_class, calculate_ap,
)


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
    return pb.cpu().numpy(), ps.cpu().numpy(), pc.cpu().numpy()


def write_submission_format(pred_boxes, pred_scores, pred_class):
    """EXACT copy of export_submission.py's write logic - cls first, then
    cx cy cz dx dy dz yaw score, matching the real submission.zip contract."""
    lines = []
    for box, score, cls in zip(pred_boxes, pred_scores, pred_class):
        cx, cy, cz, dx, dy, dz, yaw = box
        lines.append(f"{int(cls)} {cx:.4f} {cy:.4f} {cz:.4f} "
                     f"{dx:.4f} {dy:.4f} {dz:.4f} {yaw:.4f} {score:.4f}")
    return "\n".join(lines)


def read_submission_format(text):
    """Reverse of the above - what a scorer consuming submission.zip does."""
    boxes, scores, classes = [], [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        cx, cy, cz, dx, dy, dz, yaw, score = (float(x) for x in parts[1:9])
        boxes.append([cx, cy, cz, dx, dy, dz, yaw])
        scores.append(score)
        classes.append(cls)
    if not boxes:
        return np.zeros((0, 7)), np.zeros((0,)), np.zeros((0,), dtype=int)
    return np.array(boxes), np.array(scores), np.array(classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--limit', type=int, default=500)
    opt = ap.parse_args()

    hypes = yaml_utils.load_yaml(str(Path(opt.model_dir) / 'config.yaml'))
    hypes['flag_train'] = False
    dataset = build_dataset(hypes, visualize=False, train=False)

    model = train_utils.create_model(hypes)
    model.cuda()
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()
    device = torch.device('cuda')

    n_total = min(len(dataset), opt.limit) if opt.limit else len(dataset)

    max_diff = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'L': 0.0, 'W': 0.0, 'H': 0.0, 'yaw': 0.0, 'score': 0.0}
    class_mismatches = 0
    count_mismatches = 0
    total_boxes = 0

    stat_direct = make_result_stat()
    stat_roundtrip = make_result_stat()

    out_zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(out_zip_buf, 'w', zipfile.ZIP_DEFLATED)

    for idx in tqdm(range(n_total)):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)

        pb, ps, pc = run_inference(dataset, model, device, agents_full)

        # --- write out exactly like export_submission.py, then read back ---
        text = write_submission_format(pb, ps, pc)
        zf.writestr(f"frame_{idx}.txt", text)
        pb2, ps2, pc2 = read_submission_format(text)

        n = pb.shape[0]
        total_boxes += n
        if pb2.shape[0] != n:
            count_mismatches += 1
        else:
            if n > 0:
                max_diff['x'] = max(max_diff['x'], np.abs(pb[:, 0] - pb2[:, 0]).max())
                max_diff['y'] = max(max_diff['y'], np.abs(pb[:, 1] - pb2[:, 1]).max())
                max_diff['z'] = max(max_diff['z'], np.abs(pb[:, 2] - pb2[:, 2]).max())
                max_diff['L'] = max(max_diff['L'], np.abs(pb[:, 3] - pb2[:, 3]).max())
                max_diff['W'] = max(max_diff['W'], np.abs(pb[:, 4] - pb2[:, 4]).max())
                max_diff['H'] = max(max_diff['H'], np.abs(pb[:, 5] - pb2[:, 5]).max())
                yaw_diff = np.abs(pb[:, 6] - pb2[:, 6])
                max_diff['yaw'] = max(max_diff['yaw'], yaw_diff.max())
                max_diff['score'] = max(max_diff['score'], np.abs(ps - ps2).max())
                if not np.array_equal(pc.astype(int), pc2.astype(int)):
                    class_mismatches += 1

        # --- score both the direct in-memory predictions and the
        # round-tripped (re-parsed-from-text) predictions against the SAME GT ---
        gt_info_dict = dataset.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        gt_boxes, gt_class = dataset.preprocess_gt_boxes(
            gt_info_dict['gt_boxes_in_top'], gt_info_dict['gt_class'], gt_info_dict['gt_visibility'],
            [dataset.convert_agent_name_to_indicator(a) for a in agents_full.keys()],
        )
        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
        gt_class_t = torch.tensor(gt_class, dtype=torch.long, device=device)

        pb_t, ps_t, pc_t = torch.tensor(pb, dtype=torch.float32, device=device), \
                            torch.tensor(ps, dtype=torch.float32, device=device), \
                            torch.tensor(pc, dtype=torch.long, device=device)
        pb2_t, ps2_t, pc2_t = torch.tensor(pb2, dtype=torch.float32, device=device), \
                               torch.tensor(ps2, dtype=torch.float32, device=device), \
                               torch.tensor(pc2, dtype=torch.long, device=device)

        for cls_id in CLS_NAMES:
            mg = gt_class_t == cls_id
            gb = gt_boxes_t[mg].contiguous()

            m1 = pc_t == cls_id
            match_class(pb_t[m1].contiguous(), ps_t[m1].contiguous(), gb, stat_direct[cls_id])

            m2 = pc2_t == cls_id
            match_class(pb2_t[m2].contiguous(), ps2_t[m2].contiguous(), gb, stat_roundtrip[cls_id])

    zf.close()

    print(f"\n=== Round-trip field comparison ({n_total} frames, {total_boxes} boxes) ===")
    for k, v in max_diff.items():
        flag = "  <-- SUSPICIOUS" if v > 0.001 else ""
        print(f"  max |{k} - {k}'| = {v:.6f}{flag}")
    print(f"  frames with box-count mismatch: {count_mismatches}")
    print(f"  frames with class-array mismatch: {class_mismatches}")

    def comp_map(stat):
        ap_car = calculate_ap(stat[1][0.5])
        ap_bike = calculate_ap(stat[2][0.5])
        ap_ped = calculate_ap(stat[3][0.3])
        return (ap_car + ap_bike + ap_ped) / 3, ap_car, ap_bike, ap_ped

    m_direct, c_direct, b_direct, p_direct = comp_map(stat_direct)
    m_rt, c_rt, b_rt, p_rt = comp_map(stat_roundtrip)

    print(f"\n=== Scoring comparison ===")
    print(f"  DIRECT (in-memory predictions):      car={c_direct:.4f} bike={b_direct:.4f} ped={p_direct:.4f}  mAP={m_direct:.4f}")
    print(f"  ROUND-TRIPPED (export -> re-parsed):  car={c_rt:.4f} bike={b_rt:.4f} ped={p_rt:.4f}  mAP={m_rt:.4f}")
    print(f"  mAP diff = {abs(m_direct - m_rt):.6f}")


if __name__ == '__main__':
    main()
