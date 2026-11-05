"""
Score calibration of the released checkpoint on DEV: the fraction of
detections that are true positives (precision) in each confidence band, per
class, at the IoU threshold each class is scored at (car and bike 0.5,
pedestrian 0.3). Run it on a model directory whose post-processing
score_threshold is 0.05 to see how reliable the 0.05-0.20 band is.

Also prints the expected calibration error (ECE): the detection-weighted mean
absolute gap between a band's mean confidence and its precision.

Usage:
    .venv/bin/python msig_compat/score_calibration.py --model_dir <run dir with score_threshold 0.05>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Mixed-Signals-Dataset"))

from msig_compat.fusion_baselines import copy_nodes, infer, iterate_dev, load_model_and_data, report  # noqa: E402
from msig_compat.bootstrap_utils import FrameRecorder  # noqa: E402

EDGES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0001)
SCORED_AT = {1: ('car', 0.5), 2: ('bike', 0.5), 3: ('pedestrian', 0.3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--limit', type=int, default=None)
    opt = ap.parse_args()

    hypes, dataset, model, device = load_model_and_data(opt.model_dir)
    print('score_threshold:', hypes['postprocess']['score_threshold'])
    rec = FrameRecorder(['model'])
    for seq_idx, nodes, gt_boxes_t, gt_class_t in iterate_dev(dataset, device, opt.limit):
        rec.add(seq_idx, gt_boxes_t, gt_class_t, {'model': infer(dataset, model, device, copy_nodes(nodes))})

    report(rec, ['model'], 'model', 0, 30)
    for cls, (name, thr) in SCORED_AT.items():
        bins = rec.score_bins('model', cls, thr, EDGES)
        n_all = sum(b[2] for b in bins)
        s = np.concatenate([f['model'][cls][thr][0] for f in rec.frames])
        ece = 0.0
        print(f'\n{name} (IoU {thr}): band | detections | precision | mean score')
        for lo, hi, n, p in bins:
            if n == 0:
                continue
            mean_s = float(s[(s >= lo) & (s < hi)].mean())
            ece += n / n_all * abs(p - mean_s)
            print(f'  {lo:.2f}-{min(hi, 1.0):.2f} | {n:6d} | {p:.3f} | {mean_s:.3f}')
        low = [b for b in bins if b[0] < 0.20]
        n_low = sum(b[2] for b in low)
        p_low = sum(b[2] * b[3] for b in low if b[2]) / max(n_low, 1)
        print(f'  0.05-0.20 band: {n_low} detections ({100 * n_low / max(n_all, 1):.1f}% of all), precision {p_low:.3f}')
        print(f'  ECE {ece:.3f}')


if __name__ == '__main__':
    main()
