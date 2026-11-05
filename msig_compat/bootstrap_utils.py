"""
Per-frame detection records and block-bootstrap confidence intervals for the
corrected BEV-IoU evaluation.

A FrameRecorder keeps, for every evaluated frame, the score-sorted TP flags of
each class at each IoU threshold plus the ground-truth count, so AP can be
recomputed on any resampled set of frames. AP uses the same VOC all-point
interpolation as opencood.utils.eval_utils.voc_ap (vectorised here) and the
same stable score sort as eval_correct_bev_iou.calculate_ap, so the point
estimate on the full set is identical to that evaluator.

Consecutive frames of a sequence are strongly correlated, so resampling single
frames would understate the variance. ci() therefore resamples non-overlapping
blocks of consecutive frames (default 30 frames) with replacement. paired_ci()
resamples the same blocks for two methods evaluated on the same frames, which
gives an interval for their difference.
"""
import numpy as np
import torch

from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou_bev

CLS_IDS = (1, 2, 3)               # car, bike, pedestrian
IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def ap_from_records(scores, tp, n_gt):
    """VOC all-point AP from detection scores and TP flags (any order)."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind='stable')
    tp = tp[order].astype(np.float64)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def match_frame(pred_boxes, pred_scores, gt_boxes):
    """Greedy score-ordered matching, identical to eval_correct_bev_iou.match_class.
    Returns {thr: (scores, tp)} as numpy arrays."""
    n_pred = pred_boxes.shape[0]
    scores = pred_scores.detach().cpu().numpy().astype(np.float64) if n_pred else np.zeros(0)
    if n_pred == 0:
        return {t: (np.zeros(0), np.zeros(0, dtype=np.int8)) for t in IOU_THRESHOLDS}
    order = np.argsort(-scores)
    scores = scores[order]
    if gt_boxes.shape[0] == 0:
        return {t: (scores, np.zeros(n_pred, dtype=np.int8)) for t in IOU_THRESHOLDS}
    iou = boxes_iou_bev(pred_boxes, gt_boxes).cpu().numpy()[order]
    out = {}
    for thr in IOU_THRESHOLDS:
        matched = np.zeros(gt_boxes.shape[0], dtype=bool)
        tp = np.zeros(n_pred, dtype=np.int8)
        for i, row in enumerate(iou):
            for g in np.argsort(-row):
                if row[g] < thr:
                    break
                if not matched[g]:
                    matched[g] = True
                    tp[i] = 1
                    break
        out[thr] = (scores, tp)
    return out


class FrameRecorder:
    """Collects per-frame records for one or more named methods."""

    def __init__(self, methods):
        self.methods = list(methods)
        self.frames = []        # list of {method: {cls: {thr: (scores, tp)}}}
        self.gt = []            # list of {cls: n_gt}
        self.seq = []           # sequence id per frame (for blocking)

    def add(self, seq_id, gt_boxes, gt_class, preds):
        """preds: {method: (boxes, scores, classes)} for this frame."""
        rec, gt = {}, {}
        for c in CLS_IDS:
            gt[c] = int((gt_class == c).sum().item())
        for m, (pb, ps, pc) in preds.items():
            rec[m] = {}
            for c in CLS_IDS:
                mp, mg = pc == c, gt_class == c
                rec[m][c] = match_frame(pb[mp].contiguous(), ps[mp].contiguous(), gt_boxes[mg].contiguous())
        self.frames.append(rec)
        self.gt.append(gt)
        self.seq.append(seq_id)

    # ---- metrics ---------------------------------------------------------
    def _ap(self, method, cls, thr, frame_idx):
        s = np.concatenate([self.frames[i][method][cls][thr][0] for i in frame_idx])
        t = np.concatenate([self.frames[i][method][cls][thr][1] for i in frame_idx])
        n = sum(self.gt[i][cls] for i in frame_idx)
        return ap_from_records(s, t, n)

    def metrics(self, method, frame_idx=None):
        """{(cls, thr): AP, 'mAP': competition mAP} on the given frames."""
        idx = range(len(self.frames)) if frame_idx is None else frame_idx
        idx = list(idx)
        out = {(c, t): self._ap(method, c, t, idx) for c in CLS_IDS for t in IOU_THRESHOLDS}
        out['mAP'] = (out[(1, 0.5)] + out[(2, 0.5)] + out[(3, 0.3)]) / 3
        return out

    def blocks(self, block=30):
        """Non-overlapping blocks of consecutive frames within each sequence."""
        out, cur, last = [], [], None
        for i, s in enumerate(self.seq):
            if s != last or len(cur) == block:
                if cur:
                    out.append(cur)
                cur = []
            cur.append(i)
            last = s
        if cur:
            out.append(cur)
        return out

    def _block_cache(self, method, block):
        """Per block: {(cls, thr): (scores, tp)} and {cls: n_gt}, concatenated once."""
        key = (method, block)
        if not hasattr(self, '_cache'):
            self._cache = {}
        if key not in self._cache:
            cache = []
            for b in self.blocks(block):
                recs = {(c, t): (np.concatenate([self.frames[i][method][c][t][0] for i in b]),
                                 np.concatenate([self.frames[i][method][c][t][1] for i in b]))
                        for c in CLS_IDS for t in IOU_THRESHOLDS}
                gts = {c: sum(self.gt[i][c] for i in b) for c in CLS_IDS}
                cache.append((recs, gts))
            self._cache[key] = cache
        return self._cache[key]

    def _metric_from_blocks(self, method, block, pick, key):
        cache = self._block_cache(method, block)
        keys = [(1, 0.5), (2, 0.5), (3, 0.3)] if key == 'mAP' else [key]
        aps = []
        for c, t in keys:
            s = np.concatenate([cache[b][0][(c, t)][0] for b in pick])
            tp = np.concatenate([cache[b][0][(c, t)][1] for b in pick])
            n = sum(cache[b][1][c] for b in pick)
            aps.append(ap_from_records(s, tp, n))
        return float(np.mean(aps))

    def _picks(self, n_boot, block, seed):
        n_blocks = len(self.blocks(block))
        rng = np.random.default_rng(seed)
        return [rng.integers(0, n_blocks, n_blocks) for _ in range(n_boot)]

    def ci(self, method, key='mAP', n_boot=1000, block=30, seed=0, alpha=0.05):
        vals = [self._metric_from_blocks(method, block, p, key) for p in self._picks(n_boot, block, seed)]
        return tuple(np.quantile(vals, [alpha / 2, 1 - alpha / 2]))

    def paired_ci(self, method_a, method_b, key='mAP', n_boot=1000, block=30, seed=0, alpha=0.05):
        """CI of metric(method_a) - metric(method_b) over the same resampled blocks."""
        vals = [self._metric_from_blocks(method_a, block, p, key) - self._metric_from_blocks(method_b, block, p, key)
                for p in self._picks(n_boot, block, seed)]
        return tuple(np.quantile(vals, [alpha / 2, 1 - alpha / 2]))

    def score_bins(self, method, cls, thr, edges):
        """Precision (TP fraction) and count of detections per score bin."""
        s = np.concatenate([f[method][cls][thr][0] for f in self.frames])
        t = np.concatenate([f[method][cls][thr][1] for f in self.frames])
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (s >= lo) & (s < hi)
            out.append((lo, hi, int(m.sum()), float(t[m].mean()) if m.any() else float('nan')))
        return out


def to_tensor(x, device, dtype):
    return torch.as_tensor(x, dtype=dtype, device=device)
