"""
Compare two FINAL prediction archives box by box.

The FINAL split's labels were never released and its evaluation service has
closed, so FINAL scores can no longer be re-obtained. What can still be checked
is that the released code regenerates the predictions that were scored: this
script matches every box of the reference archive to an unused box of the same
class in the candidate archive whose center lies within --tol metres, and
reports the matched fraction.

The archives that were scored are attached to the repository release
`final-predictions` (with SHA-256 checksums).

Usage:
    .venv/bin/python msig_compat/compare_prediction_archives.py \
        --reference submission_reference_71.98.zip \
        --candidate submission.zip [--tol 0.05]
"""
import argparse
import zipfile

import numpy as np


def load(path):
    """{frame file name: (N, 9) array of cls cx cy cz dx dy dz yaw score}"""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('.txt'):
                continue
            rows = [line.split() for line in z.read(name).decode().splitlines() if line.strip()]
            out[name] = np.array(rows, dtype=float).reshape(-1, 9)
    return out


def compare(reference, candidate, tol):
    matched = total = 0
    missing_frames = 0
    for name, ref_boxes in reference.items():
        total += len(ref_boxes)
        cand = candidate.get(name)
        if cand is None:
            missing_frames += 1
            continue
        used = np.zeros(len(cand), dtype=bool)
        for box in ref_boxes:
            same_cls = (cand[:, 0] == box[0]) & ~used
            if not same_cls.any():
                continue
            dist = np.full(len(cand), np.inf)
            dist[same_cls] = np.abs(cand[same_cls, 1:4] - box[1:4]).max(axis=1)
            j = int(np.argmin(dist))
            if dist[j] <= tol:
                used[j] = True
                matched += 1
    return matched, total, missing_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True, help='archive that was scored')
    ap.add_argument('--candidate', required=True, help='regenerated archive')
    ap.add_argument('--tol', type=float, default=0.05, help='max center difference per axis (m)')
    opt = ap.parse_args()

    ref, cand = load(opt.reference), load(opt.candidate)
    matched, total, missing = compare(ref, cand, opt.tol)
    n_cand = sum(len(v) for v in cand.values())
    print(f'reference: {len(ref)} frames, {total} boxes | candidate: {len(cand)} frames, {n_cand} boxes')
    print(f'frames missing from candidate: {missing}')
    print(f'reference boxes matched within {opt.tol} m: {matched}/{total} = {matched / max(total, 1):.4f}')


if __name__ == '__main__':
    main()
