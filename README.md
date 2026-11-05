# Mixed Signals V2X — Cooperative 3D Object Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24027618.svg)](https://doi.org/10.5281/zenodo.24027618)

Archived on Zenodo: version 1.0.0 is [10.5281/zenodo.24027619](https://doi.org/10.5281/zenodo.24027619) (commit `c1d5257`, tag `v1.0.0`); [10.5281/zenodo.24027618](https://doi.org/10.5281/zenodo.24027618) always resolves to the latest version. Citation details are in [`CITATION.cff`](CITATION.cff).

V2X cooperative 3D object detection (car, bike, pedestrian) on the [Mixed
Signals dataset](https://huggingface.co/datasets/sberrio/Mixed-Signals-V2X),
Built using OpenCOOD starter kit
(`Mixed-Signals-Dataset/`) and dataset devkit (`mixed-signals-devkit/`), with
a compat layer (`msig_compat/`) bridging the starter kit to the current
dataset release.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for how the pipeline and model work,
and the results on the held-out evaluation split.

## Repo layout

```
Mixed-Signals-Dataset/     starter kit (OpenCOOD fork), provided by the competition
mixed-signals-devkit/      dataset devkit, provided by the competition
msig_compat/               bridges the starter kit to the current dataset release
requirements.txt           full pinned environment used to train/evaluate the submitted model
ARCHITECTURE.md            pipeline, model, and results
MSig_Training_Pipeline.ipynb   walks through setup -> train -> eval
Detection_Viewer.ipynb         loads the submitted model and visualizes its predictions
```

## Setup

Requires Python 3.10 and an NVIDIA GPU (CUDA 12.1 compatible driver). The
dataset itself is large (~230 GB extracted) and is downloaded separately, not
included in this repo.

```bash
git clone <this repo> && cd v2x-cooperative-detection

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

### Get the dataset

```bash
python msig_compat/download_dataset.py
```

Downloads and extracts every `train/` sequence (the labeled pool this model
is trained and evaluated on) plus `val/`/`test/` for completeness, into
`data/mixed-signals/` and `data/mixed-signals-test-competition/`. Resumable —
re-running skips anything already extracted. Pass `--files train/mini_2.tar
...` to fetch specific sequences instead of everything.

The dataset is released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/):
non-commercial use with attribution, share-alike on redistributed
derivatives. Cite the Mixed Signals paper if you use this data or these
results in a publication.

Code licensing is described in [`LICENSE.md`](LICENSE.md): the original code
here is MIT, the bundled starter kit keeps its academic-only licence, and the
released checkpoint and prediction archives follow the dataset's
CC BY-NC-SA 4.0 licence.

### Build ground-truth caches

```bash
cd Mixed-Signals-Dataset
PYTHONPATH=..:. ../.venv/bin/python ../msig_compat/run_preprocessing.py
```

Builds `data/mixed-signals/{train,test}_gt_infos.pkl` — run once, before
training or evaluating.

## Train / evaluate

```bash
# from Mixed-Signals-Dataset/, same PYTHONPATH as above
../.venv/bin/python ../msig_compat/run_train.py \
    --hypes_yaml opencood/hypes_yaml/msig_pillar_attnfuse_local.yaml

../.venv/bin/python ../msig_compat/run_eval.py \
    --model_dir opencood/logs/<run_name> --fusion_method intermediate
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md#running-it) for resuming a run and
more detail on each script.

## Notebooks

Open in Jupyter or VS Code and select the **"V2X (.venv)"**
kernel (`v2x-msig`) — the one pointed at this repo's own `.venv`.

- **`MSig_Training_Pipeline.ipynb`** — setup, config, train, eval, in order.
- **`Detection_Viewer.ipynb`** — loads the submitted checkpoint
  (`Mixed-Signals-Dataset/opencood/logs/final_model/`) and renders the point
  cloud with predicted boxes (colored by class) against ground truth, for
  any sample in the held-out test set.

## Competition submission

```bash
.venv/bin/python msig_compat/export_submission.py \
    --model_dir Mixed-Signals-Dataset/opencood/logs/final_model \
    --out_zip submission.zip
```

Runs the submitted checkpoint over the competition's own held-out frames
(`data/mixed-signals-test-competition/`) and writes `submission.zip` in the
format the Codabench scoring program expects - one `<frame_id>.txt` per
frame, ready to upload on the Submit tab.

## Result

Submitted checkpoint, evaluated on the 6-sequence held-out split (1791
frames) with `msig_compat/eval_correct_bev_iou.py` (see
[`ARCHITECTURE.md`](ARCHITECTURE.md#results) for why this, not the starter
kit's own `eval.yaml`, is the trustworthy number):

| Class      | AP@0.3 | AP@0.5 | AP@0.7 |
|------------|--------|--------|--------|
| Car        | 0.943  | 0.933  | 0.679  |
| Bike       | 0.940  | 0.924  | 0.752  |
| Pedestrian | 0.679  | 0.477  | 0.038  |

mAP (`mean(Vehicle AP@0.5, Bike AP@0.5, Pedestrian AP@0.3)`):
**84.5%** on the held-out split, **71.98%** on the real Codabench
leaderboard. Full writeup, including a lower-threshold change that looked
safe locally but scored far worse for real, in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Reproducing the development experiments

| Change | How to reproduce | DEV mAP | FINAL mAP |
|---|---|---|---|
| Reference (epoch 55, threshold 0.20) | `eval_correct_bev_iou.py --model_dir .../final_model` | 84.51 | 71.98 |
| Confidence threshold 0.05 | copy `final_model/`, set the three `postprocess.score_threshold` values to 0.05, evaluate that copy | 88.63 | 31.18 |
| CAV pose frame not inverted | `export_submission.py ... --cav_pose_no_inverse` (acts only on the competition frames' `transforms.json`, so there is no DEV equivalent) | n/a | 69.35 |
| Random node dropout (p = 0.7) | train with `opencood/hypes_yaml/msig_pillar_attnfuse_dropout070.yaml` (the original run's schedule and checkpoint were not kept) | not recorded | 67.86 |

DEV numbers above were re-run from this commit; repeated evaluation of the
same checkpoint varies by at most 0.02 pp.

FINAL scores are historical: they were returned by the challenge's Codabench
evaluation service in September 2026, the service has since closed, and the
test labels were never released, so they can't be re-scored. The exact
scored archives for the reference, the 0.05 threshold and the pose-frame
variant are attached to the
[`final-predictions`](https://github.com/nitvishnoi/v2x-cooperative-detection/releases/tag/final-predictions)
release with SHA-256 checksums (the node-dropout archive couldn't be
identified with certainty among the retained files, so it isn't included).
Regenerated predictions can be checked against them box by box:

```bash
.venv/bin/python msig_compat/compare_prediction_archives.py \
    --reference submission_reference_71.98.zip --candidate submission.zip
```

Regenerated archives for the reference and the pose-frame variant match the
scored ones on 99.8% / 99.6% of boxes (centers within 5 cm).

### Fusion baselines

`msig_compat/fusion_baselines.py` runs the same epoch-55 weights with
different fusion strategies and scores them all against the same ground
truth as the reference (DEV, 1791 frames; every frame has at least one CAV
in range):

| Method | Nodes / fusion | DEV mAP |
|---|---|---|
| No fusion | roadside `top` LiDAR only | 73.43 |
| Early fusion | all point clouds merged before the network | 61.78 |
| RSU-only | `top` + `dome`, intermediate fusion, CAVs ignored | 81.16 |
| Late fusion | each node detected separately, boxes merged by NMS | 81.79 |
| Intermediate fusion (reference) | all nodes, attention fusion of BEV features | 84.51 |

```bash
.venv/bin/python msig_compat/fusion_baselines.py \
    --model_dir Mixed-Signals-Dataset/opencood/logs/final_model
```

The weights were trained for intermediate fusion, so the other rows are
inference-time comparisons rather than separately trained models; early
fusion in particular feeds the network denser pillars than it was trained on.
The script also prints every class at IoU 0.3/0.5/0.7, a 95% block-bootstrap
confidence interval for each mAP, and a paired interval for each method's
difference to intermediate fusion (`--n_boot`, `--block`; `--stride 4`
evaluates every fourth frame, which is how the paper's intervals were made).

### Further DEV measurements

All use the same weights and ground truth as the reference:

| Script | What it measures |
|---|---|
| `msig_compat/pose_sensitivity.py` | mAP when every CAV's pose is shifted (0.5/1/2 m) or rotated (1/2/5°), inverted, or the CAVs are removed |
| `msig_compat/score_calibration.py` | precision per confidence band and expected calibration error; run it on a copy of `final_model/` with `score_threshold` 0.05 |
| `msig_compat/final_composition_dev.py` | DEV mAP when each frame keeps only as many CAVs as FINAL frames have (70.6% of FINAL frames have no CAV in range, against none on DEV) |

`msig_compat/bootstrap_utils.py` holds the per-frame records and the block
bootstrap they share.

Retraining note: `opencood/tools/train.py` now clips gradients after
`backward()`; the released epoch-55 checkpoint was trained when the clip ran
before it (a no-op), so a retrained model won't match it bit-for-bit.
Dataset paths stored in configs are re-rooted to this clone's `data/`
automatically by `msig_compat.dataset_patch`.
