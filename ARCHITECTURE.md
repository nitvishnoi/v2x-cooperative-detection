# Architecture

V2X cooperative 3D object detection (car, bike, pedestrian) on the Mixed
Signals dataset, using an OpenCOOD-based intermediate-fusion PointPillar
detector.

## Repository layout

```
v2x-cooperative-detection/
├── Mixed-Signals-Dataset/     starter kit (OpenCOOD fork), provided by the competition
│   └── opencood/                model, training loop, config, postprocessing
├── mixed-signals-devkit/      dataset devkit, provided by the competition
├── msig_compat/               bridges the starter kit to the current dataset release
├── data/                      downloaded dataset + generated ground-truth caches
└── logs/                      run logs
```

Trained runs live under `Mixed-Signals-Dataset/opencood/logs/<run_name>/`:
`config.yaml` (resolved config for that run), `net_epoch<N>.pth` checkpoints,
`eval.yaml` (last eval result). The submitted model is in
`opencood/logs/final_model/`.

## Data pipeline

```
HuggingFace sberrio/Mixed-Signals-V2X (flat per-frame .txt labels)
        │
        ▼
msig_compat/txt_explorer.py (MixedSignalsExplorerTxt)
        │  reads .txt labels + point clouds per sequence
        ▼
opencood/data_utils/datasets/msig_base_dataset.py
        │  (patched at runtime, see msig_compat/dataset_patch.py)
        ▼
opencood/tools/msig_dataset_preprocessing.py
        │  builds train_gt_infos.pkl / test_gt_infos.pkl
        ▼
data/mixed-signals/{train,test}_gt_infos.pkl
        │
        ▼
MSigIntermediateFusion dataset → SpVoxelPreprocessor → model
```

29 sequences are downloaded locally under `train/`. The competition's own
val/test split ships point clouds only, no labels, so it can't be used to
compute a local accuracy score. Instead, 6 of the 29 labeled sequences
(31, 32, 33, 34, 35, 37) are held out as our own evaluation split; the
remaining 23 are used for training.

## Why `msig_compat/` exists

The starter kit and devkit were built against an older release of the
dataset (a single `V2X_dataset-v0.4-labels.json` file and a different set of
sequence ids). The current HuggingFace release ships flat per-frame `.txt`
labels and different sequence ids. Rather than editing the starter kit in
place, `msig_compat/dataset_patch.py` applies two runtime patches to
`MSigBaseDataset` before any dataset is built:

1. **Label format** - swaps in `MixedSignalsExplorerTxt` (reads the current
   flat-`.txt` label format) in place of the devkit's JSON-based loader.
2. **Train/test split** - points the held-out split at real sequence ids
   from this release instead of the old hardcoded ones.

`msig_compat/run_{preprocessing,train,eval}.py` apply these patches and then
call straight into the starter kit's own `opencood/tools/*.py` entry points.

## Model

`point_pillar_intermediate` - each connected node (the roadside unit's "top"
LiDAR, which also defines the ego frame, its "dome" LiDAR, and any connected
vehicle - "laser", "003", "004" - within communication range) computes BEV
features locally; features are fused mid-network via attention across nodes,
then a shared detection head runs on the fused map.

```
per node: raw point cloud
        │
        ▼
SpVoxelPreprocessor         voxel size 0.2m × 0.2m × 7m, range ±51.2m
        ▼
PillarVFE                   per-pillar point encoder → 64-dim pillar features
        ▼
PointPillarScatter          scatter to a dense (64, 512, 512) BEV grid
        ▼
AttBEVBackbone (3 scales)   conv stack per scale → attention fusion across
        │                    nodes → upsample → concat all 3 scales
        ▼
spatial_features_2d          (384, 256, 256) fused BEV feature map
        ├──▶ cls_head (1×1 conv)  →  per-anchor class scores
        └──▶ reg_head (1×1 conv)  →  per-anchor box regression
```

Per-class anchors (length/width/height, meters): car [5.13, 2.12, 1.84],
bike [1.76, 0.6, 1.73], pedestrian [0.8, 0.6, 1.73], 2 orientations each.

Training config: `opencood/hypes_yaml/msig_pillar_attnfuse_local.yaml`. Adam,
lr 5e-4, multistep decay (gamma 0.3) at epochs 50 and 80.

## Running it

Environment: `requirements.txt` at the repo root pins every package used to
train/evaluate the submitted model (see the file's header for the extra
package index needed for the CUDA-12.1 torch build).

```bash
cd Mixed-Signals-Dataset
PYTHONPATH=<repo_root>:. <venv_python> ../msig_compat/run_preprocessing.py      # gt_infos, once
PYTHONPATH=<repo_root>:. <venv_python> ../msig_compat/run_train.py \
    --hypes_yaml opencood/hypes_yaml/msig_pillar_attnfuse_local.yaml \
    [--model_dir opencood/logs/<run_name>]                                     # train / resume
PYTHONPATH=<repo_root>:. <venv_python> ../msig_compat/run_eval.py \
    --model_dir opencood/logs/<run_name> --fusion_method intermediate         # eval
```

`MSig_Training_Pipeline.ipynb` (repo root) walks through the same steps.
`Detection_Viewer.ipynb` (repo root) loads the submitted checkpoint and
visualizes its predictions against ground truth on any sample.

## Competition submission

`msig_compat/export_submission.py` runs the submitted checkpoint over the
competition's own held-out frames (`data/mixed-signals-test-competition/`,
no labels) and writes one prediction file per frame in the format the
Codabench scoring program expects (`cls cx cy cz dx dy dz yaw score`),
zipped at the archive root:

```bash
.venv/bin/python msig_compat/export_submission.py \
    --model_dir Mixed-Signals-Dataset/opencood/logs/final_model \
    --out_zip submission.zip
```

Each competition frame ships one already-synchronized snapshot (a `.pcd`
per connected node plus `transforms.json` for their poses at that instant)
rather than the sequence/odometry layout the labeled train/eval data uses,
so this script rebuilds the per-node frame corrections and intensity
normalization `MixedSignalsExplorer.return_agent_point_cloud` applies
before handing off to the same voxelizer, model, and post-processor used
for the results below.

The Codabench evaluation service has since closed and the competition labels
were never released, so a new archive can't be scored. The archives that were
scored are attached to the `final-predictions` GitHub release;
`msig_compat/compare_prediction_archives.py` checks a regenerated archive
against them box by box.

`--cav_pose_no_inverse` uses each CAV's `transforms.json` matrix directly
instead of its inverse. That reproduces the pose-frame experiment
(0.6935 on the leaderboard vs 0.7198 for the default); its regenerated
predictions match the scored `submission_ver3.zip` on 99.6% of boxes.

## Results

Submitted checkpoint: `opencood/logs/final_model/net_epoch55.pth` - the
original 50-epoch run, resumed for 5 more epochs at a manually-set decayed
LR (see note below). Evaluated on the 6-sequence held-out split (1791
frames) with `msig_compat/eval_correct_bev_iou.py`:

| Class      | AP@0.3 | AP@0.5 | AP@0.7 |
|------------|--------|--------|--------|
| Car        | 0.943  | 0.933  | 0.679  |
| Bike       | 0.940  | 0.924  | 0.752  |
| Pedestrian | 0.678  | 0.477  | 0.038  |

The competition ranks by `mAP = mean(Vehicle AP@0.5, Bike AP@0.5,
Pedestrian AP@0.3)` = (0.933 + 0.924 + 0.678) / 3 ≈ **84.5%** on this split,
which scored **71.98%** on the real Codabench leaderboard.

Fusion baselines (`msig_compat/fusion_baselines.py`, same weights, same
ground truth): no fusion (`top` only) 73.43, early fusion 61.78, RSU-only
intermediate 81.16, late fusion 81.79, intermediate fusion 84.51 DEV mAP.
Intermediate fusion beats late fusion mainly on car AP@0.5 (0.933 vs 0.887)
and pedestrian AP@0.7 (0.038 vs 0.021). Pedestrian AP@0.3 is slightly higher
with the RSU pair alone (0.695) than with CAVs added (0.679). See the
README for the full table and caveats.

Post-processor `score_threshold` is 0.2. A lower threshold (0.05) was tried:
VOC-style AP integrates the full precision-recall curve, so in theory any
true positive sitting between 5% and 20% confidence was being discarded
before it ever reached the AP calculation - a free recall cap for no
benefit. It did raise the held-out mAP to 0.886, every class improving. But
on the real competition test set it scored 0.31, far worse than the 0.2
baseline - the model's low-confidence tail is dominated by false positives
on genuinely unseen data, not the mostly-in-distribution held-out split
this was validated against. Reverted to 0.2. This is the sharpest example
in the whole project of a change that looked like a safe, theoretically-
grounded improvement locally and was actively harmful in reality - a
reminder that the held-out split (drawn from the same labeled recording
sessions as training) is an optimistic proxy, not a substitute for testing
against the actual hidden test set.

Note on resuming: `opencood/tools/train.py`'s `--model_dir` resume path only
reloads the model weights, not the optimizer or LR scheduler state, so a
plain resume from epoch 50 silently restarts at the full undecayed LR
(5e-4) instead of the schedule's already-decayed value. Epoch 55 (LR
manually set to 1.5e-4, the correct post-epoch-50 value) is the best point
found - epochs 56 through 85 were also tried and consistently scored worse
even after the second decay milestone, most likely overfitting on the same
6655-sample training pool, so training was stopped and epoch 55 kept as
final.

Note: `eval.yaml` (produced by the starter kit's own
`opencood/utils/eval_utils.py`) reports different, less trustworthy numbers
(car/bike AP nearly identical across all three IoU thresholds). That's a bug
in the provided `caluclate_tp_fp_gpu` - it matches with
`boxes_overlap_bev_gpu`, which returns raw BEV intersection *area in m²*, not
a normalized IoU, so the 0.3/0.5/0.7 threshold comparison is comparing area
against a ratio. It's large enough to matter for car/bike (~10 m² and ~1 m²
boxes) and coincidentally small enough to look plausible for pedestrian
(~0.5 m² boxes). `eval_correct_bev_iou.py` reuses the same trained model,
predictions and `voc_ap` integration, just with a properly union-normalized
IoU (`iou3d_nms_utils.boxes_iou_bev`) - the table above is the trustworthy
one.

Pedestrian accuracy trails the other two classes badly at the stricter
IoU@0.7 threshold, which points to a localization precision limit for
small objects at the current voxel/detection-grid resolution rather than
an undertrained model. A finer detection grid, or a proper resume path
that restores optimizer/scheduler state so longer training doesn't
overfit past epoch 55, are the two most direct ways to close that gap.
