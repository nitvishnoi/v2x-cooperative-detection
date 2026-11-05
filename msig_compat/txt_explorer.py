"""
Compatibility adapter for mixed-signals-devkit against the *current*
Mixed-Signals-V2X HuggingFace release.

Why this exists
----------------
`mixedsignals.mixed_signals.MixedSignalsExplorer` (the devkit shipped by
quan-dao/mixed-signals-devkit, referenced from the OpenCOOD starter kit
acfr/Mixed-Signals-Dataset) hard-requires a single
`V2X_dataset-v0.4-labels.json` file at the dataset root
(see `_make_label_generator_for_avail_seqs`). The dataset actually published
on HuggingFace (sberrio/Mixed-Signals-V2X) no longer ships that file -
labels are one flat text file per synchronized frame instead:

    labels/mini_<N>/mini_<N>_<syncIndex>.txt
    each line: cls cx cy cz dx dy dz yaw   (8 fields, no track_id)

This module provides `MixedSignalsExplorerTxt`, a drop-in subclass of
`MixedSignalsExplorer` that builds its label generators from those .txt
files instead of the missing JSON. Everything else (point cloud loading,
odometry, coordinate frames, category grouping) is untouched, since the
`PointClouds/` and `Odometry/` layout is unchanged between releases.

Per-frame indices were verified to be dense (no gaps) within their labeled
range, so no timestamp interpolation is needed - each labeled sync index
maps 1:1 to the label file of the same index.

Track IDs are not present in the new format. Since deduplication
(`remove_repetitive_boxes` in `msig_dataset_preprocessing.py`) only needs
uniqueness *within a single frame*, we assign a synthetic track_id equal to
the row's line number in its file. This is NOT a real object identity and
must not be relied on across frames (e.g. `return_tracks_traj`).
"""
from pathlib import Path
from typing import Dict, List

import numpy as np

from mixedsignals.mixed_signals import MixedSignalsExplorer, TYPE_SEQ_INDEX
from mixedsignals.utils.mixed_signals_utils import TYPE_TIMESTAMP
from mixedsignals.utils.geometry import apply_se3_


class SequenceLabeledFramesTxt:
    """Same public interface as mixedsignals.utils.mixed_signals_utils.SequenceLabeledFrames,
    backed by flat per-frame .txt files instead of the v0.4 JSON blob."""

    def __init__(self, seq_index: int, labels_dir: Path, agent_top_timestamps: List[TYPE_TIMESTAMP]):
        self.seq_index = seq_index

        label_files = sorted(
            labels_dir.glob(f"mini_{seq_index}_*.txt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        assert len(label_files) > 0, f"no label files found in {labels_dir} for seq {seq_index}"

        self.labeled_frames_timestamp: List[TYPE_TIMESTAMP] = []
        self.labeled_frames_gt: List[np.ndarray] = []

        for f in label_files:
            sync_idx = int(f.stem.split("_")[-1])
            assert sync_idx < len(agent_top_timestamps), \
                f"label sync_idx {sync_idx} out of range for {len(agent_top_timestamps)} 'top' frames"

            boxes = []
            with open(f) as fh:
                for row_idx, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    cls, cx, cy, cz, dx, dy, dz, yaw = (float(x) for x in line.split())
                    # devkit box order: x, y, z, l, w, h, yaw, category_id, track_id
                    boxes.append([cx, cy, cz, dx, dy, dz, yaw, cls, row_idx])

            gt_boxes = np.array(boxes, dtype=np.float32).reshape(-1, 9)
            self.labeled_frames_timestamp.append(agent_top_timestamps[sync_idx])
            self.labeled_frames_gt.append(gt_boxes)

        sorted_ids = np.argsort(self.labeled_frames_timestamp)
        assert np.all(sorted_ids == np.arange(len(self.labeled_frames_timestamp))), \
            "label sync indices must be monotonic with 'top' agent timestamps"

    def return_labeled_frames_timestamp(self) -> List[TYPE_TIMESTAMP]:
        return self.labeled_frames_timestamp

    def return_annos(self, timestamp: TYPE_TIMESTAMP, debug: bool = False) -> np.ndarray:
        """Exact-match lookup (dense per-frame labels -> no interpolation needed).

        Returns a COPY, not a reference to the cached array. Callers upstream
        (MSigBaseDataset.return_gt_boxes_in_ego -> apply_se3_) mutate the
        returned array in place, and some callers (e.g.
        check_idx_in_seq_has_gt_boxes, which calls return_gt_boxes_in_ego
        once to test for emptiness and __getitem__ calls it again right
        after to actually fetch the same frame) end up calling this twice
        for the same frame. Without copying here, the in-place SE3 transform
        gets applied twice (or more) to the cached label array, permanently
        corrupting it and scattering every box far from its real position.
        """
        idx = np.searchsorted(self.labeled_frames_timestamp, timestamp)
        if idx >= len(self.labeled_frames_timestamp) or \
                abs(self.labeled_frames_timestamp[idx] - timestamp) > 1e-3:
            if idx > 0 and abs(self.labeled_frames_timestamp[idx - 1] - timestamp) <= 1e-3:
                idx = idx - 1
            else:
                if debug:
                    print(f"DEBUG | {timestamp} is not labeled")
                return np.zeros((0, 9), dtype=np.float32)
        return self.labeled_frames_gt[idx].copy()


class MixedSignalsExplorerTxt(MixedSignalsExplorer):
    """MixedSignalsExplorer that reads the current flat-.txt label format."""

    def __init__(self, dataset_root: str, verbose: bool = False) -> None:
        # Build pc_generators before label_generators (parent order is reversed
        # and label-building needs 'top' timestamps to key the txt labels).
        from mixedsignals.utils.mixed_signals_utils import SequencePointClouds

        self.root = Path(dataset_root)
        self.verbose = verbose

        self.name_vehicles = ('laser', '003', '004')
        self.name_rsu_lidars = ('top', 'dome')
        self.name_agents = self.name_rsu_lidars + self.name_vehicles
        self.dict_agent_indicator = dict(zip(self.name_agents, range(len(self.name_agents))))

        self.available_sequences_index = self._find_available_sequences()
        if self.verbose:
            print(f"MixedSignalsExplorerTxt | available_sequences_index: {self.available_sequences_index}")

        self.pc_generators = {
            seq_idx: SequencePointClouds(self.root, seq_idx)
            for seq_idx in self.available_sequences_index
        }

        self.label_generators = self._make_label_generator_for_avail_seqs()

        self.odom_generators, self.dict_exist_cavs = self._make_odom_generator_for_avail_seqs()

        # === pose of RSU (identical to parent)
        self.map_se3_dome = np.array([
            [1.0, 0.0, 0.0, -41.507],
            [0.0, 1.0, 0.0, -51.864],
            [0.0, 0.0, 1.0, -1.340],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.dome_se3_map = np.linalg.inv(self.map_se3_dome)

        self.map_se3_top = np.array([
            [1.0, 0.0, 0.0, -41.551],
            [0.0, 1.0, 0.0, -51.878],
            [0.0, 0.0, 1.0, -1.077],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.top_se3_map = np.linalg.inv(self.map_se3_top)

        self.time_anchor_agent = 'top'
        self._category_indices = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self._category_names = ['car', 'pedestrian', 'truck', 'motorbike', 'bicycle',
                                 'emergency_vehicle', 'bus', 'portable_personal_mobility',
                                 'electric_vehicle', 'trailer']
        self._translate_car_along_z = -3.25

    def return_gt_boxes_in_map(self, seq_idx: TYPE_SEQ_INDEX, sync_time_idx: int) -> np.ndarray:
        """
        Override of MixedSignalsExplorer.return_gt_boxes_in_map.

        Empirically verified (by matching raw label xyz against the 'top'
        RSU's own point cloud - closest point 0.16m away, 797 points within
        3m, z-range consistent with a car-height box) that in the current
        HuggingFace release, box labels are published directly in the
        'top' RSU's own sensor frame, NOT in an intermediate "map" frame.

        Callers (MSigBaseDataset.return_gt_boxes_in_ego) unconditionally
        apply `top_se3_map` on top of whatever this method returns, to
        convert from "map frame" to "top/ego frame". Since our labels are
        already top-frame-correct, we pre-apply the inverse (`map_se3_top`)
        here so that downstream `apply_se3_(top_se3_map, ...)` exactly
        cancels it out and the labels pass through unchanged. Without this,
        every box gets shifted by the hardcoded ~(41.5, 51.9, 1.1) RSU-pose
        offset and ends up nowhere near any lidar point (verified: with the
        offset applied, closest point to a box was >24m away; every box in
        both train/test gt_infos.pkl was then discarded by
        process_boxes_based_on_points's min-points-in-box filter).
        """
        query_timestamp = self.return_timestamp_for_query_gt(seq_idx, sync_time_idx)
        gt_boxes = self.label_generators[seq_idx].return_annos(query_timestamp, self.verbose)
        apply_se3_(self.map_se3_top, boxes_=gt_boxes, boxes_has_velocity=False)
        return gt_boxes

    def return_squence_length(self, seq_index: TYPE_SEQ_INDEX) -> int:
        # Upstream MixedSignalsExplorer.return_squence_length calls
        # len(SequencePointClouds instance), but that class defines no
        # __len__, so it always raises TypeError. Compute it directly instead.
        return len(self.pc_generators[seq_index].agents_timestamp[self.time_anchor_agent])

    def _find_available_sequences(self) -> List[int]:
        """Robust replacement for mixedsignals_utils.find_available_sequences,
        which crashes (UnboundLocalError) when a third top-level dir such as
        'labels/' is present alongside PointClouds/ and Odometry/."""
        avail_seq_pcs, avail_seq_odom = set(), set()
        for directory_data in self.root.iterdir():
            if not directory_data.is_dir():
                continue
            if directory_data.name == 'PointClouds':
                target = avail_seq_pcs
            elif directory_data.name == 'Odometry':
                target = avail_seq_odom
            else:
                continue
            for directory_seq in directory_data.iterdir():
                if directory_seq.is_dir():
                    target.add(int(directory_seq.name.split('_')[1]))

        assert avail_seq_pcs == avail_seq_odom, \
            f"PointClouds seqs {avail_seq_pcs} != Odometry seqs {avail_seq_odom}"
        return sorted(avail_seq_pcs)

    def _make_label_generator_for_avail_seqs(self) -> Dict[TYPE_SEQ_INDEX, SequenceLabeledFramesTxt]:
        dict_label_generators = {}
        for seq_idx in self.available_sequences_index:
            labels_dir = self.root / "labels" / f"mini_{seq_idx}"
            if not labels_dir.exists():
                # val/test scenes ship point clouds only, labels withheld
                if self.verbose:
                    print(f"MixedSignalsExplorerTxt | seq {seq_idx} has no labels dir, skipping")
                continue
            top_timestamps = self.pc_generators[seq_idx].return_all_timestamps_of_agent('top')
            dict_label_generators[seq_idx] = SequenceLabeledFramesTxt(seq_idx, labels_dir, top_timestamps)
        return dict_label_generators
