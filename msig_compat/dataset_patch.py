"""
Monkeypatches applied to `opencood.data_utils.datasets.msig_base_dataset` so the
OpenCOOD MSig* datasets work against the current (flat .txt label) HuggingFace
release of Mixed-Signals-V2X, with a real held-out evaluation split.

Patch 1 - label format (`apply_explorer_patch`)
-------------------------------------------------
`msig_base_dataset.py` instantiates `MixedSignalsExplorer` directly, which
expects the old `V2X_dataset-v0.4-labels.json` blob that this release doesn't
ship. This patch swaps in `MixedSignalsExplorerTxt` (`msig_compat/
txt_explorer.py`), which reads the flat per-frame `.txt` labels instead.

Patch 2 - train/test split (`apply_test_sequences_patch`)
-------------------------------------------------------------
`MSigBaseDataset` hardcodes `_test_sequences = {4, 5, 17, 21}`, ids that
don't exist in this release. This patch replaces `_filter_sequences_` with a
version that reads a module-level `_TEST_SEQUENCES` set instead.

The competition's official val/test split ships point clouds only, no
labels, so it can't be used to compute a local AP score. `_TEST_SEQUENCES`
instead holds out the 6 highest-numbered of the 29 labeled `train/`
sequences ({31, 32, 33, 34, 35, 37}), keeping the other 23 (~79%) for
training - a reproducible sequence-level split used only for our own
held-out evaluation, separate from the competition's own scoring.
"""
import os
from pathlib import Path

import numpy as np

_TEST_SEQUENCES = {31, 32, 33, 34, 35, 37}


def apply_explorer_patch() -> None:
    import opencood.data_utils.datasets.msig_base_dataset as msig_base_dataset_mod
    from msig_compat.txt_explorer import MixedSignalsExplorerTxt

    msig_base_dataset_mod.MixedSignalsExplorer = MixedSignalsExplorerTxt


def _filter_sequences_patched(self) -> None:
    """Same logic as upstream `_filter_sequences_`, except it reads the
    module-level `_TEST_SEQUENCES` (set via `apply_test_sequences_patch`)
    instead of the hardcoded `self._test_sequences = {4, 5, 17, 21}`."""
    if self._flag_train:
        kept_seqs = [s for s in self._available_seqs if s not in _TEST_SEQUENCES]
    else:
        kept_seqs = [s for s in self._available_seqs if s in _TEST_SEQUENCES]

    self._available_seqs = np.array(kept_seqs)

    if self._flag_verbose:
        print(f"MSigBaseDataset[patched] | train: {self._flag_train} | "
              f"avail seqs: {self._available_seqs} | test_sequences: {_TEST_SEQUENCES}")
    return


def apply_test_sequences_patch(test_sequences=(31, 32, 33, 34, 35, 37)) -> None:
    global _TEST_SEQUENCES
    _TEST_SEQUENCES = set(test_sequences)

    import opencood.data_utils.datasets.msig_base_dataset as msig_base_dataset_mod
    msig_base_dataset_mod.MSigBaseDataset._filter_sequences_ = _filter_sequences_patched


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _remap_data_paths(node):
    """Configs (and the run dirs' saved config.yaml) record absolute dataset
    paths from the machine they were trained on. If such a path doesn't exist
    here, re-root its `data/mixed-signals...` tail under this clone's `data/`
    so a fresh checkout runs without hand-editing every yaml."""
    if isinstance(node, dict):
        return {k: _remap_data_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_remap_data_paths(v) for v in node]
    if isinstance(node, str) and os.path.isabs(node) and not os.path.exists(node):
        marker = '/data/mixed-signals'
        idx = node.find(marker)
        if idx != -1:
            candidate = str(_PROJECT_ROOT) + node[idx:]
            if os.path.exists(candidate):
                return candidate
    return node


def apply_data_path_patch() -> None:
    import opencood.hypes_yaml.yaml_utils as yaml_utils_mod
    if getattr(yaml_utils_mod, '_msig_paths_patched', False):
        return
    for name in ('load_yaml', 'msig_load_yaml'):
        original = getattr(yaml_utils_mod, name)

        def wrapped(file, opt=None, _original=original):
            return _remap_data_paths(_original(file, opt))

        setattr(yaml_utils_mod, name, wrapped)
    yaml_utils_mod._msig_paths_patched = True


def apply_all_patches(test_sequences=(31, 32, 33, 34, 35, 37)) -> None:
    apply_explorer_patch()
    apply_test_sequences_patch(test_sequences)
    apply_data_path_patch()
