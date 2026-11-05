"""
Driver for opencood/tools/train.py that applies our compat patches (label
format + train/test split, see msig_compat/dataset_patch.py)
before opencood builds any dataset. Needed because train.py's own main()
calls build_dataset() directly with no patch hook.

Run from the Mixed-Signals-Dataset/ directory with:
    <venv_python> ../msig_compat/run_train.py --hypes_yaml opencood/hypes_yaml/msig_pillar_attnfuse_local.yaml [other train.py args]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for msig_compat

from msig_compat.dataset_patch import apply_all_patches

apply_all_patches()  # must happen BEFORE opencood.tools.train builds any dataset

from opencood.tools.train import main

if __name__ == '__main__':
    main()
