"""
Driver for opencood/tools/msig_dataset_preprocessing.py that applies our
compat patches (label format + train/test split, see
msig_compat/dataset_patch.py) before building train_gt_infos.pkl /
test_gt_infos.pkl for the locally downloaded dataset.

Run from the Mixed-Signals-Dataset/ directory with:
    PYTHONPATH=<repo_root>:. <venv_python> ../msig_compat/run_preprocessing.py

(PYTHONPATH needs the repo root so `msig_compat` is importable, and '.' so
`opencood` resolves the same way the upstream script expects.)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for msig_compat

from msig_compat.dataset_patch import apply_all_patches

apply_all_patches()  # must happen BEFORE opencood.tools.msig_dataset_preprocessing builds any dataset

from opencood.tools.msig_dataset_preprocessing import make_partition_info
from opencood.hypes_yaml.yaml_utils import msig_load_yaml


def main():
    params = msig_load_yaml('opencood/hypes_yaml/msig_preprocessing_local.yaml')

    params['flag_train'] = True
    make_partition_info(params)

    params['flag_train'] = False
    make_partition_info(params)


if __name__ == '__main__':
    main()
