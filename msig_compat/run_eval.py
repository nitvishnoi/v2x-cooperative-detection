"""Run opencood's inference.py AP evaluation with the msig_compat patches applied first."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Mixed-Signals-Dataset"))

from msig_compat.dataset_patch import apply_all_patches

apply_all_patches()

from opencood.tools import inference

if __name__ == "__main__":
    inference.main()
