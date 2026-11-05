from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import lovely_tensors as lt
lt.monkey_patch()

from opencood.tools import train_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.freezed_rsu_fusion_utils import openpcdet_load_weights


def main():
    path_ckpt = 'ckpt/debug_pillar_late/net_epoch22.pth'

    hypes_yaml = 'opencood/hypes_yaml/point_pillar_late_multi.yaml'
    hypes = load_yaml(hypes_yaml)

    model = train_utils.create_model(hypes)
    
    openpcdet_load_weights(model, path_ckpt)


if __name__ == '__main__':
    main()

