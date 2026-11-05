"""
Per-agent forensic inspection for a frame with all 5 agents present:
raw point cloud stats, coordinate transform (ego_se3_agent), and pre-fusion
BEV feature magnitude for each agent - to see whether 004 stands out from
laser/003 on any of these axes.

Usage:
    .venv/bin/python msig_compat/inspect_004.py \
        --model_dir Mixed-Signals-Dataset/opencood/logs/final_model
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "Mixed-Signals-Dataset"))

from msig_compat.dataset_patch import apply_all_patches

apply_all_patches()

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from mixedsignals.utils.geometry import apply_se3_

from msig_compat.export_submission import drop_agents_with_no_points_in_range

AGENT_ORDER = ('top', 'dome', 'laser', '003', '004')


def rotation_to_yaw(R):
    return np.degrees(np.arctan2(R[1, 0], R[0, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    opt = ap.parse_args()

    hypes = yaml_utils.load_yaml(str(Path(opt.model_dir) / 'config.yaml'))
    hypes['flag_train'] = False
    dataset = build_dataset(hypes, visualize=False, train=False)

    model = train_utils.create_model(hypes)
    model.cuda()
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()
    device = torch.device('cuda')

    required = {'laser', '003', '004'}
    found = None
    for idx in range(len(dataset)):
        seq_idx, idx_in_seq = dataset._convert_idx_in_dataset_to_idx_in_sequence(idx)
        idx_in_seq = dataset.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        agents_full = dataset._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
        if required.issubset(set(agents_full.keys())):
            agents_full = drop_agents_with_no_points_in_range(agents_full, dataset._cav_lidar_range)
            if required.issubset(set(agents_full.keys())):
                found = (seq_idx, idx_in_seq, agents_full)
                break

    if found is None:
        print("No frame found with laser+003+004 all present in the scanned range.")
        return

    seq_idx, idx_in_seq, agents_full = found
    print(f"Using seq_idx={seq_idx} idx_in_seq={idx_in_seq}, agents present: {list(agents_full.keys())}\n")

    print("=== Raw point cloud + coordinate transform, per agent ===")
    for agent_name in AGENT_ORDER:
        if agent_name not in agents_full:
            continue
        data = agents_full[agent_name]
        pts = data['points']  # in agent's own body frame
        ego_se3_agent = data['ego_se3_agent']

        pts_ego = pts.copy()
        apply_se3_(ego_se3_agent, points_=pts_ego)

        t = ego_se3_agent[:3, -1]
        yaw = rotation_to_yaw(ego_se3_agent[:3, :3])

        print(f"[{agent_name}]")
        print(f"  n_points = {pts.shape[0]}")
        print(f"  body-frame  xyz range: x[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
              f"y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
        print(f"  ego-frame   xyz range: x[{pts_ego[:,0].min():.2f},{pts_ego[:,0].max():.2f}] "
              f"y[{pts_ego[:,1].min():.2f},{pts_ego[:,1].max():.2f}] z[{pts_ego[:,2].min():.2f},{pts_ego[:,2].max():.2f}]")
        print(f"  ego_se3_agent translation = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}], yaw = {yaw:.2f} deg")
        print(f"  intensity: mean={pts[:,3].mean():.4f} max={pts[:,3].max():.4f}")
        print()

    # --- pre-fusion BEV feature magnitude, per agent slot ---
    captured = {}

    def hook(module, inp, out):
        captured['x'] = inp[0].detach()

    handle = model.backbone.fuse_modules[0].register_forward_hook(hook)

    ego_part, agg_points = dataset._get_projected_voxel_grid_of_every_agent(agents_full)
    data = dataset.init_output_of_getitem()
    data['ego'].update(ego_part)
    data['ego']['anchor_box'] = dataset._anchor_box
    data['metadata']['points'] = agg_points
    batch = dataset.collate_batch_test([data])
    batch = train_utils.to_device(batch, device)

    with torch.no_grad():
        model(batch['ego'])

    handle.remove()

    x = captured['x']  # (cav_num, C, H, W), agent order = dict insertion order = agents_full.keys()
    agent_names_in_order = list(agents_full.keys())
    print("=== Pre-fusion BEV feature magnitude, per agent (scale 0) ===")
    print(f"feature tensor shape: {tuple(x.shape)}\n")
    for i, agent_name in enumerate(agent_names_in_order):
        feat = x[i]
        print(f"[{agent_name}] (slot {i})")
        print(f"  abs().mean() = {feat.abs().mean().item():.4f}")
        print(f"  abs().max()  = {feat.abs().max().item():.4f}")
        print(f"  std()        = {feat.std().item():.4f}")
        print(f"  nonzero frac = {(feat != 0).float().mean().item():.4f}")
        print()


if __name__ == '__main__':
    main()
