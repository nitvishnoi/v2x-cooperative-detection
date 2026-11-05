# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy


def inference_late_fusion(batch_data, model, dataset, return_output_dict: bool = False, bandwidth_dict: dict = None):
    """
    Model inference for late fusion.
    Faster inference -> batch cav_content

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    input_dict = dict()
    idx_to_cavId = list()
    idx = 0
    for cav_name, cav_content in batch_data.items():
        if cav_name == 'metadata':
            continue
        idx_to_cavId.append(cav_name)

        for k in cav_content['processed_lidar'].keys():
            if k not in input_dict:
                input_dict[k] = list()
            if k == 'voxel_coords':
                assert len(cav_content['processed_lidar']['voxel_coords'].shape) == 2, \
                    f"{len(cav_content['processed_lidar']['voxel_coords'].shape)} != 2"
                cav_content['processed_lidar']['voxel_coords'][:, 0] *= 0
                cav_content['processed_lidar']['voxel_coords'][:, 0] += idx
            input_dict[k].append(cav_content['processed_lidar'][k])

        # ---
        idx += 1

    for k, v in input_dict.items():
        input_dict[k] = torch.cat(v)
    
    input_dict = {
        'processed_lidar': input_dict
    }
    with torch.no_grad():
        out = model(input_dict)
    
    output_dict = OrderedDict()
    for b_idx, cav_name in enumerate(idx_to_cavId):
        output_dict[cav_name] = dict()
        for k in out.keys():
            output_dict[cav_name][k] = out[k][b_idx].unsqueeze(0)

        # output_dict[cav_name] = {
        #     'prob': out['prob'][b_idx].unsqueeze(0),
        #     'rm': out['rm'][b_idx].unsqueeze(0),
        # }

    if return_output_dict:
        # early return to be used by freeze_rsu_inference_late_fusion
        return output_dict

    pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
        dataset.post_process(batch_data, output_dict, bandwidth_dict)

    return pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class


def inference_late_fusion_slow(batch_data, model, dataset):
    output_dict = dict()
    for agent_name, agent_content in batch_data.items():
        if agent_name == 'metadata':
            continue

        with torch.no_grad():
            output_dict[agent_name] = model(agent_content)

    pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
        dataset.post_process(batch_data, output_dict)

    return pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class
        

def freeze_rsu_inference_late_fusion(batch_data, model_rsu, model_cav, dataset):
    rsu_dict = {
        'top': batch_data['top'],
        'dome': batch_data['dome'],
    }

    cav_dict = dict()
    for agent_name, agent_content in batch_data.items():
        if agent_name in dataset.name_all_cars:
            cav_dict[agent_name] = agent_content

    rsu_output_dict = inference_late_fusion(rsu_dict, model_rsu, dataset, return_output_dict=True)
    cav_output_dict = inference_late_fusion(cav_dict, model_cav, dataset, return_output_dict=True)

    # merge rsu_output_dict & cav_output_dict
    output_dict = dict()
    output_dict.update(rsu_output_dict)
    output_dict.update(cav_output_dict)

    # post-process
    # NOTE: late fusion is done by NMS which takes place inside voxel_postprocess_multi.py
    pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class


def inference_early_fusion(batch_data, model, dataset, bandwidth_dict: dict = None):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    cav_content['flag_measure_bandwidth'] = isinstance(bandwidth_dict, dict)

    with torch.no_grad():
        output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
        dataset.post_process(batch_data,
                             output_dict)
    
    if isinstance(bandwidth_dict, dict):
        bandwidth_dict['size_exchanged_info'] += output_dict['ego']['size_exchanged_info']
        bandwidth_dict['num_cavs'] += output_dict['ego']['num_cavs']

    return pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class


def inference_intermediate_fusion(batch_data, model, dataset, bandwidth_dict: dict = None):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset, bandwidth_dict)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy_test' % timestamp), gt_np)
