import argparse
from tqdm import tqdm
from collections import OrderedDict
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.freezed_rsu_fusion_utils import openpcdet_load_weights


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    
    parser.add_argument('--model_dir', type=str, default='', help='Continued training path')
    parser.add_argument('--hypes_yaml', type=str, default='', help='')
    parser.add_argument('--ckpt', type=str, default='', help='')
    parser.add_argument('--fusion_method', required=True, type=str, help='')
    parser.add_argument('--save_eval_result_to_directory', type=str, default='')
    parser.add_argument('--compute_avg_bandwidth', action='store_true')
    parser.add_argument('--workers', default=4, type=int)

    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def make_result_stat_dict() -> dict:
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    return result_stat


def main():
    opt = test_parser()
    # assert opt.fusion_method in ['late', 'early', 'intermediate']

    if opt.model_dir != '':
        print(f'EVAL INFO | loading config from {opt.model_dir}/config.yaml')
        print(f'EVAL INFO | eval result is saved to {opt.model_dir}/eval.yaml')
        hypes = yaml_utils.load_yaml(None, opt)
        opt.save_eval_result_to_directory = opt.model_dir
    else:
        assert opt.hypes_yaml != '', f"opt.hypes_yaml must not be ''"
        assert opt.ckpt != '', f"opt.ckpt must not be ''"
        assert opt.save_eval_result_to_directory != '', f"opt.ckpt must not be ''"

        print(f'EVAL INFO | loading config from {opt.hypes_yaml}')
        print(f'EVAL INFO | eval result is saved to {opt.save_eval_result_to_directory}/eval.yaml')
        os.makedirs(opt.save_eval_result_to_directory, exist_ok=True)
        hypes = yaml_utils.msig_load_yaml(opt.hypes_yaml)

    hypes['flag_train'] = False

    print('Dataset Building')
    dataset = build_dataset(hypes, visualize=False, train=False)
    print(f"{len(dataset)} samples found.")
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             num_workers=opt.workers,
                             collate_fn=dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    model.cuda()
    device = torch.device('cuda')

    if opt.model_dir != '':
        print(f'EVAL INFO | loading model weight from the last ckpt in {opt.model_dir}')
        _, model = train_utils.load_saved_model(opt.model_dir, model)
    else:
        print(f'EVAL INFO | loading model weight from {opt.ckpt}')
        train_utils.openpcdet_load_weights(model, opt.ckpt, strict=True)
    model.eval()

    if opt.fusion_method == 'late_freeze_rsu':
        rsu_cfg = {'model': hypes['freezed_rsu_config']}
        rsu_cfg['model']['args']['point_pillar_scatter']['grid_size'] = \
            hypes['model']['args']['point_pillar_scatter']['grid_size']
        
        model_rsu = train_utils.create_model(rsu_cfg)
        openpcdet_load_weights(model_rsu, rsu_cfg['model']['path_ckpt'])
        model_rsu.eval().cuda()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    cls_idx_to_name = ('dummy', 'car', 'bike', 'pedestrian')
    result_multi = OrderedDict({
        'car': make_result_stat_dict(),
        'bike': make_result_stat_dict(),
        'pedestrian': make_result_stat_dict(),
    })

    if opt.compute_avg_bandwidth:
        bandwidth_dict = {
            'size_exchanged_info': 0,
            'num_cavs': 0,
        }
    else:
        bandwidth_dict = None

    for i, batch_data in tqdm(enumerate(data_loader), total=len(dataset)):
        batch_data = train_utils.to_device(batch_data, device)

        if len(batch_data.keys()) == 1:
            # doesn't have any agents in this batch
            # this is the case when eval single-agent model and the chosen agent
            # doesn't have any points in the labeld regions of the dataset
            assert list(batch_data.keys()) == ['metadata'], f"??? keys of batch_data: {list(batch_data.keys())}"
            continue

        if opt.fusion_method == 'late':
            pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
                inference_utils.inference_late_fusion_slow(batch_data, model, dataset)
                # inference_utils.inference_late_fusion(
                #     batch_data, model, dataset, return_output_dict=False, bandwidth_dict=bandwidth_dict)
            
        elif opt.fusion_method == 'early' or opt.fusion_method == 'laly':
            pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
                inference_utils.inference_early_fusion(
                    batch_data, model, dataset, bandwidth_dict)
            
        elif opt.fusion_method == 'intermediate':
            pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
                inference_utils.inference_intermediate_fusion(
                    batch_data, model, dataset, bandwidth_dict)
            
        elif opt.fusion_method == 'late_freeze_rsu':
            pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
                inference_utils.freeze_rsu_inference_late_fusion(
                    batch_data, model_rsu, model, dataset)
            
        else:
            raise NotImplementedError('Only early, late and intermediate'
                                        'fusion is supported.')
        
        if gt_box_tensor.shape[0] == 0:
            # this happen when eval with condition: gt_boxes visible to a certain agent
            continue

        unq_cls_idx = torch.unique(gt_class)

        for cls_idx in unq_cls_idx:
            mask_this_cls = pred_class == cls_idx
            if len(pred_box_tensor) == 0:
                this_cls_pred_box = torch.zeros(0, 7)
                this_cls_pred_score = torch.zeros(0)
                
            this_cls_pred_box = pred_box_tensor[mask_this_cls].contiguous()
            this_cls_pred_score = pred_score[mask_this_cls].contiguous()
            
            mask_gt_this_cls = gt_class == cls_idx
            this_cls_gt_box = gt_box_tensor[mask_gt_this_cls].contiguous()
            eval_utils.caluclate_tp_fp_gpu(this_cls_pred_box,
                                    this_cls_pred_score,
                                    this_cls_gt_box,
                                    result_multi[cls_idx_to_name[cls_idx]],
                                    0.3)
            eval_utils.caluclate_tp_fp_gpu(this_cls_pred_box,
                                    this_cls_pred_score,
                                    this_cls_gt_box,
                                    result_multi[cls_idx_to_name[cls_idx]],
                                    0.5)
            eval_utils.caluclate_tp_fp_gpu(this_cls_pred_box,
                                    this_cls_pred_score,
                                    this_cls_gt_box,
                                    result_multi[cls_idx_to_name[cls_idx]],
                                    0.7)

    if isinstance(bandwidth_dict, dict):
        avg_bandwidth = float(bandwidth_dict['size_exchanged_info'])  / (1e3 * float(bandwidth_dict['num_cavs']))
        print(f'EVAL INFO | avg bandwidth: {avg_bandwidth} (KB)')

    eval_utils.eval_final_results(result_multi,
                                  opt.save_eval_result_to_directory,
                                  opt.global_sort_detections)


if __name__ == '__main__':
    main()
