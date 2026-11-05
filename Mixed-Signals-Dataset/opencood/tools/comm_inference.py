import argparse
import os
import yaml
import re
from tqdm import tqdm
from collections import OrderedDict

import torch
import open3d as o3d
from torch.utils.data import DataLoader

from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils



def load_yaml(cfg_file: str):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    stream = open(cfg_file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param


def load_weights(model, path_ckpt):
    checkpoint = torch.load(
        path_ckpt,
        map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)

    del checkpoint
    return


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    
    parser.add_argument('--cfg_rsu', type=str, required=True,
                        help='config of model for rsu')
    
    parser.add_argument('--ckpt_rsu', type=str, required=True,
                        help='ckpt of model for rsu')
    
    parser.add_argument('--cfg_car', type=str, required=True,
                        help='config of model for car')
    
    parser.add_argument('--ckpt_car', type=str, required=True,
                        help='ckpt of model for car')
    
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    
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
    assert torch.cuda.is_available()
    
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    
    hypes_rsu = load_yaml(opt.cfg_rsu)
    hypes_car = load_yaml(opt.cfg_car)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes_rsu, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=0,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model & load ckpt')
    model_rsu = train_utils.create_model(hypes_rsu)
    load_weights(model_rsu, opt.ckpt_rsu)
    model_rsu.eval().cuda()

    model_car = train_utils.create_model(hypes_car)
    load_weights(model_car, opt.ckpt_car)
    model_car.eval().cuda()
    
    device = torch.device('cuda')


    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    cls_idx_to_name = ('dummy', 'car', 'bike', 'pedestrian')
    result_multi = OrderedDict({
        'car': make_result_stat_dict(),
        'bike': make_result_stat_dict(),
        'pedestrian': make_result_stat_dict(),
    })

    for i, batch_data in tqdm(enumerate(data_loader), total=len(opencood_dataset)):
        # print(i)
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
                    inference_utils.comm_inference_late_fusion(batch_data,
                                                               model_rsu,
                                                               model_car,
                                                               opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model_rsu,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, pred_class, gt_box_tensor, gt_class = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model_rsu,
                                                                  opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
            unq_cls_idx = torch.unique(pred_class).cpu().numpy()
            for cls_idx in unq_cls_idx:
                mask_this_cls = pred_class == cls_idx
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
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)
            

    eval_utils.eval_final_results(result_multi,
                                  opt.model_dir,
                                  opt.global_sort_detections)
    
if __name__ == '__main__':
    main()
