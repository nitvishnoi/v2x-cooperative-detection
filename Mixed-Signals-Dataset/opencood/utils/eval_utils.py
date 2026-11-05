# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils
from opencood.pcdet_utils.iou3d_nms import iou3d_nms_cuda
from opencood.utils import box_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

        result_stat[iou_thresh]['score'] += det_score.tolist()

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def caluclate_tp_fp_gpu(det_boxes: torch.Tensor, det_score: torch.Tensor, gt_boxes: torch.Tensor, 
                        result_stat: dict, iou_thresh: float):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : 
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :
        The confidence score for each preditect bounding box.
    gt_boxes :
        The groundtruth bounding box.
    result_stat:
        A dictionary contains fp, tp and gt number.
    iou_thresh :
        The iou thresh.
    """

    # det_boxes, _ = common_utils.check_numpy_to_torch(det_boxes)
    # det_score, _ = common_utils.check_numpy_to_torch(det_score)
    # gt_boxes, _ = common_utils.check_numpy_to_torch(gt_boxes)

    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes.shape[0] > 0:
        sorted_scores, sorted_det_indices = torch.sort(det_score, descending=True)
        sorted_scores = sorted_scores.cpu().numpy()
        sorted_det_indices = sorted_det_indices.cpu().numpy()

        overlaps_bev = torch.tensor(
            torch.zeros((det_boxes.shape[0], gt_boxes.shape[0])), 
            dtype=torch.float32, 
            device='cuda'
        )

        if gt_boxes.shape[0] > 0:
            # iou between pred & gt
            iou3d_nms_cuda.boxes_overlap_bev_gpu(
                det_boxes.cuda().contiguous(), gt_boxes.cuda().contiguous(), 
                overlaps_bev
            )

        overlaps_bev, sorted_gt_indices = torch.sort(overlaps_bev, dim=1, descending=True)
        overlaps_bev = overlaps_bev.cpu().numpy()
        sorted_gt_indices = sorted_gt_indices.cpu().numpy()

        mask_matched_gt_boxes = np.zeros(gt_boxes.shape[0], dtype=bool)  # True if a gt is matched with a det
        num_columns = overlaps_bev.shape[1]
        for ii in sorted_det_indices:
            column_idx_overlaps_bev = 0
            matched = False 
            if not np.all(mask_matched_gt_boxes):  # not all gt_boxes are matched
                while column_idx_overlaps_bev < num_columns and not matched:
                    # check if this matched_gt_idx is already taken by another det
                    _this_gt_idx = sorted_gt_indices[ii, column_idx_overlaps_bev]

                    if not mask_matched_gt_boxes[_this_gt_idx]:
                        # check if this matched_gt_idx has sufficiently high iou
                        matched = overlaps_bev[ii, column_idx_overlaps_bev] >= iou_thresh

                    if not matched:
                        # check the next column
                        column_idx_overlaps_bev += 1
                    else:
                        # matched !
                        # mark this gt_box is matched with a det
                        mask_matched_gt_boxes[_this_gt_idx] = True
            
            if matched:
                # ii is a true positive
                fp.append(0)
                tp.append(1)
            else:
                # ii is a false positive
                fp.append(1)
                tp.append(0)

            result_stat[iou_thresh]['score'] += sorted_scores.tolist()

    if gt_boxes.shape[0] == 0:
        # this class does not have any ground truth
        # all detections are False Positive
        if det_boxes.shape[0] > 0:
            fp = [1] * det_boxes.shape[0]
            tp = [0] * det_boxes.shape[0]

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def calculate_ap(result_stat, iou, global_sort_detections):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
        
    iou : float
        The threshold of iou.

    global_sort_detections : bool
        Whether to sort the detection results globally.
    """
    iou_5 = result_stat[iou]

    if global_sort_detections:
        fp = np.array(iou_5['fp'])
        tp = np.array(iou_5['tp'])
        score = np.array(iou_5['score'])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()
        
    else:
        fp = iou_5['fp']
        tp = iou_5['tp']
        assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_multi, save_path, global_sort_detections):
    dump_dict = {}

    for cls_name, result_stat in result_multi.items():
        ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30, global_sort_detections)
        ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50, global_sort_detections)
        ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70, global_sort_detections)

        dump_dict.update({f'{cls_name}_ap30': ap_30,
                        f'{cls_name}_ap_50': ap_50,
                        f'{cls_name}_ap_70': ap_70,
                        f'{cls_name}_mpre_50': mpre_50,
                        f'{cls_name}_mrec_50': mrec_50,
                        f'{cls_name}_mpre_70': mpre_70,
                        f'{cls_name}_mrec_70': mrec_70,
                        })
        print('-------------------------------------------')
        print(f'eval result of class {cls_name}')
        print('\tAP @ IOU 0.3 : %.2f \n'
            '\tAP @ IOU 0.5 : %.2f \n'
            '\tAP @ IOU 0.7 : %.2f \n' % (ap_30, ap_50, ap_70))
        print('-------------------------------------------')
    
    output_file = 'eval.yaml' if not global_sort_detections else 'eval_global_sort.yaml'
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))
