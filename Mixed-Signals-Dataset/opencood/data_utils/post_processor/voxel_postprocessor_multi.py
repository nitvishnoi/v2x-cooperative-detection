from typing import Tuple

import math
import numpy as np
import torch
from einops import rearrange

from opencood.utils import box_utils
from opencood.utils.type_utils import type_check_torch_tensor
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.utils.geometry import torch_apply_se3_
from opencood.utils.common_utils import measure_tensor_size_in_bytes


class VoxelPostprocessorMulti(object):
    def __init__(self, anchor_params, train=True) -> None:
        self.params = anchor_params
        self.train = train
        assert self.params['order'] == 'hwl', f"{self.params['order']} != 'hwl'"
        self.anchor_num_per_position = self.params['anchor_args']['num_per_position']
        self.num_classes = self.params['num_classes']
        self.anchor_num_total = self.anchor_num_per_position * self.num_classes

        if 'feature_stride' in self.params['anchor_args']:
            self.feature_stride = self.params['anchor_args']['feature_stride']
        else:
            print('WARNING | @VoxelPostprocessorMulti, feature_stride is set to the default value of 2')
            self.feature_stride = 2

        # ---
        # anchor spatial shape
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']
        self.anchor_hw = (H // self.feature_stride, W // self.feature_stride)

    def generate_gt_bbx(self, batch: dict) -> Tuple[torch.Tensor]:
        """
        This function is called by fnc dataset.postprocess to get ground truth bboxes & classes

        Parameter:
        ---------
        batch : 
            output of dataset.collate_batch_test
            NOTE: while testing, batch size always = 1
            {
                metadata: {
                    gt_boxes_in_ego: (M, 7)
                    gt_class: (M,)
                }
                ...
            }

        Returns:
        -------
        gt_boxes_in_top : 
            (M, 7) - [cx, cy, cz, l, w, h, yaw] in TOP

        gt_class :
            (M,) - gt_class[i] in (1, 2, 3)

        gt_visibility :
            (M, max_num_agents) - bool, [i, j] = True if gt_i is visible to agent_j
        """
        if 'gt_boxes_in_top' in batch['metadata']:
            gt_boxes = batch['metadata']['gt_boxes_in_top']
        elif 'gt_boxes_in_ego' in batch['metadata']:
            gt_boxes = batch['metadata']['gt_boxes_in_ego']
        else:
            raise ValueError("don't know where to find gt_boxes")
        
        return gt_boxes, batch['metadata']['gt_class'], \
            batch['metadata']['gt_visibility']
    
    def generate_anchor_box(self) -> np.ndarray:
        """
        Return
        ------

        anchors:
            (h, w, num_cls, num_anchor_per_position, 7)
        """
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        anchor_lwh = self.params['anchor_args']['anchor_lwh']  # list[list[float]]
        
        r = self.params['anchor_args']['r']
        assert self.anchor_num_per_position == len(r)
        r = [math.radians(ele) for ele in r]

        voxel_h = self.params['anchor_args']['vh']
        voxel_w = self.params['anchor_args']['vw']

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        feature_stride = self.feature_stride

        x = np.linspace(xrange[0] + voxel_w, xrange[1] - voxel_w, W // feature_stride)
        y = np.linspace(yrange[0] + voxel_h, yrange[1] - voxel_h, H // feature_stride)

        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num_total)  # (H/str, W/str, anchor_num_total)
        cy = np.tile(cy[..., np.newaxis], self.anchor_num_total)  # (H/str, W/str, anchor_num_total)
        cz = np.ones_like(cx) * -3.0  # (H/str, W/str, anchor_num_total)

        cx = rearrange(cx, 'h w (n_cls n_anchor) -> h w n_cls n_anchor', n_cls=self.num_classes)
        cy = rearrange(cy, 'h w (n_cls n_anchor) -> h w n_cls n_anchor', n_cls=self.num_classes)
        cz = rearrange(cz, 'h w (n_cls n_anchor) -> h w n_cls n_anchor', n_cls=self.num_classes)

        l = np.ones_like(cx)
        w = np.ones_like(cx)
        h = np.ones_like(cx)
        for cls_idx, cls_lwh in enumerate(anchor_lwh):
            l[:, :, cls_idx, :] *= cls_lwh[0]
            w[:, :, cls_idx, :] *= cls_lwh[1]
            h[:, :, cls_idx, :] *= cls_lwh[2]

        r_ = np.ones_like(cx)  # (h, w, n_cls, n_anchor)
        for i, angle in enumerate(r):
            r_[:, :, :, i] = angle

        anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1)        
        # anchors: (h, w, n_cls, n_anchor, 7)

        return anchors

    def generate_label_1cls(self, gt_box_center: np.ndarray, anchors: np.ndarray) -> dict:
        """
        Generate targets for training.

        Parameters
        ----------
        gt_box_center:(n, 7) | already filtered out invalid
        anchor:(h, w, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            {
                'pos_equal_one': (h, w, 2) - classification target  
                'neg_equal_one': (h, w, 2) - classification target
                'targets': (h, w, anchor_num_per_position * 7) - regression target
            }
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        
        feature_map_shape = anchors.shape[:2]  # (h, w)

        anchors = rearrange(anchors, 'h w anchor_num C -> (h w anchor_num) C', C=7)
        # normalization factor
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)  # (h * w * anchor_num,)

        if gt_box_center.shape[0] == 0:
            # this class doesn't have any ground truth box
            # classification
            pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num_per_position))  # (h, w, 2)
            neg_equal_one = np.ones((*feature_map_shape, self.anchor_num_per_position))  # (h, w, 2)
            # regression
            targets = np.zeros((*feature_map_shape, self.anchor_num_per_position * 7))  # (h, w, anchor_num * 7)    

            label_dict = {'pos_equal_one': pos_equal_one,
                          'neg_equal_one': neg_equal_one,
                          'targets': targets}
            return label_dict

        # ----------
        # init label
        # classification
        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num_per_position))  # (h, w, 2)
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num_per_position))  # (h, w, 2)
        # regression
        targets = np.zeros((*feature_map_shape, self.anchor_num_per_position * 7))  # (h, w, anchor_num * 7)
        
        # --------------------------
        # find positive & negative anchors
        #   convert gt & anchors from bbox to 8-corner form
        #   compute axis-aligned IOU in BEV
        # --------------------------
        gt_box_corner_valid = box_utils.boxes_to_corners_3d(
            gt_box_center, 
            self.params['order']
        )  # (n, 8, 3)
               
        anchors_corner = box_utils.boxes_to_corners_3d(
            anchors,
            self.params['order']
        )  # (h * w * anchor_num, 8, 3)

        
        anchors_standup_2d = box_utils.corner2d_to_standup_box(anchors_corner)  # (h * w * anchor_num, 4)
        
        gt_standup_2d = box_utils.corner2d_to_standup_box(gt_box_corner_valid)  # (n, 4)

        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )  # (h * w * anchor_num, n)

        # the anchor boxes has the largest iou across
        # shape: (n)
        id_highest = np.argmax(iou.T, axis=1)
        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = \
            np.where(iou >
                     self.params['target_args']['pos_threshold'])
        #  find anchors iou < params['neg_iou']
        id_neg = np.where(np.sum(iou <
                                 self.params['target_args']['neg_threshold'],
                                 axis=1) == iou.shape[1])[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, self.anchor_num_per_position))
        pos_equal_one[index_x, index_y, index_z] = 1

        # calculate the targets
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[
                id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, self.anchor_num_per_position))
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, self.anchor_num_per_position))
        neg_equal_one[index_x, index_y, index_z] = 0

        label_dict = {'pos_equal_one': pos_equal_one,
                      'neg_equal_one': neg_equal_one,
                      'targets': targets}

        return label_dict
    
    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), 
            gt_class: (max_num, 7)
            mask: (max_num,) - =1 for valid gt, =0 for padded box (to uniformnize #gt per data sample in batch)
            anchor: (h, w, n_cls, n_anchor, 7) with h = H / feature_stride, w = W / feature_stride

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        mask_valid = kwargs['mask'] == 1
        gt_boxes = kwargs['gt_box_center'][mask_valid]  # (N, 7)
        gt_class = kwargs['object_class'][mask_valid]  # (N,)
        anchors = kwargs['anchors']  # (h w n_cls n_anchor, 7) - output of self.generate_anchor_box

        label_dict = {
            'pos_equal_one': list(),
            'neg_equal_one': list(),
            'targets': list(),
        }
        
        for cls_idx in range(1, self.num_classes + 1):
            mask_this_cls = gt_class == cls_idx
            label_dict_cls = self.generate_label_1cls(gt_boxes[mask_this_cls], anchors[:, :, cls_idx - 1])
            # NOTE: absent class, which is class having no ground truth, is handled inside self.generate_label_1cls
            
            # add label of this cls to label_dict
            label_dict['pos_equal_one'].append(label_dict_cls['pos_equal_one'])
            label_dict['neg_equal_one'].append(label_dict_cls['neg_equal_one'])
            label_dict['targets'].append(label_dict_cls['targets'])

        label_dict['pos_equal_one'] = np.stack(label_dict['pos_equal_one'], axis=-1)  # (h, w, 2, n_cls)
        label_dict['neg_equal_one'] = np.stack(label_dict['neg_equal_one'], axis=-1)  # (h, w, 2, n_cls)
        label_dict['targets'] = np.stack(label_dict['targets'], axis=-1)  # (h, w, anchor_num_per_position * 7, n_cls)

        return label_dict

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = \
            torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = \
            torch.from_numpy(np.array(neg_equal_one))
        targets = \
            torch.from_numpy(np.array(targets))

        return {'targets': targets,  # (B, h, w, anchor_num_per_position * 7, n_cls)
                'pos_equal_one': pos_equal_one,  # (B, h, w, 2, n_cls)
                'neg_equal_one': neg_equal_one,  # (B, h, w, 2, n_cls)
                }

    def post_process_1cls(self, 
                          cls_idx: int,
                          data_dict: dict, 
                          output_dict: dict,
                          score_threshold: float,
                          nms_thresh: float,
                          bandwidth_dict: dict = None) -> Tuple[torch.Tensor]:
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        NOTE: only eval for ego

        Parameters
        ----------
        cls_idx:
            in (1, 2, 3)

        data_dict :
            output of dataset.__getitem__
            {
                cav_id : {
                    'anchor_box': np.ndarray - (h, w, n_cls, n_anchor, 7) 

                    NOTE: anchor_box is used in __getitem__ => not batched!
                    ...
                }
            }
        
        output_dict :
            output of model.forward()
            {
                cav_id : {
                    prob : torch.Tensor - (B, num_cls, num_anchor, h, w) - probability, not logits
                    reg : torch.Tensor - (B, num_cls, 7*num_anchor, h, w) - regression map
                }
            }

        score_threshold :
            score threshold of this class

        nms_thresh : 
            nms thresh of this class

        Returns
        -------
        pred_boxes : torch.Tensor
            (N, 7)

        pred_scores : torch.Tensor
            (N,)
        """
        assert cls_idx in (1, 2, 3), "1=car, 2=bike, 3=ped"
        _class_index = cls_idx - 1  # use this to index into anchors and output
        
        # init output
        pred_boxes = list()
        pred_scores = list()

        for agent_name, agent_content in data_dict.items():
            if agent_name == 'metadata':
                continue
            assert agent_name in output_dict, f"keys of output_dict: {list(output_dict.keys())}"
            # anchor_box of 1 class
            anchor_box = agent_content['anchor_box'][:, :, _class_index]  # (H, W, anchor_num, 7)
            type_check_torch_tensor(anchor_box, strict=True)

            # classification probability
            prob = output_dict[agent_name]['prob'][:, _class_index]
            prob = rearrange(prob, 'B num_anchor h w -> 1 (B h w num_anchor)', 
                             B=1)  # add B=1 to assure we have the right dim

            # convert regression map back to bounding box
            reg = output_dict[agent_name]['rm'][:, _class_index]
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)  
            # NOTE: batch_box3d is in 'hwl' !
            
            mask_in_range = torch.gt(prob, score_threshold)  # (1, B * h * w * num_anchor)
            mask_reg = mask_in_range.unsqueeze(2).repeat(1, 1, 7)  # (1, B * h * w * num_anchor, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask_in_range[0])

            # convert output to bounding box
            if boxes3d.shape[0] > 0:
                if self.params['order'] == 'hwl':
                    box_utils.bbox_hwl_to_lwh(boxes3d)  # (N, 7)

                # transform boxes3d from agent frame to TOP frame
                transformation_matrix = agent_content['transformation_matrix']
                type_check_torch_tensor(transformation_matrix, strict=True)
                assert transformation_matrix.shape == (4, 4), f"{transformation_matrix.shape} != (4, 4)"
                torch_apply_se3_(transformation_matrix, boxes_=boxes3d)

                pred_boxes.append(boxes3d)
                pred_scores.append(scores)

                if isinstance(bandwidth_dict, dict):
                    bandwidth_dict['num_cavs'] += 1

        if len(pred_boxes) == 0:
            return None, None
        
        pred_boxes = torch.cat(pred_boxes, dim=0)
        pred_scores = torch.cat(pred_scores, dim=0)
        
        if isinstance(bandwidth_dict, dict):
            bandwidth_dict['size_exchanged_info'] += measure_tensor_size_in_bytes(pred_boxes)
            bandwidth_dict['size_exchanged_info'] += measure_tensor_size_in_bytes(pred_scores)

        # nms
        keep_index = box_utils.nms_rotated_gpu(pred_boxes, pred_scores, nms_thresh)
        pred_boxes = pred_boxes[keep_index]
        pred_scores = pred_scores[keep_index]

        # filter the prediction out of the range.
        mask_in_range = box_utils.get_mask_center_in_range(pred_boxes, self.params['anchor_args']['cav_lidar_range'])
        pred_boxes = pred_boxes[mask_in_range]
        pred_scores = pred_scores[mask_in_range]
        
        return pred_boxes, pred_scores

    def post_process(self, data_dict: dict, output_dict: dict, bandwidth_dict: dict = None) -> Tuple[torch.Tensor]:
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        NOTE: only eval for ego

        Parameters
        ----------
        data_dict :
            output of dataset.__getitem__
            {
                cav_id : {
                    'anchor_box': np.ndarray - (h, w, n_cls, n_anchor, 7) 

                    NOTE: anchor_box is used in __getitem__ => not batched!
                    ...
                }
            }
        
        output_dict :
            output of model.forward()
            {
                cav_id : 
                {
                    prob: (B, num_cls, num_anchor, h, w), PROBABILITY => the normalization be done in the model
                    rm: (B, num_cls, 7*num_anchor, h, w) - regression map
                }
            }

        Returns
        -------
        pred_boxes : torch.Tensor
            (N, 7)

        pred_scores : torch.Tensor
            (N,)

        pred_class : torch.Tensor
            (N,)
        """
        score_threshold_multi = self.params['score_threshold']
        nms_thresh_multi = self.params['nms_thresh']

        all_pred_boxes = list()
        all_pred_scores = list()
        all_pred_class = list()

        for cls_idx, score_thresh, nms_thresh in \
            zip(range(1, self.num_classes + 1), score_threshold_multi, nms_thresh_multi):

            pred_boxes, pred_scores = self.post_process_1cls(
                cls_idx, data_dict, output_dict, score_thresh, nms_thresh, bandwidth_dict=bandwidth_dict)
            # NOTE: here nms is done for pred_boxes of each class
            # NOTE: => possible to have 1 car and 1 ped occupy the same space

            if isinstance(pred_boxes, torch.Tensor):  # if there is no pred_boxes of this cls, pred_box3d is None
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)
                all_pred_class.append(torch.zeros_like(pred_scores).long() + cls_idx)

        if len(all_pred_boxes) > 0:
            all_pred_boxes = torch.cat(all_pred_boxes)
            all_pred_scores = torch.cat(all_pred_scores)
            all_pred_class = torch.cat(all_pred_class)
        else:
            all_pred_boxes = torch.zeros(0, 7)
            all_pred_scores = torch.zeros(0)
            all_pred_class = torch.zeros(0).long()
            print('WARNING | @voxel_postprocessor_multi : there is no prediction whose score > threshold')

        return all_pred_boxes, all_pred_scores, all_pred_class

    @staticmethod
    def delta_to_boxes3d(deltas, anchors, channel_swap=True):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr
        channel_swap : bool
            Whether to swap the channel of deltas. It is only false when using
            FPV-RCNN

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        if channel_swap:
            deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        else:
            deltas = deltas.contiguous().view(N, -1, 7)

        boxes3d = torch.zeros_like(deltas)
        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        anchors_reshaped = rearrange(anchors, 'W L n_anchor C -> (W L n_anchor) C').float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d
