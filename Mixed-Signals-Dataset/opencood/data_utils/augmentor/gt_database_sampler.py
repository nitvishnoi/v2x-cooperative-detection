import pickle
from pathlib import Path
from typing import List, Tuple
from collections import OrderedDict

import numpy as np

from opencood.utils.box_overlaps import bbox_overlaps
from opencood.utils import box_utils
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu


class GtSampler(object):
    def __init__(self, config: dict):
        self.root_gt_database = Path(config['root_gt_database'])
        self.name_classes = ['car', 'bike', 'pedestrian']
        self._cls_idx_to_name = ['dummy',] + self.name_classes  # pad dummy to have idx=1 points to 'car'

        self._iou_thresh = 0.2  # to determine if two boxes overlap

        self.dir_classes = [self.root_gt_database / f"{cls_name}_database" for cls_name in self.name_classes]
        for dir_cls in self.dir_classes:
            assert dir_cls.exists(), f"{dir_cls} does not exist"

        with open(self.root_gt_database / 'database_infos.pkl', 'rb') as f:
            self.database_infos = pickle.load(f)

        self.target_num_gt = config['target_num_gt']  # (num_car, num_bike, num_ped)
        assert len(self.target_num_gt) == len(self.name_classes), \
            f"{len(self.target_num_gt)} != {len(self.name_classes)}"
        
        self.target_num_gt = OrderedDict(dict(zip(self.name_classes, self.target_num_gt)))
        
        self._indices = OrderedDict(dict([
            (cls_name, np.arange(len(self.database_infos[cls_name]))) 
            for cls_name in self.name_classes
        ]))
        for cls_name in self.name_classes:
            np.random.shuffle(self._indices[cls_name])

        self._pointers = OrderedDict(dict([(cls_name, 0) for cls_name in self.name_classes]))
    
    def take_sample(self, cls_name: str, num_sample_to_take: int) -> np.ndarray:
        assert num_sample_to_take > 0, f"{num_sample_to_take} !> 0"
        
        pointer = self._pointers[cls_name]
        out = self._indices[cls_name][pointer: pointer + num_sample_to_take]

        # move pointer forward
        self._pointers[cls_name] += num_sample_to_take
        
        # check if there are sufficient gt to sample
        # if there aren't, re-shuffle & reset pointer
        if self._pointers[cls_name] > len(self.database_infos[cls_name]) - self.target_num_gt[cls_name]:
           np.random.shuffle(self._indices[cls_name])
           self._pointers[cls_name]  = 0
        
        return out
        
    def __call__(self, 
                 cur_gt_boxes: np.ndarray, 
                 cur_gt_class: np.ndarray, 
                 cur_agents_indices: List[int],
                 cur_agents_name: List[str]) -> Tuple[np.ndarray]:
        """
        
        Parameters:
        ----------
        cur_gt_boxes : 
            (M, 7) - [cx, cy, cz, l, w, h, theta], in TOP frame

        cur_gt_class :
            (M,)
        
        cur_agents_ids : 
            (num_agent) - index of agents that are presenting

        Returns:
        -------
        sample_gt_boxes :
            (B_tot, 7)

        sample_gt_class :
            (B_tot,)

        sample_points :
            (N, 4+1) - x, y, z, intensity, agent_indicator
        """
        assert cur_gt_class.shape[0] == cur_gt_boxes.shape[0], f"{cur_gt_class.shape[0]} != {cur_gt_boxes.shape[0]}"
        assert np.all(cur_gt_class > 0), f"gt_class must start from 1"

        unq_cls, cur_cls_counts = np.unique(cur_gt_class.astype(int), return_counts=True)

        candidate_infos: List[dict] = list()

        for cls_idx, num_cur in zip(unq_cls, cur_cls_counts):
            cls_name = self._cls_idx_to_name[cls_idx]

            # determine num of gt to sample for this class
            num_sample_to_take = self.target_num_gt[cls_name] - num_cur
            if num_sample_to_take < 1:
                continue

            indices = self.take_sample(cls_name, num_sample_to_take)
            for ii in indices:
                candidate_infos.append(self.database_infos[cls_name][ii])

        candidate_boxes = np.stack([info['box'] for info in candidate_infos], axis=0)  # (B, 7)    

        # -------------------------------------------------
        # determine valid boxes <-no overlap with cur_gt_boxes
        candidate_boxes, indices_valid_candidates = self.filter_by_overlap(candidate_boxes, cur_gt_boxes)
        if candidate_boxes.shape[0] == 0:
            # there is no valid candidate
            return np.array([]), np.array([]), dict()
        
        # shuffle candidate_boxes to add variety 
        # to avoid prioritizing cars in the subsequence step of filter_self_overlap
        _indices = np.arange(candidate_boxes.shape[0])
        np.random.shuffle(_indices)
        candidate_boxes = candidate_boxes[_indices]
        indices_valid_candidates = indices_valid_candidates[_indices]

        # keep valid infos
        candidate_infos = [candidate_infos[ii] for ii in indices_valid_candidates]  # (B,)
        
        # -------------------------------------------------
        # determine valid boxes <-no overlap with other candidates
        candidate_boxes, indices_valid_candidates = self.filter_self_overlap(candidate_boxes)
        if candidate_boxes.shape[0] == 0:
            # there is no valid candidate
            return np.array([]), np.array([]), dict()
        
        # keep valid infos
        candidate_infos = [candidate_infos[ii] for ii in indices_valid_candidates]  # (B,)

        # -------------------------------------------------
        # process points
        valid_candidate_indices = list()
        valid_candidate_class = list()
        all_agent_points = dict([(name, list()) for name in cur_agents_name])
        
        _agents_indices = np.array(cur_agents_indices).reshape(1, -1)  # (1, n_cur_agents)

        for idx, info in enumerate(candidate_infos):
            pts = np.load(info['path_points'])  # (n,4+1)
            
            # check if these pts come from cur_agents_indices
            pts_indicator = pts[:, -1].astype(int)  # (n,)
            mask_from_cur_agents = pts_indicator.reshape(-1, 1) == _agents_indices  # (n, n_cur_agents)
            
            is_this_box_valid = np.any(mask_from_cur_agents.any(axis=1))
            if is_this_box_valid:
                valid_candidate_indices.append(idx)
                valid_candidate_class.append(info['cls_idx'])

                for _ii, agent_name in enumerate(cur_agents_name):
                    all_agent_points[agent_name].append(pts[mask_from_cur_agents[:, _ii]])
        
        if len(valid_candidate_indices) > 0:
            candidate_boxes = candidate_boxes[np.array(valid_candidate_indices)]
            valid_candidate_class = np.array(valid_candidate_class)
            for agent_name in all_agent_points.keys():
                all_agent_points[agent_name] = np.concatenate(all_agent_points[agent_name])
            
            return candidate_boxes, valid_candidate_class, all_agent_points
        else:
            # there is no valid candidate
            return np.array([]), np.array([]), all_agent_points

    def filter_by_overlap(self, 
                          candidate_boxes: np.ndarray, 
                          gt_boxes: np.ndarray) -> Tuple[np.ndarray]:
        iou = self.compute_axis_aligned_iou(candidate_boxes, gt_boxes)  # (B, M)
        
        candidate_max_iou = np.max(iou, axis=1)
        mask_valid_candidates = candidate_max_iou < self._iou_thresh
        indices_valid_candidates = np.arange(candidate_boxes.shape[0])[mask_valid_candidates]
        
        return candidate_boxes[mask_valid_candidates], indices_valid_candidates
    
    def filter_self_overlap(self, boxes) -> Tuple[np.ndarray]:
        iou = self.compute_axis_aligned_iou(boxes, boxes)  # (B, M)

        # zero-out diagonal to avoid excluding every box
        indices = np.arange(boxes.shape[0])
        iou[indices, indices] = 0.0

        # chose box 1-by-1, following NonMaxSuppression style
        chosen_indices = [0,]
        for box_idx in range(1, boxes.shape[0]):
            overlap_with_chosen = iou[box_idx, chosen_indices]
            if np.max(overlap_with_chosen) < self._iou_thresh:
                # valid
                chosen_indices.append(box_idx)
        return boxes[chosen_indices], chosen_indices

    @staticmethod
    def compute_axis_aligned_iou(boxes1: np.ndarray, boxes2: np.ndarray, order='lwh') -> np.ndarray:
        """
        Parameters:
        ----------
        boxes1 :
            (M1, 7)

        boxes2 :
            (M2, 7)

        Return:
        ------
        iou :
            (M1, M2)
        """
        return boxes_bev_iou_cpu(boxes1, boxes2)
        # boxes1_corners = box_utils.boxes_to_corners_3d(boxes1, order)
        # boxes1_standup_2d = box_utils.corner2d_to_standup_box(boxes1_corners)  # (M1 ,4)

        # boxes2_corners = box_utils.boxes_to_corners_3d(boxes2, order)
        # boxes2_standup_2d = box_utils.corner2d_to_standup_box(boxes2_corners)  # (M2 ,4)

        # iou = bbox_overlaps(
        #     np.ascontiguousarray(boxes1_standup_2d).astype(np.float32),
        #     np.ascontiguousarray(boxes2_standup_2d).astype(np.float32),
        # )  # (M1, M2)

        # assert iou.shape == (boxes1.shape[0], boxes2.shape[0]), f"{iou.shape} != {(boxes1.shape[0], boxes2.shape[0])}"
        # return iou
    
