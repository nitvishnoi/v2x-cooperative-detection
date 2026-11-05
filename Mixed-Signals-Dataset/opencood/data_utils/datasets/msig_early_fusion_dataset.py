import numpy as np
import torch

from opencood.tools.type_def import *
from opencood.data_utils.datasets.msig_base_dataset import MSigBaseDataset

from mixedsignals.utils.geometry import apply_se3_


class MSigEarlyFusion(MSigBaseDataset):
    def __init__(self, params: Dict,
                 flag_return_only_early_pc_and_gt: bool = False,
                 **kwargs):
        super().__init__(params)

        # ===
        self._flag_return_only_early_pc_and_gt = flag_return_only_early_pc_and_gt
        # this option is used only for generating gt_info.pkl using the script "msig_dataset_preprocessing.py"
        if self._flag_return_only_early_pc_and_gt:
            print(f"MSigEarlyFusion | _flag_return_only_early_pc_and_gt: True | use this only to generate gt_info")
        else:
            print(f'MSigEarlyFusion | load gt infos')
            self.load_gt_infos()

        # ===
        print(f'MSigEarlyFusion | building pre- & post-processor for training:={self.is_train}')
        self.pre_processor = self._build_preprocessor(params['preprocess'])
        self.post_processor = self._build_postprocessor(params['postprocess'])
        self._cav_lidar_range = np.array(params['preprocess']['cav_lidar_range'])
        
        self._anchor_box = self.post_processor.generate_anchor_box()

    def get_early_collab_pointcloud(self, seq_idx, idx_in_seq, flag_pad_agent_indicator: bool = False) \
        -> Tuple[np.ndarray]:
        dict_agents_pts_poses = self._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)

        agg_points = list()
        avail_agents_inds = list()  # keep track of which agents contrib to agg point clouds
        for agent_name, agent_pts_pose in dict_agents_pts_poses.items():
            agent_pc, ego_se3_agent = agent_pts_pose['points'], agent_pts_pose['ego_se3_agent']
            # agent_pc: (N, 4) - x, y, z, intensity
            
            # transform to the frame of the ego agent
            if agent_name != self.name_ego:
                apply_se3_(ego_se3_agent, points_=agent_pc)
            
            if flag_pad_agent_indicator:
                indicator = self.convert_agent_name_to_indicator(agent_name)
                agent_pc = np.pad(agent_pc, pad_width=[(0, 0), (0, 1)], constant_values=indicator)

            agg_points.append(agent_pc)
            avail_agents_inds.append(self.convert_agent_name_to_indicator(agent_name))

        agg_points = np.concatenate(agg_points)
        avail_agents_inds = np.array(avail_agents_inds)

        return agg_points, avail_agents_inds
    
    def init_output_of_getitem(self):
        data = {
            'ego': {'processed_lidar': dict()},
        }

        if self.is_train:
            data['ego'].update({'label_dict': dict()})
        else:
            data['ego'].update({'anchor_box': np.ndarray([], dtype=np.float32)})
            data.update({
                'metadata': {
                    'gt_boxes_in_ego': np.zeros([0, 7]),
                    'gt_class': np.zeros([0]),
                    'points': np.zeros([0, 4])
                }
            })
    
        return data
    
    def __getitem__(self, idx_in_dataset: int):
        """
        Note:
            agg_points: (N, 4) - x, y, z, i | in ego (i.e. top) frame
            gt_boxes: (N, 7+2) - box-7, class_id, track_is | in ego (i.e. top) frame
        """
        seq_idx, idx_in_seq = self._convert_idx_in_dataset_to_idx_in_sequence(idx_in_dataset)
        if self._flag_verbose:
            print(f'MSigEarlyFusion | seq_idx: {seq_idx} | original idx_in_seq: {idx_in_seq}')

        idx_in_seq = self.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)
        if self._flag_verbose:
            print(f'MSigEarlyFusion | checked-gt idx_in_seq: {idx_in_seq}')

        agg_points, avail_agents_inds = \
            self.get_early_collab_pointcloud(seq_idx, idx_in_seq, self.is_pad_agent_indicator)

        if self._flag_return_only_early_pc_and_gt:
            # NOTE: agg_points & gt_boxes are in ego frame
            gt_boxes = self.return_gt_boxes_in_ego(seq_idx, idx_in_seq)  # unprocessed gt_boxes
            return {
                'agg_points': agg_points,  # (N, 5) - x, y, z, i, agent_indicator
                'gt_boxes': gt_boxes,  # (M, 7+2) - box-7, class_id, track_id
                'seq_idx': seq_idx,
                'idx_in_seq': idx_in_seq,
            }

        # ===
        data = self.init_output_of_getitem()
        data['ego']['processed_lidar'], agg_points = self.hard_voxelize_point_cloud(agg_points)
        
        gt_info_dict = self.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        gt_boxes = gt_info_dict['gt_boxes_in_top']  # (N, 7)
        gt_class = gt_info_dict['gt_class']
        gt_visibility = gt_info_dict['gt_visibility']

        gt_boxes, gt_class = self.preprocess_gt_boxes(gt_boxes, gt_class, gt_visibility, avail_agents_inds)

        if self.is_train:
            data['ego']['label_dict'] = self.post_processor.generate_label(
                gt_box_center=self.convert_boxes_from_lwh_to_hwl(gt_boxes),
                mask=np.ones(gt_boxes.shape[0]),  # legacy | to be compatible with post_processor
                object_class=gt_class,
                anchors=self._anchor_box,
            )
        else:
            data['ego']['anchor_box'] = self._anchor_box
            data['metadata']['gt_boxes_in_ego'] = gt_boxes
            data['metadata']['gt_class'] = gt_class
            data['metadata']['points'] = agg_points
        return data
