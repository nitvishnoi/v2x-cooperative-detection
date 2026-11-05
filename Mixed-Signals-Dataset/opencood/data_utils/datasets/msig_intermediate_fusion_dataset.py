import numpy as np
import torch
from typing import Optional

from opencood.tools.type_def import *
from opencood.data_utils.datasets.msig_base_dataset import MSigBaseDataset
from opencood.tools.point_cloud_tools import remove_points_outside_range
from opencood.data_utils.augmentor import augment_utils

from mixedsignals.utils.geometry import apply_se3_


class MSigIntermediateFusion(MSigBaseDataset):
    def __init__(self, params: Dict, **kwargs):
        super().__init__(params)

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.proj_first = True

        # scene-level flip augmentation (train only): the same random flip is
        # applied to every agent's points AND the shared gt_boxes so the scene
        # stays geometrically consistent. Configure via `data_augment: [{NAME:
        # random_world_flip_x}, {NAME: random_world_flip_y}]` in the hypes yaml.
        augment_names = {cfg['NAME'] for cfg in params.get('data_augment', []) if 'NAME' in cfg}
        self._augment_flip_x = 'random_world_flip_x' in augment_names
        self._augment_flip_y = 'random_world_flip_y' in augment_names

        # Node dropout (train only): the labeled sequences almost always have
        # 3-5 connected nodes (top+dome RSU pair plus 1-3 CAVs), but most
        # frames in the competition's own test set only have the RSU pair -
        # no CAV in communication range at all. The model never saw that
        # case during training, so it generalizes poorly to it. This drops
        # each present CAV independently at random (RSU pair always kept)
        # so training sees the same mix of node counts the real test set
        # does. Configure via `data_augment: [{NAME: random_node_dropout}]`
        # and `node_dropout_p`. The pre-rename names (random_agent_dropout,
        # agent_dropout_p) are still accepted so older run configs load.
        self._augment_node_dropout = bool(
            {'random_node_dropout', 'random_agent_dropout'} & set(augment_names))
        self._node_dropout_p = float(params.get('node_dropout_p', params.get('agent_dropout_p', 0.7)))

        # Diagnostic-only, off by default: force every sample down to just
        # the RSU pair regardless of train/eval mode, to directly measure
        # accuracy in the no-CAV case (set externally by eval scripts, not
        # driven by hypes yaml).
        self.force_rsu_only = False

        # ===
        print(f'MSigIntermediateFusion | load gt infos')
        self.load_gt_infos()

        print(f'MSigIntermediateFusion | building pre- & post-processor for training:={self.is_train}')
        self.pre_processor = self._build_preprocessor(params['preprocess'])
        self.post_processor = self._build_postprocessor(params['postprocess'])
        self._cav_lidar_range = np.array(params['preprocess']['cav_lidar_range'])
        
        self._anchor_box = self.post_processor.generate_anchor_box()

    @staticmethod
    def convert_list_of_dict_to_dict_of_list(list_dicts: List[Dict]) -> Dict:
        output = OrderedDict()

        for i in range(len(list_dicts)):
            for feature_name, feature in list_dicts[i].items():
                if feature_name not in output:
                    output[feature_name] = list()
                if isinstance(feature, list):
                    output[feature_name] += feature
                else:
                    output[feature_name].append(feature)

        return output
    
    def init_output_of_getitem(self):
        data = {
            'ego': {
                'processed_lidar': list(),
                'cav_num': 0,
                # legacy of the origin OpenCOOD
                'velocity': list(),
                'time_delay': list(),
                'infra': list(),
                'spatial_correction_matrix': list(),
                'pairwise_t_matrix': list()
            }
        }

        if self.is_train:
            data['ego'].update({'label_dict': dict()})
        else:
            data['ego'].update({'anchor_box': np.array([])})
            data.update({
                'metadata': {
                    'gt_boxes_in_ego': np.zeros([0, 7]),
                    'gt_class': np.zeros([0]),
                    'points': np.zeros([0, 5])  # point-4, agent_indicator
                }
            })

        return data

    def _sample_node_dropout(self, dict_agents_pts_poses: Dict) -> Dict:
        if self.force_rsu_only:
            return {node_name: node_data for node_name, node_data in dict_agents_pts_poses.items()
                    if node_name in self.name_rsu}

        if not self.is_train or not self._augment_node_dropout:
            return dict_agents_pts_poses

        kept = {
            node_name: node_data
            for node_name, node_data in dict_agents_pts_poses.items()
            if node_name in self.name_rsu or np.random.random() >= self._node_dropout_p
        }
        # name_rsu (top, dome) is always present in dict_agents_pts_poses, so
        # kept can never drop below 2.
        return kept

    def _sample_scene_augmentation(self) -> Optional[Dict[str, bool]]:
        """
        Decide the flip(s) to apply to this sample - same decision is used for
        every agent's points and for the shared gt_boxes, so the scene stays
        geometrically consistent. Returns None if no augmentation is configured
        or this is not a training split (matches DataAugmentorForMidFusion's
        own train-only behavior).
        """
        if not self.is_train or not (self._augment_flip_x or self._augment_flip_y):
            return None
        return {
            'flip_x': self._augment_flip_x and bool(np.random.choice([False, True])),
            'flip_y': self._augment_flip_y and bool(np.random.choice([False, True])),
        }

    @staticmethod
    def _apply_scene_augmentation_to_points(points: np.ndarray, aug: Dict[str, bool]) -> np.ndarray:
        # dummy_boxes is a throwaway placeholder so we can reuse augment_utils'
        # paired (gt_boxes, points) transform functions for a points-only call;
        # its own transformed value is discarded.
        dummy_boxes = np.array([[100., 100., 0., 1., 1., 1., 0.]])
        if aug['flip_x']:
            dummy_boxes, points = augment_utils.random_flip_along_x(dummy_boxes, points, enable=True)
        if aug['flip_y']:
            dummy_boxes, points = augment_utils.random_flip_along_y(dummy_boxes, points, enable=True)
        return points

    @staticmethod
    def _apply_scene_augmentation_to_boxes(gt_boxes: np.ndarray, aug: Dict[str, bool]) -> np.ndarray:
        gt_boxes = gt_boxes.copy()  # get_1sample_from_gt_infos returns a live reference into
                                     # the cached gt_infos dict - must not mutate it in place
        # dummy_points is a throwaway placeholder, symmetric to the points-only helper above.
        dummy_points = np.array([[100., 100., 100., 0.]])
        if aug['flip_x']:
            gt_boxes, dummy_points = augment_utils.random_flip_along_x(gt_boxes, dummy_points, enable=True)
        if aug['flip_y']:
            gt_boxes, dummy_points = augment_utils.random_flip_along_y(gt_boxes, dummy_points, enable=True)
        return gt_boxes

    def __getitem__(self, idx_in_dataset):
        seq_idx, idx_in_seq = self._convert_idx_in_dataset_to_idx_in_sequence(idx_in_dataset)
        idx_in_seq = self.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)

        dict_agents_pts_poses = self._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
        dict_agents_pts_poses = self._sample_node_dropout(dict_agents_pts_poses)
        gt_info_dict = self.get_1sample_from_gt_infos(seq_idx, idx_in_seq)
        scene_aug = self._sample_scene_augmentation()

        data = self.init_output_of_getitem()

        data['ego'], agg_points = self._get_projected_voxel_grid_of_every_agent(
            dict_agents_pts_poses, scene_aug)

        gt_boxes_in_top = gt_info_dict['gt_boxes_in_top']
        if scene_aug is not None:
            gt_boxes_in_top = self._apply_scene_augmentation_to_boxes(gt_boxes_in_top, scene_aug)

        gt_boxes, gt_class = self.preprocess_gt_boxes(
            gt_boxes_in_top,  # (N, 7)
            gt_info_dict['gt_class'],
            gt_info_dict['gt_visibility'],
            [self.convert_agent_name_to_indicator(agent_name) for agent_name in dict_agents_pts_poses.keys()]
        )

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
            data['metadata']['points'] = agg_points  # (N, 4+1) - x, y, z, i, agent_indicator | in frame of top
        
        return data

    def _get_projected_voxel_grid_of_every_agent(self, dict_agents_pts_poses: Dict[str, np.ndarray],
                                                 scene_aug: Optional[Dict[str, bool]] = None) \
        -> Tuple[Dict, np.ndarray]:
        """
        cav's lidar will first be projected to the ego's coordinate framem then voxelized
        """
        processed_lidar: List[Dict] = list()
        cav_num = 0
        velocity = [0. for _ in range(self.num_agents)]
        time_delay = [0. for _ in range(self.num_agents)]
        spatial_correction_matrix = np.stack([np.eye(4) for _ in range(self.num_agents)])
        pairwise_t_matrix = np.stack([np.eye(4) for _ in range(self.num_agents)])
        infra = [0. for _ in range(self.num_agents)]

        agg_points = list()

        _iter = 0
        for agent_name, agent_data in dict_agents_pts_poses.items():
            points = agent_data['points']
            ego_se3_agent = agent_data['ego_se3_agent']

            apply_se3_(ego_se3_agent, points_=points)

            if scene_aug is not None:
                points = self._apply_scene_augmentation_to_points(points, scene_aug)

            agent_processed_lidar, points = self.hard_voxelize_point_cloud(points)

            processed_lidar.append(agent_processed_lidar)
            cav_num += 1

            if agent_name in self.name_rsu:
                infra[_iter] = 1.0

            points = np.pad(points, pad_width=[(0, 0), (0, 1)], 
                            constant_values=self.convert_agent_name_to_indicator(agent_name))
            agg_points.append(points)

            _iter += 1
        
        processed_lidar = self.convert_list_of_dict_to_dict_of_list(processed_lidar)
        
        return {
            'processed_lidar': processed_lidar,
            'cav_num': cav_num,
            'velocity': velocity,
            'time_delay': time_delay,
            'infra': infra,
            'spatial_correction_matrix': spatial_correction_matrix,
            'pairwise_t_matrix': pairwise_t_matrix,
        }, np.concatenate(agg_points)
    
    def collate_batch(self, batch):
        output_dict = {'ego': {}}

        processed_lidar_list = []
        record_len = []

        # used for PriorEncoding for models
        velocity = []
        time_delay = []
        infra = []

        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            
            processed_lidar_list.append(ego_dict['processed_lidar'])
            record_len.append(ego_dict['cav_num'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            
            spatial_correction_matrix_list.append(
                ego_dict['spatial_correction_matrix']
            )

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.convert_list_of_dict_to_dict_of_list(processed_lidar_list)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))

        # (B, max_cav)
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = \
            torch.from_numpy(np.array(spatial_correction_matrix_list))
        # (B, max_cav, 3)
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()
        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({
            'processed_lidar': processed_lidar_torch_dict,
            'record_len': record_len,
            'prior_encoding': prior_encoding,
            'spatial_correction_matrix': spatial_correction_matrix_list,
            'pairwise_t_matrix': pairwise_t_matrix,
            'transformation_matrix': torch.from_numpy(np.identity(4)).float()})

        if self.is_train:
            label_torch_dict = self.post_processor.collate_batch(
                [data['ego']['label_dict'] for data in batch])
            
            output_dict['ego'].update({
                'label_dict': label_torch_dict,
            })
        else:
            metadata = {
                'points': list(),
                'gt_boxes_in_ego': list(),
                'gt_class': list(),
            }

            output_dict['ego']['anchor_box'] = \
                torch.from_numpy(batch[0]['ego']['anchor_box']).float()

            for b_idx, data in enumerate(batch):
                metadata['gt_boxes_in_ego'].append(data['metadata']['gt_boxes_in_ego'])
                metadata['gt_class'].append(data['metadata']['gt_class'])
                metadata['points'].append(data['metadata']['points'])
            
            metadata['gt_boxes_in_ego'] = torch.from_numpy(
                np.concatenate(metadata['gt_boxes_in_ego'])).float()
            
            metadata['gt_class'] = torch.from_numpy(
                np.concatenate(metadata['gt_class'])).long()
            
            metadata['points'] = torch.from_numpy(
                np.concatenate(metadata['points']))
            
            output_dict['metadata'] = metadata

        return output_dict

    def collate_batch_train(self, batch):
        return self.collate_batch(batch)
    
    def collate_batch_test(self, batch):
        return self.collate_batch(batch)
