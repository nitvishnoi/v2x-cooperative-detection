import numpy as np
import torch

from opencood.tools.type_def import *
from opencood.data_utils.datasets.msig_base_dataset import MSigBaseDataset
from opencood.tools.point_cloud_tools import remove_points_outside_range

from mixedsignals.utils.geometry import apply_se3_


class MSigLateFusion(MSigBaseDataset):
    def __init__(self, params: Dict, flag_make_laly_database: bool = False, **kwargs):
        super().__init__(params)
        
        self._flag_make_laly_database = flag_make_laly_database
        if self._flag_make_laly_database:
            print(f"MSigLateFusion | _flag_make_laly_database: True |"
                    " use this only to make prediction database for Laly fusion")
        # ===
        print(f'MSigLateFusion | load gt infos')
        self.load_gt_infos()

        print(f'MSigLateFusion | building pre- & post-processor for training:={self.is_train}')
        self.pre_processor = self._build_preprocessor(params['preprocess'])
        self.post_processor = self._build_postprocessor(params['postprocess'])
        self._cav_lidar_range = np.array(params['preprocess']['cav_lidar_range'])
        
        self._anchor_box = self.post_processor.generate_anchor_box()
        
    def init_output_of_getitem(self, avail_agents_names: List[str] = '') -> Dict:
        if self.is_train:
            # data contains only 1 cav that is randomly chosen to be ego
            data = {
                'ego': {
                    'processed_lidar': dict(),
                    'label_dict': dict(),
                }
            }
        else:
            # data contains every cav that is available
            assert len(avail_agents_names) > 0
            data = dict()
            for agent_name in avail_agents_names:
                data[agent_name] = {
                    'processed_lidar': dict(),
                    'anchor_box': np.array([], dtype=np.float32),
                    'transformation_matrix': np.array([], dtype=np.float32),  # top_se3_agent, 
                    # transformation_matrix is to be used by PostProcessor
                }

            data.update({
                'metadata': {
                    'gt_boxes_in_ego': np.zeros([0, 7]),
                    'gt_class': np.zeros([0]),
                    'points': np.zeros([0, 5])  # point-4, agent_indicator
                }
            })

        return data
    
    def __getitem__(self, idx_in_dataset):
        """
        if self.is_train:
            data contains only 1 cav that is randomly chosen to be ego
        else:
            data contains every agent that is available 
        """
        seq_idx, idx_in_seq = self._convert_idx_in_dataset_to_idx_in_sequence(idx_in_dataset)
        idx_in_seq = self.check_idx_in_seq_has_gt_boxes(seq_idx, idx_in_seq)

        dict_agents_pts_poses = self._get_all_pointclouds_in_1sample(seq_idx, idx_in_seq)
        gt_info_dict = self.get_1sample_from_gt_infos(seq_idx, idx_in_seq)

        if self.is_train:
            return self._get_item_train(dict_agents_pts_poses, gt_info_dict)
        else:
            return self._get_item_test(dict_agents_pts_poses, gt_info_dict)

    def _get_item_train(self, dict_agents_pts_poses: Dict, gt_info_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Randomly choose 1 agent as the ego agent to make data for training individual detection model
        
        Args:
            dict_agents_pts_poses: {
                agent_name: {
                    points: (N, 4) - x, y, z, intensity | in agent's frame
                    ego_se3_agent: (4, 4) - pose of agent in the top's frame (ego:=top)
                }
            }

            gt_info_dict: {
                gt_boxes_in_top: (N, 7)
                gt_class: (N,)
                gt_visibility: (N, num_agents)
            }

        Returns:
            chosen_agent_name: str
            points: (N, 4) - x, y, z, intensity | in agent's frame
            ego_se3_chosen_agent: (4, 4) - pose of agent in the top's frame (ego:=top)
        """
        def randomly_choose_1agent(list_agents_names: List[str]) -> str:
            _shuffled_inds = np.random.permutation(len(list_agents_names))
            chosen_agent_name = list_agents_names[_shuffled_inds[0]]
            return chosen_agent_name

        chosen_agent_name = randomly_choose_1agent(list(dict_agents_pts_poses.keys()))
        points = dict_agents_pts_poses[chosen_agent_name]['points']
        top_se3_agent = dict_agents_pts_poses[chosen_agent_name]['ego_se3_agent']
        
        data = self.init_output_of_getitem()
        data['ego']['processed_lidar'], points = self.hard_voxelize_point_cloud(points)

        # transform gt_boxes to agent's frame
        apply_se3_(np.linalg.inv(top_se3_agent), boxes_=gt_info_dict['gt_boxes_in_top'])

        gt_boxes, gt_class = self.preprocess_gt_boxes(
            gt_info_dict['gt_boxes_in_top'], 
            gt_info_dict['gt_class'], 
            gt_info_dict['gt_visibility'], 
            [self.convert_agent_name_to_indicator(chosen_agent_name),]
        )

        data['ego']['label_dict'] = self.post_processor.generate_label(
            gt_box_center=self.convert_boxes_from_lwh_to_hwl(gt_boxes),
            mask=np.ones(gt_boxes.shape[0]),  # legacy | to be compatible with post_processor
            object_class=gt_class,
            anchors=self._anchor_box,
        )

        return data

    def _get_item_test(self, dict_agents_pts_poses: Dict, gt_info_dict: Dict[str, np.ndarray]) -> Dict:
        data = self.init_output_of_getitem(list(dict_agents_pts_poses.keys()))
        data['metadata']['points'] = list()
        avail_agents_inds = list()

        for agent_name, agent_data in dict_agents_pts_poses.items():
            points = agent_data['points']
            ego_se3_agent = agent_data['ego_se3_agent']

            # crop points in by make_gt_info_range defined in the frame of top
            apply_se3_(ego_se3_agent, points_=points)
            points = remove_points_outside_range(points, self.make_gt_info_range)
            apply_se3_(np.linalg.inv(ego_se3_agent), points_=points)

            data[agent_name]['processed_lidar'], agent_points = \
                self.hard_voxelize_point_cloud(points)  # in agent's frame
            
            data[agent_name]['anchor_box'] = self._anchor_box  # in agent's frame
            data[agent_name]['transformation_matrix'] = agent_data['ego_se3_agent']

            # store agent_points for visualization
            agent_indi = self.convert_agent_name_to_indicator(agent_name)
            avail_agents_inds.append(agent_indi)
            agent_points = np.pad(agent_points, pad_width=[(0, 0), (0, 1)], constant_values=agent_indi)
            apply_se3_(agent_data['ego_se3_agent'], points_=agent_points)
            data['metadata']['points'].append(agent_points)

        data['metadata']['points'] = np.concatenate(data['metadata']['points'])

        gt_boxes, gt_class = self.preprocess_gt_boxes(
            gt_info_dict['gt_boxes_in_top'],  # (N, 7)
            gt_info_dict['gt_class'], 
            gt_info_dict['gt_visibility'], 
            avail_agents_inds
        )
        data['metadata']['gt_boxes_in_ego'] = gt_boxes
        data['metadata']['gt_class'] = gt_class

        return data
    
    def collate_batch_test(self, batch):
        assert len(batch) == 1, f"while eval, it is too complicated to support batch_size > 1"
        _data = batch[0]

        datas = dict()  # output

        for agent_name, agent_data in _data.items():
            if agent_name == 'metadata':
                continue

            processed_lidar = self.pre_processor.collate_batch([agent_data['processed_lidar'],])
            anchor_box = torch.from_numpy(agent_data['anchor_box']).float()
            transformation_matrix = torch.from_numpy(agent_data['transformation_matrix']).float()

            datas[agent_name] = {
                'processed_lidar': processed_lidar,
                'anchor_box': anchor_box,
                'transformation_matrix': transformation_matrix
            }

        datas['metadata'] = {
            'gt_boxes_in_ego': torch.from_numpy(_data['metadata']['gt_boxes_in_ego']).float(),
            'gt_class': torch.from_numpy(_data['metadata']['gt_class']).long(),
            'points': torch.from_numpy(_data['metadata']['points']).float(),
        }

        return datas
