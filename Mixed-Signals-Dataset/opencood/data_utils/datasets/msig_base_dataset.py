import pickle
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset

from opencood.tools.type_def import *

from mixedsignals.mixed_signals import MixedSignalsExplorer
from mixedsignals.utils.geometry import apply_se3_
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.tools.point_cloud_tools import remove_points_outside_range
from opencood.utils.pcd_utils import shuffle_points
from opencood.utils import box_utils


class MSigBaseDataset(Dataset):
    def __init__(self, params: Dict, **kwargs):
        """
        Base dataset class for MSig dataset.
        
        Args:
            data_root: Root directory of the dataset.
            flag_train: 
            flag_verbose: if true, there will a lot of printing
        """
        self._data_root = params['data_root']
        self._flag_train = params['flag_train']
        self._flag_verbose = params['flag_verbose']
        self._flag_pad_agent_indicator = params['flag_pad_agent_indicator']
        self._test_sequences = set((4, 5, 17, 21))

        self._msig = MixedSignalsExplorer(self._data_root, self._flag_verbose)
        self._available_seqs = np.array(self._msig.return_available_sequences())
        self._filter_sequences_()
        self._check_sequences_in_ascending_order()
        self._acc_seq_len = self._accumulate_seq_length()

        # === CONSTANTS
        self._name_ego = 'top'
        self._name_rsu = self._msig.name_rsu_lidars
        self._ego_se3_map = self._msig.top_se3_map
        self._communication_range = 70.0
        # Raw label ids from the HuggingFace release already use the 3-class
        # scheme (1=car, 2=bike, 3=pedestrian), not the old 10-category devkit
        # taxonomy, so map them directly.
        self._vehicle_category_indices = np.array([[1]])
        self._cyclist_category_indices = np.array([[2]])
        self._pedestrian_category_indices = np.array([[3]])
        self._num_min_points_in_boxes = 5
        self._make_gt_info_range = np.array([-70., -70., -6., 70., 70., 1.])
        self._gt_infos: Dict[str, Dict] = dict()  # placeholder of gt_infos
        self._path_gt_infos: Dict[str, str] = params['path_gt_infos'] if 'path_gt_infos' in params else None  
        # {'train': path to train_gt_infos.pkl, 'test': ...}

        # === placeholder
        self.pre_processor = None
        self.post_processor = None
        self._cav_lidar_range = None  # to crop point cloud before passing to preprocessor for voxelization
        self._anchor_box = np.array([])  # (h, w, n_cls, n_anchor_per_pixel, 7)
        if 'order' in params['postprocess']:
            assert params['postprocess']['order'] == 'hwl', "Only suppert anchors in `hwl` order"

    @property
    def name_ego(self):
        return self._name_ego
    
    @property
    def name_rsu(self):
        return self._name_rsu
    
    @property
    def ego_se3_map(self):
        return self._ego_se3_map
    
    @property
    def is_pad_agent_indicator(self):
        return self._flag_pad_agent_indicator
    
    @property
    def num_agents(self):
        return self._msig.num_agents
    
    @property
    def num_min_points_in_boxes(self):
        return self._num_min_points_in_boxes
    
    @property
    def make_gt_info_range(self):
        return self._make_gt_info_range
    
    @property
    def is_train(self):
        return self._flag_train
    
    def load_gt_infos(self):
        _path = Path(self._path_gt_infos['train' if self._flag_train else 'test'])
        assert _path.exists(), f"flag_train: {self._flag_train} | {_path} does not exist"
        with open(_path, 'rb') as f:
            self._gt_infos = pickle.load(f)
    
    def get_1sample_from_gt_infos(self, seq_idx: int, idx_in_seq: int) -> Dict[str, np.ndarray]:
        return self._gt_infos[self.make_sample_token(seq_idx, idx_in_seq)]

    def _check_sequences_in_ascending_order(self) -> None:
        self._available_seqs.sort()
        return
    
    def _filter_sequences_(self) -> None:
        """
        This function mutates `self._available_seqs`.
        Filter sequences to make train/ test partition.
        """
        if self._flag_train:
            kept_seqs = [seq_idx for seq_idx in self._available_seqs if seq_idx not in self._test_sequences]
        else:
            kept_seqs = [seq_idx for seq_idx in self._available_seqs if seq_idx in self._test_sequences]
        
        self._available_seqs = np.array(kept_seqs)

        if self._flag_verbose:
            print(f"MSigBaseDataset | train: {self._flag_train} | avail seqs: {self._available_seqs}")
        return
    
    def _accumulate_seq_length(self) -> np.ndarray:
        """
        The output of this function is used to convert the index of the dataset into the index 
        of a data point in a sequence
        """
        lengths_seqs = np.array([self._msig.return_squence_length(seq_idx) for seq_idx in self._available_seqs])
        
        if self._flag_verbose:
            print(f"MSigBaseDataset | lengths_seqs: {lengths_seqs}")
        
        return np.cumsum(lengths_seqs)

    def __len__(self) -> int:
        return self._acc_seq_len[-1]

    def _convert_idx_in_dataset_to_idx_in_sequence(self, idx_in_dataset: int) -> Tuple[int]:
        """
        Returns:
            seq_index: in the case of test set, this would be a value taken from (4, 5, 17, 21).
                Note that seq_index is not contiguous
            
            idx_in_sequence: index of a data sample in the sequence identified by seq_index

        """
        # we loop the accumulated length list to see get the scenario index
        iter_avail_seq = 0
        while self._acc_seq_len[iter_avail_seq] <= idx_in_dataset:
            iter_avail_seq += 1
        
        # convert idx (of the whole dataset) into idx of sample of this scene
        idx_in_seq = idx_in_dataset if iter_avail_seq == 0 else \
            idx_in_dataset - self._acc_seq_len[iter_avail_seq - 1]

        return self._available_seqs[iter_avail_seq], idx_in_seq

    def __getitem__(self, idx):
        raise NotImplementedError("Abstract method, needs to be define by the children class.")

    def _get_all_pointclouds_in_1sample(self, seq_idx: int, idx_in_seq: int) -> Dict[str, np.ndarray]:
        """
        Return:
            {
                agent_name: {
                    points: (N, 4) point cloud of this agent in its body frame,
                    ego_se3_agent: (4, 4) - pose of the agent in the ego frame at the timestamp of the pointcloud
                }
            }
        """
        avail_cavs = self._msig.return_name_cavs_in_seq(seq_idx)
        avail_agents = list(self.name_rsu) + avail_cavs

        dict_agents_pts_poses = dict()
        for agent_name in avail_agents:
            agent_pc, agent_timestamp = self._msig.return_agent_point_cloud(seq_idx, agent_name, idx_in_seq)
            # agent_pc: (N, 4) - x, y, z, intensity

            map_se3_agent = self._msig.return_map_se3_agent(seq_idx, agent_name, agent_timestamp)
            ego_se3_agent = self.ego_se3_map @ map_se3_agent

            if self._is_agent_in_comm_range(ego_se3_agent[:3, -1]):
                dict_agents_pts_poses[agent_name] = {
                    'points': agent_pc,
                    'ego_se3_agent': ego_se3_agent
                }

        assert len(dict_agents_pts_poses) >= 2, \
            f"dict_agents_pts_poses must have at least top and dome: {list(dict_agents_pts_poses.keys())}"

        return dict_agents_pts_poses
    
    def _is_agent_in_comm_range(self, agent_position: np.ndarray) -> bool:
        assert agent_position.shape == (3,) or agent_position.shape == (2,), \
            f"agent_position.shape: {agent_position.shape}"
        
        return np.linalg.norm(agent_position[:2]) <= self._communication_range
    
    def return_gt_boxes_in_ego(self, seq_idx: int, idx_in_seq: int) -> np.ndarray:
        """
        Return:
        =======
        gt_boxes: (N, 7+2) - cx, cy, cz, dx, dy, dz, yaw, class_id, track_id
        """
        gt_boxes = self._msig.return_gt_boxes_in_map(seq_idx, idx_in_seq)
        # gt_boxes: (N, 7+2) - cx, cy, cz, dx, dy, dz, yaw, msig_category_id, track_id
        apply_se3_(self.ego_se3_map, boxes_=gt_boxes, boxes_has_velocity=False)

        gt_class = self.group_mixed_signals_classes_to_meta_classes(gt_boxes[:, -2])
        # overwrite category_id of gt_boxes with gt_class
        gt_boxes[:, -2] = gt_class
        return gt_boxes
    
    def group_mixed_signals_classes_to_meta_classes(self, gt_category_indices: np.ndarray) -> np.ndarray:
        """
        Args:
            gt_category_indices: (N,) - 1 (car), 2 (bike), 3 (pedestrian)

        Returns:
            gt_class: (N,) - 1 (vehicle), 2, (cyclist) 3 (pedestrian)
        """
        assert len(gt_category_indices.shape) == 1, f"{len(gt_category_indices.shape)} != 1"
        assert np.all(np.logical_and(gt_category_indices >= 1, gt_category_indices <= 3)), \
            "raw catergory_id of MixedSignals (this HuggingFace release) are in {1, 2, 3}"

        gt_class = np.zeros(gt_category_indices.shape[0])
        
        _gt_cat = gt_category_indices.astype(int).reshape(-1, 1)

        for class_idx, cat_indices_of_class in \
            enumerate((self._vehicle_category_indices, 
                       self._cyclist_category_indices, 
                       self._pedestrian_category_indices)):

            cls_mask = _gt_cat == cat_indices_of_class  # (N, num_indices_of_cls)
            cls_mask = np.any(cls_mask, axis=1)  # (N,)
            gt_class[cls_mask] = 1 + class_idx  # class_idx starts from 1

        return gt_class

    def check_idx_in_seq_has_gt_boxes(self, seq_idx: int, idx_in_seq: int) -> int:
        """
        Return new idx_in_seq if the input idx_in_seq points to a data sample that are not annotated
        """
        gt_boxes = self.return_gt_boxes_in_ego(seq_idx, idx_in_seq)
        offset = 1 if idx_in_seq < 100 else -1  
        # half of seq = 150, if before half move to the future 1 frame, else move to the past 1 frame
        while gt_boxes.shape[0] == 0:
            idx_in_seq += offset
            gt_boxes = self.return_gt_boxes_in_ego(seq_idx, idx_in_seq)
        
        return idx_in_seq

    @staticmethod
    def convert_agent_name_to_indicator(agent_name):
        if agent_name == 'top' or agent_name == 'ego':
            agent_indicator = 0
        elif agent_name == 'dome':
            agent_indicator = 1
        elif agent_name == '003':
            agent_indicator = 2
        elif agent_name == '004':
            agent_indicator = 3
        elif agent_name == 'laser':
            agent_indicator = 4
        else:
            raise ValueError(f"unknown cav_id {agent_name}")

        return agent_indicator
    
    @staticmethod
    def convert_agent_indicator_to_name(agent_indicator):
        if agent_indicator == 0:
            return 'top'
        elif agent_indicator == 1:
            return 'dome'
        elif agent_indicator == 2:
            return '003'
        elif agent_indicator == 3:
            return '004'
        elif agent_indicator == 4:
            return 'laser'
        else:
            raise ValueError(f"unknown agent indicator {agent_indicator}")
    
    @staticmethod
    def make_sample_token(seq_idx: int, idx_in_seq: int) -> str:
        """
        The sample_token made by this function is used to access gt_info in every dataset class
        """
        return f"seq{seq_idx}_idx{idx_in_seq}" 

    def _build_preprocessor(self, preprocessor_params: dict):
        return build_preprocessor(preprocessor_params, self.is_train)
    
    def _build_postprocessor(self, postprocessor_params: dict):
        return build_postprocessor(postprocessor_params, self.is_train)

    def init_output_of_getitem(self) -> Dict:
        """
        This function is to know what to expect from __getitem__
        """
        raise NotImplementedError("Abstract method, needs to be define by the children class.")
    
    def hard_voxelize_point_cloud(self, points: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """
        Args:
            points: (N, 4) - x, y, z, i
        
        Returns:
            processed_lidar: {
                    voxel_features: (N_vox, num_max_pts_per_vox, num_pts_feat)
                    voxel_coords: (N_vox, z+y+x) -> to be prepended with batch_idx in collate_func
                    voxel_num_points: (N_vox,)
            }
            points: (N, 4)
        """
        assert isinstance(points, np.ndarray), f"expect np.ndarray, got {type(points)}"
        points = remove_points_outside_range(points, self._cav_lidar_range)
        points = shuffle_points(points)
        dict_process_lidar = self.pre_processor.preprocess(points)
        return dict_process_lidar, points

    def preprocess_gt_boxes(self, 
                         gt_boxes: np.ndarray, 
                         gt_class: np.ndarray, 
                         gt_visibility: np.ndarray,
                         avail_agents_inds: np.ndarray) -> dict:
        """
        Filter gt_boxes outside of range and not visible to avail agents
        """
        assert gt_boxes.shape[1] == 7, f"{gt_boxes.shape[1]} != 7"
        
        gt_class = gt_class.astype(int)
        assert gt_class.shape == (gt_boxes.shape[0],), f"{gt_class.shape} != {(gt_boxes.shape[0],)}"
        assert np.all(gt_class >= 1), "gt_class must start from 1"

        assert gt_boxes.shape[0] == gt_visibility.shape[0], f"{gt_boxes.shape[0]} != {gt_visibility.shape[0]}"
        assert gt_visibility.shape[1] == self.num_agents, f"{gt_visibility.shape[1]} != {self.num_agents}"

        assert np.max(avail_agents_inds) < gt_visibility.shape[1], \
            f"{np.max(avail_agents_inds)} !< {gt_visibility.shape[1]}"

        # to filter gt_boxes based on visibility of avail agents
        mask_visible_gt = gt_visibility[:, avail_agents_inds].any(axis=1)
        
        # to filter gt_boxes outside of cav_lidar_range
        mask_in_range_gt = np.logical_and(gt_boxes[:, :3] >= self._cav_lidar_range[:3], 
                                          gt_boxes[:, :3] < self._cav_lidar_range[3:]).all(axis=1)
    
        # apply filter 
        mask_kept = np.logical_and(mask_visible_gt, mask_in_range_gt)
        gt_boxes = gt_boxes[mask_kept]
        gt_class = gt_class[mask_kept]

        return gt_boxes, gt_class

    def collate_batch_train(self, batch: List[Dict]) -> Dict:
        # init output
        datas = {'ego': {
            'processed_lidar': list(),
            'label_dict': list(),}
        }

        for _data in batch:
            datas['ego']['processed_lidar'].append(_data['ego']['processed_lidar'])
            datas['ego']['label_dict'].append(_data['ego']['label_dict'])

        datas['ego']['processed_lidar'] = \
            self.pre_processor.collate_batch(datas['ego']['processed_lidar'])
        
        datas['ego']['label_dict'] = \
            self.post_processor.collate_batch(datas['ego']['label_dict'])

        return datas
    
    def collate_batch_test(self, batch: List[Dict]) -> Dict:
        assert len(batch) == 1, f"{len(batch)} != 1"
        # init output
        datas = {'ego': {
            'processed_lidar': list(),
            'anchor_box': None,
            'transformation_matrix': torch.from_numpy(np.identity(4)).float(),
            },

            'metadata': {
                'gt_boxes_in_ego': None,
                'gt_class': None,
            }
        }

        _data = batch[0]

        datas['ego']['processed_lidar'].append(_data['ego']['processed_lidar'])
        datas['ego']['processed_lidar'] = \
            self.pre_processor.collate_batch(datas['ego']['processed_lidar'])
        
        datas['ego']['anchor_box'] = torch.from_numpy(self._anchor_box).float()
        
        datas['metadata']['gt_boxes_in_ego'] = torch.from_numpy(_data['metadata']['gt_boxes_in_ego']).float()
        datas['metadata']['gt_class'] = torch.from_numpy(_data['metadata']['gt_class']).long()
        datas['metadata']['points'] = torch.from_numpy(_data['metadata']['points']).float()

        return datas
    
    @staticmethod
    def convert_boxes_from_lwh_to_hwl(boxes: np.ndarray) -> np.ndarray:
        assert boxes.shape[1] == 7, f"{boxes.shape[1]} != 7"
        _boxes = deepcopy(boxes)
        box_utils.bbox_lwh_to_hwl(_boxes)
        return _boxes
    
    def post_process(self, data_dict: dict, output_dict: dict, **kwargs) -> Tuple[torch.Tensor]:
        pred_boxes_in_top, pred_scores, pred_class = \
            self.post_processor.post_process(data_dict, output_dict)
        
        gt_boxes_in_top = data_dict['metadata']['gt_boxes_in_ego']
        gt_class = data_dict['metadata']['gt_class']

        return pred_boxes_in_top, pred_scores, pred_class, gt_boxes_in_top, gt_class
        