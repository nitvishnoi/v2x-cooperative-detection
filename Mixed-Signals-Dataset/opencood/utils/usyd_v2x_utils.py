from pathlib import Path
from typing import Dict, List, Set
import csv
import json

import numpy as np
import torch
from pypcd4 import PointCloud
from pyquaternion import Quaternion

from opencood.utils.v2x_sim_utils import make_se3, apply_se3_, apply_se3


def type_check(o, target_type) -> None:
    assert isinstance(o, target_type), f"must have type {target_type}"


def _legacy_check_stamp(stamp: str):
    """
    From Katie

    Parameter
    ---------
    stamp : str
        the last part of the a file in the Pointclouds directory

    Return
    ------
    stamp_with_the part after the "." prepended with 0
    """
    stamp_split = stamp.split(".")
    if len(stamp_split[-1]) < 9:
        stamp_split[-1] = "0" * (9 - len(stamp_split[-1])) + stamp_split[-1]
    return ".".join(stamp_split)


def read_odometry_file(path_odom_file: Path) -> Dict:
    """
    Parse csv file storing agent's odometry into dict

    Parameter
    ---------
    path_odom_file : Path
        path to odometry file
    
    Return
    ------
    odom_data : Dict
        dict represents the info stored in odometry file
    """
    type_check(path_odom_file, Path)
    assert path_odom_file.exists(), f"{str(path_odom_file)} does not exist"

    odom_data = {}
    header_idx_map = {}
    with open(path_odom_file, mode ='r')as file:
        odo_reader = csv.reader(file)
        for odom_time_idx, lines in enumerate(odo_reader):
            if odom_time_idx == 0:  # header line
                for j, header_title in enumerate(lines):
                    odom_data[header_title] = []
                    header_idx_map[j] = header_title
            else:  # datas
                for j, data in enumerate(lines):
                    odom_data[header_idx_map[j]].append(data)
    return odom_data


def build_seq_database(path_pcd: Path, path_odom: Path) -> Dict:
    """
    Organize Pointclouds and Odometry into dictionary

    Parameters
    ----------
    patch_pcd : Path
        path to the directory storing the point clouds of this sequence
    
    path_odom : Path
        path to the directory storing odom of every CAV of this sequence
    
    Return
    ------
    seq_database : 
        {
            agent_name : {
                time_idx0 : {
                    'timestamp_pc': value of timestamp of point cloud
                    'pc_file': full path to the pc file of this agent at time_idx0
                    'timestamp_odom': value of timestamp of agent's pose
                    'translation': coordinate in map frame
                    'quaternion': quaternion representing orientation in map frame
                }
                time_idx1 : {...}
            }    
        }
    """
    type_check(path_pcd, Path)
    type_check(path_odom, Path)

    seq_database = dict()

    # ================================
    # get pointcloud path & timestamps
    # ================================
    for path_to_file in path_pcd.glob('*.pcd'):
        filename = str(path_to_file.parts[-1])[:-4]  # exclude '.pcd'
        filename = filename.split('_')
        agent_name = filename[0]
        pc_time_idx = filename[1]
        timestamp_pc = float(_legacy_check_stamp(filename[2]))

        if agent_name not in seq_database:
            seq_database.update({agent_name: dict()})

        seq_database[agent_name][pc_time_idx] = {
            'timestamp_pc': timestamp_pc,
            'pc_file': path_to_file 
        }

    # ============================================================
    # for each agent, find the pose of its lidar at each timestamp
    # ============================================================
    for agent_name in seq_database.keys():
        if agent_name in ('dome', 'top'):
            # these two agents are static -> constant pose
            continue
        
        odom_data = read_odometry_file(path_odom / f'odometry_{agent_name}.csv')
        
        # loop over each time_idx of agent's pointclouds
        agent_content = seq_database[agent_name]
        _list_time_idx_offset = np.array([0, 1, -1])
        for t_idx in agent_content.keys():
            pc_timestamp = agent_content[t_idx]['timestamp_pc']
            
            # jiggle the odom time index to see which one is closest to 
            # the point cloud's timestamp
            candidate_odom_ts = np.array([
                float(odom_data['field.header.stamp'][int(t_idx) + o]) / 1e9 
                for o in _list_time_idx_offset
            ])
            idx_min_diff_ts = np.argmin(np.abs(pc_timestamp - candidate_odom_ts))
            
            # extract timestamp
            odom_timestamp = candidate_odom_ts[idx_min_diff_ts]

            # extract pose
            chosen_offset = _list_time_idx_offset[idx_min_diff_ts]
            odom_time_idx = int(t_idx) + chosen_offset
            translation = np.array([
                odom_data['field.pose.pose.position.x'][odom_time_idx],
                odom_data['field.pose.pose.position.y'][odom_time_idx],
                odom_data['field.pose.pose.position.z'][odom_time_idx]
            ])
            
            quaternion = np.array([
                odom_data['field.pose.pose.orientation.w'][odom_time_idx],
                odom_data['field.pose.pose.orientation.x'][odom_time_idx],
                odom_data['field.pose.pose.orientation.y'][odom_time_idx],
                odom_data['field.pose.pose.orientation.z'][odom_time_idx]
            ])

            # save pose and timestamp of pose to agent_content
            agent_content[t_idx].update({
                'timestamp_odom': odom_timestamp,
                'translation': translation,
                'quaternion': quaternion
            })

        # save new agent_content to seq_database
        seq_database[agent_name] = agent_content
    
    return seq_database


def read_pointcloud(pointcloud_file: Path) -> np.ndarray:
    """
    Read point cloud from file

    Parameter
    ---------
    pointcloud_file : Path
        path to point cloud file

    Return
    ------
    pc_numpy : np.ndarray
        (N, 4) - x, y, z, intensity
    """
    type_check(pointcloud_file, Path)
    pc = PointCloud.from_path(pointcloud_file)
    return pc.numpy(('x', 'y', 'z', 'intensity'))


def get_rsu_dome_pose_in_map() -> np.ndarray:
    # TODO: double check if this is true
    return np.eye(4)


def get_rsu_top_pose_in_map() -> np.ndarray:
    # TODO: double check if this is true
    return np.eye(4)


def linear_interp(time_query: float, 
                  val_start: float, 
                  val_end: float, 
                  time_start: float, 
                  time_end: float) -> float:
    assert (time_end - time_start) > 0.0, f"{time_end - time_start} not > 0.0"
    assert (time_query - time_start) >= 0.0, f"{time_query - time_start} not >= 0.0"

    val_query = val_start + \
        (val_end - val_start) * (time_query - time_start) / (time_end - time_start)
    return val_query

def interpolate_yaw(time_query: float,
                    yaw_start: float,
                    yaw_end: float,
                    time_start: float,
                    time_end: float) -> float:
    q0 = Quaternion(axis=[0, 0, 1], angle=yaw_start)
    q1 = Quaternion(axis=[0, 0, 1], angle=yaw_end)
    q = Quaternion.slerp(q0, q1, amount=(time_query - time_start) / (time_end - time_start))
    rot_mat = q.rotation_matrix
    '''
    [c -s 0]
    [s c  0]
    [0 0 1]
    '''
    yaw = np.arctan2(-rot_mat[0, 1], rot_mat[0, 0])
    return yaw

class SequenceLabeledFrames(object):
    def __init__(self, labels_dataset_samples: List[dict], sequence_id: int):
        """
        Parameters
        ----------
        labels_dataset_samples: 
            result of reading json file (:= labels['dataset']['samples'])
        
        sequence_id:
            index of this sequence going from 1 to 37.
        """
        # find labels of this sequence
        _found = False
        _id = 0
        seq_samples = None
        while not _found and _id < len(labels_dataset_samples):
            seq_samples = labels_dataset_samples[_id]
            _this_seq_id = seq_samples['name'].strip().split('_')[-1]
            _found = int(_this_seq_id) == sequence_id
            if not _found:
                _id += 1

        assert _found, f"sequence {sequence_id} does not have labels"

        self._frames = seq_samples['labels']['ground-truth']['attributes']['frames']

        self._offs = 1.7e18  # for numerical stability
        self._timestamps = [int(fr['timestamp']) - self._offs for fr in self._frames]
    
    def return_min_max_timestamp(self):
        return (self._timestamps[0], self._timestamps[-1])

    def _find_indices_annotated_frames(self, timestamp: float) -> List[int]:
        # offset the query timestamp by self._offs for numerical stability
        _timestamp = int(timestamp * 1e9) - self._offs

        labeled_idx = np.searchsorted(self._timestamps, _timestamp)

        # process extreme left & right
        # queried timestamp has only 1 neighboring labeled timestamp
        # => invalid frame because this frame is not annotated
        if labeled_idx == 0 and _timestamp < self._timestamps[0]:
            # extreme left
            return -100, -100  # -100 is the dummy value
        
        if labeled_idx == len(self._timestamps):
            # extreme right
            return -100, -100  # -100 is the dummy value


        if np.abs(_timestamp - self._timestamps[labeled_idx]) < 1e-5:
            # a keyframe because the query timestamp matches 
            # perfectly with a labeled timestamp
            return -100, labeled_idx
        else:
            return labeled_idx - 1, labeled_idx

    def _return_annos_labeled_frames(self, labeled_idx: int) -> np.ndarray:
        """
        Return annotation of a labeled frame

        Parameter
        ---------
        labeled_idx:
            index of a labeled frame        

        Return
        ------
        gt_boxes:
            (N, 7+2) - [x, y, z, l, w, h, yaw, category_id, track_id]
        """
        labeled_frame = self._frames[labeled_idx]
        gt_boxes = list()
        for anno in labeled_frame['annotations']:
            box = np.array([
                anno['position']['x'], 
                anno['position']['y'], 
                anno['position']['z'],

                anno['dimensions']['x'],
                anno['dimensions']['y'],
                anno['dimensions']['z'],

                anno['yaw'],

                anno['category_id'],
                anno['track_id'],
            ])
            gt_boxes.append(box)

        gt_boxes = np.stack(gt_boxes, axis=0)
        return gt_boxes

    def return_annos(self, timestamp: float, debug: bool = False) -> np.ndarray:
        """
        Return annotation of a query timestamp 
        (might be timestamp of labeled_frame or might not)

        Parameter
        ---------
        timestamp:
            query timestamp comes from timestamp of pc of an agent
        
        debug:
            if True, print keyframe (i.e., is_labeled) status of timestamp

        Return
        ------
        gt_boxes:
            (N, 7+1) - [x, y, z, l, w, h, yaw, category_id]
        """
        labeled_ids = self._find_indices_annotated_frames(timestamp)

        # check if this timestamp is annotated
        if labeled_ids[0] < 0 and labeled_ids[1] < 0:
            if debug:
                print(f'{timestamp} is not labeled')
            return np.zeros((0, 8))

        # check if this timestamp is a keyframe
        if labeled_ids[0] < 0 and labeled_ids[1] >= 0:
            if debug:
                print(f"{timestamp} is a keyframe")
            return self._return_annos_labeled_frames(labeled_ids[1])
        
        if debug:
            print(f'{timestamp} is inside the labeled range, but not a keyframe')
        # -------------------------------
        # interpolate gt_boxes in frames 
        # -------------------------------
        prev_gt = self._return_annos_labeled_frames(labeled_ids[0])
        next_gt = self._return_annos_labeled_frames(labeled_ids[1])
        
        # only interpolate on objects that present in both prev & next frame
        common_gt = dict()
        # {
        #   track_id_1: (pose_t-1, pose_t),
        #   track_id_2: (pose_t-1, pose_t),
        # }
        
        for prev_box in prev_gt:
            _track_id = int(prev_box[-1])
            common_gt[_track_id] = [prev_box,]
        
        for next_box in next_gt:
            _track_id = int(next_box[-1])
            if _track_id in common_gt:
                common_gt[_track_id].append(next_box)
        
        interp_boxes = list()
        _timestamp = int(timestamp * 1e9) - self._offs  # for numberical stability
        t_start = self._timestamps[labeled_ids[0]]
        t_end = self._timestamps[labeled_ids[1]]
        for _track_id, _item in common_gt.items():
            if len(_item) < 2:
                # this _track_id doesn't have 2 items
                # --> the obj doesn't present in the next frame
                continue
            prev_box, next_box = _item
            x = linear_interp(_timestamp, prev_box[0], next_box[0], t_start, t_end)
            y = linear_interp(_timestamp, prev_box[1], next_box[1], t_start, t_end)
            z = linear_interp(_timestamp, prev_box[2], next_box[2], t_start, t_end)
            yaw = interpolate_yaw(_timestamp, prev_box[6], next_box[6], t_start, t_end)

            box = [x, y, z, 
                   *next_box[3: 6].tolist(),  # dimension
                   yaw, 
                   next_box[7],  # category_id
                   next_box[8],  # track_id
                ]
            interp_boxes.append(box)

        interp_boxes = np.stack(interp_boxes)
        return interp_boxes


class USydExplorer(object):
    def __init__(self, dataset_root: Path) -> None:
        type_check(dataset_root, Path)
        
        self._root = dataset_root
        self._seq_names = self._find_available_sequences()
        self.database = dict()
        for seq_name in self._seq_names:
            self.database[seq_name] = build_seq_database(
                self._root / 'PointClouds' / f'mini_{seq_name}', 
                self._root / 'Odometry' / f'mini_{seq_name}'
            )
        
        label_file = dataset_root / 'V2X_dataset-v0.4-labels.json'
        with open(label_file, 'r') as f:
            labels = json.load(f)
        
        self.label_database: Dict[str, SequenceLabeledFrames] = dict()
        for seq_name in self._seq_names:
            label_db = SequenceLabeledFrames(labels['dataset']['samples'], int(seq_name))
            self.label_database[seq_name] = label_db

        self.labeled_time_idx = \
            dict([(seq_name, self.return_seq_labeled_frames(seq_name)) for seq_name in self._seq_names])
        
        # position of 'top' to translate pointclouds and boxes to [0, 0, 0] of map frame
        # self._coord_top = np.array([-41.551, -51.878, -1.077 - 2.50])
        self._map_se3_top = np.array([
            [1.0, 0.0, 0.0, -41.551],
            [0.0, 1.0, 0.0, -51.878],
            [0.0, 0.0, 1.0, -1.077],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._top_se3_map = np.linalg.inv(self._map_se3_top)

        self._map_se3_dome = np.array([
            [1.0, 0.0, 0.0, -41.507],
            [0.0, 1.0, 0.0, -51.864],
            [0.0, 0.0, 1.0, -1.340],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._dome_se3_map = np.linalg.inv(self._map_se3_dome)

        self._class_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self._class_name = ['car',  # 1
                            'pedestrian',  # 2
                            'truck',  # 3
                            'motorbike',  # 4
                            'bicycle',  # 5
                            'emergency_vehicle',  # 6
                            'bus',  # 7
                            'portable_personal_mobility',  # 8
                            'electric_vehicle',  # 9
                            'trailer'  # 10
                            ]
        self._name_cars = set(('003', '004', 'laser'))
        self._translate_car_along_z = -3.25
        self.max_num_agents = 5  # top, dome, 003, 004, laser

    @staticmethod
    def _find_vehicles(object_class: np.ndarray) -> np.ndarray:
        assert len(object_class.shape) == 1, f"len({object_class.shape}) != 1"
        mask_vehicles = (object_class == 1) | (object_class == 3) | (object_class == 6) \
                    | (object_class == 7) | (object_class == 9) | (object_class == 10)
        return mask_vehicles
    
    @staticmethod
    def _find_bikes(object_class: np.ndarray) -> np.ndarray:
        assert len(object_class.shape) == 1, f"len({object_class.shape}) != 1"
        mask_bike = (object_class == 4) | (object_class == 5) | (object_class == 8)
        return mask_bike
    
    @staticmethod
    def _find_pedestrians(object_class: np.ndarray) -> np.ndarray:
        assert len(object_class.shape) == 1, f"len({object_class.shape}) != 1"
        mask_pedestrian = object_class == 2
        return mask_pedestrian
        
    def return_seq_labeled_frames(self, seq_name: str):
        top_content = self.database[seq_name]['top']
        min_labeled_timestamp, max_labeled_timestamp = self.label_database[seq_name].return_min_max_timestamp()
        labeled_time_idx: List[int] = list()
        for str_time_idx in top_content.keys():
            timestamp = self.return_agent_timestamp(int(seq_name), 'top', int(str_time_idx))
            _timestamp = int(timestamp * 1e9) - 1.7e18  # for numerical stability
            if _timestamp >= min_labeled_timestamp and _timestamp <= max_labeled_timestamp:
                # check if vehicles available
                gt_boxes_in_map = self.label_database[seq_name].return_annos(timestamp)
                gt_class = gt_boxes_in_map[:, -2].astype(int)
                mask_vehicle = self._find_vehicles(gt_class)
                mask_bike = self._find_bikes(gt_class)
                mask_ped = self._find_pedestrians(gt_class)
                mask_have_labels = mask_vehicle | mask_bike | mask_ped
                if np.any(mask_have_labels):
                    labeled_time_idx.append(int(str_time_idx))

        labeled_time_idx.sort()
        return labeled_time_idx
    
    def return_seq_len(self, seq_name: str) -> int:
        return len(self.labeled_time_idx[seq_name])
        
    def return_avail_seq_name(self) -> List[str]:
        return self._seq_names

    def _find_available_sequences(self) -> List[str]:
        """
        Find the name of available sequences in the dataset root

        Return
        ------
        avail_seq_pcs:
            sorted list of name of sequences (e.g., {'11', '18'})
        """
        data_dirs = [f for f in self._root.iterdir() if f.is_dir()]
        avail_seq_pcs = set()
        avail_seq_odom = set()
        for directory_data in data_dirs:
            data_type = directory_data.parts[-1]
            if data_type == 'PointClouds':
                save_to = avail_seq_pcs
            elif data_type == 'Odometry':
                save_to = avail_seq_odom
            else:
                RuntimeWarning(f'{directory_data} is unknown')
        
            for directory_seq in directory_data.iterdir():
                if directory_seq.is_dir():
                    seq_name = directory_seq.parts[-1].split('_')[1]
                    save_to.add(seq_name)
        
        assert avail_seq_pcs == avail_seq_odom, f"{avail_seq_pcs} != {avail_seq_odom}"
        avail_seq_pcs = sorted([int(seq_name) for seq_name in avail_seq_pcs])
        return [str(seq_id) for seq_id in avail_seq_pcs]

    def return_pointcloud(self, seq_id: int, agent_name: str, idx: int, return_intensity: bool = True) -> np.ndarray:
        """
        Get agent's point cloud in its body frame

        Parameters
        ----------
        seq_id: 
            id of a sequence (in {1, 2, ..., 37})
        
        agent_name:

        time_idx:

        Return
        ------
        pc: 
            (N, 3+1) - x, y, z, intensity
        """
        type_check(idx, int)
        time_idx = self.labeled_time_idx[str(seq_id)][idx]
        _offset = 0 if agent_name != 'top' else 1  # hack to make pc of top align better with the rest
        data = self.database[str(seq_id)][agent_name][str(time_idx + _offset)]
        pc = read_pointcloud(data['pc_file'])

        # undo the transformation that is overwritten to points of top & dome
        if agent_name == 'top':
            apply_se3_(self._top_se3_map, pc)
        elif agent_name == 'dome':
            apply_se3_(self._dome_se3_map, pc)

        if agent_name in self._name_cars:
            # translate point cloud of cars (003, 004, laser) to make 
            # the origin of their body frame has roughly the same height as RSU
            pc[:, 2] += self._translate_car_along_z

        # normalize intensity
        pc[:, 3] = np.clip(pc[:, 3] / 3500.0, a_min=0.0, a_max=1.0)

        if return_intensity:
            return pc
        else:
            # use [x, y, z] only
            return pc[:, :3]

    def return_agent_pose_in_map(self, 
                               seq_id: int, 
                               agent_name: str, 
                               idx: int) -> np.ndarray:
        """
        Get connected vehicles' pose in MAP frame

        Parameters
        ----------
        seq_id: 
            id of a sequence (in {1, 2, ..., 37})
        
        agent_name:

        time_idx:

        Return
        ------
        map_se3_agent: 
            (4, 4) - rigid body transformation from CAV's body frame to the MAP frame
        """
        if agent_name == 'top':
            return self._map_se3_top
        elif agent_name == 'dome':
            return self._map_se3_dome
        else:
            assert agent_name in ('003', '004', 'laser'), f"{agent_name} ! in ('003', '004', 'laser')"
            type_check(idx, int)
            time_idx = self.labeled_time_idx[str(seq_id)][idx]
            data = self.database[str(seq_id)][agent_name][str(time_idx)]
            map_se3_agent = make_se3(
                data['translation'], 
                quaternion=data['quaternion']
            )

            # translate the origin of the body frame of cars
            # to make them resemble the origin of the body frame of RSU
            agent_se3_agent_translated = np.eye(4)
            agent_se3_agent_translated[:3, -1] = np.array([0, 0, -self._translate_car_along_z])
            
            map_se3_agent = map_se3_agent @ agent_se3_agent_translated

            return map_se3_agent

    def return_agent_timestamp(self, seq_id: int, agent_name: str, time_idx: int) -> float:
        return self.database[str(seq_id)][agent_name][str(time_idx)]['timestamp_pc']

    def return_ground_truth(self, 
                            seq_id: int, 
                            agent_name: str, 
                            idx: int) -> np.ndarray:
        """
        Get ground truth of aggregated point clouds in MAP frame at a specific timestamp

        Parameters
        ----------
        seq_id: 

        agent_name:
            name of the agent whose database is indexed by time_idx

        time_idx:
            in {0, 1, 2, ..., len of a sequence}
        
        Return
        ------
        gt_boxes: 
            (M, 7+2) - x, y, z, l, w, h, yaw, category_id, track_id
        """
        type_check(idx, int)
        time_idx = self.labeled_time_idx[str(seq_id)][idx]
        timestamp = self.return_agent_timestamp(seq_id, agent_name, time_idx)
        gt_boxes_in_map = self.label_database[str(seq_id)].return_annos(timestamp)

        # interpret category_id to class
        gt_class = gt_boxes_in_map[:, -2].astype(int)
        mask_vehicle = self._find_vehicles(gt_class)
        mask_bike = self._find_bikes(gt_class)
        mask_ped = self._find_pedestrians(gt_class)

        gt_boxes_in_map[mask_vehicle, -2] = 1
        gt_boxes_in_map[mask_bike, -2] = 2
        gt_boxes_in_map[mask_ped, -2] = 3

        return gt_boxes_in_map

    @staticmethod
    def convert_agent_name_to_agent_indicator(agent_name):
        if agent_name == 'top':
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
    def convert_agent_indicator_to_agent_name(agent_indicator):
        if agent_indicator == 0:
            agent_name = 'top'
        elif agent_indicator == 1:
            agent_name = 'dome'
        elif agent_indicator == 2:
            agent_name = '003'
        elif agent_indicator == 3:
            agent_name = '004'
        elif agent_indicator == 4:
            agent_name = 'laser'
        else:
            raise ValueError(f"unknown agent indicator: {agent_indicator}")
        return agent_name


def cache_ground_truth_in_TOP(dataset_root: Path,
                              min_num_points_in_box: int = 5):
    """
    Make ground truth info to skip filtering empty boxes on the fly
    -> accelerate training & eval

    Parameter:
    ---------
    dataset_root: root of USyD dataset
    """
    import pickle
    from tqdm import tqdm
    from einops import rearrange

    from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

    usyd = USydExplorer(dataset_root)

    # init output
    gt_infos = dict()

    for seq_name in usyd._seq_names:
        print(f'building gt_infos of sequence: {seq_name}')
        
        seq_id = int(seq_name)
        seq_info = usyd.database[seq_name]
        name_agents_in_seq = list(seq_info.keys())

        for idx in tqdm(range(len(usyd.labeled_time_idx[seq_name]))):
            all_lidars_in_map = list()
            all_gt_boxes_in_map = list()
            for agent_name in name_agents_in_seq:
                # get aggregated point cloud & ground truth (because gt_boxes are organized per agent)
                lidar_in_agent = usyd.return_pointcloud(seq_id, agent_name, idx)  # (N, 4)
                gt_boxes_in_map = usyd.return_ground_truth(seq_id, agent_name, idx)

                map_se3_agent = usyd.return_agent_pose_in_map(seq_id, agent_name, idx)  # (4, 4)
                lidar_in_map = apply_se3(map_se3_agent, lidar_in_agent)
                
                # pad lidar with agent indicator
                agent_indicator = usyd.convert_agent_name_to_agent_indicator(agent_name)
                lidar_in_map = np.pad(lidar_in_map, pad_width=[(0, 0), (0, 1)], constant_values=agent_indicator)

                # ---
                all_lidars_in_map.append(lidar_in_map)
                if gt_boxes_in_map.shape[0] > 0:
                    all_gt_boxes_in_map.append(gt_boxes_in_map)
            
            all_lidars_in_map = np.concatenate(all_lidars_in_map)  # (N_tot, 4+1) - x, y, z, intensity, agent_indicator
            all_gt_boxes_in_map = np.concatenate(all_gt_boxes_in_map)  # (M_tot, 7+2) - box-7, category_id, track_id

            # to torch.Tensor for fast processing
            all_lidars_in_map = torch.from_numpy(all_lidars_in_map).float().cuda()
            all_gt_boxes_in_map = torch.from_numpy(all_gt_boxes_in_map).float().cuda()

            # remove repetitive gt_boxes
            gt_track_id = all_gt_boxes_in_map[:, -1].long()
            unq_track_id, indices = torch_unique_return_indices(gt_track_id)
            all_gt_boxes_in_map = all_gt_boxes_in_map[indices]  # (M, 7+2) - box-7, category_id, track_id

            # ------------------------
            # point-to-box correspondant
            box_index_of_points = points_in_boxes_gpu(
                all_lidars_in_map[:, :3].unsqueeze(0).contiguous(),
                all_gt_boxes_in_map[:, :7].unsqueeze(0).contiguous()
            ).squeeze(0)  # (N,)

            points_to_boxes_correspondance = rearrange(box_index_of_points, "N -> N 1") == \
                rearrange(torch.arange(all_gt_boxes_in_map.shape[0]).cuda(), "M -> 1 M")  # (N, M)
            
            # filter empty boxes
            num_points_in_boxes = points_to_boxes_correspondance.long().sum(dim=0)  # (M,)
            mask_kept = num_points_in_boxes > min_num_points_in_box
            all_gt_boxes_in_map = all_gt_boxes_in_map[mask_kept]

            # compute gt_visibility
            num_gt = mask_kept.long().sum()
            gt_visibility = all_gt_boxes_in_map.new_zeros(
                num_gt, usyd.max_num_agents
            ).bool()

            points_agent_indicator = all_lidars_in_map[:, -1].long()
            for idx_gt in range(num_gt):
                mask_points_in_this_gt = points_to_boxes_correspondance[:, idx_gt]  # (N,)
                unq_indicator = torch.unique(points_agent_indicator[mask_points_in_this_gt])
                gt_visibility[idx_gt, unq_indicator] = True

            # ---
            all_gt_boxes_in_map = all_gt_boxes_in_map.cpu().numpy().astype(np.float32)
            sample_token = f"seq{seq_id}_idx{idx}"
            gt_infos[sample_token] = {
                'gt_boxes_in_map': all_gt_boxes_in_map[:, :7],
                'gt_class': all_gt_boxes_in_map[:, -2],
                'gt_track_id': all_gt_boxes_in_map[:, -1],
                'gt_visibility': gt_visibility.cpu().numpy(),
            }
        
    path_file_info = dataset_root / f"infos_gt.pkl"
    print(f'save gt_infos of to {path_file_info}')
    with open(str(path_file_info), 'wb') as f:
        pickle.dump(gt_infos, f)
    
    return
            

def torch_unique_return_indices(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


if __name__ == '__main__':
    import argparse
        
    parser = argparse.ArgumentParser(description='parse input for cache_ground_truth_in_RSU')
    parser.add_argument('--dataroot', type=str)
    args = parser.parse_args()

    cache_ground_truth_in_TOP(Path(args.dataroot))

