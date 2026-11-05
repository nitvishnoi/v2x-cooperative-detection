import numpy as np
from pathlib import Path
from typing import Union, List
from pyquaternion import Quaternion
from nuscenes import NuScenes


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def rotation_matrix_to_yaw(rot: np.ndarray) -> float:
    return np.arctan2(rot[1, 0], rot[0, 0])


def apply_se3_(se3_tf: np.ndarray, 
               points_: np.ndarray = None, 
               boxes_: np.ndarray = None, boxes_has_velocity: bool = False, 
               vector_: np.ndarray = None) -> None:
    """
    Inplace function

    Args:
        se3_tf: (4, 4) - homogeneous transformation
        points_: (N, 3 + C) - x, y, z, [others]
        boxes_: (N, 7 + 2 + C) - x, y, z, dx, dy, dz, yaw, [vx, vy], [others]
        boxes_has_velocity: make boxes_velocity explicit
        vector_: (N, 2 [+ 1]) - x, y, [z]
    """
    if points_ is not None:
        assert points_.shape[1] >= 3, f"points_.shape: {points_.shape}"
        points_[:, :3] = points_[:, :3] @  se3_tf[:3, :3].T + se3_tf[:3, -1]

    if boxes_ is not None:
        assert boxes_.shape[1] >= 7, f"boxes_.shape: {boxes_.shape}"
        boxes_[:, :3] = boxes_[:, :3] @  se3_tf[:3, :3].T + se3_tf[:3, -1]
        boxes_[:, 6] += rotation_matrix_to_yaw(se3_tf[:3, :3])
        boxes_[:, 6] = np.arctan2(np.sin(boxes_[:, 6]), np.cos(boxes_[:, 6]))
        if boxes_has_velocity:
            boxes_velo = np.pad(boxes_[:, 7: 9], pad_width=[(0, 0), (0, 1)], constant_values=0.0)  # (N, 3) - vx, vy, vz
            boxes_velo = boxes_velo @ se3_tf[:3, :3].T
            boxes_[:, 7: 9] = boxes_velo[:, :2]

    if vector_ is not None:
        if vector_.shape[1] == 2:
            vector = np.pad(vector_, pad_width=[(0, 0), (0, 1)], constant_values=0.)
            vector_[:, :2] = (vector @ se3_tf[:3, :3].T)[:, :2]
        else:
            assert vector_.shape[1] == 3, f"vector_.shape: {vector_.shape}"
            vector_[:, :3] = vector_ @ se3_tf[:3, :3].T

    return


def apply_se3(se3_tf: np.ndarray, 
              points_: np.ndarray = None, 
              boxes_: np.ndarray = None, 
              boxes_has_velocity: bool = False, 
              vector_: np.ndarray = None) -> np.ndarray:
    """
    Non-inplace version of apply_se3_
    """
    points = points_.copy() if points_ is not None else None
    boxes = boxes_.copy() if boxes_ is not None else None
    vectors = vector_.copy() if vector_ is not None else None
    apply_se3_(se3_tf, 
               points_=points, 
               boxes_=boxes, 
               vector_=vectors, 
               boxes_has_velocity=boxes_has_velocity)
    out = [ele for ele in [points, boxes, vectors] if ele is not None]
    if len(out) == 1:
        return out[0]
    else:
        return out


def make_se3(translation: Union[List[float], np.ndarray], yaw: float = None, rotation_matrix: np.ndarray = None, 
             quaternion: Union[List[float], np.ndarray] = None):
    if yaw is not None:
        assert quaternion is None
        assert rotation_matrix is None
        rotation_matrix = make_rotation_around_z(yaw)
    elif quaternion is not None:
        assert yaw is None
        assert rotation_matrix is None
        assert isinstance(quaternion, list) or isinstance(quaternion, np.ndarray), f"{type(quaternion)} is not supported"
        rotation_matrix = Quaternion(quaternion).rotation_matrix
    else:
        assert rotation_matrix is not None
        
    
    out = np.zeros((4, 4))
    out[-1, -1] = 1.0

    out[:3, :3] = rotation_matrix

    if not isinstance(translation, np.ndarray):
        translation = np.array(translation)
    out[:3, -1] = translation.reshape(3)

    return out


def make_rotation_around_z(yaw: float) -> np.ndarray:
    cos, sin = np.cos(yaw), np.sin(yaw)
    out = np.array([
        [cos, -sin, 0.],
        [sin, cos, 0.],
        [0., 0., 1.]
    ])
    return out


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


class NuScenesHelper(object):
    def __init__(self, nuscenes_path: Union[str, Path], nuscenes_version: str):
        self.nusc = NuScenes(dataroot=nuscenes_path, version=nuscenes_version, verbose=False)

    def get_lidar_path(self, lidar_token):
        """
        Parameters
        ----------
        lidar_token : str
            sample data token

        Return
        ------
        path_to_lidar_file : str
        """
        return self.nusc.get_sample_data_path(lidar_token)
    
    def get_sample_tokens_of_1scene(self, scene_token):
        """
        Parameters
        ----------
        lidar_token : str
            sample data token

        Return
        ------
        list_sample_tokens : List[str]
        """
        scene_record = self.nusc.get('scene', scene_token)
        sample_token = scene_record['first_sample_token']

        list_sample_tokens = list()

        while sample_token != '':
            list_sample_tokens.append(sample_token)

            sample_record = self.nusc.get('sample', sample_token)
            sample_token = sample_record['next']
        
        return list_sample_tokens


    def get_one_pointcloud(self, lidar_data_token: str) -> np.ndarray:
        """
        Return:
            pointcloud: (N, 4) - x, y, z, reflectant
        """
        pcfile = self.nusc.get_sample_data_path(lidar_data_token)
        points = np.fromfile(str(pcfile), dtype=np.float32, count=-1)
        
        # the following is for Lyft
        if points.shape[0] % 5 != 0:
            points = points[: points.shape[0] - (points.shape[0] % 5)]
        
        points = points.reshape([-1, 5])[:, :4]

        return points

    def get_ground_truth_in_agent_frame(self, sample_token, lidar_token):
        """
        Parameters
        ----------
        sample_token : str
            sample token.
        
        lidar_token : str
            lidar_token of agent where ground truth are transformed into

        Return
        ------
        boxes : np.ndarray
            (N, 7) - [x, y, z, l, w, h, heading]
        
        boxes_class : np.ndarray
            (N,) - str
        """
        nusc = self.nusc
        boxes = list()
        boxes_class = list()
        sample = nusc.get('sample', sample_token)

        for anno_token in sample['anns']:
            anno = nusc.get('sample_annotation', anno_token)
            box = np.zeros(7)
            box[:3] = np.array(anno['translation'])
            box[3: 6] = np.array([anno['size'][1], anno['size'][0], anno['size'][2]])
            box[6] = quaternion_yaw(Quaternion(anno['rotation']))
            name = map_name_from_general_to_detection[anno['category_name']]

            boxes.append(box)
            boxes_class.append(name)

        boxes = np.array(boxes)
        boxes_class = np.array(boxes_class)

        # transform boxes to LIDAR of the ego vehicle frame
        world_se3_lidar = self.get_nuscenes_sensor_pose_in_global(lidar_token)
        apply_se3_(np.linalg.inv(world_se3_lidar), boxes_=boxes)

        return boxes, boxes_class

    def get_nuscenes_sensor_pose_in_ego_vehicle(self, sensor_data_token: str) -> np.ndarray:
        curr_rec = self.nusc.get('sample_data', sensor_data_token)
        curr_cs_rec = self.nusc.get('calibrated_sensor', curr_rec['calibrated_sensor_token'])
        ego_from_curr = make_se3(curr_cs_rec['translation'], quaternion=curr_cs_rec['rotation'])
        return ego_from_curr

    def get_nuscenes_sensor_pose_in_global(self, sensor_data_token: str) -> np.ndarray:
        ego_from_curr = self.get_nuscenes_sensor_pose_in_ego_vehicle(sensor_data_token)
        curr_rec = self.nusc.get('sample_data', sensor_data_token)
        curr_ego_rec = self.nusc.get('ego_pose', curr_rec['ego_pose_token'])
        glob_from_ego = make_se3(curr_ego_rec['translation'], quaternion=curr_ego_rec['rotation'])
        glob_from_curr = glob_from_ego @ ego_from_curr
        return glob_from_curr

    def v2x_sim_get_num_agents(self, scene_token):
        """
        Parameters
        ----------
        scene_token : str
            scene token.

        Return
        ------
        num_agents : int
            num CAVs + num RSU (=1) in this scene.
            Note that #CAVs is constant for all samples of a scene
        """
        scene = self.nusc.get('scene', scene_token)
        sample_data = self.nusc.get('sample', scene['first_sample_token'])['data']
        num_agents = 0
        for sensor_name, _ in sample_data.items():
            if 'LIDAR_TOP' in sensor_name and 'SEMLIDAR_TOP' not in sensor_name:
                num_agents += 1
        
        return num_agents

    def return_sample_token(self, scene_token, sample_idx):
        """
        Parameters
        ----------
        scene_token : str
            scene token.

        sample_idx : int
            index of a sample of this scene

        Return
        ------
        sample_token : str
        """
        scene = self.nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        sidx = 0
        while sidx < sample_idx:
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']
            sidx += 1
        return sample_token


