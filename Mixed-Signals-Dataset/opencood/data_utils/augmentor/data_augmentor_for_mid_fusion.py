# -*- coding: utf-8 -*-
"""
Class for data augmentation
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
from typing import Tuple

from opencood.data_utils.augmentor import augment_utils


class DataAugmentorForMidFusion(object):
    """
    Data Augmentor.

    Parameters
    ----------
    augment_config : list
        A list of augmentation configuration.

    Attributes
    ----------
    data_augmentor_queue : list
        The list of data augmented functions.
    """

    def __init__(self, augment_config, train=True):
        self.data_augmentor_queue = []
        self.train = train
        
        self._rotation_range = None
        self._scaling_range = None

        self._list_aug = list()

        for cur_cfg in augment_config:
            if 'NAME' not in cur_cfg:
                # skip 'use_for_mid
                continue

            if cur_cfg['NAME'] == 'random_world_rotation':
                assert isinstance(cur_cfg['WORLD_ROT_ANGLE'], list) \
                    or isinstance(cur_cfg['WORLD_ROT_ANGLE'], tuple)
                self._rotation_range = cur_cfg['WORLD_ROT_ANGLE']

            if cur_cfg['NAME'] == 'random_world_scaling':
                assert isinstance(cur_cfg['WORLD_SCALE_RANGE'], list) \
                    or isinstance(cur_cfg['WORLD_SCALE_RANGE'], tuple)
                self._scaling_range = cur_cfg['WORLD_SCALE_RANGE']

            cur_augmentor = getattr(self, cur_cfg['NAME'])
            self.data_augmentor_queue.append(cur_augmentor)

            self._list_aug.append(cur_cfg['NAME'])

    def random_world_flip_x(self, gt_boxes: np.ndarray, points: np.ndarray, enable: bool) -> Tuple[np.ndarray]:
        gt_boxes, points = augment_utils.random_flip_along_x(gt_boxes, points, enable)
        return gt_boxes, points
    
    def random_world_flip_y(self, gt_boxes: np.ndarray, points: np.ndarray, enable: bool) -> Tuple[np.ndarray]:
        gt_boxes, points = augment_utils.random_flip_along_y(gt_boxes, points, enable)
        return gt_boxes, points

    def random_world_rotation(self, gt_boxes: np.ndarray, points: np.ndarray, noise_rotation: float) \
        -> Tuple[np.ndarray]:
        gt_boxes, points = augment_utils.global_rotation(gt_boxes, points, self._rotation_range, noise_rotation)
        return gt_boxes, points

    def random_world_scaling(self, gt_boxes: np.ndarray, points: np.ndarray, noise_scale: float) \
        -> Tuple[np.ndarray]:
        gt_boxes, points = augment_utils.global_scaling(gt_boxes, points, self._scaling_range, noise_scale)
        return gt_boxes, points

    def define_augmentation_amount(self, aug_amount: list = None) -> list:
        """
        decide which augmentation will be done

        Parameter:
        ----------
        aug_amount: for debugging purpose
        """
        if aug_amount is None:
            aug_amount = list()
            for aug_method in self._list_aug:
                if aug_method == 'random_world_flip_x' or aug_method == 'random_world_flip_y':
                    action = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                
                elif aug_method == 'random_world_rotation':
                    action = np.random.uniform(self._rotation_range[0], self._rotation_range[1])
                
                elif aug_method == 'random_world_scaling':
                    action = np.random.uniform(self._scaling_range[0], self._scaling_range[1])

                else:
                    raise ValueError(f"{aug_method} is unknown")

                aug_amount.append(action)
        
        return aug_amount

    def forward(self, gt_boxes: np.ndarray, points: np.ndarray, aug_amount: list = None) -> Tuple[np.ndarray]:
        """
        Args:
            gt_boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
            points: (N, 3 + C_in)
            aug_amount: for debugging

        Returns:
            aug_points:
            aug_gt_boxes
        """
        if not self.train:
            # validation -> no augmentation
            return gt_boxes, points

        # make sure gt_boxes are not padded
        assert not np.any(np.all(np.abs(gt_boxes) < 1e-4, axis=1)), \
            f"gt_boxes must not be padded with 0, gt_boxes.shape: {gt_boxes.shape}"
        
        # apply data_aug
        for cur_augmentor, action in zip(self.data_augmentor_queue, aug_amount):
            gt_boxes, points = cur_augmentor(gt_boxes, points, action)
            assert np.any(np.all(np.abs(points[:, :3]) > 1e-3, axis=1))
        
        return gt_boxes, points

