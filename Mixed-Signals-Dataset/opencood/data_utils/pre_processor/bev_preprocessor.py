# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Convert lidar to bev
"""

import numpy as np
import torch
import torch_scatter
from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class BevPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(BevPreprocessor, self).__init__(preprocess_params, train)
        self.lidar_range = self.params['cav_lidar_range']
        self.geometry_param = preprocess_params["geometry_param"]

        height, width, channels = self.geometry_param['input_shape']
        self._scale_yz = width * channels
        self._scale_z = channels
        self._scale_y = width

    def preprocess(self, pcd_raw):
        """
        Preprocess the lidar points to BEV representations.

        Parameters
        ----------
        pcd_raw : np.ndarray
            The raw lidar.

        Returns
        -------
        data_dict : the structured output dictionary.
        """
        bev = torch.zeros(self.geometry_param['input_shape']).float()
        # bev: (800, 800, 51)

        bev_origin = torch.tensor([
            self.geometry_param["L1"], 
            self.geometry_param["W1"], 
            self.geometry_param["H1"]
        ]).float()
        
        pcd_tensor = torch.from_numpy(pcd_raw).float()
        coords = torch.floor(
            (pcd_tensor[:, :3] - bev_origin) / self.geometry_param['res']
        ).long()
        merge_coords = coords[:, 0] * self._scale_yz + \
              coords[:, 1] * self._scale_z + coords[:, 2]
        unq_coords = torch.unique(merge_coords)

        # unpack unq_coords
        unq_coords_3d = torch.stack([
            unq_coords // self._scale_yz,
            (unq_coords % self._scale_yz) // self._scale_z,
            unq_coords % self._scale_z
        ], dim=1)

        # scatter points to bev
        bev[unq_coords_3d[:, 0], unq_coords_3d[:, 1], unq_coords_3d[:, 2]] = 1.0

        # avg intensity
        if pcd_raw.shape[1] == 4:
            # points have intensity
            merge_coords_xy = coords[:, 0] * self._scale_y + coords[:, 1]
            unq_coords_xy, unq_inv = torch.unique(merge_coords_xy, return_inverse=True)
            mean_intensity = torch_scatter.scatter_mean(pcd_tensor[:, 3], unq_inv, dim=0)

            # unpack unq_coords_xy
            unq_coords_2d = torch.stack([
                unq_coords_xy // self._scale_y,
                unq_coords_xy % self._scale_y
            ], dim=1)

            # scatter avg intensity to bev
            bev[unq_coords_2d[:, 0], unq_coords_2d[:, 1], -1] = mean_intensity

        bev = bev.numpy()
        data_dict = {
            "bev_input": np.transpose(bev, (2, 0, 1))
        }
        return data_dict

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        bev_input_list = [
            x["bev_input"][np.newaxis, ...] for x in batch
        ]
        processed_batch = {
            "bev_input": torch.from_numpy(
                np.concatenate(bev_input_list, axis=0))
        }
        return processed_batch

    @staticmethod
    def collate_batch_dict(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : dict
            Dict of list. Each element represents a CAV.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        bev_input_list = [
            x[np.newaxis, ...] for x in batch["bev_input"]
        ]
        processed_batch = {
            "bev_input": torch.from_numpy(
                np.concatenate(bev_input_list, axis=0))
        }
        return processed_batch

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list / dict
            Batched data.
        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            raise NotImplemented
