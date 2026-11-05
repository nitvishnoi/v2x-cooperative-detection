# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.utils.freezed_rsu_fusion_utils import extract_rsu_cav_sparse_processed_lidar, RSUPointPillarBackBone

from opencood.utils.common_utils import measure_tensor_size_in_bytes


class PointPillarIntermediate(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediate, self).__init__()
        self.args = args
        if 'freezed_rsu_config' in args:
            rsu_cfg = args['freezed_rsu_config']
            rsu_cfg['point_pillar_scatter']['grid_size'] = args['point_pillar_scatter']['grid_size']
            self.rsu_backbone = RSUPointPillarBackBone(rsu_cfg, save_downsample_feat_map=True)
            for param in self.rsu_backbone.parameters():
                param.requires_grad = False

        self.num_class = args['num_classes']
        self.num_anchors_per_loc = args['anchor_number']
        self.num_anchors_total = self.num_class * self.num_anchors_per_loc

        # ======================================================
        # ======================================================
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=args['num_point_features'],
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
            
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, self.num_anchors_total,
                                  kernel_size=1)
        
        self.reg_head = nn.Conv2d(128 * 3, 7 * self.num_anchors_total,
                                  kernel_size=1)

    def forward(self, data_dict):
        if 'freezed_rsu_config' in self.args:
            self.rsu_backbone.eval()
            rsu_data_dict, cavs_data_dict, top_dome_indices, cav_indices = extract_rsu_cav_sparse_processed_lidar(data_dict)
            rsu_data_dict = self.rsu_backbone(rsu_data_dict)
            
            # ---------------
            # prepared input for CAV's pillar_vfe & scatter
            batch_dict = cavs_data_dict

            # prepared input for fusion backbone
            batch_dict['freezed_rsu_downsampled_feat_maps'] = rsu_data_dict['freezed_rsu_downsampled_feat_maps']
            batch_dict['top_dome_indices'] = top_dome_indices
            batch_dict['cav_indices'] = cav_indices
            batch_dict['record_len'] = data_dict['record_len']
        else:
            # normal stuff -> RSUs are trained together with CAVs
            batch_dict = {'voxel_features': data_dict['processed_lidar']['voxel_features'],
                          'voxel_coords': data_dict['processed_lidar']['voxel_coords'],
                          'voxel_num_points': data_dict['processed_lidar']['voxel_num_points'],
                          'record_len': data_dict['record_len']}

        # ---
        # ---
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        # put psm & rm in the right dim (check voxel_postprocessor_multi)
        psm = rearrange(psm, 'B (num_class num_anchors) h w -> B num_class num_anchors h w', 
                        num_class=self.num_class)
        
        rm = rearrange(rm, 'B (num_class C_num_anchors) h w -> B num_class C_num_anchors h w',
                       num_class=self.num_class)
        # NOTE: C_num_anchors means C * num_anchors

        if self.training:
            output_dict = {'psm': psm,  # un-normalize logit
                           'rm': rm}
        else:
            # evaluation
            prob = torch.sigmoid(psm)
            output_dict = {'prob': prob, 
                           'rm': rm,
                           'psm': psm}
            
            if 'flag_measure_bandwidth' in data_dict and data_dict['flag_measure_bandwidth']:
                output_dict['size_exchanged_info'] = measure_tensor_size_in_bytes(batch_dict['voxel_features'])
                output_dict['size_exchanged_info'] += measure_tensor_size_in_bytes(batch_dict['voxel_coords'])
                output_dict['size_exchanged_info'] += measure_tensor_size_in_bytes(batch_dict['voxel_num_points'])
                output_dict['num_cavs'] = int(data_dict['record_len'])

        return output_dict