# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv


class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        # self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
        #                           kernel_size=1)
        # self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
        #                           kernel_size=1)
        self.cls_head = nn.Conv2d(args['cls_head_dim'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['cls_head_dim'],
                                  7 * args['anchor_number'],
                                  kernel_size=1)


    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        # put psm & rm in the right dim (check voxel_postprocessor_multi)
        psm = rearrange(psm, 'B (num_class num_anchors) h w -> B num_class num_anchors h w', 
                        num_class=1)
        
        rm = rearrange(rm, 'B (num_class C_num_anchors) h w -> B num_class C_num_anchors h w',
                       num_class=1)

        # output_dict = {'psm': psm,
        #                'rm': rm}
        if self.training:
            output_dict = {'psm': psm,  # un-normalize logit
                           'rm': rm}
        else:
            # evaluation
            prob = torch.sigmoid(psm)
            output_dict = {'prob': prob, 
                           'rm': rm,
                           'psm': psm}

        return output_dict