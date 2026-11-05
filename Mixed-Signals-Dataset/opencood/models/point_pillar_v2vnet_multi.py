# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.v2v_fuse import V2VNetFusion
from opencood.utils.freezed_rsu_fusion_utils import \
    extract_rsu_cav_sparse_processed_lidar, RSUPointPillarBackBone, combine_rsu_and_cav_dense_features

from opencood.utils.common_utils import measure_tensor_size_in_bytes


class PointPillarV2VNetMulti(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if 'freezed_rsu_config' in args:
            rsu_cfg = args['freezed_rsu_config']
            rsu_cfg['point_pillar_scatter']['grid_size'] = args['point_pillar_scatter']['grid_size']
            self.rsu_backbone = RSUPointPillarBackBone(rsu_cfg, save_spatial_feature_2d=True)
            for param in self.rsu_backbone.parameters():
                param.requires_grad = False

        self.max_cav = args['max_cav']

        self.num_class = args['num_classes']
        self.num_anchors_per_loc = args['anchor_number']
        self.num_anchors_total = self.num_class * self.num_anchors_per_loc

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=args['num_point_features'],
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        lidar_range = torch.tensor(args['lidar_range'])
        voxel_size = torch.tensor(args['voxel_size'])
        feat_map_WH = torch.floor(
            (lidar_range[3: 5] - lidar_range[:2]) / voxel_size[:2] / args['v2vfusion']['downsample_rate']
        ).long()

        args['v2vfusion']['conv_gru'].update({'W': feat_map_WH[0], 'H': feat_map_WH[1]})
        
        self.fusion_net = V2VNetFusion(args['v2vfusion'])

        self.cls_head = nn.Conv2d(128 * 2, self.num_anchors_total,
                                  kernel_size=1)
        
        self.reg_head = nn.Conv2d(128 * 2, 7 * self.num_anchors_total,
                                  kernel_size=1)
        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        if 'freezed_rsu_config' in self.args:
            self.rsu_backbone.eval()
            rsu_data_dict, cavs_data_dict, top_dome_indices, cav_indices = extract_rsu_cav_sparse_processed_lidar(data_dict)
            rsu_data_dict = self.rsu_backbone(rsu_data_dict)
            
            # ---------------
            # prepared input for CAV's pillar_vfe & scatter
            batch_dict = cavs_data_dict

            # prepared input for fusion backbone
            batch_dict['rsu_spatial_features_2d'] = rsu_data_dict['freezed_rsu_spatial_features_2d']
            batch_dict['top_dome_indices'] = top_dome_indices
            batch_dict['cav_indices'] = cav_indices
            batch_dict['record_len'] = data_dict['record_len']
        else:
            # normal stuff -> RSUs are trained together with CAVs
            batch_dict = {'voxel_features': data_dict['processed_lidar']['voxel_features'],
                          'voxel_coords': data_dict['processed_lidar']['voxel_coords'],
                          'voxel_num_points': data_dict['processed_lidar']['voxel_num_points'],
                          'record_len': data_dict['record_len']}

        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        if 'freezed_rsu_config' in self.args:
            # combine 'freezed_rsu_downsampled_feat_maps' and 'spatial_features_2d'
            spatial_features_2d = combine_rsu_and_cav_dense_features(
                batch_dict['rsu_spatial_features_2d'],
                spatial_features_2d,
                batch_dict['top_dome_indices'],
                batch_dict['cav_indices']
            )
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        fused_feature = self.fusion_net(spatial_features_2d,
                                        data_dict['record_len'],
                                        pairwise_t_matrix)

        psm = self.cls_head(fused_feature)  # (B, num_anchor_total, h, w)
        rm = self.reg_head(fused_feature)  # (B, 7 * num_anchor_total, h, w) 

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
                output_dict['size_exchanged_info'] = measure_tensor_size_in_bytes(spatial_features_2d)
                output_dict['num_cavs'] = int(data_dict['record_len'])

        return output_dict
