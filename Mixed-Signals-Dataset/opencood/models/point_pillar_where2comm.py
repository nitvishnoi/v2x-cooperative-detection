import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

from opencood.utils.common_utils import measure_tensor_size_in_bytes


class PointPillarWhere2comm(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2comm, self).__init__()
        self.max_cav = 5
        self.num_class = args['num_classes']
        self.num_anchors_per_loc = args['anchor_number']
        self.num_anchors_total = self.num_class * self.num_anchors_per_loc

        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        self.fusion_net = Where2comm(args['where2comm_fusion'])
        self.multi_scale = args['where2comm_fusion']['multi_scale']

        self.cls_head = nn.Conv2d(args['head_dim'], self.num_anchors_total, kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * self.num_anchors_total, kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
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
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # N, C, H', W': [N, 256, 48, 176]
        spatial_features_2d = batch_dict['spatial_features_2d']
        # Down-sample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)

        # Compressor
        if self.compression:
            # The ego feature is also compressed
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        if self.multi_scale:
            # Bypass communication cost, communicate at high resolution, neither shrink nor compress
            fused_feature, communication_rates = self.fusion_net(batch_dict['spatial_features'],
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix,
                                                                 self.backbone)
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates = self.fusion_net(spatial_features_2d,
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

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
                if not self.multi_scale:
                    output_dict['size_exchanged_info'] = measure_tensor_size_in_bytes(spatial_features_2d)
                else:
                    output_dict['size_exchanged_info'] = measure_tensor_size_in_bytes(batch_dict['spatial_features'])
                output_dict['num_cavs'] = int(data_dict['record_len'])
            
        return output_dict
