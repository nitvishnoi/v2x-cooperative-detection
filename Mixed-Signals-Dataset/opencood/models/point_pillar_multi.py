import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.dyn_pillar_vfe import DynamicPillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone


class PointPillarMulti(nn.Module):
    def __init__(self, args):
        super(PointPillarMulti, self).__init__()

        self.num_class = args['num_classes']
        self.num_anchors_per_loc = args['anchor_number']
        self.num_anchors_total = self.num_class * self.num_anchors_per_loc

        # PIllar VFE
        self._use_dyn_voxelization = 'dyn_pillar_vfe' in args
        if self._use_dyn_voxelization:
            self.pillar_vfe = DynamicPillarVFE(args['dyn_pillar_vfe'],
                                               num_point_features=args['num_point_features'],
                                               voxel_size=args['voxel_size'],
                                               point_cloud_range=args['lidar_range'])
        else:
            self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                        num_point_features=args['num_point_features'],
                                        voxel_size=args['voxel_size'],
                                        point_cloud_range=args['lidar_range'])

        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, self.num_anchors_total,
                                  kernel_size=1)
        
        self.reg_head = nn.Conv2d(128 * 3, 7 * self.num_anchors_total,
                                  kernel_size=1)
        
    def forward(self, data_dict):
        
        if not self._use_dyn_voxelization:
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

            batch_dict = {'voxel_features': voxel_features,
                          'voxel_coords': voxel_coords,
                          'voxel_num_points': voxel_num_points}
        else:
            batch_dict = {'points': data_dict['points']}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']  # (B, C, h, w)

        psm = self.cls_head(spatial_features_2d)  # (B, num_anchor_total, h, w)
        rm = self.reg_head(spatial_features_2d)  # (B, 7 * num_anchor_total, h, w)

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

        return output_dict