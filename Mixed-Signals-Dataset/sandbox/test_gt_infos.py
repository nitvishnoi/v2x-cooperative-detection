import numpy as np

from opencood.data_utils.datasets import MSigEarlyFusion
from opencood.tools.point_cloud_tools import remove_points_outside_range

from mixedsignals.utils.mixed_signals_utils import AGENT_COLOR
from mixedsignals.utils.o3d_viz_utils import PointPainter


def main():
    params = {
        'data_root': 'data/usyd-v2x-mini',
		'flag_train': False,
		'flag_verbose': True,
		'flag_pad_agent_indicator': True,
        'path_gt_infos': {
            'test': 'data/usyd-v2x-mini/test_gt_infos.pkl'
		}
	}
    dataset = MSigEarlyFusion(params)
    agg_points, gt_info_dict = dataset[0]
    gt_boxes = gt_info_dict['gt_boxes_in_top']

    agg_points = remove_points_outside_range(agg_points, dataset.make_gt_info_range)
    print(f'agg_points: {agg_points.shape}')

    print('showing aggregated point clouds and all gt boxes')

    painter = PointPainter()
    pts_indicators = agg_points[:, -1].astype(int)
    unq_indicators = np.unique(pts_indicators)
    colors_agg_points = np.zeros((agg_points.shape[0], 3))
    for ind in unq_indicators:
        mask_agent = pts_indicators == ind
        agent_name = dataset.convert_agent_indicator_to_name(ind)
        colors_agg_points[mask_agent] = AGENT_COLOR[agent_name]

    painter.add_pointclouds_(agg_points, colors_agg_points)
    painter.add_boxes_(gt_boxes)
    painter.show()

    chosen_agent_name = '003'
    chosen_agent_ind = dataset.convert_agent_name_to_indicator(chosen_agent_name)
    print(f'showing point clouds of agent {chosen_agent_name}')

    mask_agent_points = agg_points[:, -1].astype(int) == chosen_agent_ind
    agg_points = agg_points[mask_agent_points]
    colors_agg_points = colors_agg_points[mask_agent_points]

    gt_visibility = gt_info_dict['gt_visibility'].astype(bool)
    print(f'gt_visibility:\n{gt_visibility}')
    mask_agent_gt = gt_visibility[:, chosen_agent_ind]
    gt_boxes = gt_boxes[mask_agent_gt]

    painter = PointPainter()
    painter.add_pointclouds_(agg_points, colors_agg_points)
    painter.add_boxes_(gt_boxes)
    painter.show()



if __name__ == '__main__':
    main()
