import numpy as np

from opencood.data_utils.datasets import MSigEarlyFusion

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
    print(f'agg_points: {agg_points.shape}')


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


if __name__ == '__main__':
    main()

'''
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 259.81436157226562, 128.04051208496094, 17.315069198608398 ],
			"boundingbox_min" : [ -151.18905639648438, -136.19036865234375, -27.285449981689453 ],
			"field_of_view" : 60.0,
			"front" : [ -0.19224158979819125, -0.52013436641531618, 0.83216789893963217 ],
			"lookat" : [ 7.1681487831940816, 1.9957238174838041, -4.9559045190009492 ],
			"up" : [ 0.38038461119377359, 0.74219587035595269, 0.55177245091933635 ],
			"zoom" : 0.099999999999999617
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''
