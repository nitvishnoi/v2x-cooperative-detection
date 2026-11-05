from opencood.data_utils.datasets.msig_early_fusion_dataset import MSigEarlyFusion
from opencood.data_utils.datasets.msig_late_fusion_dataset import MSigLateFusion
from opencood.data_utils.datasets.msig_intermediate_fusion_dataset import MSigIntermediateFusion

__all__ = {
    'MSigEarlyFusion': MSigEarlyFusion,
    'MSigLateFusion': MSigLateFusion,
    'MSigIntermediateFusion': MSigIntermediateFusion,
}


def build_dataset(dataset_cfg, visualize=False, train=True, **kwargs):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        **kwargs
    )

    return dataset
