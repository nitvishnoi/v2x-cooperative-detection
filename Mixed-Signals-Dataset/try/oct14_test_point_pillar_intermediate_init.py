from opencood.tools import train_utils
from opencood.hypes_yaml.yaml_utils import load_yaml


def main():
    hypes_yaml = 'opencood/hypes_yaml/point_pillar_intermediate_multi_freeze_rsu.yaml'
    hypes = load_yaml(hypes_yaml)

    model = train_utils.create_model(hypes)
    

if __name__ == '__main__':
    main()
