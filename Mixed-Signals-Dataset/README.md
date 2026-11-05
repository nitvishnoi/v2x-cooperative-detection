# Installation

## Install the devkit of Mixed Signals dataset

```bash
git clone github.com/quan-dao/mixed-signals-devkit
cd mixed-signals-devkit
pip install -r requirements.txt
python setup.py develop
```

## Install OpenCOOD

```bash
git clone <this_repo>
cd OpenCOOD
pip install -r requirements.txt
python opencood/utils/setup.py build_ext --inplace
python opencood/pcdet_utils/setup.py build_ext --inplace
python setup.py develop
```

# Prepare Mixed Signals dataset
## Download Mixed Signales dataset
Download our dataset from `???`.
After unpacking the `???.zip` file, the structure of the dataset directory is as following

```bash
mixed-signals
├── Odometry
│   ├── mini_17
│   ├── mini_21
│   ├── mini_4
│   └── ...
├── PointClouds
│   ├── mini_17
│   ├── mini_21
│   ├── mini_4
│   └── ...
└── V2X_dataset-v0.4-labels.json
```

The **Odometry** directory contains csv files that store the localisation of vehicles existing in each segment of our dataset. For example,

```bash
mini_17
├── odometry_003.csv
├── odometry_004.csv
└── odometry_laser.csv
```

The **PointClouds** directory contains pcd files that store the point cloud of each agent, including 3 vehicles and 2 road-side lidars, at every syncrhonized time step of each segment. For example,

```bash
mini_17
├── top_1_1712121711.218860737.pcd
├── dome_1_1712121711.224186066.pcd
├── 003_1_1712121711.188474014.pcd
├── 004_1_1712121711.204070535.pcd
├── laser_1_1712121711.215227296.pcd
├── ...
```

The naming convetion of point cloud files is 

```bash
<agent_name>_<index_of_synchronized_timestep_in_a_segment>_<time_stamp_this_pointcloud>.pcd
```

The time stamp contains two parts: 
- before the decimal point: seconds
- after the decimal point: nanoseconds

## Preprocess ground truths

```bash
cd OpenCOOD
python opencood/tools/msig_dataset_preprocessing.py
```

This script creates `train_gt_infos.pkl` and `test_gt_infos.pkl` in the root directory of Mixed Signals dataset.

# Evaluate our models

Download check points from `???` and put them in `OpenCOOD/ckpt`

Create a directory to store the evaluation result

```bash
cd OpenCOOD
mkdir output
```

Evaluate Early fusion
```bash
cd OpenCOOD
python opencood/tools/inference.py --hypes_yaml opencood/hypes_yaml/msig_pillar_early.yaml --ckpt pointpillar_early_fusion.pth --fusion_method early --save_eval_result_to_directory output
```

Late Fusion
```bash
cd OpenCOOD
python opencood/tools/inference.py --hypes_yaml opencood/hypes_yaml/msig_pillar_late.yaml --ckpt pointpillar_no_fusion_net_epoch20.pth --fusion_method late --save_eval_result_to_directory output
```
