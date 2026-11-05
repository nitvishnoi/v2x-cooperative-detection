#!/bin/bash

# Modify these paths and GPU ids
DATA_PATH="/home/mdao/Downloads/dataset_zip/opv2v/"
CODE_PATH="/home/mdao/Desktop/usyd_dataset/OpenCOOD"
GPU_ID="0"
DOCKER_IMAGE="test-opencood"

ENVS="  --env=NVIDIA_VISIBLE_DEVICES=$GPU_ID
        --env=CUDA_VISIBLE_DEVICES=$GPU_ID
        --env=NVIDIA_DRIVER_CAPABILITIES=all"


# Setup environmetns for pop-up visualization of point cloud (open3d)
VISUAL="        --env=DISPLAY
                --env=QT_X11_NO_MITSHM=1
                --volume=/tmp/.X11-unix:/tmp/.X11-unix"
xhost +local:docker

echo "Running the docker image [GPUS: ${GPU_ID}]"

# Start docker image
docker  run -d -it --rm \
$ENVS \
$VISUAL \
--mount type=bind,source=$DATA_PATH,target=/data \
--mount type=bind,source=$CODE_PATH,target=/code \
--runtime=nvidia \
--gpus all \
--privileged \
--net=host \
--ipc=host \
--shm-size=30G \
--workdir=/code \
$DOCKER_IMAGE   