FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt update && apt install locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8
RUN apt install software-properties-common -y
RUN add-apt-repository universe
RUN apt update && apt install curl -y
RUN apt update && apt upgrade -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt install python3-pip -y
RUN pip install scikit-image==0.19.3
RUN pip install numpy==1.22.4
RUN pip install scipy==1.12.0
RUN pip install spconv-cu117	
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install nuscenes-devkit einops
RUN pip install open3d==0.18.0
RUN pip install cython==3.0.11

WORKDIR /root
