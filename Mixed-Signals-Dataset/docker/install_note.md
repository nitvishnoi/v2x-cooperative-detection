# To build the docker image
Install nvidia-docker-tookit after building docker image https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt

In the `docker` directory, execute
```bash
docker build --tag=test-opencood .
```
# After building docker image
## Adapt run.sh to your computer
In the `docker` directory, look for the file `run.sh` and change the variable `DATA_PATH` to the global path to where you store the OPV2V dataset. 
The sturcture of my OPV2V directory is as the following:

```
opv2v
├── validate
│   ├── 2021_08_20_21_48_35
│   ├── 2021_08_21_17_30_41
│   ├── 2021_08_22_13_37_16
│   ├── 2021_08_22_22_01_17
│   ├── 2021_08_23_10_51_24
│   ├── 2021_08_23_13_17_21
│   ├── 2021_08_23_19_42_07
│   ├── 2021_09_09_19_27_35
│   └── 2021_09_11_00_33_16
```
In the `docker` directory, look for the file `run.sh` and change the variable `CODE_PATH` to the global path to where you unzip the OpenCOOD.zip I sent you.

## Launch the container
In the `docker` directory, execute
```bash
sudo ./run.sh
sudo docker exec -it ${CONTAINER_NAME} bash
```
Now, you are in the docker container.

In the directory `/code`, looks for the file `setup.py` and comment the line 29: `install_requires=_read_requirements_file()` if it is not already commented out. 

In the directory `/code`, execute the following commands

```bash
python3 opencood/utils/setup.py build_ext --inplace
```
then,
``` bash
python3 setup.py develop
```
Remember to save the state of your container.

To visualize data of the OPV2V dataset,
```bash
python3 opencood/visualization/vis_data_sequence.py
```

**Note:**
The build command for dependencies for FPV-RCNN (below) doesn't work. However, it is safe to skip to this command
```bash
python opencood/pcdet_utils/setup.py build_ext --inplace
```
