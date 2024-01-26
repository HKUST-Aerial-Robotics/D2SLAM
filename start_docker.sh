#!/bin/bash
# Usage: ./start_docker.sh  0 to start docker with current file-dir, anything you modify in docker will be saved in current file-dir
# Usage: ./start_docker.sh  1 to start docker only for image transportation.
# Please do not move this file to other dir, it will cause the docker container can not find the current dir.

script_path=$(dirname "$(readlink -f "$0")")

source $script_path/docker_files_config_local.sh

if [ $# -eq 0 ]; then
  echo "[INFO] No start option, will start docker container only for application"
  START_OPTION=0
else
  echo "[INFO] Start option is ${1}"
  START_OPTION=$1
fi
xhost +
if [ ${START_OPTION} == 1 ]; then
  echo "[INFO] Start docker container with mapping current dir to docker container"
  CURRENT_DIR=$script_path
  echo "${CURRENT_DIR} will be mapped in to docker container with start option 1"
  docker run -it --rm --runtime=nvidia --gpus all  --net=host \
    -v ${CURRENT_DIR}:${SWARM_WS}/src/D2SLAM \
    -v ${CONFIGS}:${SWARM_WS}/src/configs-drone-swarm \
    -v ${NN_MODELS}:${SWARM_WS}/src/NNmodels_generator \
    -v /dev/:/dev/  -v ${DATA_SET}:/data/ --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  --name="d2slam_container" ${DOCKERIMAGE} /bin/bash 
else
  echo "Start docker container for image transportation only"
  docker run -it --rm --runtime=nvidia --gpus all  --net=host -v /dev/:/dev/ \
    -v ${CONFIGS}:${SWARM_WS}/src/configs-drone-swarm \
    -v ${NN_MODELS}:${SWARM_WS}/src/NNmodels_generator \
    --privileged -e DISPLAY  -v ${DATA_SET}:/data/  -v /tmp/.X11-unix:/tmp/.X11-unix --name="d2slam_container"  ${DOCKERIMAGE} /bin/bash \
    -c "source ./devel/setup.bash && roslaunch d2vins quadcam.launch"
fi