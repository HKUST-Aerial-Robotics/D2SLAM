#!/bin/bash

# This script is used to debug in the docker container
# Options: arm64/jetson/pc
# Usage: ./debug_in_docker.sh arm64

if [ $# -ne 1 ]; then
    echo "Usage: $0 arm64/jetson/pc"
    exit 1
fi

# Read arguments
ARCH=$1

# Set the docker image
if [ $ARCH == "arm64" ]; then
    DOCKER_IMAGE="hkustswarm/d2slam:arm64_base_ros1_noetic"
elif [ $ARCH == "orin" ]; then
    DOCKER_IMAGE="hkustswarm/d2slam:jetson_orin_base_35.3.1"
elif [ $ARCH == "pc" ]; then
    DOCKER_IMAGE="hkustswarm/d2slam:x86"
else
    echo "Invalid argument: $ARCH"
    exit 1
fi

# Set the docker container name
CONTAINER_NAME="d2slam_debug"

# Create space to store the ros workspace
mkdir -p $(pwd)/../../d2slam_docker_ros_space

# Run the docker container
# Is pc or orin, enable gpu
if [ $ARCH == "pc" ] || [ $ARCH == "orin" ]; then
    xhost +local:root
    docker run -it --rm --name $CONTAINER_NAME \
        --gpus all \
        --net=host \
        --privileged \
        --rm \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$(pwd)/../../d2slam_docker_ros_space:/root/ros_space" \
        --volume="$(pwd)/../:/root/ros_space/src/D2SLAM" \
        $DOCKER_IMAGE \
        /bin/bash
else
    docker run -it --rm --name $CONTAINER_NAME \
        --net=host \
        --privileged \
        --rm \
        --name="d2slam" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --env="DISPLAY=host.docker.internal:0" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$(pwd)/../../d2slam_docker_ros_space:/root/swarm_ws" \
        --volume="$(pwd)/../:/root/swarm_ws/src/D2SLAM" \
        --volume="/Users/xuhao/Dropbox/data:/root/data" \
        $DOCKER_IMAGE \
        /bin/bash
fi
