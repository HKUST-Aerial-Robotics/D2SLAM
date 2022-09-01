# FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
FROM nvcr.io/nvidia/tensorrt:22.08-py3
ARG CERES_VERSION=2.1.0
ARG CMAKE_VERSION=3.23.1
ARG ONNX_VERSION=1.12.1
ARG OPENCV_VERSION=4.6.0
ENV SWARM_WS=/root/swarm_ws

#Some basic dependencies
RUN  apt-get -y update && \
      DEBIAN_FRONTEND=noninteractive TZ=Asia/Beijing apt-get -y install tzdata && \
      apt-get install -y wget curl lsb-release git \
      libatlas-base-dev \
      libeigen3-dev \
      libgoogle-glog-dev \
      libsuitesparse-dev \
      libglib2.0-dev \
      libyaml-cpp-dev \
      libdw-dev

#Install ROS
# update ros repository
# some code copied from https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/docker/Dockerfile
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' &&\
RUN sh -c 'echo "deb http://mirrors.ustc.edu.cn/ros/ubuntu/ `lsb_release -cs` main" > /etc/apt/sources.list.d/ros-latest.list' && \
      curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
      apt-get update && \
      if [ "x$(nproc)" = "x1" ] ; then export USE_PROC=1 ; \
      else export USE_PROC=$(($(nproc)/2)) ; fi && \
      apt-get update && apt-get install -y \
      ros-noetic-ros-base \
      ros-noetic-nav-msgs \
      ros-noetic-sensor-msgs \
      ros-noetic-cv-bridge \
      ros-noetic-rviz \
      python3-rosdep \
      python3-rosinstall \
      python3-rosinstall-generator \
      python3-wstool \
      build-essential \
      python3-rosdep \
      ros-noetic-catkin \
      net-tools \
      python3-catkin-tools \
      htop \
      xterm \
      gdb && \
      rosdep init && \
      rosdep update

#Replace CMAKE
RUN   rm /usr/local/bin/cmake && \
      wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/ \
      && rm /tmp/cmake-install.sh \
      && cmake --version

#Install ceres
RUN   git clone https://github.com/HKUST-Swarm/ceres-solver -b D2SLAM && \
      cd ceres-solver && \
      mkdir build && cd build && \
      cmake  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF .. && \
      make -j$(USE_PROC) install && \
      rm -rf ../../ceres-solver && \
      apt-get clean all

#Install ONNXRuntime with CUDA
RUN   git clone --recursive https://github.com/Microsoft/onnxruntime
RUN   cd onnxruntime && \
      git checkout tags/v${ONNX_VERSION} && \
      ./build.sh --config Release --build_shared_lib --parallel \
      --use_cuda --cudnn_home /usr/ --cuda_home /usr/local/cuda --skip_test \
      --use_tensorrt --tensorrt_home /usr/ &&\
      cd build/Linux/Release && \
      make install && \
      rm -rf ../../../onnxruntime

#Install Libtorch (CUDA)
RUN   wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip && \
      unzip libtorch-cxx11-abi-shared-with-deps-latest.zip && \
      rm libtorch-cxx11-abi-shared-with-deps-latest.zip && \
      cp -r libtorch/* /usr/local/

#Install LCM
RUN   git clone https://github.com/lcm-proj/lcm && \
      cd lcm && \
      git checkout tags/v1.4.0 && \
      mkdir build && cd build && \
      cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF .. && \
      make -j$(USE_PROC) install && \
      pip install lcm

#Install Faiss
RUN   git clone https://github.com/facebookresearch/faiss.git && \
      cd faiss && \
      cmake -B build -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DFAISS_ENABLE_GPU=OFF . && \
      make -C build -j faiss && \
      make -C build install && \
      rm -rf ../faiss

#Install OpenCV4 with CUDA
RUN   apt update && \
      apt install libgtk2.0-dev -y && \
      wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip && \
      unzip opencv.zip && \
      rm opencv.zip && \
      git clone https://github.com/opencv/opencv_contrib.git -b ${OPENCV_VERSION} && \
      cd opencv-${OPENCV_VERSION} && \
      mkdir build && cd build && \
      cmake .. \
            -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX=/usr/local \
            -D WITH_CUDA=ON \
            -D WITH_CUDNN=ON \
            -D WITH_CUBLAS=ON \
            -D WITH_TBB=ON \
            -D OPENCV_DNN_CUDA=ON \
            -D OPENCV_ENABLE_NONFREE=ON \
            -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
            -D BUILD_EXAMPLES=OFF \
            -D BUILD_opencv_java=OFF \
            -D BUILD_opencv_python=OFF \
            -D BUILD_TESTS=OFF \
            -D BUILD_PERF_TESTS=OFF \
            -D BUILD_opencv_apps=OFF \
            -D BUILD_LIST=calib3d,features2d,highgui,dnn,imgproc,imgcodecs,\
cudev,cudaoptflow,cudaimgproc,cudalegacy,cudaarithm,cudacodec,cudastereo,\
cudafeatures2d,xfeatures2d,tracking,stereo,\
aruco,videoio && \
      make -j $(USE_PROC) && \
      make install 

#Install OpenGV
RUN git clone https://github.com/laurentkneip/opengv && \
      mkdir opengv/build && cd opengv/build && cmake .. && make && \
      make install

#Install Backward
RUN wget https://raw.githubusercontent.com/bombela/backward-cpp/master/backward.hpp -O /usr/local/include/backward.hpp

#Build D2SLAM
RUN mkdir -p ${SWARM_WS}/src/ && \
      cd ${SWARM_WS}/src/ && \
      git clone https://github.com/HKUST-Swarm/swarm_msgs.git -b D2SLAM && \
      git clone https://github.com/HKUST-Swarm/sync_bag_player.git && \
      git clone https://github.com/ros-perception/vision_opencv.git -b noetic
COPY ./ ${SWARM_WS}/src/
WORKDIR $SWARM_WS
RUN   source "/opt/ros/noetic/setup.bash" && \
      catkin build