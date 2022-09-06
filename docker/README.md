# Docker for D2SLAM
Our docker including: ros-noetic, ceres-2.1.0, onnxruntime-gpu-1.12.1, libtorch-latest, LCM, faiss, OpenCV4 with CUDA, OPenGV, Backward, and $D^2$SLAM

## Docker pc
Build with 
```
make pc
```

## Docker for Jetson
This docker file is recommend to build on PC with [qemu support](https://www.stereolabs.com/docs/docker/building-arm-container-on-x86/), build on Jetson will be extreme slow.

To build docker for $D^2$SLAM
```
make jetson
```
To build the __base__ image for $D^2$SLAM (which contains the environment for those who would like to modify them)
```
make jetson_base
```
And change in __Dockerfile_jetson_D2SLAM__
```
FROM xuhao1/d2slam:jetson_base_35.1.0
```
to your own image name
```
FROM your-image-name/d2slam:jetson_base_35.1.0
```

Currently this docker support Jetpack 5.0.2/35.1.0. Tested on Xaiver NX.
