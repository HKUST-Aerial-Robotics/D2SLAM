# Docker for $D^2$SLAM

Our Docker image includes: 

- ros-noetic 
- ceres-2.1.0 
- onnxruntime-gpu-1.12.1 
- libtorch-latest 
- LCM 
- faiss 
- OpenCV4 with CUDA 
- OpenGV 
- Backward 
- $D^2$SLAM

## Docker PC

To build the Docker image for PC, run the following command:

```
$ make pc
```


## Docker for Jetson

This Docker file can be built on a MacBook with Apple Silicon (M1 or M2), X86_64 PC with [qemu support](https://www.stereolabs.com/docs/docker/building-arm-container-on-x86/) or on Jetson. However, in our tests, building on Jetson is takes hours and building on Qemu is even more slow.

We highly recommend building the image on a MacBook Pro with M1/M2 Max. This is possibly the fastest way.

To build the Docker image for $D^2$ SLAM, run:

```
$ make jetson
```


### Build Base Container (Optional)

To build the __base__ image for $D^2$SLAM (which contains the environment for those who would like to modify it), run:
```
$ make jetson_base
```

Then, in __Dockerfile_jetson_D2SLAM__, change:

```
FROM xuhao1/d2slam:jetson_base_35.1.0
```


to your own image name:

```
FROM your-image-name/d2slam:jetson_base_35.1.0
```


This Docker image has been tested on Jetpack 5.0.2/35.1.0 with Xavier NX.
