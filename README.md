# Intro

## Prerequirements
Ubuntu 20.04 with gcc-9.
OpenCV 4.
Ceres latest version.

Compile onnx.

```bash
$ ./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_cuda --cudnn_home /usr/local/cuda --cuda_home /usr/local/cuda
cd build/Linux/RelWithDebInfo
sudo make install
```