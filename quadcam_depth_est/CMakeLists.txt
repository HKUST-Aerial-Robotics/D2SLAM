cmake_minimum_required(VERSION 3.0.2)
project(quadcam_depth_est)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
add_compile_options(-Wno-deprecated-declarations -Wno-reorder  -Wno-format -Wno-sign-compare)

find_package(catkin REQUIRED)
SET("OpenCV_DIR"  "/usr/local/share/OpenCV/")
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CUDA REQUIRED)
find_package(d2frontend REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosmsg
  rospy
  std_msgs
  swarm_msgs
  cv_bridge
  d2common
  message_generation
  camera_models
  message_filters
  image_transport
  pcl_ros
)

set(ONNXRUNTIME_LIB_DIR "/home/xuhao/source/onnxruntime-linux-x64-gpu-1.12.1/lib/" CACHE STRING "Path of ONNXRUNTIME_LIB_DIR")
set(ONNXRUNTIME_INC_DIR "/home/xuhao/source/onnxruntime-linux-x64-gpu-1.12.1/include/" CACHE STRING "Path of ONNXRUNTIME_INC_DIR")

add_definitions("-D USE_ONNX")

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES quadcam_depth_est
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

link_directories(${ONNXRUNTIME_LIB_DIR})

include_directories(
# include
${catkin_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
${d2frontend_INCLUDE_DIRS}
${YAML_CPP_INCLUDE_DIRS}
${ONNXRUNTIME_INC_DIR}
${CUDA_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/quadcam_depth_est.cpp
  src/virtual_stereo.cpp
  src/hitnet.cpp
  src/quadcam_depth_est_trt.cpp
)

target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${d2frontend_LIBRARIES}
  ${CUDA_LIBRARIES}
  onnxruntime
  nvinfer
  nvinfer_plugin
  nvonnxparser
)

add_definitions("-Wall -g")

add_executable(${PROJECT_NAME}_node src/quadcam_depth_est_node.cpp)
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
  ${CUDA_LIBRARIES}
  nvinfer
  nvinfer_plugin
  nvonnxparser
)

# add_executable(${PROJECT_NAME}_test src/quadcam_depth_est.cpp test/test_quadcam_depth_est.cpp)
# target_link_libraries(${PROJECT_NAME}_test
#   ${catkin_LIBRARIES}
#   ${YAML_CPP_LIBRARIES}
#   ${PROJECT_NAME}
#   ${d2frontend_LIBRARIES}
#   onnxruntime
# )
