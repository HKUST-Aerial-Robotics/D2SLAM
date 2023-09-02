cmake_minimum_required(VERSION 2.8.3)
project(d2frontend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
add_compile_options(-Wno-deprecated-declarations -Wno-reorder  -Wno-format -Wno-sign-compare)
set(USE_ONNX on)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosmsg
  rospy
  std_msgs
  cv_bridge
  swarm_msgs
  swarmcomm_msgs
  d2common
  message_generation
  camera_models
  message_filters
  image_transport
)

SET("OpenCV_DIR"  "/usr/local/share/OpenCV/")
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(lcm REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(opengv REQUIRED)

set(Torch_DIR "$ENV{HOME}/source/libtorch/share/cmake/Torch" CACHE STRING "Path of libtorch")
set(TORCH_INSTALL_PREFIX "$ENV{HOME}/source/libtorch" CACHE STRING "Path of libtorch")
#Find torch Optional
find_package(Torch)
if (NOT Torch_FOUND)
  message("PyTorch not found")
  set(TORCH_INCLUDE_DIRS "$ENV{HOME}/source/libtorch/" CACHE STRING "Path of torch install path")
  set(TORCH_INCLUDE_DIRS
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)
  set(TORCH_LIBRARIES
    ${TORCH_INSTALL_PREFIX}/lib/libtorch.so
    ${TORCH_INSTALL_PREFIX}/lib/libc10.so
    ${TORCH_INSTALL_PREFIX}/lib/libtorch_cpu.so)
endif()
find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${TORCH_INCLUDE_DIRS})

add_definitions("-D USE_ONNX")
set(ONNXRUNTIME_LIB_DIR "/home/xuhao/source/onnxruntime-linux-x64-gpu-1.12.1/lib/" CACHE STRING "Path of ONNXRUNTIME_LIB_DIR")
set(ONNXRUNTIME_INC_DIR "/home/xuhao/source/onnxruntime-linux-x64-gpu-1.12.1/include/" CACHE STRING "Path of ONNXRUNTIME_INC_DIR")

link_directories(${ONNXRUNTIME_LIB_DIR})
link_directories("/usr/local/lib/")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES loop_cnn libd2frontend
  CATKIN_DEPENDS std_msgs cv_bridge roscpp rosmsg rospy swarm_msgs swarmcomm_msgs d2common
  #DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  /usr/local/include/
  include
  ${opengv_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${YAML_CPP_INCLUDE_DIRS}
  ${ONNXRUNTIME_INC_DIR}
)

add_library(libd2frontend
  src/loop_cam.cpp
  src/loop_detector.cpp
  src/loop_net.cpp
  src/d2frontend_params.cpp
  src/d2frontend.cpp
  src/d2featuretracker.cpp
  src/loop_utils.cpp
  src/d2landmark_manager.cpp
)

add_library(${PROJECT_NAME}_nodelet
  src/d2frontend_nodelet.cpp
)

add_executable(${PROJECT_NAME}_node
  src/d2frontend_node.cpp
)

add_executable(${PROJECT_NAME}_spy
  src/d2frontend_spy.cpp
)

add_executable(${PROJECT_NAME}_net_tester
  tests/d2frontend_network_tester.cpp
)

set_property(TARGET ${PROJECT_NAME}_nodelet PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_NAME}_node PROPERTY CXX_STANDARD 14)
set_property(TARGET libd2frontend PROPERTY CXX_STANDARD 14)

add_library(loop_cnn
  src/CNN/superpoint_common.cpp
  src/CNN/superpoint_onnx.cpp
  src/CNN/superglue_onnx.cpp
  src/loop_utils.cpp
  src/d2frontend_params.cpp
)
set_property(TARGET loop_cnn PROPERTY CXX_STANDARD 14)


#Use tensorrt and onnx
target_link_libraries(loop_cnn opencv_dnn 
  onnxruntime
  ${TORCH_LIBRARIES}
  opengv
)

add_executable(loop_tensorrt_test
  tests/loop_tensorrt_test.cpp
)

target_link_libraries(loop_tensorrt_test
  loop_cnn
  dw
  ${YAML_CPP_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES})

add_executable(camera_undistort_test
  tests/camera_undistort_test.cpp
  src/d2frontend_params.cpp
)

target_link_libraries(camera_undistort_test
  dw
  loop_cnn
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES})

add_dependencies(${PROJECT_NAME}_nodelet
    ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_dependencies(${PROJECT_NAME}_spy
    ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(libd2frontend
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  lcm
  faiss
  dw
  loop_cnn
  opengv
)

target_link_libraries(${PROJECT_NAME}_nodelet
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
  lcm
  faiss
  dw
  libd2frontend
)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
  lcm
  dw
  libd2frontend
)


target_link_libraries(${PROJECT_NAME}_net_tester
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
  lcm
  dw
  libd2frontend
)

target_link_libraries(${PROJECT_NAME}_spy
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  lcm
  dw
  libd2frontend
)
