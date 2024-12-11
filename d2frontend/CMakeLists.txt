cmake_minimum_required(VERSION 3.0.0)
project(d2frontend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
add_compile_options(-Wno-deprecated-declarations -Wno-reorder -Wno-format -Wno-sign-compare)

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

SET("OpenCV_DIR" "/usr/local/share/OpenCV/")
SET("TENSORRT_UTILS_INC" "../tensorrt_utils/include/")

find_package(OpenCV REQUIRED)
message("opencv: ", ${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
find_package(lcm REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(opengv REQUIRED)

# if CUDA_TOOLKIT_ROOT_DIR not set in environment, set it here
if(NOT DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
  set(ENV{CUDA_TOOLKIT_ROOT_DIR} "/usr/local/cuda")
endif()

if(USE_CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    message("CUDA found")
    add_definitions(-DUSE_CUDA)
    set(ONNXRUNTIME_LIB_DIR "/home/xuhao/source/onnxruntime-linux-x64-gpu-1.12.1/lib/" CACHE STRING "Path of ONNXRUNTIME_LIB_DIR")
    set(ONNXRUNTIME_INC_DIR "/home/xuhao/source/onnxruntime-linux-x64-gpu-1.12.1/include/" CACHE STRING "Path of ONNXRUNTIME_INC_DIR")
    add_subdirectory(../tensorrt_utils tensorrt_utils)
  else()
    message("CUDA not found")
  endif()
endif()

find_package(Torch)
if(Torch_FOUND)
  message("Torch found")
  add_definitions(-DUSE_TORCH)
else()
  message("Torch not found")
  set(Torch_DIR "$ENV{HOME}/source/libtorch/share/cmake/Torch" CACHE STRING "Path of libtorch")
  set(TORCH_INSTALL_PREFIX "$ENV{HOME}/source/libtorch" CACHE STRING "Path of libtorch")
  set(TORCH_INCLUDE_DIRS "/root/libtorch/include")
  set(TORCH_LIBRARIES "/root/libtorch/lib")
endif()

if (NOT USE_CUDA)
  # This is because TORCH compile with wrong opencv with pip install torch==1.12, needs to be fixed
  set(TORCH_LIBRARIES "")
endif()

#Find torch Optional
find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${TORCH_INCLUDE_DIRS})

link_directories(${ONNXRUNTIME_LIB_DIR})
link_directories("/usr/local/lib/")

catkin_package(
  INCLUDE_DIRS include 
  LIBRARIES loop_cnn libd2frontend 
  CATKIN_DEPENDS std_msgs cv_bridge roscpp rosmsg rospy swarm_msgs swarmcomm_msgs d2common
)

include_directories(
  include/
  /usr/local/include/
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
  src/opticaltrack_utils.cpp
  src/pnp_utils.cpp
  src/d2landmark_manager.cpp
  src/feature_matcher.cpp
)

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


#Use onnx
if(USE_CUDA)
  add_library(loop_cnn
    src/CNN/superpoint_common.cpp
    src/CNN/superpoint_tensorrt.cpp
    src/CNN/superglue_onnx.cpp
    src/loop_utils.cpp
    src/d2frontend_params.cpp
  )

  target_link_libraries(loop_cnn
    opencv_dnn
    onnxruntime
    opengv
    nvinfer
    nvinfer_plugin
    nvonnxparser
    tensorrt_utils
    cudart
  )
else()
  add_library(loop_cnn
    src/CNN/superpoint_common.cpp
    src/CNN/superpoint_onnx.cpp
    src/CNN/superglue_onnx.cpp
    src/d2frontend_params.cpp
  )

  target_link_libraries(loop_cnn
    opencv_dnn
    opengv
    onnxruntime
  )
endif()

target_include_directories(loop_cnn PUBLIC
  /usr/include
  /usr/local/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ONNXRUNTIME_INC_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_UTILS_INC}
  ${Boost_LIBRARIES}
)


add_executable(camera_undistort_test
  tests/camera_undistort_test.cpp
  src/d2frontend_params.cpp
)

target_link_libraries(camera_undistort_test
  dw
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  onnxruntime
  )

target_include_directories(camera_undistort_test PUBLIC
  /usr/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_UTILS_INC}
)

# add_executable(${PROJECT_NAME}_net_tester
#   tests/d2frontend_network_tester.cpp)
# target_link_libraries(${PROJECT_NAME}_net_tester
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   ${TORCH_LIBRARIES}
#   lcm
#   dw
#   libd2frontend
# )

# add_executable(${PROJECT_NAME}_spy
#   src/d2frontend_spy.cpp
# )
# add_dependencies(${PROJECT_NAME}_spy
#   ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(${PROJECT_NAME}_spy
#   libd2frontend
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   lcm
#   dw
# )
