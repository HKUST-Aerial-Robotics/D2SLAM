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
SET("TENSORRT_UTILS_INC" "../tensorrt_utils/include/")
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
if(USE_CUDA)
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

  catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES quadcam_depth_est
  #  CATKIN_DEPENDS other_catkin_pkg
  #  DEPENDS system_lib
  )

  include_directories(
  # include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TENSORRT_UTILS_INC}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${d2frontend_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${ONNXRUNTIME_INC_DIR}
  ${CUDA_INCLUDE_DIRS}
  )

  add_subdirectory(../tensorrt_utils  tensorrt_utils)

  ## Declare a C++ library
  add_library(${PROJECT_NAME}
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
    tensorrt_utils
    yaml-cpp
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
    tensorrt_utils
  )

  # add_executable(${PROJECT_NAME}_test src/quadcam_depth_est.cpp test/test_quadcam_depth_est.cpp)
  # target_link_libraries(${PROJECT_NAME}_test
  #   ${catkin_LIBRARIES}
  #   ${YAML_CPP_LIBRARIES}
  #   ${PROJECT_NAME}
  #   ${d2frontend_LIBRARIES}
  #   onnxruntime
  # )
endif()