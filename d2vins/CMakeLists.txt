cmake_minimum_required(VERSION 3.0.2)
project(d2vins)

set(CMAKE_CXX_STANDARD 17)
## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-g -O3 -Wall -Wno-deprecated-declarations -Wno-format -Wno-reorder)
# add_compile_options(-g -O0)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  swarm_msgs
)
find_package(d2frontend REQUIRED)

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES d2vins
 CATKIN_DEPENDS d2frontend roscpp rospy swarm_msgs
)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
set(Torch_DIR "$ENV{HOME}/source/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

## Your package locations should be listed before other locations
include_directories(
  /usr/local/include/
  include
  ${catkin_INCLUDE_DIRS}
  ${d2frontend_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
)

# Declare a C++ library
add_library(${PROJECT_NAME}_MSCKF
  src/MSCKF/MSCKF.cpp
  src/MSCKF/MSCKF_state.cpp
)

add_dependencies(${PROJECT_NAME}_MSCKF ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_library(${PROJECT_NAME}_estimator
  src/estimator/d2estimator.cpp
  src/d2imu.cpp
  src/d2vins_types.cpp
  src/d2vins_params.cpp
  src/visualization/visualization.cpp
  src/visualization/CameraPoseVisualization.cpp
  src/estimator/landmark_manager.cpp
  src/estimator/d2vinsstate.cpp
  src/estimator/marginalization/marginalization.cpp
  src/estimator/marginalization/marginalization_info.cpp
  src/factors/projectionTwoFrameOneCamFactor.cpp
  src/factors/projectionTwoFrameOneCamDepthFactor.cpp
  src/factors/projectionTwoFrameTwoCamFactor.cpp
  src/factors/projectionOneFrameTwoCamFactor.cpp
  src/factors/prior_factor.cpp
  src/factors/pose_local_parameterization.cpp
)

add_dependencies(${PROJECT_NAME}_MSCKF ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(${PROJECT_NAME}_node 
  src/d2vins_node.cpp
)

add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${d2frontend_LIBRARIES}
  ${PROJECT_NAME}_MSCKF
  ${PROJECT_NAME}_estimator
  ${CERES_LIBRARIES}
  dw
)

