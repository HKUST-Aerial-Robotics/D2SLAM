cmake_minimum_required(VERSION 3.0.2)
project(d2vins)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
add_compile_options(-Wno-deprecated-declarations -Wno-reorder  -Wno-format -Wno-sign-compare)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  swarm_msgs
  d2common
  tf
)
find_package(d2frontend REQUIRED)
find_package(d2common REQUIRED)
find_package(lcm REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES d2vins
 CATKIN_DEPENDS d2frontend roscpp rospy swarm_msgs d2common
)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc calib3d)

## Your package locations should be listed before other locations
include_directories(
  /usr/local/include/
  include/
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
  src/d2vins_params.cpp
  src/visualization/visualization.cpp
  src/visualization/CameraPoseVisualization.cpp
  src/estimator/landmark_manager.cpp
  src/estimator/d2vinsstate.cpp
  src/estimator/marginalization/marginalization.cpp
  src/estimator/ParamResidualInfo.cpp
  src/estimator/solver/VINSConsenusSolver.cpp
  src/estimator/solver/ConsensusSync.cpp
  src/factors/projectionTwoFrameOneCamFactor.cpp
  src/factors/projectionTwoFrameOneCamDepthFactor.cpp
  src/factors/projectionTwoFrameTwoCamFactor.cpp
  src/factors/projectionOneFrameTwoCamFactor.cpp
  src/factors/prior_factor.cpp
  src/network/d2vins_net.cpp
  src/utils/solve_5pts.cpp
)

target_link_libraries(${PROJECT_NAME}_estimator
  ${d2frontend_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node 
  src/d2vins_node.cpp
)

add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${d2frontend_LIBRARIES}
  ${d2common_LIBRARIES}
  ${PROJECT_NAME}_MSCKF
  ${PROJECT_NAME}_estimator
  ${CERES_LIBRARIES}
  ${OpenCV_LIBRARIES}
  dw
  lcm
  spdlog
)

