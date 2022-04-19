cmake_minimum_required(VERSION 3.0.2)
project(d2vins)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -Wall -Wno-deprecated-declarations -Wno-format -Wno-reorder")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  d2frontend
  roscpp
  rospy
  swarm_msgs
)

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES d2vins
 CATKIN_DEPENDS d2frontend roscpp rospy swarm_msgs
#  DEPENDS system_lib
)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

## Your package locations should be listed before other locations
include_directories(
  /usr/local/include/
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
)

# Declare a C++ library
add_library(${PROJECT_NAME}_MSCKF
  src/MSCKF/MSCKF.cpp
  src/MSCKF/MSCKF_state.cpp
)

link_directories(/home/xuhao/swarm_ws/devel/lib)

add_dependencies(${PROJECT_NAME}_MSCKF ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_library(${PROJECT_NAME}_estimator
  src/estimator/d2estimator.cpp
  src/d2imu.cpp
)

add_dependencies(${PROJECT_NAME}_MSCKF ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(${PROJECT_NAME}_node 
  src/d2vins_node.cpp
  src/d2vins_params.cpp
)

add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_MSCKF
  ${PROJECT_NAME}_estimator
  ${CERES_LIBRARIES}
  libd2frontend
  dw
)

