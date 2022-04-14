cmake_minimum_required(VERSION 3.0.2)
project(d2vins)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

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
#  CATKIN_DEPENDS d2frontend roscpp rospy swarm_msgs
#  DEPENDS system_lib
)

find_package(Eigen3 REQUIRED)

## Your package locations should be listed before other locations
include_directories(
  /usr/local/include/
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Declare a C++ library
add_library(${PROJECT_NAME}_MSCKF
  src/MSCKF/MSCKF.cpp
  src/MSCKF/MSCKF_state.cpp
)

add_dependencies(${PROJECT_NAME}_MSCKF ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(${PROJECT_NAME}_MSCKF_node src/MSCKF/MSCKF_node.cpp)
set_target_properties(${PROJECT_NAME}_MSCKF_node PROPERTIES OUTPUT_NAME node PREFIX "")
add_dependencies(${PROJECT_NAME}_MSCKF_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_MSCKF_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_MSCKF
)

