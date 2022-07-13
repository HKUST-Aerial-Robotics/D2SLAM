cmake_minimum_required(VERSION 3.0.2)
project(d2comm)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosmsg
  rospy
  sensor_msgs
  swarm_msgs
  swarmcomm_msgs
  d2common
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(lcm REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES d2comm
#  CATKIN_DEPENDS roscpp rosmsg rospy sensor_msgs swarm_msgs swarmcomm_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_LIBRARIES}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/d2comm.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  lcm
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(${PROJECT_NAME}_node src/d2comm_node.cpp)

add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)
