cmake_minimum_required(VERSION 3.0.2)
project(d2common)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-g -O3 -Wall -Wno-deprecated-declarations -Wno-format -Wno-reorder)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosmsg
  rospy
  sensor_msgs
  swarm_msgs
  swarmcomm_msgs
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES d2common
 CATKIN_DEPENDS roscpp rosmsg rospy sensor_msgs swarm_msgs swarmcomm_msgs
#  DEPENDS system_lib
)

include_directories(
  /usr/local/include/
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/d2imu.cpp
  src/d2vinsframe.cpp
  src/solver/BaseParamResInfo.cpp
  src/solver/BaseSolverWrapper.cpp
  src/solver/ConsensusSolver.cpp
  src/solver/consenus_factor.cpp
  src/solver/ARock.cpp
  src/solver/pose_local_parameterization.cpp
)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(${PROJECT_NAME}_test 
  src/test.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}_test 
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)
