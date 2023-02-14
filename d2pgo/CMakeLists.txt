cmake_minimum_required(VERSION 3.0.2)
project(d2pgo)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  d2common
  roscpp
  swarm_msgs
)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
SET("OpenCV_DIR"  "/usr/local/share/OpenCV/")
find_package(OpenCV REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES d2pgo
#  CATKIN_DEPENDS d2common d2frontend d2vins roscpp swarm_msgs
#  DEPENDS system_lib
)

include_directories(
  # include
  third_party/
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/d2pgo.cpp
  src/ARockPGO.cpp
  src/rot_init/rotation_initialization.cpp
  src/swarm_outlier_rejection/swarm_outlier_rejection.cpp
  third_party/fast_max-clique_finder/src/findCliqueHeu.cpp
  third_party/fast_max-clique_finder/src/findCliqueHeuInc.cpp
  third_party/fast_max-clique_finder/src/findClique.cpp
  third_party/fast_max-clique_finder/src/graphIO.cpp
  third_party/fast_max-clique_finder/src/utils.cpp
  test/posegraph_g2o.cpp
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node src/d2pgo_node.cpp)
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  dw
)


add_executable(${PROJECT_NAME}_test 
  test/d2pgo_test.cpp 
)
add_dependencies(${PROJECT_NAME}_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  dw
)

add_executable(test_pgo
  test/test_pgo.cpp 
)
add_dependencies(test_pgo ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(test_pgo
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  dw
)

