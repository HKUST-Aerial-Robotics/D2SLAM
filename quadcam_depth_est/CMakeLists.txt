cmake_minimum_required(VERSION 3.0.2)
project(quadcam_depth_est)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
add_compile_options(-Wno-deprecated-declarations -Wno-reorder  -Wno-format -Wno-sign-compare)

find_package(catkin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
set(Torch_DIR "$ENV{HOME}/source/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
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
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES quadcam_depth_est
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
# include
${catkin_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
${YAML_CPP_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/quadcam_depth_est.cpp
)

target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${d2frontend_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node src/quadcam_depth_est_node.cpp)
add_executable(${PROJECT_NAME}_test src/quadcam_depth_est.cpp test/test_quadcam_depth_est.cpp)
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
)
target_link_libraries(${PROJECT_NAME}_test
  ${catkin_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${d2frontend_LIBRARIES}
  ${TORCH_LIBRARIES}
)

