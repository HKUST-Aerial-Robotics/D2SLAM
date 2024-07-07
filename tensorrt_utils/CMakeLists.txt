cmake_minimum_required(VERSION 3.0.2)
project(tensorrt_utils)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
add_compile_options(-Wno-deprecated-declarations -Wno-reorder  -Wno-format -Wno-sign-compare)

find_package(CUDA REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tensorrt_utils
  ${CUDA_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/logger.cpp
)

target_link_libraries(${PROJECT_NAME}
  nvinfer
  nvinfer_plugin
  nvonnxparser
)

