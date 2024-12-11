cmake_minimum_required(VERSION 3.0.2)
project(tensorrt_utils)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
add_compile_options(-Wno-deprecated-declarations -Wno-reorder  -Wno-format -Wno-sign-compare)

if (NOT DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
  set(ENV{CUDA_TOOLKIT_ROOT_DIR} "/usr/local/cuda")
endif()

if (USE_CUDA)
  find_package(CUDA)
  if (CUDA_FOUND)
    message("CUDA found")
    add_definitions(-DUSE_CUDA)
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
  else()
    message("CUDA not found, skipping tensorrt_utils")
  endif()
else()
  message("CUDA disabled, skipping tensorrt_utils")
endif()
