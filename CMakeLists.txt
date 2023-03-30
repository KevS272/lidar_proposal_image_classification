cmake_minimum_required(VERSION 3.0.2)
project(lidar_proposal_image_classification)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  fs_msgs
  image_transport
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  tf2
)

# CUDA configuration
find_package(CUDA REQUIRED)
message("-- CUDA version: ${CUDA_VERSION}")
set(CUDA_VERBOSE_BUILD ON)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
  find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()

# catkin specific configuration
catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS
    cv_bridge
    image_transport
    pcl_ros
    pcl_conversions
    roscpp
    rospy
    sensor_msgs
    tf2
    fs_msgs
)

# Build
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
CUDA_INCLUDE_DIRECTORIES(
  ${CUDNN_INCLUDE_DIR} 
  ${TENSORRT_INCLUDE_DIR}
  )

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} 
  src/common.cpp
  src/Projector.cpp 
  src/Classifier.cpp 
  src/liproic_node.cpp 
  )

target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${TENSORRT_LIBRARY} )

