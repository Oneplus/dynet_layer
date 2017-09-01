cmake_minimum_required(VERSION 2.8.7 FATAL_ERROR)
project (dynet_layer)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_FAST_MATH")

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")   # Disable min/max macros in windef.h
endif()

if (DEFINED ENV{EIGEN3_INCLUDE_DIR} AND NOT DEFINED EIGEN3_INCLUDE_DIR) # use env variable if not set
  set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
endif()
get_filename_component(EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
message("-- Eigen dir is " ${EIGEN3_INCLUDE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

if (DEFINED ENV{DYNET_INCLUDE_DIR} AND NOT DEFINED DYNET_INCLUDE_DIR) # use env variable if not set
  set(DYNET_INCLUDE_DIR $ENV{DYNET_INCLUDE_DIR})
endif()
get_filename_component(DYNET_INCLUDE_DIR "${DYNET_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
message("-- Dynet dir is " ${DYNET_INCLUDE_DIR})
find_package(Dynet REQUIRED)
include_directories(${DYNET_INCLUDE_DIR})

add_subdirectory (dynet_layer)

