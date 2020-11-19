cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/hardswish.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

add_executable(yolov5s_TRT_infer ${PROJECT_SOURCE_DIR}/yolov5s_TRT_infer.cpp)
target_link_libraries(yolov5s_TRT_infer nvinfer)
target_link_libraries(yolov5s_TRT_infer cudart)
target_link_libraries(yolov5s_TRT_infer myplugins)
target_link_libraries(yolov5s_TRT_infer ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

