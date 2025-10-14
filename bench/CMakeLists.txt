cmake_minimum_required(VERSION 3.16)
project(celerity_backend_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SYCL (Intel oneAPI DPC++)
if(NOT CMAKE_CXX_COMPILER MATCHES "icpx|dpcpp")
    message(STATUS "Attempting to find Intel DPC++ compiler...")
    find_program(DPCPP_COMPILER NAMES icpx dpcpp PATHS ENV PATH)
    if(DPCPP_COMPILER)
        set(CMAKE_CXX_COMPILER ${DPCPP_COMPILER})
        message(STATUS "Found DPC++ compiler: ${DPCPP_COMPILER}")
    else()
        message(FATAL_ERROR "Intel DPC++ compiler (icpx/dpcpp) not found. Please source oneAPI setvars.sh and use: cmake -DCMAKE_CXX_COMPILER=icpx ..")
    endif()
endif()

# SYCL compile and link flags for Intel oneAPI
set(SYCL_FLAGS "-fsycl")

# Intel GPU target flags (Arc/Data Center GPUs)
# Separate flags as list for proper CMake handling
set(INTEL_GPU_FLAGS 
    "-fsycl-targets=spir64_gen"
    "-Xsycl-target-backend=spir64_gen"
    "-device dg2"
)

add_executable(memcpy_linear micro/memcpy_linear.cpp)
target_include_directories(memcpy_linear PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(memcpy_linear PRIVATE ${SYCL_FLAGS} ${INTEL_GPU_FLAGS})
target_link_options(memcpy_linear PRIVATE ${SYCL_FLAGS} ${INTEL_GPU_FLAGS})

add_executable(event_overhead micro/event_overhead.cpp)
target_include_directories(event_overhead PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(event_overhead PRIVATE ${SYCL_FLAGS} ${INTEL_GPU_FLAGS})
target_link_options(event_overhead PRIVATE ${SYCL_FLAGS} ${INTEL_GPU_FLAGS})
