cmake_minimum_required(VERSION 3.16)
project(celerity_backend_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(memcpy_linear micro/memcpy_linear.cpp)
target_include_directories(memcpy_linear PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(event_overhead micro/event_overhead.cpp)
target_include_directories(event_overhead PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
