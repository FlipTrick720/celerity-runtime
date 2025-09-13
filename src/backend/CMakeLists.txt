# celerity-runtime/src/backend/CMakeLists.txt:
set(CUDA_OR_CUDA_TOOLKIT_FOUND FALSE)
if(NOT CELERITY_SYCL_IMPL STREQUAL "SimSYCL")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
    find_package(CUDAToolkit)
    set(CUDA_OR_CUDA_TOOLKIT_FOUND ${CUDAToolkit_FOUND})
  else()
    find_package(CUDA)
    set(CUDA_OR_CUDA_TOOLKIT_FOUND ${CUDA_FOUND})
  endif()
endif()

option(CELERITY_ENABLE_CUDA_BACKEND "Enable optimized code paths for CUDA backends" ${CUDA_OR_CUDA_TOOLKIT_FOUND})
option(CELERITY_ENABLE_LEVEL_ZERO_BACKEND "Enable optimized code paths for Intel Level Zero backends" ON)

if(CELERITY_ENABLE_CUDA_BACKEND AND NOT CUDA_OR_CUDA_TOOLKIT_FOUND)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
    find_package(CUDAToolkit REQUIRED)
  else()
    find_package(CUDA REQUIRED)
  endif()
endif()

add_library(celerity_backends STATIC sycl_backend.cc sycl_generic_backend.cc)
target_include_directories(celerity_backends PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(celerity_backends PRIVATE celerity_runtime)
add_sycl_to_target(TARGET celerity_backends SOURCES)

if(CELERITY_ENABLE_CUDA_BACKEND)
  target_sources(celerity_backends PRIVATE sycl_cuda_backend.cc)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
    target_link_libraries(celerity_backends PUBLIC CUDA::cudart)
  else()
    target_link_libraries(celerity_backends PUBLIC ${CUDA_LIBRARIES})
    target_include_directories(celerity_backends 
    PUBLIC
      ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_SOURCE_DIR}/include/backend
    )
  endif()
  target_compile_definitions(celerity_backends PUBLIC CELERITY_DETAIL_BACKEND_CUDA_ENABLED=1)
  message(STATUS "CUDA backend enabled")
else()
  target_compile_definitions(celerity_backends PUBLIC CELERITY_DETAIL_BACKEND_CUDA_ENABLED=0)
endif()

# === Level Zero backend integration ===
if(CELERITY_ENABLE_LEVEL_ZERO_BACKEND)
  message(STATUS "Level Zero backend enabled")
  target_sources(celerity_backends PRIVATE oneapi_backend.cc)
  find_library(ZE_LOADER ze_loader REQUIRED)
  target_link_libraries(celerity_backends PUBLIC ${ZE_LOADER})
  target_compile_definitions(celerity_backends PUBLIC CELERITY_DETAIL_BACKEND_LEVEL_ZERO_ENABLED=1)
endif()
