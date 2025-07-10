cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
project(PointCloudRenderer LANGUAGES CXX CUDA)
add_executable(PointCloudRenderer)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*")
file(GLOB sources1 "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*")
file(GLOB sources2 "${CMAKE_CURRENT_SOURCE_DIR}/src/octree/*.c*")
file(GLOB sources3 "${CMAKE_CURRENT_SOURCE_DIR}/src/util/*.c*")
set(sources ${sources1} ${sources2} ${sources3})


include(FetchContent)

FetchContent_Declare(
	glm
	GIT_REPOSITORY https://github.com/g-truc/glm.git
	GIT_TAG bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
)

FetchContent_MakeAvailable(glm)
set_target_properties(PointCloudRenderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(PointCloudRenderer PROPERTIES CUDA_ARCHITECTURES "86")
target_sources(PointCloudRenderer PRIVATE ${sources})
target_include_directories(PointCloudRenderer PRIVATE include)
target_link_libraries(PointCloudRenderer PRIVATE glm::glm)