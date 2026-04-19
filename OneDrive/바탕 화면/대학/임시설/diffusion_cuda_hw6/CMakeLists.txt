cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_ARCHITECTURES 87)
project(
    diffusion_cpp # project name
    VERSION 1.0.0 # version
    DESCRIPTION "Diffusion Inference on Embedded Devices (ESD2026Spring)"
    LANGUAGES CXX CUDA# C/C++
)

set(CMAKE_CXX_STANDARD 20) # use C++ 20
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

add_subdirectory(src)
add_subdirectory(test)

add_compile_options(-fopenmp -O3)

add_executable(sample main.cpp) # create an executable

find_package(OpenMP REQUIRED)
target_link_libraries(sample PRIVATE
        OpenMP::OpenMP_CXX
)

target_link_libraries(sample PRIVATE ${PROJECT_NAME}_lib) # link the library

add_dependencies(sample functionality ${PROJECT_NAME}_lib) # add dependency