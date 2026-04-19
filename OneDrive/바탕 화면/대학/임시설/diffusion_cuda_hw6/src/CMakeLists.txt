add_compile_options(-fopenmp -O3)

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE CUDA_SOURCE_FILES CONFIGURE_DEPENDS "*.cu")

add_library(${PROJECT_NAME}_lib ${SOURCE_FILES} ${CUDA_SOURCE_FILES})
target_compile_options(${PROJECT_NAME}_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_87>)     #sm_87 for Jetson Orin Nano

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME}_lib PRIVATE
        OpenMP::OpenMP_CXX
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME}_lib PRIVATE
        CUDA::cudart
)

target_include_directories(
        ${PROJECT_NAME}_lib
        PUBLIC
            ${CMAKE_SOURCE_DIR}/third_party # add third_party directory to include path
            ${CUDAToolkit_INCLUDE_DIRS}
            include
        PRIVATE
)
