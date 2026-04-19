add_compile_options(-fopenmp -O3)

add_executable(functionality functionality.cpp)
target_link_libraries(functionality PRIVATE ${PROJECT_NAME}_lib)