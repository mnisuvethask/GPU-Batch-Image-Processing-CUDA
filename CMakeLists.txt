cmake_minimum_required(VERSION 3.18)

project(GPU_Batch_Image_Processing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(batch_processor
    src/batch_processor.cpp
    src/image_processing.cu
)

target_include_directories(batch_processor
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    src
)

target_link_libraries(batch_processor
    PRIVATE
    ${OpenCV_LIBS}
)

set_target_properties(batch_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
