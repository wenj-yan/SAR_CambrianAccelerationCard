cmake_minimum_required(VERSION 3.8)
project(BPA_SAR CUDA CXX)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 89)  # 根据您的GPU调整

# 查找 CUDA 包
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 添加可执行文件
add_executable(BPA_SAR
    main.cpp
    BPA_CUDA.cpp
    BPA_CUDA_kernels.cu
)

# 链接 CUDA 库
target_link_libraries(BPA_SAR
    ${CUDA_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
)

# 设置 C++ 标准
set_property(TARGET BPA_SAR PROPERTY CXX_STANDARD 14)