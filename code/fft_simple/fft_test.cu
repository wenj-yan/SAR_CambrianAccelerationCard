#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

// 错误检查宏
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult_t err = call; \
    if (err != CUFFT_SUCCESS) { \
        printf("cuFFT error %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
}

void test_fft() {
    // 设置维度
    const int height = 500;
    const int width = 128 * 1024;
    const size_t input_size = height * width;
    
    // 分配主机内存
    cufftComplex* h_input = (cufftComplex*)malloc(input_size * sizeof(cufftComplex));
    cufftComplex* h_output = (cufftComplex*)malloc(input_size * sizeof(cufftComplex));
    
    // 初始化输入数据
    for (size_t i = 0; i < input_size; i++) {
        h_input[i].x = 1.0f;  // 实部
        h_input[i].y = 0.0f;  // 虚部
    }

    // 打印部分输入数据
    std::cout << "------------------step0: input complex signal----------------\n";
    for (int i = 0; i < 4; i++) {
        printf("(%.1f,%.1f) ", h_input[i].x, h_input[i].y);
    }
    printf("\n");

    // 分配设备内存
    cufftComplex *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input, input_size * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, input_size * sizeof(cufftComplex)));

    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 拷贝数据到设备
    auto h2d_start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(cufftComplex), 
                         cudaMemcpyHostToDevice));
    auto h2d_end = std::chrono::high_resolution_clock::now();

    // 创建 FFT 计划
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, width, CUFFT_C2C, height));

    // 执行 FFT
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUFFT(cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // 拷贝结果回主机
    auto d2h_start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(h_output, d_output, input_size * sizeof(cufftComplex), 
                         cudaMemcpyDeviceToHost));
    auto d2h_end = std::chrono::high_resolution_clock::now();

    // 打印部分结果
    std::cout << "\n------------------step1: fft result----------------\n";
    for (int i = 0; i < 30; i++) {
        printf("(%.8f,%.8f) ", h_output[i].x, h_output[i].y);
        if ((i + 1) % 5 == 0) printf("\n");
    }

    // 计算并打印时间
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    auto h2d_time = std::chrono::duration_cast<std::chrono::microseconds>
                    (h2d_end - h2d_start).count() / 1000.0f;
    auto d2h_time = std::chrono::duration_cast<std::chrono::microseconds>
                    (d2h_end - d2h_start).count() / 1000.0f;

    std::cout << "\n------------------Compute time----------------\n";
    printf("GPU Compute Time (ms): %.3f\n", gpu_time);
    printf("Host to Device Time (ms): %.3f\n", h2d_time);
    printf("Device to Host Time (ms): %.3f\n", d2h_time);
    printf("Total Time (ms): %.3f\n", gpu_time + h2d_time + d2h_time);

    // 清理资源
    cufftDestroy(plan);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
}

int main() {
    test_fft();
    return 0;
}