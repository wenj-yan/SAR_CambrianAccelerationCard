#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <complex>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

// CUDA错误检查宏
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(err); \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        printf("CUFFT error %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(err); \
    } \
}

// 系统参数结构体
struct SARParams {
    float c = 3e8;          // 光速
    float fc = 5.3e9;       // 载频
    float lambda;           // 波长
    float Kr = 4e12;        // 调频率
    float Tr = 2.5e-6;      // 脉冲持续时间
    float Br = 1e8;         // 带宽
    float Fr = 1.2e8;       // 采样率
    float vr = 150;         // 平台速度
    float H = 5000;         // 平台高度
    float La = 4;           // 天线长度
    float Rg0 = 10000;      // 场景中心距离
    float Az0 = 0;          // 场景中心方位
    int Nfast;              // 距离向采样点数
    int Mslow;              // 方位向采样点数
    int Ntarget = 5;        // 目标数量
    
    SARParams() {
        lambda = c/fc;
        Nfast = static_cast<int>(Tr * Fr);
        Mslow = 1024;  // 可以根据需要调整
    }
};

class SAR_Processor {
public:
    SAR_Processor();
    ~SAR_Processor();
    void processImage();
    void saveResult(const char* filename);

private:
    void initializeArrays();
    void generateEchoData();
    void applyBPAlgorithm();

    SARParams params;
    cufftHandle fft_plan;
    
    // 设备内存
    cufftComplex *d_Srnm;    // 回波数据
    cufftComplex *d_Image;   // 图像数据
    float *d_Ptarget;        // 目标位置
    float *d_ta;             // 方位时间
    float *d_tr;             // 距离时间
    
    // 主机内存
    std::vector<std::complex<float>> h_Image;
    std::vector<float> h_Ptarget;
    std::vector<float> h_ta;
    std::vector<float> h_tr;
    
    // 图像尺寸
    size_t imageSize;
};