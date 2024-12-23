#include "BPA_CUDA.h"

// CUDA核函数声明
extern void launchGenerateEcho(cufftComplex* Srnm, float* Ptarget, 
                             float* ta, float* tr, const SARParams params,
                             int Mslow, int Nfast);

extern void launchBackProjection(cufftComplex* Srnm, cufftComplex* Image,
                               const SARParams params, int Na, int Nr);

SAR_Processor::SAR_Processor() {
    initializeArrays();
}

SAR_Processor::~SAR_Processor() {
    // 清理CUDA资源
    cudaFree(d_Srnm);
    cudaFree(d_Image);
    cudaFree(d_Ptarget);
    cudaFree(d_ta);
    cudaFree(d_tr);
}

void SAR_Processor::initializeArrays() {
    // 初始化目标位置
    h_Ptarget = {
        params.Az0-10, params.Rg0-20, 1.0f,
        params.Az0+20, params.Rg0+30, 0.8f,
        params.Az0-30, params.Rg0+10, 1.2f,
        params.Az0+40, params.Rg0-40, 0.9f,
        params.Az0,    params.Rg0,    1.5f
    };
    
    // 初始化时间数组
    float ta_start = -params.La / (2 * params.vr);
    float ta_end = params.La / (2 * params.vr);
    float ta_step = (ta_end - ta_start) / (params.Mslow - 1);
    
    h_ta.resize(params.Mslow);
    for(int i = 0; i < params.Mslow; i++) {
        h_ta[i] = ta_start + i * ta_step;
    }
    
    float tr_step = 1.0f / params.Fr;
    h_tr.resize(params.Nfast);
    for(int i = 0; i < params.Nfast; i++) {
        h_tr[i] = i * tr_step;
    }
    
    // 分配设备内存
    imageSize = params.Mslow * params.Nfast * sizeof(cufftComplex);
    CHECK_CUDA(cudaMalloc(&d_Srnm, imageSize));
    CHECK_CUDA(cudaMalloc(&d_Image, imageSize));
    CHECK_CUDA(cudaMalloc(&d_Ptarget, h_Ptarget.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ta, h_ta.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tr, h_tr.size() * sizeof(float)));
    
    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_Ptarget, h_Ptarget.data(), 
                         h_Ptarget.size() * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ta, h_ta.data(), 
                         h_ta.size() * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tr, h_tr.data(), 
                         h_tr.size() * sizeof(float), 
                         cudaMemcpyHostToDevice));
}

void SAR_Processor::processImage() {
    // 生成回波数据
    generateEchoData();
    
    // 应用BP算法
    applyBPAlgorithm();
}

void SAR_Processor::generateEchoData() {
    launchGenerateEcho(d_Srnm, d_Ptarget, d_ta, d_tr, params,
                      params.Mslow, params.Nfast);
}

void SAR_Processor::applyBPAlgorithm() {
    launchBackProjection(d_Srnm, d_Image, params, params.Mslow, params.Nfast);
    
    // 拷贝结果回主机
    h_Image.resize(params.Mslow * params.Nfast);
    CHECK_CUDA(cudaMemcpy(h_Image.data(), d_Image, imageSize, 
                         cudaMemcpyDeviceToHost));
}

void SAR_Processor::saveResult(const char* filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if(!outFile) {
        throw std::runtime_error("Cannot open file for writing");
    }
    
    outFile.write(reinterpret_cast<const char*>(h_Image.data()), 
                 h_Image.size() * sizeof(std::complex<float>));
}