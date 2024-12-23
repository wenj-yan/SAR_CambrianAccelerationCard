#include "BPA_CUDA.h"

__global__ void generateEchoKernel(cufftComplex* Srnm, float* Ptarget, 
                                 float* ta, float* tr, const SARParams params,
                                 int Mslow, int Nfast) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Mslow * Nfast) return;
    
    int i = idx / Nfast;    // 方位向索引
    int j = idx % Nfast;    // 距离向索引
    
    float t_a = ta[i];
    float t_r = tr[j];
    
    cufftComplex sum = make_cuFloatComplex(0.0f, 0.0f);
    
    // 对每个目标计算回波
    for(int k = 0; k < params.Ntarget; k++) {
        float Az_k = t_a * params.vr - Ptarget[k*3];
        float Rg_k = Ptarget[k*3 + 1];
        float sigma_k = Ptarget[k*3 + 2];
        
        float R_k = sqrtf(Az_k*Az_k + Rg_k*Rg_k + params.H*params.H);
        float tau_k = 2.0f * R_k / params.c;
        float t_k = t_r - tau_k;
        
        if(t_k > 0 && t_k < params.Tr && fabsf(Az_k) < params.La/2) {
            float phase = PI * params.Kr * t_k * t_k - 
                         (4.0f * PI / params.lambda) * R_k;
            
            float cos_val, sin_val;
            sincosf(phase, &sin_val, &cos_val);
            
            cufftComplex echo = make_cuFloatComplex(
                sigma_k * cos_val,
                sigma_k * sin_val
            );
            
            sum.x += echo.x;
            sum.y += echo.y;
        }
    }
    
    Srnm[idx] = sum;
}

__global__ void backProjectionKernel(cufftComplex* Srnm, cufftComplex* Image,
                                   const SARParams params, int Na, int Nr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Na * Nr) return;
    
    int i = idx / Nr;    // 方位向索引
    int j = idx % Nr;    // 距离向索引
    
    // BP算法实现
    float x = (float)j - Nr/2;
    float y = (float)i - Na/2;
    
    cufftComplex sum = make_cuFloatComplex(0.0f, 0.0f);
    
    // 对每个方位位置进行积分
    for(int m = 0; m < params.Mslow; m++) {
        float R = sqrtf(x*x + y*y + params.H*params.H);
        float phase = -4.0f * PI * R / params.lambda;
        
        float cos_val, sin_val;
        sincosf(phase, &sin_val, &cos_val);
        
        cufftComplex factor = make_cuFloatComplex(cos_val, sin_val);
        cufftComplex echo = Srnm[m * params.Nfast + j];
        
        sum.x += echo.x * factor.x - echo.y * factor.y;
        sum.y += echo.x * factor.y + echo.y * factor.x;
    }
    
    Image[idx] = sum;
}

void launchGenerateEcho(cufftComplex* Srnm, float* Ptarget, 
                       float* ta, float* tr, const SARParams params,
                       int Mslow, int Nfast) {
    dim3 blockSize(256);
    dim3 gridSize((Mslow * Nfast + blockSize.x - 1) / blockSize.x);
    
    generateEchoKernel<<<gridSize, blockSize>>>(Srnm, Ptarget, ta, tr, 
                                              params, Mslow, Nfast);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void launchBackProjection(cufftComplex* Srnm, cufftComplex* Image,
                         const SARParams params, int Na, int Nr) {
    dim3 blockSize(256);
    dim3 gridSize((Na * Nr + blockSize.x - 1) / blockSize.x);
    
    backProjectionKernel<<<gridSize, blockSize>>>(Srnm, Image, params, Na, Nr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}