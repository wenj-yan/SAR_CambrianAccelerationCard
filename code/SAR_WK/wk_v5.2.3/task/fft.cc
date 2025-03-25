#include <time.h>
#include <string>
#include "string.h"
#include <complex>
#include <iostream>
#include <complex.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <algorithm> 
#include <unordered_set>


#include "task.h"


void perform2DFFT(cnnlHandle_t handle, 
                    cnrtQueue_t queue,
                    void* d_data,
                    size_t Na, 
                    size_t Nr) 
{   
    cout << "\n执行2D FFT..." << endl;
    // 创建张量描述符
    cnnlTensorDescriptor_t fft_Na_Nr_desc,fft_Nr_Na_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&fft_Na_Nr_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&fft_Nr_Na_desc));
    // FFT公共参数
    int rank = 1;  
    // 先进行距离向FFT
    int fft_dims_1[] = {static_cast<int>(Na), static_cast<int>(Nr)};
    int fft_dims_2[] = {static_cast<int>(Nr), static_cast<int>(Na)};  
    CNNL_CHECK(cnnlSetTensorDescriptor(fft_Na_Nr_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, fft_dims_1));
    CNNL_CHECK(cnnlSetTensorDescriptor(fft_Nr_Na_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, fft_dims_2));
    CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(fft_Na_Nr_desc, CNNL_DTYPE_FLOAT));
    CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(fft_Nr_Na_desc, CNNL_DTYPE_FLOAT));
    // 配置距离向FFT
    size_t fft_range_workspace_size = 0;
    void *fft_range_workspace = nullptr;
    size_t fft_range_reservespace_size = 0;
    void *fft_range_reservespace = nullptr;

    cnnlFFTPlan_t fft_range_desc;
    CNNL_CHECK(cnnlCreateFFTPlan(&fft_range_desc));
    //性能计时器
    float hardware_time;
    cnrtNotifier_t start = nullptr, end = nullptr;
    cnrtNotifierCreate(&start);
    cnrtNotifierCreate(&end);
    // 设置FFT参数
    int n_range[] = {static_cast<int>(Nr)};  // FFT长度
    
    // 初始化FFT计划
    CNNL_CHECK(cnnlMakeFFTPlanMany(handle, fft_range_desc, fft_Na_Nr_desc, fft_Na_Nr_desc, 
                                rank, n_range, &fft_range_reservespace_size, 
                                &fft_range_workspace_size));

    // 分配工作空间
    if (fft_range_workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&fft_range_workspace, fft_range_workspace_size));
    }
    if (fft_range_reservespace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&fft_range_reservespace, fft_range_reservespace_size));
        CNNL_CHECK(cnnlSetFFTReserveArea(handle, fft_range_desc, fft_range_reservespace));
    }
    void* d_fft;
    CNRT_CHECK(cnrtMalloc((void **)&d_fft, Na * Nr * sizeof(complex<float>)));
    // 执行距离向FFT，添加缩放因子 1.0
    
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(start, queue);
    }
    CNNL_CHECK(cnnlExecFFT(handle, fft_range_desc, d_data, 1.0, 
                        fft_range_workspace, d_fft, 0));
    
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "距离向FFT执行时间: " << hardware_time / 1000 << " ms" << endl;
    }      
    
    // 转置数据 - 第一次转置
    void* d_transposed;
    CNRT_CHECK(cnrtMalloc((void **)&d_transposed, Na * Nr * sizeof(complex<float>)));
    // 设置转置描述符
    cnnlTransposeDescriptor_t trans_desc;
    CNNL_CHECK(cnnlCreateTransposeDescriptor(&trans_desc));
    int perm[] = {1, 0};  // 交换维度
    CNNL_CHECK(cnnlSetTransposeDescriptor(trans_desc, 2, perm));
    // 获取工作空间大小
    size_t transpose_workspace_size = 0;
    CNNL_CHECK(cnnlGetTransposeWorkspaceSize(handle, fft_Na_Nr_desc, trans_desc, &transpose_workspace_size));
    
    // 分配工作空间
    void* transpose_workspace = nullptr;
    if (transpose_workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&transpose_workspace, transpose_workspace_size));
    }

    // 执行第一次转置：[Na, Nr] -> [Nr, Na]
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(start, queue);
    }
    
    CNNL_CHECK(cnnlTranspose_v2(handle, trans_desc, fft_Na_Nr_desc, d_fft,
                                fft_Nr_Na_desc, d_transposed,
                                transpose_workspace, transpose_workspace_size));
    CNRT_CHECK(cnrtQueueSync(queue));
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "第一次转置执行时间: " << hardware_time / 1000 << " ms" << endl;
    }
    // 配置方位向FFT
    size_t fft_azimuth_workspace_size = 0;
    void *fft_azimuth_workspace = nullptr;
    size_t fft_azimuth_reservespace_size = 0;
    void *fft_azimuth_reservespace = nullptr;
    
    cnnlFFTPlan_t fft_azimuth_desc;
    CNNL_CHECK(cnnlCreateFFTPlan(&fft_azimuth_desc));
    int n_azimuth[] = {static_cast<int>(Na)};  // 方位向FFT长度
    
    // 初始化FFT计划，使用新的描述符
    CNNL_CHECK(cnnlMakeFFTPlanMany(handle, fft_azimuth_desc, fft_Nr_Na_desc, fft_Nr_Na_desc, 
                                rank, n_azimuth, &fft_azimuth_reservespace_size, 
                                &fft_azimuth_workspace_size));

    // 分配工作空间
    if (fft_azimuth_workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&fft_azimuth_workspace, fft_azimuth_workspace_size));
    }
    if (fft_azimuth_reservespace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&fft_azimuth_reservespace, fft_azimuth_reservespace_size));
        CNNL_CHECK(cnnlSetFFTReserveArea(handle, fft_azimuth_desc, fft_azimuth_reservespace));
    }

    // 执行方位向FFT，添加缩放因子 1.0
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(start, queue);
    }
    CNNL_CHECK(cnnlExecFFT(handle, fft_azimuth_desc, d_transposed, 1.0, 
                        fft_azimuth_workspace, d_fft, 0));
    CNRT_CHECK(cnrtQueueSync(queue)); 
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "方位向FFT执行时间: " << hardware_time / 1000 << " ms" << endl;
    }
    
    // 最后再次转置回原始维度：[Nr, Na] -> [Na, Nr]
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(start, queue);
    }
    CNNL_CHECK(cnnlTranspose_v2(handle, trans_desc, fft_Nr_Na_desc, d_fft,
                                fft_Na_Nr_desc, d_data,
                                transpose_workspace, transpose_workspace_size));
    CNRT_CHECK(cnrtQueueSync(queue));
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "第二次转置执行时间: " << hardware_time / 1000 << " ms" << endl;
    }
    // 清理资源
    CNNL_CHECK(cnnlDestroyTensorDescriptor(fft_Na_Nr_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(fft_Nr_Na_desc));
    CNNL_CHECK(cnnlDestroyTransposeDescriptor(trans_desc));
    if (transpose_workspace) {
        CNRT_CHECK(cnrtFree(transpose_workspace));
    }
    CNRT_CHECK(cnrtFree(d_transposed));
    CNRT_CHECK(cnrtFree(d_fft));
    CNNL_CHECK(cnnlDestroyFFTPlan(fft_range_desc));
    CNNL_CHECK(cnnlDestroyFFTPlan(fft_azimuth_desc));

    if (fft_range_workspace) {
        CNRT_CHECK(cnrtFree(fft_range_workspace));
        CNRT_CHECK(cnrtFree(fft_range_reservespace));
    }
    if (fft_azimuth_workspace) {
        CNRT_CHECK(cnrtFree(fft_azimuth_workspace));
        CNRT_CHECK(cnrtFree(fft_azimuth_reservespace));
    }

};

void perform2DIFFT(cnnlHandle_t handle, 
                    cnrtQueue_t queue,
                    void* d_data,
                    size_t Na, 
                    size_t Nr)
{
    //2D IFFT
    cout << "\n执行2D IFFT..." << endl;
    // 创建张量描述符
    cnnlTensorDescriptor_t fft_Na_Nr_desc,fft_Nr_Na_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&fft_Na_Nr_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&fft_Nr_Na_desc));
    // FFT公共参数
    int rank = 1;  
    // 先进行距离向FFT
    int fft_dims_1[] = {static_cast<int>(Na), static_cast<int>(Nr)};
    int fft_dims_2[] = {static_cast<int>(Nr), static_cast<int>(Na)};  
    CNNL_CHECK(cnnlSetTensorDescriptor(fft_Na_Nr_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, fft_dims_1));
    CNNL_CHECK(cnnlSetTensorDescriptor(fft_Nr_Na_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, fft_dims_2));
    CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(fft_Na_Nr_desc, CNNL_DTYPE_FLOAT));
    CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(fft_Nr_Na_desc, CNNL_DTYPE_FLOAT));

    // 分配MLU内存
    void *d_ifft,*d_ifft_transposed;
    size_t ifft_size = Na * Nr * 2*sizeof(float);
    CNRT_CHECK(cnrtMalloc(&d_ifft, ifft_size));
    CNRT_CHECK(cnrtMalloc(&d_ifft_transposed, ifft_size));

    // 配置距离向IFFT
    size_t ifft_range_workspace_size = 0;
    void *ifft_range_workspace = nullptr;
    size_t ifft_range_reservespace_size = 0;
    void *ifft_range_reservespace = nullptr;

    cnnlFFTPlan_t ifft_range_desc;
    CNNL_CHECK(cnnlCreateFFTPlan(&ifft_range_desc));

    int ifft_n_range[] = {static_cast<int>(Nr)};   
    // 初始化IFFT计划
    CNNL_CHECK(cnnlMakeFFTPlanMany(handle, ifft_range_desc, fft_Na_Nr_desc, fft_Na_Nr_desc, 
                                rank, ifft_n_range, &ifft_range_reservespace_size, 
                                &ifft_range_workspace_size));

    // 分配工作空间
    if (ifft_range_workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&ifft_range_workspace, ifft_range_workspace_size));
    }
    if (ifft_range_reservespace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&ifft_range_reservespace, ifft_range_reservespace_size));
        CNNL_CHECK(cnnlSetFFTReserveArea(handle, ifft_range_desc, ifft_range_reservespace));
    }

    // 执行距离向IFFT，direction=1表示IFFT
    float range_scale = 1.0f / Nr;  // IFFT需要除以N
    cnrtNotifier_t start = nullptr, end = nullptr;
    cnrtNotifierCreate(&start);
    cnrtNotifierCreate(&end);
    float hardware_time;
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(start, queue);
    }
    CNNL_CHECK(cnnlExecFFT(handle, ifft_range_desc, d_data, range_scale, 
                        ifft_range_workspace, d_ifft, 1));
    CNRT_CHECK(cnrtQueueSync(queue));
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "距离向IFFT执行时间: " << hardware_time / 1000 << " ms" << endl;
    }                    
    // 设置转置描述符
    cnnlTransposeDescriptor_t ifft_trans_desc;  
    CNNL_CHECK(cnnlCreateTransposeDescriptor(&ifft_trans_desc));
    int ifft_perm[] = {1, 0};   
    CNNL_CHECK(cnnlSetTransposeDescriptor(ifft_trans_desc, 2, ifft_perm));

    // 获取转置工作空间
    size_t ifft_transpose_workspace_size = 0;   
    CNNL_CHECK(cnnlGetTransposeWorkspaceSize(handle, fft_Na_Nr_desc, ifft_trans_desc, &ifft_transpose_workspace_size));

    void* ifft_transpose_workspace = nullptr;   
    if (ifft_transpose_workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&ifft_transpose_workspace, ifft_transpose_workspace_size));
    }

    // 执行转置
    CNNL_CHECK(cnnlTranspose_v2(handle, ifft_trans_desc, fft_Na_Nr_desc, d_ifft,
                            fft_Nr_Na_desc, d_ifft_transposed,
                            ifft_transpose_workspace, ifft_transpose_workspace_size));
    CNRT_CHECK(cnrtQueueSync(queue));
    // 配置方位向IFFT
    size_t ifft_azimuth_workspace_size = 0;
    void *ifft_azimuth_workspace = nullptr;
    size_t ifft_azimuth_reservespace_size = 0;
    void *ifft_azimuth_reservespace = nullptr;

    cnnlFFTPlan_t ifft_azimuth_desc;
    CNNL_CHECK(cnnlCreateFFTPlan(&ifft_azimuth_desc));

    int ifft_n_azimuth[] = {static_cast<int>(Na)};   

    // 初始化方位向IFFT计划
    CNNL_CHECK(cnnlMakeFFTPlanMany(handle, ifft_azimuth_desc, fft_Nr_Na_desc, fft_Nr_Na_desc, 
                                rank, ifft_n_azimuth, &ifft_azimuth_reservespace_size, 
                                &ifft_azimuth_workspace_size));

    // 分配工作空间
    if (ifft_azimuth_workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&ifft_azimuth_workspace, ifft_azimuth_workspace_size));
    }
    if (ifft_azimuth_reservespace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&ifft_azimuth_reservespace, ifft_azimuth_reservespace_size));
        CNNL_CHECK(cnnlSetFFTReserveArea(handle, ifft_azimuth_desc, ifft_azimuth_reservespace));
    }

    // 执行方位向IFFT
    float azimuth_scale = 1.0f / Na;  // IFFT需要除以N
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(start, queue);
    }
    CNNL_CHECK(cnnlExecFFT(handle, ifft_azimuth_desc, d_ifft_transposed, azimuth_scale, 
                        ifft_azimuth_workspace, d_ifft, 1));
    CNRT_CHECK(cnrtQueueSync(queue));
    if(G_DEBUG == true)
    {
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "方位向IFFT执行时间: " << hardware_time / 1000 << " ms" << endl;
    }    
    // 最后的转置
    CNNL_CHECK(cnnlTranspose_v2(handle, ifft_trans_desc, fft_Nr_Na_desc, d_ifft,
                            fft_Na_Nr_desc, d_data,
                            ifft_transpose_workspace, ifft_transpose_workspace_size));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 清理资源
    CNRT_CHECK(cnrtFree(d_ifft));
    CNRT_CHECK(cnrtFree(d_ifft_transposed));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(fft_Na_Nr_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(fft_Nr_Na_desc));
    CNNL_CHECK(cnnlDestroyTransposeDescriptor(ifft_trans_desc));
    CNNL_CHECK(cnnlDestroyFFTPlan(ifft_range_desc));
    CNNL_CHECK(cnnlDestroyFFTPlan(ifft_azimuth_desc));

    if (ifft_range_workspace) {
        CNRT_CHECK(cnrtFree(ifft_range_workspace));
        CNRT_CHECK(cnrtFree(ifft_range_reservespace));
    }
    if (ifft_azimuth_workspace) {
        CNRT_CHECK(cnrtFree(ifft_azimuth_workspace));
        CNRT_CHECK(cnrtFree(ifft_azimuth_reservespace));
    }
    if (ifft_transpose_workspace) {
        CNRT_CHECK(cnrtFree(ifft_transpose_workspace));
    }
};

            