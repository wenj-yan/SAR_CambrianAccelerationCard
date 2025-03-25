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


#include "complex.h"

void SplitComplexTensor(cnnlHandle_t handle,
                       cnrtQueue_t queue,
                       const void* complex_data, // [Na, Nr] - 复数数据
                       int Na,
                       int Nr,
                       void* d_real,  // [Na, Nr] - 实部设备指针
                       void* d_imag)  // [Na, Nr] - 虚部设备指针
{
    // 创建计时器
    cnrtNotifier_t start = nullptr, end = nullptr;
    CNRT_CHECK(cnrtNotifierCreate(&start));
    CNRT_CHECK(cnrtNotifierCreate(&end));
    
    // 放置开始计时点
    CNRT_CHECK(cnrtPlaceNotifier(start, queue));

    // 1. 创建输入tensor描述符
    cnnlTensorDescriptor_t input_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&input_desc));
    int input_dims[] = {Na * Nr, 2}; 
    CNNL_CHECK(cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, input_dims));

    // 2. 创建输出tensor描述符
    cnnlTensorDescriptor_t output_desc[2];
    CNNL_CHECK(cnnlCreateTensorDescriptor(&output_desc[0]));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&output_desc[1]));
    
    // 输出维度需要与输入维度相同
    int output_dims[] = {Na * Nr, 1};  // 保持二维，但第二维为1
    CNNL_CHECK(cnnlSetTensorDescriptor(output_desc[0], CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, output_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(output_desc[1], CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, output_dims));

    // 5. 获取工作空间大小
    size_t workspace_size = 0;
    CNNL_CHECK(cnnlGetSplitWorkspaceSize(handle, 2, &workspace_size));
    
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&workspace, workspace_size));
    }

    // 6. 准备输出数组
    void* outputs[] = {d_real, d_imag};

    // 7. 执行split操作 - 在最后一维(axis=1)上分离
    CNNL_CHECK(cnnlSplit(handle, 
                        2,           // split_num = 2 (实部和虚部)
                        1,           // axis = 1
                        input_desc, 
                        complex_data,
                        workspace,
                        workspace_size,
                        output_desc,
                        outputs));

    // 放置结束计时点
    CNRT_CHECK(cnrtPlaceNotifier(end, queue));
    CNRT_CHECK(cnrtQueueSync(queue));
    
    // 计算执行时间
    float hardware_time;
    CNRT_CHECK(cnrtNotifierDuration(start, end, &hardware_time));
    cout << "SplitComplexTensor 执行时间: " << hardware_time / 1000 << " ms" << endl;

    // 10. 清理计时器
    CNRT_CHECK(cnrtNotifierDestroy(start));
    CNRT_CHECK(cnrtNotifierDestroy(end));

    // 10. 清理资源
    CNNL_CHECK(cnnlDestroyTensorDescriptor(input_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc[0]));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc[1]));
    
    if (workspace) {
        CNRT_CHECK(cnrtFree(workspace));
    }
}
void CombineComplexTensor(cnnlHandle_t handle,
                         cnrtQueue_t queue,
                         void* d_real,        // 实部数据指针 [Na,Nr,1]
                         void* d_imag,        // 虚部数据指针 [Na,Nr,1] 
                         int Na,              // 输入维度
                         int Nr,              // 输入维度
                         void* d_output)      // 输出数据指针 [Na,Nr,2]
{
    cout << "开始执行CombineComplexTensor" << endl;
    // 创建计时器
    cnrtNotifier_t start = nullptr, end = nullptr;
    CNRT_CHECK(cnrtNotifierCreate(&start));
    CNRT_CHECK(cnrtNotifierCreate(&end));
    
    // 放置开始计时点
    CNRT_CHECK(cnrtPlaceNotifier(start, queue));

    // 1. 创建输入tensor描述符 [Na,Nr,1]
    cnnlTensorDescriptor_t real_desc, imag_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&real_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&imag_desc));
    int input_dims[] = {Na, Nr, 1};
    CNNL_CHECK(cnnlSetTensorDescriptor(real_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, input_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(imag_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, input_dims));

    // 2. 创建输出tensor描述符 [Na,Nr,2]
    cnnlTensorDescriptor_t output_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&output_desc));
    int output_dims[] = {Na, Nr, 2};
    CNNL_CHECK(cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, output_dims));

    // 3. 准备输入描述符数组和指针数组
    cnnlTensorDescriptor_t input_descs[] = {real_desc, imag_desc};
    const void* inputs[] = {d_real, d_imag};

    // 4. 查询所需workspace大小
    size_t workspace_size = 0;
    CNNL_CHECK(cnnlGetConcatWorkspaceSize(handle, 2, &workspace_size));
    
    // 5. 分配workspace内存
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&workspace, workspace_size));
    }

    // 6. 执行concat操作 - 在最后一维(axis=2)上拼接
    CNNL_CHECK(cnnlConcat(handle,
                         2,            // 输入数量
                         2,            // axis = 2 (最后一维)
                         input_descs,  // 输入描述符数组
                         inputs,       // 输入数据指针数组
                         workspace,
                         workspace_size,
                         output_desc,
                         d_output));

    // 放置结束计时点
    CNRT_CHECK(cnrtPlaceNotifier(end, queue));
    CNRT_CHECK(cnrtQueueSync(queue));
    
    // 计算执行时间
    float hardware_time;
    CNRT_CHECK(cnrtNotifierDuration(start, end, &hardware_time));
    cout << "CombineComplexTensor 执行时间: " << hardware_time / 1000 << " ms" << endl;

    // 清理计时器
    CNRT_CHECK(cnrtNotifierDestroy(start));
    CNRT_CHECK(cnrtNotifierDestroy(end));

    // 清理资源
    CNNL_CHECK(cnnlDestroyTensorDescriptor(real_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(imag_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc));

    if (workspace) {
        CNRT_CHECK(cnrtFree(workspace));
    }
    cout << "CombineComplexTensor 执行完成" << endl;
}
// 实数矩阵乘法函数
cnnlStatus_t MatrixMultiply(cnnlHandle_t handle,
                           const void* A,  // 修改为void*以兼容设备指针
                           const void* B,  // 修改为void*以兼容设备指针
                           void* C,        // 修改为void*以兼容设备指针
                           int M,  int N) {
    try {
        // 创建张量描述符
        cnnlTensorDescriptor_t a_desc, b_desc, c_desc;
        CNNL_CHECK(cnnlCreateTensorDescriptor(&a_desc));
        CNNL_CHECK(cnnlCreateTensorDescriptor(&b_desc));
        CNNL_CHECK(cnnlCreateTensorDescriptor(&c_desc));

        // 设置维度
        int dims[] = {M, N};
        // 设置张量描述符
        CNNL_CHECK(cnnlSetTensorDescriptor(a_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, dims));
        CNNL_CHECK(cnnlSetTensorDescriptor(b_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, dims));
        CNNL_CHECK(cnnlSetTensorDescriptor(c_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, dims));

        // 设置指针数组
        const cnnlTensorDescriptor_t desc_array[] = {a_desc, b_desc};
        const void* data_array[] = {A, B};

        // 执行逐元素相乘
        CNNL_CHECK(cnnlMulN(handle, desc_array, data_array, 2, c_desc, C));

        // 清理资源
        CNNL_CHECK(cnnlDestroyTensorDescriptor(a_desc));
        CNNL_CHECK(cnnlDestroyTensorDescriptor(b_desc));
        CNNL_CHECK(cnnlDestroyTensorDescriptor(c_desc));

        return CNNL_STATUS_SUCCESS;
    }
    catch (const std::exception& e) {
        cerr << "矩阵乘法错误: " << e.what() << endl;
        return CNNL_STATUS_EXECUTION_FAILED;
    }
}

// 在 MatrixMultiply 函数之前添加复数矩阵乘法函数
cnnlStatus_t ComplexMatrixMultiply(cnnlHandle_t handle,
                                  cnrtQueue_t queue,
                                  const void* A,
                                  const void* B,
                                  void* C,
                                  int M,  int N) {
    try {
        // 为实数矩阵分配设备内存
        void *d_A_real, *d_A_imag, *d_B_real, *d_B_imag, *d_C_real, *d_C_imag;
        size_t size = M * N * sizeof(float);
        CNRT_CHECK(cnrtMalloc(&d_A_real, size));
        CNRT_CHECK(cnrtMalloc(&d_A_imag, size));
        CNRT_CHECK(cnrtMalloc(&d_B_real, size));
        CNRT_CHECK(cnrtMalloc(&d_B_imag, size));
        CNRT_CHECK(cnrtMalloc(&d_C_real, size));
        CNRT_CHECK(cnrtMalloc(&d_C_imag, size));

        // 分离实部和虚部
        SplitComplexTensor(handle, queue, A, M, N, d_A_real, d_A_imag);
        SplitComplexTensor(handle, queue, B, M, N, d_B_real, d_B_imag);
        // 临时矩阵用于存储中间结果
        void *d_temp1, *d_temp2;
        size_t temp_size = M * N * sizeof(float);
        CNRT_CHECK(cnrtMalloc(&d_temp1, temp_size));
        CNRT_CHECK(cnrtMalloc(&d_temp2, temp_size));

        // 计算实部：real = A_real * B_real - A_imag * B_imag
        CNNL_CHECK(MatrixMultiply(handle, d_A_real, d_B_real, d_C_real, M,  N));
        CNNL_CHECK(MatrixMultiply(handle, d_A_imag, d_B_imag, d_temp1, M,  N));
        CNRT_CHECK(cnrtQueueSync(queue));
        // 创建输入和输出张量描述符
        cnnlTensorDescriptor_t a_desc, c_desc;
        CNNL_CHECK(cnnlCreateTensorDescriptor(&a_desc));
        CNNL_CHECK(cnnlCreateTensorDescriptor(&c_desc));
        // 设置张量维度
        int dims[] = {M * N};
        CNNL_CHECK(cnnlSetTensorDescriptor(a_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, dims));
        CNNL_CHECK(cnnlSetTensorDescriptor(c_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, dims));

        // 获取AssignSub操作所需workspace大小
        size_t assign_sub_workspace_size = 0;
        CNNL_CHECK(cnnlGetAssignSubWorkspaceSize(handle, a_desc, c_desc, &assign_sub_workspace_size));
        
        // 分配workspace内存
        void* assign_sub_workspace = nullptr;
        CNRT_CHECK(cnrtMalloc(&assign_sub_workspace, assign_sub_workspace_size));

        // 设置alpha和beta参数
        float alpha = 1.0f;
        float beta = 1.0f;

        // 执行AssignSub操作
        CNNL_CHECK(cnnlAssignSub(handle, &alpha, a_desc, d_temp1, assign_sub_workspace, assign_sub_workspace_size,
                               &beta, c_desc, d_C_real));

        // 释放资源
        CNRT_CHECK(cnrtFree(assign_sub_workspace));
        CNNL_CHECK(cnnlDestroyTensorDescriptor(a_desc));
        CNNL_CHECK(cnnlDestroyTensorDescriptor(c_desc));

        // 计算虚部：imag = A_real * B_imag + A_imag * B_real
        CNNL_CHECK(MatrixMultiply(handle, d_A_real, d_B_imag, d_C_imag, M,  N));
        CNNL_CHECK(MatrixMultiply(handle, d_A_imag, d_B_real, d_temp2, M,  N));
        CNRT_CHECK(cnrtQueueSync(queue));
        // 创建输入和输出张量描述符
        cnnlTensorDescriptor_t inputs_desc[2], output_desc;
        CNNL_CHECK(cnnlCreateTensorDescriptor(&inputs_desc[0]));
        CNNL_CHECK(cnnlCreateTensorDescriptor(&inputs_desc[1]));
        CNNL_CHECK(cnnlCreateTensorDescriptor(&output_desc));
        
        // 设置张量描述符
        CNNL_CHECK(cnnlSetTensorDescriptor(inputs_desc[0], CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, dims));
        CNNL_CHECK(cnnlSetTensorDescriptor(inputs_desc[1], CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, dims));
        CNNL_CHECK(cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, dims));

        // 准备输入输出指针
        const void* inputs[2] = {d_C_imag, d_temp2};
        void* output = d_C_imag;

        // 获取所需workspace大小并分配内存
        size_t workspace_size = 0;
        CNNL_CHECK(cnnlGetAddNWorkspaceSize(handle, inputs_desc, 2, output_desc, &workspace_size));
        void* workspace = nullptr;
        CNRT_CHECK(cnrtMalloc(&workspace, workspace_size));

        // 执行AddN操作
        CNNL_CHECK(cnnlAddN_v2(handle,inputs_desc, inputs, 2,output_desc, output, workspace, workspace_size));

        // 释放资源
        CNRT_CHECK(cnrtFree(workspace));
        CNNL_CHECK(cnnlDestroyTensorDescriptor(inputs_desc[0]));
        CNNL_CHECK(cnnlDestroyTensorDescriptor(inputs_desc[1]));
        CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc));
        // 组合实部和虚部
        CombineComplexTensor(handle, queue, d_C_real, d_C_imag, M, N, C);

        // 清理资源
        CNRT_CHECK(cnrtFree(d_A_real));
        CNRT_CHECK(cnrtFree(d_A_imag));
        CNRT_CHECK(cnrtFree(d_B_real));
        CNRT_CHECK(cnrtFree(d_B_imag));
        CNRT_CHECK(cnrtFree(d_C_real));
        CNRT_CHECK(cnrtFree(d_C_imag));
        CNRT_CHECK(cnrtFree(d_temp1));
        CNRT_CHECK(cnrtFree(d_temp2));

        
    return CNNL_STATUS_SUCCESS;
    }
    catch (const std::exception& e) {
        cerr << "复数矩阵乘法错误: " << e.what() << endl;
        return CNNL_STATUS_EXECUTION_FAILED;
    }
    cout << "ComplexMatrixMultiply 执行完成" << endl;
}

void PadComplexTensor(cnnlHandle_t handle,
                     cnrtQueue_t queue,
                     void* d_input,        // 设备端输入数据指针 [Na,Nr,2]
                     int Na,               // 输入维度
                     int Nr,               // 输入维度  
                     int pad_before_Na,    // Na维度前填充
                     int pad_after_Na,     // Na维度后填充
                     int pad_before_Nr,    // Nr维度前填充
                     int pad_after_Nr,     // Nr维度后填充
                     complex<float> padding_value, // 填充值
                     void* d_output)       // 设备端输出数据指针
{
    // 创建计时器
    cnrtNotifier_t start = nullptr, end = nullptr;
    CNRT_CHECK(cnrtNotifierCreate(&start));
    CNRT_CHECK(cnrtNotifierCreate(&end));
    // 放置开始计时点
    CNRT_CHECK(cnrtPlaceNotifier(start, queue));
    // 1. 创建输入tensor描述符 [Na,Nr,2]
    cnnlTensorDescriptor_t input_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&input_desc));
    int input_dims[] = {Na, Nr, 2}; 
    CNNL_CHECK(cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, input_dims));
    // 2. 创建输出tensor描述符 [Na+pad,Nr+pad,2] 
    cnnlTensorDescriptor_t output_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&output_desc));
    int output_dims[] = {Na + pad_before_Na + pad_after_Na, 
                        Nr + pad_before_Nr + pad_after_Nr,
                        2};
    CNNL_CHECK(cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, output_dims));
    // 3. 准备padding参数
    int32_t paddings[] = {
        pad_before_Na, pad_after_Na,  // Na维度的padding
        pad_before_Nr, pad_after_Nr,  // Nr维度的padding
        0, 0                          // 复数维度不需要padding
    };

    // 4. 准备padding值 - 转换complex<float>为float[2]
    float padding_value_arr[2] = {padding_value.real(), padding_value.imag()};

    // 5. 执行pad操作
    CNNL_CHECK(cnnlPad(handle,
                       input_desc,
                       d_input, 
                       paddings,
                       padding_value_arr,
                       output_desc,
                       d_output));

    // 放置结束计时点
    CNRT_CHECK(cnrtPlaceNotifier(end, queue));
    CNRT_CHECK(cnrtQueueSync(queue));
    
    // 计算执行时间
    float hardware_time;
    CNRT_CHECK(cnrtNotifierDuration(start, end, &hardware_time));
    cout << "PadComplexTensor 执行时间: " << hardware_time / 1000 << " ms" << endl;

    // 清理计时器
    CNRT_CHECK(cnrtNotifierDestroy(start));
    CNRT_CHECK(cnrtNotifierDestroy(end));

    // 清理资源
    CNNL_CHECK(cnnlDestroyTensorDescriptor(input_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc));
}