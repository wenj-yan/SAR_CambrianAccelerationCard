/**
 * @file wk_V4.cc
 * @brief SAR-WK成像算法实现(基于MLU平台)
 * @author wenjyan
 * @email wenjyan@outlook.com
 * @version 4.0
 * @date 2025-01-13
 *
 * @note 更新内容:
 * - 2025-01-02 v1.0
 *   - 初始版本
 *   - 实现了基本的无斜视角SAR成像算法
 *   - 支持MLU硬件加速
 *   - 包含距离向压缩和方位向压缩
 *   - 实现了FFT/IFFT等基础操作
 *   - 未实现Stolt插值
 * 
 * -2025-01-06 v2.0
 *   - 实现了CPU上参考函数的生成
 * 
 * -2025-01-08 v3.0
 *   - 实现了Stolt插值
 * 
 * -2025-01-13 v4.0
 *   - 算子优化，padding、split、combine、mul进行优化
 *   - 优化了数据传输，减少了数据传输的次数
 *   - 运算效率变高
 * 
 * @copyright Copyright (c) 2024
 */




#include <time.h>
#include <string>
#include "string.h"
#include "tool.h"
#include <complex>
#include <iostream>
#include <complex.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <algorithm>  // 为min_element和max_element

using namespace std;

// 定义参数结构体
struct WKParams {
    double Fr;    // 距离向采样率
    double PRF;   // 方位向采样率
    double f0;    // 中心频率
    double Tr;    // 脉冲持续时间
    double R0;    // 最近点斜距
    double Kr;    // 线性调频率
    double c;     // 光速
    double Vr;    // 等效雷达速度
    double Ka;    // 方位向调频率
    double f_nc;  // 多普勒中心频率
};

void initDevice(int &dev, cnrtQueue_t &queue, cnnlHandle_t &handle) {
    CNRT_CHECK(cnrtGetDevice(&dev));
    CNRT_CHECK(cnrtSetDevice(dev));
    CNRT_CHECK(cnrtQueueCreate(&queue));
    CNNL_CHECK(cnnlCreate(&handle));
    CNNL_CHECK(cnnlSetQueue(handle, queue));
}

// 性能统计结构体
struct Optensor {
    float hardware_time = 0.0;
    float interface_time = 0.0;
    float end2end_time = 0.0;
    float memcpyH2D_time = 0.0;
    float memcpyD2H_time = 0.0;
};

// 读取复数数据
bool readComplexData(const string& filename, vector<complex<float>>& data) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return false;
    }

    data.clear();
    float real, imag;
    
    // 读取实部和虚部
    while (file >> real >> imag) {
        data.push_back(complex<float>(real, imag));
    }

    cout << "成功读取 " << data.size() << " 个复数" << endl;
    file.close();
    return true;
}
// 读取乘法结束后的数据
bool readComplexData_mul(const string& filename, vector<complex<float>>& data) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return false;
    }

    data.clear();
    float real, imag;
    
    // 读取实部和虚部
    while (file >> real >> imag) {
        data.push_back(complex<float>(real, imag));
    }

    cout << "成功读取 " << data.size() << " 个复数" << endl;
    file.close();
    return true;
}
// 读取参数文件
bool readParams(const string& filename, WKParams& params) {
    cout << "开始读取参数！" << endl;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开参数文件: " << filename << endl;
        return false;
    }

    string line;
    while (getline(file, line)) {
        size_t pos = line.find(':');
        if (pos == string::npos) continue;
        
        string param_name = line.substr(0, pos);
        double value = stod(line.substr(pos + 1));
        
        if (param_name == "Fr") params.Fr = value;
        else if (param_name == "PRF") params.PRF = value;
        else if (param_name == "f0") params.f0 = value;
        else if (param_name == "Tr") params.Tr = value;
        else if (param_name == "R0") params.R0 = value;
        else if (param_name == "Kr") params.Kr = value;
        else if (param_name == "c") params.c = value;
    }

    file.close();

    // 打印所有参数
    cout << "读取到的参数：" << endl;
    cout << "Fr: " << params.Fr << " Hz" << endl;
    cout << "PRF: " << params.PRF << " Hz" << endl;
    cout << "f0: " << params.f0 << " Hz" << endl;
    cout << "Tr: " << params.Tr << " s" << endl;
    cout << "R0: " << params.R0 << " m" << endl;
    cout << "Kr: " << params.Kr << " Hz/s" << endl;
    cout << "c: " << params.c << " m/s" << endl;
    cout << "Vr: " << params.Vr << " m/s" << endl;
    cout << "Ka: " << params.Ka << " Hz/s" << endl;
    cout << "f_nc: " << params.f_nc << " Hz" << endl;

    return true;
}
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

// 首先计算新的频率映射
vector<float> calculateNewFrequencyMapping(const vector<double>& fr_axis, 
                                         const vector<double>& fa_axis,
                                         double f0, double c, double Vr,
                                         int Na, int Nr) {
    vector<float> fr_new_mtx(Na * Nr);
    for (int i = 0; i < Na; ++i) {
        double fa = fa_axis[i];
        for (int j = 0; j < Nr; ++j) {
            // fr_new_mtx = sqrt((f0+fr_axis).^2-c^2/4/Vr^2.*fa_axis.^2)-f0
            double f0_plus_fr = f0 + fr_axis[j];
            double term1 = f0_plus_fr * f0_plus_fr;
            double term2 = (c * c * fa * fa) / (4.0 * Vr * Vr);
            fr_new_mtx[i * Nr + j] = sqrt(term1 - term2) - f0;
        }
    }
    return fr_new_mtx;
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
// Stolt插值函数
void performStoltInterpolation(cnnlHandle_t handle, 
                             cnrtQueue_t queue,
                             const vector<complex<float>>& input,
                             vector<complex<float>>& output,
                             const vector<double>& fr_axis,
                             const vector<double>& fa_axis,
                             double f0, double c, double Vr,
                             int Na, int Nr,double fr) {
    cnrtNotifier_t start = nullptr, end = nullptr;
    
    cout << "\n执行Stolt插值..." << endl;

    // 首先计算新的频率映射
    vector<float> fr_new_mtx = calculateNewFrequencyMapping(fr_axis, fa_axis, f0, c, Vr, Na, Nr);
    
    float fr_new_min = *min_element(fr_new_mtx.begin(), fr_new_mtx.end());
    float fr_new_max = *max_element(fr_new_mtx.begin(), fr_new_mtx.end());
    float fr_min = *min_element(fr_axis.begin(), fr_axis.end());
    float fr_max = *max_element(fr_axis.begin(), fr_axis.end());

    cout << "fr_new_mtx范围: [" << fr_new_min << ", " << fr_new_max << "]" << endl;
    cout << "fr_axis范围: [" << fr_min << ", " << fr_max << "]" << endl;

    // 3. 设置输入参数
    const int batch_size = 1;
    const int channels = 1;
    const int height = Na;
    const int width = Nr;
    const int out_height = Na;
    const int out_width = Nr;

    // 4. 创建描述符
    cnnlGridSampleDescriptor_t grid_sample_desc;
    cnnlTensorDescriptor_t input_desc, grid_desc, output_desc;
    
    CNNL_CHECK(cnnlCreateGridSampleDescriptor(&grid_sample_desc));
    CNNL_CHECK(cnnlSetGridSampleDescriptor(
        grid_sample_desc,
        CNNL_INTERP_BILINEAR,
        CNNL_GRIDSAMPLE_PADDING_ZEROS,
        true  // align_corners
    ));

    // 5. 设置tensor描述符 - NHWC布局
    int input_dims[] = {batch_size, height, width, channels};  // NHWC
    int grid_dims[] = {batch_size, out_height, out_width, 2}; // NHWC 
    int output_dims[] = {batch_size, out_height, out_width, channels}; // NHWC

    CNNL_CHECK(cnnlCreateTensorDescriptor(&input_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&grid_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&output_desc));

    CNNL_CHECK(cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, input_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(grid_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, grid_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, output_dims));

    // 获取workspace大小
    size_t workspace_size = 0;
    CNNL_CHECK(cnnlGetGridSampleForwardWorkspaceSize(
        handle,
        input_desc,
        grid_desc, 
        output_desc,
        &workspace_size));
    // 分配workspace内存
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&workspace, workspace_size));
    }

    // 6. 准备输入数据 - 分别处理实部和虚部
    std::vector<float> host_input_real(batch_size * height * width * channels, 0.0f);
    std::vector<float> host_input_imag(batch_size * height * width * channels, 0.0f);
    std::vector<float> host_grid(batch_size * out_height * out_width * 2, 0.0f);
    std::vector<float> host_output_real(batch_size * out_height * out_width * channels, 0.0f);
    std::vector<float> host_output_imag(batch_size * out_height * out_width * channels, 0.0f);

    // 创建临时向量存储平移后的数据
    std::vector<float> shifted_input_real(batch_size * height * width * channels, 0.0f);
    std::vector<float> shifted_input_imag(batch_size * height * width * channels, 0.0f);
    
    
    //数据平移
    int mid_point = width / 2;
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            //循环移位
            int src_w = (w + width - mid_point) % width;
            int dst_idx = h * width + w;
            shifted_input_real[dst_idx] = input[h * width + src_w].real();
            shifted_input_imag[dst_idx] = input[h * width + src_w].imag();
        }
    }

    // 打印CPU平移结果的5x5区域
    std::cout << "\nCPU平移结果的5x5区域:" << std::endl;
    for (int h = 0; h < std::min(5, height); h++) {
        for (int w = 0; w < std::min(5, width); w++) {
            int idx = h * width + w;
            std::cout << "(" << shifted_input_real[idx] << "," << shifted_input_imag[idx] << ") ";
        }
        std::cout << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // // 使用CNNL进行移位操作
    // cnnlTensorDescriptor_t roll_desc;
    // CNNL_CHECK(cnnlCreateTensorDescriptor(&roll_desc));
    // int dims[4] = {batch_size, height, width, channels};
    // CNNL_CHECK(cnnlSetTensorDescriptor(roll_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_COMPLEX_FLOAT, 4, dims));
    // // 获取workspace大小
    // size_t roll_workspace_size = 0;
    // CNNL_CHECK(cnnlGetRollWorkspaceSize(handle, roll_desc, &roll_workspace_size));
    // void* roll_workspace = nullptr;
    // if (roll_workspace_size > 0) {
    //     CNRT_CHECK(cnrtMalloc(&roll_workspace, roll_workspace_size));
    // }
    // // 分配设备内存
    // void *d_input, *d_output;
    // size_t array_size = batch_size * height * width * channels * sizeof(complex<float>);
    // CNRT_CHECK(cnrtMalloc(&d_input, array_size));
    // CNRT_CHECK(cnrtMalloc(&d_output, array_size));
    // // 拷贝数据到设备
    // CNRT_CHECK(cnrtMemcpy(d_input, (void*)input.data(), array_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    // // 执行roll操作
    // int shifts[2] = {0, width/2};  // 垂直和水平方向的移动量
    // int axes[2] = {1, 2};  // 对应的轴
    // CNNL_CHECK(cnnlRoll(handle, roll_desc, d_input, 
    //                     shifts, 2, axes, 2,
    //                     roll_workspace, roll_workspace_size,
    //                     roll_desc, d_output));
    // // 拷贝结果回主机
    // std::vector<complex<float>> roll_result(batch_size * height * width * channels);
    // CNRT_CHECK(cnrtMemcpy(roll_result.data(), d_output, array_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    // // 打印CNNL平移结果的5x5区域
    // std::cout << "\nCNNL平移结果的5x5区域:" << std::endl;
    // for (int h = 0; h < std::min(5, height); h++) {
    //     for (int w = 0; w < std::min(5, width); w++) {
    //         int idx = h * width + w;
    //         std::cout << "(" << roll_result[idx].real() << "," << roll_result[idx].imag() << ") ";
    //     }
    //     std::cout << std::endl;
    // }




    // // 清理内存
    // if (roll_workspace) {
    //     CNRT_CHECK(cnrtFree(roll_workspace));
    // }
    // CNRT_CHECK(cnrtFree(d_input));
    // CNRT_CHECK(cnrtFree(d_output));
    // CNNL_CHECK(cnnlDestroyTensorDescriptor(roll_desc));
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    host_input_real = shifted_input_real;
    host_input_imag = shifted_input_imag;
    //     // 生成网格数据 
    for (int h = 0; h < out_height; h++) {
        for (int w = 0; w < out_width; w++) {
            int idx = h * out_width * 2 + w * 2;
            float normalized_h = 2.0f * h / (out_height - 1) - 1.0f;
            host_grid[idx] = fr_new_mtx[h * Nr + w]/(fr*Nr*0.5);  // x使用频率映射数据
            host_grid[idx + 1] = normalized_h;  // y保持原位置
        }
    }

    // 检查grid数据范围和统计超出[-1,1]范围的数据比例
    float min_val = host_grid[0];
    float max_val = host_grid[0];
    int out_of_range = 0;
    int total = host_grid.size();
    for(int i = 0; i < total; i++) {
        float val = host_grid[i];
        min_val = min(min_val, val);
        max_val = max(max_val, val); 
        if(val < -1.0f || val > 1.0f) {
            out_of_range++;
        }
    }
    cout << "Grid数据范围: 最小值 = " << min_val << ", 最大值 = " << max_val << endl;
    cout << "超出[-1,1]范围的数据比例: " << (float)out_of_range/total * 100 << "%" << endl;

    // 7. 分配设备内存
    void *dev_input_real, *dev_input_imag, *dev_grid, *dev_output_real, *dev_output_imag;
    size_t input_size = host_input_real.size() * sizeof(float);
    size_t grid_size = host_grid.size() * sizeof(float);
    size_t output_size = host_output_real.size() * sizeof(float);

    CNRT_CHECK(cnrtMalloc(&dev_input_real, input_size));
    CNRT_CHECK(cnrtMalloc(&dev_input_imag, input_size));
    CNRT_CHECK(cnrtMalloc(&dev_grid, grid_size));
    CNRT_CHECK(cnrtMalloc(&dev_output_real, output_size));
    CNRT_CHECK(cnrtMalloc(&dev_output_imag, output_size));

    // 8. 拷贝数据到设备
    CNRT_CHECK(cnrtMemcpy(dev_input_real, host_input_real.data(), input_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(dev_input_imag, host_input_imag.data(), input_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(dev_grid, host_grid.data(), grid_size, CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 9. 分别执行实部和虚部的GridSampleForward
    // 处理实部
    cnrtNotifierCreate(&start);
    cnrtNotifierCreate(&end);
    cnrtPlaceNotifier(start, queue);
    CNNL_CHECK(cnnlGridSampleForward(
        handle,
        grid_sample_desc,
        input_desc,
        dev_input_real,
        grid_desc,
        dev_grid,
        output_desc,
        dev_output_real,
        workspace,
        workspace_size
    ));
    CNRT_CHECK(cnrtQueueSync(queue));
    // 处理虚部
    CNNL_CHECK(cnnlGridSampleForward(
        handle,
        grid_sample_desc,
        input_desc,
        dev_input_imag,
        grid_desc,
        dev_grid,
        output_desc,
        dev_output_imag,
        workspace,
        workspace_size
    ));
    cnrtPlaceNotifier(end, queue);
    CNRT_CHECK(cnrtQueueSync(queue));

    float hardware_time;        
    cnrtNotifierDuration(start, end, &hardware_time);
    cout << "Stolt插值执行时间: " << hardware_time / 1000 << " ms" << endl;

    // 10. 拷贝结果回主机
    CNRT_CHECK(cnrtMemcpy(host_output_real.data(), dev_output_real, output_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(host_output_imag.data(), dev_output_imag, output_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    // 组合实部和虚部
    output.resize(Na * Nr);
    for (int h = 0; h < Na; h++) {
        for (int w = 0; w < Nr; w++) {
            int idx = h * Nr  + w ;
            output[idx] = complex<float>(host_output_real[idx], host_output_imag[idx]);
        }
    }

    cout << "\nStolt插值后的结果前5x5个值：" << endl;
    for (int i = 0; i < min(5, (int)Na); ++i) {
        for (int j = 0; j < min(5, (int)Nr); ++j) {
            complex<float> val = output[i * Nr + j];
            printf("(%.3e,%.3e) ", val.real(), val.imag());
        }
        cout << endl;
    }

    // 12. 清理资源
    CNNL_CHECK(cnnlDestroyTensorDescriptor(input_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(grid_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc));
    CNNL_CHECK(cnnlDestroyGridSampleDescriptor(grid_sample_desc));

    CNRT_CHECK(cnrtFree(dev_input_real));
    CNRT_CHECK(cnrtFree(dev_input_imag));
    CNRT_CHECK(cnrtFree(dev_grid));
    CNRT_CHECK(cnrtFree(dev_output_real));
    CNRT_CHECK(cnrtFree(dev_output_imag));

    if (workspace != nullptr) {
        CNRT_CHECK(cnrtFree(workspace));
    }

    cout << "Stolt插值完成" << endl;
}

void generateReferenceFunction_CPU(cnnlHandle_t handle,
                             cnrtQueue_t queue,
                             vector<complex<float>>& theta_ft_fa,
                             const vector<double>& fr_axis,
                             const vector<double>& fa_axis,
                             double f0, double c, double Vr, double Kr, double R0,
                             int Na, int Nr) {
    // 初始化输出向量
    theta_ft_fa.resize(Na * Nr);

    // 计算常量项
    double bb = (c * c) / (4.0 * Vr * Vr);
    double d1 = M_PI / Kr * (-1.0);
    double scale = 4 * M_PI * R0 / c;

    // 对每个点计算参考函数
    for (int i = 0; i < Na; ++i) {
        double fa = fa_axis[i];  // 当前fa值
        double fa_term = bb * fa * fa;  // (c^2/4/Vr^2) * fa^2

        for (int j = 0; j < Nr; ++j) {
            double fr = fr_axis[j];  // 当前fr值
            double f_sum = f0 + fr;  // f0 + fr
            
            // 计算第一项: 4*pi*R0/c * sqrt((f0+fr)^2 - c^2/4/Vr^2 * fa^2)
            double under_sqrt = f_sum * f_sum - fa_term;
            if(under_sqrt < 0) {
                // 处理负数情况,设为0避免nan
                under_sqrt = 0;
            }
            double first_term = scale * sqrt(under_sqrt);
            
            // 计算第二项: pi/Kr * fr^2
            double second_term = d1 * fr * fr;
            
            // 计算总相位
            double theta = first_term + second_term;
            
            // 使用欧拉公式计算复指数
            theta_ft_fa[i * Nr + j] = complex<float>(cos(theta), sin(theta));
        }
    }

    // 打印前5x5个点的值以及相位值进行验证
    cout << "\n参考函数前5x5个点的值及其相位：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            size_t idx = i * Nr + j;
            complex<float> val = theta_ft_fa[idx];
            double phase = atan2(val.imag(), val.real());
            cout << "值:(" << val.real() << "," << val.imag() << ") 相位:" << phase << " ";
        }
        cout << endl;



           
    }

    //  //保存结果到文件
    //         cout << "\n保存结果到cankao.txt..." << endl;
    //         ofstream outfile("cankao.txt");
    //         if (!outfile) {
    //             cerr << "无法创建输出文件" << endl;
    //             return;
    //         }

    //         // 设置输出精度
    //         outfile << scientific;  // 使用科学计数法
    //         outfile.precision(15);   // 设置精度为6位

    //         // 写入数据
    //         for (int i = 0; i < Na; ++i) {
    //             for (int j = 0; j < Nr; ++j) {
    //                 complex<float> val = theta_ft_fa[i * Nr + j];
    //                 outfile << val.real() << " " << val.imag();
    //                 if (j < Nr - 1) {
    //                     outfile << " ";  // 在每行的数之间添加空格
    //                 }
    //             }
    //             outfile << "\n";  // 每行结束添加换行
    //         }

    //         outfile.close();
    //         cout << "cankao.txt" << endl;
}

void process_wk() {
    try {
    //////////////////////////////////////////////////////
    // 初始化设备与参数读取
    /////////////////////////////////////////////////////
        // 初始化设备
        int dev;
        cnrtQueue_t queue = nullptr;
        cnnlHandle_t handle = nullptr;
        initDevice(dev, queue, handle);

        // 创建性能计时
        cnrtNotifier_t e_t0, e_t1;
        cnrtNotifierCreate(&e_t0);
        cnrtNotifierCreate(&e_t1);
        Optensor optensor;

        // 读取参数
        WKParams params = {
            .Vr = 7062,   // 等效雷达速度
            .Ka = 1733,   // 方位向调频率
            .f_nc = -6900 // 多普勒中心频率
        };
        if (!readParams("data/paras.txt", params)) {
            cerr << "读取参数失败" << endl;
            return;
        }
        // 读取回波数据
        vector<complex<float>> echo;
        cout << "开始读取数据文件..." << endl;
        if (!readComplexData("data/echo.txt", echo)) {
            cerr << "读取回波数据失败" << endl;
            return;
        }
        // 设置回波数据维度
        size_t Na = 3072;  // 方位向采样点数
        size_t Nr = 2048;  // 距离向采样点数

        // 检查数据大小是否正确
        if (echo.size() != Na * Nr) {
            cerr << "数据大小不正确，期望: " << Na * Nr << ", 实际: " << echo.size() << endl;
            return;
        }

        // 计算填充参数
        int pad_before_Na = round(Na/6.0f);
        int pad_after_Na = round(Na/6.0f);
        int pad_before_Nr = round(Nr/3.0f); 
        int pad_after_Nr = round(Nr/3.0f);
        size_t Na_padded = Na + pad_before_Na + pad_after_Na;
        size_t Nr_padded = Nr + pad_before_Nr + pad_after_Nr;

        cout << "填充后维度: Na_padded = " << Na_padded << ", Nr_padded = " << Nr_padded << endl;
        cout << "原始维度: Na = " << Na << ", Nr = " << Nr << endl;
        cout << "填充量: Na_pad = " << pad_before_Na + pad_after_Na 
             << ", Nr_pad = " << pad_before_Nr + pad_after_Nr << endl;

        // 准备设备内存
        void *d_input, *d_output2;
        size_t input_size = Na * Nr * 2 * sizeof(float);
        size_t output_size = Na_padded * Nr_padded * 2 * sizeof(float);
        
        CNRT_CHECK(cnrtMalloc(&d_input, input_size));
        CNRT_CHECK(cnrtMalloc(&d_output2, output_size));

        // 拷贝输入数据到设备
        CNRT_CHECK(cnrtMemcpy(d_input, echo.data(), input_size, CNRT_MEM_TRANS_DIR_HOST2DEV));

        // 执行pad操作
        PadComplexTensor(handle, queue, d_input, Na, Nr,
                        pad_before_Na, pad_after_Na,
                        pad_before_Nr, pad_after_Nr,
                        complex<float>(0.0f, 0.0f), d_output2);

        // 分配主机内存接收结果
        vector<complex<float>> echo_padded(Na_padded * Nr_padded);

        // 拷贝结果回主机
        CNRT_CHECK(cnrtMemcpy(echo_padded.data(), d_output2, output_size, CNRT_MEM_TRANS_DIR_DEV2HOST));

        // 释放设备内存
        CNRT_CHECK(cnrtFree(d_input));
        CNRT_CHECK(cnrtFree(d_output2));


        
    //////////////////////////////////////////////////////
    // 方位向变频（频谱搬移）
    /////////////////////////////////////////////////////
        // 生成方位向时间轴
        float Fa = params.PRF;  // 方位向采样率等于PRF
        vector<float> ta_axis(Na_padded);
        float ta_step = 1.0f / Fa;
        for (size_t i = 0; i < Na_padded; ++i) {
            ta_axis[i] = (static_cast<float>(i) - Na_padded/2) * ta_step;
        }

        // 执行方位向下变频
        float f_nc = params.f_nc;  // 多普勒中心频率
        for (size_t i = 0; i < Na_padded; ++i) {
            float phase = -2 * M_PI * f_nc * ta_axis[i];
            complex<float> exp_factor(cos(phase), sin(phase));
            for (size_t j = 0; j < Nr_padded; ++j) {
                echo_padded[i * Nr_padded + j] *= exp_factor;
            }
        }

        // 更新维度变量，使用填充后��维度
        Na = Na_padded;
        Nr = Nr_padded;
        echo = std::move(echo_padded);
    //////////////////////////////////////////////////////
    // 数据预处理结束
    // 以下是FFT部分
    /////////////////////////////////////////////////////
            // 分配MLU内存并拷贝数据
            void* d_echo;
            void* d_output;  // FFT输出
            size_t echo_size = Na * Nr * sizeof(complex<float>);
            cout << "准备分配MLU内存，大小: " << echo_size << " 字节" << endl;
            
            CNRT_CHECK(cnrtMalloc((void **)&d_echo, echo_size));
            CNRT_CHECK(cnrtMalloc((void **)&d_output, echo_size));
            
            HostTimer copyin_timer;
            copyin_timer.start();
            CNRT_CHECK(cnrtMemcpy(d_echo, echo.data(), echo_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
            copyin_timer.stop();
            optensor.memcpyH2D_time = copyin_timer.tv_usec;

            // 创建张量描述符
            cnnlTensorDescriptor_t input_desc, output_desc;
            CNNL_CHECK(cnnlCreateTensorDescriptor(&input_desc));
            CNNL_CHECK(cnnlCreateTensorDescriptor(&output_desc));
            
            // FFT公共参数
            int rank = 1;  
            
            // 先进行距离向FFT
            int input_dims[] = {static_cast<int>(Na), static_cast<int>(Nr)};
            CNNL_CHECK(cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, input_dims));
            CNNL_CHECK(cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, input_dims));
            
            CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(input_desc, CNNL_DTYPE_FLOAT));
            CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(output_desc, CNNL_DTYPE_FLOAT));

            // 配置距离向FFT
            size_t fft_range_workspace_size = 0;
            void *fft_range_workspace = nullptr;
            size_t fft_range_reservespace_size = 0;
            void *fft_range_reservespace = nullptr;
            
            cnnlFFTPlan_t fft_range_desc;
            CNNL_CHECK(cnnlCreateFFTPlan(&fft_range_desc));
            
            // 设置FFT参数
            int n_range[] = {static_cast<int>(Nr)};  // FFT长度
            
            // 初始化FFT计划
            CNNL_CHECK(cnnlMakeFFTPlanMany(handle, fft_range_desc, input_desc, output_desc, 
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

            // 执行距离向FFT，添加缩放因子 1.0
            cnrtNotifier_t start = nullptr, end = nullptr;
            cnrtNotifierCreate(&start);
            cnrtNotifierCreate(&end);
            cnrtPlaceNotifier(start, queue);

            CNNL_CHECK(cnnlExecFFT(handle, fft_range_desc, d_echo, 1.0, 
                                  fft_range_workspace, d_output, 0));

            cnrtPlaceNotifier(end, queue);
            CNRT_CHECK(cnrtQueueSync(queue));
            
            float hardware_time;
            cnrtNotifierDuration(start, end, &hardware_time);
            cout << "距离向FFT执行时间: " << hardware_time / 1000 << " ms" << endl;

            // 转置数据 - 第一次转置
            void* d_transposed;
            CNRT_CHECK(cnrtMalloc((void **)&d_transposed, echo_size));
            
            // 创建转置后的张量描述符
            cnnlTensorDescriptor_t transposed_desc;
            CNNL_CHECK(cnnlCreateTensorDescriptor(&transposed_desc));
            int transposed_dims[] = {static_cast<int>(Nr), static_cast<int>(Na)};  // 交换维度
            CNNL_CHECK(cnnlSetTensorDescriptor(transposed_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, transposed_dims));
            CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(transposed_desc, CNNL_DTYPE_FLOAT));
            
            // 设置转置描述符
            cnnlTransposeDescriptor_t trans_desc;
            CNNL_CHECK(cnnlCreateTransposeDescriptor(&trans_desc));
            int perm[] = {1, 0};  // 交换维度
            CNNL_CHECK(cnnlSetTransposeDescriptor(trans_desc, 2, perm));

            // 获取工作空间大小
            size_t transpose_workspace_size = 0;
            CNNL_CHECK(cnnlGetTransposeWorkspaceSize(handle, input_desc, trans_desc, &transpose_workspace_size));
            
            // 分配工作空间
            void* transpose_workspace = nullptr;
            if (transpose_workspace_size > 0) {
                CNRT_CHECK(cnrtMalloc(&transpose_workspace, transpose_workspace_size));
            }

            // 执行第一次转置：[Na, Nr] -> [Nr, Na]
            cnrtNotifierCreate(&start);
            cnrtNotifierCreate(&end);
            cnrtPlaceNotifier(start, queue);
            CNNL_CHECK(cnnlTranspose_v2(handle, trans_desc, input_desc, d_output,
                                       transposed_desc, d_transposed,
                                       transpose_workspace, transpose_workspace_size));
            cnrtPlaceNotifier(end, queue);
            CNRT_CHECK(cnrtQueueSync(queue));
            
            cnrtNotifierDuration(start, end, &hardware_time);
            cout << "第一次转置执行时间: " << hardware_time / 1000 << " ms" << endl;
            
            
            // 配置方位向FFT
            size_t fft_azimuth_workspace_size = 0;
            void *fft_azimuth_workspace = nullptr;
            size_t fft_azimuth_reservespace_size = 0;
            void *fft_azimuth_reservespace = nullptr;
            
            cnnlFFTPlan_t fft_azimuth_desc;
            CNNL_CHECK(cnnlCreateFFTPlan(&fft_azimuth_desc));
            
            // 使用转置后的维度创建新的描述符
            cnnlTensorDescriptor_t azimuth_input_desc, azimuth_output_desc;
            CNNL_CHECK(cnnlCreateTensorDescriptor(&azimuth_input_desc));
            CNNL_CHECK(cnnlCreateTensorDescriptor(&azimuth_output_desc));
            
            int azimuth_dims[] = {static_cast<int>(Nr), static_cast<int>(Na)};  // 使用转置后的维度
            CNNL_CHECK(cnnlSetTensorDescriptor(azimuth_input_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, azimuth_dims));
            CNNL_CHECK(cnnlSetTensorDescriptor(azimuth_output_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, azimuth_dims));
            
            CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(azimuth_input_desc, CNNL_DTYPE_FLOAT));
            CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(azimuth_output_desc, CNNL_DTYPE_FLOAT));
            
            int n_azimuth[] = {static_cast<int>(Na)};  // 方位向FFT长度
            
            // 初始化FFT计划，使用新的描述符
            CNNL_CHECK(cnnlMakeFFTPlanMany(handle, fft_azimuth_desc, azimuth_input_desc, azimuth_output_desc, 
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
            cnrtNotifierCreate(&start);
            cnrtNotifierCreate(&end);
            cnrtPlaceNotifier(start, queue);

            CNNL_CHECK(cnnlExecFFT(handle, fft_azimuth_desc, d_transposed, 1.0, 
                                  fft_azimuth_workspace, d_echo, 0));

            cnrtPlaceNotifier(end, queue);
            CNRT_CHECK(cnrtQueueSync(queue));
            
            cnrtNotifierDuration(start, end, &hardware_time);
            cout << "方位向FFT执行时间: " << hardware_time / 1000 << " ms" << endl;

            // 最后再次转置回原始维度：[Nr, Na] -> [Na, Nr]
            // 注意：这里我们需要从d_echo转置到d_output，并且使用相反的维度顺序
            cnnlTensorDescriptor_t final_transposed_desc;
            CNNL_CHECK(cnnlCreateTensorDescriptor(&final_transposed_desc));
            int final_dims[] = {static_cast<int>(Na), static_cast<int>(Nr)};  // 最终维度
            CNNL_CHECK(cnnlSetTensorDescriptor(final_transposed_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, final_dims));
            CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(final_transposed_desc, CNNL_DTYPE_FLOAT));

            // 执行最后的转置
            cnrtNotifierCreate(&start);
            cnrtNotifierCreate(&end);
            cnrtPlaceNotifier(start, queue);
            
            CNNL_CHECK(cnnlTranspose_v2(handle, trans_desc, transposed_desc, d_echo,
                                       final_transposed_desc, d_output,
                                       transpose_workspace, transpose_workspace_size));

            cnrtPlaceNotifier(end, queue);
            CNRT_CHECK(cnrtQueueSync(queue));
            
            cnrtNotifierDuration(start, end, &hardware_time);
            cout << "第二次转置执行时间: " << hardware_time / 1000 << " ms" << endl;

            // 清理资源
            CNNL_CHECK(cnnlDestroyTensorDescriptor(final_transposed_desc));
            CNNL_CHECK(cnnlDestroyTensorDescriptor(transposed_desc));
            CNNL_CHECK(cnnlDestroyTransposeDescriptor(trans_desc));
            if (transpose_workspace) {
                CNRT_CHECK(cnrtFree(transpose_workspace));
            }
            CNRT_CHECK(cnrtFree(d_transposed));

            // 拷贝结果回主机
            vector<complex<float>> result(Na * Nr);
            CNRT_CHECK(cnrtMemcpy(result.data(), d_output, echo_size, CNRT_MEM_TRANS_DIR_DEV2HOST));  

            // 清理资源
            CNNL_CHECK(cnnlDestroyTensorDescriptor(input_desc));
            CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc));
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

            CNRT_CHECK(cnrtFree(d_echo));
            CNRT_CHECK(cnrtFree(d_output));

            // 打印部分结果用于验证
            cout << "FFT结果前几个值：" << endl;
            for (int i = 1395; i < 1400; ++i) {
                cout << i << ": " << result[i] << " ";
            }
            cout << endl;

            CNNL_CHECK(cnnlDestroyTensorDescriptor(azimuth_input_desc));
            CNNL_CHECK(cnnlDestroyTensorDescriptor(azimuth_output_desc));

            // 生成距离向频率轴 fr_axis
            vector<double> fr_axis(Nr);
            double fr_gap = params.Fr / Nr;
            // fr_axis = fftshift(-Nr/2:Nr/2-1).*fr_gap
            for (size_t i = 0; i < Nr; ++i) {
                int idx;
                if (i < Nr/2) {
                    idx = i + Nr/2;   
                } else {
                    idx = (int)(i - Nr/2);    
                }
                fr_axis[i] = (idx - (int)(Nr/2)) * fr_gap;  
            }

            // 生成方位向频率轴
            vector<double> fa_axis(Na);
            double fa_gap = params.PRF / Na;
            for (size_t i = 0; i < Na; ++i) {
                int idx = i;
                if (i < Na/2) {
                    idx = i + Na/2;    // 后半部分移到前面
                } else {
                    idx = (int)(i - Na/2);    // 前半部分移到后面
                }
                double shift_val = (idx - (int)(Na/2)) * fa_gap;
                fa_axis[i] = params.f_nc + shift_val;
            }

            // 定义参考函数计算所需的常量
            const double c = params.c;
            const double Vr = params.Vr;
            const double R0 = params.R0;
            const double f0 = params.f0;
            const double Kr = -params.Kr;  // 注意这里要取负值,多普勒频域在负值

            // 生成参考函数矩阵
            vector<complex<float>> theta_ft_fa(Na * Nr);

            // // 使用CPU生成参考函数
             generateReferenceFunction_CPU(handle, queue, theta_ft_fa, fr_axis, fa_axis,
                                     f0, c, Vr, Kr, R0, Na, Nr);
            // 分配设备内存
            void *d_theta_ft_fa, *d_result;
            size_t theta_size = Na * Nr * sizeof(complex<float>);
            CNRT_CHECK(cnrtMalloc(&d_theta_ft_fa, theta_size));
            size_t result_size = Na * Nr * sizeof(complex<float>);
            CNRT_CHECK(cnrtMalloc(&d_result, result_size));
            // 拷贝原始结果到设备
            CNRT_CHECK(cnrtMemcpy(d_result, result.data(), result_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
            CNRT_CHECK(cnrtMemcpy(d_theta_ft_fa, theta_ft_fa.data(), theta_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
            // 执行复数矩阵乘法
            cout << "\n执行点乘操作..." << endl;
            HostTimer mul_timer;
            mul_timer.start();

            CNNL_CHECK(ComplexMatrixMultiply(handle, 
                                            queue,
                                            d_result, 
                                            d_theta_ft_fa, 
                                            d_result, 
                                            Na, Nr));
            // 同步MLU队列
            CNRT_CHECK(cnrtQueueSync(queue));
            mul_timer.stop();
            float mul_time = mul_timer.tv_usec;
            cout << "点乘执行时间: " << mul_time/1000 << " ms" << endl;
            // 将点乘结果从设备拷贝回主机
            CNRT_CHECK(cnrtMemcpy(result.data(), d_result, result_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
            // 释放设备内存
            CNRT_CHECK(cnrtFree(d_theta_ft_fa));
            CNRT_CHECK(cnrtFree(d_result));
            cout << "\n点乘后的结果前5x5个值：" << endl;
            for (int i = 0; i < min(5, (int)Na); ++i) {
                for (int j = 0; j < min(5, (int)Nr); ++j) {
                    complex<float> val = result[i * Nr + j];
                    printf("(%.3e,%.3e) ", val.real(), val.imag());
                }
                cout << endl;
            }

            // 在点乘操作后添加：
             vector<complex<float>> stolt_result;
            performStoltInterpolation(handle, queue, result, stolt_result, fr_axis, fa_axis,
                                    f0, c, Vr, Na, Nr,fr_gap);
            CNRT_CHECK(cnrtQueueSync(queue));

            result = stolt_result;
            //2D IFFT
            cout << "\n执行2D IFFT..." << endl;

            // 分配MLU内存
            void* d_ifft_in;
            void* d_ifft_out;
            size_t ifft_size = Na * Nr * sizeof(complex<float>);
            CNRT_CHECK(cnrtMalloc(&d_ifft_in, ifft_size));
            CNRT_CHECK(cnrtMalloc(&d_ifft_out, ifft_size));
            
            // 拷贝插值结果到设备
            CNRT_CHECK(cnrtMemcpy(d_ifft_in, result.data(), ifft_size, CNRT_MEM_TRANS_DIR_HOST2DEV));

            // 创建张量描述符
            cnnlTensorDescriptor_t ifft_input_desc, ifft_output_desc;
            CNNL_CHECK(cnnlCreateTensorDescriptor(&ifft_input_desc));
            CNNL_CHECK(cnnlCreateTensorDescriptor(&ifft_output_desc));
            
            // 设置距离向IFFT的描述符
            int ifft_dims[] = {static_cast<int>(Na), static_cast<int>(Nr)};
            CNNL_CHECK(cnnlSetTensorDescriptor(ifft_input_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, ifft_dims));
            CNNL_CHECK(cnnlSetTensorDescriptor(ifft_output_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, ifft_dims));
            
            CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(ifft_input_desc, CNNL_DTYPE_FLOAT));
            CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(ifft_output_desc, CNNL_DTYPE_FLOAT));

            // 配置距离向IFFT
            size_t ifft_range_workspace_size = 0;
            void *ifft_range_workspace = nullptr;
            size_t ifft_range_reservespace_size = 0;
            void *ifft_range_reservespace = nullptr;
            
            cnnlFFTPlan_t ifft_range_desc;
            CNNL_CHECK(cnnlCreateFFTPlan(&ifft_range_desc));
            
            int ifft_n_range[] = {static_cast<int>(Nr)};  // 改名
            
            // 初始化IFFT计划
            CNNL_CHECK(cnnlMakeFFTPlanMany(handle, ifft_range_desc, ifft_input_desc, ifft_output_desc, 
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

            cnrtNotifierCreate(&start);
            cnrtNotifierCreate(&end);
            cnrtPlaceNotifier(start, queue);

            CNNL_CHECK(cnnlExecFFT(handle, ifft_range_desc, d_ifft_in, range_scale, 
                                  ifft_range_workspace, d_ifft_out, 1));
            
            cnrtPlaceNotifier(end, queue);
            CNRT_CHECK(cnrtQueueSync(queue));
            
            cnrtNotifierDuration(start, end, &hardware_time);
            cout << "距离向IFFT执行时间: " << hardware_time / 1000 << " ms" << endl;
                                  
            // 打印5*5检查点
            complex<float>* check_point = new complex<float>[25];
            CNRT_CHECK(cnrtMemcpy(check_point, d_ifft_out, 25 * sizeof(complex<float>), CNRT_MEM_TRANS_DIR_DEV2HOST));
            cout << "距离向IFFT结果前5*5:" << endl;
            for(int i = 0; i < 5; i++) {
                for(int j = 0; j < 5; j++) {
                    cout << check_point[i*5 + j] << " ";
                }
                cout << endl;
            }
            delete[] check_point;
            
            // 转置数据
            void* d_ifft_transposed;  // 改名
            CNRT_CHECK(cnrtMalloc(&d_ifft_transposed, ifft_size));
            
            // 创建转置描述符
            cnnlTensorDescriptor_t ifft_transposed_desc;  // 改名
            CNNL_CHECK(cnnlCreateTensorDescriptor(&ifft_transposed_desc));
            int ifft_transposed_dims[] = {static_cast<int>(Nr), static_cast<int>(Na)};  // 改名
            CNNL_CHECK(cnnlSetTensorDescriptor(ifft_transposed_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, ifft_transposed_dims));
            CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(ifft_transposed_desc, CNNL_DTYPE_FLOAT));
            
            // 设置转置描述符
            cnnlTransposeDescriptor_t ifft_trans_desc;  // 改名
            CNNL_CHECK(cnnlCreateTransposeDescriptor(&ifft_trans_desc));
            int ifft_perm[] = {1, 0};  // 改名
            CNNL_CHECK(cnnlSetTransposeDescriptor(ifft_trans_desc, 2, ifft_perm));

            // 获取转置工作空间
            size_t ifft_transpose_workspace_size = 0;  // 改名
            CNNL_CHECK(cnnlGetTransposeWorkspaceSize(handle, ifft_input_desc, ifft_trans_desc, &ifft_transpose_workspace_size));
            
            void* ifft_transpose_workspace = nullptr;  // 改名
            if (ifft_transpose_workspace_size > 0) {
                CNRT_CHECK(cnrtMalloc(&ifft_transpose_workspace, ifft_transpose_workspace_size));
            }

            // 执行转置
            CNNL_CHECK(cnnlTranspose_v2(handle, ifft_trans_desc, ifft_input_desc, d_ifft_out,
                                       ifft_transposed_desc, d_ifft_transposed,
                                       ifft_transpose_workspace, ifft_transpose_workspace_size));
            CNRT_CHECK(cnrtQueueSync(queue));
            // 配置方位向IFFT
            size_t ifft_azimuth_workspace_size = 0;
            void *ifft_azimuth_workspace = nullptr;
            size_t ifft_azimuth_reservespace_size = 0;
            void *ifft_azimuth_reservespace = nullptr;
            
            cnnlFFTPlan_t ifft_azimuth_desc;
            CNNL_CHECK(cnnlCreateFFTPlan(&ifft_azimuth_desc));
            
            int ifft_n_azimuth[] = {static_cast<int>(Na)};  // 改名
            
            // 初始化方位向IFFT计划
            CNNL_CHECK(cnnlMakeFFTPlanMany(handle, ifft_azimuth_desc, ifft_transposed_desc, ifft_transposed_desc, 
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
            cnrtNotifierCreate(&start);
            cnrtNotifierCreate(&end);
            cnrtPlaceNotifier(start, queue);

            CNNL_CHECK(cnnlExecFFT(handle, ifft_azimuth_desc, d_ifft_transposed, azimuth_scale, 
                                  ifft_azimuth_workspace, d_ifft_in, 1));
            
            cnrtPlaceNotifier(end, queue);
            CNRT_CHECK(cnrtQueueSync(queue));
            
            cnrtNotifierDuration(start, end, &hardware_time);
            cout << "方位向IFFT执行时间: " << hardware_time / 1000 << " ms" << endl;

            
            // 最后的转置
            CNNL_CHECK(cnnlTranspose_v2(handle, ifft_trans_desc, ifft_transposed_desc, d_ifft_in,
                                       ifft_input_desc, d_ifft_out,
                                       ifft_transpose_workspace, ifft_transpose_workspace_size));
            CNRT_CHECK(cnrtQueueSync(queue));
            // 拷贝结果回主机
            vector<complex<float>> ifft_result(Na * Nr);
            CNRT_CHECK(cnrtMemcpy(ifft_result.data(), d_ifft_out, ifft_size, CNRT_MEM_TRANS_DIR_DEV2HOST));

            // 清理资源
            CNNL_CHECK(cnnlDestroyTensorDescriptor(ifft_input_desc));
            CNNL_CHECK(cnnlDestroyTensorDescriptor(ifft_output_desc));
            CNNL_CHECK(cnnlDestroyTensorDescriptor(ifft_transposed_desc));
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

            CNRT_CHECK(cnrtFree(d_ifft_in));
            CNRT_CHECK(cnrtFree(d_ifft_out));
            CNRT_CHECK(cnrtFree(d_ifft_transposed));

            // 打印IFFT结果
            cout << "\nIFFT结果前5x5个值：" << endl;
            for (int i = 0; i < min(5, (int)Na); ++i) {
                for (int j = 0; j < min(5, (int)Nr); ++j) {
                    complex<float> val = ifft_result[i * Nr + j];
                    printf("(%.3e,%.3e) ", val.real(), val.imag());
                }
                cout << endl;
            }

            // //保存IFFT结果到文件
            // cout << "\n保存结果到result.txt..." << endl;
            // ofstream outfile("result.txt");
            // if (!outfile) {
            //     cerr << "无法创建输出文件" << endl;
            //     return;
            // }

            // // 设置输出精度
            // outfile << scientific;  // 使用科学计数法
            // outfile.precision(6);   // 设置精度为6位

            // // 写入数据
            // for (int i = 0; i < Na; ++i) {
            //     for (int j = 0; j < Nr; ++j) {
            //         complex<float> val = ifft_result[i * Nr + j];
            //         outfile << val.real() << " " << val.imag();
            //         if (j < Nr - 1) {
            //             outfile << " ";  // 在每行的数之间添加空格
            //         }
            //     }
            //     outfile << "\n";  // 每行结束添加换行
            // }

            // outfile.close();
            // cout << "结果已保存到result.txt" << endl;

    } catch (const std::exception& e) {
        cerr << "数据处理错误: " << e.what() << endl;
        return;
    }
    
}

int main() {
    process_wk();
    return 0;
} 