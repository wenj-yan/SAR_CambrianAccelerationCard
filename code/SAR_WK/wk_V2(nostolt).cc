/**
 * @file wk_V2(nostolt).cc
 * @brief SAR-WK成像算法实现(基于MLU平台)
 * @author wenjyan
 * @email wenjyan@outlook.com
 * @version 2.0
 * @date 2025-01-06
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

// 实数矩阵乘法函数
cnnlStatus_t MatrixMultiply(cnnlHandle_t handle,
                           const float* A,  // 输入矩阵 A
                           const float* B,  // 输入矩阵 B
                           float* C,        // 输出矩阵 C
                           int M,  int N) {
    try {
        // 分配设备内存
        void* d_A;
        void* d_B;
        void* d_C;
        size_t size = M * N * sizeof(float);
        
        CNRT_CHECK(cnrtMalloc(&d_A, size));
        CNRT_CHECK(cnrtMalloc(&d_B, size));
        CNRT_CHECK(cnrtMalloc(&d_C, size));

        // 拷贝数据到设备
        CNRT_CHECK(cnrtMemcpy(d_A, const_cast<void*>(static_cast<const void*>(A)), 
                             size, CNRT_MEM_TRANS_DIR_HOST2DEV));
        CNRT_CHECK(cnrtMemcpy(d_B, const_cast<void*>(static_cast<const void*>(B)), 
                             size, CNRT_MEM_TRANS_DIR_HOST2DEV));

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
        const void* data_array[] = {d_A, d_B};

        // 执行逐元素相乘
        CNNL_CHECK(cnnlMulN(handle, desc_array, data_array, 2, c_desc, d_C));

        // 拷贝结果回主机
        CNRT_CHECK(cnrtMemcpy(C, d_C, size, CNRT_MEM_TRANS_DIR_DEV2HOST));

        // 清理资源
        CNNL_CHECK(cnnlDestroyTensorDescriptor(a_desc));
        CNNL_CHECK(cnnlDestroyTensorDescriptor(b_desc));
        CNNL_CHECK(cnnlDestroyTensorDescriptor(c_desc));
        
        CNRT_CHECK(cnrtFree(d_A));
        CNRT_CHECK(cnrtFree(d_B));
        CNRT_CHECK(cnrtFree(d_C));

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
                                  const complex<float>* A,
                                  const complex<float>* B,
                                  complex<float>* C,
                                  int M,  int N) {
    try {
        // 为实数矩阵分配内存
        vector<float> A_real(M * N);  // A的实部
        vector<float> A_imag(M * N);  // A的虚部
        vector<float> B_real(M * N);  // B的实部
        vector<float> B_imag(M * N);  // B的虚部
        vector<float> C_real(M * N);  // C的实部
        vector<float> C_imag(M * N);  // C的虚部

        // 分离实部和虚部
        for (int i = 0; i < M * N; ++i) {
            A_real[i] = A[i].real();
            A_imag[i] = A[i].imag();
            B_real[i] = B[i].real();
            B_imag[i] = B[i].imag();
        }

        // 临时矩阵用于存储中间结果
        vector<float> temp1(M * N);
        vector<float> temp2(M * N);

        // 计算实部：real = A_real * B_real - A_imag * B_imag
        CNNL_CHECK(MatrixMultiply(handle, A_real.data(), B_real.data(), C_real.data(), M,  N));
        CNNL_CHECK(MatrixMultiply(handle, A_imag.data(), B_imag.data(), temp1.data(), M,  N));
        CNRT_CHECK(cnrtQueueSync(queue));
        for (int i = 0; i < M * N; ++i) {
            C_real[i] -= temp1[i];
        }

        // 计算虚部：imag = A_real * B_imag + A_imag * B_real
        CNNL_CHECK(MatrixMultiply(handle, A_real.data(), B_imag.data(), C_imag.data(), M,  N));
        CNNL_CHECK(MatrixMultiply(handle, A_imag.data(), B_real.data(), temp2.data(), M,  N));
        CNRT_CHECK(cnrtQueueSync(queue));
        for (int i = 0; i < M * N; ++i) {
            C_imag[i] += temp2[i];
        }

        // 组合实部和虚部
        for (int i = 0; i < M * N; ++i) {
            C[i] = complex<float>(C_real[i], C_imag[i]);
        }

        return CNNL_STATUS_SUCCESS;
    }
    catch (const std::exception& e) {
        cerr << "复数矩阵乘法错误: " << e.what() << endl;
        return CNNL_STATUS_EXECUTION_FAILED;
    }
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

// 修改后的Stolt插值函数
void performStoltInterpolation(cnnlHandle_t handle, 
                             cnrtQueue_t queue,
                             const vector<complex<float>>& input,
                             vector<complex<float>>& output,
                             const vector<double>& fr_axis,
                             const vector<double>& fa_axis,
                             double f0, double c, double Vr,
                             int Na, int Nr) {
    
    cout << "\n执行Stolt插值..." << endl;

    // 首先计算新的频率映射
    vector<float> fr_new_mtx = calculateNewFrequencyMapping(fr_axis, fa_axis, f0, c, Vr, Na, Nr);

    // 计算fr_new_mtx的最大值和最小值，用于归一化
    float fr_new_min = *min_element(fr_new_mtx.begin(), fr_new_mtx.end());
    float fr_new_max = *max_element(fr_new_mtx.begin(), fr_new_mtx.end());
    float fr_min = *min_element(fr_axis.begin(), fr_axis.end());
    float fr_max = *max_element(fr_axis.begin(), fr_axis.end());

    cout << "fr_new_mtx范围: [" << fr_new_min << ", " << fr_new_max << "]" << endl;
    cout << "fr_axis范围: [" << fr_min << ", " << fr_max << "]" << endl;

  

    cout << "Stolt插值完成" << endl;
}

// 生成参考函数
void generateReferenceFunction(cnnlHandle_t handle,
                             cnrtQueue_t queue,
                             vector<complex<float>>& theta_ft_fa,
                             const vector<double>& fr_axis,
                             const vector<double>& fa_axis,
                             double f0, double c, double Vr, double Kr, double R0,
                             int Na, int Nr) {
    
    // 创建OpTensor描述符用于基本运算
    cnnlOpTensorDescriptor_t op_desc;
    CNNL_CHECK(cnnlCreateOpTensorDescriptor(&op_desc));
    
    // 创建另一个用于乘法的描述符
    cnnlOpTensorDescriptor_t mul_desc;
    CNNL_CHECK(cnnlCreateOpTensorDescriptor(&mul_desc));
    
    // 设置加法描述符
    CNNL_CHECK(cnnlSetOpTensorDescriptor(op_desc, CNNL_OP_TENSOR_ADD, CNNL_DTYPE_FLOAT, CNNL_NOT_PROPAGATE_NAN));
    
    // 设置乘法描述符
    CNNL_CHECK(cnnlSetOpTensorDescriptor(mul_desc, CNNL_OP_TENSOR_MUL, CNNL_DTYPE_FLOAT, CNNL_NOT_PROPAGATE_NAN));

    // 创建张量描述符
    cnnlTensorDescriptor_t grid_desc, temp_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&grid_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&temp_desc));

    // 设置网格维度 [Na, Nr]
    int grid_dims[] = {Na, Nr};
    CNNL_CHECK(cnnlSetTensorDescriptor(grid_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, grid_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(temp_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, grid_dims));
    // 设置onchip数据类型
    CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(grid_desc, CNNL_DTYPE_FLOAT));

    // 分配设备内存
    void *d_fr_grid, *d_fa_grid, *d_temp1, *d_temp2, *d_output;
    size_t grid_size = Na * Nr * sizeof(float);
    
    CNRT_CHECK(cnrtMalloc(&d_fr_grid, grid_size));
    CNRT_CHECK(cnrtMalloc(&d_fa_grid, grid_size));
    CNRT_CHECK(cnrtMalloc(&d_temp1, grid_size));
    CNRT_CHECK(cnrtMalloc(&d_temp2, grid_size));
    CNRT_CHECK(cnrtMalloc(&d_output, grid_size * 2));  // 复数输出

    // 打印原始的fa_axis
    cout << "\n原始fa_axis的前几个值：" << endl;
    for (int i = 0; i < 5; i++) {
        cout << fa_axis[i] << " ";
    }
    cout << endl;

    // 手动生成网格
    vector<float> fr_grid(Na * Nr);
    vector<float> fa_grid(Na * Nr);
    for (int i = 0; i < Na; i++) {
        for (int j = 0; j < Nr; j++) {
            fr_grid[i * Nr + j] = static_cast<float>(fr_axis[j]);  // 添加类型转换
            fa_grid[i * Nr + j] = static_cast<float>(fa_axis[i]);  // 添加类型转换
        }
    }

    // 打印生成的网格数据
    // cout << "\nfr_grid的前几个值（第一行）：" << endl;
    // for (int j = 0; j < 5; j++) {
    //     cout << fr_grid[j] << " ";
    // }
    // cout << endl;

    // cout << "\nfa_grid的前几个值（第一列）：" << endl;
    // for (int i = 0; i < 5; i++) {
    //     cout << fa_grid[i * Nr] << " ";
    // }
    // cout << endl;

    // // 打印fa_grid的一个小块，验证2D结构
    // cout << "\nfa_grid的5x5块：" << endl;
    // for (int i = 0; i < 5; i++) {
    //     for (int j = 0; j < 5; j++) {
    //         cout << fa_grid[i * Nr + j] << " ";
    //     }
    //     cout << endl;
    // }

    // 拷贝网格数据到设备
    CNRT_CHECK(cnrtMemcpy(d_fr_grid, fr_grid.data(), grid_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_fa_grid, fa_grid.data(), grid_size, CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 定义常量
    float alpha1 = 1.0f;
    float alpha2 = 1.0f;
    float beta = 0.0f;
    float f0_val = static_cast<float>(f0);

    // 获取工作空间大小
    size_t workspace_size = 0;
    CNNL_CHECK(cnnlGetOpTensorWorkspaceSize(handle, grid_desc, grid_desc, grid_desc, &workspace_size));
    
    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&workspace, workspace_size));
    }

    // 修改f0的添加方式
    vector<float> f0_grid(Na * Nr, f0_val);  // 创建一个全是f0的网格
    void* d_f0_grid;
    CNRT_CHECK(cnrtMalloc(&d_f0_grid, grid_size));
    CNRT_CHECK(cnrtMemcpy(d_f0_grid, f0_grid.data(), grid_size, CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 1. 计算 (f0+fr_axis)
    CNNL_CHECK(cnnlOpTensor(handle, op_desc, &alpha1, grid_desc, d_f0_grid,
                   &alpha1, grid_desc, d_fr_grid, workspace, workspace_size,
                   &beta, grid_desc, d_temp1));
    
    // 打印第一步结果
    vector<float> debug_data(Na * Nr);
    CNRT_CHECK(cnrtMemcpy(debug_data.data(), d_temp1, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cout << "\n步骤1 - (f0+fr_axis) 的前5*5个值：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << debug_data[i * Nr + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    // 打印fa_grid的值
    vector<float> fa_debug(Na * Nr);
    CNRT_CHECK(cnrtMemcpy(fa_debug.data(), d_fa_grid, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cout << "\nfa_grid的前5x5个值（计算平方前）：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << fa_debug[i * Nr + j] << " ";
        }
        cout << endl;
    }
    // 2. 计算 fa_axis^2 
    CNNL_CHECK(cnnlSquare(handle,
                         grid_desc,    // 输入描述符
                         d_fa_grid,    // 输入数据
                         grid_desc,    // 输出描述符
                         d_temp2));    // 输出数据
    
    // 添加同步点
    CNRT_CHECK(cnrtQueueSync(queue));
    
    // 立即检查结果
    CNRT_CHECK(cnrtMemcpy(debug_data.data(), d_temp2, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));

    cout << "\n步骤2 - fa_axis^2 的前5x5个值：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << debug_data[i * Nr + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    // 3. 计算 c^2/4/Vr^2 * fa^2
    float scale = c * c / (4 * Vr * Vr);
    cout << "\n步骤3 - scale = " << scale << endl;
    
    CNNL_CHECK(cnnlOpTensor(handle, op_desc, &scale, grid_desc, d_temp2,
                           &beta, grid_desc, d_temp2, workspace, workspace_size,
                           &beta, grid_desc, d_temp2));

    CNRT_CHECK(cnrtQueueSync(queue));

    CNRT_CHECK(cnrtMemcpy(debug_data.data(), d_temp2, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cout << "步骤3 - c^2/4/Vr^2 * fa^2 的前5x5个值：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << debug_data[i * Nr + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    // 4. 计算 (f0+fr)^2
    // CNNL_CHECK(cnnlOpTensor(handle, op_desc, &alpha1, grid_desc, d_temp1,
    //                        &alpha1, grid_desc, d_temp1, workspace, workspace_size,
    //                        &beta, grid_desc, d_temp1));
    CNNL_CHECK(cnnlSquare(handle,
                         grid_desc,    // 输入描述符
                         d_temp1,    // 输入数据
                         grid_desc,    // 输出描述符
                         d_temp1));    // 输出数据

    CNRT_CHECK(cnrtQueueSync(queue));
    
    CNRT_CHECK(cnrtMemcpy(debug_data.data(), d_temp1, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cout << "\n步骤4 - (f0+fr)^2 的前5x5个值：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << debug_data[i * Nr + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    // 5. 计算 (f0+fr)^2 - c^2/4/Vr^2 * fa^2
    //CNNL_CHECK(cnnlSetOpTensorDescriptor(op_desc, CNNL_OP_TENSOR_SUB, CNNL_DTYPE_FLOAT, CNNL_NOT_PROPAGATE_NAN));
    float minus_one = -1.0f;
    CNNL_CHECK(cnnlOpTensor(handle, op_desc, &alpha1, grid_desc, d_temp1,
                           &minus_one, grid_desc, d_temp2, workspace, workspace_size,
                           &beta, grid_desc, d_temp1));
    CNRT_CHECK(cnrtQueueSync(queue));
    // 缩放操作
    // float scale_factor = 1e-12f;  // 根据实际数值大小调整缩放因子
    // CNNL_CHECK(cnnlOpTensor(handle, op_desc, &scale_factor, grid_desc, d_temp1,
    //                     &beta, grid_desc, d_temp1, workspace, workspace_size,
    //                     &beta, grid_desc, d_temp1));
    // CNRT_CHECK(cnrtQueueSync(queue));

    CNRT_CHECK(cnrtMemcpy(debug_data.data(), d_temp1, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cout << "\n步骤5 - (f0+fr)^2 - c^2/4/Vr^2 * fa^2 的前5x5个值：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << debug_data[i * Nr + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    // 转换为double类型
    vector<double> data_double(Na * Nr);
    for (int i = 0; i < Na * Nr; i++) {
        data_double[i] = static_cast<double>(debug_data[i]);
    }
    // 6. 计算平方根
    // CNNL_CHECK(cnnlSqrt_v2(handle,
    //                       CNNL_COMPUTATION_HIGH_PRECISION,
    //                       grid_desc,
    //                       d_temp1,
    //                       grid_desc,
    //                       d_temp2));

    // CNRT_CHECK(cnrtQueueSync(queue));
    vector<double> data_sqrt(Na * Nr);
    for (int i = 0; i < Na * Nr; i++) {
        data_sqrt[i] = sqrt(data_double[i]);
    }
    // //恢复原始比例
    // float restore_scale = 1.0f / sqrt(scale_factor);  // 使用平方根因为后面做了sqrt运算
    // cout << "restore_scale = " << restore_scale << endl;
    // CNNL_CHECK(cnnlOpTensor(handle, op_desc, &restore_scale, grid_desc, d_temp2,
    //                     &beta, grid_desc, d_temp2, workspace, workspace_size,
    //                     &beta, grid_desc, d_temp2));
    // CNRT_CHECK(cnrtQueueSync(queue));
    //CNRT_CHECK(cnrtMemcpy(debug_data.data(), d_temp2, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cout << "\n步骤6 - sqrt 的前5x5个值：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << data_sqrt[i * Nr + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    // 7. 计算 4*pi*R0/c * sqrt(...)
    float four_pi_R0_c = 4 * M_PI * R0 / c;
    CNNL_CHECK(cnnlOpTensor(handle, op_desc, &four_pi_R0_c, grid_desc, d_temp2,
                           &beta, grid_desc, d_temp2, workspace, workspace_size,
                           &beta, grid_desc, d_temp1));
    CNRT_CHECK(cnrtQueueSync(queue));

    CNRT_CHECK(cnrtMemcpy(debug_data.data(), d_temp1, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cout << "\n步骤7 - 4*pi*R0/c * sqrt(...) 的前5x5个值：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << debug_data[i * Nr + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    // 8. 计算 fr^2
    // CNNL_CHECK(cnnlOpTensor(handle, op_desc, &alpha1, grid_desc, d_fr_grid,
    //                        &alpha1, grid_desc, d_fr_grid, workspace, workspace_size,
    //                        &beta, grid_desc, d_temp2));
    CNNL_CHECK(cnnlSquare(handle,
                         grid_desc,    // 输入描述符
                         d_fr_grid,    // 输入数据
                         grid_desc,    // 输出描述符
                         d_temp2));    // 输出数据
    CNRT_CHECK(cnrtQueueSync(queue));
    CNRT_CHECK(cnrtMemcpy(debug_data.data(), d_temp2, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cout << "\n步骤8 - fr^2 的前5x5个值：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << debug_data[i * Nr + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    // 9. 计算 pi/Kr * fr^2
    double pi_Kr = M_PI / Kr * (-1.0);  // 使用 double 而不是 float
    cout << std::scientific;   // 使用科学计数法显示
    cout << "M_PI = " << M_PI << endl;
    cout << "Kr = " << Kr << endl;
    cout << "pi_Kr = " << pi_Kr << endl;
    cout << std::defaultfloat; // 恢复默认显示格式
    float scale1 = 1e12f;
    CNNL_CHECK(cnnlOpTensor(handle, op_desc, &scale1, grid_desc, d_temp2,
                        &beta, grid_desc, d_temp2, workspace, workspace_size,
                        &beta, grid_desc, d_temp2));
    CNRT_CHECK(cnrtQueueSync(queue));
// 然后乘以 (pi/Kr * 1e-12)
    float adjusted_pi_Kr = static_cast<float>(pi_Kr * 1e-12);
    CNNL_CHECK(cnnlOpTensor(handle, op_desc, &adjusted_pi_Kr, grid_desc, d_temp2,
                        &beta, grid_desc, d_temp2, workspace, workspace_size,
                        &beta, grid_desc, d_temp2));

    CNRT_CHECK(cnrtQueueSync(queue));
    CNRT_CHECK(cnrtMemcpy(debug_data.data(), d_temp2, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cout << "\n步骤9 - pi/Kr * fr^2 的前5x5��值：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << debug_data[i * Nr + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    // 10. 计算最终相位
    CNNL_CHECK(cnnlOpTensor(handle, op_desc, &alpha1, grid_desc, d_temp1,
                   &alpha1, grid_desc, d_temp2, workspace, workspace_size,
                   &beta, grid_desc, d_temp1));
    CNRT_CHECK(cnrtQueueSync(queue));
    CNRT_CHECK(cnrtMemcpy(debug_data.data(), d_temp1, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cout << "\n最终相位值的前5x5个值：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            size_t idx = i * Nr + j;
            cout << fixed << "(" << debug_data[idx] << ") ";
        }
        cout << endl;
    }

    // 计算sin和cos
    cnnlTensorDescriptor_t complex_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&complex_desc));
    int complex_dims[] = {Na, Nr, 2};  // 最后一维存储实部和虚部
    CNNL_CHECK(cnnlSetTensorDescriptor(complex_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, complex_dims));

    // 分别计算sin和cos
    void *d_cos, *d_sin;
    CNRT_CHECK(cnrtMalloc(&d_cos, grid_size));
    CNRT_CHECK(cnrtMalloc(&d_sin, grid_size));

    // 创建sin和cos的输入输出描述符
    cnnlTensorDescriptor_t phase_desc, result_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&phase_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&result_desc));
    
    // 设置描述符
    int phase_dims[] = {Na, Nr};
    CNNL_CHECK(cnnlSetTensorDescriptor(phase_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, phase_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(result_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, phase_dims));

    // 计算cos和sin之前，分配归一化相位的内存
    void *d_normalized;
    CNRT_CHECK(cnrtMalloc(&d_normalized, grid_size));

    // 创建一个包含1/(2π)的设备内存
    float inv_two_pi = 1.0f / (2.0f * M_PI);
    cout << "\n1/(2π) = " << inv_two_pi << endl;

    // 使用乘法代替除法进行归一化
    CNNL_CHECK(cnnlOpTensor(handle,
                           op_desc,         // 操作描述符
                           &inv_two_pi,     // alpha1 = 1/(2π)
                           phase_desc,      // a_desc
                           d_temp1,         // a (相位)
                           &beta,           // alpha2 = 0
                           phase_desc,      // b_desc
                           d_temp1,         // b (不使用)
                           workspace,       // workspace
                           workspace_size,  // workspace_size
                           &beta,          // beta
                           phase_desc,      // c_desc
                           d_normalized));  // c (归一化后的相位)

    CNRT_CHECK(cnrtQueueSync(queue));

    // 打印归一化前后的相位值进行对比
    vector<float> phase_before(Na * Nr);
    vector<float> phase_after(Na * Nr);
    CNRT_CHECK(cnrtMemcpy(phase_before.data(), d_temp1, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(phase_after.data(), d_normalized, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    
    cout << "\n归一化前后的相位对比（前5个值）：" << endl;
    for (int i = 0; i < 5; i++) {
        cout << "原始相位: " << phase_before[i] 
             << ", 归一化后: " << phase_after[i] 
             << ", 手动验证: " << phase_before[i] * inv_two_pi << endl;
    }

    // 计算cos 
    CNNL_CHECK(cnnlCos_v2(handle,
                     CNNL_COMPUTATION_HIGH_PRECISION,
                     grid_desc,        // x_desc
                     d_normalized,     // 使用归一化后的相位
                     grid_desc,        // y_desc
                     d_cos));          // y (cos结果)
    CNRT_CHECK(cnrtQueueSync(queue));

    // 计算sin 
    CNNL_CHECK(cnnlSin_v2(handle,
                     CNNL_COMPUTATION_HIGH_PRECISION,
                     grid_desc,        // x_desc
                     d_normalized,     // 使用归一化后的相位
                     grid_desc,        // y_desc
                     d_sin));          // y (sin结果)
    CNRT_CHECK(cnrtQueueSync(queue));

    // 合并cos和sin到复数输出
    vector<float> cos_data(Na * Nr);
    vector<float> sin_data(Na * Nr);
    CNRT_CHECK(cnrtMemcpy(cos_data.data(), d_cos, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(sin_data.data(), d_sin, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));

    // 转换为复数形式
    theta_ft_fa.resize(Na * Nr);
    for (int i = 0; i < Na * Nr; i++) {
        theta_ft_fa[i] = complex<float>(cos_data[i], sin_data[i]);
    }

    // 打印sin和cos的结果
    vector<float> cos_debug(Na * Nr);
    vector<float> sin_debug(Na * Nr);
    CNRT_CHECK(cnrtMemcpy(cos_debug.data(), d_cos, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(sin_debug.data(), d_sin, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    
    cout << "\ncos和sin的前5x5个值：" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            size_t idx = i * Nr + j;
            cout << "(" << cos_debug[idx] << "," << sin_debug[idx] << ") ";
        }
        cout << endl;
    }

    // 清理资源 - 只保留一次释放
    CNNL_CHECK(cnnlDestroyOpTensorDescriptor(op_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(grid_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(temp_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(complex_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(phase_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(result_desc));
    
    if (workspace) {
        CNRT_CHECK(cnrtFree(workspace));
    }
    CNRT_CHECK(cnrtFree(d_fr_grid));
    CNRT_CHECK(cnrtFree(d_fa_grid));
    CNRT_CHECK(cnrtFree(d_temp1));
    CNRT_CHECK(cnrtFree(d_temp2));
    CNRT_CHECK(cnrtFree(d_output));
    CNRT_CHECK(cnrtFree(d_f0_grid));
    CNRT_CHECK(cnrtFree(d_cos));
    CNRT_CHECK(cnrtFree(d_sin));
    CNNL_CHECK(cnnlDestroyOpTensorDescriptor(mul_desc));

    // 在清理资源时添加
    CNRT_CHECK(cnrtFree(d_normalized));
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

        // 计算填充后的维度
        size_t Na_padded = Na + 2 * round(Na/6.0f);  
        size_t Nr_padded = Nr + 2 * round(Nr/3.0f);  
        cout << "填充后维度: Na_padded = " << Na_padded << ", Nr_padded = " << Nr_padded << endl;
        cout << "原始维度: Na = " << Na << ", Nr = " << Nr << endl;
        cout << "填充量: Na_pad = " << 2 * round(Na/6.0f) << ", Nr_pad = " << 2 * round(Nr/3.0f) << endl;

        // 创建填充后的数据数组
        vector<complex<float>> echo_padded(Na_padded * Nr_padded, complex<float>(0, 0));

        // 执行填充（将原始数据复制到中心位置）
        size_t Na_start = round(Na/6);
        size_t Nr_start = round(Nr/3);
        for (size_t i = 0; i < Na; ++i) {
            for (size_t j = 0; j < Nr; ++j) {
                echo_padded[(i + Na_start) * Nr_padded + (j + Nr_start)] = echo[i * Nr + j];
            } 
        }
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
            CNNL_CHECK(cnnlTranspose_v2(handle, trans_desc, input_desc, d_output,
                                       transposed_desc, d_transposed,
                                       transpose_workspace, transpose_workspace_size));

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
            CNNL_CHECK(cnnlTranspose_v2(handle, trans_desc, transposed_desc, d_echo,
                                       final_transposed_desc, d_output,
                                       transpose_workspace, transpose_workspace_size));

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

            // 在清理时添加新描述符的销毁
            CNNL_CHECK(cnnlDestroyTensorDescriptor(azimuth_input_desc));
            CNNL_CHECK(cnnlDestroyTensorDescriptor(azimuth_output_desc));

            // 生成距离向频率轴 fr_axis
            vector<double> fr_axis(Nr);
            double fr_gap = params.Fr / Nr;
            // fr_axis = fftshift(-Nr/2:Nr/2-1).*fr_gap
            for (size_t i = 0; i < Nr; ++i) {
                int idx;
                if (i < Nr/2) {
                    idx = i + Nr/2;    // 后半部分移到前面
                } else {
                    idx = (int)(i - Nr/2);    // 前半部分移到后面
                }
                fr_axis[i] = (idx - (int)(Nr/2)) * fr_gap;  
            }

            // 生成方位向频率轴
            vector<double> fa_axis(Na);
            double fa_gap = params.PRF / Na;
            // fa_axis = f_nc + fftshift(-Na/2:Na/2-1).*fa_gap
            for (size_t i = 0; i < Na; ++i) {
                int idx = i;
                if (i < Na/2) {
                    idx = i + Na/2;    // 后半部分移到前面
                } else {
                    idx = (int)(i - Na/2);    // 前半部分移到后面
                }
                // 修正：先计算-Na/2:Na/2-1范围内的值，再加上f_nc
                double shift_val = (idx - (int)(Na/2)) * fa_gap;
                fa_axis[i] = params.f_nc + shift_val;
            }

            // 定义参考函数计算所需的常量
            // cout.setf(ios::fixed);              // 使用定点表示法
            // cout.unsetf(ios::scientific);       // 禁用科学计数法
            const double c = params.c;
            const double Vr = params.Vr;
            const double R0 = params.R0;
            const double f0 = params.f0;
            const double Kr = -params.Kr;  // 注意这里要取负值，与MATLAB代码一致

            // 预计算常量
            const double four_pi_R0_c = 4.0 * M_PI * R0 / c;
            const double c_2Vr_2 = c * c / (4.0 * Vr * Vr);
            const double pi_Kr = M_PI / Kr;

            // 生成参考函数矩阵
            vector<complex<float>> theta_ft_fa(Na * Nr);

            // // 使用CPU生成参考函数
             generateReferenceFunction_CPU(handle, queue, theta_ft_fa, fr_axis, fa_axis,
                                     f0, c, Vr, Kr, R0, Na, Nr);
            
            // cout << "开始读取数据文件..." << endl;
            // if (!readComplexData("data/ref.txt", theta_ft_fa)) {
            //     cerr << "读取数据失败" << endl;
            //     return;
            // }

            // 执行复数矩阵乘法
            cout << "\n执行点乘操作..." << endl;
            CNNL_CHECK(ComplexMatrixMultiply(handle, 
                                            queue,
                                            result.data(), 
                                            theta_ft_fa.data(), 
                                            result.data(), 
                                            Na, Nr));
            // 同步MLU队列
            CNRT_CHECK(cnrtQueueSync(queue));

            // 在点乘操作后添加：
             vector<complex<float>> stolt_result;
            // performStoltInterpolation(handle, queue, result, stolt_result, fr_axis, fa_axis,
            //                         f0, c, Vr, Na, Nr);
            //result = stolt_result;
            cout << "\n插值后的结果前5x5个值：" << endl;
            for (int i = 0; i < min(5, (int)Na); ++i) {
                for (int j = 0; j < min(5, (int)Nr); ++j) {
                    complex<float> val = result[i * Nr + j];
                    printf("(%.3e,%.3e) ", val.real(), val.imag());
                }
                cout << endl;
            }



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
            CNNL_CHECK(cnnlExecFFT(handle, ifft_range_desc, d_ifft_in, range_scale, 
                                  ifft_range_workspace, d_ifft_out, 1));
            CNRT_CHECK(cnrtQueueSync(queue));
                                  
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
            CNNL_CHECK(cnnlExecFFT(handle, ifft_azimuth_desc, d_ifft_transposed, azimuth_scale, 
                                  ifft_azimuth_workspace, d_ifft_in, 1));
            CNRT_CHECK(cnrtQueueSync(queue));
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

            // // 打印IFFT结果
            // cout << "\nIFFT结果前5x5个值：" << endl;
            // for (int i = 0; i < min(5, (int)Na); ++i) {
            //     for (int j = 0; j < min(5, (int)Nr); ++j) {
            //         complex<float> val = ifft_result[i * Nr + j];
            //         printf("(%.3e,%.3e) ", val.real(), val.imag());
            //     }
            //     cout << endl;
            // }

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