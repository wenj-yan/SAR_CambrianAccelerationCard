#include <time.h>
#include <string>
#include "string.h"
#include "./tool/tool.h"
#include <complex>
#include <iostream>
#include <complex.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <algorithm>  // 为min_element和max_element
#include <unordered_set>

#include "./readfile/read.h"
#include "./complex/complex.h"
#include "./task/task.h"

using namespace std;

// 全局调试标志
bool G_DEBUG = false;
bool G_OUTPUT = false;

void initDevice(int &dev, cnrtQueue_t &queue, cnnlHandle_t &handle) {
    CNRT_CHECK(cnrtGetDevice(&dev));
    cout << "当前使用的MLU设备号: " << dev << endl;
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
        WKParams params ;
params.Vr = 7062;
params.Ka = 1733;
params.f_nc = -6900;
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
            //CNRT_CHECK(cnrtMemcpy(result.data(), d_result, result_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
            // 释放设备内存
            CNRT_CHECK(cnrtFree(d_theta_ft_fa));
            

            HostTimer grid_timer;
            grid_timer.start();
            // 在点乘操作后添加：
             vector<complex<float>> stolt_result(Na * Nr);
            //performStoltInterpolation(handle, queue, result, stolt_result, fr_axis, fa_axis,f0, c, Vr, Na, Nr,fr_gap);
            StoltInterp_sinc(handle, queue, d_result, d_result, fr_axis, fa_axis, f0, c, Vr, Na, Nr, params.Fr,6);
            CNRT_CHECK(cnrtQueueSync(queue));

            grid_timer.stop();
        float grid_time = grid_timer.tv_usec;
        cout << "GridSampleForward执行时间: " << grid_time/1000 << " ms" << endl;

            //result = stolt_result;

//////////////////////////////////////////////////////////////////////
// // 检查NaN的数量
// int nan_count = 0;
// int first_nan = -1;
// cout << "\n检查stolt_result中的NaN值:" << endl;
// for(size_t i = 0; i < stolt_result.size(); i++) {
//     if(isnan(stolt_result[i].real()) || isnan(stolt_result[i].imag())) {
//         if(first_nan == -1) {
//             first_nan = i;
//         }
//         nan_count++;
//         if(nan_count <= 10) {  // 只打印前10个NaN的位置
//             cout << "发现NaN在位置 " << i << " 值: (" 
//                  << stolt_result[i].real() << "," 
//                  << stolt_result[i].imag() << ")" << endl;
//         }
//     }
// }

// cout << "总共发现 " << nan_count << " 个NaN值" << endl;
// cout << "NaN值占比: " << (float)nan_count/stolt_result.size() * 100 << "%" << endl;

// if(nan_count > 0 && first_nan != -1) {
//     // 打印第一个NaN值周围的数据
//     cout << "\n第一个NaN值周围的数据:" << endl;
//     for(int i = max(0, first_nan-5); i < min((int)stolt_result.size(), first_nan+6); i++) {
//         cout << "位置 " << i << ": (" 
//              << stolt_result[i].real() << "," 
//              << stolt_result[i].imag() << ")" << endl;
//     }
// }

// cout << "\nstolt后的结果前5x5个值：" << endl;
// for (int i = 0; i < min(5, (int)Na); ++i) {
//     for (int j = 0; j < min(5, (int)Nr); ++j) {
//         complex<float> val = stolt_result[i * Nr + j];
//         printf("(%.3e,%.3e) ", val.real(), val.imag());
//     }
//     cout << endl;
// }



//////////////////////////////////////////////////////////////////////
            //2D IFFT
            cout << "\n执行2D IFFT..." << endl;

            // 分配MLU内存
            
            void* d_ifft_out;
            size_t ifft_size = Na * Nr * 2*sizeof(float);
            CNRT_CHECK(cnrtMalloc(&d_ifft_out, ifft_size));
            //CNRT_CHECK(cnrtMalloc(&d_result, ifft_size));
            // 拷贝插值结果到设备
            //CNRT_CHECK(cnrtMemcpy(d_result, d_result, ifft_size, cnrtMemcpyDevToDev));

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
            
            int ifft_n_range[] = {static_cast<int>(Nr)};   
            
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

            CNNL_CHECK(cnnlExecFFT(handle, ifft_range_desc, d_result, range_scale, 
                                  ifft_range_workspace, d_ifft_out, 1));
            CNRT_CHECK(cnrtQueueSync(queue));

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
            void* d_ifft_transposed;   
            CNRT_CHECK(cnrtMalloc(&d_ifft_transposed, ifft_size));
            
            // 创建转置描述符
            cnnlTensorDescriptor_t ifft_transposed_desc;  
            CNNL_CHECK(cnnlCreateTensorDescriptor(&ifft_transposed_desc));
            int ifft_transposed_dims[] = {static_cast<int>(Nr), static_cast<int>(Na)};   
            CNNL_CHECK(cnnlSetTensorDescriptor(ifft_transposed_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, ifft_transposed_dims));
            CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(ifft_transposed_desc, CNNL_DTYPE_FLOAT));
            
            // 设置转置描述符
            cnnlTransposeDescriptor_t ifft_trans_desc;  
            CNNL_CHECK(cnnlCreateTransposeDescriptor(&ifft_trans_desc));
            int ifft_perm[] = {1, 0};   
            CNNL_CHECK(cnnlSetTransposeDescriptor(ifft_trans_desc, 2, ifft_perm));

            // 获取转置工作空间
            size_t ifft_transpose_workspace_size = 0;   
            CNNL_CHECK(cnnlGetTransposeWorkspaceSize(handle, ifft_input_desc, ifft_trans_desc, &ifft_transpose_workspace_size));
            
            void* ifft_transpose_workspace = nullptr;   
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
            
            int ifft_n_azimuth[] = {static_cast<int>(Na)};   
            
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
                                  ifft_azimuth_workspace, d_result, 1));
            
            cnrtPlaceNotifier(end, queue);
            CNRT_CHECK(cnrtQueueSync(queue));
            
            cnrtNotifierDuration(start, end, &hardware_time);
            cout << "方位向IFFT执行时间: " << hardware_time / 1000 << " ms" << endl;

            
            // 最后的转置
            CNNL_CHECK(cnnlTranspose_v2(handle, ifft_trans_desc, ifft_transposed_desc, d_result,
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

            

            // 打印IFFT结果
            cout << "\nIFFT结果前5x5个值：" << endl;
            for (int i = 0; i < min(5, (int)Na); ++i) {
                for (int j = 0; j < min(5, (int)Nr); ++j) {
                    complex<float> val = ifft_result[i * Nr + j];
                    printf("(%.3e,%.3e) ", val.real(), val.imag());
                }
                cout << endl;
            }
            if(G_OUTPUT == true)
            {
//保存IFFT结果到文件
cout << "\n保存结果到result.txt..." << endl;
ofstream outfile("result.txt");
if (!outfile) {
    cerr << "无法创建输出文件" << endl;
    return;
}

// 设置输出精度
outfile << scientific;  // 使用科学计数法
outfile.precision(6);   // 设置精度为6位

// 写入数据
for (int i = 0; i < Na; ++i) {
    for (int j = 0; j < Nr; ++j) {
        complex<float> val = ifft_result[i * Nr + j];
        outfile << val.real() << " " << val.imag();
        if (j < Nr - 1) {
            outfile << " ";  // 在每行的数之间添加空格
        }
    }
    outfile << "\n";  // 每行结束添加换行
}

outfile.close();
cout << "结果已保存到result.txt" << endl;



            }
            

            CNRT_CHECK(cnrtFree(d_ifft_out));
            CNRT_CHECK(cnrtFree(d_result));
            CNRT_CHECK(cnrtFree(d_ifft_transposed));

    } catch (const std::exception& e) {
        cerr << "数据处理错误: " << e.what() << endl;
        return;
    }
    
}

void test() {
    printf("开始reduce测试...\n");
    // 初始化设备
    int dev;
    cnrtQueue_t queue = nullptr;
    cnnlHandle_t handle = nullptr;
    initDevice(dev, queue, handle);

    // 创建测试数据
    const int P = 6;    // 第一维大小
    const int Na = 2;   // 第二维大小
    const int Nr = 3;   // 第三维大小
    const int channels = 2; // 复数数据有2个通道

    // 创建测试数据
    vector<float> input_data(P * Na * Nr * channels);
    // 用简单的递增数据填充
    for(int i = 0; i < P * Na * Nr * channels; i++) {
        input_data[i] = static_cast<float>(i + 1);
    }

    // 创建tensor描述符
    cnnlTensorDescriptor_t input_desc, output_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&input_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&output_desc));

    // 设置维度
    int input_dims[4] = {P, Na, Nr, channels};  // [6,2,3,2]
    int output_dims[4] = {1, Na, Nr, channels}; // [1,2,3,2]
    
    CNNL_CHECK(cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, input_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, output_dims));

    // 分配设备内存
    void *d_input, *d_output;
    size_t input_size = P * Na * Nr * channels * sizeof(float);
    size_t output_size = Na * Nr * channels * sizeof(float);
    
    CNRT_CHECK(cnrtMalloc(&d_input, input_size));
    CNRT_CHECK(cnrtMalloc(&d_output, output_size));

    // 拷贝输入数据到设备
    CNRT_CHECK(cnrtMemcpy(d_input, input_data.data(), input_size, CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 创建reduce描述符
    cnnlReduceDescriptor_t reduce_desc;
    CNNL_CHECK(cnnlCreateReduceDescriptor(&reduce_desc));
    
    // 设置reduce参数
    int reduce_axis[] = {0};  // 在第0维上进行reduce
    int num_axes = 1;        
    CNNL_CHECK(cnnlSetReduceDescriptor(
        reduce_desc, 
        reduce_axis,
        num_axes, 
        CNNL_REDUCE_ADD,      
        CNNL_DTYPE_FLOAT,     
        CNNL_NOT_PROPAGATE_NAN,
        CNNL_REDUCE_NO_INDICES,
        CNNL_32BIT_INDICES
    ));

    // 获取workspace大小
    size_t workspace_size = 0;
    CNNL_CHECK(cnnlGetReduceOpWorkspaceSize(
        handle,
        input_desc,
        output_desc,
        reduce_desc,
        &workspace_size
    ));

    // 分配workspace
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&workspace, workspace_size));
    }

    // 执行reduce操作
    float alpha = 1.0f;
    float beta = 0.0f;
    
    CNNL_CHECK(cnnlReduce(
        handle,
        reduce_desc,
        workspace,
        workspace_size,
        &alpha,
        input_desc,
        d_input,
        0,
        nullptr,
        &beta,
        output_desc,
        d_output
    ));
    // 同步队列
    CNRT_CHECK(cnrtQueueSync(queue));

    // 获取结果
    vector<float> output_data(Na * Nr * channels);
    CNRT_CHECK(cnrtMemcpy(output_data.data(), d_output, output_size, CNRT_MEM_TRANS_DIR_DEV2HOST));

    // 打印结果
    cout << "Reduce结果:" << endl;
    for(int i = 0; i < Na; i++) {
        for(int j = 0; j < Nr; j++) {
            cout << "(" << output_data[(i*Nr + j)*2] << "," 
                 << output_data[(i*Nr + j)*2 + 1] << ") ";
        }
        cout << endl;
    }

    // 清理资源
    CNNL_CHECK(cnnlDestroyReduceDescriptor(reduce_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(input_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc));
    
    if (workspace) {
        CNRT_CHECK(cnrtFree(workspace));
    }
    CNRT_CHECK(cnrtFree(d_input));
    CNRT_CHECK(cnrtFree(d_output));

    cout << "Reduce测试完成" << endl;
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    std::unordered_set<std::string> args;
    for (int i = 1; i < argc; ++i) {
        args.insert(argv[i]);
    }
    
    // 设置调试标志
    if (args.count("--info") || args.count("-i")) {
        G_DEBUG = true;
    }
    if (args.count("--out") || args.count("-o")) {
        G_OUTPUT = true;
    }

    process_wk();

    //test();
    return 0;
}


