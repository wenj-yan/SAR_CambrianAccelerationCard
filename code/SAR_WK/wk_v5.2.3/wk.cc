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

        //前处理，包括
        //1.padding
        //2.方位向变频
                // 准备设备内存
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
        void *d_input,*d_output;
        size_t input_size = Na * Nr * 2 * sizeof(float);
        size_t echo_size = Na_padded * Nr_padded * sizeof(complex<float>);
        CNRT_CHECK(cnrtMalloc(&d_input, input_size));
        CNRT_CHECK(cnrtMalloc(&d_output, echo_size));

        // 拷贝输入数据到设备
        CNRT_CHECK(cnrtMemcpy(d_input, echo.data(), input_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
        CNRT_CHECK(cnrtQueueSync(queue));
        preprocess_wk(handle, queue, d_input, Na,Nr,Na_padded, Nr_padded,params,d_output);

        CNRT_CHECK(cnrtFree(d_input));    
        // 更新维度变量
        Na = Na_padded;
        Nr = Nr_padded;

    //     // 准备设备内存
    //     void *d_input, *d_output2;
    //     size_t input_size = Na * Nr * 2 * sizeof(float);
    //     size_t output_size = Na_padded * Nr_padded * 2 * sizeof(float);
        
    //     CNRT_CHECK(cnrtMalloc(&d_input, input_size));
    //     CNRT_CHECK(cnrtMalloc(&d_output2, output_size));

    //     // 拷贝输入数据到设备
    //     CNRT_CHECK(cnrtMemcpy(d_input, echo.data(), input_size, CNRT_MEM_TRANS_DIR_HOST2DEV));

    //     // 执行pad操作
    //     PadComplexTensor(handle, queue, d_input, Na, Nr,
    //                     pad_before_Na, pad_after_Na,
    //                     pad_before_Nr, pad_after_Nr,
    //                     complex<float>(0.0f, 0.0f), d_output2);

    //     // 分配主机内存接收结果
    //     vector<complex<float>> echo_padded(Na_padded * Nr_padded);

    //     // 拷贝结果回主机
    //     CNRT_CHECK(cnrtMemcpy(echo_padded.data(), d_output2, output_size, CNRT_MEM_TRANS_DIR_DEV2HOST));

    //     // 释放设备内存
    //     CNRT_CHECK(cnrtFree(d_input));
    //     CNRT_CHECK(cnrtFree(d_output2));


        
    // //////////////////////////////////////////////////////
    // // 方位向变频（频谱搬移）
    // /////////////////////////////////////////////////////
    //     // 生成方位向时间轴
    //     float Fa = params.PRF;  // 方位向采样率等于PRF
    //     vector<float> ta_axis(Na_padded);
    //     float ta_step = 1.0f / Fa;
    //     for (size_t i = 0; i < Na_padded; ++i) {
    //         ta_axis[i] = (static_cast<float>(i) - Na_padded/2) * ta_step;
    //     }

    //     // 执行方位向下变频
    //     float f_nc = params.f_nc;  // 多普勒中心频率
    //     for (size_t i = 0; i < Na_padded; ++i) {
    //         float phase = -2 * M_PI * f_nc * ta_axis[i];
    //         complex<float> exp_factor(cos(phase), sin(phase));
    //         for (size_t j = 0; j < Nr_padded; ++j) {
    //             echo_padded[i * Nr_padded + j] *= exp_factor;
    //         }
    //     }

    //     // 更新维度变量，使用填充后��维度
    //     Na = Na_padded;
    //     Nr = Nr_padded;
    //     echo = std::move(echo_padded);
    //     // 分配MLU内存并拷贝数据
    //     void* d_output;
    //     size_t echo_size = Na * Nr * sizeof(complex<float>);
        
    //     CNRT_CHECK(cnrtMalloc((void **)&d_output, echo_size));
        
    //     HostTimer copyin_timer;
    //     copyin_timer.start();
    //     CNRT_CHECK(cnrtMemcpy(d_output, echo.data(), echo_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    //     copyin_timer.stop();
    //     optensor.memcpyH2D_time = copyin_timer.tv_usec;
    // //////////////////////////////////////////////////////
    // // 数据预处理结束
    // // 以下是FFT部分
    // /////////////////////////////////////////////////////
            //2dfft
            perform2DFFT(handle, queue,d_output,Na,Nr) ;

                // 拷贝结果回主机
            vector<complex<float>> result(Na * Nr);
            CNRT_CHECK(cnrtMemcpy(result.data(), d_output, echo_size, CNRT_MEM_TRANS_DIR_DEV2HOST));  
            CNRT_CHECK(cnrtQueueSync(queue));
            CNRT_CHECK(cnrtFree(d_output)); 
            CNRT_CHECK(cnrtQueueSync(queue));
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

            // 释放设备内存
            CNRT_CHECK(cnrtFree(d_theta_ft_fa));
            

            HostTimer grid_timer;
            float grid_time ;
            grid_timer.start();
            // 在点乘操作后添加：
             vector<complex<float>> stolt_result(Na * Nr);
            //performStoltInterpolation(handle, queue, result, stolt_result, fr_axis, fa_axis,f0, c, Vr, Na, Nr,fr_gap);
            StoltInterp_sinc(handle, queue, d_result, d_result, fr_axis, fa_axis, f0, c, Vr, Na, Nr, params.Fr,6);

            grid_timer.stop();
            
            grid_time= grid_timer.tv_usec;
            cout << "STOLT插值执行时间: " << grid_time/1000 << " ms" << endl;


            // //2D IFFT
            HostTimer Ifft_timer;
            float ifft_time ;
            Ifft_timer.start();
            perform2DIFFT(handle, queue,d_result,Na, Nr);
            Ifft_timer.stop();
            ifft_time = Ifft_timer.tv_usec;
            cout << "IFFT执行时间: " << ifft_time/1000 << " ms" << endl;

            // 拷贝结果回主机
            vector<complex<float>> ifft_result(Na * Nr);
            size_t ifft_size = Na * Nr * 2*sizeof(float);
            CNRT_CHECK(cnrtMemcpy(ifft_result.data(), d_result, ifft_size, CNRT_MEM_TRANS_DIR_DEV2HOST));

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
            

            CNRT_CHECK(cnrtFree(d_result));


    } catch (const std::exception& e) {
        cerr << "数据处理错误: " << e.what() << endl;
        return;
    }
    
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
    return 0;
}
