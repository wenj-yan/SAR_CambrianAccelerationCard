#include <time.h>
#include <string>
#include "string.h"
#include "tool.h"
#include <complex>
#include <iostream> 
#include <complex.h>

using namespace std;

void initDevice(int &dev, cnrtQueue_t &queue, cnnlHandle_t &handle) {
    CNRT_CHECK(cnrtGetDevice(&dev));
    CNRT_CHECK(cnrtSetDevice(dev));
    CNRT_CHECK(cnrtQueueCreate(&queue));
    CNNL_CHECK(cnnlCreate(&handle));
    CNNL_CHECK(cnnlSetQueue(handle, queue));
}

void SplitComplexTensor(cnnlHandle_t handle,
                       cnrtQueue_t queue,
                       const complex<float>* complex_data,
                       int Na,
                       int Nr,
                       float* real_part,
                       float* imag_part)
{
    // 创建Host计时器
    HostTimer h2d_timer, d2h_timer, total_timer;
    
    // 创建设备计时器用于计算时间
    cnrtNotifier_t start_compute = nullptr, end_compute = nullptr;
    CNRT_CHECK(cnrtNotifierCreate(&start_compute));
    CNRT_CHECK(cnrtNotifierCreate(&end_compute));
    
    // 开始总计时
    total_timer.start();

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

    // 3. 分配设备内存
    void *d_input, *d_real, *d_imag;
    CNRT_CHECK(cnrtMalloc(&d_input, Na * Nr * 2 * sizeof(float)));
    CNRT_CHECK(cnrtMalloc(&d_real, Na * Nr * sizeof(float)));
    CNRT_CHECK(cnrtMalloc(&d_imag, Na * Nr * sizeof(float)));

    // 4. 创建临时缓冲区并拷贝数据
    void* temp_buffer = (void*)complex_data;
    CNRT_CHECK(cnrtMemcpy(d_input, temp_buffer, Na * Nr * 2 * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 5. 获取工作空间大小
    size_t workspace_size = 0;
    CNNL_CHECK(cnnlGetSplitWorkspaceSize(handle, 2, &workspace_size));
    
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc(&workspace, workspace_size));
    }

    // 6. 准备输出数组
    void* outputs[] = {d_real, d_imag};

    // Host->Device传输时间
    h2d_timer.start();
    CNRT_CHECK(cnrtMemcpy(d_input, temp_buffer, Na * Nr * 2 * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV));
    h2d_timer.stop();

    // 计算时间
    CNRT_CHECK(cnrtPlaceNotifier(start_compute, queue));
    CNNL_CHECK(cnnlSplit(handle, 
                        2,
                        1,
                        input_desc, 
                        d_input,
                        workspace,
                        workspace_size,
                        output_desc,
                        outputs));
    CNRT_CHECK(cnrtPlaceNotifier(end_compute, queue));

    // Device->Host传输时间
    d2h_timer.start();
    CNRT_CHECK(cnrtMemcpy(real_part, d_real, Na * Nr * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(imag_part, d_imag, Na * Nr * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
    d2h_timer.stop();

    // 结束总计时
    total_timer.stop();
    
    // 同步队列以确保所有操作完成
    CNRT_CHECK(cnrtQueueSync(queue));
    
    // 计算设备计算时间
    float compute_time;
    CNRT_CHECK(cnrtNotifierDuration(start_compute, end_compute, &compute_time));

    // 打印各阶段时间
    cout << "\n时间统计:" << endl;
    cout << "Host->Device 传输时间: " << h2d_timer.tv_usec << " us" << endl;
    cout << "计算时间: " << compute_time / 1000.0 << " ms" << endl;
    cout << "Device->Host 传输时间: " << d2h_timer.tv_usec << " us" << endl;
    cout << "总时间: " << total_timer.tv_usec << " us" << endl;

    // 清理计时器
    CNRT_CHECK(cnrtNotifierDestroy(start_compute));
    CNRT_CHECK(cnrtNotifierDestroy(end_compute));

    // 10. 清理资源
    CNNL_CHECK(cnnlDestroyTensorDescriptor(input_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc[0]));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc[1]));
    
    if (workspace) {
        CNRT_CHECK(cnrtFree(workspace));
    }
    CNRT_CHECK(cnrtFree(d_input));
    CNRT_CHECK(cnrtFree(d_real));
    CNRT_CHECK(cnrtFree(d_imag));
}




int main() {
    // 初始化设备
    int dev;
    cnrtQueue_t queue = nullptr;
    cnnlHandle_t handle = nullptr;
    initDevice(dev, queue, handle);

    // 测试参数
    int Na = 5;
    int Nr = 5;

    // 准备复数数据 [2,3]
    complex<float> complex_data[] = {
        {1.0f, 0.1f}, {2.0f, 0.2f}, {3.0f, 0.3f}, {4.0f, 0.4f   }, {5.0f, 0.5f},  // 第一行
        {6.0f, 0.6f}, {7.0f, 0.7f}, {8.0f, 0.8f}, {9.0f, 0.9f}, {10.0f, 1.0f},  // 第二行
        {11.0f, 1.1f}, {12.0f, 1.2f}, {13.0f, 1.3f}, {14.0f, 1.4f}, {15.0f, 1.5f},  // 第三行
        {16.0f, 1.6f}, {17.0f, 1.7f}, {18.0f, 1.8f}, {19.0f, 1.9f}, {20.0f, 2.0f},  // 第四行
        {21.0f, 2.1f}, {22.0f, 2.2f}, {23.0f, 2.3f}, {24.0f, 2.4f}, {25.0f, 2.5f}   // 第五行
    };

    // 打印原始复数数据
    cout << "原始复数数据:" << endl;
    for(int i = 0; i < Na; i++) {
        for(int j = 0; j < Nr; j++) {
            cout << complex_data[i * Nr + j].real() << "+" << complex_data[i * Nr + j].imag() << "i ";
        }
        cout << endl;
    }

    float real_part[25];  // [5,5]
    float imag_part[25];  // [5,5]

    SplitComplexTensor(handle, queue, complex_data, Na, Nr, real_part, imag_part);

    // 打印结果
    cout << "Real part:" << endl;
    for(int i = 0; i < Na; i++) {
        for(int j = 0; j < Nr; j++) {
            cout << real_part[i * Nr + j] << " ";
        }
        cout << endl;
    }

    cout << "Imaginary part:" << endl;
    for(int i = 0; i < Na; i++) {
        for(int j = 0; j < Nr; j++) {
            cout << imag_part[i * Nr + j] << " ";
        }
        cout << endl;
    }

    // 清理资源
    CNNL_CHECK(cnnlDestroy(handle));
    CNRT_CHECK(cnrtQueueDestroy(queue));

    return 0;
} 