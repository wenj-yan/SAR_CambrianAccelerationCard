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

int main() {
    // 初始化设备
    int dev;
    cnrtQueue_t queue = nullptr;
    cnnlHandle_t handle = nullptr;
    initDevice(dev, queue, handle);

    // 测试参数
    int Na = 2;
    int Nr = 3;
    int pad_before_Na = 2;
    int pad_after_Na = 2; 
    int pad_before_Nr = 2;
    int pad_after_Nr = 2;
    complex<float> padding_value(0.0f, 0.0f);

    // 准备输入数据
    complex<float> input_data[] = {
        {1.0f, 0.1f}, {2.0f, 0.2f}, {3.0f, 0.3f},  // 第一行
        {4.0f, 0.4f}, {5.0f, 0.5f}, {6.0f, 0.6f}   // 第二行
    };

    // 分配设备内存
    void *d_input, *d_output;
    size_t input_size = Na * Nr * 2 * sizeof(float);
    size_t output_size = (Na + pad_before_Na + pad_after_Na) * 
                        (Nr + pad_before_Nr + pad_after_Nr) * 
                        2 * sizeof(float);
    
    CNRT_CHECK(cnrtMalloc(&d_input, input_size));
    CNRT_CHECK(cnrtMalloc(&d_output, output_size));

    // 拷贝输入数据到设备
    CNRT_CHECK(cnrtMemcpy(d_input, input_data, input_size, CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 执行pad操作
    PadComplexTensor(handle, queue, d_input, Na, Nr,
                    pad_before_Na, pad_after_Na,
                    pad_before_Nr, pad_after_Nr,
                    padding_value, d_output);

    // 分配主机内存接收结果
    int out_Na = Na + pad_before_Na + pad_after_Na;
    int out_Nr = Nr + pad_before_Nr + pad_after_Nr;
    complex<float> output_data[out_Na * out_Nr];

    // 拷贝结果回主机
    CNRT_CHECK(cnrtMemcpy(output_data, d_output, output_size, CNRT_MEM_TRANS_DIR_DEV2HOST));

    // 打印结果
    cout << "Padded output:" << endl;
    for(int i = 0; i < out_Na; i++) {
        for(int j = 0; j < out_Nr; j++) {
            cout << output_data[i * out_Nr + j].real() << "+" 
                 << output_data[i * out_Nr + j].imag() << "i ";
        }
        cout << endl;
    }

    // 清理资源
    CNRT_CHECK(cnrtFree(d_input));
    CNRT_CHECK(cnrtFree(d_output));
    CNNL_CHECK(cnnlDestroy(handle));
    CNRT_CHECK(cnrtQueueDestroy(queue));

    return 0;
} 