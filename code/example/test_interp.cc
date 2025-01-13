#include <time.h>
#include <string>
#include "string.h"
#include "tool.h"
#include <complex>
#include <iostream>
#include <vector>

using namespace std;

void initDevice(int &dev, cnrtQueue_t &queue, cnnlHandle_t &handle) {
    CNRT_CHECK(cnrtGetDevice(&dev));
    CNRT_CHECK(cnrtSetDevice(dev));
    CNRT_CHECK(cnrtQueueCreate(&queue));
    CNNL_CHECK(cnnlCreate(&handle));
    CNNL_CHECK(cnnlSetQueue(handle, queue));
}

// 简化后的插值函数
void performInterpolation(cnnlHandle_t handle, cnrtQueue_t queue,
                         const vector<float>& input, vector<float>& output) {
    cout << "\n执行线性插值..." << endl;
    
    // 创建插值描述符
    cnnlInterpDescriptor_t interp_desc;
    CNNL_CHECK(cnnlCreateInterpDescriptor(&interp_desc));
    
    // 设置插值描述符
    CNNL_CHECK(cnnlSetInterpDescriptor(interp_desc, 
                                      CNNL_INTERP_LINEAR,  // 使用线性插值
                                      CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO2)); // align_corners=true

    // 创建输入输出张量描述符
    cnnlTensorDescriptor_t input_desc, output_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&input_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&output_desc));

    // 设置输入输出张量维度
    int input_dims[] = {1, 5, 1};  // [N, W, C]
    int output_dims[] = {1, 81, 1}; // [N, W, C] 
    
    CNNL_CHECK(cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NLC, CNNL_DTYPE_FLOAT, 3, input_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NLC, CNNL_DTYPE_FLOAT, 3, output_dims));

    // 设置插值描述符的额外参数
    float scale_factor = 81.0f / 5.0f;
    float scale_factors[] = {scale_factor};
    
    CNNL_CHECK(cnnlSetInterpDescriptorEx(interp_desc,
                                        input_desc,
                                        CNNL_INTERP_ROUND_PERFER_CEIL,
                                        scale_factors,
                                        nullptr,
                                        -0.75f,
                                        false));

    // 分配设备内存和主机内存
    void *d_input, *d_output;
    size_t input_size = 5 * sizeof(float);
    size_t output_size = 81 * sizeof(float);
    
    CNRT_CHECK(cnrtMalloc(&d_input, input_size));
    CNRT_CHECK(cnrtMalloc(&d_output, output_size));

    // 使用const_cast去除const限定
    CNRT_CHECK(cnrtMemcpy(d_input, const_cast<void*>(static_cast<const void*>(input.data())), 
                         input_size, CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 执行插值
    CNNL_CHECK(cnnlInterp_v3(handle,
                            interp_desc,
                            input_desc,
                            d_input,
                            output_desc,
                            d_output));
                            
    CNRT_CHECK(cnrtQueueSync(queue));

    // 拷贝结果回主机
    CNRT_CHECK(cnrtMemcpy(output.data(), d_output, output_size, CNRT_MEM_TRANS_DIR_DEV2HOST));

    // 清理资源
    CNRT_CHECK(cnrtFree(d_input));
    CNRT_CHECK(cnrtFree(d_output));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(input_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(output_desc));
    CNNL_CHECK(cnnlDestroyInterpDescriptor(interp_desc));

    cout << "插值完成" << endl;
}

int main() {
    // 初始化设备
    int dev;
    cnrtQueue_t queue = nullptr;
    cnnlHandle_t handle = nullptr;
    initDevice(dev, queue, handle);

    // 准备输入数据
    vector<float> input = {2, 6, 4, 8, 5};  // 你提供的y坐标值
    vector<float> output(81, 0);  // 输出数组,大小为81(1到9,步长0.1)

    // 执行插值
    performInterpolation(handle, queue, input, output);

    // 打印部分结果
    cout << "\n插值结果(部分):" << endl;
    for(int i = 0; i < 81; i++) {  // 只打印前10个结果
        cout << output[i] << " ";
    }
    cout << "..." << endl;

    // 清理资源
    CNNL_CHECK(cnnlDestroy(handle));
    CNRT_CHECK(cnrtQueueDestroy(queue));

    return 0;
} 