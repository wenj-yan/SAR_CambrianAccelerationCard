#include <time.h>
#include <string>
#include "string.h"
#include "tool.h"
#include <complex>
#include <iostream>
#include <complex.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>  

using namespace std;

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

void process_test(){
    // 初始化设备
    int dev;
    cnrtQueue_t queue = nullptr;
    cnnlHandle_t handle = nullptr;

    initDevice(dev, queue, handle);

    // cnrt: time
    //创建性能计时器
    cnrtNotifier_t e_t0, e_t1;
    cnrtNotifierCreate(&e_t0);
    cnrtNotifierCreate(&e_t1);
    Optensor optensor;  

// 创建张量描述符
    cnnlTensorDescriptor_t grid_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&grid_desc));

    // 设置网格维度 
    int Na = 10;
    int Nr = 10;
    int grid_dims[] = {Na, Nr};
    CNNL_CHECK(cnnlSetTensorDescriptor(grid_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, grid_dims));
    // 设置onchip数据类型
    CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(grid_desc, CNNL_DTYPE_FLOAT));
    // 分配设备内存
    void *d_temp1, *d_temp2;
    size_t grid_size = Na * Nr * sizeof(float);
    CNRT_CHECK(cnrtMalloc(&d_temp1, grid_size));
    CNRT_CHECK(cnrtMalloc(&d_temp2, grid_size));

    // 创建并填充测试数据
    vector<float> data_test(Na * Nr);
    for(int i = 0; i < Na * Nr; i++) {
        data_test[i] = i % 100; 
    }
    cout.setf(ios::fixed);              // 使用定点表示法
    cout.unsetf(ios::scientific);       // 禁用科学计数法
    data_test[0] = 28068548338144772096;
    //data_test[0] = 2806854.8338144772096;
    cout << "原始数据值：" << endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << data_test[i * Nr + j] << " ";
        }
        cout << endl;
    }
    
    // 转换为double类型
    vector<double> data_double(Na * Nr);
    for (int i = 0; i < Na * Nr; i++) {
        data_double[i] = static_cast<double>(data_test[i]);
    }
    
    cout << "转换为double后的数据：" << endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << data_double[i * Nr + j] << " ";
        }
        cout << endl;
    }
    
    // 计算开方
    vector<double> data_sqrt(Na * Nr);
    for (int i = 0; i < Na * Nr; i++) {
        data_sqrt[i] = sqrt(data_double[i]);
    }
    
    cout << "开方后的数据：" << endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << data_sqrt[i * Nr + j] << " ";
        }
        cout << endl;
    }
    CNRT_CHECK(cnrtMemcpy(d_temp1, data_test.data(), grid_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNNL_CHECK(cnnlSqrt_v2(handle,
                        CNNL_COMPUTATION_HIGH_PRECISION,
                        grid_desc,
                        d_temp1,
                        grid_desc,
                        d_temp2));

    CNRT_CHECK(cnrtQueueSync(queue));
    CNRT_CHECK(cnrtMemcpy(data_test.data(), d_temp2, grid_size, CNRT_MEM_TRANS_DIR_DEV2HOST));    
    cout << "结果数据值：" << endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << data_test[i * Nr + j] << " ";
        }
        cout << endl;
    }

    // 释放内存和资源
    CNNL_CHECK(cnnlDestroyTensorDescriptor(grid_desc));
    CNRT_CHECK(cnrtFree(d_temp1));
    CNRT_CHECK(cnrtFree(d_temp2));
    CNRT_CHECK(cnrtNotifierDestroy(e_t0));
    CNRT_CHECK(cnrtNotifierDestroy(e_t1));
    CNNL_CHECK(cnnlDestroy(handle));
    CNRT_CHECK(cnrtQueueDestroy(queue));
}


int main() {
    process_test();
    return 0;
} 