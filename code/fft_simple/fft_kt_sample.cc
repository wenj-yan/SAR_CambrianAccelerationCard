/*
SAR multioptest
@hmguo 2024.04.16

运算标准流程：
输入 2*4 复数向量
FFT                    输出  2*4  复数向量
与同维度的向量点乘     输出  2*4  复数向量
IFFT                   输出  2*4  复数向量
输出  2*4  复数向量

运行结果：

------------------step0: input_cpu complex signal----------------
                    (1,0) (1,0) (1,0) (1,0) 
                    (1,0) (1,0) (1,0) (1,0)

------------------step1: fft result------------------------------
                    (4,0) (0,0) (0,0) (0,0) 
                    (4,0) (0,0) (0,0) (0,0)
                    
------------------step2: dotmultiple result----------------------
                    (4,0) (0,0) (0,0) (0,0)
                    (4,0) (0,0) (0,0) (0,0)
                    
------------------step3: ifft result-----------------------------
                    (4,0) (4,0) (4,0) (4,0)
                    (4,0) (4,0) (4,0) (4,0)

*/

#include <time.h>
#include <string>
#include "string.h"
#include "tool.h"
#include "opencv2/opencv.hpp"
#include <complex>
#include <iostream> 
#include <complex.h>
using namespace std;

void initDevice(int &dev, cnrtQueue_t &queue, cnnlHandle_t &handle) {
  CNRT_CHECK(cnrtGetDevice(&dev));//获取设备对应的设备号
  CNRT_CHECK(cnrtSetDevice(dev));//绑定当前线程所使用的设备

  CNRT_CHECK(cnrtQueueCreate(&queue));//创建一个计算队列

  CNNL_CHECK(cnnlCreate(&handle));//创建一个Cambricon CNNL句柄。句柄将与当前线程所使用的设备绑定
  CNNL_CHECK(cnnlSetQueue(handle, queue));//将队列和Cambricon CNNL句柄绑定
}

/**
 * @brief 性能统计结构体
 * 用于记录各阶段的时间开销
 * 包括硬件计算时间、接口调用时间、端到端时间、数据传输时间等
 */
struct Optensor {
  float hardware_time = 0.0;    // NPU硬件计算时间
  float interface_time = 0.0;   // 主机端接口调用时间
  float end2end_time = 0.0;     // 端到端总执行时间
  float memcpyH2D_time = 0.0;   // Host到Device的数据拷贝时间(包括输入、权重、偏置)
  float memcpyD2H_time = 0.0;   // Device到Host的数据拷贝时间(输出结果)
};



void test_fft(){
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
    Optensor optensor;  //用于记录各阶段的时间

    //设计数据维度
  //  int in_height = 2;
   int in_height =500;
  //  int in_width = 4;
    int in_width = 128*1024;
    int dstHeight = in_height;
    int dstWidth = in_width;

    /////////////////////////////////////////////////////
    // Prepare in/output tensor descriptor 
    /////////////////////////////////////////////////////

    // Na*Nr COMPLEX_FLOAT
    int input_dim1[] = {in_height, in_width};
    //创建张量描述符
    cnnlTensorDescriptor_t NaNr_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&NaNr_desc));
    //CNNL_LAYOUT_ARRAY: 数组布局方式
    //CNNL_DTYPE_COMPLEX_FLOAT: 复数浮点数据类型
    //2: 数据维度
    //input_dim1: 数据维度数组
    CNNL_CHECK(cnnlSetTensorDescriptor(NaNr_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_COMPLEX_FLOAT, 2, input_dim1));
    // Na*Nr FLOAT use for complexdot, store abs and angle
    int abs_angle_dim1[] = {in_height, in_width};
    cnnlTensorDescriptor_t abs_angle_NaNr_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&abs_angle_NaNr_desc));
    CNNL_CHECK(cnnlSetTensorDescriptor(abs_angle_NaNr_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, abs_angle_dim1));
    
    //两个张量描述符的区别：
    //NaNr_desc: 用于存储和处理FFT运算的输入输出数据
    //abs_angle_NaNr_desc: 用于存储浮点数据，存储幅度和角度
    //第二个没有使用到



    /////////////////////////////////////////////////////
    // Prepare function layer descriptor
    /////////////////////////////////////////////////////
  
    CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(NaNr_desc, CNNL_DTYPE_FLOAT));

    //////////////// cnnl Layer1: The descriptor of FFT function that holds cnnlExecFFT  //////////////////////////////////////////////
    size_t fft1_workspace_size = 0;
    size_t *fft1_workspace_size_pt = &fft1_workspace_size;
    void *fft1_workspace = nullptr;
    size_t fft1_reservespace_size = 0;
    size_t *fft1_reservespace_size_pt = &fft1_reservespace_size;
    void *fft1_reservespace = nullptr;
    
    //配置FFT计划
    cnnlFFTPlan_t fft1_desc;
    CNNL_CHECK(cnnlCreateFFTPlan(&fft1_desc));
    CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(NaNr_desc, CNNL_DTYPE_FLOAT));
    CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(NaNr_desc, CNNL_DTYPE_FLOAT));
    //配置FFT变换参数
    int rank1 = 1;//1维FFT变换
    int fft1_n[] = {in_width};//FFT变换长度为输入宽度
    //初始化FFT计划
    // Initializes the FFT descriptor
    CNNL_CHECK(cnnlMakeFFTPlanMany(handle, fft1_desc, NaNr_desc, NaNr_desc, rank1, fft1_n, fft1_reservespace_size_pt, fft1_workspace_size_pt));
    // cout << " fft1_workspace_size: " << fft1_workspace_size << ", " << " fft1_reservespace_size: " << fft1_reservespace_size << endl; 

    // set Workspace
    if (fft1_workspace_size > 0) {
        CNRT_CHECK(cnrtMalloc((void **)&fft1_workspace, fft1_workspace_size));
    }
    // set Reservespace
    if (fft1_reservespace_size > 0) {
        CNRT_CHECK(cnrtMalloc((void **)&fft1_reservespace, fft1_reservespace_size));
    }
    CNNL_CHECK(cnnlSetFFTReserveArea(handle, fft1_desc, fft1_reservespace));



    void* input; // Na*Nr
    void* output1; // Na*Nr


    // cnrt: malloc input and output

    size_t input_length = in_height * in_width;
    size_t output_length = dstHeight * dstWidth;

    // cout << "------------- input length:" << input_length << "--------------" << endl;
    // cout << "------------- output length:" << output_length << "--------------" << endl;
    //分配内存
    CNRT_CHECK(cnrtMalloc((void**)&input, 2 * input_length * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void**)&output1, 2 * output_length * sizeof(float)));


    complex<float>* input_cpu = (complex<float> *)malloc(input_length * sizeof(complex<float>));
    complex<float>* output_cpu1 = (complex<float> *)malloc(output_length * sizeof(complex<float>));


    //input 数据赋值
    cout << "------------------step0: input_cpu complex signal----------------" << endl;
    for (size_t j = 0; j < input_length; ++j) { 
        reinterpret_cast<float*>(input_cpu)[2*j] = 1.0f;
        reinterpret_cast<float*>(input_cpu)[2*j+1] = 0.0f;
        }
    //  for (size_t i = 0; i < input_length; ++i) {
    //      cout << input_cpu[i] << " ";
    //  }
    //  cout << endl;

   
    //////////////////////////////////////////////////////
    // Start compute
    /////////////////////////////////////////////////////
    HostTimer copyin_timer;
    copyin_timer.start();
    //数据拷贝到设备
    CNRT_CHECK(cnrtMemcpy(input, input_cpu, 2 * input_length * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV));
    copyin_timer.stop();
    optensor.memcpyH2D_time = copyin_timer.tv_usec;

    HostTimer interface_timer;
    cnrtNotifier_t start = nullptr, end = nullptr;
    cnrtNotifierCreate(&start);
    cnrtNotifierCreate(&end);
    cnrtPlaceNotifier(start, queue);
 
    //Inference section
    
    // Step 1: FFT along the second dimension (Na)
    cnnlExecFFT(handle, fft1_desc, input, 1.0, fft1_workspace, output1, 0);
    
    cnrtPlaceNotifier(end, queue);
    //同步队列
    CNRT_CHECK(cnrtQueueSync(queue));
    cnrtNotifierDuration(start, end, &(optensor.hardware_time));
    CNRT_CHECK(cnrtNotifierDestroy(start));
    CNRT_CHECK(cnrtNotifierDestroy(end)); 
    // cnrt: memcpy D2H
    HostTimer copyout_timer;
    copyout_timer.start();
    CNRT_CHECK(cnrtMemcpy(output_cpu1, output1, 2 * output_length * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
    copyout_timer.stop();
    optensor.memcpyD2H_time = copyout_timer.tv_usec;
    
   cout << "!! ------------------step1: fft result----------------" << endl;
    for (size_t i = 0; i < output_length; ++i) {
        cout << output_cpu1[i] << " ";
    }
    cout << endl;
    free(output_cpu1);
    output_cpu1 = NULL;

    cout << "------------------Compute time----------------" << endl;
    // print compute time
    std::stringstream end2end_time;
    end2end_time << "All Time(ms):" << (optensor.hardware_time + optensor.memcpyH2D_time + optensor.memcpyD2H_time  )/ 1000;
    LOG(end2end_time.str());

    std::stringstream hardware_time;
    hardware_time << "npu compute Time(ms):" << optensor.hardware_time / 1000;
    LOG(hardware_time.str());

    std::stringstream copyin_time;
    copyin_time << "Copy data Host2Device Time(ms):" << optensor.memcpyH2D_time / 1000;
    LOG(copyin_time.str());

    std::stringstream copyout_time;
    copyout_time << "Copy data Device2Host Time(ms):" << optensor.memcpyD2H_time / 1000;
    LOG(copyout_time.str());


  

    //释放内存
    CNNL_CHECK(cnnlDestroyTensorDescriptor(NaNr_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(abs_angle_NaNr_desc));
    CNNL_CHECK(cnnlDestroyFFTPlan(fft1_desc));
     // cnrt: free
    if (fft1_workspace) {
        CNRT_CHECK(cnrtFree(fft1_workspace));
        CNRT_CHECK(cnrtFree(fft1_reservespace));
    }

    CNRT_CHECK(cnrtFree(input));
    CNRT_CHECK(cnrtFree(output1));

    free(input_cpu);
    input_cpu = NULL;

    free(output_cpu1);
    output_cpu1 = NULL;

    // cnnl: free
    CNNL_CHECK(cnnlDestroy(handle));

    // cnrt: free
    CNRT_CHECK(cnrtQueueSync(queue));


    //cnnlDestroyActivationDescriptor
    printf("========= test fft success !!! ========= \n");

}
int main()
{
    test_fft();
    return 0;
}