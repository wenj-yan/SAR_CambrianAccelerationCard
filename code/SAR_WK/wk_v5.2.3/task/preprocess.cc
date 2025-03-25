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
#include <unordered_set>


#include "task.h"

void preprocess_wk(cnnlHandle_t handle, cnrtQueue_t queue,
                    void *d_input,size_t Na, size_t Nr,size_t Na_padded,size_t Nr_padded,
                    WKParams params,void *d_output)
{
    int pad_before_Na = round(Na/6.0f);
    int pad_after_Na = round(Na/6.0f);
    int pad_before_Nr = round(Nr/3.0f); 
    int pad_after_Nr = round(Nr/3.0f);
    void  *d_output2;
    size_t output_size = Na_padded * Nr_padded * 2 * sizeof(float);
    CNRT_CHECK(cnrtMalloc(&d_output2, output_size));
            // 执行pad操作
    PadComplexTensor(handle, queue, d_input, Na, Nr,
                    pad_before_Na, pad_after_Na,
                    pad_before_Nr, pad_after_Nr,
                    complex<float>(0.0f, 0.0f), d_output2);
    CNRT_CHECK(cnrtQueueSync(queue));                
//////////////////////////////////////////////////////
// 方位向变频（频谱搬移）
/////////////////////////////////////////////////////
    // 生成方位向时间轴
    float Fa = params.PRF;  // 方位向采样率等于PRF
    float f_nc = params.f_nc;  // 多普勒中心频率
    float xishu = -2 * M_PI * f_nc;
    vector<float> ta_axis(Na_padded);
    vector<float> phase(Na_padded);
    float ta_step = 1.0f / Fa;
    for (size_t i = 0; i < Na_padded; ++i) {
        ta_axis[i] = (static_cast<float>(i) - Na_padded/2) * ta_step ;
        phase[i] = ta_axis[i] * xishu;
    }

    void * d_sin,*d_cos,*d_phase;
    size_t na_size = Na_padded * sizeof(float);

    CNRT_CHECK(cnrtMalloc(&d_sin, na_size));
    CNRT_CHECK(cnrtMalloc(&d_cos, na_size));
    CNRT_CHECK(cnrtMalloc(&d_phase, na_size));
    CNRT_CHECK(cnrtMemcpy(d_phase, phase.data(), na_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtQueueSync(queue));
    
    cnnlTensorDescriptor_t Na_float_desc,Na_complex_desc,Na_Nr_complex_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&Na_float_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&Na_complex_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&Na_Nr_complex_desc));
    int dims_Na[3] = {Na_padded,1,1};
    int dims_Na_2[3] = {Na_padded,1,2};
    int dims_Na_Nr_2[3] = {Na_padded,Nr_padded,2};
    CNNL_CHECK(cnnlSetTensorDescriptor(Na_float_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, dims_Na));
    CNNL_CHECK(cnnlSetTensorDescriptor(Na_complex_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, dims_Na_2));
    CNNL_CHECK(cnnlSetTensorDescriptor(Na_Nr_complex_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, dims_Na_Nr_2));
    CNNL_CHECK(cnnlCos_v2(handle,CNNL_COMPUTATION_HIGH_PRECISION, 
                            Na_float_desc, d_phase, 
                            Na_float_desc, d_cos));
    CNRT_CHECK(cnrtQueueSync(queue));
    CNNL_CHECK(cnnlSin_v2(handle,CNNL_COMPUTATION_HIGH_PRECISION, 
                            Na_float_desc, d_phase, 
                            Na_float_desc, d_sin));
    CNRT_CHECK(cnrtQueueSync(queue));
    ///////////////////////////////////////////////////////
    // vector<float> sin_vals(Na_padded);
    // vector<float> cos_vals(Na_padded);
    
    // // 在CPU上计算sin和cos值
    // for (size_t i = 0; i < Na_padded; ++i) {
    //     cos_vals[i] = cos(phase[i]);
    //     sin_vals[i] = sin(phase[i]);
    // }
    // // 将计算好的值拷贝到设备内存
    // CNRT_CHECK(cnrtMemcpy(d_cos, cos_vals.data(), na_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    // CNRT_CHECK(cnrtMemcpy(d_sin, sin_vals.data(), na_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    // CNRT_CHECK(cnrtQueueSync(queue));
    //////////////////////////////////////////////////////
    void *d_phase_complex,*d_phase_expand;
    size_t phase_size = Na_padded * sizeof(complex<float>);
    CNRT_CHECK(cnrtMalloc(&d_phase_complex, phase_size));  
    size_t echo_size = Na_padded * Nr_padded * sizeof(complex<float>);
    CNRT_CHECK(cnrtMalloc(&d_phase_expand, echo_size));
    CombineComplexTensor(handle,queue,
                        d_cos,        
                        d_sin,        
                        Na_padded,             
                        1,              
                        d_phase_complex);    
    CNRT_CHECK(cnrtQueueSync(queue));
    
    CNNL_CHECK(cnnlExpand(handle, 
                            Na_complex_desc, d_phase_complex, 
                            Na_Nr_complex_desc, d_phase_expand));
    CNRT_CHECK(cnrtQueueSync(queue));  
///////////////////////////////////////////
//     // 添加调试代码
// vector<complex<float>> debug_phase_expand(Na_padded * Nr_padded);
// CNRT_CHECK(cnrtMemcpy(debug_phase_expand.data(), d_phase_expand, 
//            Na_padded * Nr_padded * sizeof(complex<float>), 
//            CNRT_MEM_TRANS_DIR_DEV2HOST));

// // 打印部分值进行验证
// cout << "Expanded phase values:" << endl;
// for (size_t i = 0; i < min(size_t(5), Na_padded); ++i) {
//     for (size_t j = 0; j < min(size_t(5), Nr_padded); ++j) {
//         complex<float> val = debug_phase_expand[i * Nr_padded + j];
//         cout << "(" << i << "," << j << "): " 
//              << val.real() << " + " << val.imag() << "j" << endl;
//     }
// }
//////////////////////////////////////////
    // cnnlTensorDescriptor_t inputs_desc[] = {Na_Nr_complex_desc,Na_Nr_complex_desc};    
    // void *const inputs[]= {d_output2,d_phase_expand};
    // CNNL_CHECK(cnnlMulN(handle, inputs_desc,inputs, 2, Na_Nr_complex_desc, d_output));  
    // CNRT_CHECK(cnrtQueueSync(queue)); 
    ComplexMatrixMultiply(handle,queue,d_output2,d_phase_expand,d_output,Na_padded,Nr_padded);

///////////////////////////////////////////
// // 添加乘法结果验证代码
// vector<complex<float>> debug_multiply_result(Na_padded * Nr_padded);
// vector<complex<float>> debug_input(Na_padded * Nr_padded);

// // 拷贝输入和输出数据到主机内存
// CNRT_CHECK(cnrtMemcpy(debug_multiply_result.data(), d_output,
//            Na_padded * Nr_padded * sizeof(complex<float>),
//            CNRT_MEM_TRANS_DIR_DEV2HOST));
// CNRT_CHECK(cnrtMemcpy(debug_input.data(), d_output2,
//            Na_padded * Nr_padded * sizeof(complex<float>),
//            CNRT_MEM_TRANS_DIR_DEV2HOST));

// // 打印部分结果进行验证
// cout << "乘法运算结果验证:" << endl;
// for (size_t i = 0; i < min(size_t(3), Na_padded); ++i) {
//     for (size_t j = 0; j < min(size_t(3), Nr_padded); ++j) {
//         size_t idx = i * Nr_padded + j;
//         complex<float> input_val = debug_input[idx];
//         complex<float> phase_val = debug_phase_expand[idx];
//         complex<float> result_val = debug_multiply_result[idx];
        
//         cout << "位置(" << i << "," << j << "):" << endl;
//         cout << "  输入值: " << input_val.real() << " + " << input_val.imag() << "j" << endl;
//         cout << "  相位值: " << phase_val.real() << " + " << phase_val.imag() << "j" << endl;
//         cout << "  结果值: " << result_val.real() << " + " << result_val.imag() << "j" << endl;
//         cout << "-----------------" << endl;
//     }
// }
//////////////////////////////////////////

    CNRT_CHECK(cnrtFree(d_output2));    
    CNRT_CHECK(cnrtFree(d_sin));
    CNRT_CHECK(cnrtFree(d_cos));
    CNRT_CHECK(cnrtFree(d_phase));
    CNRT_CHECK(cnrtFree(d_phase_complex));
    CNRT_CHECK(cnrtFree(d_phase_expand));
}