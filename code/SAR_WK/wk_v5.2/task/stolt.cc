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
    // int dims_Na_Nr[4] = {batch_size, height, width, channels};
    // CNNL_CHECK(cnnlSetTensorDescriptor(roll_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_COMPLEX_FLOAT, 4, dims_Na_Nr));
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
        for (int w = 0; w <Nr; w++) {
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




//矩阵运算实现stolt插值——sinc插值
void StoltInterp_sinc(cnnlHandle_t handle, 
                             cnrtQueue_t queue,
                             void * input,
                             vector<complex<float>>& output,
                             const vector<double>& fr_axis,
                             const vector<double>& fa_axis,
                             double f0, double c, double Vr,
                             int Na, int Nr,double fr,int P)
{
    cout << "\n执行Stolt插值V2(sinc)..." << endl;
    // 创建性能计时
    cnrtNotifier_t start = nullptr, end = nullptr;
    cnrtNotifierCreate(&start);
    cnrtNotifierCreate(&end);
    HostTimer Host_Timer;
    float host_time;
    float hardware_time;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// 1. 计算新频率映射////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(G_DEBUG == true){
        cnrtPlaceNotifier(start, queue);
        Host_Timer.start();
    }
    vector<float> fr_new_mtx = calculateNewFrequencyMapping(fr_axis, fa_axis, f0, c, Vr, Na, Nr);

    if(G_DEBUG == true){
        float fr_new_min = *min_element(fr_new_mtx.begin(), fr_new_mtx.end());
        float fr_new_max = *max_element(fr_new_mtx.begin(), fr_new_mtx.end());
        float fr_min = *min_element(fr_axis.begin(), fr_axis.end());
        float fr_max = *max_element(fr_axis.begin(), fr_axis.end());
        cout << "fr_new_mtx范围: [" << fr_new_min << ", " << fr_new_max << "]" << endl;
        cout << "fr_axis范围: [" << fr_min << ", " << fr_max << "]" << endl;

        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));  
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "计算频率映射执行时间(hardware): " << hardware_time / 1000 << " ms" << endl;
        Host_Timer.stop();
        host_time=Host_Timer.tv_usec;
        cout << "计算频率映射执行时间(host): " << host_time/1000 << " ms" << endl;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// 2. 计算每个点的原始坐标偏移量delta_n  delta_n=((f_tao_pie-f_tao_mtx)/Fr*Nr/2);////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    //创建操作描述符
    cnnlOpTensorDescriptor_t optensor_desc;
    CNNL_CHECK(cnnlCreateOpTensorDescriptor(&optensor_desc));//(cnnlOpTensorDescriptor_t *op_tensor_desc)
    CNNL_CHECK(cnnlSetOpTensorDescriptor(optensor_desc, 
                                       CNNL_OP_TENSOR_ADD, 
                                       CNNL_DTYPE_FLOAT, 
                                       CNNL_NOT_PROPAGATE_NAN));
    //设置输入输出tensor描述符-float
    cnnlTensorDescriptor_t Na_Nr_float_desc, Nr_float_desc,P_Na_Nr_float_desc,Na_Nr_2_float_desc,P_Na_Nr_2_float_desc,P_float_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&Na_Nr_float_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&Nr_float_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&P_Na_Nr_float_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&Na_Nr_2_float_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&P_Na_Nr_2_float_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&P_float_desc));
    //设置输入输出tensor描述符-int
    cnnlTensorDescriptor_t P_Na_Nr_int32_desc,P_Na_Nr_2_int32_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&P_Na_Nr_int32_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&P_Na_Nr_2_int32_desc));
    //设置维度
    int dims_Na_Nr[4] = {1,Na, Nr,1};
    int dims_Nr[4] = {1,1,Nr,1};
    int dims_p[4] = {P,1,1,1};
    int dims_P_Na_Nr[4] = {P,Na,Nr,1};
    int dims_P_Nr[4] = {P,1,Nr,1};
    int dims_Na_Nr_2[4] = {1,Na, Nr,2};
    int dims_P_Na_Nr_2[4] = {P,Na,Nr,2};

    CNNL_CHECK(cnnlSetTensorDescriptor(Na_Nr_float_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dims_Na_Nr));
    CNNL_CHECK(cnnlSetTensorDescriptor(Nr_float_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dims_Nr));
    CNNL_CHECK(cnnlSetTensorDescriptor(P_Na_Nr_float_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dims_P_Na_Nr));
    CNNL_CHECK(cnnlSetTensorDescriptor(Na_Nr_2_float_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dims_Na_Nr_2));
    CNNL_CHECK(cnnlSetTensorDescriptor(P_Na_Nr_2_float_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dims_P_Na_Nr_2));
    CNNL_CHECK(cnnlSetTensorDescriptor(P_float_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,4, dims_p));
    CNNL_CHECK(cnnlSetTensorDescriptor(P_Na_Nr_int32_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32,4,dims_P_Na_Nr));
    CNNL_CHECK(cnnlSetTensorDescriptor(P_Na_Nr_2_int32_desc,CNNL_LAYOUT_ARRAY,CNNL_DTYPE_INT32,4,dims_P_Na_Nr_2));
    //准备输入数据
    float alpha1 = 1.0f * Nr/fr  ;  //1.0f * Nr/fr 
    float alpha2 = -1.0f * Nr/fr ;  //-1.0f * Nr/fr
    float beta = 0.0f;
    float a = 1.0f;
    // 分配设备内存
    void *dev_fr_new_mtx, *dev_fr_axis, *dev_delta_n,*dev_deltan_expand,*dev_input,*dev_input2,*dev_gather_out,*center_idx;
    size_t data_size = Na * Nr * sizeof(float);
    size_t data_size_1d =  Nr * sizeof(float);
    size_t data_size_4d =  P*Na*Nr * sizeof(float);
    size_t data_size_mask =  P*Nr * sizeof(float);
    size_t data_size_complex = Na * Nr * 2 * sizeof(float);
    size_t data_size_complex_out = P*Na*Nr * 2 * sizeof(float);
    
    CNRT_CHECK(cnrtMalloc(&dev_fr_new_mtx, data_size));
    CNRT_CHECK(cnrtMalloc(&dev_fr_axis, data_size_1d));
    CNRT_CHECK(cnrtMalloc(&dev_delta_n, data_size));
    CNRT_CHECK(cnrtMalloc(&dev_deltan_expand, data_size_4d));
    CNRT_CHECK(cnrtMalloc(&center_idx, data_size_4d));
    CNRT_CHECK(cnrtMalloc(&dev_input,data_size_complex));
    CNRT_CHECK(cnrtMalloc(&dev_input2,data_size_complex_out));
    CNRT_CHECK(cnrtMalloc(&dev_gather_out,data_size_complex_out));



    //CNRT_CHECK(cnrtMemcpy(dev_input, (void*)input.data(), data_size_complex, CNRT_MEM_TRANS_DIR_HOST2DEV));
    dev_input = input;
    //数据类型转换
    if(G_DEBUG == true){
        cnrtPlaceNotifier(start, queue);
        Host_Timer.start();
    }
    vector<float> fr_axis_float(fr_axis.size());
    std::transform(fr_axis.begin(), fr_axis.end(), fr_axis_float.begin(),
    [](double val) { return static_cast<float>(val); });

    CNRT_CHECK(cnrtMemcpy(dev_fr_new_mtx, fr_new_mtx.data(), data_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(dev_fr_axis, (void*)fr_axis_float.data(), data_size_1d, CNRT_MEM_TRANS_DIR_HOST2DEV));
    if(G_DEBUG == true){
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));  
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "数据类型转换与搬移执行时间(hardware): " << hardware_time / 1000 << " ms" << endl;
        Host_Timer.stop();
        host_time=Host_Timer.tv_usec;
        cout << "数据类型转换与搬移执行时间(host): " << host_time/1000 << " ms" << endl;
    }
    if(G_DEBUG == true){
        cnrtPlaceNotifier(start, queue);
        Host_Timer.start();
    }
    // 获取workspace大小
    size_t workspace_size = 0;
    CNNL_CHECK(cnnlGetOpTensorWorkspaceSize(handle, Na_Nr_float_desc, Nr_float_desc, Na_Nr_float_desc, &workspace_size));
    void *dev_workspace;
    CNRT_CHECK(cnrtMalloc(&dev_workspace, workspace_size));
    CNNL_CHECK(cnnlOpTensor(handle, optensor_desc,
                          &alpha1, Na_Nr_float_desc, dev_fr_new_mtx,
                          &alpha2, Nr_float_desc, dev_fr_axis,
                          dev_workspace, workspace_size, &beta,
                          Na_Nr_float_desc, dev_delta_n));
    CNRT_CHECK(cnrtQueueSync(queue));
    CNRT_CHECK(cnrtFree(dev_fr_axis));
    CNRT_CHECK(cnrtFree(dev_workspace));
    if(G_DEBUG == true){
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));  
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "计算dev_delta_n执行时间:(hardware) " << hardware_time / 1000 << " ms" << endl;
        Host_Timer.stop();
        host_time=Host_Timer.tv_usec;
        cout << "计算dev_delta_n执行时间(host): " << host_time/1000 << " ms" << endl;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// //确定插值位置position=n-delta_n////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(G_DEBUG == true){
        cnrtPlaceNotifier(start, queue);
        Host_Timer.start();
    }
    float startP = 0;
    float step = 1;
    void *dev_NrOut,*dev_mask,*dev_mask_out,*signal_expand;
    CNRT_CHECK(cnrtMalloc(&dev_NrOut, data_size_1d));
    CNNL_CHECK(cnnlArange_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, 
                                &startP, &step, Nr_float_desc, dev_NrOut));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 获取workspace大小
    size_t workspace_size_position = 0;
    CNNL_CHECK(cnnlGetOpTensorWorkspaceSize(handle, Nr_float_desc, Na_Nr_float_desc, Na_Nr_float_desc, &workspace_size_position));
    void *dev_workspace_position;
    CNRT_CHECK(cnrtMalloc(&dev_workspace_position, workspace_size_position));
    alpha1 = 1.0f  ;  
    alpha2 = -1.0f ; 
    CNNL_CHECK(cnnlOpTensor(handle, optensor_desc,
                          &alpha1, Nr_float_desc, dev_NrOut,
                          &alpha2, Na_Nr_float_desc, dev_delta_n,
                          dev_workspace_position, workspace_size_position, &beta,
                          Na_Nr_float_desc, dev_delta_n));
    CNRT_CHECK(cnrtQueueSync(queue));
    CNRT_CHECK(cnrtFree(dev_workspace_position));
    if(G_DEBUG == true){
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));  
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "计算插值位置执行时间:(hardware) " << hardware_time / 1000 << " ms" << endl;
        Host_Timer.stop();
        host_time=Host_Timer.tv_usec;
        cout << "计算插值位置执行时间(host): " << host_time/1000 << " ms" << endl;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// // 3. 生成邻域索引和权重索引P[6]=[-3,-2,-1,0,1,2]，权重尺寸NrxP////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(G_DEBUG == true){
        cnrtPlaceNotifier(start, queue);
        Host_Timer.start();
    }
    // 定义张量数据
    //float data[6] = {-3, -2, -1, 0, 1, 2};  // 输入数据
    // 分配设备内存
    void* dev_Pdata = nullptr;
    CNRT_CHECK(cnrtMalloc(&dev_Pdata, P * sizeof(float)));
    //直接生成P_DATA
    startP = P*0.5f*(-1);
    step = 1;
    CNNL_CHECK(cnnlArange_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, 
        &startP, &step, P_float_desc, dev_Pdata));
    CNRT_CHECK(cnrtQueueSync(queue));
    // 将数据拷贝到设备
    CNRT_CHECK(cnrtMalloc(&dev_mask, data_size_4d));
    CNRT_CHECK(cnrtMalloc(&dev_mask_out, data_size_4d));
    CNRT_CHECK(cnrtMalloc(&signal_expand, data_size_4d));
    
    //生成Nr*P，并且进行mask掩膜计算
    size_t workspace_size_mask = 0;
    cnnlTensorDescriptor_t descs[] = {Na_Nr_float_desc, P_float_desc};
    CNNL_CHECK(cnnlGetAddNWorkspaceSize(handle, descs, 2, P_Na_Nr_float_desc, &workspace_size_mask));
    void *dev_workspace_mask;
    CNRT_CHECK(cnrtMalloc(&dev_workspace_mask, workspace_size_mask));
    
    void* inputs[] = {dev_delta_n, dev_Pdata};
    CNNL_CHECK(cnnlAddN_v2(handle, descs, inputs, 
                            2, P_Na_Nr_float_desc, dev_mask, 
                            dev_workspace_mask, workspace_size_mask));
    CNRT_CHECK(cnrtQueueSync(queue));
    CNRT_CHECK(cnrtFree(dev_workspace_mask));  

    CNNL_CHECK(cnnlFloor(handle, P_Na_Nr_float_desc, dev_mask, P_Na_Nr_float_desc, dev_mask));
    CNRT_CHECK(cnrtQueueSync(queue));
    float min_data = 0;
    float max_data = Nr-1;
    CNNL_CHECK(cnnlClip_v2(handle, CNNL_POINTER_MODE_HOST, 
                            P_Na_Nr_float_desc, dev_mask, &min_data, &max_data, 
                            P_Na_Nr_float_desc, dev_mask_out));
    CNRT_CHECK(cnrtQueueSync(queue));
    
    //input维度扩展 1*Na*Nr*2 -> P*Na*Nr*2

    CNNL_CHECK(cnnlExpand(handle, Na_Nr_2_float_desc, dev_input, 
                        P_Na_Nr_2_float_desc, dev_input2));
    CNRT_CHECK(cnrtQueueSync(queue));
    if(G_DEBUG == true){
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));  
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "计算索引矩阵执行时间:(hardware) " << hardware_time / 1000 << " ms" << endl;
        Host_Timer.stop();
        host_time=Host_Timer.tv_usec;
        cout << "计算索引矩阵执行时间(host): " << host_time/1000 << " ms" << endl;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// // 2.2 求sinc////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(G_DEBUG == true){
        cnrtPlaceNotifier(start, queue);
        Host_Timer.start();
    }
    // CNNL_CHECK(cnnlExpand(handle, P_float_desc, dev_Pdata,
    //     P_Na_Nr_float_desc, dev_deltan_expand));
    CNRT_CHECK(cnrtQueueSync(queue));
    CNNL_CHECK(cnnlExpand(handle,Na_Nr_float_desc, dev_delta_n,P_Na_Nr_float_desc,center_idx));
    //center_idx - idx   P*Na*Nr*1   P_Na_Nr_float_desc, dev_mask
    // 获取workspace大小
    size_t workspace_size_sub = 0;
    CNNL_CHECK(cnnlGetOpTensorWorkspaceSize(handle, P_Na_Nr_float_desc, P_Na_Nr_float_desc, P_Na_Nr_float_desc, &workspace_size_sub));
    void *dev_workspace_sub;
    CNRT_CHECK(cnrtMalloc(&dev_workspace_sub, workspace_size_sub));
    alpha1 = 1.0f  * 3.1415926f;  //
    alpha2 = -1.0f * 3.1415926f; 
    CNNL_CHECK(cnnlOpTensor(handle, optensor_desc,
                          &alpha1, P_Na_Nr_float_desc, center_idx,
                          &alpha2, P_Na_Nr_float_desc, dev_mask_out,
                          dev_workspace_sub, workspace_size_sub, &beta,
                          P_Na_Nr_float_desc, dev_deltan_expand));
    CNRT_CHECK(cnrtQueueSync(queue));
    CNRT_CHECK(cnrtFree(dev_workspace_sub));   
    
//     // 在dev_deltan_expand操作后添加检查代码


// // 检查dev_deltan_expand中的0值
// vector<float> check_data(P * Na * Nr);  
// CNRT_CHECK(cnrtMemcpy(check_data.data(), dev_deltan_expand, data_size_4d, CNRT_MEM_TRANS_DIR_DEV2HOST));

// int zero_count = 0;
// int first_zero = -1;
// cout << "\n检查dev_deltan_expand中的0值:" << endl;

// for(int i = 0; i < check_data.size(); i++) {
//     if(abs(check_data[i]) < 1e-6) { // 使用小阈值来判断接近0的值
//         if(first_zero == -1) {
//             first_zero = i;
//         }
//         zero_count++;
//         if(zero_count <= 10) {  // 只打印前10个0的位置
//             cout << "发现0在位置 " << i << " (维度坐标: P=" 
//                  << i/(Na*Nr) << ", Na=" 
//                  << (i%(Na*Nr))/Nr << ", Nr=" 
//                  << i%Nr << ")" << endl;
//         }
//     }
// }

// cout << "总共发现 " << zero_count << " 个0值" << endl;
// cout << "0值占比: " << (float)zero_count/check_data.size() * 100 << "%" << endl;

// // 打印最大值和最小值
// float min_val = *min_element(check_data.begin(), check_data.end());
// float max_val = *max_element(check_data.begin(), check_data.end());
// cout << "数据范围: [" << min_val << ", " << max_val << "]" << endl;

// if(zero_count > 0 && first_zero != -1) {
//     // 打印第一个0值周围的数据
//     cout << "\n第一个0值周围的数据:" << endl;
//     int start_idx = max(0, first_zero-5);
//     int end_idx = min((int)check_data.size(), first_zero+6);
//     for(int i = start_idx; i < end_idx; i++) {
//         cout << "位置 " << i << " (维度坐标: P=" 
//              << i/(Na*Nr) << ", Na=" 
//              << (i%(Na*Nr))/Nr << ", Nr=" 
//              << i%Nr << "): " 
//              << check_data[i] << endl;
//     }
// }


    //sinc计算
    void *dev_sinc_result;
    CNRT_CHECK(cnrtMalloc(&dev_sinc_result, data_size_4d));
    cnnlSin_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, 
                    P_Na_Nr_float_desc, dev_deltan_expand, 
                    P_Na_Nr_float_desc, dev_sinc_result);
    CNRT_CHECK(cnrtQueueSync(queue));

    size_t workspace_size_sinc = 0;
    CNNL_CHECK(cnnlGetDivNoNanWorkspaceSize(handle, P_Na_Nr_float_desc, P_Na_Nr_float_desc, P_Na_Nr_float_desc, &workspace_size_sinc));
    void *dev_workspace_sinc;
    CNRT_CHECK(cnrtMalloc(&dev_workspace_sinc, workspace_size_sinc));
    // CNNL_CHECK(cnnlDiv_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION,
    //                     P_Na_Nr_float_desc, dev_sinc_result, 
    //                     P_Na_Nr_float_desc, dev_deltan_expand, 
    //                     dev_workspace_sinc, workspace_size_sinc, 
    //                     P_Na_Nr_float_desc, dev_sinc_result));
    CNNL_CHECK(cnnlDivNoNan_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, 
                                P_Na_Nr_float_desc, dev_sinc_result,  
                                P_Na_Nr_float_desc, dev_deltan_expand, 
                                dev_workspace_sinc, workspace_size_sinc, 
                                 P_Na_Nr_float_desc, dev_sinc_result));
    CNRT_CHECK(cnrtQueueSync(queue));
    CNRT_CHECK(cnrtFree(dev_workspace_sinc)); 
    if(G_DEBUG == true){
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));  
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "计算sinc执行时间: (hardware)" << hardware_time / 1000 << " ms" << endl;
        Host_Timer.stop();
        host_time=Host_Timer.tv_usec;
        cout << "计算sinc执行时间(host): " << host_time/1000 << " ms" << endl;
    }
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// // 4. 执行Gather操作进行花式索引////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(G_DEBUG == true){
        cnrtPlaceNotifier(start, queue);
        Host_Timer.start();
    }
    // 2. 创建整数类型的设备内存
    void *dev_indices,*dev_indices2;
    CNRT_CHECK(cnrtMalloc(&dev_indices, data_size_4d));
    CNRT_CHECK(cnrtMalloc(&dev_indices2, data_size_complex_out));



    // void* host_mask = malloc(data_size_4d);
    // CNRT_CHECK(cnrtMemcpy(host_mask, dev_mask_out, data_size_4d, CNRT_MEM_TRANS_DIR_DEV2HOST));
    // float* mask_array = (float*)host_mask;
    // for(int i = 890; i < 900; i++) {
    //         cout <<"Mask values : "<< i << ":" << mask_array[i] << " "<<endl;

    // }
    // cout << endl;
    // free(host_mask);

    // 3. 使用正确的类型转换参数
    CNNL_CHECK(cnnlCastDataType(handle,
                           P_Na_Nr_float_desc,        // 输入描述符
                           dev_mask_out,     // 输入数据
                           CNNL_CAST_FLOAT_TO_INT32,  // 使用正确的转换类型
                           P_Na_Nr_int32_desc,     // 输出描述符
                           dev_indices));    // 输出数据
    CNRT_CHECK(cnrtQueueSync(queue));


//     //检查转换后的indices值
// void* host_indices_pre = malloc(data_size_4d);
// CNRT_CHECK(cnrtMemcpy(host_indices_pre, dev_indices, data_size_4d, CNRT_MEM_TRANS_DIR_DEV2HOST));
// int* indices_pre = (int*)host_indices_pre;
// cout << "Indices before expand (first 10): ";
// for(int i = 890; i < 900; i++) {
   
//                  cout << i << ":" << indices_pre[i] << " "<<endl;
             
// }
// cout << endl;
// free(host_indices_pre);
    CNNL_CHECK(cnnlExpand(handle, P_Na_Nr_int32_desc, dev_indices,
                            P_Na_Nr_2_int32_desc, dev_indices2));
    CNRT_CHECK(cnrtQueueSync(queue));
    
// 6. 最后检查expand后的indices值
// void* host_indices = malloc(data_size_complex_out);
// CNRT_CHECK(cnrtMemcpy(host_indices, dev_indices2, data_size_complex_out, CNRT_MEM_TRANS_DIR_DEV2HOST));
// int* indices_array = (int*)host_indices;
// bool has_invalid = false;
// cout << "Indices after expand (first 10): ";
// for(int i = 0; i < P*Na*Nr*2; i++) {
    
//     if(indices_array[i] < 0 || indices_array[i] >= Nr) {
//         has_invalid = true;
//         cout << i << " ";
//     }
// }
// cout << endl;
// if(has_invalid) {
//     cout << "Warning: Invalid indices detected!" << endl;
// }
// free(host_indices);
    CNNL_CHECK(cnnlGather(handle, 2, 
                            P_Na_Nr_2_float_desc, dev_input2,  
                            P_Na_Nr_2_int32_desc, dev_indices2, 
                            P_Na_Nr_2_float_desc, dev_gather_out));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 5.乘权重
    void *dev_sinc_result2;
    CNRT_CHECK(cnrtMalloc(&dev_sinc_result2, data_size_complex_out));
    CNNL_CHECK(cnnlExpand(handle, P_Na_Nr_float_desc, dev_sinc_result,
                            P_Na_Nr_2_float_desc, dev_sinc_result2));
    CNRT_CHECK(cnrtQueueSync(queue));

    cnnlTensorDescriptor_t descs2[] = {P_Na_Nr_2_float_desc, P_Na_Nr_2_float_desc};
    void* inputs2[] = {dev_gather_out, dev_sinc_result2};
    CNNL_CHECK(cnnlMulN(handle, descs2, inputs2, 2, P_Na_Nr_2_float_desc, dev_input2));
    CNRT_CHECK(cnrtQueueSync(queue));

// // 检查dev_input2中的NaN值
// vector<float> check_data(P * Na * Nr * 2);  // 2表示复数的实部和虚部
// CNRT_CHECK(cnrtMemcpy(check_data.data(), dev_input2, data_size_complex_out, CNRT_MEM_TRANS_DIR_DEV2HOST));

// int nan_count = 0;
// int first_nan = -1;
// cout << "\n检查dev_input2中的NaN值:" << endl;

// // 检查实部和虚部
// for(int i = 0; i < check_data.size(); i++) {
//     if(isnan(check_data[i])) {
//         if(first_nan == -1) {
//             first_nan = i;
//         }
//         nan_count++;
//         if(nan_count <= 10) {  // 只打印前10个NaN的位置
//             cout << "发现NaN在位置 " << i << " (维度坐标: P=" 
//                  << i/(Na*Nr*2) << ", Na=" 
//                  << (i%(Na*Nr*2))/(Nr*2) << ", Nr=" 
//                  << ((i%(Na*Nr*2))%(Nr*2))/2 << ", 实/虚=" 
//                  << i%2 << ")" << endl;
//         }
//     }
// }

// cout << "总共发现 " << nan_count << " 个NaN值" << endl;
// cout << "NaN值占比: " << (float)nan_count/check_data.size() * 100 << "%" << endl;

// if(nan_count > 0 && first_nan != -1) {
//     // 打印第一个NaN值周围的数据
//     cout << "\n第一个NaN值周围的数据:" << endl;
//     int start_idx = max(0, first_nan-10);
//     int end_idx = min((int)check_data.size(), first_nan+11);
//     for(int i = start_idx; i < end_idx; i++) {
//         cout << "位置 " << i << " (维度坐标: P=" 
//              << i/(Na*Nr*2) << ", Na=" 
//              << (i%(Na*Nr*2))/(Nr*2) << ", Nr=" 
//              << ((i%(Na*Nr*2))%(Nr*2))/2 << ", 实/虚=" 
//              << i%2 << "): " 
//              << check_data[i] << endl;
//     }
// }
    if(G_DEBUG == true){
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));  
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "计算花式索引执行时间: (hardware)" << hardware_time / 1000 << " ms" << endl;
        Host_Timer.stop();
        host_time=Host_Timer.tv_usec;
        cout << "计算花式索引执行时间(host): " << host_time/1000 << " ms" << endl;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// // 5.求和////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(G_DEBUG == true){
        cnrtPlaceNotifier(start, queue);
        Host_Timer.start();
    }
    cnnlReduceDescriptor_t reduce_desc;
    CNNL_CHECK(cnnlCreateReduceDescriptor(&reduce_desc));
    int reduce_axis[] = {0};  // 在第0维上进行reduce
    int num_axes = 1;        // reduce的维度数量
    CNNL_CHECK(cnnlSetReduceDescriptor(reduce_desc, reduce_axis, num_axes, CNNL_REDUCE_ADD, CNNL_DTYPE_FLOAT, 
                                    CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES));
    size_t workspace_reduce_size = 0;
    CNNL_CHECK(cnnlGetReduceOpWorkspaceSize(handle,
                                        P_Na_Nr_2_float_desc,
                                        Na_Nr_2_float_desc,
                                     reduce_desc,
                                     &workspace_reduce_size));
    void *workspace = nullptr;
    if (workspace_reduce_size > 0) {
        CNRT_CHECK(cnrtMalloc(&workspace, workspace_reduce_size));
    }                                
    void *dev_output;
    CNRT_CHECK(cnrtMalloc(&dev_output, data_size_complex));
    float alpha = 1.0f;
    float beta_final = 0.0f;
    CNNL_CHECK(cnnlReduce(handle, reduce_desc, workspace, workspace_reduce_size,
                            &alpha, P_Na_Nr_2_float_desc, dev_input2, 
                            0, nullptr, 
                            &beta_final, Na_Nr_2_float_desc, dev_output));

    CNNL_CHECK(cnnlDestroyReduceDescriptor(reduce_desc));
    CNRT_CHECK(cnrtQueueSync(queue));
    if(G_DEBUG == true){
        cnrtPlaceNotifier(end, queue);
        CNRT_CHECK(cnrtQueueSync(queue));  
        cnrtNotifierDuration(start, end, &hardware_time);
        cout << "降维度执行时间:(hardware) " << hardware_time / 1000 << " ms" << endl;
        Host_Timer.stop();
        host_time=Host_Timer.tv_usec;
        cout << "降维度执行时间(host): " << host_time/1000 << " ms" << endl;
    }
    ////////////////////////////////////////////////////
//     void* host_indices_pre = malloc(data_size_complex);
// CNRT_CHECK(cnrtMemcpy(host_indices_pre, dev_output, data_size_complex, CNRT_MEM_TRANS_DIR_DEV2HOST));
// float* indices_pre = (float*)host_indices_pre;
// cout << "Indices before expand (first 10): ";
// for(int i = 0; i < 10; i++) {
   
//                  cout << i << ":" << indices_pre[i] << " "<<endl;
             
// }
// cout << endl;
// free(host_indices_pre);
////////////////////////////////////////////////////
    CNRT_CHECK(cnrtMemcpy(output.data(), dev_output, data_size_complex, CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtQueueSync(queue));
    //output = dev_output;
    // for (int i = 0; i < 5; i++) {
    //     for (int j = 0; j < 5; j++) {
    //         cout << output[i * Nr + j] << " ";
    //     }
    //     cout << endl;
    // }

    // 释放所有申请的资源
    // 释放描述符
    CNNL_CHECK(cnnlDestroyOpTensorDescriptor(optensor_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(Na_Nr_float_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(Nr_float_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(Na_Nr_float_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(P_Na_Nr_float_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(Na_Nr_2_float_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(P_Na_Nr_2_float_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(Nr_float_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(P_float_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(P_Na_Nr_float_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(P_Na_Nr_int32_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(P_Na_Nr_2_int32_desc));

    // 释放设备内存
    CNRT_CHECK(cnrtFree(dev_fr_new_mtx));
    CNRT_CHECK(cnrtFree(dev_delta_n));
    CNRT_CHECK(cnrtFree(dev_deltan_expand));
    CNRT_CHECK(cnrtFree(dev_input));
    CNRT_CHECK(cnrtFree(dev_input2));
    CNRT_CHECK(cnrtFree(dev_NrOut));
    CNRT_CHECK(cnrtFree(dev_Pdata));
    CNRT_CHECK(cnrtFree(dev_sinc_result));
    CNRT_CHECK(cnrtFree(dev_mask));
    CNRT_CHECK(cnrtFree(dev_mask_out));
    CNRT_CHECK(cnrtFree(signal_expand));
    CNRT_CHECK(cnrtFree(dev_indices));
    CNRT_CHECK(cnrtFree(dev_indices2));
    CNRT_CHECK(cnrtFree(dev_sinc_result2));
    CNRT_CHECK(cnrtFree(dev_output));

    // 释放workspace内存
    if (workspace != nullptr) {
        CNRT_CHECK(cnrtFree(workspace));
    }

    cout << "Stolt插值V2(sinc)完成" << endl;
}