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
    // int dims[4] = {batch_size, height, width, channels};
    // CNNL_CHECK(cnnlSetTensorDescriptor(roll_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_COMPLEX_FLOAT, 4, dims));
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
        for (int w = 0; w < Nr; w++) {
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
                             const vector<complex<float>>& input,
                             vector<complex<float>>& output,
                             const vector<double>& fr_axis,
                             const vector<double>& fa_axis,
                             double f0, double c, double Vr,
                             int Na, int Nr,double fr,int P)
{
    cout << "\n执行Stolt插值V2(sinc)..." << endl;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// 1. 计算新频率映射////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    vector<float> fr_new_mtx = calculateNewFrequencyMapping(fr_axis, fa_axis, f0, c, Vr, Na, Nr);
    if(G_DEBUG == true){
        float fr_new_min = *min_element(fr_new_mtx.begin(), fr_new_mtx.end());
        float fr_new_max = *max_element(fr_new_mtx.begin(), fr_new_mtx.end());
        float fr_min = *min_element(fr_axis.begin(), fr_axis.end());
        float fr_max = *max_element(fr_axis.begin(), fr_axis.end());
        cout << "fr_new_mtx范围: [" << fr_new_min << ", " << fr_new_max << "]" << endl;
        cout << "fr_axis范围: [" << fr_min << ", " << fr_max << "]" << endl;
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
    //设置输入输出tensor描述符
    cnnlTensorDescriptor_t a_desc, b_desc, c_desc,out_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&a_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&b_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&c_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&out_desc));
    //设置维度
    int dims[4] = {1,Na, Nr,1};
    int dims_2[4] = {1,1,Nr,1};
    int dims_4d[4] = {P,1,Nr,1};
    int dims_4d_out[4] = {P,Na,Nr,1};
    CNNL_CHECK(cnnlSetTensorDescriptor(a_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(b_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dims_2));
    CNNL_CHECK(cnnlSetTensorDescriptor(c_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(out_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, dims_4d_out));
    //准备输入数据
    float alpha1 = 1.0f * Nr/ (fr  * 2.0f);  
    float alpha2 = -1.0f * Nr/ (fr  * 2.0f); 
    float beta = 0.0f;
    float a = 1.0f;
    // 分配设备内存
    void *dev_fr_new_mtx, *dev_fr_axis, *dev_delta_n,*dev_deltan_expand;
    size_t data_size = Na * Nr * sizeof(float);
    size_t data_size_1d =  Nr * sizeof(float);
    size_t data_size_4d =  P*Na*Nr * sizeof(float);
    
    CNRT_CHECK(cnrtMalloc(&dev_fr_new_mtx, data_size));
    CNRT_CHECK(cnrtMalloc(&dev_fr_axis, data_size_1d));
    CNRT_CHECK(cnrtMalloc(&dev_delta_n, data_size));
    CNRT_CHECK(cnrtMalloc(&dev_deltan_expand, data_size_4d));
    
////////////////////////////////////////
    // 创建测试数据
    std::vector<float> fr_new_mtx2(Na * Nr, 1.0f);  // 尺寸为Na*Nr，初始化为1.0
    std::vector<float> fr_axis2(Nr);                // 尺寸为Nr
    
    // 生成简单的线性数据
    for (int i = 0; i < Na; ++i) {
        for (int j = 0; j < Nr; ++j) {
            fr_new_mtx2[i * Nr + j] = static_cast<float>(i + j);  // 线性递增
        }
    }
    for (int i = 0; i < Nr; ++i) {
        fr_axis2[i] = static_cast<float>(i);  // 0, 1, 2, ..., Nr-1
    }

    // 打印输入数据
    std::cout << "Input fr_new_mtx (first 10 elements): ";
    for (int i = 0; i < std::min(10, (int)fr_new_mtx2.size()); ++i) {
        std::cout << fr_new_mtx2[i] << " ";
    }
    std::cout << "\nInput fr_axis: ";
    for (auto val : fr_axis2) std::cout << val << " ";
    std::cout << std::endl;
    //////////////////////////////////////
    // 将数据拷贝到设备
    CNRT_CHECK(cnrtMemcpy(dev_fr_new_mtx, fr_new_mtx2.data(), data_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(dev_fr_axis, (void*)fr_axis2.data(), data_size_1d, CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 获取workspace大小
    size_t workspace_size = 0;
    CNNL_CHECK(cnnlGetOpTensorWorkspaceSize(handle, a_desc, b_desc, c_desc, &workspace_size));
    void *dev_workspace;
    CNRT_CHECK(cnrtMalloc(&dev_workspace, workspace_size));
    CNNL_CHECK(cnnlOpTensor(handle, optensor_desc,
                          &a, a_desc, dev_fr_new_mtx,
                          &a, b_desc, dev_fr_axis,
                          dev_workspace, workspace_size, &beta,
                          c_desc, dev_delta_n));
    CNRT_CHECK(cnrtQueueSync(queue));
    // //将结果拷贝回主机
    // vector<float> delta_n( Na*Nr);
    // CNRT_CHECK(cnrtMemcpy(delta_n.data(), dev_delta_n, data_size, CNRT_MEM_TRANS_DIR_DEV2HOST));
    // // 打印计算结果
    // std::cout << "Output result: ";
    // for (auto val : delta_n) std::cout << val << " ";
    // std::cout << std::endl;
    // 打印调试信息
    // if(G_DEBUG == true){
    //     cout << "delta_n计算完成，范围: [" 
    //         << *min_element(delta_n.begin(), delta_n.end()) << ", "
    //         << *max_element(delta_n.begin(), delta_n.end()) << "]" << endl;
    // }

    // 释放设备内存
    CNRT_CHECK(cnrtFree(dev_fr_new_mtx));
    CNRT_CHECK(cnrtFree(dev_fr_axis));
    CNRT_CHECK(cnrtFree(dev_workspace));
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// // 2.2 求sinc(delta_n(m,n)-i）////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 定义张量维度
    int dims_p[4] = {P, 1, 1, 1};  // 6*1*1*1
    // 定义张量数据
    float data[6] = {-3, -2, -1, 0, 1, 2};  // 输入数据
    // 创建张量描述符
    cnnlTensorDescriptor_t tensor_desc;
    cnnlCreateTensorDescriptor(&tensor_desc);
    CNNL_CHECK(cnnlSetTensorDescriptor(tensor_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,4, dims_p));
    // 分配设备内存
    void* dev_Pdata = nullptr;
    CNRT_CHECK(cnrtMalloc(&dev_Pdata, P * sizeof(float)));
    // 将数据拷贝到设备
    cnrtMemcpy(dev_Pdata, data, P * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV);
    alpha1 = 1.0f ;  
    alpha2 = -1.0f ; 
    beta = 0.0f;
    // 获取workspace大小
    size_t workspace_size2 = 0;
    CNNL_CHECK(cnnlGetOpTensorWorkspaceSize(handle, c_desc, tensor_desc, out_desc, &workspace_size2));
    void *dev_workspace2;
    CNRT_CHECK(cnrtMalloc(&dev_workspace2, workspace_size2));
    //delta_n(m,n)-i   P*M*N*1
    CNNL_CHECK(cnnlOpTensor(handle, optensor_desc,
                          &alpha1, c_desc, dev_delta_n,
                          &alpha2, tensor_desc, dev_Pdata,
                          dev_workspace2, workspace_size2, &beta,
                          out_desc, dev_deltan_expand));
    CNRT_CHECK(cnrtQueueSync(queue));
    CNRT_CHECK(cnrtFree(dev_workspace2));
    // // 将结果拷贝回主机
    // vector<float> delta_n_ex( Na*Nr*P);
    // CNRT_CHECK(cnrtMemcpy(delta_n_ex.data(), dev_deltan_expand, data_size_4d, CNRT_MEM_TRANS_DIR_DEV2HOST));
    // // 打印计算结果
    // std::cout << "Output result: ";
    // for (auto val : delta_n_ex) std::cout << val << " ";
    // std::cout << std::endl;
    //sinc计算
    


    // 3. 生成邻域索引和权重索引P[6]=[-3,-2,-1,0,1,2]，权重尺寸NrxP
   

   //  3.2 将输入数据和权重数据进行维度对齐

    // 4. 执行Gather操作进行花式索引

    // 5.乘权重

    // 6.求和
}