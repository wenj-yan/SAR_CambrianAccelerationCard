#include <cnnl.h>
#include <cnrt.h>

cnnlStatus_t stolt_interpolation(
    cnnlHandle_t handle,
    const float* fold_m,    // [Na, Nr]
    const float* ftau,      // [Nr]
    const float* S1,        // [Na, Nr]
    float* Sstolt,         // [Na, Nr]
    const int Na,
    const int Nr,
    const float Fr) {
    
    // 1. 创建三维tensor描述符
    cnnlTensorDescriptor_t fold_m_desc, ftau_desc, S1_desc, Sstolt_desc;
    int fold_m_dims[] = {1, Na, Nr};  // 添加batch维度
    int ftau_dims[] = {1, 1, Nr};     // 扩展维度以便广播
    
    CNNL_CHECK(cnnlCreateTensorDescriptor(&fold_m_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&ftau_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&S1_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&Sstolt_desc));
    
    // 设置为3D tensor
    CNNL_CHECK(cnnlSetTensorDescriptor(fold_m_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, fold_m_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(ftau_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, ftau_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(S1_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, fold_m_dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(Sstolt_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 3, fold_m_dims));

    // 2. 计算Delta = (fold_m - ftau) / (Fr/Nr)
    float* delta;
    CNRT_CHECK(cnrtMalloc(&delta, Na * Nr * sizeof(float)));
    
    // 使用3D广播减法
    float scale = Nr / Fr;
    CNNL_CHECK(cnnlBroadcastSub(
        handle,
        fold_m_desc, fold_m,
        ftau_desc, ftau,
        fold_m_desc, delta
    ));
    
    // 标量乘法
    CNNL_CHECK(cnnlScale(
        handle,
        fold_m_desc,
        delta,
        fold_m_desc,
        delta,
        scale,
        0.0f
    ));
    
    // 4. 计算IntNum = ceil(Delta)
    float* int_num;
    CNRT_CHECK(cnrtMalloc(&int_num, Na * Nr * sizeof(float)));
    CNNL_CHECK(cnnlCeil(
        handle,
        fold_m_desc,
        delta,
        fold_m_desc,
        int_num
    ));
    
    // 5. 生成网格索引并计算kk
    float* kk;
    CNRT_CHECK(cnrtMalloc(&kk, Na * Nr * sizeof(float)));
    
    // 使用arange和broadcast生成索引矩阵
    float* range;
    CNRT_CHECK(cnrtMalloc(&range, Nr * sizeof(float)));
    CNNL_CHECK(cnnlArange(
        handle,
        1,
        Nr + 1,
        1,
        CNNL_DTYPE_FLOAT,
        range
    ));
    
    // 广播相加得到kk
    CNNL_CHECK(cnnlBroadcastAdd(
        handle,
        ftau_desc, range,
        fold_m_desc, int_num,
        fold_m_desc, kk
    ));
    
    // 6. 计算掩码 (5 <= kk <= Nr-3)
    float* mask;
    CNRT_CHECK(cnrtMalloc(&mask, Na * Nr * sizeof(float)));
    
    // 使用比较算子和逻辑与
    float* temp_mask1, *temp_mask2;
    CNRT_CHECK(cnrtMalloc(&temp_mask1, Na * Nr * sizeof(float)));
    CNRT_CHECK(cnrtMalloc(&temp_mask2, Na * Nr * sizeof(float)));
    
    CNNL_CHECK(cnnlGreaterOrEqual(
        handle,
        fold_m_desc, kk,
        nullptr, &(float){5.0f},
        fold_m_desc, temp_mask1
    ));
    
    CNNL_CHECK(cnnlLessOrEqual(
        handle,
        fold_m_desc, kk,
        nullptr, &(float){(float)(Nr-3)},
        fold_m_desc, temp_mask2
    ));
    
    CNNL_CHECK(cnnlLogicalAnd(
        handle,
        fold_m_desc, temp_mask1,
        fold_m_desc, temp_mask2,
        fold_m_desc, mask
    ));
    
    // 7. 计算sinc插值
    // DecNum = IntNum - Delta
    float* dec_num;
    CNRT_CHECK(cnrtMalloc(&dec_num, Na * Nr * sizeof(float)));
    CNNL_CHECK(cnnlSub(
        handle,
        fold_m_desc, int_num,
        fold_m_desc, delta,
        fold_m_desc, dec_num
    ));

    // 8. 实现sinc插值计算 - 使用4D tensor来处理8个点
    int sinc_dims[] = {1, Na, Nr, 8};  // 添加第四维来存储8个sinc值
    cnnlTensorDescriptor_t sinc_desc;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&sinc_desc));
    CNNL_CHECK(cnnlSetTensorDescriptor(sinc_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 4, sinc_dims));

    float* sinc_result;
    CNRT_CHECK(cnrtMalloc(&sinc_result, Na * Nr * 8 * sizeof(float)));

    // 计算DecNum + offsets 并生成sinc值
    float offsets[8] = {-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    
    // 使用4D tensor进行计算
    for(int i = 0; i < 8; i++) {
        float* curr_sinc = sinc_result + i * Na * Nr;
        CNNL_CHECK(cnnlAdd(
            handle,
            fold_m_desc, dec_num,
            nullptr, &offsets[i],
            fold_m_desc, curr_sinc
        ));
        
        CNNL_CHECK(cnnlSinc(
            handle,
            fold_m_desc,
            curr_sinc,
            fold_m_desc,
            curr_sinc
        ));
    }

    // 9. 使用4D tensor进行gather操作
    float* gathered_s1;
    CNRT_CHECK(cnrtMalloc(&gathered_s1, Na * Nr * 8 * sizeof(float)));

    // 10. 最终的卷积和求和
    CNNL_CHECK(cnnlConvolution(
        handle,
        gathered_s1,    // 输入
        sinc_result,    // 卷积核
        Sstolt,         // 输出
        /* 其他卷积参数 */
    ));

    // 清理资源
    CNRT_CHECK(cnrtFree(sinc_result));
    CNRT_CHECK(cnrtFree(gathered_s1));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(sinc_desc));
    
    return CNNL_STATUS_SUCCESS;
}