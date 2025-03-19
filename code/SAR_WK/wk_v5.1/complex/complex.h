#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#include <time.h>
#include <string>
#include "string.h"
#include "../tool/tool.h"
#include <complex>
#include <iostream>
#include <complex.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <algorithm> 

using namespace std;

extern bool G_DEBUG;


void SplitComplexTensor(cnnlHandle_t handle,
                       cnrtQueue_t queue,
                       const void* complex_data, // [Na, Nr] - 复数数据
                       int Na,
                       int Nr,
                       void* d_real,  // [Na, Nr] - 实部设备指针
                       void* d_imag);  // [Na, Nr] - 虚部设备指针

void CombineComplexTensor(cnnlHandle_t handle,
                         cnrtQueue_t queue,
                         void* d_real,        // 实部数据指针 [Na,Nr,1]
                         void* d_imag,        // 虚部数据指针 [Na,Nr,1] 
                         int Na,              // 输入维度
                         int Nr,              // 输入维度
                         void* d_output);      // 输出数据指针 [Na,Nr,2]

// 实数矩阵乘法函数
cnnlStatus_t MatrixMultiply(cnnlHandle_t handle,
                           const void* A,  // 修改为void*以兼容设备指针
                           const void* B,  // 修改为void*以兼容设备指针
                           void* C,        // 修改为void*以兼容设备指针
                           int M,  int N) ;


// 在 MatrixMultiply 函数之前添加复数矩阵乘法函数
cnnlStatus_t ComplexMatrixMultiply(cnnlHandle_t handle,
                                  cnrtQueue_t queue,
                                  const void* A,
                                  const void* B,
                                  void* C,
                                  int M,  int N);


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
                     void* d_output);       // 设备端输出数据指针


#endif