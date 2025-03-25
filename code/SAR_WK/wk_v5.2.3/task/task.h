#ifndef __TASK_H__
#define __TASK_H__

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

#include "../readfile/read.h"
#include "../complex/complex.h"


using namespace std;

extern bool G_DEBUG;

// Stolt插值函数
//使用gridsample算子进行采样，效率较低
void performStoltInterpolation(cnnlHandle_t handle, 
                             cnrtQueue_t queue,
                             const vector<complex<float>>& input,
                             vector<complex<float>>& output,
                             const vector<double>& fr_axis,
                             const vector<double>& fa_axis,
                             double f0, double c, double Vr,
                             int Na, int Nr,double fr);

//矩阵运算实现stolt插值——sinc插值
void StoltInterp_sinc(cnnlHandle_t handle, 
                             cnrtQueue_t queue,
                             void * input,
                             void * output,
                             const vector<double>& fr_axis,
                             const vector<double>& fa_axis,
                             double f0, double c, double Vr,
                             int Na, int Nr,double fr,int P);

//2dfft
void perform2DFFT(cnnlHandle_t handle, 
    cnrtQueue_t queue,
    void* d_data,
    size_t Na, 
    size_t Nr);
//2difft
void perform2DIFFT(cnnlHandle_t handle, 
    cnrtQueue_t queue,
    void* d_data,
    size_t Na, 
    size_t Nr);


//参考函数生成-CPU
void generateReferenceFunction_CPU(cnnlHandle_t handle,
        cnrtQueue_t queue,
        vector<complex<float>>& theta_ft_fa,
        const vector<double>& fr_axis,
        const vector<double>& fa_axis,
        double f0, double c, double Vr, double Kr, double R0,
        int Na, int Nr);

//前处理        
void preprocess_wk(cnnlHandle_t handle, cnrtQueue_t queue,
                    void *d_input,size_t Na, size_t Nr,size_t Na_padded,size_t Nr_padded,
                    WKParams params,void *d_output);

#endif // !__TASK_H__
