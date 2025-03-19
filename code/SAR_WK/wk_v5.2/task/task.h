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
                             vector<complex<float>>& output,
                             const vector<double>& fr_axis,
                             const vector<double>& fa_axis,
                             double f0, double c, double Vr,
                             int Na, int Nr,double fr,int P);



#endif // !__TASK_H__
