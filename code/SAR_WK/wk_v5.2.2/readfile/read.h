#ifndef __READ_H__
#define __READ_H__

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

// 定义参数结构体
struct WKParams {
    double Fr;    // 距离向采样率
    double PRF;   // 方位向采样率
    double f0;    // 中心频率
    double Tr;    // 脉冲持续时间
    double R0;    // 最近点斜距
    double Kr;    // 线性调频率
    double c;     // 光速
    double Vr;    // 等效雷达速度
    double Ka;    // 方位向调频率
    double f_nc;  // 多普勒中心频率
};

// 读取复数数据
bool readComplexData(const string& filename, vector<complex<float>>& data);

// 读取乘法结束后的数据
bool readComplexData_mul(const string& filename, vector<complex<float>>& data);

// 读取参数文件
bool readParams(const string& filename, WKParams& params);



#endif // !1