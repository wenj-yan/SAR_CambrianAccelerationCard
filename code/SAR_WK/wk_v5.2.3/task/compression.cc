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

void generateReferenceFunction_CPU(cnnlHandle_t handle,
    cnrtQueue_t queue,
    vector<complex<float>>& theta_ft_fa,
    const vector<double>& fr_axis,
    const vector<double>& fa_axis,
    double f0, double c, double Vr, double Kr, double R0,
    int Na, int Nr) {
    // 初始化输出向量
    theta_ft_fa.resize(Na * Nr);

    // 计算常量项
    double bb = (c * c) / (4.0 * Vr * Vr);
    double d1 = M_PI / Kr * (-1.0);
    double scale = 4 * M_PI * R0 / c;

    // 对每个点计算参考函数
    for (int i = 0; i < Na; ++i) {
    double fa = fa_axis[i];  // 当前fa值
    double fa_term = bb * fa * fa;  // (c^2/4/Vr^2) * fa^2

    for (int j = 0; j < Nr; ++j) {
    double fr = fr_axis[j];  // 当前fr值
    double f_sum = f0 + fr;  // f0 + fr

    // 计算第一项: 4*pi*R0/c * sqrt((f0+fr)^2 - c^2/4/Vr^2 * fa^2)
    double under_sqrt = f_sum * f_sum - fa_term;
    if(under_sqrt < 0) {
    // 处理负数情况,设为0避免nan
    under_sqrt = 0;
    }
    double first_term = scale * sqrt(under_sqrt);

    // 计算第二项: pi/Kr * fr^2
    double second_term = d1 * fr * fr;

    // 计算总相位
    double theta = first_term + second_term;

    // 使用欧拉公式计算复指数
    theta_ft_fa[i * Nr + j] = complex<float>(cos(theta), sin(theta));
    }
    }

    // 打印前5x5个点的值以及相位值进行验证
    if (G_DEBUG == true) {
        cout << "\n参考函数前5x5个点的值及其相位：" << endl;
        for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
        size_t idx = i * Nr + j;
        complex<float> val = theta_ft_fa[idx];
        double phase = atan2(val.imag(), val.real());
        cout << "值:(" << val.real() << "," << val.imag() << ") 相位:" << phase << " ";
        }
        cout << endl;
        }
    }
}