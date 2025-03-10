#include "read.h"
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

// 读取复数数据
bool readComplexData(const string& filename, vector<complex<float>>& data) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return false;
    }

    data.clear();
    float real, imag;
    
    // 读取实部和虚部
    while (file >> real >> imag) {
        data.push_back(complex<float>(real, imag));
    }
    if(G_DEBUG == true) cout << "成功读取 " << data.size() << " 个复数" << endl;
    file.close();
    return true;
}
// 读取乘法结束后的数据
bool readComplexData_mul(const string& filename, vector<complex<float>>& data) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return false;
    }

    data.clear();
    float real, imag;
    
    // 读取实部和虚部
    while (file >> real >> imag) {
        data.push_back(complex<float>(real, imag));
    }

    cout << "成功读取 " << data.size() << " 个复数" << endl;
    file.close();
    return true;
}
// 读取参数文件
bool readParams(const string& filename, WKParams& params) {
    cout << "开始读取参数！" << endl;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开参数文件: " << filename << endl;
        return false;
    }

    string line;
    while (getline(file, line)) {
        size_t pos = line.find(':');
        if (pos == string::npos) continue;
        
        string param_name = line.substr(0, pos);
        double value = stod(line.substr(pos + 1));
        
        if (param_name == "Fr") params.Fr = value;
        else if (param_name == "PRF") params.PRF = value;
        else if (param_name == "f0") params.f0 = value;
        else if (param_name == "Tr") params.Tr = value;
        else if (param_name == "R0") params.R0 = value;
        else if (param_name == "Kr") params.Kr = value;
        else if (param_name == "c") params.c = value;
    }

    file.close();

    // 打印所有参数
    if(G_DEBUG == true){
        cout << "读取到的参数：" << endl;
        cout << "Fr: " << params.Fr << " Hz" << endl;
        cout << "PRF: " << params.PRF << " Hz" << endl;
        cout << "f0: " << params.f0 << " Hz" << endl;
        cout << "Tr: " << params.Tr << " s" << endl;
        cout << "R0: " << params.R0 << " m" << endl;
        cout << "Kr: " << params.Kr << " Hz/s" << endl;
        cout << "c: " << params.c << " m/s" << endl;
        cout << "Vr: " << params.Vr << " m/s" << endl;
        cout << "Ka: " << params.Ka << " Hz/s" << endl;
        cout << "f_nc: " << params.f_nc << " Hz" << endl;
    }

    return true;
}