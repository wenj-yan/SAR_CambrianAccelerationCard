%{
    本代码用于对雷达的回波数据进行一维FFT分析
    包括水平方向(距离向)和垂直方向(方位向)的FFT处理
    创建日期：2024
%}
close all;
clear;

%% 数据读取
% 从txt文件读取回波数据
echo_data = load('echo.txt');
[rows, cols] = size(echo_data);
% 将实部和虚部重组为复数
Nr = cols/2;
echo = complex(zeros(rows, Nr));
for i = 1:Nr
    echo(:,i) = complex(echo_data(:,2*i-1), echo_data(:,2*i));
end

% 从txt文件读取参数
fid = fopen('paras.txt', 'r');
params_text = textscan(fid, '%s %f', 'Delimiter', ':');
fclose(fid);

% 构建参数结构体
param_names = params_text{1};
param_values = params_text{2};
para = struct();
for i = 1:length(param_names)
    para.(strtrim(param_names{i})) = param_values(i);
end

% 提取需要的参数
Fr = para.Fr;   % 距离向采样率
Fa = para.PRF;  % 方位向采样率
f0 = para.f0;   % 中心频率
R0 = para.R0;   % 最近点斜距 
c = para.c;     % 光速

%% 计算基本参数
[Na, Nr] = size(echo);

% 距离向时间轴及频率轴
tr_axis = 2*R0/c + (0:Nr-1)/Fr;   % 距离向时间轴
fr_gap = Fr/Nr;
fr_axis = (0:Nr-1).*fr_gap;   % 距离向频率轴

% 方位向时间轴及频率轴
ta_axis = (0:Na-1)/Fa;    % 方位向时间轴
ta_gap = Fa/Na; 
fa_axis = (0:Na-1).*ta_gap;    % 方位向频率轴

%% 时域信号显示
figure;
subplot(2,1,1);
plot(tr_axis, abs(echo(round(Na/2),:)));
title('距离向时域信号（中间方位向）');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

subplot(2,1,2);
plot(ta_axis, abs(echo(:,round(Nr/2))));
title('方位向时域信号（中间距离门）');
xlabel('时间 (s)');
ylabel('幅度');
grid on;

%% 二维频谱分析
% 先进行水平方向（距离向）FFT
echo_h = fft(echo,[],2);

% 再进行垂直方向（方位向）FFT
echo_2d_fft = fft(echo_h,[],1);

% 显示二维频谱
figure;
imagesc(fr_axis, fa_axis, 20*log10(abs(echo_2d_fft)));
colorbar;
title('二维频谱（dB）');
xlabel('距离向频率 (Hz)');
ylabel('方位向频率 (Hz)'); 