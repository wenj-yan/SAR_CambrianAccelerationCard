%{
    本代码用于对雷达的回波数据，利用wka算法进行成像，利用电平饱和法以及直方图均衡法，
    提高成像质量。
    2023/11/26 21:16
%}
%close all;
%% 数据读取
% 加载数据
echo1 = importdata('CDdata1.mat');
echo2 = importdata('CDdata2.mat');
% 将回波拼装在一起
echo = double([echo1;echo2]);
% 将复数数据保存为txt文件
%fid = fopen('echo_data.txt', 'w');
%for i = 1:6144
%    for j = 1:4096
%        fprintf(fid, '%.6f %.6f ', real(echo(i,j)), imag(echo(i,j)));
%    end
%    fprintf(fid, '\n');
%end
%fclose(fid);
% 加载参数
para = importdata('CD_run_params.mat');
Fr = para.Fr;   % 距离向采样率
Fa = para.PRF;  % 方位向采样率
f0 = para.f0;   % 中心频率
Tr = para.Tr;   % 脉冲持续时间
R0 = para.R0;   % 最近点斜距 
Kr = -para.Kr;   % 线性调频率
c = para.c;     % 光速
% 以下参数来自课本附录A
Vr = 7062;      % 等效雷达速度
Ka = 1733;      % 方位向调频率
f_nc = -6900;   % 多普勒中心频率

%% 图像填充
% 计算参数
[Na,Nr] = size(echo);
% 按照全尺寸对图像进行补零
% 设置插值倍数
%factor_a = 3;  % 方位向扩展2倍
%factor_r = 3;  % 距离向扩展2倍
% 选择一种插值方法使用
%echo_interpolated = internal_linear_interp(echo, factor_a, factor_r);
echo = padarray(echo,[round(Na/6), round(Nr/3)]);
%random_padding = rand(size(echo) + [2*round(Na/6), 2*round(Nr/3)]);
%random_padding(round(Na/6)+1:end-round(Na/6), round(Nr/3)+1:end-round(Nr/3)) = echo;
%echo = random_padding;
% 计算参数
[Na,Nr] = size(echo);

%% 轴产生
% 距离向时间轴及频率轴
tr_axis = 2*R0/c + (-Nr/2:Nr/2-1)/Fr;   % 距离向时间轴
fr_gap = Fr/Nr;
aa = fftshift(-Nr/2:Nr/2-1);
fr_axis = aa.*fr_gap;   % 距离向频率轴

% 方位向时间轴及频率轴
ta_axis = (-Na/2:Na/2-1)/Fa;    % 方位向时间轴
ta_gap = Fa/Na; 
fa_axis = f_nc + fftshift(-Na/2:Na/2-1).*ta_gap;    % 方位向频率轴
% 方位向对应纵轴，应该转置成列向量
ta_axis = ta_axis';
fa_axis = fa_axis';

%% 第一步 二维傅里叶变换
tic;
% 方位向下变频
phase = exp(-2i*pi*f_nc.*ta_axis);
echo = echo .* phase;
% 二维傅里叶变换
echo_s1 = fft2(echo);
t1 = toc;
fprintf('2D FFT 耗时: %.2f 秒\n', t1);

%% 第二步 参考函数相乘(一致压缩)
tic;
% 生成参考函数
a = pi/Kr;
theta_ft_fa = 4*pi*R0/c.*sqrt((f0+fr_axis).^2-c^2/4/Vr^2.*fa_axis.^2)+a.*fr_axis.^2;
theta_ft_fa = exp(1i.*theta_ft_fa);
% 一致压缩
echo_s2 = echo_s1 .* theta_ft_fa;
t2 = toc;
fprintf('一致压缩 耗时: %.2f 秒\n', t2);

%% 第三步 在距离域进行Stolt插值操作(补余压缩)
tic;
% 计算映射后的距离向频率
fr_new_mtx = sqrt((f0+fr_axis).^2-c^2/4/Vr^2.*fa_axis.^2)-f0;

Signal_stolt = zeros(Na,Nr);
P = 6;  % 插值核长度
for m = 1:Na
    % 计算当前方位向频率对应的插值位置
    delta = (fr_new_mtx(m,:) - fr_axis)./fr_gap;
    for n = 1:Nr
        % 计算插值核的中心位置
        center_idx = n - delta(n);
        % 确定插值范围
        idx_start = max(1, floor(center_idx - P/2));
        idx_end = min(Nr, ceil(center_idx + P/2));
        % 对范围内的点进行sinc插值
        for idx = idx_start:idx_end
            x = center_idx - idx;
            if abs(x) <= P/2
                Signal_stolt(m,n) = Signal_stolt(m,n) + ...
                    echo_s2(m,idx) * sinc(x);
            end
        end
    end
end
t3 = toc;
fprintf('stolt插值 耗时: %.2f 秒\n', t3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 第四步 二维逆傅里叶变换
tic;
echo_s4 = ifft2(Signal_stolt);
t4 = toc;
fprintf('2D IFFT 耗时: %.2f 秒\n', t4);

%% 第五步 图像纠正
echo_s5 = circshift(echo_s4,-1800,2);
echo_s5 = circshift(echo_s5,-3365,1);
echo_s5 = flipud(echo_s5);

%% 画图
saturation = 50;
figure;
echo_s6 = abs(echo_s5);
echo_s6(echo_s6 > saturation) = saturation;
imagesc(tr_axis.*c,ta_axis.*c,echo_s6);
title('ωk处理结果(精确版本)');
% 绘制处理结果灰度图
% 做一些图像处理。。。
echo_res = gather(echo_s6 ./ saturation);
% 直方图均衡
echo_res = adapthisteq(echo_res,"ClipLimit",0.004,"Distribution","exponential","Alpha",0.5);
figure;
imshow(echo_res);
