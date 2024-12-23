echo1 = importdata('CDdata1.mat');
echo2 = importdata('CDdata2.mat');
% 将回波拼装在一起
echo = double([echo1;echo2]);
% 加载参数
para_data = importdata('CD_run_params.mat');

% 创建参数字典
para = struct();
para.Fr = para_data.Fr;    % 距离向采样率
para.PRF = para_data.PRF;  % 方位向采样率
para.f0 = para_data.f0;    % 中心频率
para.Tr = para_data.Tr;    % 脉冲持续时间
para.R0 = para_data.R0;    % 最近点斜距
para.Kr = -para_data.Kr;   % 线性调频率
para.c = para_data.c;      % 光速

% 创建输出矩阵，交替存储实部和虚部
[rows, cols] = size(echo);
output_data = zeros(rows, 2*cols);
for i = 1:cols
    output_data(:, 2*i-1) = real(echo(:,i));  % 实部
    output_data(:, 2*i) = imag(echo(:,i));    % 虚部
end

% 保存为txt文件，使用空格分隔
writematrix(output_data, 'echo.txt', 'Delimiter', ' ');

% 保存参数字典为txt文件
fid = fopen('paras.txt', 'w');
fields = fieldnames(para);
for i = 1:length(fields)
    fprintf(fid, '%s: %f\n', fields{i}, para.(fields{i}));
end
fclose(fid);