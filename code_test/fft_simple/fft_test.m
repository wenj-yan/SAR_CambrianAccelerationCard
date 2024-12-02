% 设置输入参数
height = 500;
width = 128*1024;

% 创建输入数据 - 全1复数矩阵
input_data = ones(height, width) + 1i * zeros(height, width);

% 显示部分输入数据
disp('------------------step0: input complex signal----------------');
disp(input_data(1:2, 1:4));  % 只显示一小部分用于验证

% 对每一行进行FFT运算
output_fft = zeros(size(input_data));
for i = 1:height
    output_fft(i,:) = fft(input_data(i,:));
end

% 显示FFT结果
disp('------------------step1: fft result----------------');
% 显示第一行的前30个结果用于验证
result = output_fft(1,1:30);
for i = 1:length(result)
    fprintf('(%.8f,%.8f) ', real(result(i)), imag(result(i)));
    if mod(i,5) == 0
        fprintf('\n');
    end
end

% 计算执行时间
tic;
for i = 1:height
    fft(input_data(i,:));
end
execution_time = toc;

fprintf('\n\n------------------Compute time----------------\n');
fprintf('Computation Time(ms): %.2f\n', execution_time * 1000);