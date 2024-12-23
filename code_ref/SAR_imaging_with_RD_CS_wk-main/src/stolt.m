x = [1, 3, 5, 7, 9];        % x坐标
y = [2, 6, 4, 8, 5];        % y坐标

% 画出这些点
plot(x, y, 'ro', 'MarkerSize', 10)
hold on
grid on

% 我们想知道这些位置的值
xq = 1:0.1:9;  % 从1到9，每隔0.5取一个点

% 用不同方式连接
y1 = interp1(x, y, xq, 'linear');     % 方式1：直接用直线连接
y2 = interp1(x, y, xq, 'nearest');    % 方式2：找最近的点
y3 = interp1(x, y, xq, 'spline');     % 方式3：用平滑的曲线连接

% 画出来看看区别
plot(xq, y1, 'b-', 'LineWidth', 2)    % 蓝线-直线连接
plot(xq, y2, 'g--', 'LineWidth', 2)   % 绿线-最近点
plot(xq, y3, 'r-.', 'LineWidth', 2)   % 红线-平滑曲线

legend('原始点', '直线连接', '最近点', '平滑曲线')