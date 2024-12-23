% 读取文件开头的一些字节
fid = fopen('1.dat', 'r');
% 读取前100个字节
header = fread(fid, 100, 'uint8');
fclose(fid);

% 显示为字符
char(header')

% 显示为十六进制
fprintf('%02X ', header);