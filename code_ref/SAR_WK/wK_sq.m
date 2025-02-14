%% wK算法
clear;close all;clc;

%% 程序说明
% 斜视成像
% wKA算法
% 合成时长由距离、波束宽度决定
% 基于走停模式生成回波信号(不考虑由于距离导致合成时间的差异）

%% 参数设置
% 载频信号参数
c = 3e8;
fc = 1e9;                               % 信号载频
lambda = c/fc;                          % 载波波长

% 探测范围(地面范围）
Az0 = 9000;
AL = 1000;
Azmin = Az0-AL/2;                       % 方位向范围
Azmax = Az0+AL/2;                  
Rg0 = 9000;                            % 中心地距
RgL = 1000;                             % 测绘带宽

% 平台参数
theta = -50;                            % 斜视角(负号表示前视)
vr = 20;                                % SAR搭载平台速度
fd = -2*vr*sind(theta)/lambda;
H = 1000;                               % 平台高度
R0 = sqrt(Rg0^2+H^2);                   % 中心地距对应最短斜距（斜视角零度）
Ka = -2*vr^2/lambda/R0*cosd(theta)^3;   % 慢时间维调频率
R1 = sqrt((Rg0-RgL/2)^2+H^2);           % 最小地距对应最短斜距（斜视角零度）
R2 = sqrt((Rg0+RgL/2)^2+H^2);           % 最大地距对应最短斜距（斜视角零度）

%天线参数
D = 4;                                  % 方位向天线长度
phi = lambda/D;                         % 波束宽度
As = phi*R0/cosd(theta)^2;              % 合成孔径长度

% 方位维/慢时间维参数
tmin = Azmin/vr;
tmax = Azmax/vr;

t11 = tmin+R1*tand(theta)/vr-phi*R1/cosd(theta)^2/2/vr;
t12 = tmin+R2*tand(theta)/vr-phi*R2/cosd(theta)^2/2/vr;
t21 = tmax+R1*tand(theta)/vr+phi*R1/cosd(theta)^2/2/vr;
t22 = tmax+R2*tand(theta)/vr+phi*R2/cosd(theta)^2/2/vr;

t1 = min(t11,t12);
t2 = max(t21,t22);
Ba = 2*vr/D*cosd(theta);                % 慢时间维带宽
PRF = 1.2*Ba;                           % 脉冲重复频率
Mslow = ceil((t2-t1)*PRF);              % 慢时间维点数/脉冲数
Mslow = 2^nextpow2(Mslow);              % 用于慢时间维FFT的点数
ta = linspace(t1,t2,Mslow);  
PRF = 1/((t2-t1)/Mslow);                % 与慢时间维FFT点数相符的脉冲重复频率

% 距离维/快时间维参数
Tw = 5e-6;                              % 脉冲持续时间
Br = 30e6;                              % 发射信号带宽
Kr = Br/Tw;                             % 调频率
Fr = 2*Br;                              % 快时间维采样频率
if (t2>tmin && t1<tmin) 
    Rmin = sqrt((Rg0-RgL/2)^2+H^2);
elseif tmin>t2
    Rmin = sqrt((Rg0-RgL/2)^2+H^2+((tmin-t2)*vr)^2);
end
if (t1< tmax && t2>tmax) 
    Rmin = sqrt((Rg0-RgL/2)^2+H^2);
elseif tmax<t1
    Rmin = sqrt((Rg0-RgL/2)^2+H^2+((t1-tmax)*vr)^2);
end
if Rmin<Tw*c/2
    error('距离太近');
end
Rmax = sqrt((R2*tand(abs(theta))+phi*R2/cosd(theta)^2/2)^2+R2^2);
if Rmax>c/PRF/2
    error('距离太远');
end
Nfast = ceil(2*(Rmax-Rmin)/c*Fr+Tw*Fr); % 快时间维点数
Nfast = 2^nextpow2(Nfast);              % 用于快时间维FFT的点数
tr = linspace(2*Rmin/c-Tw/2,2*Rmax/c+Tw/2,Nfast);  
Fr = 1/((2*Rmax/c+Tw-2*Rmin/c)/Nfast);  % 与快时间维FFT点数相符的采样率

% 点目标参数
Ptarget=[Az0+500,Rg0-500,1;Az0+500,Rg0,1;Az0+500,Rg0+500,1;
    Az0,Rg0-500,1;Az0,Rg0,1;Az0,Rg0+500,1;
    Az0-500,Rg0-500,1;Az0-500,Rg0,1;Az0-500,Rg0+500,1];  
Ntarget = size(Ptarget,1);              % 目标数量
fprintf('仿真参数：\n');     
fprintf('快时间/距离维过采样率：%.4f\n',Fr/Br);     
fprintf('快时间/距离维采样点数：%d\n',Nfast);     
fprintf('慢时间/方位维过采样率：%.4f\n',PRF/Ba);     
fprintf('慢时间/方位维采样点数：%d\n',Mslow);     
disp('目标方位/地距/斜距：');
disp([Ptarget(:,1),Ptarget(:,2),sqrt(Ptarget(:,2).^2+H^2)])

%% 回波信号生成(距离时域-方位时域）
snr = 0;                                % 信噪比
Srnm=zeros(Mslow,Nfast);
for k=1:1:Ntarget
    sigmak=Ptarget(k,3);
    R0k = sqrt(Ptarget(k,2)^2+H^2);
    Ack = R0k*tand(theta);
    Ask = phi*R0k/cosd(theta)^2;
    Azk = ta*vr-Ptarget(k,1);
    Rtk=sqrt(Azk.^2+Ptarget(k,2)^2+H^2);
    tauk=2*Rtk/c;
    tk=ones(Mslow,1)*tr-tauk'*ones(1,Nfast);
    phasek=pi*Kr*tk.^2-(4*pi/lambda)*(Rtk'*ones(1,Nfast));
    Srnm=Srnm+sigmak*exp(1i*phasek).*(-Tw/2<tk&tk<Tw/2).*((abs(Azk-Ack)<Ask/2)'*ones(1,Nfast));
end
% Srnm = awgn(Srnm,snr,'measured');
figure;
mesh(tr,ta,abs(Srnm));
xlabel('距离时域');
ylabel('方位时域');
title('回波在距离时域方位时域上的表示');
view(2);
%-----------------------------wK成像算法---------------------------
%% 方位维傅里叶变换（距离多普勒域）
% 方位频偏搬移
H_fd = exp(-1j*2*pi*fd*ta).'*ones(1,Nfast);
Srnm = Srnm.*H_fd;
Srnm1 = fftshift(fft(Srnm,Mslow,1),1); 

% 方位频域划分
ft = linspace(-PRF/2,PRF/2,Mslow).'+fd;
% figure;
% mesh(tr,ft,abs(Srnm1));
% xlabel('距离时域');
% ylabel('方位频域');
% title('回波在距离多普勒域上的表示');
% view(2);

%% 距离维傅里叶变换（距离频域方位频域）
Srnm2 = fftshift(fft(Srnm1,Nfast,2),2); 
% 方位频域划分
ftau = linspace(-Fr/2,Fr/2,Nfast);
figure;
mesh(ftau,ft,abs(Srnm2));
xlabel('距离频域');
ylabel('方位频域');
title('回波在距离频域方位频域上的表示');
view(2);

%% 一致压缩
% 参考距离
R_ref = R0;
% 线性调频变标方程
H_RFM = exp(1i*4*pi*R_ref/c*sqrt((ones(Mslow,1)*(fc+ftau).^2)-c^2*ft.^2*ones(1,Nfast)/4/vr^2)+1i*pi*ones(Mslow,1)*ftau.^2/Kr); 
% 变标处理
Srnm3 = Srnm2.*H_RFM;

%% Stolt插值
Srnm4 = zeros(size(Srnm3));
for i=1:1:Mslow
    xx = sqrt((fc+ftau).^2+c^2*ft(i)/4/vr^2)-fc;
    pn = (abs(xx)<Fr/2);
    PN = find(pn~=0);
    a = interp1(ftau,Srnm3(i,:),xx,'linear');
    Srnm4(i,PN) = a(PN);
end

%% 距离向IFFT
Srnm5 = ifft(ifftshift(Srnm4,2),Nfast,2); 

% figure;
% mesh(abs(Srnm5));
% xlabel('距离');
% ylabel('方位');
% title('距离维IFFT');
% view(2);

%% 方位向IFFT
Srnm6 = ifft(ifftshift(Srnm5,1),Mslow,1); 
figure;
mesh(abs(Srnm6));
xlabel('距离');
ylabel('方位');
title('方位维IFFT');
view(2);

