%% RD算法
clear;close all;clc;

%------------------------------ 参数设置 ----------------------------
%% 基本参数
% 载频信号参数
c = 3e8;
fc = 1e9;                               % 信号载频
lambda = c/fc;                          % 载波波长
% 平台参数
vr = 100;                               % SAR搭载平台速度
H = 5000;                               % 平台高度
%天线参数
D = 4;                                  % 方位向天线长度

%% 快时间参数
% 距离向范围
Rg0 = 10e3;                             % 中心地距
RgL = 1000;                             % 测绘带宽
R0 = sqrt(Rg0^2+H^2);                   % 中心斜距（斜视角零度）
% 合成孔径长度
La = lambda*R0/D;                       
% 合成时长
Ta = La/vr;                             

% 距离维/快时间维参数
Tw = 5e-6;                              % 脉冲持续时间
Br = 30e6;                              % 发射信号带宽
Kr = Br/Tw;                             % 调频率
Fr = 2*Br;                              % 快时间维采样频率
Rmin = sqrt((Rg0-RgL/2)^2+H^2);
Rmax = sqrt((Rg0+RgL/2)^2+H^2+(La/2)^2);
Nfast = ceil(2*(Rmax-Rmin)/c*Fr+Tw*Fr); % 快时间维点数
Nfast = 2^nextpow2(Nfast);              % 用于快时间维FFT的点数
tr = linspace(2*Rmin/c,2*Rmax/c+Tw,Nfast);  
Fr = 1/((2*Rmax/c+Tw-2*Rmin/c)/Nfast);  % 与快时间维FFT点数相符的采样率

%% 慢时间参数
% 方位向范围
Az0 = 10e3;
AL = 1000;
Azmin = Az0-AL/2;                       
Azmax = Az0+AL/2;                  
% 方位维/慢时间维参数
Ka = -2*vr^2/lambda/R0;                 % 慢时间维调频率
Ba = abs(Ka*Ta);                        % 慢时间维带宽
PRF = 1.2*Ba;                           % 脉冲重复频率
Mslow = ceil((Azmax-Azmin+La)/vr*PRF);  % 慢时间维点数/脉冲数
Mslow = 2^nextpow2(Mslow);              % 用于慢时间维FFT的点数
ta = linspace((Azmin-La/2)/vr,(Azmax+La/2)/vr,Mslow);  
PRF = 1/((Azmax-Azmin+La)/vr/Mslow);    % 与慢时间维FFT点数相符的脉冲重复频率

%% 性能参数
% 分辨率参数
Dr = c/2/Br;                            % 距离分辨率
Da = D/2;                               % 方位分辨率

%% 目标参数
Ntarget = 5;                            % 目标数量
Ptarget = [Az0-10,Rg0-20,1;            % 目标位置\散射信息
           Az0+20,Rg0+30,0.8;
           Az0-30,Rg0+10,1.2;
           Az0+40,Rg0-40,0.9;
           Az0,Rg0,1.5];
          
fprintf('仿真参数：\n');     
fprintf('快时间/距离维过采样率：%.4f\n',Fr/Br);     
fprintf('快时间/距离维采样点数：%d\n',Nfast);     
fprintf('慢时间/方位维过采样率：%.4f\n',PRF/Ba);     
fprintf('慢时间/方位维采样点数：%d\n',Mslow);     
fprintf('距离分辨率：%.1fm\n',Dr);     
fprintf('距离横向分辨率：%.1fm\n',Da);     
fprintf('合成孔径长度：%.1fm\n',La);     
disp('目标方位/地距/斜距：');
disp([Ptarget(:,1),Ptarget(:,2),sqrt(Ptarget(:,2).^2+H^2)])

%------------------------------ 回波信号生成 ----------------------------
%% 回波信号生成
snr = 0;                                % 信噪比
Srnm = zeros(Mslow,Nfast);
for k = 1:1:Ntarget
    sigmak = Ptarget(k,3);
    Azk = ta*vr-Ptarget(k,1);
    Rk = sqrt(Azk.^2+Ptarget(k,2)^2+H^2);
    tauk = 2*Rk/c;
    tk = ones(Mslow,1)*tr-tauk'*ones(1,Nfast);
    phasek = pi*Kr*tk.^2-(4*pi/lambda)*(Rk'*ones(1,Nfast));
    Srnm = Srnm+sigmak*exp(1i*phasek).*(0<tk&tk<Tw).*((abs(Azk)<La/2)'*ones(1,Nfast));
end                                
%%Srnm = awgn(Srnm,snr,'measured');
%------------------------------ 距离脉冲压缩 ----------------------------
%% 距离压缩
thr = tr-2*Rmin/c;
hrc = exp(1i*pi*Kr*thr.^2).*(0<thr&thr<Tw);  % 距离维匹配滤波器
SRN_FFT = fft(Srnm,Nfast,2);
HRC_FFT = fft(hrc,Nfast,2);
SSAA = (ones(Mslow,1)*conj(HRC_FFT));
SSAR = (SRN_FFT.*SSAA);

SAR1 =ifft(SSAR,Nfast,2);



%------------------------------ 方位脉冲压缩 ----------------------------
%% 距离插值
L = 8;
trs = linspace(min(tr),max(tr),L*Nfast);
SAR1f = fft(SAR1,Nfast,2);
SAR11f = [SAR1f(:,1:floor((Nfast+1)/2)),zeros(Mslow,(L-1)*Nfast),...
    SAR1f(:,floor((Nfast+1)/2)+1:end)];
SAR2 = ifft(SAR11f,L*Nfast,2);
figure;
imagesc(trs,ta,255-abs(SAR1));                       
xlabel('快时间');
ylabel('慢时间');
title('快时间维脉压(插值前）');
colormap(gray)
figure;
imagesc(trs,ta,255-abs(SAR2));                       
xlabel('快时间');
ylabel('慢时间');
title('快时间维脉压(插值后）');
colormap(gray)

%% 网格剖分
% 探测范围离散化
Rg = Rg0-50:0.1:Rg0+50;
Nr =length(Rg);
Az = Az0-50:0.05:Az0+50;
Na = length(Az);

%% SAR 成像（未进行距离维插值）
SAR3 = zeros(Na,Nr);
for i = 1:Na    
    for j = 1:Nr
        % 距离徙动曲线确定
        vr_ta= vr*ta;
        az_diff = vr_ta-Az(i) ;
        Rt = sqrt((az_diff).^2+Rg(j)^2+H^2);
        tau = 2*Rt/c;
        nr = round((tau-min(trs))*Fr);
        % 相位补偿
        rd = zeros(1,Mslow);
        for m=1:Mslow
            rd(m) = SAR1(m,nr(m));
            
        end
        rd = rd.*exp(1j*4*pi*fc/c*Rt);
        % 相干累加
        SAR3(i,j) = sum(rd);
    end
    disp(i);
end
figure;
mesh(Rg,Az,abs(SAR3));                       
xlabel('地距/m');
ylabel('方位/m');
title('SAR图像(未插值）');
colormap(gray)

figure;
imagesc(Rg,Az,255-abs(SAR3));                       
xlabel('地距/m');
ylabel('方位/m');
title('SAR图像(插值前）');
colormap(gray)

figure;
Max = max(max(abs(SAR3)));
contourf(Rg,Az,abs(SAR3),[0.707*Max,Max],'b');  
grid on
xlabel('\rightarrow\it地距/m');
ylabel('\it方位/m\leftarrow');
title('地面目标分辨率(插值前）');
colormap(gray);

%% SAR 成像（进行距离维插值）
uSAR4 = zeros(Na,Nr);
for i = 1:Na    
    for j = 1:Nr
        % 距离徙动曲线确定
        Rt = sqrt((vr*ta-Az(i)).^2+Rg(j)^2+H^2);
        tau = 2*Rt/c;
        nr = round((tau-min(trs))*Fr*L);
        % 相位补偿
        rd = zeros(1,Mslow);
        for m=1:Mslow
            rd(m) = SAR2(m,nr(m));
            disp(nr);
            
        end
        rd = rd.*exp(1j*4*pi*fc/c*Rt);
        % 相干累加
        SAR4(i,j) = sum(rd);
    end
    disp(i);
end

figure;
mesh(Rg,Az,abs(SAR4));                       
xlabel('地距/m');
ylabel('方位/m');
title('SAR图像(插值后）');
colormap(gray)

figure;
imagesc(Rg,Az,255-abs(SAR4));                       
xlabel('地距/m');
ylabel('方位/m');
title('SAR图像(插值后）');
colormap(gray)

figure;
Max = max(max(abs(SAR4)));
contourf(Rg,Az,abs(SAR4),[0.707*Max,Max],'b');  
grid on
xlabel('\rightarrow\it地距/m');
ylabel('\it方位/m\leftarrow');
title('地面目标分辨率(插值后）');
colormap(gray);


