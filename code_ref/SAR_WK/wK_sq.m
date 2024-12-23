%% wK�㷨
clear;close all;clc;

%% ����˵��
% б�ӳ���
% wKA�㷨
% �ϳ�ʱ���ɾ��롢������Ⱦ���
% ������ͣģʽ���ɻز��ź�(���������ھ��뵼�ºϳ�ʱ��Ĳ��죩

%% ��������
% ��Ƶ�źŲ���
c = 3e8;
fc = 1e9;                               % �ź���Ƶ
lambda = c/fc;                          % �ز�����

% ̽�ⷶΧ(���淶Χ��
Az0 = 9000;
AL = 1000;
Azmin = Az0-AL/2;                       % ��λ��Χ
Azmax = Az0+AL/2;                  
Rg0 = 9000;                            % ���ĵؾ�
RgL = 1000;                             % ������

% ƽ̨����
theta = -50;                            % б�ӽ�(���ű�ʾǰ��)
vr = 20;                                % SAR����ƽ̨�ٶ�
fd = -2*vr*sind(theta)/lambda;
H = 1000;                               % ƽ̨�߶�
R0 = sqrt(Rg0^2+H^2);                   % ���ĵؾ��Ӧ���б�ࣨб�ӽ���ȣ�
Ka = -2*vr^2/lambda/R0*cosd(theta)^3;   % ��ʱ��ά��Ƶ��
R1 = sqrt((Rg0-RgL/2)^2+H^2);           % ��С�ؾ��Ӧ���б�ࣨб�ӽ���ȣ�
R2 = sqrt((Rg0+RgL/2)^2+H^2);           % ���ؾ��Ӧ���б�ࣨб�ӽ���ȣ�

%���߲���
D = 4;                                  % ��λ�����߳���
phi = lambda/D;                         % �������
As = phi*R0/cosd(theta)^2;              % �ϳɿ׾�����

% ��λά/��ʱ��ά����
tmin = Azmin/vr;
tmax = Azmax/vr;

t11 = tmin+R1*tand(theta)/vr-phi*R1/cosd(theta)^2/2/vr;
t12 = tmin+R2*tand(theta)/vr-phi*R2/cosd(theta)^2/2/vr;
t21 = tmax+R1*tand(theta)/vr+phi*R1/cosd(theta)^2/2/vr;
t22 = tmax+R2*tand(theta)/vr+phi*R2/cosd(theta)^2/2/vr;

t1 = min(t11,t12);
t2 = max(t21,t22);
Ba = 2*vr/D*cosd(theta);                % ��ʱ��ά����
PRF = 1.2*Ba;                           % �����ظ�Ƶ��
Mslow = ceil((t2-t1)*PRF);              % ��ʱ��ά����/������
Mslow = 2^nextpow2(Mslow);              % ������ʱ��άFFT�ĵ���
ta = linspace(t1,t2,Mslow);  
PRF = 1/((t2-t1)/Mslow);                % ����ʱ��άFFT��������������ظ�Ƶ��

% ����ά/��ʱ��ά����
Tw = 5e-6;                              % �������ʱ��
Br = 30e6;                              % �����źŴ���
Kr = Br/Tw;                             % ��Ƶ��
Fr = 2*Br;                              % ��ʱ��ά����Ƶ��
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
    error('����̫��');
end
Rmax = sqrt((R2*tand(abs(theta))+phi*R2/cosd(theta)^2/2)^2+R2^2);
if Rmax>c/PRF/2
    error('����̫Զ');
end
Nfast = ceil(2*(Rmax-Rmin)/c*Fr+Tw*Fr); % ��ʱ��ά����
Nfast = 2^nextpow2(Nfast);              % ���ڿ�ʱ��άFFT�ĵ���
tr = linspace(2*Rmin/c-Tw/2,2*Rmax/c+Tw/2,Nfast);  
Fr = 1/((2*Rmax/c+Tw-2*Rmin/c)/Nfast);  % ���ʱ��άFFT��������Ĳ�����

% ��Ŀ�����
Ptarget=[Az0+500,Rg0-500,1;Az0+500,Rg0,1;Az0+500,Rg0+500,1;
    Az0,Rg0-500,1;Az0,Rg0,1;Az0,Rg0+500,1;
    Az0-500,Rg0-500,1;Az0-500,Rg0,1;Az0-500,Rg0+500,1];  
Ntarget = size(Ptarget,1);              % Ŀ������
fprintf('���������\n');     
fprintf('��ʱ��/����ά�������ʣ�%.4f\n',Fr/Br);     
fprintf('��ʱ��/����ά����������%d\n',Nfast);     
fprintf('��ʱ��/��λά�������ʣ�%.4f\n',PRF/Ba);     
fprintf('��ʱ��/��λά����������%d\n',Mslow);     
disp('Ŀ�귽λ/�ؾ�/б�ࣺ');
disp([Ptarget(:,1),Ptarget(:,2),sqrt(Ptarget(:,2).^2+H^2)])

%% �ز��ź�����(����ʱ��-��λʱ��
snr = 0;                                % �����
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
xlabel('����ʱ��');
ylabel('��λʱ��');
title('�ز��ھ���ʱ��λʱ���ϵı�ʾ');
view(2);
%-----------------------------wK�����㷨---------------------------
%% ��λά����Ҷ�任�������������
% ��λƵƫ����
H_fd = exp(-1j*2*pi*fd*ta).'*ones(1,Nfast);
Srnm = Srnm.*H_fd;
Srnm1 = fftshift(fft(Srnm,Mslow,1),1); 

% ��λƵ�򻮷�
ft = linspace(-PRF/2,PRF/2,Mslow).'+fd;
% figure;
% mesh(tr,ft,abs(Srnm1));
% xlabel('����ʱ��');
% ylabel('��λƵ��');
% title('�ز��ھ�����������ϵı�ʾ');
% view(2);

%% ����ά����Ҷ�任������Ƶ��λƵ��
Srnm2 = fftshift(fft(Srnm1,Nfast,2),2); 
% ��λƵ�򻮷�
ftau = linspace(-Fr/2,Fr/2,Nfast);
figure;
mesh(ftau,ft,abs(Srnm2));
xlabel('����Ƶ��');
ylabel('��λƵ��');
title('�ز��ھ���Ƶ��λƵ���ϵı�ʾ');
view(2);

%% һ��ѹ��
% �ο�����
R_ref = R0;
% ���Ե�Ƶ��귽��
H_RFM = exp(1i*4*pi*R_ref/c*sqrt((ones(Mslow,1)*(fc+ftau).^2)-c^2*ft.^2*ones(1,Nfast)/4/vr^2)+1i*pi*ones(Mslow,1)*ftau.^2/Kr); 
% ��괦��
Srnm3 = Srnm2.*H_RFM;

%% Stolt��ֵ
Srnm4 = zeros(size(Srnm3));
for i=1:1:Mslow
    xx = sqrt((fc+ftau).^2+c^2*ft(i)/4/vr^2)-fc;
    pn = (abs(xx)<Fr/2);
    PN = find(pn~=0);
    a = interp1(ftau,Srnm3(i,:),xx,'linear');
    Srnm4(i,PN) = a(PN);
end

%% ������IFFT
Srnm5 = ifft(ifftshift(Srnm4,2),Nfast,2); 

% figure;
% mesh(abs(Srnm5));
% xlabel('����');
% ylabel('��λ');
% title('����άIFFT');
% view(2);

%% ��λ��IFFT
Srnm6 = ifft(ifftshift(Srnm5,1),Mslow,1); 
figure;
mesh(abs(Srnm6));
xlabel('����');
ylabel('��λ');
title('��λάIFFT');
view(2);

