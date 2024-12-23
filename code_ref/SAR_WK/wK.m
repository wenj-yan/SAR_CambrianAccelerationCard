%% wK�㷨
clear;close all;clc;

%% ����˵��
% ���ӳ���
% wKA�㷨
% �ϳ�ʱ���̶�
% ������ͣģʽ���ɻز��ź�(���������ھ��뵼�ºϳ�ʱ��Ĳ��죩

% �ھ���Ƶ��-��λƵ�����һ��ѹ������
% �ھ���Ƶ��-��λƵ�����Stolt��ֵ����
% �ھ���Ƶ��-��λƵ���Ͻ��вο������µľ����ƶ�
% ����άIFFT����
% ��λάIFFT����

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
Rg0 = 9000;                             % ���ĵؾ�
RgL = 1000;                              % ������

% ƽ̨����
vr = 20;                               % SAR����ƽ̨�ٶ�
H = 1000;                               % ƽ̨�߶�
R0 = sqrt(Rg0^2+H^2);                   % ����б�ࣨб�ӽ���ȣ�

%���߲���
D = 4;                                  % ��λ�����߳���
La = lambda*R0/D;                       % �ϳɿ׾�����
Ta = La/vr;                             % �ϳ�ʱ��

% ��λά/��ʱ��ά����
Ka = -2*vr^2/lambda/R0;                 % ��ʱ��ά��Ƶ��
Ba = abs(Ka*Ta);                        % ��ʱ��ά����
PRF = 1.2*Ba;                           % �����ظ�Ƶ��
Mslow = ceil((Azmax-Azmin+La)/vr*PRF);  % ��ʱ��ά����/������
Mslow = 2^nextpow2(Mslow);              % ������ʱ��άFFT�ĵ���
ta = linspace((Azmin-La/2)/vr,(Azmax+La/2)/vr,Mslow);  
PRF = 1/((Azmax-Azmin+La)/vr/Mslow);    % ����ʱ��άFFT��������������ظ�Ƶ��
Az = ta*vr;

% ����ά/��ʱ��ά����
Tw = 5e-6;                              % �������ʱ��
Br = 30e6;                              % �����źŴ���
Kr = Br/Tw;                             % ��Ƶ��
Fr = 2*Br;                              % ��ʱ��ά����Ƶ��
R1 = sqrt((Rg0-RgL/2)^2+H^2);
R2 = sqrt((Rg0+RgL/2)^2+H^2);

Rmin = sqrt((Rg0-RgL/2)^2+H^2);
Rmax = sqrt((Rg0+RgL/2)^2+H^2+(La/2)^2);
Nfast = ceil(2*(Rmax-Rmin)/c*Fr+Tw*Fr); % ��ʱ��ά����
Nfast = 2^nextpow2(Nfast);              % ���ڿ�ʱ��άFFT�ĵ���
tr = linspace(2*Rmin/c-Tw/2,2*Rmax/c+Tw/2,Nfast);  
Fr = 1/((2*Rmax/c+Tw-2*Rmin/c)/Nfast);  % ���ʱ��άFFT��������Ĳ�����
R = tr*c/2;


% �ֱ��ʲ���
Dr = c/2/Br;                            % ����ֱ���
Da = D/2;                               % ��λ�ֱ���

% ��Ŀ�����
Ptarget=[Az0+300,Rg0-300,1;Az0+300,Rg0,1;Az0+300,Rg0+300,1;
    Az0,Rg0-300,1;Az0,Rg0,1;Az0,Rg0+300,1;
    Az0-300,Rg0-300,1;Az0-300,Rg0,1;Az0-300,Rg0+300,1];  
Ntarget = size(Ptarget,1);                            % Ŀ������
fprintf('���������\n');     
fprintf('��ʱ��/����ά�������ʣ�%.4f\n',Fr/Br);     
fprintf('��ʱ��/����ά����������%d\n',Nfast);     
fprintf('��ʱ��/��λά�������ʣ�%.4f\n',PRF/Ba);     
fprintf('��ʱ��/��λά����������%d\n',Mslow);     
fprintf('����ֱ��ʣ�%.1fm\n',Dr);     
fprintf('�������ֱ��ʣ�%.1fm\n',Da);     
fprintf('�ϳɿ׾����ȣ�%.1fm\n',La);     
disp('Ŀ�귽λ/�ؾ�/б�ࣺ');
disp([Ptarget(:,1),Ptarget(:,2),sqrt(Ptarget(:,2).^2+H^2)])

%% �ز��ź�����(����ʱ��-��λʱ��
snr = 0;                                % �����
Srnm=zeros(Mslow,Nfast);
for k=1:1:Ntarget
    sigmak=Ptarget(k,3);
    Azk=ta*vr-Ptarget(k,1);
    Rk=sqrt(Azk.^2+Ptarget(k,2)^2+H^2);
    tauk=2*Rk/c;
    tk=ones(Mslow,1)*tr-tauk'*ones(1,Nfast);
    phasek=pi*Kr*tk.^2-(4*pi/lambda)*(Rk'*ones(1,Nfast));
    Srnm=Srnm+sigmak*exp(1i*phasek).*(-Tw/2<tk&tk<Tw/2).*((abs(Azk)<La/2)'*ones(1,Nfast));
end
% Srnm = awgn(Srnm,snr,'measured');

figure;
mesh(tr,ta,abs(Srnm));
xlabel('����ʱ��/s');
ylabel('��λʱ��/s');
title('�ز��ھ���ʱ��λʱ���ϵı�ʾ');
view(2);
%-----------------------------wK�����㷨---------------------------
%% ��λά����Ҷ�任�������������
Srnm1 = fftshift(fft(Srnm,Mslow,1),1); 
% ��λƵ�򻮷�
ft = linspace(-PRF/2,PRF/2,Mslow).';
figure;
mesh(tr,ft,abs(Srnm1));
xlabel('����ʱ��/s');
ylabel('��λƵ��/Hz');
title('�ز��ھ�����������ϵı�ʾ');
view(2);

%% ����ά����Ҷ�任������Ƶ��λƵ��
Srnm2 = fftshift(fft(Srnm1,Nfast,2),2); 
% ��λƵ�򻮷�
ftau = linspace(-Fr/2,Fr/2,Nfast);
figure;
mesh(ftau,ft,abs(Srnm2));
xlabel('����Ƶ��/Hz');
ylabel('��λƵ��/Hz');
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

%% �ο�����ƽ��
H_ref = exp(-1j*4*pi*R_ref/c*ones(Mslow,1)*ftau);
Srnm4 = Srnm4.*H_ref;

%% ������IFFT
Srnm5 = ifft(ifftshift(Srnm4,2),Nfast,2); 

figure;
mesh(R,ft,abs(Srnm5));
xlabel('����/m');
ylabel('��λƵ��/Hz');
title('����ѹ����');
view(2);

%% ��λ��IFFT
Srnm6 = ifft(ifftshift(Srnm5,1),Mslow,1); 
figure;
mesh(R,Az,abs(Srnm6));
xlabel('����/m');
ylabel('��λ/m');
title('��λѹ����');
view(2);

%% ��������SARͼ�����ݻ�ȡ�棩

m1 = round(La/2/vr*PRF);
m2 = round(((Azmax-Azmin)/vr+La/2/vr)*PRF);
dA = vr/PRF;

n1 = round(Tw/2*Fr);
n2 = round((2*(R2-Rmin)/c+Tw/2)*Fr); 
dR = round(c/2/Fr);

figure;
mesh(R(n1:n2),Az(m1:m2),abs(Srnm6(m1:m2,n1:n2)));
xlabel('����/m');
ylabel('��λ/m');
title('���ݻ�ȡ��SARͼ��');
view(2);
%--------------------------ת��Ϊ����SARͼ��----------------------------------
%% �ؾ�ת��
Rg = linspace(Rg0-RgL/2,Rg0+RgL/2,1024);
[X,Y] = meshgrid(R,Az);
Rt = sqrt(Rg.^2+H^2);
[Xq,Yq] = meshgrid(Rt,Az);
SAR = interp2(X,Y,abs(Srnm6),Xq,Yq,'linear');

figure;
mesh(Rg,Az,SAR);
xlabel('�ؾ�/m');
ylabel('��λ/m');
title('SARͼ��ͶӰ�����棩')

%% 3dBĿ��ͼ�񣨵���SARͼ��
Max1 = max(max(SAR));
figure;
contourf(Rg,Az,SAR,[0.707*Max1,Max1],'b');grid on
xlabel('\rightarrow\it�ؾ�/m');
ylabel('\it��λ/m\leftarrow');
title('ͶӰ������Ŀ��ֱ���');
colormap(gray);








