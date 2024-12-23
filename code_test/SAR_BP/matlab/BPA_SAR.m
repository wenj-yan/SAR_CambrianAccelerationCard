%% RD�㷨
clear;close all;clc;

%------------------------------ �������� ----------------------------
%% ��������
% ��Ƶ�źŲ���
c = 3e8;
fc = 1e9;                               % �ź���Ƶ
lambda = c/fc;                          % �ز�����
% ƽ̨����
vr = 100;                               % SAR����ƽ̨�ٶ�
H = 5000;                               % ƽ̨�߶�
%���߲���
D = 4;                                  % ��λ�����߳���

%% ��ʱ�����
% ������Χ
Rg0 = 10e3;                             % ���ĵؾ�
RgL = 1000;                             % ������
R0 = sqrt(Rg0^2+H^2);                   % ����б�ࣨб�ӽ���ȣ�
% �ϳɿ׾�����
La = lambda*R0/D;                       
% �ϳ�ʱ��
Ta = La/vr;                             

% ����ά/��ʱ��ά����
Tw = 5e-6;                              % �������ʱ��
Br = 30e6;                              % �����źŴ���
Kr = Br/Tw;                             % ��Ƶ��
Fr = 2*Br;                              % ��ʱ��ά����Ƶ��
Rmin = sqrt((Rg0-RgL/2)^2+H^2);
Rmax = sqrt((Rg0+RgL/2)^2+H^2+(La/2)^2);
Nfast = ceil(2*(Rmax-Rmin)/c*Fr+Tw*Fr); % ��ʱ��ά����
Nfast = 2^nextpow2(Nfast);              % ���ڿ�ʱ��άFFT�ĵ���
tr = linspace(2*Rmin/c,2*Rmax/c+Tw,Nfast);  
Fr = 1/((2*Rmax/c+Tw-2*Rmin/c)/Nfast);  % ���ʱ��άFFT��������Ĳ�����

%% ��ʱ�����
% ��λ��Χ
Az0 = 10e3;
AL = 1000;
Azmin = Az0-AL/2;                       
Azmax = Az0+AL/2;                  
% ��λά/��ʱ��ά����
Ka = -2*vr^2/lambda/R0;                 % ��ʱ��ά��Ƶ��
Ba = abs(Ka*Ta);                        % ��ʱ��ά����
PRF = 1.2*Ba;                           % �����ظ�Ƶ��
Mslow = ceil((Azmax-Azmin+La)/vr*PRF);  % ��ʱ��ά����/������
Mslow = 2^nextpow2(Mslow);              % ������ʱ��άFFT�ĵ���
ta = linspace((Azmin-La/2)/vr,(Azmax+La/2)/vr,Mslow);  
PRF = 1/((Azmax-Azmin+La)/vr/Mslow);    % ����ʱ��άFFT��������������ظ�Ƶ��

%% ���ܲ���
% �ֱ��ʲ���
Dr = c/2/Br;                            % ����ֱ���
Da = D/2;                               % ��λ�ֱ���

%% Ŀ�����
Ntarget = 5;                            % Ŀ������
Ptarget = [Az0-10,Rg0-20,1;            % Ŀ��λ��\ɢ����Ϣ
           Az0+20,Rg0+30,0.8;
           Az0-30,Rg0+10,1.2;
           Az0+40,Rg0-40,0.9;
           Az0,Rg0,1.5];
          
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

%------------------------------ �ز��ź����� ----------------------------
%% �ز��ź�����
snr = 0;                                % �����
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
%------------------------------ ��������ѹ�� ----------------------------
%% ����ѹ��
thr = tr-2*Rmin/c;
hrc = exp(1i*pi*Kr*thr.^2).*(0<thr&thr<Tw);  % ����άƥ���˲���
SRN_FFT = fft(Srnm,Nfast,2);
HRC_FFT = fft(hrc,Nfast,2);
SSAA = (ones(Mslow,1)*conj(HRC_FFT));
SSAR = (SRN_FFT.*SSAA);

SAR1 =ifft(SSAR,Nfast,2);



%------------------------------ ��λ����ѹ�� ----------------------------
%% �����ֵ
L = 8;
trs = linspace(min(tr),max(tr),L*Nfast);
SAR1f = fft(SAR1,Nfast,2);
SAR11f = [SAR1f(:,1:floor((Nfast+1)/2)),zeros(Mslow,(L-1)*Nfast),...
    SAR1f(:,floor((Nfast+1)/2)+1:end)];
SAR2 = ifft(SAR11f,L*Nfast,2);
figure;
imagesc(trs,ta,255-abs(SAR1));                       
xlabel('��ʱ��');
ylabel('��ʱ��');
title('��ʱ��ά��ѹ(��ֵǰ��');
colormap(gray)
figure;
imagesc(trs,ta,255-abs(SAR2));                       
xlabel('��ʱ��');
ylabel('��ʱ��');
title('��ʱ��ά��ѹ(��ֵ��');
colormap(gray)

%% �����ʷ�
% ̽�ⷶΧ��ɢ��
Rg = Rg0-50:0.1:Rg0+50;
Nr =length(Rg);
Az = Az0-50:0.05:Az0+50;
Na = length(Az);

%% SAR ����δ���о���ά��ֵ��
SAR3 = zeros(Na,Nr);
for i = 1:Na    
    for j = 1:Nr
        % �����㶯����ȷ��
        vr_ta= vr*ta;
        az_diff = vr_ta-Az(i) ;
        Rt = sqrt((az_diff).^2+Rg(j)^2+H^2);
        tau = 2*Rt/c;
        nr = round((tau-min(trs))*Fr);
        % ��λ����
        rd = zeros(1,Mslow);
        for m=1:Mslow
            rd(m) = SAR1(m,nr(m));
            
        end
        rd = rd.*exp(1j*4*pi*fc/c*Rt);
        % ����ۼ�
        SAR3(i,j) = sum(rd);
    end
    disp(i);
end
figure;
mesh(Rg,Az,abs(SAR3));                       
xlabel('�ؾ�/m');
ylabel('��λ/m');
title('SARͼ��(δ��ֵ��');
colormap(gray)

figure;
imagesc(Rg,Az,255-abs(SAR3));                       
xlabel('�ؾ�/m');
ylabel('��λ/m');
title('SARͼ��(��ֵǰ��');
colormap(gray)

figure;
Max = max(max(abs(SAR3)));
contourf(Rg,Az,abs(SAR3),[0.707*Max,Max],'b');  
grid on
xlabel('\rightarrow\it�ؾ�/m');
ylabel('\it��λ/m\leftarrow');
title('����Ŀ��ֱ���(��ֵǰ��');
colormap(gray);

%% SAR ���񣨽��о���ά��ֵ��
uSAR4 = zeros(Na,Nr);
for i = 1:Na    
    for j = 1:Nr
        % �����㶯����ȷ��
        Rt = sqrt((vr*ta-Az(i)).^2+Rg(j)^2+H^2);
        tau = 2*Rt/c;
        nr = round((tau-min(trs))*Fr*L);
        % ��λ����
        rd = zeros(1,Mslow);
        for m=1:Mslow
            rd(m) = SAR2(m,nr(m));
            disp(nr);
            
        end
        rd = rd.*exp(1j*4*pi*fc/c*Rt);
        % ����ۼ�
        SAR4(i,j) = sum(rd);
    end
    disp(i);
end

figure;
mesh(Rg,Az,abs(SAR4));                       
xlabel('�ؾ�/m');
ylabel('��λ/m');
title('SARͼ��(��ֵ��');
colormap(gray)

figure;
imagesc(Rg,Az,255-abs(SAR4));                       
xlabel('�ؾ�/m');
ylabel('��λ/m');
title('SARͼ��(��ֵ��');
colormap(gray)

figure;
Max = max(max(abs(SAR4)));
contourf(Rg,Az,abs(SAR4),[0.707*Max,Max],'b');  
grid on
xlabel('\rightarrow\it�ؾ�/m');
ylabel('\it��λ/m\leftarrow');
title('����Ŀ��ֱ���(��ֵ��');
colormap(gray);


