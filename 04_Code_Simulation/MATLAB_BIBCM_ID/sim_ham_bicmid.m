clear all
tic

m = 3; P = de2bi((1:2^m-1),m,'left-msb')'; P(:,(2.^(0:m-1))) = [];
% Extended Hamming codes
P = [P; rem(sum(P) + ones(1,2^m-1-m), 2)];G = [eye(2^m-1-m),P']; H = [P,eye(m+1)];
% Hamming codes
%H = [P,eye(m)]; G = [eye(2^m-1-m),P']; 
%load ChecMatrix_Golay.mat G H;

[rows,cols]=size(H);
%4QAM conctellation
%M = 4; maprule=[1,2,3,4]; Es = 10;

%8QAM constelation and mapping rule
%M = 8; maprule = [1,7,6,4,5,3,2,8]; Es = 1;  %for PSK
%M = 8; maprule = [1,2,3,4,5,6,7,8]; Es = 10; %for QAM

%16QAM constellation and mapping rule
% M = 16; maprule = [13,6,7,16,3,12,14,5,8,15,9,2,10,1,4,11]; Es = 10;
% M = 16; maprule = [11,2,5,16,1,12,15,6,13,8,3,10,7,14,9,4]; Es = 10;
M = 16; maprule = [11,2,1,12,4,9,10,3,5,16,15,6,14,7,8,13]; Es = 10;
%====================Set Partitioning==============================
%M = 16; maprule = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]; Es = 10;
%=======================LTE Gray=================================
%M = 16; maprule = [7,8,3,4,6,5,2,1,11,12,15,16,10,9,14,13]; Es = 10; 

%====================Strar 16QAM MSEW==============================
%M = 16; maprule = [9,4,3,14,7,10,13,8,5,16,15,2,11,6,1,12]; a = 1.8;  Es = (1 + a^2)/2; 
%S = (1 + floor((maprule-1)/8)*(a-1)).*exp(1i*(maprule-1)*pi/4)./sqrt(Es);
%S = (1 + floor((maprule-1)/8)*(a-1)).*exp(1i*(maprule-1)*pi/4)./sqrt(Es).*(1i*floor((maprule-1)/8)*pi/8);

if M==4
    temp = qammod((0:15), 16);
    qamPoints = temp(1,[14,2,3,15]);
elseif M==8
    temp = qammod((0:15), 16);
    %qamPoints = temp(1,[14,5,2,8,3,12,15,9]);
    qamPoints = temp(1,[14,2,12,9,3,15,5,8]);
    %qamPoints = pskmod((0:M-1), M);
else
    qamPoints = qammod((0:M-1), M);
end

bps = log2(M);
S = qamPoints(maprule)/sqrt(Es);   
symMatrix = de2bi((0:M-1),bps,'left-msb');

maximum_blockerror=50;             % maximum blockerrors per SNR point
maximum_block=ceil((8e+8)/cols);   % maximum block number per SNR point
        

%codelen=8, bps=2
%load BIBCM-ID_Algeb_4096_8_2.mat alpha

%codelen=8, bps=4
%load BIBCM_256.mat alpha
%load BIBCM_512_128_13.mat alpha
%load BIBCM_512_random.mat alpha
%load BIBCM_512_124_0.mat alpha
%load BIBCM_512_128_0.mat alpha


load BIBCM-ID_4096Algeb.mat alpha
%load BIBCM-ID_4096AlgebInline.mat alpha
%load BIBCM-ID_2048Algeb.mat alpha
%load BIBCM-ID_2048AlgebInline.mat alpha
%load BIBCM-ID_1024Algeb.mat alpha
%load BIBCM-ID_1024AlgebInline.mat alpha
%load BIBCM-ID_512Algeb.mat alpha

%codelen=8, bps=3
%load BIBCM-ID_Algeb_2304_8_3.mat alpha
%load BIBCM_3072Algeb_8_3.mat
%load BIBCM_1536Algeb_8_3.mat
%load BIBCM_768Algeb_8_3.mat

%codelen=16, bps=4
%load BIBCM-ID_Algeb_2048_16_4.mat alpha

numchanbit = length(alpha);
framelen = numchanbit/cols;
maxiter = 10; 

dB = [(0:3),(3.5:0.5:5)];                        % range of SNR values in dB
SNRpsig = 10.^(dB/10)*bps;           % Eb/No conversion from dB to decimal
No_uncoded = 1./SNRpsig;             % since Es=1
No = No_uncoded./((cols-rows)/cols);
BER = zeros(maxiter,length(dB));     % array for Channel Error Rate

for z = 1:length(SNRpsig)           % loop for testing over range of SNR values

    if dB(z) <= 2
        maximum_blockerror = 150;
    elseif dB(z) <= 3
        maximum_blockerror = 100;
    elseif dB(z) <= 4
        maximum_blockerror = 70;
    elseif dB(z) <= 4.5
        maximum_blockerror = 50;
    else
        maximum_blockerror = 30;
    end
    biterrors = zeros(1,maxiter);
    blockerrors=0;
    block=0;
    
    while(blockerrors<maximum_blockerror && block<maximum_block)  %while loop
        u = zeros(1,framelen*(cols-rows)); v = zeros(1,framelen*cols);
        for cnt = 1:framelen
           temp = round(rand(1,cols-rows));
           u((cnt-1)*(cols-rows)+1:cnt*(cols-rows)) = temp;
           v((cnt-1)*cols+1:cnt*cols) = rem(temp*G,2);
        end

        %[~, alpha] = sort(rand(1, numchanbit)); % for random interleavers
        vv = v(alpha);
        
        chan_in = modulate16qam(vv, S);
        noise = randn(size(chan_in)) + 1i*randn(size(chan_in));
        chan_out = chan_in + sqrt(No(z)/2)*noise;
        
        demodulated = softdemod_16qam(chan_out,No(z),S);
        La = zeros(1,numchanbit); Le = La(alpha);
        b_llr = zeros(1,numchanbit);
        SF = ones(1,numchanbit)*0.85;
        for iteration = 1:maxiter
            
            Le =(La - b_llr).*SF;
            Le = Le(alpha); 
            b_llr(alpha) = softdem(demodulated, Le, symMatrix);          
  
            for cnt = 1:framelen
                La((cnt-1)*cols+1:cnt*cols) = dualdec( b_llr((cnt-1)*cols+1:cnt*cols), H);
            end  
             
            vhat = ( ( sign(La) + 1 )/2 );
            Errors=zeros(1,length(vhat));
            Errors((v~=vhat)) = 1; 
            biterrors(iteration) = biterrors(iteration) + sum(Errors);
        end
        block = block + 1;        
        if sum(Errors)~=0
            blockerrors = blockerrors+1;
        end
        
        if rem(block,1000)==0
            pause(0.01)
        end
        
    end     %while loop
    
    for iteration = maxiter%-4:2:maxiter
       BER(iteration,z)=biterrors(iteration)/(block*numchanbit);       
       semilogy(dB(1:z),BER(iteration,1:z),'-md')
       drawnow
       hold on
    end
    fprintf(1,'\n\n Simulation finished for SNR: %d \n',dB(z))
end      % loop for testing over range of SNR values

toc

%   Copyright (c) 2020 by Dinh The Cuong


