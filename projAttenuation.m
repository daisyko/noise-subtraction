clc;
clear all;
% [y_c,Fe_c] = audioread('CleanSpeech/s04.wav');
% y_c = y_c(500:end);
[y,Fe] = audioread('ns2_04.wav');
y = y(500:end);
output = SSBoll79(y,Fe,0.8);


figure;
subplot(3,1,1);
plot(y);
title('Noisy Signal');
subplot(3,1,2);
plot(output);
title('De-Noised Signal');
subplot(3,1,3);
% plot(y_c);
% title('Clean Signal');

fprintf('\nPlay Original Sound:');
sound(y,Fe);
fprintf(' OK\n');
pause(3.5);
fprintf('Play new Sound: ');
sound(output,Fe);
fprintf('OK\n');
% fprintf('Write newsiound.wav:');
% audiowrite('NoisySpeech2/denoised_ns2_4.wav',output,Fe);
fprintf('OK\n');




function output=SSBoll79(signal,fs,IS)


if (nargin<3 | isstruct(IS))
    IS=.25; %seconds
end

W=fix(.025*fs); %Window length is 25 ms
nfft=W;
SP=.5; %Shift percentage is 40% (10ms) %Overlap-Add method works good with this value(.4)
wnd=hamming(W);



NIS=fix((IS*fs-W)/(SP*W) +1);%number of initial silence segments
Gamma=1;%Magnitude Power (1 for magnitude spectral subtraction 2 for power spectrum subtraction)

y=segment(signal,W,SP,wnd);
Y=fft(y,nfft);
YPhase=angle(Y(1:fix(end/2)+1,:)); %Noisy Speech Phase
Y=abs(Y(1:fix(end/2)+1,:)).^Gamma;%Specrogram
numberOfFrames=size(Y,2);
FreqResol=size(Y,1);

N=mean(Y(:,1:NIS)')'; %initial Noise Power Spectrum mean

NRM=zeros(size(Y));% Noise Residual Maximum (Initialization)
NoiseCounter=0;
NoiseLength=9;%This is a smoothing factor for the noise updating

Beta=.025;

YS=Y; %Y Magnitude Averaged
for i=2:(numberOfFrames-1)
    YS(:,i)=(Y(:,i-1)+Y(:,i)+Y(:,i+1))/3;
end

for i=1:numberOfFrames
    D(:,i)=YS(:,i)-N;
   % D(find(D<0))=0;
    
end
     
for i=1:NIS
    N=(NoiseLength*N+Y(:,i))/(NoiseLength+1);%Update and smooth noise
    NRM=max(NRM,abs(YS(:,i)-D(:,i)));
end

for i=1:numberOfFrames
     for j=2:length(D)-1
          if D(j) < NRM(j)
               D(j)=min([D(j) D(j-1) D(j+1)]);
          end
     end
end




for i=1:numberOfFrames
    temp = 0;
    for j=1:FreqResol
        temp = temp + (D(j)/N(j));       
    end
    T(i) = 20*log10(temp)/FreqResol;
    if T(i)<-12
        D(i) = 0.03.*D(i);
    end
    X(:,i)=max(D(:,i),0);
end

    

output=OverlapAdd2(X.^(1/Gamma),YPhase,W,SP*W);
end


function [NoiseFlag, SpeechFlag, NoiseCounter, Dist]=vad(signal,noise,NoiseCounter,NoiseMargin,Hangover)


if nargin<4
    NoiseMargin=3;
end
if nargin<5
    Hangover=8;
end
if nargin<3
    NoiseCounter=0;
end
    
FreqResol=length(signal);

SpectralDist= 20*(log10(signal)-log10(noise));
SpectralDist(find(SpectralDist<0))=0;

Dist=mean(SpectralDist); 
if (Dist < NoiseMargin) 
%  if (Dist < -12)     
    NoiseFlag=1; 
    NoiseCounter=NoiseCounter+1;
else
    NoiseFlag=0;
    NoiseCounter=0;
end

% Detect noise only periods and attenuate the signal     
if (NoiseCounter > Hangover) 
    SpeechFlag=0;    
else 
    SpeechFlag=1; 
end 
end


function Seg=segment(signal,W,SP,Window)

Window=Window(:); %make it a column vector

L=length(signal);
SP=fix(W.*SP);
N=fix((L-W)/SP +1); %number of segments

Index=(repmat(1:W,N,1)+repmat((0:(N-1))'*SP,1,W))';
hw=repmat(Window,1,N);
Seg=signal(Index).*hw;
end


function ReconstructedSignal=OverlapAdd2(XNEW,yphase,windowLen,ShiftLen);


if nargin<2 %Number of function input arguments
    yphase=angle(XNEW);
end
if nargin<3
    windowLen=size(XNEW,1)*2;
end
if nargin<4
    ShiftLen=windowLen/2;
end
if fix(ShiftLen)~=ShiftLen
    ShiftLen=fix(ShiftLen);
    disp('The shift length have to be an integer as it is the number of samples.')
    disp(['shift length is fixed to ' num2str(ShiftLen)])
end

[FreqRes FrameNum]=size(XNEW);

Spec=XNEW.*exp(j*yphase);

if mod(windowLen,2) %if FreqResol is odd
    Spec=[Spec;flipud(conj(Spec(2:end,:)))];
else
    Spec=[Spec;flipud(conj(Spec(2:end-1,:)))];
end
sig=zeros((FrameNum-1)*ShiftLen+windowLen,1);
weight=sig;
for i=1:FrameNum
    start=(i-1)*ShiftLen+1;
    spec=Spec(:,i);
    sig(start:start+windowLen-1)=sig(start:start+windowLen-1)+real(ifft(spec,windowLen));
end
ReconstructedSignal=sig;
end


