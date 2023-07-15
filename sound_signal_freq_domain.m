%% Phys 420 - A2 coding part - completed by Dilyar Arkin
%% Q5 Part A
clear,clc;

dt=0.1;     % given time increament
ts = dt;    % interpreted as sampling period
fs = 1/ts;  % sampling frequency is 1/period
tmax = 50;  % total time duration
f1=1.20;    % frequency 1 given
f2=1.22;    % frequenct 2 given
N=(tmax/ts)+1;      % total time duration divided by sampling period plus 
                    % 1 (b/c we want array length to be consistant)
t = (0:ts:tmax);    % now finally, specify the time vector
y =  sin(2*pi*f1*t)+0.7*sin(2*pi*f2*t); % original function, given
y_pad = [zeros(1,200),y,zeros(1,200)];  % zeros padded function for better
                                        % resolution.
% plot time vs original function - seems to have beat frequencies.
plot(t,y); 
grid on
xlabel('time (0.1s / grid)')
ylabel('amplitude (units)')
title('temporal domain plot of original function Q5.a')

Y = fftshift(fft(y)); % fft transfor and shift zero frequencies to centre.
% frequency goes from -f/2 to +f/2 with fs/(N-1) increment.
f = (-fs/2:fs/(N-1):fs/2); 
% plot the power of Y vs f. note that Y.*Y assumed as power
figure,plot(f,(real(Y).*real(Y)+imag(Y).*imag(Y))); 
xlim([1,1.5]) % set range for plotted graph
grid on
xlabel('Frequency (Hz)')
ylabel('Power Spectrum (power units)')
title('Unresolved Frequency Resolution Q5.a')

Npad = length(y_pad); % re-calculate sample size by taking length of y_pad
fpad = (-fs/2:fs/(Npad-1):fs/2); % recalculate the frequency axis
Ypad = fftshift(fft(y_pad)); % FFT and shift the zero padded function 
% plot the power spectrum vs f
figure,plot(fpad,real(Ypad).*real(Ypad)+imag(Ypad).*imag(Ypad)); 
xlim([1,1.5]); % set bounds for frequency axis, only show relavant portion
grid on
xlabel('Frequency (Hz)')
ylabel('Power Spectrum (power units)')
title('Resolved Frequency Resolution Q5.a')

%figure,pwelch(y_pad,[],[],[],fs);
% ------------------------------------------------------------------------
% Answers: 
%   - adding lots of zeros does not introduce anything new, but  only 
%     increase the resolution of fft'd signal up to a certain point. 
%     i found a few hundred zeros padded to the original data is sufficient
%     enough When added a lot of zeros, the amplitude went slightly higher 
%     but frequency also shifted away slightly. so there is trade offs.
%      
%   - the ratio of intensities in the power spectrum supposed to be
%     proportional to the amplitude ratio of the original signals, 
%     which were 1:.7 but this ratio is off by 14%. Which is odd
%     because we should be able to capture relative ratio between the two
%     peaks exactly.
%     --------------------------------------------------------------------
%%  Q5 Part B.
clear,clc;

dt=0.1;      % given time increament in the question
ts = dt;     % interpreted as sampling period
fs = 1/ts ;  % sampling frequency is 1/period
tmax = 100;  % total time duration
f1=0.10;     % frequency 1 given
f2=0.35;     % frequenct 2 given
N = (tmax/ts);      % total time duration divided by sampling period plus 
                    % 1 (b/c we want array length to be consistant)
t = (0:ts:tmax);    % now finally, the time vector
y =  sin(2*pi*f1*t)+0.7*sin(2*pi*f2*t); % original function, given
w = hamming(N+1);   % create hamming filter
u = y .* w'; % multiplication in time domain is convolution in freq domain

figure, plot(t,u,'b'); % plot input data pre-windowing
grid on;
xlabel('time (0.1s/grid)')
ylabel('amplitude')
title('input data post-windowing Q5.b')
figure,plot(t,y,'r');  % plot input data post-windowing
grid on
xlabel('time (0.1s/grid)')
ylabel('amplitude')
title('input data pre-windowing Q5.b')
% fourier transform pre-windowed and post-windowed signals by fft function
% without zero padding
Y = fftshift(fft(y));
U = fftshift(fft(u));
f = (-fs/2:fs/(N):fs/2); % specify frequency axis
figure,plot(f,(real(U).*real(U)+imag(U).*imag(U)),'b'); % plot power spect
hold on
plot(f,(real(Y).*real(Y)+imag(Y).*imag(Y)),'r');
xlim([0,0.5])
grid on
xlabel('Frequency (Hz)')
ylabel('Power (units)')
title('frequency spectrum of pre/post windowed data without zero pad Q5.b')
hold off

% same pre/post windowed signals with zero padding
ypad = [zeros(1,10000),y,zeros(1,10000)]; % original function, given
Npad = length(ypad);
wnew = hamming(Npad);
upad = ypad .* wnew';
Ypad = fftshift(fft(ypad));
Upad = fftshift(fft(upad));
fpad = (-fs/2:fs/(Npad-1):fs/2);
figure,plot(fpad,(real(Upad).*real(Upad)+imag(Upad).*imag(Upad)),'b');
hold on
plot(fpad,(real(Ypad).*real(Ypad)+imag(Ypad).*imag(Ypad)),'r');
xlim([0,0.5]);
grid on
xlabel('Frequency (Hz)')
ylabel('Power (units)')
title('frequency spectrum of pre/post windowed data with zero pad Q5.b')
hold off;
% Notes on the side lobes: for the case when no zero padding, the hamming
% windowed data shows wider side lobes than the original data. But when the
% zero-padded windowed signal transformed into fourier domain, the side 
% lobes are smaller than the zero-padded un-windowed signal. Zero pads
% only increases the resolution in the frequency domain up to certain
% extent. It is not the reason the side lobes are getting smaller, but the
% hamming window itself that minimize the leakage from one bin to the
% another.

%% Q6
clear,clc;
% load the audio files and store the values
[audioIn1,fs1] = audioread("m1.m4a"); % A2_i.wav
[audioIn2,fs2] = audioread("m2.m4a"); % A2_ii.wav 
% sampling period register
ts1 = 1/fs1;
ts2 = 1/fs2;
% total sample
N1 = length(audioIn1);
N2 = length(audioIn2);
% total elapced time of sampling
tmax1 = ts1*(N1-1);
tmax2 = ts2*(N2-1);
% time vectors for two samples
t1 = (0:ts1:tmax1);
t2 = (0:ts2:tmax2);
% plot the temporal content on top of each other
plot(t1,audioIn1,'r');
hold on
plot(t2,audioIn2,'g');
hold off

sound(audioIn1,fs1);
pause(tmax1*1.1);
sound(audioIn2,fs2);

% DFT by fast fourier transform built in function and shift the low
% frequencies to the centre
Y1 = fftshift(fft(audioIn1));
Y1 = Y1';
Y2 = fftshift(fft(audioIn2));
Y2 = Y2';
% specify the x axis of the sample we're plotting against.
f1 = (-fs1/2: fs1/(N1-1) :fs1/2);
f2 = (-fs2/2: fs2/(N2-1) :fs2/2);
%plotting the file 1 in frequency domain
figure,plot(f1,abs(Y1));
xlim([0,1000]);
title 'm1 file recorded sound'
%plotting the file 2 in frequency domain
figure,plot(f2,abs(Y2));
xlim([0,1000]);
title 'm2 file recorded sound'

% file number 1 (A2_i.wav) is 446 hz and it's out of tune by 6hz ± 1 hz.
% as shown on the plotted graphs, file #2,(A2_ii.wav) is 440hz ±1hz.
%-------------------------------------------------------------------------
% according to wikipedia,  "electronic tuner detects and displays the pitch
% of musical notes played on the musical instrument". sound is a wave of 
% a pressure gradient of air. It oscillates the most in certain frequencies
% known as the fundemental frequency. In fourier domain, it is the highest
% impluse. Electronic tuner parhaps first measuring sound as signal and
% transforms into the fourier domain, and reads out the highest peaked
% frequency. If this frequency is off from the natural frequency
% corresponds to that particular string, one can use this read out device
% to re-tune the strings back to it's harmonic frequency.
%% End of code