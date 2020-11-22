%%Time specifications:
Fs = 8000;                   % samples per second
dt = 1/Fs;                   % seconds per sample
StopTime = 0.25;             % seconds
t = (0:dt:StopTime-dt)';     % seconds
%%Sine wave:
Fc = 10;                     % hertz
x = cos(2*pi*Fc*t);
subplot(1,2,2)
% Plot the signal versus time:

subplot(1,3,1)
plot(t,x);
xlabel('time (in seconds)');
title('Signal versus Time');
zoom xon;

f_n = 200;
Fs2 = 8000;                   % samples per second
dt2 = 1/Fs2;                   % seconds per sample
StopTime2 = 0.25;             % seconds
t2 = (0:dt2:StopTime2-dt2)';

subplot(1,3,2)
y = 0.2*cos(2*pi*f_n*t2);
ref_noise = 0.1*cos(2*pi*f_n*t2);

plot(t2,y);
xlabel('time (in seconds)');
title('Signal versus Time');
zoom xon;
noise = y;

noisy = noise + x;

subplot(1,3,3)
plot(t, noisy);
xlabel('time (in seconds)');
title('Noisy signal');
zoom xon;

filtered_out = LMS_Algo(noisy, ref_noise);
plot(t, filtered_out);
xlabel('time (in seconds)');
title('Filtered signal');
zoom xon;
