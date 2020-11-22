
close all;
clear all;
%% Fetching the input and the noise
[in_audio, sampl_rate] = audioread('inp_10sec.wav');
norm_audio = in_audio/rms(in_audio);

subplot(2,2,1)
plot(norm_audio)
title("Normalized Input Audio")
[m,n]=size(norm_audio);
result = fopen('report3.txt', 'a+');
fprintf(result, '\nBefore, SNR After,Tap,LR,Exec_secs\n');
%% Extracting 10 sec sample from the noise signal
[in_noise, sr_noise] = audioread('bomb_noise.wav');
[r,c]=size(in_noise);
dt_noise=1/sr_noise;
t_noise = dt_noise*(0:r-1);
noise_ampl = 30;
in_noise_10=  noise_ampl*in_noise(1:size(in_audio,1));
in_noise_ref = in_noise(size(in_audio,1)+1:end);
% in_noise_ref = in_noise_10;
subplot(2,2,2)
plot(in_noise_10)
title("Noise Signal")
echo on
%% Adding noise to the original signal
org_noise = norm_audio' + in_noise_10;
audiowrite('Noisy_Signal.wav',org_noise,sr_noise)
subplot(2,2,3)
plot(norm_audio, 'g')
hold on
plot(in_noise_10, 'r')
title("Voice + Noise Signal")
% 
disp("Before Denoising SNR")
SNR = 10*log(rms(norm_audio)/rms(org_noise));

%% Analyzing the effect of filter_size (M) on the noisy signal
% for i = 0.1:0.005:1
%     M = 25000;
%     disp(M)
%     mu = i;
%     tic
%     filtered_out = LMS_Algo_audio(org_noise, in_noise_ref, M, mu);
%     time_t = toc;
% %     disp("After Denoising SNR")
%     SNR_out = 10*log(rms(org_noise)/rms(filtered_out));
%     fprintf(result, '\n%f,%f,%d,%f,%f\n', SNR, SNR_out, M, mu, time_t);
% 
% end
% fclose(result);
% % % plot(out, sampl_rate);
