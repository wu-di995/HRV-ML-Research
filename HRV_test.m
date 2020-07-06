% Testing Main_HRV_Analysis on a segment of data 
% Test data - Joystick teleop, s00

run("startup.m");

% % Input path: s04 pre-sess1 resting
% sigpath = strcat(pwd,filesep,"RawECG\s04\Pre-session 1 resting.csv");

% Input path: s04 sip and puff WST 
% sigpath = strcat(pwd,filesep,"RawECG\s04\Sip-n-puff - WST.csv");

% Input path: s04 headarray mid-level autonomy 
sigpath = strcat(pwd,filesep,"RawECG\s04\Headarray - Mid level autonomy.csv");

% Input path: s04 sip and puff teleop
sigpath = strcat(pwd,filesep,"RawECG\s04\Sip-n-puff - Teleoperation.csv");

% Input path: s04 head-array teleop
sigpath = strcat(pwd,filesep,"RawECG\s04\Headarray - Teleoperation.csv");

% Input path: s05 head-array command
sigpath = strcat(pwd,filesep,"RawECG\s05\Headarray - Command.csv");

% Load ecg signal 
ecg = load(sigpath);
% Sampling frequency 
Fs = 250;
% Time vector for visualization 
tm = 0:1/Fs:(length(ecg)-1)/Fs;
% Plot raw ecg signal
figure(1);
plot(tm,ecg);
title("Raw ECG");
xlabel("Time (s)");
ylabel("Voltage (mV)");

% Detection of the R-peaks using the jqrs.m function included in the
% toolbox, requires to set initialization parameters calling the
% InitializeHRVparams.m function
HRVparams = InitializeHRVparams('Test');
HRVparams.Fs = Fs;
HRVparams.windowlength = 60; % seconds
HRVparams.increment = 5; % seconds
HRVparams.MSE.on = 0;
HRVparams.DFA.on = 0;
HRVparams.HRT.on = 0;
HRVparams.af.on = 0;

% call the function that perform peak detection
% r_peaks = jqrs(ecg,HRVparams);

% plot the detected r_peaks on the top of the ecg signal
% figure(1)
% hold on;
% plot(r_peaks./Fs, ecg(r_peaks),'o');
% legend('ecg signal', 'detected R peaks')


%Convert ECG waveform to RR intervals 
% subID = "s00";
% [t, rr, jqrs_ann, SQIvalue , tSQI] = ConvertRawDataToRRIntervals(ecg, HRVparams, subID);
% figure(1);
% plot(t,rr);
% sqi = [tSQI', SQIvalue'];

%Preprocess data
% ann = [];
% [NN, tNN, tWin, AFWindows,out] = PreparDataForHRVAnlysis(rr,t,ann,sqi,HRVparams,subID);
% plot(tNN,NN);

%HRV Analysis 
% subID = "s04-pre_sess1_rest";
% subID = "s04-snp_wst";
% subID = "s04-ha_midauto";
% subID = "s04-snp_tele";
% subID = "s04-ha_tele";
subID = "s05-ha_com";
[results, resFileName] = Main_HRV_Analysis(ecg,[],"ECGWaveform",HRVparams,subID);
