% Generate HRV metrics for all subjects, on overlapping windows, 30s HRV
% windows

run("startup.m"); 

% Base path 
base = "C:\Users\Wu Di\Documents\HRV-ML-Research\ECG_overlappingWindowsbyTask\*.csv";
csvfiles = dir(base);

% InitializeHRVparams.m function
HRVparams = InitializeHRVparams('Test');
HRVparams.Fs = 250;
HRVparams.windowlength = 60; % seconds
HRVparams.increment = 1; % seconds
HRVparams.MSE.on = 0;
HRVparams.DFA.on = 0;
HRVparams.HRT.on = 0;
HRVparams.af.on = 0;
HRVparams.sqi.LowQualityThreshold = 0.9; % Default: 0.9, Threshold for which SQI represents good data
HRVparams.sqi.windowlength = 10;         % Default: 10, seconds, length of the comparison window
HRVparams.sqi.increment = 1;             % Default: 1, seconds
HRVparams.sqi.TimeThreshold = 0.1;       % Default: 0.1, seconds
HRVparams.sqi.margin = 1;                % Default: 2, seconds, Margin time not include in comparison 


path = "C:\Users\Wu Di\Documents\HRV-ML-Research\RawECG\s00\Joystick - Teleoperation.csv";
% path = "C:\Users\Wu Di\Documents\HRV-ML-Research\ECG_overlappingWindowsbyTask\s00_HA_A0_1.csv";
ecg = load(path);
[results, resFileName] = Main_HRV_Analysis(ecg,[],"ECGWaveform",HRVparams,"111");

% for i = 1 : length(csvfiles)
%     path = strcat(csvfiles(i).folder,filesep,csvfiles(i).name);
%     subjID = erase(csvfiles(i).name,".csv");
%     disp(subjID);
%     ecg = load(path);
%     [results, resFileName] = Main_HRV_Analysis(ecg,[],"ECGWaveform",HRVparams,subjID);
% end