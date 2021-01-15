% Generate HRV metrics for all subjects, 30s and 60s windows only

% run("startup.m");

% Subjects
subjects = ["s00", "s01", "s02", "s03", "s04", "s05", "s06", "s07", "s08", ...
    "u00", "u01", "u02", "u03", "u04", "u05", "u06", "u07", "u08" ,...
    "u09", "u11", "u12", "u13", "u14"];

% Interfaces
interfaces = ["HA", "JOY", "SNP"];

% Autonomy 
% autonomies = ["A0", "A1", "A2"];
autonomies = ["A0"];

% Task numbers 
tasks = ["1", "2", "3", "4", "5", "6", "7"];

% Base path -- Change to match local
% base = "C:\Users\Wu Di\Documents\HRV-ML-Research\RawECG_byTasks\";
base = "/home/skrdown/HRV-ML-Research/ECG/RawECG_byTasks/";

% InitializeHRVparams.m function
HRVparams = InitializeHRVparams('Test');
HRVparams.Fs = 250;
HRVparams.windowlength = 30; % seconds
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

for subject = subjects
    for interface = interfaces
        for autonomy = autonomies 
            for task = tasks 
                event = strcat(subject, "_", interface, "_", autonomy, "_", task);
                disp(event);
                path = strcat(base,subject,filesep,event,".csv");
                subjID = event;
                if isfile(path)
                    ecg = load(path);
                end
                [results, resFileName] = Main_HRV_Analysis(ecg,[],"ECGWaveform",HRVparams,subjID);
            end
        end 
    end
end
