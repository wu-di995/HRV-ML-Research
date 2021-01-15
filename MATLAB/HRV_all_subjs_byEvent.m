% Generate HRV metrics for all subjects, 30s and 60s windows only

run("startup.m");

% Subjects
subjects = ["s00", "s01", "s02", "s03", "s04", "s05", "s06", "s07", "s08", ...
    "u01", "u02", "u03", "u04", "u05", "u06", "u07", "u08" ,...
    "u09", "u10", "u11", "u12", "u13", "u14"];

% Events 
events = ["Headarray - Teleoperation","Headarray - Low level autonomy", "Headarray - Mid level autonomy", ... 
           "Joystick - Teleoperation", "Joystick - Low level autonomy", "Joystick - Mid level autonomy", ...
          "Sip-n-puff - Teleoperation"];
      
% Base path - Change to match local 
% base = "E:\argall-lab-data\ECG_byEvent\";
base = "/home/skrdown/Documents/argall-lab-data/ECG_byEvent_forHRV/";

% InitializeHRVparams.m function
HRVparams = InitializeHRVparams('Test');
HRVparams.Fs = 250;
HRVparams.windowlength = 60; % seconds -- change to 30 or 60 
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
    for event = events 
        path = strcat(base,subject,filesep,event,".csv");
        % Interface
        if contains(event, "Headarray")
            interface = "HA";
        elseif contains(event, "Joystick")
            interface = "JOY";
        elseif contains(event,"Sip-n-puff")
            interface = "SNP";
        end
        % Autonomy 
        if contains(event, "Teleoperation")
            autonomy = "A0";
        elseif contains(event, "Low level autonomy")
            autonomy = "A1";
        elseif contains(event,"Mid level autonomy")
            autonomy = "A2";
        end
        if isfile(path)
            ecg = load(path);
%             disp(path);
            subjID = subject+"_"+interface+"_"+autonomy;
%             disp(subjID);
            [results, resFileName] = Main_HRV_Analysis(ecg,[],"ECGWaveform",HRVparams,subjID);
        end
        
        
        
    end
end
    
      
      
      
      