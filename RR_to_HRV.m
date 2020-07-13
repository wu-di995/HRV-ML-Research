% Edited RR intervals to HRV metrics
% Creates HRV metrics for 5s windows, 1s increments

run("startup.m");

% Subjects
subjects = ["s01", "s03", "s05", "u03", "u04", "u09", "u13", "u14"];

% Events 
events = ["Headarray - Command", "Headarray - Low level autonomy", "Headarray - Mid level autonomy", ... 
          "Headarray - Teleoperation", "Headarray - Trajectory", "Headarray - WST", "Joystick - Command", ...
          "Joystick - Low level autonomy", "Joystick - Mid level autonomy", "Joystick - Teleoperation", ...
          "Joystick - Trajectory", "Joystick - WST", "Sip-n-puff - Command", "Sip-n-puff - Teleoperation" ...
          "Sip-n-puff - Trajectory", "Sip-n-puff - WST"];

% Base path 
base = "C:\Users\Wu Di\Documents\HRV-ML-Research\RR_edited\";
% C:\Users\Wu Di\Documents\HRV-ML-Research\RR_edited\s01-Headarray -Command_edit-RR.csv
% Paths
paths = [];
% InitializeHRVparams.m function
HRVparams = InitializeHRVparams('Test');
HRVparams.Fs = 250;
HRVparams.windowlength = 5; % seconds
HRVparams.increment = 1; % seconds
HRVparams.MSE.on = 0;
HRVparams.DFA.on = 0;
HRVparams.HRT.on = 0;
HRVparams.af.on = 0;
HRVparams.sqi.LowQualityThreshold = 0.01; % Default: 0.9, Threshold for which SQI represents good data
HRVparams.sqi.windowlength = 5;         % Default: 10, seconds, length of the comparison window
HRVparams.sqi.increment = 1;             % Default: 1, seconds
HRVparams.sqi.TimeThreshold = 0.1;       % Default: 0.1, seconds
HRVparams.sqi.margin = 1;                % Default: 2, seconds, Margin time not include in comparison 

for subject = subjects
    if subject == "s01"
        for event = events
            if contains(event,"Headarray") || contains(event,"Sip")
                path = strcat(base,subject,"-",event,"_edit-RR.csv");
                subjID = subject+"_"+event;
                csv = readtable(path);
                time = table2array(csv(:,2));
                RR = table2array(csv(:,3));
                [results, resFileName] = Main_HRV_Analysis(RR,time,"RRIntervals",HRVparams,subjID);
            end
        end
        
    elseif subject == "s03"
        for event = events
            if contains(event,"Joystick") || contains(event,"Sip")
                path = strcat(base,subject,"-",event,"_edit-RR.csv");
                subjID = subject+"_"+event;
                csv = readtable(path);
                time = table2array(csv(:,2));
                RR = table2array(csv(:,3));
                [results, resFileName] = Main_HRV_Analysis(RR,time,"RRIntervals",HRVparams,subjID);
            end 
        end
         
    else
        for event = events 
            path = strcat(base,subject,"-",event,"_edit-RR.csv");
            subjID = subject+"_"+event;
            csv = readtable(path);
            time = table2array(csv(:,2));
            RR = table2array(csv(:,3));
            [results, resFileName] = Main_HRV_Analysis(RR,time,"RRIntervals",HRVparams,subjID);
        end
    end
end 