% Convert ECG to Beats
% beats.csv contains times (s) of beats 
% run("startup.m");

% Subjects
subjects = ["s01", "s03", "s05", "u03", "u04", "u09", "u13", "u14"];

% Events 
events = ["Headarray - Command", "Headarray - Low level autonomy", "Headarray - Mid level autonomy", ... 
          "Headarray - Teleoperation", "Headarray - Trajectory", "Headarray - WST", "Joystick - Command", ...
          "Joystick - Low level autonomy", "Joystick - Mid level autonomy", "Joystick - Teleoperation", ...
          "Joystick - Trajectory", "Joystick - WST", "Sip-n-puff - Command", "Sip-n-puff - Teleoperation" ...
          "Sip-n-puff - Trajectory", "Sip-n-puff - WST"];

% Base path 
base = "C:\Users\Wu Di\Documents\HRV-ML-Research\RawECG\";
% Paths
paths = [];
% InitializeHRVparams.m function
Fs = 250;
HRVparams = InitializeHRVparams('Beats');
HRVparams.Fs = Fs;


for subject = subjects
    if subject == "s01"
        for event = events
            if contains(event,"Headarray") || contains(event,"Sip")
                path =  strcat(base,subject,filesep,event,".csv");
                subjID = subject+"-"+event;
                ecg = load(path);
                get_beats(ecg,HRVparams,subjID);
            end
        end
        
    elseif subject == "s03"
        for event = events
            if contains(event,"Joystick") || contains(event,"Sip")
                path = strcat(base,subject,filesep,event,".csv");
                subjID = subject+"-"+event;
                ecg = load(path);
                get_beats(ecg,HRVparams,subjID);
            end 
        end
        
    elseif subject == "u03"
        for event = events 
            if contains(event,"Joystick - Trajectory")
                path = strcat(base,subject,filesep,event,"1.csv");
            else 
                path = strcat(base,subject,filesep,event,".csv");
            end 
            subjID = subject+"-"+event;
            ecg = load(path);
            get_beats(ecg,HRVparams,subjID);
        end 
         
    else
        for event = events 
            path = strcat(base,subject,filesep,event,".csv");
            subjID = subject+"-"+event;
            ecg = load(path);
            get_beats(ecg,HRVparams,subjID);
        end
    end
end 

function get_beats(ecg,HRVparams,subjID)
Fs = 250;
time = 0:1/Fs:(length(ecg)-1)/Fs;
QRS = run_qrsdet_by_seg(ecg,HRVparams);
beats = time(QRS);
savedir = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\Beats";
filename = strcat(savedir,filesep,subjID,"_beats.csv");
writematrix(beats, filename);
end