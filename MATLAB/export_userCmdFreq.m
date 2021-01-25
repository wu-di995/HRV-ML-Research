% Export angular and linear commands as csvs, excluding times where
% supervisor controls

% Loop through all userCmd files 
% base = "E:\argall-lab-data\Trajectory Data\";
base = "/home/skrdown/Documents/argall-lab-data/Trajectory Data/";


% Subjects
subjects = ["S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", ...
    "U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08" ,...
    "U09", "U11", "U12", "U13", "U14"];

for subject = subjects
    trajFolders = dir(strcat(base,subject));
    issub = [trajFolders(:).isdir];
    trajFolderNames = {trajFolders(issub).name};
    for trajFolder = trajFolderNames
        userCmdFile = dir(strcat(base,subject,filesep,trajFolder{1},filesep,"*user_cmd.mat"));
        userCmdFilePath = strcat(userCmdFile.folder,filesep,userCmdFile.name);
        if contains(trajFolder, "A0")
            if contains(trajFolder, "HA")
                joyCmdFile = dir(strcat(base,subject,filesep,trajFolder{1},filesep,"*joy_cont_asl.mat"));
            elseif contains(trajFolder,"SNP")
                joyCmdFile = dir(strcat(base,subject,filesep,trajFolder{1},filesep,"*joy_snp.mat"));
            elseif contains(trajFolder,"JOY")
                joyCmdFile = dir(strcat(base,subject,filesep,trajFolder{1},filesep,"*joy_cont.mat"));
            end
        joyCmdFilePath = strcat(joyCmdFile.folder,filesep,joyCmdFile.name);
        joyCmdCSVName = erase(joyCmdFile.name,".mat");
        freqs30CSVPath = strcat(joyCmdFile.folder,filesep,joyCmdCSVName,"_30_freqs.csv");
        freqs60CSVPath = strcat(joyCmdFile.folder,filesep,joyCmdCSVName,"_60_freqs.csv");
        impulsesPath = strcat(joyCmdFile.folder,filesep,joyCmdCSVName,"_userImpulses.csv");
        axesCSVPath = strcat(joyCmdFile.folder,filesep,joyCmdCSVName,"_axes.csv");
        supStatusCSVPath = strcat(userCmdFile.folder,filesep,joyCmdCSVName,"_supStatus.csv");
        disp(axesCSVPath);
        export_axes(joyCmdFilePath, axesCSVPath);
        export_supStatus(userCmdFilePath, supStatusCSVPath)
%         export_cmdFreq(userCmdFilePath,impulsesPath,freqs30CSVPath,freqs60CSVPath);
            
        end
    end

end

function export_supStatus(userCmdFilePath, supStatusCSVPath)
    file = load(userCmdFilePath);
    userCmd = file.user_cmd;
    % Create matrix 
    supStatus_mat = zeros(length(userCmd),2);
    for i = 1:length(userCmd)
        time = userCmd{i}.meta_data.time;
        % Convert time to match with ECG and Annotations time format
        time = int64(time*1000);
        supStatus = userCmd{i}.goal_source;
        if strcmp(supStatus,'teleoperation_superviser')
            supStatus = 0;
        else 
            supStatus = 1;
        end
        supStatus_mat(i,1) = time;
        supStatus_mat(i,2) = supStatus;
    end
    writematrix(supStatus_mat,supStatusCSVPath);

end

function export_axes(joyCmdFilePath, axesCSVPath)
    file = load(joyCmdFilePath);
    if contains(joyCmdFilePath,"HA")
        disp("HA");
        joyCmd = file.joy_cont_asl;
    elseif contains(joyCmdFilePath,"SNP")
        disp("SNP");
        joyCmd = file.joy_snp;
    elseif contains(joyCmdFilePath,"JOY")
        disp("JOY");
        joyCmd = file.joy_cont;
    end
    disp(length(joyCmd));
    axes = zeros(length(joyCmd),3);
    for i = 1:length(joyCmd)
        axes(i,1) = int64(joyCmd{i}.meta_data.time*1000);
        axes(i,2) = joyCmd{i}.axes(1);
        axes(i,3) = joyCmd{i}.axes(2);   
    end
    writematrix(axes,axesCSVPath);
    
end

function export_cmdFreq(joyCmdPath,impulsesPath,freqs30CSVPath,freqs60CSVPath)
    file = load(joyCmdPath);
    if contains(joyCmdPath,"HA")
        disp("HA");
        userCmd = file.joy_cont_asl;
    elseif contains(joyCmdPath,"SNP")
        disp("SNP");
        userCmd = file.joy_snp;
    elseif contains(joyCmdPath,"JOY")
        disp("JOY");
        userCmd = file.joy_cont;
    end
    
    % Sampling frequency
%     samp_freq = 25;
    
    % Create vector of 1s and 0s, where 1 indicates presence of userCmd 
    userCmd_impulses = zeros(length(userCmd),2);
    for i = 1:length(userCmd)
        userCmd_impulses(i,1) = int64(userCmd{i}.meta_data.time*1000);
        if ~ ((userCmd{i}.axes(1)==0) && (userCmd{i}.axes(2)==0))
            userCmd_impulses(i,2) = 1;
        end
    end

    % Moving average window frequencies 
    % Columns  - Start Time, Frequency 
    % Create vectors to save frequencies 
%     freqs_30 = zeros(length(1:samp_freq:(length(userCmd)-(samp_freq*30))),2);
%     freqs_60 = zeros(length(1:samp_freq:(length(userCmd)-(samp_freq*60))),2);
%     freqs_30_idx = 1;
%     freqs_60_idx = 1;
%     disp(length(1:samp_freq:(length(userCmd)-(samp_freq*30))));
    % Get 30s window frequencies
%     for i = 1:samp_freq:(length(userCmd)-(samp_freq*30))
%         % If axes == [0,0], increment userCounts
%         userCounts_30 = 0;
%         idx = i;
%         for j = idx:idx+30*samp_freq
%             if ~ ((userCmd{j}.axes(1)==0) && (userCmd{j}.axes(2)==0))
%                 userCounts_30 = userCounts_30+1;
%             end
%         end
%         freqs_30(freqs_30_idx,1) = freqs_30_idx;
%         freqs_30(freqs_30_idx,2) = userCounts_30/30;
%         freqs_30_idx = freqs_30_idx + 1;
%     end
%     % Get 60s window frequencies 
%     for i = 1:samp_freq:(length(userCmd)-(samp_freq*60))
%         % If axes == [0,0], increment userCounts
%         userCounts_60 = 0;
%         idx = i;
%         for j = idx:idx+60*samp_freq
%             if ~((userCmd{j}.axes(1)==0) && (userCmd{j}.axes(2)==0))
%                 userCounts_60 = userCounts_60+1;
%             end
%         end
%         freqs_60(freqs_60_idx,1) = freqs_60_idx;
%         freqs_60(freqs_60_idx,2) = userCounts_60/60;
%         freqs_60_idx = freqs_60_idx + 1;
%     end
    
    
    % Export matrix 
    writematrix(userCmd_impulses,impulsesPath);
%     writematrix(freqs_30,freqs30CSVPath);
%     writematrix(freqs_60,freqs60CSVPath);
end

