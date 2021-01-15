% Export angular and linear commands as csvs, excluding times where
% supervisor controls
% Loop through all odom files 
base = "E:\argall-lab-data\Trajectory Data\";

% Subjects
subjects = ["S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", ...
    "U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08" ,...
    "U09", "U11", "U12", "U13", "U14"];

for subject = subjects
    trajFolders = dir(strcat(base,subject));
    issub = [trajFolders(:).isdir];
    trajFolderNames = {trajFolders(issub).name};
    for trajFolder = trajFolderNames
        if contains(trajFolder, "A0")
            userCmdFile = dir(strcat(base,subject,filesep,trajFolder{1},filesep,"*user_cmd.mat"));
            userCmdFilePath = strcat(userCmdFile.folder,filesep,userCmdFile.name);
            userCmdCSVName = erase(userCmdFile.name,".mat");
            userCmdCSVPath = strcat(userCmdFile.folder,filesep,userCmdCSVName,".csv");
            disp(userCmdCSVPath);
            export_cmd(userCmdFilePath,userCmdCSVPath);
            
        end
    end

end


function export_cmd(userCmdPath,userCmdCSVPath)
    file = load(userCmdPath);
    userCmd = file.user_cmd;
    % Create matrix 
    export_mat = zeros(length(userCmd),4);
    % Columns - Time, Supervisor Status, Linear (1) Angular(3)
    % Loop through odom 
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
        
        lin = userCmd{i}.command.linear(1);
        ang = userCmd{i}.command.angular(3);
        export_mat(i,1) = time;
        export_mat(i,2) = supStatus;
        export_mat(i,3) = lin;
        export_mat(i,4) = ang;

    end
    writematrix(export_mat,userCmdCSVPath);
end

