% Export odom timestamp, linear and angular velocities to csv files 

% Loop through all odom files 
base = "E:\argall-lab-data\Trajectory Data\";

% Subjects
subjects = ["S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", ...
    "U00", "U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08" ,...
    "U09", "U11", "U12", "U13", "U14"];

for subject = subjects
    trajFolders = dir(strcat(base,subject));
    issub = [trajFolders(:).isdir];
    trajFolderNames = {trajFolders(issub).name};
    for trajFolder = trajFolderNames
        if contains(trajFolder, "A0")
            odomFile = dir(strcat(base,subject,filesep,trajFolder{1},filesep,"*odom.mat"));
            odomFilePath = strcat(odomFile.folder,filesep,odomFile.name);
            odomCSVName = erase(odomFile.name,".mat");
            odomCSVPath = strcat(odomFile.folder,filesep,odomCSVName,"_vel.csv");
%             odomPosCSVPath = strcat(odomFile.folder,filesep,odomCSVName,"_pos.csv");
            export_velocity(odomFilePath,odomCSVPath);
%             disp(odomPosCSVPath);
%             export_pos(odomFilePath,odomPosCSVPath);
            
        end
    end

end


function export_velocity(odomPath,csvPath)
    file = load(odomPath);
    odom = file.odom;
    % Create matrix 
    export_mat = zeros(length(odom),3);
    % Loop through odom 
    for i = 1:length(odom)
        time = odom{i}.header.stamp.time;
        time = int64(time*1000);
        linear_vel = odom{i}.twist.twist.linear(1);
        ang_vel = odom{i}.twist.twist.angular(3);
        export_mat(i,1) = time;
        export_mat(i,2) = linear_vel;
        export_mat(i,3) = ang_vel;
    end
    writematrix(export_mat,csvPath);
end

function export_pos(odomPath,csvPath)
    file = load(odomPath);
    odom = file.odom;
    % Create matrix, columns = time, position x (1), position y (2),
    % orientation (3), orientation (4)
    export_mat = zeros(length(odom),5);
    % Loop through odom 
    for i = 1:length(odom)
        time = odom{i}.header.stamp.time;
        posx = odom{i}.pose.pose.position(1);
        posy = odom{i}.pose.pose.position(2);
        orient1 = odom{i}.pose.pose.orientation(3);
        orient2 = odom{i}.pose.pose.orientation(4);
        export_mat(i,1) = int64(time*1000);
        export_mat(i,2) = posx;
        export_mat(i,3) = posy;
        export_mat(i,4) = orient1;
        export_mat(i,5) = orient2;
    end
    writematrix(export_mat,csvPath);
end

