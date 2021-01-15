% Export distance to obstacle 
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
            footprintFile = dir(strcat(base,subject,filesep,trajFolder{1},filesep,"*costmap_footprint.mat"));
            footprintFilePath = strcat(footprintFile.folder,filesep,footprintFile.name);
            obstacleFile = dir(strcat(base,subject,filesep,trajFolder{1},filesep,"*obstacles.mat"));
            obstacleFilePath = strcat(footprintFile.folder,filesep,obstacleFile.name);
            
            distCSVName = erase(obstacleFile.name,"costmap_translator_obstacles.mat");
            distCSVPath = strcat(obstacleFile.folder,filesep,distCSVName,"dist2ob.csv");
            
            disp(distCSVPath);
            export_dist(distCSVPath,footprintFilePath,obstacleFilePath);
            
            
        end
    end

end

function export_dist(distCSVPath,footprintFilePath,obstacleFilePath)
    dist = get_dist2obs(footprintFilePath,obstacleFilePath);
    dist = dist';
    dist(:,1) = int64(dist(:,1).*1000);
    % Export matrix 
    writematrix(dist,distCSVPath);
    
end


function dist = get_dist2obs(footprintPath,obstaclePath)
    
    % takes in a path for a global footprint, obstacles, and the ros start
    % time of the bag
    % name specifies the subject and op specifices the operational mode
    % (A0, A1, or A2)
    
    %pull the 3 files
    fp = load(footprintPath).move_base_global_costmap_footprint;
    obstacles = load(obstaclePath).costmap_translator_obstacles;
    
    n_o = length(obstacles);                      % should be lower the n_wc due to sampling freq
    n_wc = length(fp);
    dist = [];                                   % first row is time and minimum distance at that time, unknown length because of empty measurements
    
    wc_index = 0;
    for i = 1:n_o
       t_o = obstacles{i}.header.stamp.time;
       check = 0;
       if isempty(obstacles{i}.cells) == 0    %check if obstacles detected
           while check == 0
               wc_index = wc_index + 1;   %breaks before loop finishes to prevent error
               if wc_index >= n_wc
                   break;
               end
               if fp{wc_index}.header.stamp.time >= t_o
                  
                   if wc_index == 1
                       d = calc_min_d2o(fp{wc_index}.polygon.points, obstacles{i}.cells);
                   else
                       c1 = fp{wc_index}.header.stamp.time;
                       c2 = fp{wc_index-1}.header.stamp.time;
                       if abs(c1-t_o) > abs(c2-t_o)
                           d = calc_min_d2o(fp{wc_index-1}.polygon.points, obstacles{i}.cells);
                       else
                           d = calc_min_d2o(fp{wc_index}.polygon.points, obstacles{i}.cells);
                           if wc_index > 1
                                wc_index = wc_index - 1;
                           end
                       end
                   end
                   check = 1;
               end
               
           end
           dist = [dist, [t_o;d]];   % Originally [t_o;d]
       end
    end
    dist(1,:) = dist(1,:);
end

function d =  calc_min_d2o(fp, cells)
    n1 = length(fp(1,:));
    n2 = length(cells(1,:));
    d = 10000;
    for i = 1:n1
        for j = 1:n2
            dt = sqrt((fp(1,i)-cells(1,j))^2 + (fp(2,i)-cells(2,j))^2);
            if dt < d
                d = dt;
            end
        end
    end
end