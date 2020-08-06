# Get weighted average of HRV windows 

import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 
from scipy.io import loadmat

# HRV directory 
cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
HRV_paths = glob.glob(str(mainDir)+"\\HRV_allSubj_TasksWindows\\*.csv")

# Task times dataframe
task_times_df = pd.read_csv(str(mainDir)+"\\RawECG_byTasks\\task_times.csv")

# Save directory
savedir = str(mainDir)+"\\HRV_weightedWindows\\"

# Initialize dataframe to save weighted HRV for all tasks 
HRV_metrics_list = ["SDNN","RMSSD","ulf","vlf","lf","hf","lfhf","SD1","SD2","SD1SD2","ApEn"]
columns = ["Task"] + HRV_metrics_list
HRV_weighted_df = pd.DataFrame(columns=columns)

for HRV_path in HRV_paths:
    HRV_df = pd.read_csv(HRV_path)
    task_nameList = HRV_path.split("\\")[-1].split("_")
    task_name = task_nameList[0]+"_"+task_nameList[1]+"_"+task_nameList[2]+"_"+task_nameList[3]
    task_time = int(round(task_times_df[task_times_df["Task"] == task_name]["Task time"]))
    
    # Initialize dictionary to store weighted HRV metric values
    HRV_weighted_dict = {}
    HRV_weighted_dict["Task"] = task_name
    for metric in HRV_metrics_list:
        HRV_weighted_dict[metric] = [] 
    
    # Loop through each row in HRV_df 
    for index, row in HRV_df.iterrows():
        
        # If there is no value in NNmean, continue to next loop 
        if pd.isna(row["NNmean"]):
            continue
        else:
            # Calculate overlap weight
            win_start_time = row["t_start"]
            win_end_time = row["t_end"]
            task_start_time = 30
            task_end_time = task_start_time + task_time
            # print(win_end_time)
            if win_end_time <= task_end_time:
                overlap_weight = abs((win_end_time-task_start_time)/30)
                if overlap_weight > 1:
                    overlap_weight = 1
            elif win_end_time > task_end_time:
                overlap_weight = abs((task_end_time - win_start_time)/30)
                if overlap_weight > 1:
                    overlap_weight = 1
            # Store weighted HRV in HRV weighted dictionary
            for metric in HRV_metrics_list:
                HRV_weighted_dict[metric].append(overlap_weight*row[metric])
    # If there is nothing recorded, skip to next loop 
    if HRV_weighted_dict["SDNN"] == []:
        continue 
    # Get the weighted average for each HRV metric in the task 
    for metric in HRV_metrics_list:
        HRV_weighted_dict[metric] = np.mean(HRV_weighted_dict[metric])
    # print(HRV_weighted_dict)
    # Append dictionary to dataframe 
    task_df = pd.DataFrame(HRV_weighted_dict,index=[0])
    HRV_weighted_df = HRV_weighted_df.append(task_df,ignore_index=True)
# Export dataframe 
HRV_weighted_df.to_csv(savedir+"HRV_weightedAvg.csv")


# For each HRV dataframe, find task time, start time of task is always at 30s 
# e.g. Task time = 7.5, start time = 30s, end time = 38s 
# Find the amount of overlapping time
# If window end time < task end time:
# overlap time = window end time - task start time
# If overlap time > 30, overlap time = 30 
# overlap weight = overlap time /30 
# metric weighting = metric * overlap weight

# If window end time > task end time:
# overlap time = task end time - window start time
# metric weighting 

# Sum weighted metrics 