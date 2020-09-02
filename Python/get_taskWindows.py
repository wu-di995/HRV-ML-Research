# Find the task a given window belongs to
import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 

# Task status files
taskStatusPaths = glob.glob("E:\\argall-lab-data\\TaskStatus\\*.csv")

# Feature files 
totalPowerPaths = glob.glob("E:\\argall-lab-data\\UserCmdFreq_byEvent\\*close_30.csv")
totalPowerPaths+=(glob.glob("E:\\argall-lab-data\\UserCmdFreq_byEvent\\*close_60.csv"))

# Read event name from a path
def readEvent(path):
    filenameList = path.split("\\")[-1].split("_")
    event = filenameList[0].lower()+"_"+filenameList[1]+"_"+filenameList[2]
    return event

# Function to read the starting task 
def read_startTask(taskStatusPath):
    filenameList = taskStatusPath.split("\\")[-1].split("_")
    startTask = filenameList[-1].lstrip("S").rstrip(".csv")
    startTask = int(startTask)
    return startTask

# Function to read the start and end times for a given task 
def readTaskStartEndTime(taskStatus_df,task_no):
    # Convert to ms
    startTime = int(taskStatus_df.iloc[task_no-1,0]*1000)
    endTime = int(taskStatus_df.iloc[task_no-1,1]*1000)
    return (startTime,endTime)

# Function to get list of start and end times for all tasks, as a 
def get_allStartEndTimes(taskStatus_df):
    no_tasks = taskStatus_df.shape[0]
    taskTimesList = []
    for task_no in range(1,no_tasks+1):
        startEndTime = readTaskStartEndTime(taskStatus_df,task_no)
        taskTimesList.append(startEndTime)
    return taskTimesList

# Function to check the task(s) a given window (pair of start/end times) belongs to
def find_taskWin(taskTimesList, feat_startTime, feat_endTime, startTask):
    # Get lists for task start/end times
    task_startTimes = sorted([times[0] for times in taskTimesList])
    task_endTimes = sorted([times[1] for times in taskTimesList])
    # print(task_startTimes)
    # print(task_endTimes)
    # print(feat_startTime)
    # Get the list of task startTimes BEFORE feat_startTime
    task_startTimes_bef = [time for time in task_startTimes if time<=feat_startTime]
    # Get closest task_startTime that is BEFORE feat_startTime
    if len(task_startTimes_bef) == 0:
        start_taskNo = 1
    else:
        closest_taskStartTime = min(task_startTimes_bef,key = lambda x: abs(x-feat_startTime))
        start_taskNo = task_startTimes.index(closest_taskStartTime)+1 #Add one to convert index to task number 
    # Get the list of task endTimes AFTER feat_endTime
    task_endTimes_aft = [time for time in task_endTimes if time>=feat_endTime]
    # Get closest task_endTime that is AFTER feat_endTime
    if len(task_endTimes_aft) == 0:
        end_taskNo = 7
    else:
        closest_taskEndTime = min(task_endTimes_aft,key = lambda x: abs(x-feat_endTime))
        end_taskNo = task_endTimes.index(closest_taskEndTime)+1  #Add one to convert index to task number 
    # If start and end task numbers are the same, return task number and task number breakdown
    if start_taskNo == end_taskNo:
        feat_task = start_taskNo + startTask -1
        if feat_task>7:
            feat_task = feat_task%7
        feat_taskBreakdown = [feat_task,1] #Task number, ratio, in chronological order 
        return (feat_task, feat_taskBreakdown)
    # If start and end task numbers are not the same, calculate the task that spans the most in the window
    else:
        task_span = range(start_taskNo,end_taskNo+1)
        feat_taskBreakdown = []
        winTime = feat_endTime-feat_startTime
        for i,task_no in enumerate(task_span):
            if i == 0: # First task
                task_endTime = task_endTimes[task_no-1]
                feat_taskTimeRatio = (task_endTime-feat_startTime)/winTime
                feat_taskBreakdown.append(task_no)
                feat_taskBreakdown.append(feat_taskTimeRatio)
            elif i == len(task_span)-1: # Last task
                prevTask_endTime = task_endTimes[task_no-2] # endTime of the task before
                feat_taskTimeRatio = (feat_endTime-prevTask_endTime)/winTime
                feat_taskBreakdown.append(task_no)
                feat_taskBreakdown.append(feat_taskTimeRatio)
            else: # Tasks in between
                prevTask_endTime = task_endTimes[task_no-2]
                currTask_endTime = task_endTimes[task_no-1]
                feat_taskTimeRatio = (currTask_endTime-prevTask_endTime)/winTime
                feat_taskBreakdown.append(task_no)
                feat_taskBreakdown.append(feat_taskTimeRatio)
        # Convert task numbers to align with startTask
        for i,value in enumerate(feat_taskBreakdown):
            if i%2 == 0: #Even index, which is the task number
                taskNo_aligned = (startTask+value-1)
                if taskNo_aligned>7: # 7 is the total number of tasks
                    taskNo_aligned = taskNo_aligned%7
                feat_taskBreakdown[i] = taskNo_aligned
        # Get the task that spans the most in the window
        longestTaskTime = max(feat_taskBreakdown[1::2])
        feat_task = feat_taskBreakdown[(feat_taskBreakdown.index(longestTaskTime)-1)]
        return feat_task, feat_taskBreakdown

# Function to check tasks for a dataframe of windows
def check_tasksForWindows(taskStatusPath, featPath):
    taskStatus_df = pd.read_csv(taskStatusPath)
    feature_df = pd.read_csv(featPath)
    # Start task
    startTask = read_startTask(taskStatusPath)
    # Task times list
    taskTimesList = get_allStartEndTimes(taskStatus_df)
    # Create new dataframe to save feature task window times
    featTaskWin_df = pd.DataFrame(index=range(feature_df.shape[0]),columns=["Start Win Time", "End Win Time", "Main Task", "Tasks Breakdown"])
    # Loop through all windows
    for i,row in feature_df.iterrows():
        feat_startTime = row["Start Win Time"]
        feat_endTime = row["End Win Time"]
        feat_task, feat_taskBreakdown = find_taskWin(taskTimesList, feat_startTime, feat_endTime, startTask)
        featTaskWin_df.loc[i,"Start Win Time"] = row["Start Win Time"]
        featTaskWin_df.loc[i,"End Win Time"] = row["End Win Time"]
        featTaskWin_df.loc[i,"Main Task"] =  feat_task
        featTaskWin_df.loc[i,"Tasks Breakdown"] = feat_taskBreakdown
    return featTaskWin_df

savedir = "E:\\argall-lab-data\\totalPower_taskWindows\\"
if __name__ == "__main__":
    for featPath in totalPowerPaths:
        event = readEvent(featPath)
        print(event)
        if "30" in featPath:
            time = "30"
        elif "60" in featPath:
            time = "60"
        taskStatusPath = [path for path in taskStatusPaths if event in path][0]
        featTaskWin_df = check_tasksForWindows(taskStatusPath, featPath)
        featTaskWin_df.to_csv(savedir+event+"_totalPower_"+time+".csv")
