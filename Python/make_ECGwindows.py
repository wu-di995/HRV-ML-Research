# Create windows that overlap with task times

# Look for pair of consecutive "Start" and "End" rows 
# If for "Start" or "End" rows there is a voltage value, use that row
# Otherwise, use next row for "Start", previous row for "End"
# From the "Start" row, include 29s before. From the "End" row, include 29s after 

import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 
from scipy.io import loadmat

# ECG_task_merged directory 
cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
# ecgTasksPaths = glob.glob(str(mainDir)+"\\ECG_tasks_merged\\*.csv")
ecgTasksPaths = glob.glob("E:\\argall-lab-data\\ECG_tasks_merged\\*.csv")

# Save directory 
# savedir = str(mainDir)+"\\ECG_overlappingWindowsbyTask\\"
savedir = "E:\\argall-lab-data\\ECG_overlappingWindowsbyTask\\"
# For each start or end index, check if voltage value exists, if not choose later or earlier index
# "Sample (V)"
def selectIdx(df, indices, start=True):
    for i,idx in enumerate(indices):
        if df.iloc[idx,:].isnull().any():
            if start:
                indices[i] = idx+1
            else:
                indices[i] = idx-1
    return indices 

for path in ecgTasksPaths:
    # Filename 
    filenameList = path.split("\\")[-1].split("_")
    filename = filenameList[0]+"_"+filenameList[1]+"_"+filenameList[2]+"_"
    if filenameList[0] in ["u11","u12","u13","u14"]:
        # Read dataframe 
        ecgTask_df = pd.read_csv(path)
        # Get starting task number 
        start_task = int(path.rstrip(".csv")[-1])
        no_tasks = 7
        task_no = start_task
        # Find all instances where "Task Status" == "Start" or "End"
        startIndices = ecgTask_df[ecgTask_df["Task Status"] == "Start"].index.tolist()
        endIndices = ecgTask_df[ecgTask_df["Task Status"] == "End"].index.tolist()
        startIndices = selectIdx(ecgTask_df,startIndices,start=True)
        endIndices = selectIdx(ecgTask_df,endIndices,start=False)
        # Select windows, starting from 29s before "Start" and 29s after "End"
        for i in range(no_tasks):
            try:
                startIdx = startIndices[i] - 29*1000
            except IndexError:
                startIdx = 0
            try: 
                endIdx = endIndices[i] + 29*1000
            except IndexError:
                endIdx = ecgTask_df.shape[1]-1
            # Non Null 
            nonnull = ecgTask_df[~(ecgTask_df["Sample (V)"].isnull())].iloc[:,2]
            # print(nonnull)
            # Create new dataframe 
            task_windows_df = ecgTask_df.iloc[startIdx:endIdx+1,2].apply(lambda x: x*1000) # convert to mV
            task_windows_df = task_windows_df.dropna()
            # Save dataframe 
            print(filename+str(task_no))
            task_windows_df.to_csv(savedir+filename+str(task_no)+".csv",index=None,header=None)
            # Increment task number 
            task_no += 1
            if task_no > 7:
                task_no = 1
