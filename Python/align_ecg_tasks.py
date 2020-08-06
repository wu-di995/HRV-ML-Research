# Align ecg with tasks

import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 
from scipy.io import loadmat


# ECG directory 
cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
ECG_subjPaths = glob.glob("E:\\argall-lab-data\\ECG_combined_bySubj\\*\\")
ECGPaths = []
for i in range(len(ECG_subjPaths)):
    ECGPaths += [path for path in glob.glob(ECG_subjPaths[i]+"*.csv")]
# print((ECGPaths))

# Tasks directory
tasks_subjPaths = glob.glob("E:\\argall-lab-data\Trajectory Data\\*\\")
taskFolderPaths = []
for i in range(len(tasks_subjPaths)):
    taskFolderPaths += [path for path in glob.glob(tasks_subjPaths[i]+"*\\") if "S" in path.split("\\")[-2].split("_")[-1] ]
taskPaths = []
for i in range(len(taskFolderPaths)):
    taskPaths += [path for path in glob.glob(taskFolderPaths[i]+"*task_status_cleaned.mat")]
# print(taskPaths)

# Save directory 
# savedir = str(mainDir)+"\\ECG_tasks_merged\\"
savedir = "E:\\argall-lab-data\\ECG_tasks_merged\\"
# print(savedir)

# def getColumnPaths(array, title, path):
#     columns = []
#     array = array[title][0][0]
#     path.append(title)
#     pxth = list(path)
#     names = array.dtype.names
#     if names == None:
#         return [path]
#     for n in names:
#         columns = columns + (getColumnPaths(array, n, list(path)))
#     return columns

# def getDataByColumns(array, cols):
#     dataSet = []
#     for c in cols:
#         data = array
#         for d in c[1:]:
#             data = data[d][0][0]
#         dataSet.append(data.squeeze())
#     return dataSet

# def mat2pdDataFrame(filepath, test=False):
#     annots = loadmat(filepath)
#     keys = [k for k in annots.keys() if not k.startswith('_')]
#     title = keys[0] #usually there is only one single key that doesn't start with '__'
#     # 'columns' is of the form [[root, child1, childOfChild1], [root, child2], ... ]
#     # where each list is the path through the .mat file to some data
#     # and also where each list functions as the multi-leveled column names
#     columns = getColumnPaths(annots, title, [])
#     if test: print(title, columns) #for debugging

#     df = pd.DataFrame()
#     for a in range(0, annots[title].shape[1]):
#         df = df.append([getDataByColumns(annots[title][0][a], columns)], ignore_index = True)
#     col_depth = max([len(c) for c in columns])
#     # Creates list of tuples for each header based on path to data, and also adds '↓' to shorter paths
#     test = [tuple([a for a in c] + ['↓']*(col_depth-len(c))) for c in columns]
#     df.columns = pd.MultiIndex.from_tuples(test)

#     return df


testpath = taskPaths[0]

def read_taskStatusMat(task_status_path):
    mat_struct = loadmat(task_status_path)
    task_status = mat_struct["task_status"].flatten()
    no_tasks  = task_status.shape[0]
    # Create dataframe for task status 
    times_list = []
    task_status_list = []
    collisions_list = []
    major_collisions_list = []
    columns = ["Timestamp (ms)", "Task Status", "Collision", "Major Collision"]
    task_times_df = pd.DataFrame(columns=columns)
    # Read mat file and fill in columns 
    for task in range(no_tasks):
        # Start and End Times 
        startTime = int((np.squeeze(task_status[task][0]))*1000) # Convert to ms, multiply 1000 
        endTime = int((np.squeeze(task_status[task][1]))*1000)
        # Timestamp (ms) column 
        times_list.append(startTime)
        times_list.append(endTime)
        # Task status column 
        task_status_list.append("Start")
        task_status_list.append("End")
        # Collision 
        collision = list((task_status[task][2]).flatten())
        collision = [int(time*1000) for time in collision]
        if collision == []:
            collision = 0
        collisions_list.append(0)
        collisions_list.append(collision)
        # Major Collision 
        maj_collision = list((task_status[task][3]).flatten())
        maj_collision = [int(time*1000) for time in maj_collision]
        if maj_collision == []:
            maj_collision = 0
        major_collisions_list.append(0)
        major_collisions_list.append(maj_collision)
    # Fill in dataframe columns 
    task_times_df["Timestamp (ms)"] = times_list
    task_times_df["Task Status"] = task_status_list
    task_times_df["Collision"] = collisions_list
    task_times_df["Major Collision"] = major_collisions_list
    return task_times_df

def merge_taskECG(ECGPaths,taskPaths):
    for taskpath in taskPaths:
        subj = taskpath.split("\\")[-1].split("_")[0].lower()
        if subj in ["u11","u12","u13","u14"]:
            ecgpath = [path for path in ECGPaths if subj in path][0]
            print(ecgpath)
            task_df = read_taskStatusMat(taskpath)
            ecg_df = pd.read_csv(ecgpath).iloc[:,1:]
            merged_df = pd.merge_ordered(ecg_df,task_df,on="Timestamp (ms)",how="outer")
            # Save merged dataframe
            filenameList = taskpath.split("\\")[-1].split("_")
            filename = filenameList[0].lower()+"_"+filenameList[1]+"_"+filenameList[2]+"_"+filenameList[3]
            print(filename)
            merged_df.to_csv(savedir+filename+".csv")

merge_taskECG(ECGPaths,taskPaths)

print((ECGPaths))
# print(len(taskPaths))
# print(taskPaths)