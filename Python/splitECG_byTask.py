# Split ECG recordings by task
import pandas as pd
import numpy as np
import glob,os, pathlib
import re
from scipy.io import loadmat
from pathlib import Path 


# Trajectory Folder 
trajDir = "/home/skrdown/Documents/argall-lab-data/Trajectory Data/"
trajSubjsPaths = glob.glob(trajDir+"*"+os.sep)
# Exclude u00
trajSubjsPaths = [path for path in trajSubjsPaths if "u00" not in path.lower()]
# ECG Folder
ecgDir = "/home/skrdown/Documents/argall-lab-data/ECG_bySubj/"
ecgPaths = glob.glob(ecgDir+"*"+os.sep)
print(ecgPaths)

# Directory to save to 
# savedir = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\RawECG_byTasks\\"
savedir = "/home/skrdown/HRV-ML-Research/ECG/RawECG_byTasks/"

# For each trajectory subject folder, find the session folders 
path = trajSubjsPaths[0]
A0folders = glob.glob(path+"*A0*")

# Function to find index of closest lower neighbour of a timestamp
def find_closest(df,timestamp):
    # print(df.columns[1])
    exactmatch = (df[df.columns[1]]==timestamp)
    while (exactmatch[exactmatch==True].empty):
        timestamp -=1
        exactmatch = (df[df.columns[1]]==timestamp)
    return (exactmatch[exactmatch==True].index[0])
    # df_sort = df.iloc[(df[df.columns[1]]-timestamp).abs().argsort()[:1]]
    # index = df_sort.index.tolist()[0]
    # print(index)
    # return index 

def splitECG(trajSubjsPaths):
    # Create dataframe to save task times 
    task_time_df_cols = ["Task", "Task time", "Index diff time", "Start Time", "End Time"]
    task_time_df = pd.DataFrame(columns=task_time_df_cols)
    tasks_list = []
    task_time_list = []
    task_idxTime_list = []
    start_time_list = []
    end_time_list = []
    for path in trajSubjsPaths:
        A0folders = glob.glob(path+"*A0*")
        for folder in A0folders:
            # event = folder.split("\\")[-1]
            # event = event.rstrip(event.split("_")[-1])
            # print(event)
            nameSplitList = folder.split(os.sep)[-1].split("_")
            subj = nameSplitList[0].lower()
            print(folder)
            print(subj)
            interface = nameSplitList[1]
            autonomy = nameSplitList[2]
            start_task = nameSplitList[3][-1]
            task_no = int(start_task)
            event = subj+"_"+interface+"_"+autonomy
            # Retrieive the corresponding ECG file 
            ecgpath = glob.glob(list(filter(lambda x: subj in x, ecgPaths))[0]+"*.csv")[0]
            # print(ecgpath)
            ecgfull_df = pd.read_csv(ecgpath)
            # print(ecgfull_df.shape)
            # print(subj,interface,autonomy,start_task)
            # print(subj)
            # Read task status file 
            task_status_path = glob.glob(folder+os.sep+"*_task_status_cleaned.mat")[0]
            mat_struct = loadmat(task_status_path)
            task_status = mat_struct["task_status"].flatten()
            no_tasks = task_status.shape[0]
            for task in range(no_tasks):
                # Task Name
                taskName = event + "_" + str(task_no)
                task_no += 1 
                if (task_no>7):
                    task_no = 1
                # print(taskName)
                # startTime = int(np.squeeze(task_status[task][0])) #Round down to the nearest integer 
                # endTime = int(np.squeeze(task_status[task][1])) 
                startTime = int((np.squeeze(task_status[task][0]))*1000) # Convert to ms, multiply 1000 
                endTime = int((np.squeeze(task_status[task][1]))*1000)
                start_time_list.append(startTime)
                end_time_list.append(endTime)
                # print(startTime,endTime)
                taskTime = (endTime-startTime)/1000
                # In the ECG dataframe, select ecg between start/end times 
                startIndex = find_closest(ecgfull_df,startTime)
                endIndex = find_closest(ecgfull_df,endTime)
                indexDiff = (endIndex - startIndex)*0.004
                # print(endIndex)
                # Create new df
                task_df = ecgfull_df.iloc[startIndex:endIndex+1,2]
                # Convert to mV
                task_df = task_df.apply(lambda x:x*1000)
                # Save mV values
                subjsavedir = savedir+subj+os.sep
                Path(subjsavedir).mkdir(parents=True,exist_ok=True)
                newPath = subjsavedir+taskName+".csv"
                task_df.to_csv(newPath, index=False)
                # Append task and task times to lists
                tasks_list.append(taskName)
                task_time_list.append(taskTime)
                task_idxTime_list.append(indexDiff)
    # Save dataframe as csv 
    task_time_df["Task"] = tasks_list
    task_time_df["Task time"] = task_time_list
    task_time_df["Index diff time"] = task_idxTime_list
    task_time_df["Start Time"] = start_time_list
    task_time_df["End Time"] = end_time_list
    task_time_df.to_csv(savedir+"task_times.csv",index=False)

splitECG(trajSubjsPaths)

# filepath = "E:\\argall-lab-data\\S00\\S00_HA_A0_S4\\S00_HA_A0_S4_task_status_cleaned.mat"
# mat_struct = loadmat(filepath)
# task_status = mat_struct["task_status"].flatten() # Each element contains 1 task 
# task2 = task_status[1]
# def taskStatus_reader(matfilepath):
#     mat_struct = loadmat(matfilepath)
#     task_status = mat_struct["task_status"].flatten()
#     no_tasks = task_status.shape[0]
#     start_end_list = [0]*no_tasks
#     for task in range(no_tasks):
#         start_time = np.squeeze(task_status[task][0])
#         end_time = np.squeeze(task_status[task][1])
#         start_end_list[task] = (start_time,end_time)
# print(new_arr)
# [(array([[1.53859407e+09]]), array([[1.53859412e+09]]), array([[1.53859408e+09]]), array([], shape=(0, 0), dtype=uint8))
#  (array([[1.53859415e+09]]), array([[1.53859427e+09]]), array([[1.53859417e+09, 1.53859417e+09]]), array([], shape=(0, 0), dtype=uint8))
#  (array([[1.53859429e+09]]), array([[1.53859432e+09]]), array([], shape=(0, 0), dtype=uint8), array([], shape=(0, 0), dtype=uint8))
#  (array([[1.53859433e+09]]), array([[1.53859434e+09]]), array([], shape=(0, 0), dtype=uint8), array([], shape=(0, 0), dtype=uint8))
#  (array([[1.53859436e+09]]), array([[1.53859439e+09]]), array([], shape=(0, 0), dtype=uint8), array([], shape=(0, 0), dtype=uint8))
#  (array([[1.53859439e+09]]), array([[1.53859448e+09]]), array([[1.53859446e+09]]), array([], shape=(0, 0), dtype=uint8))
#  (array([[1.53859448e+09]]), array([[1.5385945e+09]]), array([], shape=(0, 0), dtype=uint8), array([], shape=(0, 0), dtype=uint8))]

# print(np.squeeze(task2[-2]))

# print(type(task1))
# print(task1[0])
# print(type(task1[0][0].flatten()))
# print(task_status.shape)
# first_start_time = task_status[0][1][0]
# print(type(first_start_time))
# print(first_start_time.flatten())
