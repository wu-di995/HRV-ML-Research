# Split Raw ECG according to events 

# Imports
import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 
from scipy.io import loadmat


# Full ECG dataframes directory 
subjECGDir = glob.glob("E:\\argall-lab-data\\ECG_combined_bySubj\\*\\")
# Exclude u00
subjECGDir = [path for path in subjECGDir if "u00" not in path]
# print(subjECGDir)
# Annotations folders
ECGDataPath = "E:\\argall-lab-data\\ECG Data\\"
# Subjects
subjsList = [path.split("\\")[-2] for path in subjECGDir]
# print(subjsList)
annotPaths = []
for i,subj in enumerate(subjsList):
    annotPaths.append(ECGDataPath+"laa_wc_multi_session_"+subj+"\\LAA_WC_Multi_Session\\"+subj+"\\annotations.csv")


# Save directory
savedir = "E:\\argall-lab-data\\ECG_byEventNew\\"
ECGforHRVDir = "E:\\argall-lab-data\\ECG_byEvent_forHRV\\"
for subj in subjsList:
    subjsavedir = savedir+subj+"\\"
    subjECGforHRVDir = ECGforHRVDir+subj+"\\"
    Path(subjsavedir).mkdir(parents=True,exist_ok=True)
    Path(subjECGforHRVDir).mkdir(parents=True,exist_ok=True)

# # Function to find index of closest lower neighbour of a timestamp
# def find_closest(df,timestamp,times_col_idx,test=False):
#     # times_col_idx is the timestamp column index in df 
#     if test:
#         print(df.columns[times_col_idx])
#     exactmatch = (df[df.columns[times_col_idx]]==timestamp)
#     while (exactmatch[exactmatch==True].empty):
#         timestamp -=1
#         exactmatch = (df[df.columns[times_col_idx]]==timestamp)
#     print(timestamp)
#     return (exactmatch[exactmatch==True].index[0])


# Function to find closest index and closest timestamp to reference timestamp
def find_closest(df,timestamp,method,times_col_idx,test=False,match_annot=True):
    # print(timestamp)
    # times_col_idx is the timestamp column index in df 
    if test:
        print(df.columns[times_col_idx])
    # If annot start time earlier than traj start time, use traj start time 
    if match_annot:
        if timestamp<df.iloc[0,times_col_idx]:
            timestamp = df.iloc[0,times_col_idx]

    exactmatch = (df[df.columns[times_col_idx]]==timestamp)
    # Search method - "C" find closest timestamp (search both directions, returns closest)
    if method == "C":
        timestamp_aft = timestamp 
        timestamp_bef = timestamp 
        exactmatch_aft = (df[df.columns[times_col_idx]]==timestamp_aft)
        exactmatch_bef = (df[df.columns[times_col_idx]]==timestamp_bef)
        # Search for closest timestamp after
        while (exactmatch_aft[exactmatch_aft==True].empty):
            timestamp_aft +=1
            exactmatch_aft = (df[df.columns[times_col_idx]]==timestamp_aft)
        # Search for closest timestamp before
        while (exactmatch_bef[exactmatch_bef==True].empty):
            timestamp_bef +=1
            exactmatch_bef = (df[df.columns[times_col_idx]]==timestamp_bef)
        # Check which is closest to reference timestamp 
        if abs(timestamp_aft-timestamp) < abs(timestamp_bef-timestamp):
            idx = (exactmatch_aft[exactmatch_aft==True].index[0])
            return timestamp_aft, idx 
        else:
            idx = (exactmatch_bef[exactmatch_bef==True].index[0])
            return timestamp_bef, idx 
    # Search method -  "A" find closest timestamp after
    elif method == "A":
        exactmatch = (df[df.columns[times_col_idx]]==timestamp)
        while (exactmatch[exactmatch==True].empty):
            timestamp +=1
            exactmatch = (df[df.columns[times_col_idx]]==timestamp)
        idx = (exactmatch[exactmatch==True].index[0])
        return timestamp, idx 
    # Search method - "B" find closest timestamp before
    elif method == "B":
        exactmatch = (df[df.columns[times_col_idx]]==timestamp)
        while (exactmatch[exactmatch==True].empty):
            timestamp -=1
            exactmatch = (df[df.columns[times_col_idx]]==timestamp)
        idx = (exactmatch[exactmatch==True].index[0])
        return timestamp, idx 
    # Search method - "CA" find closest timestamp (both directions) and closest after
    elif method == "CA":
        timestamp_aft = timestamp 
        timestamp_bef = timestamp 
        exactmatch_aft = (df[df.columns[times_col_idx]]==timestamp_aft)
        exactmatch_bef = (df[df.columns[times_col_idx]]==timestamp_bef)
        # Search for closest timestamp after
        while (exactmatch_aft[exactmatch_aft==True].empty):
            timestamp_aft +=1
            exactmatch_aft = (df[df.columns[times_col_idx]]==timestamp_aft)
        # Search for closest timestamp before
        while (exactmatch_bef[exactmatch_bef==True].empty):
            timestamp_bef +=1
            exactmatch_bef = (df[df.columns[times_col_idx]]==timestamp_bef)
        # Check which is closest to reference timestamp 
        if abs(timestamp_aft-timestamp) < abs(timestamp_bef-timestamp):
            idx_closest = (exactmatch_aft[exactmatch_aft==True].index[0])
            timestamp_closest = timestamp_aft
        else:
            idx_closest = (exactmatch_bef[exactmatch_bef==True].index[0])
            timestamp_closest = timestamp_bef
        idx_aft = (exactmatch_aft[exactmatch_aft==True].index[0])
        return timestamp_closest, idx_closest, timestamp_aft, idx_aft


for annotPath in annotPaths:
    annot_df = pd.read_csv(annotPath)
    # Load the corresponding ECG file for that subject 
    subj = annotPath.split("\\")[-2]
    ecgPath = [path for path in subjECGDir if subj in path][0]+"full_ecg.csv"
    ecgfull_df = pd.read_csv(ecgPath)
    # Timestamp column index = 1
    # print(ecgPath)
    for i in annot_df.index:
        event = annot_df.loc[i]["EventType"]
        print(subj,event)
        # Find indices 
        startTime = int(annot_df.loc[i]["Start Timestamp (ms)"])
        endTime = int(annot_df.loc[i]["Stop Timestamp (ms)"])
        # print(startTime,endTime) 
        _, startIndex = find_closest(ecgfull_df,startTime,method="C",times_col_idx=1,test=False,match_annot=False)
        # startIndex = find_closest(ecgfull_df,startTime,1)
        # print(startIndex)
        _, endIndex = find_closest(ecgfull_df,endTime,method="C",times_col_idx=1,test=False,match_annot=False)
        # endIndex = find_closest(ecgfull_df,endTime,1)
        # print(endIndex)
        # Create new df
        new_df = ecgfull_df.iloc[startIndex:endIndex+1,1:]
        # Convert to mV
        new_df.iloc[:,1] = new_df.iloc[:,1].apply(lambda x:x*1000)
        # Save new dataframe 
        new_df.to_csv(savedir+subj+"\\"+event+".csv",index=None,header=None)
        # ECG for HRV
        ecg4hrv_df = new_df.iloc[:,1]
        ecg4hrv_df.to_csv(ECGforHRVDir+subj+"\\"+event+".csv",index=None,header=None)
        


