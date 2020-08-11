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
# Subjets
subjsList = [path.split("\\")[-2] for path in subjECGDir]
# print(subjsList)
annotPaths = []
for i,subj in enumerate(subjsList):
    annotPaths.append(ECGDataPath+"laa_wc_multi_session_"+subj+"\\LAA_WC_Multi_Session\\"+subj+"\\annotations.csv")
# Save directory
savedir = "E:\\argall-lab-data\\ECG_byEvent\\"
for subj in subjsList:
    subjsavedir = savedir+subj+"\\"
    Path(subjsavedir).mkdir(parents=True,exist_ok=True)

# Function to find index of closest lower neighbour of a timestamp
def find_closest(df,timestamp,times_col_idx,test=False):
    # times_col_idx is the timestamp column index in df 
    if test:
        print(df.columns[times_col_idx])
    exactmatch = (df[df.columns[times_col_idx]]==timestamp)
    while (exactmatch[exactmatch==True].empty):
        timestamp -=1
        exactmatch = (df[df.columns[times_col_idx]]==timestamp)
    return (exactmatch[exactmatch==True].index[0])


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
        startIndex = find_closest(ecgfull_df,startTime,1)
        # print(startIndex)
        endIndex = find_closest(ecgfull_df,endTime,1)
        # print(endIndex)
        # Create new df
        new_df = ecgfull_df.iloc[startIndex:endIndex+1,2]
        # Convert to mV
        new_df = new_df.apply(lambda x:x*1000)
        # Save dataframe 
        new_df.to_csv(savedir+subj+"\\"+event+".csv",index=None)


