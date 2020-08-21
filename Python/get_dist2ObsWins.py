# Get sliding windows for distance to obstacles for each event

import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 
from scipy.io import loadmat
from Documents.SPARC.scripts.smoothness import sparc
from scipy.signal import welch 
from scipy.signal import find_peaks

# Trajectory data folder 
subjFolderList = glob.glob("E:\\argall-lab-data\\Trajectory Data\\*\\")
subjFolderList = [path for path in subjFolderList if "U00" not in path]
# subjs = ["U04","U05","U06","U07","U08","U09","U10","U11","U12","U13","U14"]
# subjs = ["S01"]
# subjFolderList = [path for path in subjFolderList if any(subj in path for subj in subjs)]
# print(subjFolderList)

# ECG data folder
ECGDir = "E:\\argall-lab-data\\ECG Data\\"

# Get start and end times for an event 
def readEvent_start_end(ECGDir,subj,interface,autonomy):
    subjFolder = glob.glob(ECGDir+"*"+subj.lower()+"\\")[0]
    annotPath = glob.glob(subjFolder+"\\*\\")[0]+subj.lower()+"\\annotations.csv"
    annot_df = pd.read_csv(annotPath)
    # print(annotPath)
    # Convert interface label
    if interface == "HA":
        interface = "Headarray"
    elif interface == "JOY":
        interface = "Joystick"
    elif interface == "SNP":
        interface = "Sip-n-puff"
    # Convert autonomy label
    if autonomy == "A0":
        autonomy = "Teleoperation"
    elif autonomy == "A1":
        autonomy = "Low level autonomy"
    elif autonomy == "A2":
        autonomy = "Mid level autonomy"
    # Get start and end times of event
    searchString = interface + " - " + autonomy 
    startTime = annot_df[annot_df["EventType"].str.contains(searchString)]["Start Timestamp (ms)"].values[0]
    endTime = annot_df[annot_df["EventType"].str.contains(searchString)]["Stop Timestamp (ms)"].values[0]
    # Convert to int
    startTime = int(startTime)
    endTime = int(endTime)
    
    
    return startTime, endTime

# Function to find index of closest lower neighbour of a timestamp
def find_closest(df,timestamp,idxType,times_col_idx,test=False,match_annot=True):
    # times_col_idx is the timestamp column index in df 
    if test:
        print(df.columns[times_col_idx])
    # If annot start time earlier than traj start time, use traj start time 
    if match_annot:
        if timestamp<df.iloc[0,times_col_idx]:
            timestamp = df.iloc[0,times_col_idx]
    exactmatch = (df[df.columns[times_col_idx]]==timestamp)
    while (exactmatch[exactmatch==True].empty):
        # print(timestamp)
        timestamp -=1
        exactmatch = (df[df.columns[times_col_idx]]==timestamp)
    if idxType == "Start":
        idx = (exactmatch[exactmatch==True].index[0])
    elif idxType == "End":
        idx = (exactmatch[exactmatch==True].index[-1])
    return idx
    

# Function to get moving average frequency 
def dist2ObsWins(dist, samp_freq, window):
    # window in no. of seconds
    matLength = len(range(0,len(dist)-samp_freq*window,samp_freq))
    dist2ObsWins_mat = np.zeros((matLength,2))
    seconds_count = 0
    for i in range(0,len(dist)-samp_freq*window,samp_freq):
        minDist = min(dist[i:i+samp_freq*window])            
        dist2ObsWins_mat[seconds_count,0] = seconds_count
        dist2ObsWins_mat[seconds_count,1] = minDist
        seconds_count+=1
    dist2ObsWins_df = pd.DataFrame(dist2ObsWins_mat)
    return dist2ObsWins_df


for subjFolder in subjFolderList:
    trajFolders = glob.glob(subjFolder+"\\*\\")
    for trajFolder in trajFolders:
        if "A0" in trajFolder:
            dist2ObFilename = glob.glob(trajFolder+"*dist2ob.csv")[0]
            # print(freqs30Filename)
            FilenameList = dist2ObFilename.split("\\")[-1]
            subj = FilenameList.split("_")[0]
            interface = FilenameList.split("_")[1]
            autonomy = FilenameList.split("_")[2]
            eventName = subj.lower()+"_"+interface+"_"+autonomy
            print(eventName)
            # Read user impulse dataframe
            dist2Obs_df = pd.read_csv(dist2ObFilename,header=None)
            # Get user impulse sampling frequency 
            sampTime_dist = (dist2Obs_df.iloc[-1,0] - dist2Obs_df.iloc[0,0])/(dist2Obs_df.shape[0])*0.001 # time units are ms 
            sampFreq_dist = int(round(1/sampTime_dist))
            print(sampFreq_dist)
            # Read event start and end times 
            startTime, endTime = readEvent_start_end(ECGDir,subj,interface,autonomy)
            startIdx = find_closest(dist2Obs_df,startTime,"Start",0)
            endIdx = find_closest(dist2Obs_df,endTime,"End",0)
            # User impulse event vector
            dist = dist2Obs_df.iloc[startIdx:endIdx+1,1].values
            # Get dist2Obs Windows 
            dist2ObsWins30_df = dist2ObsWins(dist, sampFreq_dist, 30)
            dist2ObsWins60_df = dist2ObsWins(dist, sampFreq_dist, 60)
            dist2ObsWins30_df.to_csv(trajFolder+eventName+"_dist2ObsWins_30.csv",header=None,index=None) 
            dist2ObsWins60_df.to_csv(trajFolder+eventName+"_dist2ObsWins_60.csv",header=None,index=None) 