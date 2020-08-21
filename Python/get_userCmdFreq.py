# Get time domain (moving average) and frequency domain (fft) user commands 

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
def movingAvg(impulses, samp_freq, window):
    # window in no. of seconds
    matLength = len(range(0,len(impulses)-samp_freq*window,samp_freq))
    movingAvg_mat = np.zeros((matLength,2))
    seconds_count = 0
    for i in range(0,len(impulses)-samp_freq*window,samp_freq):
        impulseCount = 0
        for j in range(i,i+samp_freq*window):
            if impulses[j] == 1:
                impulseCount +=1
        movingAvg_mat[seconds_count,0] = seconds_count
        movingAvg_mat[seconds_count,1] = impulseCount/window
        seconds_count+=1
    movingAvg_df = pd.DataFrame(movingAvg_mat)
    return movingAvg_df

# Function to get moving average frequency 
def peaksFreq(impulses, samp_freq, window):
    # window in no. of seconds
    matLength = len(range(0,len(impulses)-samp_freq*window,samp_freq))
    peaksFreq_mat = np.zeros((matLength,2))
    seconds_count = 0
    for i in range(0,len(impulses)-samp_freq*window,samp_freq):
        pks,_ = find_peaks(impulses[i:i+samp_freq*window])
        no_pks = len(pks)
        peaksFreq_mat[seconds_count,0] = seconds_count
        peaksFreq_mat[seconds_count,1] = no_pks/window
        seconds_count+=1
    peaksFreq_df = pd.DataFrame(peaksFreq_mat)
    return peaksFreq_df


# Function to get PSD using Welch's method 
"""Use window sampling frequency """
def getPSD(impulses, samp_freq, window):
    freqs, psd = welch(impulses[:samp_freq*window],fs=samp_freq)
    matLength = len(range(0,len(impulses)-samp_freq*window,samp_freq))
    psd_mat = np.zeros((matLength,len(freqs)))
    index = range(0,len(impulses)-samp_freq*window,samp_freq)
    psd_df = pd.DataFrame(psd_mat,columns=freqs,index=index)
    # window in no. of seconds
    for i in range(0,len(impulses)-samp_freq*window,samp_freq):
        # Welch's method
        freqs, psd = welch(impulses[i:i+samp_freq*window],fs=samp_freq)
        for j,freq in enumerate(freqs):
            psd_df.loc[i,freq] = psd[j]
    return psd_df



for subjFolder in subjFolderList:
    trajFolders = glob.glob(subjFolder+"\\*\\")
    for trajFolder in trajFolders:
        if "A0" in trajFolder:
            userImpFilename = glob.glob(trajFolder+"*userImpulses.csv")[0]
            # print(freqs30Filename)
            FilenameList = userImpFilename.split("\\")[-1]
            subj = FilenameList.split("_")[0]
            interface = FilenameList.split("_")[1]
            autonomy = FilenameList.split("_")[2]
            eventName = subj.lower()+"_"+interface+"_"+autonomy
            print(eventName)
            # Read user impulse dataframe
            userImp_df = pd.read_csv(userImpFilename,header=None)
            # Get user impulse sampling frequency 
            sampTime_userImp = (userImp_df.iloc[-1,0] - userImp_df.iloc[0,0])/(userImp_df.shape[0])*0.001 # time units are ms 
            sampFreq_userImp = int(1/sampTime_userImp)
            print(sampFreq_userImp)
            # Read event start and end times 
            startTime, endTime = readEvent_start_end(ECGDir,subj,interface,autonomy)
            startIdx = find_closest(userImp_df,startTime,"Start",0)
            endIdx = find_closest(userImp_df,endTime,"End",0)
            # User impulse event vector
            userImp = userImp_df.iloc[startIdx:endIdx+1,1].values
            # print(userImp.shape)
            # Previously assumed that sampFreq = 25 Hz
            # Get moving averages 
            # movingAvg30_df = movingAvg(userImp, sampFreq_userImp, 30)
            # movingAvg60_df = movingAvg(userImp, sampFreq_userImp, 60)
            # movingAvg30_df.to_csv(trajFolder+eventName+"_movingAvg_30.csv",header=None,index=None)
            # movingAvg60_df.to_csv(trajFolder+eventName+"_movingAvg_60.csv",header=None,index=None)
            # Get PSD and frequencies 
            # psd30_df = getPSD(userImp, sampFreq_userImp, 30)
            # psd60_df = getPSD(userImp, sampFreq_userImp, 60)
            # psd30_df.to_csv(trajFolder+eventName+"_psd_30.csv") 
            # psd60_df.to_csv(trajFolder+eventName+"_psd_60.csv") 
            # Get peaks frequency 
            pksFreq30_df = peaksFreq(userImp, sampFreq_userImp, 30)
            pksFreq60_df = peaksFreq(userImp, sampFreq_userImp, 60)
            pksFreq30_df.to_csv(trajFolder+eventName+"_pksFreq_30.csv",header=None,index=None) 
            pksFreq60_df.to_csv(trajFolder+eventName+"_pksFreq_60.csv",header=None,index=None) 

## odom.pos.position, odom.pos.orientation, distance to obstacle (raw)
## smoothness, jerk, frequency - time domain and total power,  