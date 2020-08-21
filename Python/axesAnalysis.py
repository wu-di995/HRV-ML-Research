# Plot axes outputs for analysis 
import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq, fftshift

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
def find_closest(df,timestamp,times_col_idx,test=False):
    # times_col_idx is the timestamp column index in df 
    if test:
        print(df.columns[times_col_idx])
    # If annot start time earlier than traj start time, use traj start time 
    if timestamp<df.iloc[0,times_col_idx]:
        timestamp = df.iloc[0,times_col_idx]
    exactmatch = (df[df.columns[times_col_idx]]==timestamp)
    while (exactmatch[exactmatch==True].empty):
        # print(timestamp)
        timestamp -=1
        exactmatch = (df[df.columns[times_col_idx]]==timestamp)
    return (exactmatch[exactmatch==True].index[0])

for subjFolder in subjFolderList:
    trajFolders = glob.glob(subjFolder+"\\*\\")
    for trajFolder in trajFolders:
        if "A0" in trajFolder:
            axesFilename = glob.glob(trajFolder+"*axes.csv")[0]
            # print(freqs30Filename)
            FilenameList = axesFilename.split("\\")[-1]
            subj = FilenameList.split("_")[0]
            interface = FilenameList.split("_")[1]
            autonomy = FilenameList.split("_")[2]
            eventName = subj.lower()+"_"+interface+"_"+autonomy
            print(eventName)
            # Read user impulse dataframe
            axes_df = pd.read_csv(axesFilename,header=None)
            # Read event start and end times 
            startTime, endTime = readEvent_start_end(ECGDir,subj,interface,autonomy)
            startIdx = find_closest(axes_df,startTime,0)
            endIdx = find_closest(axes_df,endTime,0)
            # Axes vector
            axes1 = axes_df.iloc[startIdx:endIdx+1,1].values
            axes2 = axes_df.iloc[startIdx:endIdx+1,2].values
            # Plotting raw axes data
            fig,ax = plt.subplots()
            pks1,_ = find_peaks(axes1)
            ax.plot(axes1)
            ax.plot(pks1,axes1[pks1],"x")
            ax.set_title(eventName)
            plt.show()
            plt.close()

            # FFT
            # fft1 = abs((fft(axes1)))
            # fft1 = fft1[range(int(len(axes1)/2))]
            # N = len(axes1)
            # samp_freq = 25
            # values = np.arange(int(N/2))
            # timePeriod= N/samp_freq # Total Time 
            # print(values)
            # freq = values/timePeriod
            # fig, ax = plt.subplots()
            # ax.plot(freq,fft1)
            # plt.show()
            # plt.close()

            # Find peaks
            # samp_freq = 25
            # pks1,_ = find_peaks(axes1)
            # pks2,_ = find_peaks(axes2)
            # totalTime1 = len(axes1)/samp_freq
            # totalTime2 = len(axes1)/samp_freq
            # avg = len(pks1)/totalTime1 #user interactions per 60s 
            # print(avg)
            # Find peaks method might be appropriate way for SNP(mapped discretely) and HA
            

# Joystick remove values over abs(1)