# Get user controlled dataframes 
import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 
from scipy.io import loadmat

# Supstatus Files 
subjFolders = glob.glob("E:\\argall-lab-data\\Trajectory Data\\*\\")
subjFolders = [path for path in subjFolders if "U00" not in path]
teleopFolders = []
for subjFolder in subjFolders:
    trajFolders = glob.glob(subjFolder+"*\\")
    for trajFolder in trajFolders:
        if "A0" in trajFolder:
            teleopFolders.append(trajFolder)
supStatusPaths = []
for teleopFolder in teleopFolders:
    supStatusPaths.append(glob.glob(teleopFolder+"*supStatus.csv")[0])

# ECG data folder
ECGDir = "E:\\argall-lab-data\\ECG Data\\"

# Save directory
savedir = "E:\\argall-lab-data\\UserControlled_byEvent\\"

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

# Generator for sliding windows
def slidingWin_gen(timeSeries,windowSize,stepSize):
    for i in range(0,len(timeSeries),stepSize):
        window = timeSeries[i:i+windowSize]
        yield window

# Generate user control dataframe
def mk_userCtrl_df(userControl,samp_freq,windowSizeSeconds,stepSizeSeconds):
    # userControl, vector of 1s and 0s, 1 means user controlled
    windowSize = windowSizeSeconds*samp_freq
    stepSize = stepSizeSeconds*samp_freq
    # Create dataframe to save SPARC outputs
    userCtrl_columns = ["Start Win Time", "End Win Time","User Controlled", "User Control Ratio"] # Need to do math for Start Win Time, End Win Time
    userCtrl_df = pd.DataFrame(columns=userCtrl_columns)
    # Fill in dataframe 
    user_windows = slidingWin_gen(userControl,windowSize,stepSize)
    # Check if user operates for that entire window 
    for i,user_window in enumerate(user_windows):
        if 0 in user_window:
            user_controlled = 0
        else:
            user_controlled = 1
        # Calculate user control ratio
        user_control_ratio= np.count_nonzero(np.array(user_window))/len(user_window)
        # Fill in values for sparc_df
        userCtrl_df.loc[i,"Start Win Time"] = i
        userCtrl_df.loc[i,"End Win Time"] = i + windowSizeSeconds
        userCtrl_df.loc[i,"User Controlled"] = user_controlled
        userCtrl_df.loc[i,"User Control Ratio"] = user_control_ratio
    
    return userCtrl_df

# Generate user controlled dataframes for a list of paths
def get_userCtrl_paths(supStatusPaths, ECGDir):
    for i,path in enumerate(supStatusPaths):
        event = path.split("\\")[-1].rstrip("supStatus.csv")
        subj = event.split("_")[0].lower()
        interface = event.split("_")[1]
        autonomy = event.split("_")[2]
        event = subj+"_"+interface+"_"+autonomy
        print(event)
        # Load supStatus df
        supStatus_df = pd.read_csv(path)
        # Read event start and end times 
        startTime, endTime = readEvent_start_end(ECGDir,subj,interface,autonomy)
        startIdx = find_closest(supStatus_df,startTime,"Start",0)
        endIdx = find_closest(supStatus_df,endTime,"End",0)
        # Get sampling frequency for user control 
        sampTime = (endTime-startTime)/(endIdx-startIdx) * 0.001 # units are in ms
        sampFreq = int(1/sampTime)
        print(sampFreq)
        # Get user controlled vector
        userCtrl = supStatus_df.iloc[startIdx:endIdx+1,1].values
        # Create and save user controlled dataframes
        userCtrl30_df = mk_userCtrl_df(userCtrl,sampFreq,30,1)
        userCtrl60_df = mk_userCtrl_df(userCtrl,sampFreq,60,1)
        userCtrl30_df.to_csv(savedir+"30s\\"+event+"_userCtrl.csv")
        userCtrl60_df.to_csv(savedir+"60s\\"+event+"_userCtrl.csv")

get_userCtrl_paths(supStatusPaths, ECGDir)
# Function to extract user controlled information 
# def extractUserControlled(userCmdPaths):
#     for path in userCmdPaths:
#         userCmd_df = pd.read_csv(path)            
#         event = path.split("\\")[-1].rstrip("sparc.csv")
#         filename = event+"userCtrl.csv"
#         userControlled_df = userCmd_df.loc[:,["Start Win Time", "End Win Time", "User Controlled", "User Control Ratio"]]
#         if "30s" in path:
#             userControlled_df.to_csv(savedir+"30s\\"+filename)
#         elif "60s" in path:
#             userControlled_df.to_csv(savedir+"60s\\"+filename)

# extractUserControlled(sparc_userCmd30Paths)
# extractUserControlled(sparc_userCmd60Paths)