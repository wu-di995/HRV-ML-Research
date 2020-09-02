# Get user controlled dataframes, user control status for each window, based on ECG
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
# print(len(supStatusPaths))

# ECG start end times files
ECGdir = "E:\\argall-lab-data\\ECG_eventStartEndTimes\\"
ECG30Paths = glob.glob(ECGdir+"*30.csv")
ECG60Paths = glob.glob(ECGdir+"*60.csv")

# print(len(ECG30Paths))
# print(len(ECG60Paths))

# Save directory
savedir = "E:\\argall-lab-data\\UserControlled_byEventNEW\\"

# Read event name from a path
def readEvent(path):
    filenameList = path.split("\\")[-1].split("_")
    event = filenameList[0].lower()+"_"+filenameList[1]+"_"+filenameList[2]
    return event

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
    # Limit, stop searching for timestamp if it exceeeds 1 second in either direction of reference, return None for time and None for index
    timestamp_ref = timestamp
    # Search method - "C" find closest timestamp (search both directions, returns closest)
    if method == "C":
        timestamp_aft = timestamp 
        timestamp_bef = timestamp 
        exactmatch_aft = (df[df.columns[times_col_idx]]==timestamp_aft)
        exactmatch_bef = (df[df.columns[times_col_idx]]==timestamp_bef)
        # Search for closest timestamp after
        while (exactmatch_aft[exactmatch_aft==True].empty) and (abs(timestamp_aft-timestamp_ref)<1000):
            timestamp_aft +=1
            exactmatch_aft = (df[df.columns[times_col_idx]]==timestamp_aft)
        # timestamp_aft is None if it exceeds 1s limit 
        if (abs(timestamp_aft-timestamp_ref)>=1000):
            timestamp_aft = None
        # Search for closest timestamp before
        while (exactmatch_bef[exactmatch_bef==True].empty) and (abs(timestamp_bef-timestamp_ref)<1000):
            timestamp_bef +=1
            exactmatch_bef = (df[df.columns[times_col_idx]]==timestamp_bef)
        # timestamp_bef is none if it exceeds 1s limit
        if (abs(timestamp_bef-timestamp_ref)>=1000):
            timestamp_bef = None 
        # Check which is closest to reference timestamp if both timestamps are not None 
        if (timestamp_aft == None) and (timestamp_bef == None):
            return None, None 
        elif (timestamp_aft == None): # Return timestamp_bef
            idx = (exactmatch_bef[exactmatch_bef==True].index[0]) 
            return timestamp_bef, idx
        elif (timestamp_bef == None): # Return timestamp_aft
            idx = (exactmatch_aft[exactmatch_aft==True].index[0])
            return timestamp_aft, idx  
        else: # Neither timestamps are None 
            if abs(timestamp_aft-timestamp) < abs(timestamp_bef-timestamp):
                idx = (exactmatch_aft[exactmatch_aft==True].index[0])
                return timestamp_aft, idx 
            else:
                idx = (exactmatch_bef[exactmatch_bef==True].index[0])
                return timestamp_bef, idx 
    # Search method -  "A" find closest timestamp after
    elif method == "A":
        exactmatch = (df[df.columns[times_col_idx]]==timestamp)
        while (exactmatch[exactmatch==True].empty) and (abs(timestamp-timestamp_ref)<1000):
            timestamp +=1
            exactmatch = (df[df.columns[times_col_idx]]==timestamp)
        if (abs(timestamp-timestamp_ref)>=1000):
            return None, None
        else:
            idx = (exactmatch[exactmatch==True].index[0])
            return timestamp, idx 
    # Search method - "B" find closest timestamp before
    elif method == "B":
        exactmatch = (df[df.columns[times_col_idx]]==timestamp)
        while (exactmatch[exactmatch==True].empty) and (abs(timestamp-timestamp_ref)<1000):
            timestamp -=1
            exactmatch = (df[df.columns[times_col_idx]]==timestamp)
        if (abs(timestamp-timestamp_ref)>=1000):
            return None, None
        else:
            idx = (exactmatch[exactmatch==True].index[0])
            return timestamp, idx 
    # Search method - "CA" find closest timestamp (both directions) and closest after
    elif method == "CA":
        timestamp_aft = timestamp 
        timestamp_bef = timestamp 
        exactmatch_aft = (df[df.columns[times_col_idx]]==timestamp_aft)
        exactmatch_bef = (df[df.columns[times_col_idx]]==timestamp_bef)
        # Search for closest timestamp after
        while (exactmatch_aft[exactmatch_aft==True].empty) and (abs(timestamp_aft-timestamp_ref)<1000):
            timestamp_aft +=1
            exactmatch_aft = (df[df.columns[times_col_idx]]==timestamp_aft)
        # print(timestamp_aft-timestamp_ref)
        # timestamp_aft is none if it exceeds 1s limit
        if (abs(timestamp_aft-timestamp_ref)>=1000):
            timestamp_aft = None
        # Search for closest timestamp before
        while (exactmatch_bef[exactmatch_bef==True].empty) and (abs(timestamp_bef-timestamp_ref)<1000):
            timestamp_bef +=1
            exactmatch_bef = (df[df.columns[times_col_idx]]==timestamp_bef)
        # print(timestamp_bef-timestamp_ref)
        # timestamp_bef is none if it exceeds 1s limit
        if (abs(timestamp_bef-timestamp_ref)>=1000):
            timestamp_bef = None 
        # If both are None, return all Nones
        if (timestamp_aft == None) and (timestamp_bef==None):
            return None, None, None, None
        # If timestamp_aft == None, timestamp_bef is the closest
        elif (timestamp_aft==None):
            idx_closest = (exactmatch_bef[exactmatch_bef==True].index[0])
            timestamp_closest = timestamp_bef
            return timestamp_closest, idx_closest, None, None 
        # If timestamp_bef == None, timestamp_aft is the closest 
        elif (timestamp_bef==None):
            idx_closest = (exactmatch_aft[exactmatch_aft==True].index[0])
            timestamp_closest = timestamp_aft
            return timestamp_closest, idx_closest, timestamp_closest, idx_closest
        else: # Neither timestamps are None 
        # Check which is closest to reference timestamp 
            if abs(timestamp_aft-timestamp) <= abs(timestamp_bef-timestamp):
                # print(exactmatch_aft[exactmatch_aft==True].index)
                idx_closest = (exactmatch_aft[exactmatch_aft==True].index[0])
                timestamp_closest = timestamp_aft
            else:
                idx_closest = (exactmatch_bef[exactmatch_bef==True].index[0])
                timestamp_closest = timestamp_bef
            idx_aft = (exactmatch_aft[exactmatch_aft==True].index[0])
            return timestamp_closest, idx_closest, timestamp_aft, idx_aft

# Function to calculate user control status and percentage 
def get_userCtrlSP(user_window):
    if 0 in user_window:
        user_controlled = 0
    else:
        user_controlled = 1
    user_control_ratio= np.count_nonzero(np.array(user_window))/len(user_window)
    return user_controlled, user_control_ratio

# Function to process user control status for a supStatus dataframe and ECG windowed dataframe
def get_userCtrlWindows(supStatus_df, ECGWin_df):
    # Create a new dataframe to save user control status
    userCtrl_columns = ["Start Win Time", "End Win Time","User Controlled", "User Control Ratio"]
    # Dataframe using closest search method for start, and before search method for end
    userCtrl_close_df = pd.DataFrame(columns=userCtrl_columns, index = range(len(ECGWin_df))) 
    # Dataframe using after search method for start, and before search method for end 
    userCtrl_aft_df = pd.DataFrame(columns=userCtrl_columns, index = range(len(ECGWin_df)))
    # Iterate through each row of ECGWin_df 
    for idx,row in ECGWin_df.iterrows():
        ECGStartTime = row["Start Times"]
        ECGEndTime = row["End Times"]
        startTime_close_exist = 0
        startTime_aft_exist = 0
        endTime_exist = 0
        # User Control start times 
        startTime_close, idx_close, startTime_aft, idx_aft = find_closest(supStatus_df,ECGStartTime,"CA",0,test=False,match_annot=False)
        if startTime_close!=None:
            startTime_close_exist = 1
        if startTime_aft!=None:
            startTime_aft_exist = 1
        # If both startTimes do not exist, continue to next loop 
        if (startTime_close_exist == 0) and (startTime_aft_exist == 0):
            continue
        # User Control end times
        endTime, idx_end = find_closest(supStatus_df,ECGEndTime,"B",0,False,False)
        if endTime!= None:
            endTime_exist = 1
        # If endTime does not exist, continue to next loop
        if endTime_exist == 0:
            continue
        # If both startTimes and end times exist
        elif (startTime_close_exist == 1) and (startTime_aft_exist == 1):
            if (endTime<=startTime_close) or (endTime<=startTime_aft):
                continue
        # Slice supStatus and calculate user control status and percentage
        if startTime_close_exist:
            user_window_close = supStatus_df.iloc[idx_close:idx_end+1,1].values
            userControlled_close, user_control_ratio_close = get_userCtrlSP(user_window_close)
            # Save start and end times
            userCtrl_close_df.loc[idx,"Start Win Time"] = startTime_close
            userCtrl_close_df.loc[idx,"End Win Time"] = endTime
            # Save user controlled status and ratio
            userCtrl_close_df.loc[idx,"User Controlled"] = userControlled_close
            userCtrl_close_df.loc[idx,"User Control Ratio"] = user_control_ratio_close
        if startTime_aft_exist:
            user_window_aft = supStatus_df.iloc[idx_aft:idx_end+1,1].values
            userControlled_aft, user_control_ratio_aft = get_userCtrlSP(user_window_aft)
            # Save start and end times
            userCtrl_aft_df.loc[idx,"Start Win Time"] = startTime_aft
            userCtrl_aft_df.loc[idx,"End Win Time"] = endTime
            # Save user controlled status and ratio
            userCtrl_aft_df.loc[idx,"User Controlled"] = userControlled_aft
            userCtrl_aft_df.loc[idx,"User Control Ratio"] = user_control_ratio_aft
    return userCtrl_close_df, userCtrl_aft_df

# For each Supstatus file, read the ECG30s windows and ECG60s windows dataframes 
# For each pair of start and end times, slice the window, calculate the userCtrl percentage, status, and save them

if len(ECG30Paths) == len(ECG60Paths):
    for supStatusPath in supStatusPaths:
        event = readEvent(supStatusPath)
        print(event)
        ECG30Path = [path for path in ECG30Paths if event in path][0]
        ECG60Path = [path for path in ECG60Paths if event in path][0]
        # Read datframes
        supStatus_df = pd.read_csv(supStatusPath)
        ECG30_df = pd.read_csv(ECG30Path)
        ECG60_df = pd.read_csv(ECG60Path)
        # Generate user control dataframes 
        print("30")
        userCtrl30_close_df, userCtrl30_aft_df = get_userCtrlWindows(supStatus_df,ECG30_df)
        print("60")
        userCtrl60_close_df, userCtrl60_aft_df = get_userCtrlWindows(supStatus_df,ECG60_df)
        # Save dataframes
        userCtrl30_close_df.to_csv(savedir+event+"_close_30.csv")
        userCtrl30_aft_df.to_csv(savedir+event+"_aft_30.csv")
        userCtrl60_close_df.to_csv(savedir+event+"_close_60.csv")
        userCtrl60_aft_df.to_csv(savedir+event+"_aft_60.csv")

"""
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
"""