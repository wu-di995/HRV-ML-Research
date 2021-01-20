# Get user frequency window dataframes, moving Average, peaks Average frequency, total power, based on ECG
import pandas as pd
import numpy as np
import glob, os, pathlib
from pathlib import Path 
from scipy.io import loadmat
from scipy.signal import welch 
from scipy.signal import find_peaks

# User Impulses Files 
# subjFolders = glob.glob("E:\\argall-lab-data\\Trajectory Data\\*\\")
subjFolders = glob.glob("/home/skrdown/Documents/argall-lab-data/Trajectory Data/"+"*"+os.sep)
subjFolders = [path for path in subjFolders if "U00" not in path]
teleopFolders = []
for subjFolder in subjFolders:
    trajFolders = glob.glob(subjFolder+"*"+os.sep)
    for trajFolder in trajFolders:
        if "A0" in trajFolder:
            teleopFolders.append(trajFolder)
userImpPaths = []
for teleopFolder in teleopFolders:
    userImpPaths.append(glob.glob(teleopFolder+"*userImpulses.csv")[0])
# print(userImpPaths)

# ECG start end times files
# ECGdir = "E:\\argall-lab-data\\ECG_eventStartEndTimes\\"
ECGdir = "/home/skrdown/Documents/argall-lab-data/ECG_eventStartEndTimes/"
ECG30Paths = glob.glob(ECGdir+"*30.csv")
ECG60Paths = glob.glob(ECGdir+"*60.csv")

# Save directory
# savedir = "E:\\argall-lab-data\\UserCmdFreq_byEvent\\"
savedir = "/home/skrdown/Documents/argall-lab-data/HRV_newgen/UserCmdFreq_byEvent/"

# Read event name from a path
def readEvent(path):
    filenameList = path.split(os.sep)[-1].split("_")
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

# Function to get moving average frequency 
def get_movingAvg(impulsesWin, startTime, endTime):
    windowTimeinSecs = (endTime-startTime)*0.001
    impulsesCount = np.count_nonzero(impulsesWin)
    movingAvg = impulsesCount/windowTimeinSecs
    return movingAvg

# Function to get peaks frequency
def get_peaksFreq(impulsesWin, startTime, endTime):
    windowTimeinSecs = (endTime-startTime)*0.001
    pks,_ = find_peaks(impulsesWin)
    no_pks = len(pks)
    peaksFreq = no_pks/windowTimeinSecs
    return peaksFreq

# Function to get total power 
def get_totalPower(impulsesWin, startTime, endTime):
    windowTimeinSecs = (endTime-startTime)*0.001
    sampFreq = len(impulsesWin)/windowTimeinSecs
    # Welch's method
    freqs, psd = welch(impulsesWin,fs=sampFreq,nperseg=int(len(impulsesWin)/8)) #8 segments, as per MATLAB
    totalPower = np.trapz(psd)
    return totalPower

# Function to process user command frequencies for a user impulses dataframe and ECG windowed dataframe
def get_userFreqWindows(userImp_df, ECGWin_df):
    # Create a new dataframe to save user control status
    freq_columns = ["Start Win Time", "End Win Time","Moving Average", "Peaks Frequency", "Total Power"]
    # Dataframe using closest search method for start, and before search method for end
    freq_close_df = pd.DataFrame(columns=freq_columns, index = range(len(freq_columns))) 
    # Dataframe using after search method for start, and before search method for end 
    freq_aft_df = pd.DataFrame(columns=freq_columns, index = range(len(freq_columns)))
    # Iterate through each row of ECGWin_df 
    for idx,row in ECGWin_df.iterrows():
        ECGStartTime = row["Start Times"]
        ECGEndTime = row["End Times"]
        startTime_close_exist = 0
        startTime_aft_exist = 0
        endTime_exist = 0
        # Freq start times 
        startTime_close, idx_close, startTime_aft, idx_aft = find_closest(userImp_df,ECGStartTime,"CA",0,test=False,match_annot=False)
        print("CA")
        if startTime_close != None:
            startTime_close_exist = 1
        if startTime_aft != None:
            startTime_aft_exist = 1
        # If both startTimes do not exist, continue to next loop 
        elif (startTime_close_exist == 0) and (startTime_aft_exist == 0):
            continue 
        # print(startTime_close_exist,startTime_aft_exist)
        # Freq end times
        endTime, idx_end = find_closest(userImp_df,ECGEndTime,"B",0,False,False)
        print("B")
        if endTime != None:
            endTime_exist = 1
        # If endTime does not exist, continue to next loop
        if endTime_exist == 0:
            continue 
        # If both startTimes exist and endTimes exist 
        elif (startTime_close_exist == 1) and (startTime_aft_exist == 1):
            if (endTime<=startTime_close) or (endTime<=startTime_aft):
                continue
        # Slice supStatus and calculate frequencies 
        if startTime_close_exist:
            userImp_window_close = userImp_df.iloc[idx_close:idx_end+1,1].values
            # Moving average
            movingAvg_close = get_movingAvg(userImp_window_close, startTime_close, endTime)
            # Peaks Frequency 
            peaksFreq_close = get_peaksFreq(userImp_window_close, startTime_close, endTime)
            # Total Power
            totalPower_close = get_totalPower(userImp_window_close, startTime_close, endTime)
            # Save start and end times
            freq_close_df.loc[idx,"Start Win Time"] = startTime_close
            freq_close_df.loc[idx,"End Win Time"] = endTime
            # Save frequencies
            freq_close_df.loc[idx,"Moving Average"] = movingAvg_close
            freq_close_df.loc[idx,"Peaks Frequency"] = peaksFreq_close
            freq_close_df.loc[idx,"Total Power"] = totalPower_close 
        if startTime_aft_exist:
            userImp_window_aft = userImp_df.iloc[idx_aft:idx_end+1,1].values
            # Moving average
            movingAvg_aft = get_movingAvg(userImp_window_aft, startTime_aft, endTime)
            # Peaks Frequency
            peaksFreq_aft = get_peaksFreq(userImp_window_aft, startTime_aft, endTime)
            # Total Power
            totalPower_aft = get_totalPower(userImp_window_aft, startTime_aft, endTime)
            # Save start and end times
            freq_aft_df.loc[idx,"Start Win Time"] = startTime_aft
            freq_aft_df.loc[idx,"End Win Time"] = endTime
            # Save frequencies 
            freq_aft_df.loc[idx,"Moving Average"] = movingAvg_aft
            freq_aft_df.loc[idx,"Peaks Frequency"] = peaksFreq_aft
            freq_aft_df.loc[idx,"Total Power"] = totalPower_aft
    return freq_close_df, freq_aft_df

# For each Supstatus file, read the ECG30s windows and ECG60s windows dataframes 
# For each pair of start and end times, slice the window, calculate the userCtrl percentage, status, and save them

if len(ECG30Paths) == len(ECG60Paths):
    for userImpPath in userImpPaths:
        event = readEvent(userImpPath)
        print(event)
        ECG30Path = [path for path in ECG30Paths if event in path][0]
        ECG60Path = [path for path in ECG60Paths if event in path][0]
        # Read datframes
        userImp_df = pd.read_csv(userImpPath)
        ECG30_df = pd.read_csv(ECG30Path)
        ECG60_df = pd.read_csv(ECG60Path)
        # Generate user control dataframes 
        print("30")
        freq30_close_df, freq30_aft_df = get_userFreqWindows(userImp_df,ECG30_df)
        print("60")
        freq60_close_df, freq60_aft_df = get_userFreqWindows(userImp_df,ECG60_df)
        # Save dataframes
        freq30_close_df.to_csv(savedir+event+"_close_30.csv")
        freq30_aft_df.to_csv(savedir+event+"_aft_30.csv")
        freq60_close_df.to_csv(savedir+event+"_close_60.csv")
        freq60_aft_df.to_csv(savedir+event+"_aft_60.csv")
