# Compare start and end times of ECG, userCtrl, userImpulse

# Imports
import pandas as pd
import numpy as np
import glob,os, pathlib

# Trajectory data folder
subjFolders = glob.glob("E:\\argall-lab-data\\Trajectory Data\\*\\")
subjFolders = [path for path in subjFolders if "U00" not in path]
teleopFolders = []
for subjFolder in subjFolders:
    trajFolders = glob.glob(subjFolder+"*\\")
    for trajFolder in trajFolders:
        if "A0" in trajFolder:
            teleopFolders.append(trajFolder) 
supStatusPaths = []
userImpPaths = []
for teleopFolder in teleopFolders:
    supStatusPaths.append(glob.glob(teleopFolder+"*supStatus.csv")[0])
    userImpPaths.append(glob.glob(teleopFolder+"*_userImpulses.csv")[0])

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
ECGPaths = [glob.glob(path+"*full_ecg.csv")[0] for path in subjECGDir]
# print(len(ECGPaths))

# Get start and end times for an event 
def readEvent_start_end(ECGDir,subj,interface,autonomy):
    # print(subj)
    subjFolder = glob.glob(ECGDir+"*"+subj.lower()+"\\")[0]
    # print("Subjfolder", subjFolder)
    annotPath = glob.glob(subjFolder+"\\*\\")[0]+subj.lower()+"\\annotations.csv"
    annot_df = pd.read_csv(annotPath)
    # print("Annotpath", annotPath)
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

# Function to get start and end times of event 
def compareTimes(ECGDataPath,ECGPaths,supStatusPaths,userImpPaths):
    # Check that the number of supStatus and userImp paths are equal 
    if len(supStatusPaths) != len(userImpPaths):
        return "Unequal number of paths!"
    # Create dataframe to save times
    compareTimesColumns = ["Event", "Annot Start Time", "Annot End Time",
                           "ECG Start Time", "ECG End Time", 
                           "UserCtrl Start Time", "UserCtrl End Time",
                           "UserImp Start Time", "UserImp End Time",]
    compareTimes_df = pd.DataFrame(columns=compareTimesColumns)
    for idx, (supStatusPath,userImpPath) in enumerate(zip(supStatusPaths,userImpPaths)):
        # Read subject, interface and autonomy 
        filenameList = supStatusPath.split("\\")[-1].split("_")
        subject = filenameList[0].lower()
        interface = filenameList[1]
        autonomy = filenameList[2]
        event = subject+"_"+interface+"_"+autonomy
        print(event)
        # Read start and end times from annotPaths
        annotStartTime, annotEndTime = readEvent_start_end(ECGDataPath,subject,interface,autonomy)
        print(annotStartTime, annotEndTime)
        # Find ECG start and end times
        ECGPath = [path for path in ECGPaths if subject in path][0]
        ECG_df = pd.read_csv(ECGPath)
        ECGStartIndex = find_closest(ECG_df,annotStartTime,"Start",times_col_idx=1,test=False,match_annot=True)
        ECGStartTime = ECG_df.iloc[ECGStartIndex,1]
        # print(ECGStartTime)
        ECGEndIndex = find_closest(ECG_df,annotEndTime,"End",times_col_idx=1,test=False,match_annot=False)
        ECGEndTime = ECG_df.iloc[ECGEndIndex,1]
        # print(ECGEndTime)
        # Find userCtrl start and end times
        supStatus_df = pd.read_csv(supStatusPath)
        supStatusStartIndex = find_closest(supStatus_df,annotStartTime,"Start",times_col_idx=0,test=False,match_annot=True)
        supStatusStartTime = supStatus_df.iloc[supStatusStartIndex,1]
        supStatusEndIndex = find_closest(supStatus_df,annotEndTime,"End",times_col_idx=0,test=False,match_annot=False)
        supStatusEndTime = supStatus_df.iloc[supStatusEndIndex,1]
        # Find userImp start and end times 
        userImp_df = pd.read_csv(userImpPath)
        userImpStartIndex = find_closest(userImp_df,annotStartTime,"Start",times_col_idx=0,test=False,match_annot=True)
        userImpStartTime = supStatus_df.iloc[userImpStartIndex,1]
        userImpEndIndex = find_closest(userImp_df,annotEndTime,"End",times_col_idx=0,test=False,match_annot=False)
        userImpEndTime = supStatus_df.iloc[userImpEndIndex,1]
        # Save times
        compareTimes_df.loc[idx,"Event"] = event
        compareTimes_df.loc[idx,"Annot Start Time"] = annotStartTime
        compareTimes_df.loc[idx,"Annot End Time"] = annotEndTime
        compareTimes_df.loc[idx,"ECG Start Time"] = ECGStartTime
        compareTimes_df.loc[idx,"ECG End Time"] = ECGEndTime
        compareTimes_df.loc[idx,"UserCtrl Start Time"] = supStatusStartTime
        compareTimes_df.loc[idx,"UserCtrl End Time"] = supStatusEndTime
        compareTimes_df.loc[idx,"UserImp Start Time"] = userImpStartTime
        compareTimes_df.loc[idx,"UserImp End Time"] = userImpEndTime
    return compareTimes_df

compareTimes_df = compareTimes(ECGDataPath,ECGPaths,supStatusPaths,userImpPaths)

# Save directory 
savedir = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\"
compareTimes_df.to_csv(savedir+"compareTimes.csv")