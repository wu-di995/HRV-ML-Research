# Generate sparc on velocity profile 
import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 
from scipy.io import loadmat
from Documents.SPARC.scripts.smoothness import sparc

# Trajectory data folder 
subjFolderList = glob.glob("E:\\argall-lab-data\\Trajectory Data\\*\\")
subjFolderList = [path for path in subjFolderList if "U00" not in path]
# subjs = ["U04","U05","U06","U07","U08","U09","U10","U11","U12","U13","U14"]
# subjs = ["S01"]
# subjFolderList = [path for path in subjFolderList if any(subj in path for subj in subjs)]

# u3_SNP_A0 timestamps do not match starttime is 1536696177568, traj start time 1536696180987
print(subjFolderList)

# ECG data folder
ECGDir = "E:\\argall-lab-data\\ECG Data\\"

# Save directory 
lin_savedir = "E:\\argall-lab-data\\SPARC_userCmd_byEvent\\lin\\"
ang_savedir = "E:\\argall-lab-data\\SPARC_userCmd_byEvent\\ang\\"

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

# Generator for sliding windows
def slidingWin_gen(timeSeries,windowSize,stepSize):
    for i in range(0,len(timeSeries),stepSize):
        window = timeSeries[i:i+windowSize]
        yield window

# Generate SPARC outputs dataframe
def makeSPARCdf(velocity,userControl,samp_freq,windowSizeSeconds,stepSizeSeconds):
    # userControl, vector of 1s and 0s, 1 means user controlled
    windowSize = windowSizeSeconds*samp_freq
    stepSize = stepSizeSeconds*samp_freq
    # Create dataframe to save SPARC outputs
    sparc_columns = ["Start Win Time", "End Win Time","sal","User Controlled", "User Control Ratio"] # Need to do math for Start Win Time, End Win Time
    sparc_index = range(len(velocity))
    sparc_df = pd.DataFrame(columns=sparc_columns)
    # Fill in dataframe 
    windows = slidingWin_gen(velocity,windowSize,stepSize)
    user_windows = slidingWin_gen(userControl,windowSize,stepSize)
    # Check if user operates for that entire window 
    for i,(window,user_window) in enumerate(zip(windows,user_windows)):
        # if window is all zeros, smooth= 0, and continue to next loop
        if max(window)==0:
            sal = 0
        else: 
            sal,_,_= sparc(window,samp_freq)
        # if all the values in the user control window are 0s
        if 0 in user_window:
            user_controlled = 0
        else:
            user_controlled = 1
        # Calculate user control ratio
        user_control_ratio= np.count_nonzero(np.array(user_window))/len(user_window)
        # Fill in values for sparc_df
        sparc_df.loc[i,"sal"] = sal
        sparc_df.loc[i,"Start Win Time"] = i
        sparc_df.loc[i,"End Win Time"] = i+30
        sparc_df.loc[i,"User Controlled"] = user_controlled
        sparc_df.loc[i,"User Control Ratio"] = user_control_ratio
    
    return sparc_df



# Function to find index of closest lower neighbour of a timestamp
def find_closest(df,timestamp,times_col_idx,test=False):
    # times_col_idx is the timestamp column index in df 
    if test:
        print(df.columns[times_col_idx])
    print(df.loc[0,df.columns[times_col_idx]])
    # If annot start time earlier than traj start time, use traj start time
    if timestamp < df.loc[0,df.columns[times_col_idx]]:
        timestamp = df.loc[0,df.columns[times_col_idx]]
    # Check for exactmatch 
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
            userCmdFilename = glob.glob(trajFolder+"*user_cmd.csv")[0]
            # print(userCmdFilename)
            userCmdFilenameList = userCmdFilename.split("\\")[-1]
            subj = userCmdFilenameList.split("_")[0]
            interface = userCmdFilenameList.split("_")[1]
            autonomy = userCmdFilenameList.split("_")[2]
            eventName = subj.lower()+"_"+interface+"_"+autonomy
            print(eventName)
            columnNames = ["Timestamp", "User Control", "Lin_userCmd", "Ang_userCmd"]
            userCmd_df = pd.read_csv(userCmdFilename,header = None, names=columnNames)
            # Read event start and end times 
            startTime, endTime = readEvent_start_end(ECGDir,subj,interface,autonomy)
            print("Start Time: ",startTime)
            startIdx = find_closest(userCmd_df,startTime,0)
            # print(startIdx)

            # print(endTime)
            endIdx = find_closest(userCmd_df,endTime,0)
            # print(endIdx)
            print(endIdx-startIdx)

            # Get user control array
            userControl = userCmd_df.loc[startIdx:endIdx+1,"User Control"].values

            # Slice velocity arrays from start to end times 
            lin_vel = userCmd_df.loc[startIdx:endIdx+1,"Lin_userCmd"].values
            ang_vel = userCmd_df.loc[startIdx:endIdx+1,"Ang_userCmd"].values

            # Sampling frequency 25 Hz
            samp_freq = 25

            # Generate SPARC dataframes
            # Linear 
            # 30s
            lin_sparc_30_df = makeSPARCdf(lin_vel,userControl,samp_freq,30,1)
            lin_sparc_30_df.to_csv(lin_savedir+"30s\\"+eventName+"_sparc.csv")
            # 60s
            lin_sparc_60_df = makeSPARCdf(lin_vel,userControl,samp_freq,60,1)
            lin_sparc_60_df.to_csv(lin_savedir+"60s\\"+eventName+"_sparc.csv")
            # Angular 
            # 30s
            ang_sparc_30_df = makeSPARCdf(ang_vel,userControl,samp_freq,30,1)
            ang_sparc_30_df.to_csv(ang_savedir+"30s\\"+eventName+"_sparc.csv")
            # 60s
            ang_sparc_60_df = makeSPARCdf(ang_vel,userControl,samp_freq,60,1)
            ang_sparc_60_df.to_csv(ang_savedir+"60s\\"+eventName+"_sparc.csv")

            
            
        
# def getColumnPaths(array, title, path):
#     columns = []
#     array = array[title][0][0]
#     path.append(title)
#     pxth = list(path)
#     names = array.dtype.names
#     if names == None:
#         return [path]
#     for n in names:
#         columns = columns + (getColumnPaths(array, n, list(path)))
#     return columns

# def getDataByColumns(array, cols):
#     dataSet = []
#     for c in cols:
#         data = array
#         for d in c[1:]:
#             data = data[d][0][0]
#         dataSet.append(data.squeeze())
#     return dataSet

# def mat2pdDataFrame(filepath, test=False):
#     annots = loadmat(filepath)
#     keys = [k for k in annots.keys() if not k.startswith('_')]
#     title = keys[0] #usually there is only one single key that doesn't start with '__'

#     # 'columns' is of the form [[root, child1, childOfChild1], [root, child2], ... ]
#     # where each list is the path through the .mat file to some data
#     # and also where each list functions as the multi-leveled column names
#     columns = getColumnPaths(annots, title, [])
#     if test: print(title, columns) #for debugging

#     df = pd.DataFrame()
#     for a in range(0, annots[title].shape[1]):
#         df = df.append([getDataByColumns(annots[title][0][a], columns)], ignore_index = True)

#     col_depth = max([len(c) for c in columns])
#     # Creates list of tuples for each header based on path to data, and also adds '↓' to shorter paths
#     test = [tuple([a for a in c] + ['↓']*(col_depth-len(c))) for c in columns]
#     df.columns = pd.MultiIndex.from_tuples(test)

#     return df



