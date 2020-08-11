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
subjs = ["S01"]
subjFolderList = [path for path in subjFolderList if any(subj in path for subj in subjs)]

# u3_SNP_A0 timestamps do not match starttime is 1536696177568, traj start time 1536696180987
print(subjFolderList)

# ECG data folder
ECGDir = "E:\\argall-lab-data\\ECG Data\\"

# Save directory 
lin_savedir = "E:\\argall-lab-data\\SPARC_linVel_byEvent\\"
ang_savedir = "E:\\argall-lab-data\\SPARC_angVel_byEvent\\"

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
def makeSPARCdf(velocity,samp_freq,windowSizeSeconds,stepSizeSeconds):
    windowSize = windowSizeSeconds*samp_freq
    stepSize = stepSizeSeconds*samp_freq
    # Create dataframe to save SPARC outputs
    sparc_columns = ["sal", "f, Mf", "f_sel, Mf_sel"]
    sparc_index = range(len(velocity))
    sparc_df = pd.DataFrame(columns=sparc_columns)
    # Fill in dataframe 
    windows = slidingWin_gen(velocity,windowSize,stepSize)
    for i,window in enumerate(windows):
        # if window is all zeros, smooth= 0, and continue to next loop
        if max(window)==0:
            sal = 0
        else: 
            sal,(f, Mf),(f_sel, Mf_sel) = sparc(window,samp_freq)
        sparc_df.loc[i,"sal"] = sal
        sparc_df.loc[i,"f, Mf"] = (f, Mf)
        sparc_df.loc[i,"f_sel, Mf_sel"] = (f_sel, Mf_sel)
    
    return sparc_df



# Function to find index of closest lower neighbour of a timestamp
def find_closest(df,timestamp,times_col_idx,test=False):
    # times_col_idx is the timestamp column index in df 
    if test:
        print(df.columns[times_col_idx])
    # If annot start time earlier than traj start time, use traj start time 
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
            odomFilename = glob.glob(trajFolder+"*odom.csv")[0]
            # print(odomFilename)
            odomFilenameList = odomFilename.split("\\")[-1]
            subj = odomFilenameList.split("_")[0]
            interface = odomFilenameList.split("_")[1]
            autonomy = odomFilenameList.split("_")[2]
            eventName = subj.lower()+"_"+interface+"_"+autonomy
            print(eventName)
            odom_df = pd.read_csv(odomFilename,header = None)
            odom_df.columns = ["Timestamp", "Lin_vel", "Ang_vel"]
            odom_df.iloc[:,0] = odom_df.iloc[:,0].apply(lambda x: int(x*1000)) # Convert ROS timestamps to ms 
            # Read event start and end times 
            startTime, endTime = readEvent_start_end(ECGDir,subj,interface,autonomy)
            print("Start Time: ",startTime)
            startIdx = find_closest(odom_df,startTime,0)
            # print(startIdx)

            # print(endTime)
            endIdx = find_closest(odom_df,endTime,0)
            # print(endIdx)
            print(endIdx-startIdx)

            # Slice velocity arrays from start to end times 
            lin_vel = odom_df.iloc[startIdx:endIdx+1,1].values
            ang_vel = odom_df.iloc[startIdx:endIdx+1,2].values
            # print(len(lin_vel))
            # Sampling frequency 25 Hz
            samp_freq = 25

            # windows = slidingWin_gen(lin_vel,30*samp_freq,30*samp_freq)
            # for window in windows:
            
            # Generate SPARC dataframes
            # Possible problems: padding? incorrect event times? creating windows of zeros?
            lin_sparc_30_df = makeSPARCdf(lin_vel,samp_freq,30,1)
            # lin_sparc_30_df.to_csv(lin_savedir+"30s\\"+eventName+"_sparc.csv")
            # lin_sparc_60_df = makeSPARCdf(lin_vel,samp_freq,60,1)
            # lin_sparc_60_df.to_csv(lin_savedir+"60s\\"+eventName+"_sparc.csv")
            # ang_sparc_30_df = makeSPARCdf(ang_vel,samp_freq,30,1)
            # ang_sparc_30_df.to_csv(ang_savedir+"30s\\"+eventName+"_sparc.csv")
            # ang_sparc_60_df = makeSPARCdf(ang_vel,samp_freq,60,1)
            # ang_sparc_60_df.to_csv(ang_savedir+"60s\\"+eventName+"_sparc.csv")

            # try:
            #     lin_sparc_30_df = makeSPARCdf(lin_vel,samp_freq,30,1)
            #     lin_sparc_30_df.to_csv(lin_savedir+"30s\\"+eventName+"_sparc.csv")
            # except IndexError:
            #     continue 
            # try: 
            #     lin_sparc_60_df = makeSPARCdf(lin_vel,samp_freq,60,1)
            #     lin_sparc_60_df.to_csv(lin_savedir+"60s\\"+eventName+"_sparc.csv")
            # except IndexError:
            #     continue 
            # try: 
            #     ang_sparc_30_df = makeSPARCdf(ang_vel,samp_freq,30,1)
            #     ang_sparc_30_df.to_csv(ang_savedir+"30s\\"+eventName+"_sparc.csv")
            # except IndexError:
            #     continue 
            # try:
            #     ang_sparc_60_df = makeSPARCdf(ang_vel,samp_freq,60,1)
            #     ang_sparc_60_df.to_csv(ang_savedir+"60s\\"+eventName+"_sparc.csv")
            # except IndexError:
            #     continue 
            # Save SPARC data
            
            
            
            
    


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



