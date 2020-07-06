# For a given subject, plot and save the raw ECG by activity segments 

# Imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os 
from pathlib import Path 

# Helper functions

# Combines ecg files to one df
def get_combinedECG(ecgFolders):
    ecgFiles = []
    for folder in ecgFolders:
        ecgFiles.append(folder+"elec.csv")
    # print(ecgFiles)
    if len(ecgFiles)==1:
        ecgfull_df = pd.read_csv(ecgFiles[0])
    elif len(ecgFiles)==2:
        ecg1_df = pd.read_csv(ecgFiles[0])
        ecg2_df = pd.read_csv(ecgFiles[1])
        ecgfull_df = pd.concat([ecg1_df,ecg2_df])
    elif len(ecgFiles)==3:
        ecg1_df = pd.read_csv(ecgFiles[0])
        ecg2_df = pd.read_csv(ecgFiles[1])
        ecg3_df = pd.read_csv(ecgFiles[2])
        ecgfull_df = pd.concat([ecg1_df,ecg2_df,ecg3_df])
    return ecgfull_df

# Function to find index of closest lower neighbour of a timestamp
def find_closest(df,timestamp):
    exactmatch = (df[df.columns[0]]==timestamp)
    while (exactmatch[exactmatch==True].empty):
        timestamp -=1
        exactmatch = (df[df.columns[0]]==timestamp) 
    return (exactmatch[exactmatch==True].index[0])

# Data folder path
dataPath = "F:\\argall-lab-data\\ECG Data\\"

# Subjects 
subjsList = []
for i in range(8+1):
    if i<10:
        subjsList.append("s0"+str(i))
    else:
        subjsList.append("s"+str(i))
for i in range(10):
    if i<10:
        subjsList.append("u0"+str(i))
    else:
        subjsList.append("u"+str(i))
for i in range(11,14+1):
    if i<10:
        subjsList.append("u0"+str(i))
    else:
        subjsList.append("u"+str(i))
# print(subjsList)

# Subject folder paths
subjsPaths = []
for i,subj in enumerate(subjsList):
    subjsPaths.append(dataPath+"laa_wc_multi_session_"+subj+"\\LAA_WC_Multi_Session\\"+subj+"\\")

# print(subjsPaths)

# Annotations file
annotPaths = []
for i,path in enumerate(subjsPaths):
    annotPaths.append(path+"annotations.csv")

# print(annotPaths)

# Loop through all subjects 

for i in range(len(subjsList)):
    # Read annotations.csv
    annot_df = pd.read_csv(annotPaths[i])
    # print(annotPaths)

    # Sensor names 
    sensors = []
    for (dirpath,dirnames,filenames) in os.walk(subjsPaths[i]+"ecg_lead_i\\"):
        sensors.extend(dirnames)
        break
    # print(i)
    # print(sensors)

    # Sensor folders
    ssrFolders = []
    for sensor in sensors:
        ssrFolders.append(subjsPaths[i]+"ecg_lead_i\\"+sensor+"\\")
    # print(i)
    # print(len(ssrFolders))
    # print("----")

    # ECG folders 
    ecgFolders= []
    for ssrFolder in ssrFolders:
        seshFolders = os.listdir(ssrFolder)
        for folder in seshFolders:
            ecgFolders.append(ssrFolder+folder+"\\")

    # Get combined ECG file
    ecgfull_df = get_combinedECG(ecgFolders)
    # print(i)
    # print(ecgfull_df.shape)

    # For each event (annot_df), get start/stop times, retrieve ECG segments, plot (mV) and save them (mV)

    # Directory to save to 
    savedir = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\"
    subjsavedir = savedir+"RawECG\\"+subjsList[i]+"\\"
    Path(subjsavedir).mkdir(parents=True,exist_ok=True)

    for i in annot_df.index:
        event = annot_df.loc[i]["EventType"]
        # Find indices 
        startTime = annot_df.loc[i]["Start Timestamp (ms)"]
        endTime = annot_df.loc[i]["Stop Timestamp (ms)"]
        # startTime = annot_df.loc[(annot_df["EventType"]==event)]["Start Timestamp (ms)"].values[0]
        # endTime = annot_df.loc[(annot_df["EventType"]==event)]["Stop Timestamp (ms)"].values[0]    
        startIndex = find_closest(ecgfull_df,startTime)
        endIndex = find_closest(ecgfull_df,endTime)
        # Create new df
        new_df = ecgfull_df.iloc[startIndex:endIndex+1,1]
        # Convert to mV
        new_df = new_df.apply(lambda x:x*1000)
        # Save mV values
        newPath = subjsavedir+str(event)+".csv"
        newFile = Path(newPath)
        num = 1
        # Check if file already exists
        while newFile.is_file():
            newPath = subjsavedir+str(event)+str(num)+".csv"
            newFile = Path(newPath)
            num += 1
        new_df.to_csv(newPath,index=False,header=False) #do not save the index values 
        # Create Plots 
        plt.plot(new_df.values)
        plt.title(event)
        plt.ylabel("mV")
        plt.savefig(newPath.rstrip("csv")+"png")
        plt.close()






