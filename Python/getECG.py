# Saves ECG data as a single dataframe for all subjects
import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path
import re

# ECG Folder
ecgDir = "E:\\argall-lab-data\\ECG Data\\"

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
subjsList = [subj for subj in subjsList if "u00" not in subj]
# print(subjsList)

# ECG subject folder paths
ecgSubjsPaths = []
for i,subj in enumerate(subjsList):
    ecgSubjsPaths.append(ecgDir+"laa_wc_multi_session_"+subj+"\\LAA_WC_Multi_Session\\"+subj+"\\")
# print(ecgSubjsPaths)

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

# For all subjects, save combined ECG dataframes 

def saveECG(ecgSubjsPaths):
    for i in range(len(ecgSubjsPaths)):
        subj = ecgSubjsPaths[i].split("\\")[-2]
        print(subj)

        # Sensor names 
        sensors = []
        for (dirpath,dirnames,filenames) in os.walk(ecgSubjsPaths[i]+"ecg_lead_i\\"):
            sensors.extend(dirnames)
            break
        # print(i)
        # print(sensors)

        # Sensor folders
        ssrFolders = []
        for sensor in sensors:
            ssrFolders.append(ecgSubjsPaths[i]+"ecg_lead_i\\"+sensor+"\\")
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

        # Directory to save to 
        savedir = "E:\\argall-lab-data\ECG_combined_bySubj\\"
        subjsavedir = savedir+subjsList[i]+"\\"
        Path(subjsavedir).mkdir(parents=True,exist_ok=True)

        # Save ECG file
        ecgfull_df.to_csv(subjsavedir+"full_ecg.csv")

saveECG(ecgSubjsPaths)
