# Calculate the mean total power for each task
import pandas as pd
import numpy as np
import glob,os, pathlib
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
import matplotlib.patches as mpatches 

# User command frequency files 
userCmdFreqs30_close_Paths = glob.glob("E:\\argall-lab-data\\UserCmdFreq_byEvent\\*close_30.csv")
userCmdFreqs60_close_Paths = glob.glob("E:\\argall-lab-data\\UserCmdFreq_byEvent\\*close_60.csv")
# Tasks for windows
tasks30_Paths = glob.glob("E:\\argall-lab-data\\totalPower_taskWindows\\*30.csv")
tasks60_Paths = glob.glob("E:\\argall-lab-data\\totalPower_taskWindows\\*60.csv")
# Savedir
savedirMean = "E:\\argall-lab-data\\meanTotalPower_byTask\\"
savedirVar = "E:\\argall-lab-data\\varTotalPower_byTask\\"

# Read event name from a path
def readEvent(path):
    filenameList = path.split("\\")[-1].split("_")
    event = filenameList[0].lower()+"_"+filenameList[1]+"_"+filenameList[2]
    return event

# Calcuate mean total power for each task in an event
def meanTP(eventTP_Path, taskPath):
    # Load datframes
    eventTP_df = pd.read_csv(eventTP_Path)
    task_df = pd.read_csv(taskPath)
    # Mean TP list 
    meanTPList = [0]*7
    for i in range(1,7+1): # 7 tasks in total
        taskTP = eventTP_df[task_df["Main Task"]==i]["Total Power"].values
        meanTP = np.mean(taskTP)
        meanTPList[i-1] = meanTP
    return meanTPList

# Calcuate variance of total power for each task in an event
def varTP(eventTP_Path, taskPath):
    # Load datframes
    eventTP_df = pd.read_csv(eventTP_Path)
    task_df = pd.read_csv(taskPath)
    # Mean TP list 
    varTPList = [0]*7
    for i in range(1,7+1): # 7 tasks in total
        taskTP = eventTP_df[task_df["Main Task"]==i]["Total Power"].values
        varTP = np.var(taskTP)
        varTPList[i-1] = varTP
    return varTPList

# Create dataframe for each interface 
def mk_interface_dfs(allPaths,columns):
    HA_paths = [path for path in allPaths if "HA" in path]
    JOY_paths = [path for path in allPaths if "JOY" in path]
    SNP_paths = [path for path in allPaths if "SNP" in path]
    HA_df = pd.DataFrame(index=range(len(HA_paths)),columns=columns)
    JOY_df = pd.DataFrame(index=range(len(JOY_paths)),columns=columns)
    SNP_df = pd.DataFrame(index=range(len(SNP_paths)),columns=columns)
    return HA_df, JOY_df, SNP_df

# Fill in dataframe with mean/var TP
def fill_df(TP_df,eventTP_Paths, taskPaths, TP_function):
    for i,tpPath in enumerate(eventTP_Paths):
        event = readEvent(tpPath)
        taskPath = [path for path in taskPaths if event in path][0]
        TP_df.loc[i,"Event"] = event
        TP_df.iloc[i,1:] = TP_function(tpPath,taskPath)
    return TP_df


if __name__ == "__main__":
    # Create dataframe to save mean values for each interface 
    columns = ["Event"]+["Task {}".format(i) for i in range(1,7+1)]
    HA_30_df, JOY_30_df, SNP_30_df = mk_interface_dfs(userCmdFreqs30_close_Paths,columns)
    HA_60_df, JOY_60_df, SNP_60_df = mk_interface_dfs(userCmdFreqs60_close_Paths,columns)
    # 30s
    HA_30_Paths = [path for path in userCmdFreqs30_close_Paths if "HA" in path]
    JOY_30_Paths = [path for path in userCmdFreqs30_close_Paths if "JOY" in path]
    SNP_30_Paths = [path for path in userCmdFreqs30_close_Paths if "SNP" in path]
    # 60s
    HA_60_Paths = [path for path in userCmdFreqs60_close_Paths if "HA" in path]
    JOY_60_Paths = [path for path in userCmdFreqs60_close_Paths if "JOY" in path]
    SNP_60_Paths = [path for path in userCmdFreqs60_close_Paths if "SNP" in path]

    # Fill dataframes
    HA_30_df = fill_df(HA_30_df,HA_30_Paths,tasks30_Paths,varTP)
    JOY_30_df = fill_df(JOY_30_df,JOY_30_Paths,tasks30_Paths,varTP)
    SNP_30_df = fill_df(SNP_30_df,SNP_30_Paths,tasks30_Paths,varTP)
    HA_60_df = fill_df(HA_60_df,HA_60_Paths,tasks60_Paths,varTP)
    JOY_60_df = fill_df(JOY_60_df,JOY_60_Paths,tasks60_Paths,varTP)
    SNP_60_df = fill_df(SNP_60_df,SNP_60_Paths,tasks60_Paths,varTP)
    # Save dataframes
    HA_30_df.to_csv(savedirVar+"HA_30.csv")
    JOY_30_df.to_csv(savedirVar+"JOY_30.csv")
    SNP_30_df.to_csv(savedirVar+"SNP_30.csv")
    HA_60_df.to_csv(savedirVar+"HA_60.csv")
    JOY_60_df.to_csv(savedirVar+"JOY_60.csv")
    SNP_60_df.to_csv(savedirVar+"SNP_60.csv")