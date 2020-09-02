# Plot total power windows, color background based on the type of tasks the window is in
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

# Supstatus -- select power windows were user controls 
userCtrl30_close_Paths = glob.glob("E:\\argall-lab-data\\UserControlled_byEventNEW\\*close_30.csv")
userCtrl60_close_Paths = glob.glob("E:\\argall-lab-data\\UserControlled_byEventNEW\\*close_60.csv")

# Mean total power files
meanTP_30 = glob.glob("E:\\argall-lab-data\\meanTotalPower_byTask\\*_30.csv")
meanTP_60 = glob.glob("E:\\argall-lab-data\\meanTotalPower_byTask\\*_60.csv")

# Read event name from a path
def readEvent(path):
    filenameList = path.split("\\")[-1].split("_")
    event = filenameList[0].lower()+"_"+filenameList[1]+"_"+filenameList[2]
    return event

# Function to plot total power, for a single event, highlighted by events
def plt_TP(userCmdFreqPath, taskPath, userCtrlPath, event, time, savedir):
    # Load userCtrl dataframe 
    userCtrl_df = pd.read_csv(userCtrlPath)
    # User controlled indices
    notUserCtrl_idx = userCtrl_df[(userCtrl_df["User Controlled"] == 0)].iloc[:,0].to_list()
    # Load userCmdFreq dataframe
    userCmd_df = pd.read_csv(userCmdFreqPath)
    # Load task window df
    taskWin_df = pd.read_csv(taskPath)
    # Convert to taskWin indices
    notUserCtrl_task_idx =  taskWin_df[taskWin_df.iloc[:,0].isin(notUserCtrl_idx)].index.to_list()
    print(userCtrl_df.shape)
    print(taskWin_df.shape)
    taskWin_df.loc[notUserCtrl_task_idx,"Main Task"] = "Sup Ctrl"
    
    # Total power values
    totalPower = userCmd_df.iloc[:,-1].values
    # Task values
    tasks = taskWin_df.loc[:,"Main Task"]
    # Find peaks 
    # pks,_ = find_peaks(totalPower, distance=5)
    pks,_ = find_peaks(totalPower)
    # Plotting
    fig,ax = plt.subplots()
    ax.set_title(event)
    ax.plot(totalPower)
    ax.plot(pks,totalPower[pks],"x")
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink", "w"]
    texts = ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Sup Ctrl"]
    patches = [mpatches.Patch(color=colors[i],label="{:s}".format(texts[i])) for i in range(len(texts))]
    ax.legend(handles=patches)
    for i,task in enumerate(tasks):
        if task == "Sup Ctrl":
            ax.axvspan(i,i+1,facecolor=colors[-1],alpha=0.5)
        else:
            ax.axvspan(i,i+1,facecolor=colors[task-1],alpha=0.5)
    # plt.show()
    plt.savefig(savedir+event+"_"+time+".png")
    plt.close()
tpPath = userCmdFreqs30_close_Paths[0]
taskPath = tasks30_Paths[0]

# Plot meanTP
def plt_meanTP(meanTP_Path,savedir,interface):
    meanTP_df = pd.read_csv(meanTP_Path)
    # Plotting
    fig,ax = plt.subplots()
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
    for i,row in meanTP_df.iterrows():
        for j in range(len(colors)):
            ax.scatter(j+1,row[2+j],color=colors[j])
    ax.set_title("Mean Total Power "+interface)
    plt.savefig(savedir+interface+".png")
    plt.close()


if __name__ == "__main__":
    # Savedir
    # savedir = "E:\\argall-lab-data\\totalPower_plots\\"
    # HA_30 = [path for path in meanTP_30 if "HA" in path][0]
    # HA_60 = [path for path in meanTP_60 if "HA" in path][0]
    # JOY_30 = [path for path in meanTP_30 if "JOY" in path][0]
    # JOY_60 = [path for path in meanTP_60 if "JOY" in path][0]
    # SNP_30 = [path for path in meanTP_30 if "SNP" in path][0]
    # SNP_60 = [path for path in meanTP_60 if "SNP" in path][0]
    # plt_meanTP(HA_30,savedir,"HA_30")
    # plt_meanTP(HA_60,savedir,"HA_60")
    # plt_meanTP(JOY_30,savedir,"JOY_30")
    # plt_meanTP(SNP_30,savedir,"JOY_60")
    # plt_meanTP(HA_30,savedir,"SNP_30")
    # plt_meanTP(SNP_60,savedir,"SNP_60")


    savedir = "E:\\argall-lab-data\\totalPower_plots\\"
    for tpPath in userCmdFreqs30_close_Paths:
        event = readEvent(tpPath)
        time = "30"
        taskPath = [path for path in tasks30_Paths if event in path][0]
        userCtrlPath = [path for path in userCtrl30_close_Paths if event in path][0]
        print(taskPath,userCtrlPath)
        plt_TP(tpPath,taskPath,userCtrlPath,event,time,savedir)
    for tpPath in userCmdFreqs60_close_Paths:
        event = readEvent(tpPath)
        time = "60"
        taskPath = [path for path in tasks60_Paths if event in path][0]
        userCtrlPath = [path for path in userCtrl60_close_Paths if event in path][0]
        plt_TP(tpPath,taskPath,userCtrlPath,event,time,savedir)