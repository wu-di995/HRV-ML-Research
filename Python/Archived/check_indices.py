from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score, r2_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression,mutual_info_regression
import pandas as pd
import numpy as np
import glob,os, pathlib
import matplotlib.pyplot as plt 
from sklearn import svm 

# userCtrl_df = pd.read_csv("E:\\argall-lab-data\\UserControlled_byEventNEW\\s01_SNP_A0_aft_30.csv")
# target_df = pd.read_csv("E:\\argall-lab-data\\UserCmdFreq_byEvent\\s01_SNP_A0_aft_30.csv")
# hrv_df = pd.read_csv("E:\\argall-lab-data\\HRV_byEventNEW\\30s\\s01_SNP_A0_HRV_results_allwindows_20200826.csv")

# targetIdx_all = target_df.iloc[:,0].to_list()
# userCtrl_idx = userCtrl_df[(userCtrl_df["User Controlled"] == 1)].iloc[:,0].to_list()
# hrv_existIdx = hrv_df[~(hrv_df["NNmean"].isnull())].index.to_list()
# featuresIdx = list(set(userCtrl_idx) & set(hrv_existIdx))
# featuresIdx2 = list(set(featuresIdx) & set(targetIdx_all))
# targetIdx = target_df[(target_df.iloc[:,0].isin(featuresIdx))].index.to_list()

# print(len(featuresIdx2))
# print(len(targetIdx))

# paths = glob.glob("E:\\argall-lab-data\\UserCmdFreq_byEvent\\*aft_30.csv")
# for path in paths:
#     df = pd.read_csv(path)
#     if (df["Moving Average"].isnull().sum()):
#         print(path)

# Feature files - SPARC (velocity, user command), user command frequencies 
## SPARC velocity 
### 30s 
sparcVel30_aft_Paths = glob.glob("E:\\argall-lab-data\\SPARC_vel_byEvent\\*aft_30.csv")
sparcVel30_close_Paths = glob.glob("E:\\argall-lab-data\\SPARC_vel_byEvent\\*close_30.csv")
### 60s 
sparcVel60_aft_Paths = glob.glob("E:\\argall-lab-data\\SPARC_vel_byEvent\\*aft_60.csv")
sparcVel60_close_Paths = glob.glob("E:\\argall-lab-data\\SPARC_vel_byEvent\\*close_60.csv")

## SPARC User command 
### 30s
sparcUserCmd30_aft_Paths = glob.glob("E:\\argall-lab-data\\SPARC_userCmd_byEventNEW\\*aft_30.csv")
sparcUserCmd30_close_Paths = glob.glob("E:\\argall-lab-data\\SPARC_userCmd_byEventNEW\\*close_30.csv")
### 60s
sparcUserCmd60_aft_Paths = glob.glob("E:\\argall-lab-data\\SPARC_userCmd_byEventNEW\\*aft_60.csv")
sparcUserCmd60_close_Paths = glob.glob("E:\\argall-lab-data\\SPARC_userCmd_byEventNEW\\*close_60.csv")

## User Command Frequencies 
### 30s
userCmdFreqs30_aft_Paths = glob.glob("E:\\argall-lab-data\\UserCmdFreq_byEvent\\*aft_30.csv")
userCmdFreqs30_close_Paths = glob.glob("E:\\argall-lab-data\\UserCmdFreq_byEvent\\*close_30.csv")
### 60s
userCmdFreqs60_aft_Paths = glob.glob("E:\\argall-lab-data\\UserCmdFreq_byEvent\\*aft_60.csv")
userCmdFreqs60_close_Paths = glob.glob("E:\\argall-lab-data\\UserCmdFreq_byEvent\\*close_60.csv")

## User Controlled status 
### 30s
userCtrl30_aft_Paths = glob.glob("E:\\argall-lab-data\\UserControlled_byEventNEW\\*aft_30.csv")
userCtrl30_close_Paths = glob.glob("E:\\argall-lab-data\\UserControlled_byEventNEW\\*close_30.csv")
##s# 60s
userCtrl60_aft_Paths = glob.glob("E:\\argall-lab-data\\UserControlled_byEventNEW\\*aft_60.csv")
userCtrl60_close_Paths = glob.glob("E:\\argall-lab-data\\UserControlled_byEventNEW\\*close_60.csv")

# HRV files 
hrv30Paths = glob.glob("E:\\argall-lab-data\\HRV_byEventNEW\\30s\\*.csv")
hrv60Paths = glob.glob("E:\\argall-lab-data\\HRV_byEventNEW\\60s\\*.csv")

# Read event name from a path
def readEvent(path):
    filenameList = path.split("\\")[-1].split("_")
    event = filenameList[0].lower()+"_"+filenameList[1]+"_"+filenameList[2]
    return event

def check_lengths(targetPaths, userCtrlPaths, HRVPaths):
    for i,(targetPath,userCtrlPath,HRVPath) in enumerate(zip(targetPaths, userCtrlPaths, HRVPaths)):
        event = readEvent(targetPath)
        # Load target dataframe
        target_df = pd.read_csv(targetPath)
        # Load HRV dataframe
        hrv_path = [path for path in HRVPaths if event in path][0]
        hrv_df = pd.read_csv(hrv_path)
        # Load user controlled dataframe
        userCtrl_path = [path for path in userCtrlPaths if event in path][0]
        userCtrl_df = pd.read_csv(userCtrl_path)
        print(hrv_path)
        print(target_df.shape,userCtrl_df.shape,hrv_df.shape)

check_lengths(sparcVel30_aft_Paths, userCtrl30_aft_Paths, hrv30Paths)