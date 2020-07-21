# For every HRV metric in each every, find the min/max/avg, then for every event/subject/type/interface
# plot min/max/avg histograms

# Import necessary libraries 
import pandas as pd
import numpy as np
import os, glob, pathlib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
# Current working directory 
cwd = os.getcwd()
# Parent directory
mainDir = str(pathlib.Path(cwd).parent)
# HRV window length
win = "60s"
# HRV metrics directory
HRV_pathsList = glob.glob(mainDir+"\\HRV_multiSubj\\Extracted-with_tlx_labels\\" + win + "\\*.csv")
# Directory to save plots to 
savedir = mainDir+"\\HRV_multiSubj\\Feature_min_max_mean_by_event\\"
plotsdir = mainDir+"\\Plots\\Feature_min_max_mean_byEvent\\"+ win +"\\"
# Print feature names 
HRV_df0 = pd.read_csv(HRV_pathsList[0])
featureNames = HRV_df0.columns[1:-5].values
# print("Feature Names")
# print(featureNames)
# Events -- Interface_AutonomyLevel
interfaces = ["HA","SNP","JOY"]
autonomyLevels = ["0","1","2"]
events = [ i+"_"+a for i in interfaces for a in autonomyLevels]
eventFeatures = [event+"_"+feature for event in events for feature in featureNames]
# print(eventFeatures)
# Create dataframe to save data 
columns = [eventF+"_"+ m for eventF in eventFeatures for m in ["min","max","mean"]]
columns = [column for column in columns if ("SNP_1" not in column) and ("SNP_2" not in column)]
# print(columns)
subjects = ["s01","s03","s05","u03","u04","u09","u13","u14"]
summary_df = pd.DataFrame(columns=columns,index = subjects)
# Change index to lowercase
summary_df.index = summary_df.index.str.lower()


# Find min/max/average for each column 
for i, path in enumerate(HRV_pathsList):
    HRV_df = pd.read_csv(path)
    # Select the feature columns only 
    subj = path.split("\\")[-1][:3]
    subj = subj.lower()
    event = path.split("\\")[-1][3:-4]
    if event[0] == "h": #headarray
        interface = "HA"
    elif event[0] == "j": #joystick
        interface = "JOY" 
    elif event[0] == "s": # sip and puff
        interface = "SNP"
    if "teleoperation" in path:
        autonomy = "0"
    elif "low level" in path:
        autonomy = "1"
    elif "mid level" in path:
        autonomy = "2"
    # Instantiate Standard scalar
    sc = StandardScaler()
    # Select feature columns only 
    features = sc.fit_transform(HRV_df.iloc[:,1:-5].values)
    featMins = features.min(axis=0)
    featMaxs = features.max(axis=0)
    feataMeans = features.mean(axis=0)
    # Fill in values in summary_df
    for i,(featMin,featMax,featMean) in enumerate(zip(featMins,featMaxs,feataMeans)):
        #interface_autonomyLevel_feature_min/max/mean
        columnMin = interface+"_"+autonomy+"_"+featureNames[i]+"_min"  
        columnMax = interface+"_"+autonomy+"_"+featureNames[i]+"_max"  
        columnMean = interface+"_"+autonomy+"_"+featureNames[i]+"_mean" 
        summary_df.at[subj,columnMin] = featMin
        summary_df.at[subj,columnMax] = featMax
        summary_df.at[subj,columnMean] = featMean

# Export summary_df 
filename = "min-max-mean-byEvent"+"_"+win+".csv"
summary_df.to_csv(savedir+filename)

# Generate bar plots for: --- autonomy level not confirmed 
# 1) each event, comparing across subjects

for col in summary_df:
    # Fig 1 
    fig,ax = plt.subplots()
    ax.bar(subjects,summary_df[col].values)
    ax.set_xlabel("Subjects")
    plt.title(col)
    # plt.show()
    plt.savefig(plotsdir+col+".png")
    plt.close()

# for interface in interfaces:
#     colsMask = summary_df.columns.str.contains(interface)
#     colNames = summary_df.columns[colsMask].values
#     # print(colNames)
#     for idx, row in summary_df.iterrows():
#         fig,ax = plt.subplots()
#         plt.title(interface)
#         xlen = len(row[colsMask])
#         ax.bar(range(xlen),row[colsMask])
#         plt.show()
#     # interface_auto_feat



