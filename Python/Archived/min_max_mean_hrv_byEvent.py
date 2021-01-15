# For every HRV metric in each every, find the min/max/avg, then for every event/subject/type/interface
# plot min/max/avg histograms

# Import necessary libraries 
import pandas as pd
import numpy as np
import os, glob, pathlib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter

# Current working directory 
cwd = os.getcwd()
# Parent directory
mainDir = str(pathlib.Path(cwd).parent)
# HRV window length
win = "5s"
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
# Raw and weighted labels 


# Combine data by labels
for i,path in enumerate(HRV_pathsList):
    HRVdf = pd.read_csv(path)
    # Do not inlcude the last 4 columns and first column
    colData = HRVdf.iloc[:,1:-5].values
    minData = colData.min(axis=0)
    maxData = colData.max(axis=0)
    meanData = colData.mean(axis=0)
    # Get raw and weighted labels
    if HRVdf["Raw Label"][0] == "Low":
        rLabel = 0
    elif HRVdf["Raw Label"][0] == "Med":
        rLabel = 1
    elif HRVdf["Raw Label"][0] == "High":
        rLabel = 2
    if HRVdf["Weighted Label"][0] == "Low":
        wLabel = 0
    elif HRVdf["Weighted Label"][0] == "Med":
        wLabel = 1
    elif HRVdf["Weighted Label"][0] == "High":
        wLabel = 2
    # Create np arrarys for raw and weighted labels, low/med/high
    if i == 0:
        # print(rLabel)
        # print(wLabel)
        if rLabel == 0:
            rLowMin_ar = minData 
            rLowMax_ar = maxData 
            rLowMean_ar = meanData 
        if rLabel == 1:
            rMedMin_ar = minData 
            rMedMax_ar = maxData 
            rMedMean_ar = meanData 
        if rLabel == 2:
            rHighMin_ar = minData 
            rHighMax_ar = maxData 
            rHighMean_ar = meanData 
        if wLabel == 0:
            wLowMin_ar = minData 
            wLowMax_ar = maxData 
            wLowMean_ar = meanData 
        if wLabel == 1:
            wMedMin_ar = minData 
            wMedMax_ar = maxData 
            wMedMean_ar = meanData 
        if wLabel == 2:
            wHighMin_ar = minData 
            wHighMax_ar = maxData 
            wHighMean_ar = meanData
    else:
        if rLabel == 0:
            try:
                rLowMin_ar = np.vstack((rLowMin_ar,minData))
                rLowMax_ar = np.vstack((rLowMax_ar,maxData))
                rLowMean_ar = np.vstack((rLowMean_ar,meanData))
            except NameError: #If the array has not been created, create it 
                rLowMin_ar = minData 
                rLowMax_ar = maxData 
                rLowMean_ar = meanData 
        if rLabel == 1:
            try:
                rMedMin_ar = np.vstack((rMedMin_ar,minData))
                rMedMax_ar = np.vstack((rMedMax_ar,maxData))
                rMedMean_ar = np.vstack((rMedMean_ar,meanData)) 
            except NameError:
                rMedMin_ar = minData 
                rMedMax_ar = maxData 
                rMedMean_ar = meanData
        if rLabel == 2:
            try:
                rHighMin_ar = np.vstack((rHighMin_ar,minData))
                rHighMax_ar = np.vstack((rHighMax_ar,maxData))
                rHighMean_ar = np.vstack((rHighMean_ar,meanData)) 
            except NameError:
                rHighMin_ar = minData 
                rHighMax_ar = maxData 
                rHighMean_ar = meanData
        if wLabel == 0:
            try:
                wLowMin_ar = np.vstack((wLowMin_ar,minData))
                wLowMax_ar = np.vstack((wLowMax_ar,maxData))
                wLowMean_ar = np.vstack((wLowMean_ar,meanData))
            except NameError:
                wLowMin_ar = minData 
                wLowMax_ar = maxData 
                wLowMean_ar = meanData 
        if wLabel == 1:
            try:
                wMedMin_ar = np.vstack((wMedMin_ar,minData))
                wMedMax_ar = np.vstack((wMedMax_ar,maxData))
                wMedMean_ar = np.vstack((wMedMean_ar,meanData))
            except NameError:
                wMedMin_ar = minData 
                wMedMax_ar = maxData 
                wMedMean_ar = meanData
        if wLabel == 2:
            try:
                wHighMin_ar = np.vstack((wHighMin_ar,minData))
                wHighMax_ar = np.vstack((wHighMax_ar,maxData))
                wHighMean_ar = np.vstack((wHighMean_ar,meanData)) 
            except NameError:
                wHighMin_ar = minData 
                wHighMax_ar = maxData 
                wHighMean_ar = meanData

 
  
print(rLowMin_ar.shape)
print(rLowMax_ar.shape)
print(rLowMean_ar.shape)

print(rMedMin_ar.shape)
print(rMedMax_ar.shape)
print(rMedMean_ar.shape)

print(rHighMin_ar.shape)
print(rHighMax_ar.shape)
print(rHighMean_ar.shape)

print(wLowMin_ar.shape)
print(wLowMax_ar.shape)
print(wLowMean_ar.shape)

print(wMedMin_ar.shape)
print(wMedMax_ar.shape)
print(wMedMean_ar.shape)

print(wHighMin_ar.shape)
print(wHighMax_ar.shape)
print(wHighMean_ar.shape)

# Plot Histograms 
## For each feature, plot histograms for each feature, colored by different labels
def feature_hist_plotter(low_ar,med_ar,high_ar,scType,featType,featureNames):
    # scType == score type ("Raw" or "Weighted")
    # featType == "min"/"max"/"mean")
    # featureNames list generated earlier 
    # Check that arrays contain same number of features
    if  not (low_ar.shape[1]==med_ar.shape[1]==high_ar.shape[1]):
        return "Arrays have unequal number of columns."
    for i in range(low_ar.shape[1]):
        featName = featureNames[i]+featType
        fig, ax = plt.subplots(1,2,tight_layout=True)
        # Counts
        ax[0].hist(low_ar[:,i],label="Low",color="tab:blue",alpha=0.3)
        ax[0].hist(med_ar[:,i],label="Med",color="tab:orange",alpha=0.3)
        ax[0].hist(high_ar[:,i],label="High",color="tab:green",alpha=0.3)
        ax[0].legend()
        ax[0].set_xlabel("Feature Value")
        ax[0].set_ylabel("Count")
        # Percentage
        ax[1].hist(low_ar[:,i],label="Low",color="tab:blue",alpha=0.3, weights=np.ones(len(low_ar[:,i]))/len(low_ar[:,i]))
        ax[1].hist(med_ar[:,i],label="Med",color="tab:orange",alpha=0.3, weights=np.ones(len(med_ar[:,i]))/len(med_ar[:,i]))
        ax[1].hist(high_ar[:,i],label="High",color="tab:green",alpha=0.3, weights=np.ones(len(high_ar[:,i]))/len(high_ar[:,i]))
        ax[1].legend()
        ax[1].set_title(scType+" Labels: "+featName)
        ax[1].set_xlabel("Feature Value")
        ax[1].set_ylabel("Percentage of respective label")
        ax[1].yaxis.set_major_formatter(PercentFormatter(1))
        plt.title(scType+" Labels: "+featName)
        # Save plot 
        filename = scType+"_"+featName+"_"+"hist_byLabels.png"
        plt.savefig(plotsdir+"\\"+scType+"\\"+featType+"\\"+filename)
        plt.close()

feature_hist_plotter(rLowMin_ar,rMedMin_ar,rHighMin_ar,"Raw", "Min",featureNames)
feature_hist_plotter(rLowMax_ar,rMedMax_ar,rHighMax_ar,"Raw", "Max",featureNames)
feature_hist_plotter(rLowMean_ar,rMedMean_ar,rHighMean_ar,"Raw", "Mean",featureNames)

feature_hist_plotter(wLowMin_ar,wMedMin_ar,wHighMin_ar,"Weighted", "Min",featureNames)
feature_hist_plotter(wLowMax_ar,wMedMax_ar,wHighMax_ar,"Weighted", "Max",featureNames)
feature_hist_plotter(wLowMean_ar,wMedMean_ar,wHighMean_ar,"Weighted", "Mean",featureNames)

# feature_hist_plotter(wLow_ar,wMed_ar,wHigh_ar,"Weighted",featureNames)

# # Find min/max/average for each column 
# for i, path in enumerate(HRV_pathsList):
#     HRV_df = pd.read_csv(path)
#     # Select the feature columns only 
#     subj = path.split("\\")[-1][:3]
#     subj = subj.lower()
#     event = path.split("\\")[-1][3:-4]
#     if event[0] == "h": #headarray
#         interface = "HA"
#     elif event[0] == "j": #joystick
#         interface = "JOY" 
#     elif event[0] == "s": # sip and puff
#         interface = "SNP"
#     if "teleoperation" in path:
#         autonomy = "0"
#     elif "low level" in path:
#         autonomy = "1"
#     elif "mid level" in path:
#         autonomy = "2"
#     # Instantiate Standard scalar
#     sc = StandardScaler()
#     # Select feature columns only 
#     features = sc.fit_transform(HRV_df.iloc[:,1:-5].values)
#     featMins = features.min(axis=0)
#     featMaxs = features.max(axis=0)
#     feataMeans = features.mean(axis=0)
#     # Fill in values in summary_df
#     for i,(featMin,featMax,featMean) in enumerate(zip(featMins,featMaxs,feataMeans)):
#         #interface_autonomyLevel_feature_min/max/mean
#         columnMin = interface+"_"+autonomy+"_"+featureNames[i]+"_min"  
#         columnMax = interface+"_"+autonomy+"_"+featureNames[i]+"_max"  
#         columnMean = interface+"_"+autonomy+"_"+featureNames[i]+"_mean" 
#         summary_df.at[subj,columnMin] = featMin
#         summary_df.at[subj,columnMax] = featMax
#         summary_df.at[subj,columnMean] = featMean

# # Export summary_df 
# filename = "min-max-mean-byEvent"+"_"+win+".csv"
# summary_df.to_csv(savedir+filename)

# Generate bar plots for: --- autonomy level not confirmed 
# 1) each event, comparing across subjects

# for col in summary_df:
#     # Fig 1 
#     fig,ax = plt.subplots()
#     ax.bar(subjects,summary_df[col].values)
#     ax.set_xlabel("Subjects")
#     plt.title(col)
#     # plt.show()
#     plt.savefig(plotsdir+col+".png")
#     plt.close()




