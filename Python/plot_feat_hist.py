# Plot histograms for each feature
import pandas as pd
import os,glob,pathlib 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib.ticker import PercentFormatter


cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
HRV_pathsList = glob.glob(str(mainDir)+"\\HRV_multiSubj\\Extracted-with_tlx_labels\\5s\\*.csv")
savedir = str(mainDir)+"\\Plots\\Feature_histograms\\5s\\"

# Print feature names 
HRV_df0 = pd.read_csv(HRV_pathsList[0])
featureNames = HRV_df0.columns[1:-5].values
print("Feature Names")
print(featureNames)

# Combine data by labels 
for i,path in enumerate(HRV_pathsList):
    HRVdf = pd.read_csv(path)
    # Instantiate Standard scalar
    sc = StandardScaler()
    # Do not inlcude the last 4 columns and first column
    colData = sc.fit_transform(HRVdf.iloc[:,1:-5].values)
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
            rLow_ar = colData 
        if rLabel == 1:
            rMed_ar = colData
        if rLabel == 2:
            rHigh_ar = colData
        if wLabel == 0:
            wLow_ar = colData
        if wLabel == 1:
            wMed_ar = colData
        if wLabel == 2:
            wHigh_ar = colData
    else:
        if rLabel == 0:
            try: 
                rLow_ar = np.vstack((rLow_ar,colData)) # Do not inlcude the last 4 columns and first column
            except NameError: #If the array has not been created, create it 
                rLow_ar = colData
        if rLabel == 1:
            try: 
                rMed_ar = np.vstack((rMed_ar,colData))
            except NameError:
                rMed_ar = colData
        if rLabel == 2:
            try:
                rHigh_ar = np.vstack((rHigh_ar,colData))
            except NameError:
                rHigh_ar = colData
        if wLabel == 0:
            try:
                wLow_ar = np.vstack((wLow_ar,colData)) 
            except NameError:
                wLow_ar = colData
        if wLabel == 1:
            try:
                wMed_ar = np.vstack((wMed_ar,colData))
            except NameError:
                wMed_ar = colData
        if wLabel == 2:
            try:
                wHigh_ar = np.vstack((wHigh_ar,colData))
            except NameError:
                wHigh_ar = colData
# Raw label arrays
print("Size of Raw TLX feature arrays (Low/Med/High)")
print(rLow_ar.shape)
print(rMed_ar.shape)
print(rHigh_ar.shape)
# Weighted label arrays
print("Size of Weighted TLX feature arrays (Low/Med/High)")
print(wLow_ar.shape)
print(wMed_ar.shape)
print(wHigh_ar.shape)

# Plot Histograms 
## For each feature, plot histograms for each feature, colored by different labels
def feature_hist_plotter(low_ar,med_ar,high_ar,scType,featureNames):
    # scType == score type ("Raw" or "Weighted")
    # featureNames list generated earlier 
    # Check that arrays contain same number of features
    if  not (low_ar.shape[1]==med_ar.shape[1]==high_ar.shape[1]):
        return "Arrays have unequal number of columns."
    for i in range(low_ar.shape[1]):
        featName = featureNames[i]
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
        plt.savefig(savedir+filename)
        plt.close()

feature_hist_plotter(rLow_ar,rMed_ar,rHigh_ar,"Raw",featureNames)
feature_hist_plotter(wLow_ar,wMed_ar,wHigh_ar,"Weighted",featureNames)