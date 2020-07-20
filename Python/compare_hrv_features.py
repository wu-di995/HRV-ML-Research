# Generate plots to compare how well different features separate labels

# Import necessary libraries 
import pandas as pd
import numpy as np
import glob, pathlib, os, re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import itertools 


cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
HRV_pathsList = glob.glob(str(mainDir)+"\\HRV_multiSubj\\Extracted-with_tlx_labels\\60s\\*.csv")
savedir = str(mainDir)+"\\Plots\\Feature_comparison\\60s\\"

# Print feature names 
HRV_df0 = pd.read_csv(HRV_pathsList[0])
featureNames = HRV_df0.columns[1:-5].values
print("Feature Names")
print(featureNames)

####################### To do 7/20 #################################

# # Combine data by labels and interfaces 
# for i,path in enumerate(HRV_pathsList):
#     HRVdf = pd.read_csv(path)
#     # Instantiate Standard scalar
#     sc = StandardScaler()
#     # Do not include the last 4 columns and the first column 
#     colData = sc.fit_transform(HRVdf.iloc[:,1:-5].values)
#     # Get raw and weighted labels
#     if HRVdf["Raw Label"][0] == "Low":
#         rLabel = 0
#     elif HRVdf["Raw Label"][0] == "Med":
#         rLabel = 1
#     elif HRVdf["Raw Label"][0] == "High":
#         rLabel = 2
#     if HRVdf["Weighted Label"][0] == "Low":
#         wLabel = 0
#     elif HRVdf["Weighted Label"][0] == "Med":
#         wLabel = 1
#     elif HRVdf["Weighted Label"][0] == "High":
#         wLabel = 2
#     # Get interface 
#     if HRVdf["Interface"][0] == "headarray":
#         interface = "ha"
#     elif HRVdf["Interface"][0] == "sip-n-puff":
#         interface = "snp"
#     elif HRVdf["Interface"][0] == "joystick":
#         interface = "joy"
#     # 

# # generates:
# rLow_ha
# rLow_snp
# rLow_joy

##############################################################


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

# Create feature set and labels 
rX = np.vstack((rLow_ar,rMed_ar,rHigh_ar)) # Raw label features
wX = np.vstack((wLow_ar,wMed_ar,wHigh_ar)) # Weighted label features
ry = np.hstack((np.zeros(len(rLow_ar)), np.ones(len(rMed_ar)), np.ones(len(rHigh_ar))*2)) # Raw labels
wy = np.hstack((np.zeros(len(wLow_ar)), np.ones(len(wMed_ar)), np.ones(len(wHigh_ar))*2)) # Weighted labels

# Generate feat1 vs feat2 plots 
# Iterate through all combinations of feature pairs, no pair with 
# Array names: rLow_ar/wLow_ar, rMed_ar/wMed_ar, rHigh_ar/wHigh_ar


combos_gen = itertools.combinations(range(len(featureNames)),2)
combos = [i for i in combos_gen]

for combo in combos:
    feat1_idx = combo[0]
    feat2_idx = combo[1]
    feat1_name = featureNames[feat1_idx]
    feat2_name = featureNames[feat2_idx]
    # Raw score label plot
    fig,ax = plt.subplots()
    ax.scatter(rLow_ar[:,feat1_idx],rLow_ar[:,feat2_idx],c="tab:blue",alpha=0.3)
    ax.scatter(rMed_ar[:,feat1_idx],rMed_ar[:,feat2_idx],c="tab:orange",alpha=0.3)
    ax.scatter(rHigh_ar[:,feat1_idx],rHigh_ar[:,feat2_idx],c="tab:green",alpha=0.3)
    ax.legend(["Low","Med","High"])
    ax.set_xlabel(feat1_name)
    ax.set_ylabel(feat2_name)
    ax.set_title("Raw TLX Score")
    plt.savefig(savedir+"\\Raw\\"+feat1_name+"_"+feat2_name+".png")
    plt.close()
    # Weighted score label plot
    fig,ax = plt.subplots()
    ax.scatter(wLow_ar[:,feat1_idx],wLow_ar[:,feat2_idx],c="tab:blue",alpha=0.3)
    ax.scatter(wMed_ar[:,feat1_idx],wMed_ar[:,feat2_idx],c="tab:orange",alpha=0.3)
    ax.scatter(wHigh_ar[:,feat1_idx],wHigh_ar[:,feat2_idx],c="tab:green",alpha=0.3)
    ax.legend(["Low","Med","High"])
    ax.set_xlabel(feat1_name)
    ax.set_ylabel(feat2_name)
    ax.set_title("Weighted TLX Score")
    plt.savefig(savedir+"\\Weighted\\"+feat1_name+"_"+feat2_name+".png")
    plt.close()

# fig,ax = plt.subplots()

# plt.show()
