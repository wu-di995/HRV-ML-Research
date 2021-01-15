# Get feature importances using a few methods 
import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path
from sklearn import svm 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import loguniform
from xgboost import XGBClassifier
import re
import matplotlib.pyplot as plt 

cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
# Have to manually change for 5s/10s/30s/60s
win = "60s"

# Change to match local 
HRV_extr_dir = "/home/skrdown/Documents/argall-lab-data/HRV_newgen/Extracted_with_tlx_labels/"
HRV_2_pathsList = glob.glob(HRV_extr_dir+win+os.sep+"2"+os.sep+"*.csv")
HRV_3_pathsList = glob.glob(HRV_extr_dir+win+os.sep+"3"+os.sep+"*.csv")
savedir = "/home/skrdown/Documents/argall-lab-data/HRV_newgen/Plots/"
# Print feature names 
HRV_df0 = pd.read_csv(HRV_2_pathsList[0])
featureNames = HRV_df0.columns[1:-5].values
print("Feature Names")
print(featureNames)

# Group data by labels to create datasets, for 2 TLX levels
def mk_2_dataset(HRV_pathsList):
    for i,path in enumerate(HRV_pathsList):
        HRVdf = pd.read_csv(path)
        # Do not include the last 4 columns and first column
        colData = HRVdf.iloc[:,1:-5].values
        # Get raw and weighted labels
        if HRVdf["Raw Label"][0] == "Low":
            rLabel = 0
        elif HRVdf["Raw Label"][0] == "High":
            rLabel = 1
        if HRVdf["Weighted Label"][0] == "Low":
            wLabel = 0
        elif HRVdf["Weighted Label"][0] == "High":
            wLabel = 1
        # Create np arrarys for raw and weighted labels, low/high
        if i == 0:
            # print(rLabel)
            # print(wLabel)
            if rLabel == 0:
                rLow_ar = colData 
            if rLabel == 1:
                rHigh_ar = colData
            if wLabel == 0:
                wLow_ar = colData
            if wLabel == 1:
                wHigh_ar = colData
        else:
            if rLabel == 0:
                try: 
                    rLow_ar = np.vstack((rLow_ar,colData)) # Do not include the last 4 columns and first column
                except NameError: #If the array has not been created, create it 
                    rLow_ar = colData
            if rLabel == 1:
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
                    wHigh_ar = np.vstack((wHigh_ar,colData))
                except NameError:
                    wHigh_ar = colData
    # Raw label arrays
    print("Size of Raw TLX feature arrays (Low/High)")
    rLowlen = rLow_ar.shape[0]
    rHighlen = rHigh_ar.shape[0]
    print(rLow_ar.shape)
    print(rHigh_ar.shape)
    # Weighted label arrays
    wLowlen = wLow_ar.shape[0]
    wHighlen = rHigh_ar.shape[0]
    print("Size of Weighted TLX feature arrays (Low/High)")
    print(wLow_ar.shape)
    print(wHigh_ar.shape)

    # Create feature set and labels 
    rX = np.vstack((rLow_ar,rHigh_ar)) # Raw label features
    wX = np.vstack((wLow_ar,wHigh_ar)) # Weighted label features
    ry = np.hstack((np.zeros(len(rLow_ar)), np.ones(len(rHigh_ar)))) # Raw labels
    wy = np.hstack((np.zeros(len(wLow_ar)), np.ones(len(wHigh_ar)))) # Weighted labels

    return rX,wX,ry,wy

# Group data by labels to create datasets, for 3 TLX levels
def mk_3_dataset(HRV_pathsList):
    for i,path in enumerate(HRV_pathsList):
        HRVdf = pd.read_csv(path)
        # Do not include the last 4 columns and first column
        colData = HRVdf.iloc[:,1:-5].values
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
                    rLow_ar = np.vstack((rLow_ar,colData)) # Do not include the last 4 columns and first column
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
    rLowlen = rLow_ar.shape[0]
    rMedlen = rMed_ar.shape[0]
    rHighlen = rHigh_ar.shape[0]
    print(rLow_ar.shape)
    print(rMed_ar.shape)
    print(rHigh_ar.shape)
    # Weighted label arrays
    wLowlen = wLow_ar.shape[0]
    wMedlen = wMed_ar.shape[0]
    wHighlen = rHigh_ar.shape[0]
    print("Size of Weighted TLX feature arrays (Low/Med/High)")
    print(wLow_ar.shape)
    print(wMed_ar.shape)
    print(wHigh_ar.shape)

    # Create feature set and labels 
    rX = np.vstack((rLow_ar,rMed_ar,rHigh_ar)) # Raw label features
    wX = np.vstack((wLow_ar,wMed_ar,wHigh_ar)) # Weighted label features
    ry = np.hstack((np.zeros(len(rLow_ar)), np.ones(len(rMed_ar)), np.ones(len(rHigh_ar))*2)) # Raw labels
    wy = np.hstack((np.zeros(len(wLow_ar)), np.ones(len(wMed_ar)), np.ones(len(wHigh_ar))*2)) # Weighted labels

    return rX,wX,ry,wy

# Create datasets
rX_2,wX_2,ry_2,wy_2 = mk_2_dataset(HRV_2_pathsList)
rX_3,wX_3,ry_3,wy_3 = mk_3_dataset(HRV_3_pathsList)

# Standardize 
# Instantiate Standard scalar
sc = StandardScaler()
# Standardize all features on full datasets
rX_2 = sc.fit_transform(rX_2)
wX_2 = sc.fit_transform(wX_2)
rX_3 = sc.fit_transform(rX_3)
wX_3 = sc.fit_transform(wX_3)

# Split into training and test sets 
sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
def apply_sss(X,y):
    for train_index, test_index in sss.split(X,y):
            X_train, X_test = X[train_index],X[test_index]
            y_train, y_test = y[train_index],y[test_index]
    return X_train,X_test,y_train,y_test

rX_train_2,rX_test_2,ry_train_2,ry_test_2 = apply_sss(rX_2,ry_2)
wX_train_2,wX_test_2,wy_train_2,wy_test_2 = apply_sss(wX_2,wy_2)

rX_train_3,rX_test_3,ry_train_3,ry_test_3 = apply_sss(rX_3,ry_3)
wX_train_3,wX_test_3,wy_train_3,wy_test_3 = apply_sss(wX_3,wy_3)

def feat_impt_multiModels(X,y,scType,featureNames, win, TLX_levels):
    # scType == score type ("Raw" or "Weighted")
    # featureNames list generated earlier 
    featureScores = [0]*len(featureNames)
    featureRanks = [0]*len(featureNames)
    featureRanks_dict = dict(zip(featureNames, featureRanks))
    models = [ExtraTreesClassifier(),DecisionTreeClassifier(),XGBClassifier(label_encoder=False)]
    modelNames = ["ExtraTreesClassifier","DecisionTreeClassifer","XGBClassifier"]
    sub_savedir = savedir+win+os.sep+TLX_levels+os.sep
    Path(sub_savedir).mkdir(parents=True,exist_ok=True)
    for i,model in enumerate(models):
        model.fit(X,y)
        importances = model.feature_importances_
        # Sum up scores for feature importances
        # featureScores = [featureScore + importance for featureScore,importance in zip(featureScores,importances)]
        # Sum up ranks 
        importances,ranked_names = zip(*sorted(zip(importances,featureNames))) 
        for rank, ranked_name in enumerate(ranked_names):
            featureRanks_dict[ranked_name] += rank
            # Higher number is more important 
        # Plotting
        # importances,names = zip(*sorted(zip(importances,featureNames)))
        # plt.barh(range(len(names)), importances, align='center')
        # plt.yticks(range(len(names)), names)
        # plt.title(scType+" Labels: "+modelNames[i]+" Feature Importances")
        # filename = scType+"_"+modelNames[i]+"_featImpt.png"
        # plt.savefig(sub_savedir+filename)
        # plt.close()

    # Feature Scores
    # featureScores,names = zip(*sorted(zip(featureScores,featureNames))) #Ascending order
    # print(list(names))
    # print(list(featureScores))

    # Feature Ranks
    sorted_ranks_dict = dict(sorted(featureRanks_dict.items(), key = lambda item: item[1])) # Sorts features in ascending order 
    print(sorted_ranks_dict)
    print(list(sorted_ranks_dict.keys()))# Get the feature names in ascending order
    # print(sorted(featureRanks_dict.keys(), key = featureRanks_dict.get)) 
# Get feature importances 
print("Raw, 2")
feat_impt_multiModels(rX_train_2,ry_train_2,"Raw",featureNames, win, "2")
print("Weighted, 2")
feat_impt_multiModels(wX_train_2,wy_train_2,"Weighted",featureNames, win, "2")

print("Raw, 3")
feat_impt_multiModels(rX_train_3,ry_train_3,"Raw",featureNames, win, "3")
print("Weighted, 3")
feat_impt_multiModels(wX_train_3,wy_train_3,"Weighted",featureNames, win, "3")

"""

# Change low/med/high arrays to standardized features
## Raw Labels
rLow_ar = rX[:rLowlen]
rMed_ar = rX[rLowlen:rLowlen+rMedlen]
rHigh_ar = rX[rLowlen+rMedlen:]
## Weighted Labels
wLow_ar = wX[:wLowlen]
wMed_ar = wX[wLowlen:wLowlen+wMedlen]
wHigh_ar = wX[wLowlen+wMedlen:]

# Raw label arrays
print("Size of Raw TLX standardized feature arrays (Low/Med/High)")
print(rLow_ar.shape)
print(rMed_ar.shape)
print(rHigh_ar.shape)
# Weighted label arrays
print("Size of Weighted TLX standardized feature arrays (Low/Med/High)")
print(wLow_ar.shape)
print(wMed_ar.shape)
print(wHigh_ar.shape)
"""