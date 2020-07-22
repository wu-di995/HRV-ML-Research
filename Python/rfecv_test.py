# Test feature selection with RFECV

# Import necessary libraries 
import pandas as pd
import numpy as np
import glob, pathlib, os, re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import itertools 
from sklearn.feature_selection import RFE, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score



cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
HRV_pathsList = glob.glob(str(mainDir)+"\\HRV_multiSubj\\Extracted-with_tlx_labels\\60s\\*.csv")

# Print feature names 
HRV_df0 = pd.read_csv(HRV_pathsList[0])
featureNames = HRV_df0.columns[1:-5].values
print("Feature Names")
print(featureNames)


# Combine data by labels
for i,path in enumerate(HRV_pathsList):
    HRVdf = pd.read_csv(path)
    # Do not inlcude the last 4 columns and first column
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

# Instantiate Standard scalar
sc = StandardScaler()
# Standardize all features on full datasets
rX = sc.fit_transform(rX)
wX = sc.fit_transform(wX)
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

# Split into training and test sets 
sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
def apply_sss(X,y):
    for train_index, test_index in sss.split(X,y):
            X_train, X_test = X[train_index],X[test_index]
            y_train, y_test = y[train_index],y[test_index]
    return X_train,X_test,y_train,y_test

rX_train,rX_test,ry_train,ry_test = apply_sss(rX,ry)
wX_train,wX_test,wy_train,wy_test = apply_sss(wX,wy)

# RFE
svc = SVC(kernel="linear")
rfe = RFE(estimator=svc,n_features_to_select=5)

rfe.fit(wX_train,wy_train)
ranking = rfe.ranking_

print("Ranking:")
print(ranking)
print([feature for i, feature in enumerate(featureNames) if ranking[i]==1])

# Using only the selected features 
feat_cols = [idx for idx, rank in enumerate(ranking) if rank==1]

wX_train_sel = wX_train[:,feat_cols]
wX_test_sel = wX_test[:,feat_cols]
svc.fit(wX_train_sel,wy_train)
wy_pred = svc.predict(wX_test_sel)
wval_acc = accuracy_score(wy_test,wy_pred)
print(wval_acc)

# Raw   
# 5s : 0.4679765246236285 Ranking: [1 1 6 3 4 1 2 1 1 7 5], Features: ['SDNN', 'RMSSD', 'hf', 'SD1', 'SD2']
# 10s: 0.4795690936106984 Ranking: [1 2 1 1 5 4 3 1 1 7 6], Features: ['SDNN', 'ulf', 'vlf', 'SD1', 'SD2']
# 30s: 0.540734109221128 Ranking: [1 1 6 5 1 3 4 1 1 2 7], Features: ['SDNN', 'RMSSD', 'lf', 'SD1', 'SD2']
# 60s: 0.5655655655655656 Ranking: [1 1 6 4 2 1 5 3 1 1 7], Features: ['SDNN', 'RMSSD', 'hf', 'SD2', 'SD1SD2']

# Weighted   
# 5s :  0.6333248277621842 Ranking: [1 1 4 2 1 5 6 1 3 7 1] , Features: ['SDNN', 'RMSSD', 'lf', 'SD1', 'ApEn']
# 10s:  0.6619613670133729 Ranking: [1 1 1 1 3 4 5 1 2 7 6], Features: ['SDNN', 'RMSSD', 'ulf', 'vlf', 'SD1']
# 30s:  0.6951656222023277 Ranking: [1 1 6 4 1 2 3 1 1 5 7] , Features: ['SDNN', 'RMSSD', 'lf', 'SD1', 'SD2']
# 60s:  Ranking: , Features: 



# RFE CV 
# rfe = RFECV(estimator=DecisionTreeClassifier(),cv=5)
# model = SVC(kernel="linear")
# pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# pipeline.fit(rX_train,ry_train)
# score = pipeline.score(rX_test,ry_test)
# print(score)
## Raw 
# # 5s : 0.46185251339627453 Ranking: 
# # 10s: 0.48699851411589895 Ranking: 
# # 30s: 0.5434198746642793 Ranking: 
# # 60s: 0.591091091091091 Ranking: 

# RFE CV 2
# rfe = RFECV(estimator=SVC(kernel="linear"),cv=5)
# model = SVC()
# params_dist = {'m__C':[1,10,100,1000],
#                'm__gamma':[0.001,0.0001],
#                'm__kernel':['rbf','linear']
# }
# pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# r_search = RandomizedSearchCV(pipeline,param_distributions=params_dist,cv=5,random_state=0)
# r_search.fit(rX_train,ry_train)
# ry_pred = r_search.predict(rX_test,ry_test) # Predict using best parameters
# # Add in code to inspect features 
# print(r_search.best_params_)
# val_acc = accuracy_score(ry_test,ry_pred)
# print(val_acc)


