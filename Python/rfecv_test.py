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
HRV_pathsList = glob.glob(str(mainDir)+"\\HRV_multiSubj\\Extracted-with_tlx_labels\\5s\\*.csv")
savedir = str(mainDir)+"\\Plots\\Feature_comparison\\"

# Print feature names 
HRV_df0 = pd.read_csv(HRV_pathsList[0])
featureNames = HRV_df0.columns[1:-5].values
print("Feature Names")
print(featureNames)


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

# Split into training and test sets 
sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
def apply_sss(X,y):
    for train_index, test_index in sss.split(X,y):
            X_train, X_test = X[train_index],X[test_index]
            y_train, y_test = y[train_index],y[test_index]
    return X_train,X_test,y_train,y_test

rX_train,rX_test,ry_train,ry_test = apply_sss(rX,ry)
wX_train,wX_test,wy_train,wy_test = apply_sss(wX,wy)

# # RFE
# svc = SVC(kernel="linear")
# rfe = RFE(estimator=svc,n_features_to_select=5)
# rfe.fit(rX_train,ry_train)
# ranking = rfe.ranking_
# print("Ranking:")
# print(ranking)

# # Using only the selected features 
# feat_cols = [idx for idx, rank in enumerate(ranking) if rank==1]
# rX_train_sel = rX_train[:,feat_cols]
# rX_test_sel = rX_test[:,feat_cols]
# svc.fit(rX_train_sel,ry_train)
# ry_pred = svc.predict(rX_test_sel)
# val_acc = accuracy_score(ry_test,ry_pred)
# print(val_acc)  # 0.4679765246236285

# # RFE CV 
# rfe = RFECV(estimator=DecisionTreeClassifier(),cv=5)
# model = SVC(kernel="linear")
# pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# pipeline.fit(rX_train,ry_train)
# score = pipeline.score(rX_test,ry_test)
# print(score) # 0.4679765246236285

# RFE CV 2
rfe = RFECV(estimator=SVC(kernel="linear"),cv=5)
model = SVC()
params_dist = {'m__C':[1,10,100,1000],
               'm__gamma':[0.001,0.0001],
               'm__kernel':['rbf','linear']
}
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
r_search = RandomizedSearchCV(pipeline,param_distributions=params_dist,cv=5,random_state=0)
r_search.fit(rX_train,ry_train)
ry_pred = r_search.predict(rX_test,ry_test) # Predict using best parameters
# Add in code to inspect features 
print(r_search.best_params_)
val_acc = accuracy_score(ry_test,ry_pred)
print(val_acc)


