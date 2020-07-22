# Classify TLX labels using HRV
# SVM Classifer
import pandas as pd
import numpy as np
import glob,os, pathlib
from sklearn import svm 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from scipy.stats import loguniform
import re


cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
HRV_pathsList = glob.glob(str(mainDir)+"\\HRV_multiSubj\\Extracted-with_tlx_labels\\60s\\*.csv")

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

# Select features
rX_select = [9,6,5,4,1]
rX = rX[:,rX_select]
wX_select = [10,9,7,5,0]
wX = wX[:,wX_select]

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

print("Size of Feature and Label sets")
print(len(rX),len(ry))
print(len(wX),len(wy))

## Split into training and test sets
sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
def apply_sss(X,y):
    for train_index, test_index in sss.split(X,y):
            X_train, X_test = X[train_index],X[test_index]
            y_train, y_test = y[train_index],y[test_index]
    return X_train,X_test,y_train,y_test

rX_train,rX_test,ry_train,ry_test = apply_sss(rX,ry)
wX_train,wX_test,wy_train,wy_test = apply_sss(wX,wy)

# SVM Classifier
clf = svm.SVC()
# SVM Hyperparamater tuning and cross-validation 
print("Cross Validation...")
# params_dist = { 'C': loguniform(1e2,1e3),
#                 'gamma': [loguniform(1e-4,1e-3),'auto','scale'],
#                 'kernel': ['linear','rbf']
# }
params_dist = {'kernel':['linear','rbf'],
                'C':loguniform(1e2,1e3)}
r_search = RandomizedSearchCV(clf,params_dist,n_iter=10,cv=5,random_state=0)
# w_search = RandomizedSearchCV(clf,params_dist,n_iter=10,cv=5,n_jobs=-1,random_state=0)
r_search.fit(rX_train,ry_train)
w_search.fit(wX_train,wy_train)
# Use the best parameters from Cross-Validation 
r_params = r_search['best_params']
w_params = w_search['best_params']
r_clf = svm.SVC(**r_params)
w_clf = svm.SVC(**w_params)
r_clf = svm.SVC()
w_clf = svm.SVC()
# Training
print("Training...")
r_clf.fit(rX_train,ry_train)
w_clf.fit(wX_train,wy_train)
# Testing 
print("Testing...")
ry_pred = r_clf.predict(rX_test)
wy_pred = w_clf.predict(wX_test)
# Get accuracy scores
r_acc = accuracy_score(ry_test,ry_pred)
w_acc = accuracy_score(wy_test,wy_pred)
# Get confusion matrices 
r_cm = confusion_matrix(ry_test,ry_pred)
w_cm = confusion_matrix(wy_test,wy_pred)
print("Accuracy scores:")
print("Raw TLX:",r_acc)
print("Weighted TLX:",w_acc)
print("Confusion Matrices:")
print("Raw TLX CM")
print(r_cm)
print("Weigted TLX CM")
print(w_cm)

# 5s --- 5 Features
# Accuracy scores:
# rX_select = [10,5,4,1,0]
# wX_select = [10,8,7,5,1]
# Raw TLX: 0.5128859402908905
# Weighted TLX: 0.673896402143404
# Confusion Matrices:
# Raw TLX CM
# [[1285  419  130]
#  [ 459  501  159]
#  [ 537  205  224]]
# Weigted TLX CM
# [[1018  382    0]
#  [ 363 1623    0]
#  [ 172  361    0]]

# 10s --- 5 features
# rX_select = [10,9,7,5,0]
# wX_select = [10,9,5,1,0]
# Raw TLX: 0.562407132243685
# Weighted TLX: 0.7076523031203567
# Confusion Matrices:
# Raw TLX CM
# [[887 315  67]
#  [257 405 124]
#  [220 195 222]]
# Weigted TLX CM
# [[ 607  397    0]
#  [  80 1298    0]
#  [  10  300    0]]

# 30s --- 5 Features
# rX_select = [9,6,5,4,1]
# wX_select = [10,9,7,5,0]
# Accuracy scores:
# Raw TLX: 0.6230975828111012
# Weighted TLX: 0.7479856759176365
# Confusion Matrices:
# Raw TLX CM
# [[724 305  46]
#  [137 411 113]
#  [ 65 176 257]]
# Weigted TLX CM
# [[ 624  263    0]
#  [  65 1047    0]
#  [  14  221    0]]


# 60s --- 5 Features
# rX_select = [9,6,5,4,1]
# wX_select = [10,9,7,5,0]
# Accuracy scores:
# Raw TLX: 0.718968968968969
# Weighted TLX: 0.7855355355355356
# Confusion Matrices:
# Raw TLX CM
# [[1466  454    6]
#  [ 160  917  161]
#  [  57  285  490]]
# Weigted TLX CM
# [[1124  461    0]
#  [ 102 1924    6]
#  [  32  256   91]]