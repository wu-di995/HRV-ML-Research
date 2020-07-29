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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import re


win = "30s"
print(win+" SVM RESULTS---using same features for all datasets: SDNN, hf, RMSSD, ApEn, SD1SD2")
cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
HRV_pathsList = glob.glob(str(mainDir)+"\\HRV_allSubj\\Extracted-with_tlx_labels\\"+win+"\\*.csv")

# Combine data by labels, 
# 1) All interfaces, all autonomy levels
# 2) Separate by interface and autonomy
# 3) Separate by interface 

featureNames = ['SDNN','RMSSD','ulf','vlf','lf','hf','lfhf','SD1','SD2','SD1SD2','ApEn']
# Selected features: SDNN, hf, RMSSD, ApEn, SD1SD2
selFeatMask = [0,1,5,-2,-1]
interfaces = ["Headarray", "Sip-n-puff", "Joystick"]
autonomies = ["A0", "A1", "A2"]
int_autoList = [interface+"_"+autonomy for interface in interfaces for autonomy in autonomies]
int_autoList = [i for i in int_autoList if (("Sip-n-puff_A1" not in i) and ("Sip-n-puff_A2" not in i))]

int_auto_datasets_dict = {}
int_datasets_dict = {}

def get_data(HRVpaths):
    for i,path in enumerate(HRVpaths):
        HRVdf = pd.read_csv(path)
        # Do not inlcude the last 4 columns and first column
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
        # Create np arrarys for raw and weighted labels, low/med/high
        if rLabel == 0:
            try: 
                rLow_ar = np.vstack((rLow_ar,colData)) # Do not inlcude the last 4 columns and first column
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
    return [rLow_ar,rHigh_ar,wLow_ar,wHigh_ar]

# Create dictionary of datsets, where data is separated by interface, autonomy and label 
for int_auto in int_autoList:
    HRV_paths = [path for path in HRV_pathsList if int_auto in path]
    int_auto_datasets_dict[int_auto] = get_data(HRV_paths)
# Create dictionary of datasets, where data is separated by interface and label 
for interface in interfaces:
    HRV_paths = [path for path in HRV_pathsList if interface in path]
    int_datasets_dict[interface] = get_data(HRV_paths)

# Feature scale all datasets, 
def featureScale_dataDict(dataDict):
    sc_dataDict = {} # Scaled dataset dictionary 
    sc = StandardScaler()
    for key,datasetList in dataDict.items():
        # Raw Labels
        rX = np.vstack((datasetList[0],datasetList[1]))
        rX = sc.fit_transform(rX)
        ry = np.hstack( (np.zeros(len(datasetList[0])),np.ones(len(datasetList[1])) ))
        # print("rX:", len(rX))
        # print("ry:", len(ry))
        # Weighted Labels
        wX = np.vstack((datasetList[2],datasetList[3]))
        wX = sc.fit_transform(rX)
        wy = np.hstack( (np.zeros(len(datasetList[2])),np.ones(len(datasetList[3])) ))
        # print("wX:", len(wX))
        # print("wy:", len(wy))
        sc_dataDict[key] = [rX,ry,wX,wy]
    return sc_dataDict

sc_int_auto_datasets = featureScale_dataDict(int_auto_datasets_dict)
sc_int_datasets = featureScale_dataDict(int_datasets_dict)

# Split into training and test sets
sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
def apply_sss(X,y):
    for train_index, test_index in sss.split(X,y):
            X_train, X_test = X[train_index],X[test_index]
            y_train, y_test = y[train_index],y[test_index]
    return X_train,X_test,y_train,y_test


# Calculate feature important and find top 5 features
def feat_impt_multiModels(X_train,y_train,X_test,models,featureNames,datasetName):
    featureScores = [0]*len(featureNames)
    for i,model in enumerate(models):
        model.fit(X_train,y_train)
        importances = model.feature_importances_
        featureScores = [featureScore + importance for featureScore,importance in zip(featureScores,importances)]
    featureScores,names = zip(*sorted(zip(featureScores,featureNames))) #Ascending order
    featureScores = list(featureScores)
    names = list(names)
    # Choose top 5 features 
    # print(featureScores)
    # print(names)
    topFeatures = names[-5:]
    # Check if RMSSD and SD1 are both selected, only choose the more important one, and add the 6th feature
    if ("RMSSD" in topFeatures) and ("SD1" in topFeatures) :
        if topFeatures.index("RMSSD") < topFeatures.index("SD1"):
            topFeatures.remove("SD1")
        else:
            topFeatures.remove("RMSSD")
        topFeatures.append(names[-6])
    topFeaturesMask = [i for i,feature in enumerate(featureNames) if feature in topFeatures]
    X_train_sel = X_train[:,topFeaturesMask]
    X_test_sel = X_test[:,topFeaturesMask]
    print(datasetName+" top features: ", topFeatures)
    return X_train_sel,X_test_sel

# Select features 
def selectFeatures(sc_dataDict,featureNames):
    models = [ExtraTreesClassifier(random_state=0),DecisionTreeClassifier(random_state=0),XGBClassifier(random_state=0)]
    modelNames = ["ExtraTreesClassifier","DecisionTreeClassifer","XGBClassifier"]
    bf_dataDict = {}
    for key,datasetList in sc_dataDict.items():
        rX = datasetList[0]
        ry = datasetList[1]
        wX = datasetList[2]
        wy = datasetList[3]
        # Split into training and test sets
        rX_train,rX_test,ry_train,ry_test = apply_sss(rX,ry)
        wX_train,wX_test,wy_train,wy_test = apply_sss(wX,wy)

        # (either) Select same features for all 
        rX_train,rX_test = [rX_train[:,selFeatMask],rX_test[:,selFeatMask]]
        wX_train,wX_test = [wX_train[:,selFeatMask],wX_test[:,selFeatMask]]
        # (or) Select best features 
        # rX_train,rX_test = feat_impt_multiModels(rX_train,ry_train,rX_test,models,featureNames,key+"_rawLabel")
        # wX_train,wX_test = feat_impt_multiModels(wX_train,wy_train,wX_test,models,featureNames,key+"_weightedLabel")
        # Fill in dataset dictionary 
        bf_dataDict[key] = [rX_train,rX_test,ry_train,ry_test,wX_train,wX_test,wy_train,wy_test]
    return bf_dataDict

# Datasets with best features (bf)
bf_int_auto_dataDict  = selectFeatures(sc_int_auto_datasets,featureNames)
bf_int_dataDict  = selectFeatures(sc_int_datasets,featureNames)

# SVM Classifier 
def applySVM(dataDict):
    clf = svm.SVC()
    for key, datasetList in dataDict.items():
        rX_train,rX_test,ry_train,ry_test,wX_train,wX_test,wy_train,wy_test = datasetList
        # Raw Label Classification 
        clf.fit(rX_train,ry_train)
        ry_pred = clf.predict(rX_test)
        r_acc = accuracy_score(ry_test,ry_pred)
        r_cm = confusion_matrix(ry_test,ry_pred)
        # Weighted Label Classification 
        clf.fit(wX_train,wy_train)
        wy_pred = clf.predict(wX_test)
        w_acc = accuracy_score(wy_test,wy_pred)
        w_cm = confusion_matrix(wy_test,wy_pred)
        # Print Results 
        print("***********************")
        print("Dataset: ",key)
        print("Raw label accuracy: ",r_acc)
        print("Raw Label confusion matrix: \n",r_cm)
        print("----------------------")
        print("Weighted label accuracy: ",w_acc)
        print("Weighted Label confusion matrix: \n",w_cm)
        print("***********************")

print("***********************")
print("Datasets separated by autonomy and interface")
applySVM(bf_int_auto_dataDict)
print("***********************")
print("Datasets separated by interface only")
applySVM(bf_int_dataDict)




##### ALL SUBJECTS RESULTS #####

# 30s SVM RESULTS---using same features for all datasets: SDNN, hf, RMSSD, ApEn, SD1SD2
# ***********************
# Datasets separated by autonomy and interface
# ***********************
# Dataset:  Headarray_A0
# Raw label accuracy:  0.7942583732057417
# Raw Label confusion matrix:
#  [[133   3]
#  [ 40  33]]
# ----------------------
# Weighted label accuracy:  0.7942583732057417
# Weighted Label confusion matrix:
#  [[133   3]
#  [ 40  33]]
# ***********************
# ***********************
# Dataset:  Headarray_A1
# Raw label accuracy:  0.7816593886462883
# Raw Label confusion matrix:
#  [[232  44]
#  [ 56 126]]
# ----------------------
# Weighted label accuracy:  0.8275109170305677
# Weighted Label confusion matrix:
#  [[302  10]
#  [ 69  77]]
# ***********************
# ***********************
# Dataset:  Headarray_A2
# Raw label accuracy:  0.9650793650793651
# Raw Label confusion matrix:
#  [[ 99  17]
#  [  5 509]]
# ----------------------
# Weighted label accuracy:  0.9650793650793651
# Weighted Label confusion matrix:
#  [[ 99  17]
#  [  5 509]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff_A0
# Raw label accuracy:  0.8595271210013908
# Raw Label confusion matrix:
#  [[158  53]
#  [ 48 460]]
# ----------------------
# Weighted label accuracy:  0.847009735744089
# Weighted Label confusion matrix:
#  [[157  56]
#  [ 54 452]]
# ***********************
# ***********************
# Dataset:  Joystick_A0
# Raw label accuracy:  0.8969072164948454
# Raw Label confusion matrix:
#  [[96 14]
#  [ 6 78]]
# ----------------------
# Weighted label accuracy:  0.8969072164948454
# Weighted Label confusion matrix:
#  [[96 14]
#  [ 6 78]]
# ***********************
# ***********************
# Dataset:  Joystick_A1
# Raw label accuracy:  0.9417989417989417
# Raw Label confusion matrix:
#  [[144   1]
#  [ 10  34]]
# ----------------------
# Weighted label accuracy:  0.8835978835978836
# Weighted Label confusion matrix:
#  [[127   0]
#  [ 22  40]]
# ***********************
# ***********************
# Dataset:  Joystick_A2
# Raw label accuracy:  0.9609375
# Raw Label confusion matrix:
#  [[129  17]
#  [  3 363]]
# ----------------------
# Weighted label accuracy:  0.9609375
# Weighted Label confusion matrix:
#  [[129  17]
#  [  3 363]]
# ***********************
# ***********************
# Datasets separated by interface only
# ***********************
# Dataset:  Headarray
# Raw label accuracy:  0.8024691358024691
# Raw Label confusion matrix:
#  [[345 182]
#  [ 74 695]]
# ----------------------
# Weighted label accuracy:  0.7924382716049383
# Weighted Label confusion matrix:
#  [[376 187]
#  [ 82 651]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff
# Raw label accuracy:  0.8595271210013908
# Raw Label confusion matrix:
#  [[158  53]
#  [ 48 460]]
# ----------------------
# Weighted label accuracy:  0.847009735744089
# Weighted Label confusion matrix:
#  [[157  56]
#  [ 54 452]]
# ***********************
# ***********************
# Dataset:  Joystick
# Raw label accuracy:  0.8064876957494407
# Raw Label confusion matrix:
#  [[251 148]
#  [ 25 470]]
# ----------------------
# Weighted label accuracy:  0.8031319910514542
# Weighted Label confusion matrix:
#  [[213 168]
#  [  8 505]]
# ***********************



# 60s SVM RESULTS---using same features for all datasets: SDNN, hf, RMSSD, ApEn, SD1SD2
# ***********************
# Datasets separated by autonomy and interface
# ***********************
# Dataset:  Headarray_A0
# Raw label accuracy:  0.8143712574850299
# Raw Label confusion matrix:
#  [[108   1]
#  [ 30  28]]
# ----------------------
# Weighted label accuracy:  0.8143712574850299
# Weighted Label confusion matrix:
#  [[108   1]
#  [ 30  28]]
# ***********************
# ***********************
# Dataset:  Headarray_A1
# Raw label accuracy:  0.8941176470588236
# Raw Label confusion matrix:
#  [[235  18]
#  [ 27 145]]
# ----------------------
# Weighted label accuracy:  0.8588235294117647
# Weighted Label confusion matrix:
#  [[264  19]
#  [ 41 101]]
# ***********************
# ***********************
# Dataset:  Headarray_A2
# Raw label accuracy:  0.9835766423357665
# Raw Label confusion matrix:
#  [[100   4]
#  [  5 439]]
# ----------------------
# Weighted label accuracy:  0.9835766423357665
# Weighted Label confusion matrix:
#  [[100   4]
#  [  5 439]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff_A0
# Raw label accuracy:  0.9024390243902439
# Raw Label confusion matrix:
#  [[188  16]
#  [ 48 404]]
# ----------------------
# Weighted label accuracy:  0.9024390243902439
# Weighted Label confusion matrix:
#  [[188  16]
#  [ 48 404]]
# ***********************
# ***********************
# Dataset:  Joystick_A0
# Raw label accuracy:  0.9877300613496932
# Raw Label confusion matrix:
#  [[79  2]
#  [ 0 82]]
# ----------------------
# Weighted label accuracy:  0.9877300613496932
# Weighted Label confusion matrix:
#  [[79  2]
#  [ 0 82]]
# ***********************
# ***********************
# Dataset:  Joystick_A1
# Raw label accuracy:  1.0
# Raw Label confusion matrix:
#  [[110   0]
#  [  0  32]]
# ----------------------
# Weighted label accuracy:  1.0
# Weighted Label confusion matrix:
#  [[103   0]
#  [  0  39]]
# ***********************
# ***********************
# Dataset:  Joystick_A2
# Raw label accuracy:  0.9681528662420382
# Raw Label confusion matrix:
#  [[116  15]
#  [  0 340]]
# ----------------------
# Weighted label accuracy:  0.9681528662420382
# Weighted Label confusion matrix:
#  [[116  15]
#  [  0 340]]
# ***********************
# ***********************
# Datasets separated by interface only
# ***********************
# Dataset:  Headarray
# Raw label accuracy:  0.8482456140350877
# Raw Label confusion matrix:
#  [[359 106]
#  [ 67 608]]
# ----------------------
# Weighted label accuracy:  0.8359649122807018
# Weighted Label confusion matrix:
#  [[372 123]
#  [ 64 581]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff
# Raw label accuracy:  0.9024390243902439
# Raw Label confusion matrix:
#  [[188  16]
#  [ 48 404]]
# ----------------------
# Weighted label accuracy:  0.9024390243902439
# Weighted Label confusion matrix:
#  [[188  16]
#  [ 48 404]]
# ***********************
# ***********************
# Dataset:  Joystick
# Raw label accuracy:  0.845360824742268
# Raw Label confusion matrix:
#  [[217 105]
#  [ 15 439]]
# ----------------------
# Weighted label accuracy:  0.8621134020618557
# Weighted Label confusion matrix:
#  [[212 103]
#  [  4 457]]
# ***********************

# 30s SVM RESULTS
# Headarray_A0_rawLabel top features:  ['vlf', 'hf', 'RMSSD', 'lf', 'ApEn']
# Headarray_A0_weightedLabel top features:  ['vlf', 'hf', 'RMSSD', 'lf', 'ApEn']
# Headarray_A1_rawLabel top features:  ['SDNN', 'hf', 'RMSSD', 'lfhf', 'lf']
# Headarray_A1_weightedLabel top features:  ['SDNN', 'SD1', 'lf', 'hf', 'lfhf']
# Headarray_A2_rawLabel top features:  ['SDNN', 'SD1', 'SD1SD2', 'hf', 'lfhf']
# Headarray_A2_weightedLabel top features:  ['SDNN', 'SD1', 'SD1SD2', 'hf', 'lfhf']
# Sip-n-puff_A0_rawLabel top features:  ['ApEn', 'SD1SD2', 'SD1', 'hf', 'SDNN']
# Sip-n-puff_A0_weightedLabel top features:  ['ApEn', 'SD1SD2', 'SD1', 'hf', 'SDNN']
# Joystick_A0_rawLabel top features:  ['SD2', 'RMSSD', 'lfhf', 'SD1SD2', 'SDNN']
# Joystick_A0_weightedLabel top features:  ['SD2', 'RMSSD', 'lfhf', 'SD1SD2', 'SDNN']
# Joystick_A1_rawLabel top features:  ['ulf', 'SDNN', 'lf', 'SD1', 'hf']
# Joystick_A1_weightedLabel top features:  ['ApEn', 'lf', 'RMSSD', 'SDNN', 'hf']
# Joystick_A2_rawLabel top features:  ['SD1SD2', 'SDNN', 'hf', 'RMSSD', 'ApEn']
# Joystick_A2_weightedLabel top features:  ['SD1SD2', 'SDNN', 'hf', 'RMSSD', 'ApEn']
# Headarray_rawLabel top features:  ['RMSSD', 'hf', 'SDNN', 'lfhf', 'ApEn']
# Headarray_weightedLabel top features:  ['RMSSD', 'hf', 'SDNN', 'lfhf', 'SD1SD2']
# Sip-n-puff_rawLabel top features:  ['ApEn', 'SD1SD2', 'SD1', 'hf', 'SDNN']
# Sip-n-puff_weightedLabel top features:  ['ApEn', 'SD1SD2', 'SD1', 'hf', 'SDNN']
# Joystick_rawLabel top features:  ['lf', 'hf', 'SDNN', 'RMSSD', 'SD1SD2']
# Joystick_weightedLabel top features:  ['lf', 'hf', 'SDNN', 'RMSSD', 'SD1SD2']
# ***********************
# Datasets separated by autonomy and interface
# ***********************
# Dataset:  Headarray_A0
# Raw label accuracy:  0.8373205741626795
# Raw Label confusion matrix:
#  [[129   7]
#  [ 27  46]]
# ----------------------
# Weighted label accuracy:  0.8373205741626795
# Weighted Label confusion matrix:
#  [[129   7]
#  [ 27  46]]
# ***********************
# ***********************
# Dataset:  Headarray_A1
# Raw label accuracy:  0.8013100436681223
# Raw Label confusion matrix:
#  [[233  43]
#  [ 48 134]]
# ----------------------
# Weighted label accuracy:  0.8493449781659389
# Weighted Label confusion matrix:
#  [[288  24]
#  [ 45 101]]
# ***********************
# ***********************
# Dataset:  Headarray_A2
# Raw label accuracy:  0.973015873015873
# Raw Label confusion matrix:
#  [[104  12]
#  [  5 509]]
# ----------------------
# Weighted label accuracy:  0.973015873015873
# Weighted Label confusion matrix:
#  [[104  12]
#  [  5 509]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff_A0
# Raw label accuracy:  0.8595271210013908
# Raw Label confusion matrix:
#  [[158  53]
#  [ 48 460]]
# ----------------------
# Weighted label accuracy:  0.847009735744089
# Weighted Label confusion matrix:
#  [[157  56]
#  [ 54 452]]
# ***********************
# ***********************
# Dataset:  Joystick_A0
# Raw label accuracy:  0.9175257731958762
# Raw Label confusion matrix:
#  [[100  10]
#  [  6  78]]
# ----------------------
# Weighted label accuracy:  0.9175257731958762
# Weighted Label confusion matrix:
#  [[100  10]
#  [  6  78]]
# ***********************
# ***********************
# Dataset:  Joystick_A1
# Raw label accuracy:  0.9047619047619048
# Raw Label confusion matrix:
#  [[138   7]
#  [ 11  33]]
# ----------------------
# Weighted label accuracy:  0.8783068783068783
# Weighted Label confusion matrix:
#  [[126   1]
#  [ 22  40]]
# ***********************
# ***********************
# Dataset:  Joystick_A2
# Raw label accuracy:  0.9609375
# Raw Label confusion matrix:
#  [[129  17]
#  [  3 363]]
# ----------------------
# Weighted label accuracy:  0.9609375
# Weighted Label confusion matrix:
#  [[129  17]
#  [  3 363]]
# ***********************
# ***********************
# Datasets separated by interface only
# ***********************
# Dataset:  Headarray
# Raw label accuracy:  0.8063271604938271
# Raw Label confusion matrix:
#  [[355 172]
#  [ 79 690]]
# ----------------------
# Weighted label accuracy:  0.7862654320987654
# Weighted Label confusion matrix:
#  [[352 211]
#  [ 66 667]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff
# Raw label accuracy:  0.8595271210013908
# Raw Label confusion matrix:
#  [[158  53]
#  [ 48 460]]
# ----------------------
# Weighted label accuracy:  0.847009735744089
# Weighted Label confusion matrix:
#  [[157  56]
#  [ 54 452]]
# ***********************
# ***********************
# Dataset:  Joystick
# Raw label accuracy:  0.7930648769574944
# Raw Label confusion matrix:
#  [[235 164]
#  [ 21 474]]
# ----------------------
# Weighted label accuracy:  0.7841163310961968
# Weighted Label confusion matrix:
#  [[202 179]
#  [ 14 499]]
# ***********************


# 60s SVM RESULTS
# Headarray_A0_rawLabel top features:  ['vlf', 'lfhf', 'RMSSD', 'SDNN', 'lf']
# Headarray_A0_weightedLabel top features:  ['vlf', 'lfhf', 'RMSSD', 'SDNN', 'lf']
# Headarray_A1_rawLabel top features:  ['lf', 'SD1', 'SDNN', 'lfhf', 'hf']
# Headarray_A1_weightedLabel top features:  ['hf', 'RMSSD', 'ApEn', 'lf', 'SDNN']
# Headarray_A2_rawLabel top features:  ['ApEn', 'SD1', 'SDNN', 'hf', 'lfhf']
# Headarray_A2_weightedLabel top features:  ['ApEn', 'SD1', 'SDNN', 'hf', 'lfhf']
# Sip-n-puff_A0_rawLabel top features:  ['SD1SD2', 'ApEn', 'RMSSD', 'hf', 'lfhf']
# Sip-n-puff_A0_weightedLabel top features:  ['SD1SD2', 'ApEn', 'RMSSD', 'hf', 'lfhf']
# Joystick_A0_rawLabel top features:  ['SD2', 'SDNN', 'SD1SD2', 'RMSSD', 'vlf']
# Joystick_A0_weightedLabel top features:  ['SD2', 'SDNN', 'SD1SD2', 'RMSSD', 'vlf']
# Joystick_A1_rawLabel top features:  ['hf', 'SDNN', 'lfhf', 'SD1', 'lf']
# Joystick_A1_weightedLabel top features:  ['hf', 'ApEn', 'RMSSD', 'SDNN', 'lf']
# Joystick_A2_rawLabel top features:  ['SDNN', 'SD1SD2', 'hf', 'RMSSD', 'vlf']
# Joystick_A2_weightedLabel top features:  ['SDNN', 'SD1SD2', 'hf', 'RMSSD', 'vlf']
# Headarray_rawLabel top features:  ['SDNN', 'lf', 'RMSSD', 'hf', 'lfhf']
# Headarray_weightedLabel top features:  ['hf', 'ApEn', 'lf', 'RMSSD', 'lfhf']
# Sip-n-puff_rawLabel top features:  ['SD1SD2', 'ApEn', 'RMSSD', 'hf', 'lfhf']
# Sip-n-puff_weightedLabel top features:  ['SD1SD2', 'ApEn', 'RMSSD', 'hf', 'lfhf']
# Joystick_rawLabel top features:  ['lf', 'hf', 'SDNN', 'RMSSD', 'SD1SD2']
# Joystick_weightedLabel top features:  ['hf', 'SDNN', 'lf', 'SD1', 'SD1SD2']
# ***********************
# Datasets separated by autonomy and interface
# ***********************
# Dataset:  Headarray_A0
# Raw label accuracy:  0.9041916167664671
# Raw Label confusion matrix:
#  [[104   5]
#  [ 11  47]]
# ----------------------
# Weighted label accuracy:  0.9041916167664671
# Weighted Label confusion matrix:
#  [[104   5]
#  [ 11  47]]
# ***********************
# ***********************
# Dataset:  Headarray_A1
# Raw label accuracy:  0.8964705882352941
# Raw Label confusion matrix:
#  [[249   4]
#  [ 40 132]]
# ----------------------
# Weighted label accuracy:  0.8564705882352941
# Weighted Label confusion matrix:
#  [[276   7]
#  [ 54  88]]
# ***********************
# ***********************
# Dataset:  Headarray_A2
# Raw label accuracy:  0.9854014598540146
# Raw Label confusion matrix:
#  [[ 96   8]
#  [  0 444]]
# ----------------------
# Weighted label accuracy:  0.9854014598540146
# Weighted Label confusion matrix:
#  [[ 96   8]
#  [  0 444]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff_A0
# Raw label accuracy:  0.9253048780487805
# Raw Label confusion matrix:
#  [[174  30]
#  [ 19 433]]
# ----------------------
# Weighted label accuracy:  0.9253048780487805
# Weighted Label confusion matrix:
#  [[174  30]
#  [ 19 433]]
# ***********************
# ***********************
# Dataset:  Joystick_A0
# Raw label accuracy:  0.9754601226993865
# Raw Label confusion matrix:
#  [[79  2]
#  [ 2 80]]
# ----------------------
# Weighted label accuracy:  0.9754601226993865
# Weighted Label confusion matrix:
#  [[79  2]
#  [ 2 80]]
# ***********************
# ***********************
# Dataset:  Joystick_A1
# Raw label accuracy:  1.0
# Raw Label confusion matrix:
#  [[110   0]
#  [  0  32]]
# ----------------------
# Weighted label accuracy:  0.9929577464788732
# Weighted Label confusion matrix:
#  [[103   0]
#  [  1  38]]
# ***********************
# ***********************
# Dataset:  Joystick_A2
# Raw label accuracy:  0.9681528662420382
# Raw Label confusion matrix:
#  [[116  15]
#  [  0 340]]
# ----------------------
# Weighted label accuracy:  0.9681528662420382
# Weighted Label confusion matrix:
#  [[116  15]
#  [  0 340]]
# ***********************
# ***********************
# Datasets separated by interface only
# ***********************
# Dataset:  Headarray
# Raw label accuracy:  0.8403508771929824
# Raw Label confusion matrix:
#  [[339 126]
#  [ 56 619]]
# ----------------------
# Weighted label accuracy:  0.8666666666666667
# Weighted Label confusion matrix:
#  [[407  88]
#  [ 64 581]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff
# Raw label accuracy:  0.9253048780487805
# Raw Label confusion matrix:
#  [[174  30]
#  [ 19 433]]
# ----------------------
# Weighted label accuracy:  0.9253048780487805
# Weighted Label confusion matrix:
#  [[174  30]
#  [ 19 433]]
# ***********************
# ***********************
# Dataset:  Joystick
# Raw label accuracy:  0.8402061855670103
# Raw Label confusion matrix:
#  [[215 107]
#  [ 17 437]]
# ----------------------
# Weighted label accuracy:  0.8530927835051546
# Weighted Label confusion matrix:
#  [[205 110]
#  [  4 457]]
# ***********************






################################

# 5s SVM RESULTS
# Headarray_A0_rawLabel top features:  ['RMSSD', 'lfhf', 'SD1SD2', 'ApEn', 'hf']
# Headarray_A0_weightedLabel top features:  ['RMSSD', 'lfhf', 'SD1SD2', 'ApEn', 'hf']
# Headarray_A1_rawLabel top features:  ['SD1', 'hf', 'lfhf', 'ApEn', 'SDNN']
# Headarray_A1_weightedLabel top features:  ['SD1', 'hf', 'lfhf', 'ApEn', 'SDNN']
# Headarray_A2_rawLabel top features:  ['SD1SD2', 'RMSSD', 'hf', 'lfhf', 'ApEn']
# Headarray_A2_weightedLabel top features:  ['SD1SD2', 'RMSSD', 'hf', 'lfhf', 'ApEn']
# Sip-n-puff_A0_rawLabel top features:  ['SDNN', 'ApEn', 'SD1', 'hf', 'lfhf']
# Sip-n-puff_A0_weightedLabel top features:  ['SDNN', 'ApEn', 'SD1', 'hf', 'lfhf']
# Joystick_A0_rawLabel top features:  ['RMSSD', 'SDNN', 'hf', 'ApEn', 'lfhf']
# Joystick_A0_weightedLabel top features:  ['RMSSD', 'SDNN', 'hf', 'ApEn', 'lfhf']
# Joystick_A1_rawLabel top features:  ['RMSSD', 'hf', 'SDNN', 'ApEn', 'lf']
# Joystick_A1_weightedLabel top features:  ['SDNN', 'RMSSD', 'hf', 'ApEn', 'SD1SD2']
# Joystick_A2_rawLabel top features:  ['hf', 'SD1', 'SDNN', 'ApEn', 'SD2']
# Joystick_A2_weightedLabel top features:  ['hf', 'SD1', 'SDNN', 'ApEn', 'SD2']
# Headarray_rawLabel top features:  ['SDNN', 'lfhf', 'hf', 'RMSSD', 'ApEn']
# Headarray_weightedLabel top features:  ['SDNN', 'lfhf', 'hf', 'RMSSD', 'ApEn']
# Sip-n-puff_rawLabel top features:  ['SDNN', 'ApEn', 'SD1', 'hf', 'lfhf']
# Sip-n-puff_weightedLabel top features:  ['SDNN', 'ApEn', 'SD1', 'hf', 'lfhf']
# Joystick_rawLabel top features:  ['SD1', 'SDNN', 'hf', 'ApEn', 'SD1SD2']
# Joystick_weightedLabel top features:  ['RMSSD', 'hf', 'SDNN', 'ApEn', 'SD2']
# ***********************
# Datasets separated by autonomy and interface
# ***********************
# Dataset:  Headarray_A0
# Raw label accuracy:  0.7470238095238095
# Raw Label confusion matrix:
#  [[232  13]
#  [ 72  19]]
# ----------------------
# Weighted label accuracy:  0.7470238095238095
# Weighted Label confusion matrix:
#  [[232  13]
#  [ 72  19]]
# ***********************
# ***********************
# Dataset:  Headarray_A1
# Raw label accuracy:  0.7935943060498221
# Raw Label confusion matrix:
#  [[320  56]
#  [ 60 126]]
# ----------------------
# Weighted label accuracy:  0.7935943060498221
# Weighted Label confusion matrix:
#  [[320  56]
#  [ 60 126]]
# ***********************
# ***********************
# Dataset:  Headarray_A2
# Raw label accuracy:  0.8261376896149358
# Raw Label confusion matrix:
#  [[107 121]
#  [ 28 601]]
# ----------------------
# Weighted label accuracy:  0.8261376896149358
# Weighted Label confusion matrix:
#  [[107 121]
#  [ 28 601]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff_A0
# Raw label accuracy:  0.6910229645093946
# Raw Label confusion matrix:
#  [[  2 290]
#  [  6 660]]
# ----------------------
# Weighted label accuracy:  0.6910229645093946
# Weighted Label confusion matrix:
#  [[  2 290]
#  [  6 660]]
# ***********************
# ***********************
# Dataset:  Joystick_A0
# Raw label accuracy:  0.8604651162790697
# Raw Label confusion matrix:
#  [[166   7]
#  [ 29  56]]
# ----------------------
# Weighted label accuracy:  0.8604651162790697
# Weighted Label confusion matrix:
#  [[166   7]
#  [ 29  56]]
# ***********************
# ***********************
# Dataset:  Joystick_A1
# Raw label accuracy:  0.7987804878048781
# Raw Label confusion matrix:
#  [[225  12]
#  [ 54  37]]
# ----------------------
# Weighted label accuracy:  0.7408536585365854
# Weighted Label confusion matrix:
#  [[125  44]
#  [ 41 118]]
# ***********************
# ***********************
# Dataset:  Joystick_A2
# Raw label accuracy:  0.9694533762057878
# Raw Label confusion matrix:
#  [[106  15]
#  [  4 497]]
# ----------------------
# Weighted label accuracy:  0.9694533762057878
# Weighted Label confusion matrix:
#  [[106  15]
#  [  4 497]]
# ***********************
# ***********************
# Datasets separated by interface only
# ***********************
# Dataset:  Headarray
# Raw label accuracy:  0.7371721778791334
# Raw Label confusion matrix:
#  [[532 317]
#  [144 761]]
# ----------------------
# Weighted label accuracy:  0.7371721778791334
# Weighted Label confusion matrix:
#  [[532 317]
#  [144 761]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff
# Raw label accuracy:  0.6910229645093946
# Raw Label confusion matrix:
#  [[  2 290]
#  [  6 660]]
# ----------------------
# Weighted label accuracy:  0.6910229645093946
# Weighted Label confusion matrix:
#  [[  2 290]
#  [  6 660]]
# ***********************
# ***********************
# Dataset:  Joystick
# Raw label accuracy:  0.6785418392709196
# Raw Label confusion matrix:
#  [[196 335]
#  [ 53 623]]
# ----------------------
# Weighted label accuracy:  0.7415078707539354
# Weighted Label confusion matrix:
#  [[161 302]
#  [ 10 734]]
# ***********************

# 10s SVM RESULTS
# Headarray_A0_rawLabel top features:  ['lf', 'SDNN', 'RMSSD', 'lfhf', 'ApEn']
# Headarray_A0_weightedLabel top features:  ['lf', 'SDNN', 'RMSSD', 'lfhf', 'ApEn']
# Headarray_A1_rawLabel top features:  ['SDNN', 'hf', 'RMSSD', 'ApEn', 'SD1SD2']
# Headarray_A1_weightedLabel top features:  ['SDNN', 'hf', 'RMSSD', 'ApEn', 'SD1SD2']
# Headarray_A2_rawLabel top features:  ['SD1', 'hf', 'lfhf', 'ApEn', 'SD1SD2']
# Headarray_A2_weightedLabel top features:  ['SD1', 'hf', 'lfhf', 'ApEn', 'SD1SD2']
# Sip-n-puff_A0_rawLabel top features:  ['SDNN', 'SD1', 'ApEn', 'hf', 'SD1SD2']
# Sip-n-puff_A0_weightedLabel top features:  ['SDNN', 'SD1', 'ApEn', 'hf', 'SD1SD2']
# Joystick_A0_rawLabel top features:  ['SD1SD2', 'lf', 'RMSSD', 'hf', 'ApEn']
# Joystick_A0_weightedLabel top features:  ['SD1SD2', 'lf', 'RMSSD', 'hf', 'ApEn']
# Joystick_A1_rawLabel top features:  ['SD1', 'SDNN', 'hf', 'ApEn', 'SD2']
# Joystick_A1_weightedLabel top features:  ['SD1SD2', 'SDNN', 'SD1', 'ApEn', 'hf']
# Joystick_A2_rawLabel top features:  ['hf', 'ApEn', 'SDNN', 'RMSSD', 'SD2']
# Joystick_A2_weightedLabel top features:  ['hf', 'ApEn', 'SDNN', 'RMSSD', 'SD2']
# Headarray_rawLabel top features:  ['SDNN', 'lfhf', 'hf', 'SD1', 'ApEn']
# Headarray_weightedLabel top features:  ['SDNN', 'lfhf', 'hf', 'SD1', 'ApEn']
# Sip-n-puff_rawLabel top features:  ['SDNN', 'SD1', 'ApEn', 'hf', 'SD1SD2']
# Sip-n-puff_weightedLabel top features:  ['SDNN', 'SD1', 'ApEn', 'hf', 'SD1SD2']
# Joystick_rawLabel top features:  ['SDNN', 'hf', 'RMSSD', 'ApEn', 'SD1SD2']
# Joystick_weightedLabel top features:  ['SD1SD2', 'hf', 'RMSSD', 'ApEn', 'SDNN']
# ***********************
# Datasets separated by autonomy and interface
# ***********************
# Dataset:  Headarray_A0
# Raw label accuracy:  0.7209302325581395
# Raw Label confusion matrix:
#  [[103  30]
#  [ 30  52]]
# ----------------------
# Weighted label accuracy:  0.7209302325581395
# Weighted Label confusion matrix:
#  [[103  30]
#  [ 30  52]]
# ***********************
# ***********************
# Dataset:  Headarray_A1
# Raw label accuracy:  0.7855477855477856
# Raw Label confusion matrix:
#  [[241  26]
#  [ 66  96]]
# ----------------------
# Weighted label accuracy:  0.7855477855477856
# Weighted Label confusion matrix:
#  [[241  26]
#  [ 66  96]]
# ***********************
# ***********************
# Dataset:  Headarray_A2
# Raw label accuracy:  0.8930390492359932
# Raw Label confusion matrix:
#  [[100  35]
#  [ 28 426]]
# ----------------------
# Weighted label accuracy:  0.8930390492359932
# Weighted Label confusion matrix:
#  [[100  35]
#  [ 28 426]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff_A0
# Raw label accuracy:  0.7555886736214605
# Raw Label confusion matrix:
#  [[161  66]
#  [ 98 346]]
# ----------------------
# Weighted label accuracy:  0.7555886736214605
# Weighted Label confusion matrix:
#  [[161  66]
#  [ 98 346]]
# ***********************
# ***********************
# Dataset:  Joystick_A0
# Raw label accuracy:  0.7807486631016043
# Raw Label confusion matrix:
#  [[107  17]
#  [ 24  39]]
# ----------------------
# Weighted label accuracy:  0.7807486631016043
# Weighted Label confusion matrix:
#  [[107  17]
#  [ 24  39]]
# ***********************
# ***********************
# Dataset:  Joystick_A1
# Raw label accuracy:  0.9214659685863874
# Raw Label confusion matrix:
#  [[139   5]
#  [ 10  37]]
# ----------------------
# Weighted label accuracy:  0.8429319371727748
# Weighted Label confusion matrix:
#  [[108  10]
#  [ 20  53]]
# ***********************
# ***********************
# Dataset:  Joystick_A2
# Raw label accuracy:  0.9757281553398058
# Raw Label confusion matrix:
#  [[115   7]
#  [  3 287]]
# ----------------------
# Weighted label accuracy:  0.9757281553398058
# Weighted Label confusion matrix:
#  [[115   7]
#  [  3 287]]
# ***********************
# ***********************
# Datasets separated by interface only
# ***********************
# Dataset:  Headarray
# Raw label accuracy:  0.7631792376317924
# Raw Label confusion matrix:
#  [[310 224]
#  [ 68 631]]
# ----------------------
# Weighted label accuracy:  0.7631792376317924
# Weighted Label confusion matrix:
#  [[310 224]
#  [ 68 631]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff
# Raw label accuracy:  0.7555886736214605
# Raw Label confusion matrix:
#  [[161  66]
#  [ 98 346]]
# ----------------------
# Weighted label accuracy:  0.7555886736214605
# Weighted Label confusion matrix:
#  [[161  66]
#  [ 98 346]]
# ***********************
# ***********************
# Dataset:  Joystick
# Raw label accuracy:  0.7291139240506329
# Raw Label confusion matrix:
#  [[193 197]
#  [ 17 383]]
# ----------------------
# Weighted label accuracy:  0.7265822784810126
# Weighted Label confusion matrix:
#  [[155 209]
#  [  7 419]]
# ***********************

# 30s SVM RESULTS
# Headarray_A0_rawLabel top features:  ['RMSSD', 'SDNN', 'lfhf', 'lf', 'hf']
# Headarray_A0_weightedLabel top features:  ['RMSSD', 'SDNN', 'lfhf', 'lf', 'hf']
# Headarray_A1_rawLabel top features:  ['SDNN', 'SD2', 'RMSSD', 'hf', 'lfhf']
# Headarray_A1_weightedLabel top features:  ['SDNN', 'SD2', 'RMSSD', 'hf', 'lfhf']
# Headarray_A2_rawLabel top features:  ['SDNN', 'SD1SD2', 'SD1', 'hf', 'lfhf']
# Headarray_A2_weightedLabel top features:  ['SDNN', 'SD1SD2', 'SD1', 'hf', 'lfhf']
# Sip-n-puff_A0_rawLabel top features:  ['SDNN', 'SD1', 'SD1SD2', 'hf', 'lfhf']
# Sip-n-puff_A0_weightedLabel top features:  ['SDNN', 'SD1', 'SD1SD2', 'hf', 'lfhf']
# Joystick_A0_rawLabel top features:  ['SDNN', 'RMSSD', 'lfhf', 'SD1SD2', 'hf']
# Joystick_A0_weightedLabel top features:  ['SDNN', 'RMSSD', 'lfhf', 'SD1SD2', 'hf']
# Joystick_A1_rawLabel top features:  ['SD2', 'hf', 'SDNN', 'RMSSD', 'SD1SD2']
# Joystick_A1_weightedLabel top features:  ['hf', 'RMSSD', 'SD1SD2', 'SDNN', 'ulf']
# Joystick_A2_rawLabel top features:  ['SD1SD2', 'SDNN', 'hf', 'RMSSD', 'lfhf']
# Joystick_A2_weightedLabel top features:  ['SD1SD2', 'SDNN', 'hf', 'RMSSD', 'lfhf']
# Headarray_rawLabel top features:  ['RMSSD', 'hf', 'SDNN', 'lfhf', 'ApEn']
# Headarray_weightedLabel top features:  ['RMSSD', 'hf', 'SDNN', 'lfhf', 'ApEn']
# Sip-n-puff_rawLabel top features:  ['SDNN', 'SD1', 'SD1SD2', 'hf', 'lfhf']
# Sip-n-puff_weightedLabel top features:  ['SDNN', 'SD1', 'SD1SD2', 'hf', 'lfhf']
# Joystick_rawLabel top features:  ['SD1SD2', 'hf', 'SD2', 'RMSSD', 'SDNN']
# Joystick_weightedLabel top features:  ['SDNN', 'SD2', 'hf', 'RMSSD', 'ApEn']
# ***********************
# Datasets separated by autonomy and interface
# ***********************
# Dataset:  Headarray_A0
# Raw label accuracy:  0.8700564971751412
# Raw Label confusion matrix:
#  [[102   6]
#  [ 17  52]]
# ----------------------
# Weighted label accuracy:  0.8700564971751412
# Weighted Label confusion matrix:
#  [[102   6]
#  [ 17  52]]
# ***********************
# ***********************
# Dataset:  Headarray_A1
# Raw label accuracy:  0.8583333333333333
# Raw Label confusion matrix:
#  [[197  20]
#  [ 31 112]]
# ----------------------
# Weighted label accuracy:  0.8583333333333333
# Weighted Label confusion matrix:
#  [[197  20]
#  [ 31 112]]
# ***********************
# ***********************
# Dataset:  Headarray_A2
# Raw label accuracy:  0.9612244897959183
# Raw Label confusion matrix:
#  [[100  16]
#  [  3 371]]
# ----------------------
# Weighted label accuracy:  0.9612244897959183
# Weighted Label confusion matrix:
#  [[100  16]
#  [  3 371]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff_A0
# Raw label accuracy:  0.8714788732394366
# Raw Label confusion matrix:
#  [[186  12]
#  [ 61 309]]
# ----------------------
# Weighted label accuracy:  0.8714788732394366
# Weighted Label confusion matrix:
#  [[186  12]
#  [ 61 309]]
# ***********************
# ***********************
# Dataset:  Joystick_A0
# Raw label accuracy:  0.974025974025974
# Raw Label confusion matrix:
#  [[97  3]
#  [ 1 53]]
# ----------------------
# Weighted label accuracy:  0.974025974025974
# Weighted Label confusion matrix:
#  [[97  3]
#  [ 1 53]]
# ***********************
# ***********************
# Dataset:  Joystick_A1
# Raw label accuracy:  1.0
# Raw Label confusion matrix:
#  [[109   0]
#  [  0  34]]
# ----------------------
# Weighted label accuracy:  0.958041958041958
# Weighted Label confusion matrix:
#  [[90  1]
#  [ 5 47]]
# ***********************
# ***********************
# Dataset:  Joystick_A2
# Raw label accuracy:  0.9739130434782609
# Raw Label confusion matrix:
#  [[122   1]
#  [  8 214]]
# ----------------------
# Weighted label accuracy:  0.9739130434782609
# Weighted Label confusion matrix:
#  [[122   1]
#  [  8 214]]
# ***********************
# ***********************
# Datasets separated by interface only
# ***********************
# Dataset:  Headarray
# Raw label accuracy:  0.8312195121951219
# Raw Label confusion matrix:
#  [[330 110]
#  [ 63 522]]
# ----------------------
# Weighted label accuracy:  0.8312195121951219
# Weighted Label confusion matrix:
#  [[330 110]
#  [ 63 522]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff
# Raw label accuracy:  0.8714788732394366
# Raw Label confusion matrix:
#  [[186  12]
#  [ 61 309]]
# ----------------------
# Weighted label accuracy:  0.8714788732394366
# Weighted Label confusion matrix:
#  [[186  12]
#  [ 61 309]]
# ***********************
# ***********************
# Dataset:  Joystick
# Raw label accuracy:  0.7737909516380655
# Raw Label confusion matrix:
#  [[208 124]
#  [ 21 288]]
# ----------------------
# Weighted label accuracy:  0.7769110764430577
# Weighted Label confusion matrix:
#  [[180 134]
#  [  9 318]]
# ***********************

# 60s SVM RESULTS
# Headarray_A0_rawLabel top features:  ['SDNN', 'SD1SD2', 'RMSSD', 'lfhf', 'lf']
# Headarray_A0_weightedLabel top features:  ['SDNN', 'SD1SD2', 'RMSSD', 'lfhf', 'lf']
# Headarray_A1_rawLabel top features:  ['hf', 'RMSSD', 'lf', 'SDNN', 'lfhf']
# Headarray_A1_weightedLabel top features:  ['hf', 'RMSSD', 'lf', 'SDNN', 'lfhf']
# Headarray_A2_rawLabel top features:  ['SDNN', 'RMSSD', 'hf', 'SD2', 'lfhf']
# Headarray_A2_weightedLabel top features:  ['SDNN', 'RMSSD', 'hf', 'SD2', 'lfhf']
# Sip-n-puff_A0_rawLabel top features:  ['SD1', 'SDNN', 'SD1SD2', 'hf', 'lfhf']
# Sip-n-puff_A0_weightedLabel top features:  ['SD1', 'SDNN', 'SD1SD2', 'hf', 'lfhf']
# Joystick_A0_rawLabel top features:  ['lfhf', 'RMSSD', 'SDNN', 'SD1SD2', 'vlf']
# Joystick_A0_weightedLabel top features:  ['lfhf', 'RMSSD', 'SDNN', 'SD1SD2', 'vlf']
# Joystick_A1_rawLabel top features:  ['SD2', 'RMSSD', 'SDNN', 'hf', 'lf']
# Joystick_A1_weightedLabel top features:  ['SD1SD2', 'hf', 'SDNN', 'RMSSD', 'lfhf']
# Joystick_A2_rawLabel top features:  ['SD1SD2', 'SDNN', 'hf', 'RMSSD', 'lfhf']
# Joystick_A2_weightedLabel top features:  ['SD1SD2', 'SDNN', 'hf', 'RMSSD', 'lfhf']
# Headarray_rawLabel top features:  ['hf', 'lf', 'RMSSD', 'SDNN', 'lfhf']
# Headarray_weightedLabel top features:  ['hf', 'lf', 'RMSSD', 'SDNN', 'lfhf']
# Sip-n-puff_rawLabel top features:  ['SD1', 'SDNN', 'SD1SD2', 'hf', 'lfhf']
# Sip-n-puff_weightedLabel top features:  ['SD1', 'SDNN', 'SD1SD2', 'hf', 'lfhf']
# Joystick_rawLabel top features:  ['SDNN', 'lf', 'hf', 'RMSSD', 'SD1SD2']
# Joystick_weightedLabel top features:  ['lfhf', 'hf', 'lf', 'RMSSD', 'SDNN']
# ***********************
# Datasets separated by autonomy and interface
# ***********************
# Dataset:  Headarray_A0
# Raw label accuracy:  0.9042553191489362
# Raw Label confusion matrix:
#  [[154  11]
#  [ 16 101]]
# ----------------------
# Weighted label accuracy:  0.9042553191489362
# Weighted Label confusion matrix:
#  [[154  11]
#  [ 16 101]]
# ***********************
# ***********************
# Dataset:  Headarray_A1
# Raw label accuracy:  0.9319526627218935
# Raw Label confusion matrix:
#  [[382  10]
#  [ 36 248]]
# ----------------------
# Weighted label accuracy:  0.9319526627218935
# Weighted Label confusion matrix:
#  [[382  10]
#  [ 36 248]]
# ***********************
# ***********************
# Dataset:  Headarray_A2
# Raw label accuracy:  0.9820627802690582
# Raw Label confusion matrix:
#  [[196  11]
#  [  5 680]]
# ----------------------
# Weighted label accuracy:  0.9820627802690582
# Weighted Label confusion matrix:
#  [[196  11]
#  [  5 680]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff_A0
# Raw label accuracy:  0.9378585086042065
# Raw Label confusion matrix:
#  [[381   8]
#  [ 57 600]]
# ----------------------
# Weighted label accuracy:  0.9378585086042065
# Weighted Label confusion matrix:
#  [[381   8]
#  [ 57 600]]
# ***********************
# ***********************
# Dataset:  Joystick_A0
# Raw label accuracy:  1.0
# Raw Label confusion matrix:
#  [[158   0]
#  [  0 104]]
# ----------------------
# Weighted label accuracy:  1.0
# Weighted Label confusion matrix:
#  [[158   0]
#  [  0 104]]
# ***********************
# ***********************
# Dataset:  Joystick_A1
# Raw label accuracy:  1.0
# Raw Label confusion matrix:
#  [[154   0]
#  [  0  65]]
# ----------------------
# Weighted label accuracy:  1.0
# Weighted Label confusion matrix:
#  [[141   0]
#  [  0  78]]
# ***********************
# ***********************
# Dataset:  Joystick_A2
# Raw label accuracy:  1.0
# Raw Label confusion matrix:
#  [[234   0]
#  [  0 388]]
# ----------------------
# Weighted label accuracy:  1.0
# Weighted Label confusion matrix:
#  [[234   0]
#  [  0 388]]
# ***********************
# ***********************
# Datasets separated by interface only
# ***********************
# Dataset:  Headarray
# Raw label accuracy:  0.8875067604110329
# Raw Label confusion matrix:
#  [[662 102]
#  [106 979]]
# ----------------------
# Weighted label accuracy:  0.8875067604110329
# Weighted Label confusion matrix:
#  [[662 102]
#  [106 979]]
# ***********************
# ***********************
# Dataset:  Sip-n-puff
# Raw label accuracy:  0.9378585086042065
# Raw Label confusion matrix:
#  [[381   8]
#  [ 57 600]]
# ----------------------
# Weighted label accuracy:  0.9378585086042065
# Weighted Label confusion matrix:
#  [[381   8]
#  [ 57 600]]
# ***********************
# ***********************
# Dataset:  Joystick
# Raw label accuracy:  0.8647912885662432
# Raw Label confusion matrix:
#  [[440 106]
#  [ 43 513]]
# ----------------------
# Weighted label accuracy:  0.9083484573502723
# Weighted Label confusion matrix:
#  [[473  60]
#  [ 41 528]]
# ***********************


# # SVM Classifier
# clf = svm.SVC()
# # SVM Hyperparamater tuning and cross-validation 
# print("Cross Validation...")
# # params_dist = { 'C': loguniform(1e2,1e3),
# #                 'gamma': [loguniform(1e-4,1e-3),'auto','scale'],
# #                 'kernel': ['linear','rbf']
# # }
# params_dist = {'kernel':['linear','rbf'],
#                 'C':loguniform(1e2,1e3)}
# r_search = RandomizedSearchCV(clf,params_dist,n_iter=10,cv=5,random_state=0)
# # w_search = RandomizedSearchCV(clf,params_dist,n_iter=10,cv=5,n_jobs=-1,random_state=0)
# r_search.fit(rX_train,ry_train)
# w_search.fit(wX_train,wy_train)
# # Use the best parameters from Cross-Validation 
# r_params = r_search['best_params']
# w_params = w_search['best_params']
# r_clf = svm.SVC(**r_params)
# w_clf = svm.SVC(**w_params)
# r_clf = svm.SVC()
# w_clf = svm.SVC()
# # Training
# print("Training...")
# r_clf.fit(rX_train,ry_train)
# w_clf.fit(wX_train,wy_train)
# # Testing 
# print("Testing...")
# ry_pred = r_clf.predict(rX_test)
# wy_pred = w_clf.predict(wX_test)
# # Get accuracy scores
# r_acc = accuracy_score(ry_test,ry_pred)
# w_acc = accuracy_score(wy_test,wy_pred)
# # Get confusion matrices 
# r_cm = confusion_matrix(ry_test,ry_pred)
# w_cm = confusion_matrix(wy_test,wy_pred)
# print("Accuracy scores:")
# print("Raw TLX:",r_acc)
# print("Weighted TLX:",w_acc)
# print("Confusion Matrices:")
# print("Raw TLX CM")
# print(r_cm)
# print("Weigted TLX CM")
# print(w_cm)






############### 7/22 Testing ########################

# 3 labels - low/med/high, datasets separated by labels only, i.e. each labelled set contains
# all interfaces and autonomy levels

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