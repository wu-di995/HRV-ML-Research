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
import paths

def get_data(HRVpaths, TLX_levels):
    for i,path in enumerate(HRVpaths):
        HRVdf = pd.read_csv(path)
        # Do not include the last 4 columns and first column
        colData = HRVdf.iloc[:,1:-5].values
        if TLX_levels == "2":
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
        elif TLX_levels == "3":
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
    try:    
        rLow_ar.shape
    except UnboundLocalError:
        rLow_ar = []
    try:    
        rMed_ar.shape
    except UnboundLocalError:
        rMed_ar = []
    try:    
        rHigh_ar.shape
    except UnboundLocalError:
        rHigh_ar = []
    try:    
        wLow_ar.shape
    except UnboundLocalError:
        wLow_ar = []
    try:    
        wMed_ar.shape
    except UnboundLocalError:
        wMed_ar = []
    try:    
        wHigh_ar.shape
    except UnboundLocalError:
        wHigh_ar = []
    if TLX_levels == "2":
            return [rLow_ar,rHigh_ar,wLow_ar,wHigh_ar]
    elif TLX_levels == "3":
        return [rLow_ar,rMed_ar,rHigh_ar,wLow_ar,wMed_ar,wHigh_ar]

# Feature scale all datasets, 
def featureScale_dataDict(dataDict):
    sc_dataDict = {} # Scaled dataset dictionary 
    sc = StandardScaler()
    for key,datasetList in dataDict.items():
        if len(datasetList) == 4: # Low/High
            # Raw Labels
            r_datasetList = datasetList[:2]
            r_datasetList = [ar for ar in r_datasetList if len(ar)!=0]
            if len(r_datasetList)<2:
                rX,ry = [[], []] # Must have both labels 
            else:
                rX = np.vstack(r_datasetList)
                rX = sc.fit_transform(rX)
                ry = np.hstack( (np.zeros(len(r_datasetList[0])),np.ones(len(r_datasetList[1])) ))
                # print("rX:", rX.shape)
                # print("ry:", ry.shape)
            # Weighted Labels 
            w_datasetList = datasetList[2:]
            w_datasetList = [ar for ar in w_datasetList if len(ar)!=0]
            if len(w_datasetList)<2:
                wX, wy = [[], []] # Must have both labels
            else:
                wX = np.vstack(w_datasetList)
                wX = sc.fit_transform(wX)
                wy = np.hstack( (np.zeros(len(w_datasetList[0])),np.ones(len(w_datasetList[1])) ))
                # print("wX:", wX.shape)
                # print("wy:", wy.shape)
            
        
        elif len(datasetList) == 6: # Low/Med/High
            # Raw Labels
            r_datasetList = datasetList[:3]
            r_datasetList_sel = [ar for ar in r_datasetList if len(ar)!=0]
            if len(r_datasetList_sel)<2: # Must have at least 2 labels 
                rX, ry = [[], []]
            else:
                rX = np.vstack(r_datasetList_sel)
                rX = sc.fit_transform(rX)
                ry = np.array([0]*len(r_datasetList[0]) + [1]*len(r_datasetList[1]) + [2]*len(r_datasetList[2]))
                # print("rX:", rX.shape)
                # print("ry:", ry.shape)
            # Weighted Labels
            w_datasetList = datasetList[3:]
            w_datasetList_sel = [ar for ar in w_datasetList if len(ar)!=0]
            if len(w_datasetList_sel)<2: # Must have at least 2 labels 
                wX, wy = [[], []]
            else:
                wX = np.vstack(w_datasetList_sel)
                wX = sc.fit_transform(wX)
                wy = np.array([0]*len(w_datasetList[0]) + [1]*len(w_datasetList[1]) + [2]*len(w_datasetList[2]))
                # print("wX:", wX.shape)
                # print("wy:", wy.shape)
        else:
            print("Unexpected datasetList size")

        sc_dataDict[key] = [rX,ry,wX,wy]
        
    return sc_dataDict

def apply_sss(X,y):
    for train_index, test_index in sss.split(X,y):
            X_train, X_test = X[train_index],X[test_index]
            y_train, y_test = y[train_index],y[test_index]
    return X_train,X_test,y_train,y_test


# Calculate feature important and find top 5 features
def feat_impt_multiModels(X_train,y_train,X_test,models,featureNames,datasetName):
    #Chooses the top 5 features for the same dataset 
    featureRanks = [0]*len(featureNames)
    featureRanks_dict = dict(zip(featureNames, featureRanks))
    for i,model in enumerate(models):
        model.fit(X_train,y_train)
        importances = model.feature_importances_
        # featureScores = [featureScore + importance for featureScore,importance in zip(featureScores,importances)]
        # Sum up ranks 
        importances,ranked_names = zip(*sorted(zip(importances,featureNames))) 
        for rank, ranked_name in enumerate(ranked_names):
            featureRanks_dict[ranked_name] += rank
            # Higher number is more important 
    # featureScores,names = zip(*sorted(zip(featureScores,featureNames))) #Ascending order
    # featureScores = list(featureScores)
    # names = list(names)
    # Choose top 5 features 
    # print(featureScores)
    # print(names)
    # topFeatures = names[-5:]
    # Feature Ranks
    sorted_ranks_dict = dict(sorted(featureRanks_dict.items(), key = lambda item: item[1])) # Sorts features in ascending order 
    print(sorted_ranks_dict)
    rankedFeatures = list(sorted_ranks_dict.keys())# Get the feature names in ascending order
    topFeatures = rankedFeatures[-5:]
    # Check if RMSSD and SD1 are both selected, only choose the more important one, and add the 6th feature
    if ("RMSSD" in topFeatures) and ("SD1" in topFeatures) :
        if topFeatures.index("RMSSD") < topFeatures.index("SD1"):
            topFeatures.remove("SD1")
        else:
            topFeatures.remove("RMSSD")
        topFeatures.append(rankedFeatures[-6])
    topFeaturesMask = [i for i,feature in enumerate(featureNames) if feature in topFeatures]
    X_train_sel = X_train[:,topFeaturesMask]
    X_test_sel = X_test[:,topFeaturesMask]
    print(datasetName+" top features: ", topFeatures)
    return X_train_sel,X_test_sel


# Select features 
def selectFeatures(sc_dataDict,featureNames,selFeatMask,method):
    models = [ExtraTreesClassifier(random_state=0),DecisionTreeClassifier(random_state=0),XGBClassifier(random_state=0, label_encoder=False)]
    modelNames = ["ExtraTreesClassifier","DecisionTreeClassifer","XGBClassifier"]
    bf_dataDict = {}
    for key,datasetList in sc_dataDict.items():
        # Raw 
        rX = datasetList[0]
        ry = datasetList[1]
        if len(rX) == 0:
            rX_train,rX_test,ry_train,ry_test = [ [], [], [], [] ]
        else:
            # Split into training and test sets
            rX_train,rX_test,ry_train,ry_test = apply_sss(rX,ry)
            
            if method == "all":
                # (either) Select same features for all datasets
                rX_train,rX_test = [rX_train[:,selFeatMask],rX_test[:,selFeatMask]]
            elif method == "self":
                # (or) Select best features from the same dataset 
                num_classes = len(np.unique(np.array(ry_train)))
                if num_classes == 2:
                    models[-1] = XGBClassifier(random_state=0, label_encoder=False, objective="binary:logistic")
                rX_train,rX_test = feat_impt_multiModels(rX_train,ry_train,rX_test,models,featureNames,key+"_rawLabel")
        # Weighted 
        wX = datasetList[2]
        wy = datasetList[3]
        if len(wX) == 0:
            wX_train,wX_test,wy_train,wy_test = [ [], [], [], [] ]
        else:
            # Split into training and test sets
            wX_train,wX_test,wy_train,wy_test = apply_sss(wX,wy)
            if method == "all":
                # (either) Select same features for all datasets
                wX_train,wX_test = [wX_train[:,selFeatMask],wX_test[:,selFeatMask]]
            elif method == "self":
                # (or) Select best features from the same dataset 
                wX_train,wX_test = feat_impt_multiModels(wX_train,wy_train,wX_test,models,featureNames,key+"_weightedLabel")
        # Fill in dataset dictionary 
        bf_dataDict[key] = [rX_train,rX_test,ry_train,ry_test,wX_train,wX_test,wy_train,wy_test]
    return bf_dataDict

# SVM Classifier 
def applySVM(dataDict):
    clf = svm.SVC()
    for key, datasetList in dataDict.items():
        rX_train,rX_test,ry_train,ry_test,wX_train,wX_test,wy_train,wy_test = datasetList
        # Raw Label Classification
        if len(rX_train) == 0:
            print(f"{key}, raw score, not enough classes")
        else:
            clf.fit(rX_train,ry_train)
            ry_pred = clf.predict(rX_test)
            r_acc = accuracy_score(ry_test,ry_pred)
            r_cm = confusion_matrix(ry_test,ry_pred)
            print("***********************")
            print("Dataset: ",key)
            print("Raw label accuracy: ",r_acc)
            print("Raw Label confusion matrix: \n",r_cm)
            print("----------------------")
        # Weighted Label Classification
        if len(wX_train) == 0:
            print(f"{key}, weighted score, not enough classes")
        else:
            clf.fit(wX_train,wy_train)
            wy_pred = clf.predict(wX_test)
            w_acc = accuracy_score(wy_test,wy_pred)
            w_cm = confusion_matrix(wy_test,wy_pred)
            # Print Results 
            print("Weighted label accuracy: ",w_acc)
            print("Weighted Label confusion matrix: \n",w_cm)
            print("***********************")


if __name__ == "__main__":
    # Combine data by labels, 
    # 1) All interfaces, all autonomy levels
    # 2) Separate by interface and autonomy
    # 3) Separate by interface 

    featureNames = ['SDNN','RMSSD','ulf','vlf','lf','hf','lfhf','SD1','SD2','SD1SD2','ApEn']
    # OLD Selected features: SDNN, hf, RMSSD, ApEn, SD1SD2
    # CURRENT Selected features" RMSSD, lf, hf, lfhf, ApEn
    selFeatMask = [1,4,5,6,-1]
    interfaces = ["HA", "SNP", "JOY"]
    autonomies = ["A0", "A1", "A2"]
    int_autoList = [interface+"_"+autonomy for interface in interfaces for autonomy in autonomies]
    int_autoList = [i for i in int_autoList if (("SNP_A1" not in i) and ("SNP_A2" not in i))] # No SNP_A1 and SNP_A2

    int_auto_datasets_2_dict = {}
    int_datasets_2_dict = {}
    int_auto_datasets_3_dict = {}
    int_datasets_3_dict = {}

    wins = ["30s", "60s"]
    HRV_extr_dir = paths.HRV_byEvent_TLX_path

    for win in wins:
        HRV_2_pathsList = glob.glob(HRV_extr_dir+win+os.sep+"2"+os.sep+"*.csv")
        HRV_3_pathsList = glob.glob(HRV_extr_dir+win+os.sep+"3"+os.sep+"*.csv")

        # Create dictionary of datsets, where data is separated by interface, autonomy and label 
        for int_auto in int_autoList:
            # 2 levels
            HRV_2_paths = [path for path in HRV_2_pathsList if int_auto in path]
            int_auto_datasets_2_dict[int_auto] = get_data(HRV_2_paths, "2")
            # 3 levels
            HRV_3_paths = [path for path in HRV_3_pathsList if int_auto in path]
            int_auto_datasets_3_dict[int_auto] = get_data(HRV_3_paths, "3")

        # Create dictionary of datasets, where data is separated by interface and label 
        for interface in interfaces:
            # 2 levels
            HRV_2_paths = [path for path in HRV_2_pathsList if interface in path]
            int_datasets_2_dict[interface] = get_data(HRV_2_paths, "2")
            # 3 levels
            HRV_3_paths = [path for path in HRV_3_pathsList if interface in path]
            int_datasets_3_dict[interface] = get_data(HRV_3_paths, "3")

        # Feature scale datasets 
        sc_int_auto_2_datasets = featureScale_dataDict(int_auto_datasets_2_dict)
        sc_int_2_datasets = featureScale_dataDict(int_datasets_2_dict)
        sc_int_auto_3_datasets = featureScale_dataDict(int_auto_datasets_3_dict)
        sc_int_3_datasets = featureScale_dataDict(int_datasets_3_dict)

        # Split into training and test sets
        sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)

        # Datasets with best features (bf)
        self_int_auto_2_dataDict  = selectFeatures(sc_int_auto_2_datasets,featureNames,selFeatMask,"self")
        self_int_2_dataDict  = selectFeatures(sc_int_2_datasets,featureNames,selFeatMask,"self")
        self_int_auto_3_dataDict  = selectFeatures(sc_int_auto_3_datasets,featureNames,selFeatMask,"self")
        self_int_3_dataDict  = selectFeatures(sc_int_3_datasets,featureNames,selFeatMask,"self")

        all_int_auto_2_dataDict  = selectFeatures(sc_int_auto_2_datasets,featureNames,selFeatMask,"all")
        all_int_2_dataDict  = selectFeatures(sc_int_2_datasets,featureNames,selFeatMask,"all")
        all_int_auto_3_dataDict  = selectFeatures(sc_int_auto_3_datasets,featureNames,selFeatMask,"all")
        all_int_3_dataDict  = selectFeatures(sc_int_3_datasets,featureNames,selFeatMask,"all")

        print("***********************Using same features for all **********************************")
        print("Datasets separated by autonomy and interface, 2 levels, all")
        applySVM(all_int_auto_2_dataDict)
        print("***********************")
        print("Datasets separated by interface only, 2 levels, all")
        applySVM(all_int_2_dataDict)
        print("Datasets separated by autonomy and interface, 3 levels, all")
        applySVM(all_int_auto_3_dataDict)
        print("***********************")
        print("Datasets separated by interface only, 3 levels, all")
        applySVM(all_int_3_dataDict)

        print("***********************Using best features for single dataset **********************************")
        print("Datasets separated by autonomy and interface, 2 levels, self")
        applySVM(self_int_auto_2_dataDict)
        print("***********************")
        print("Datasets separated by interface only, 2 levels, self")
        applySVM(self_int_2_dataDict)
        print("Datasets separated by autonomy and interface, 3 levels, self")
        applySVM(self_int_auto_3_dataDict)
        print("***********************")
        print("Datasets separated by interface only, 3 levels, self")
        applySVM(self_int_3_dataDict)
