# Testing GMM models 

# Import necessary libraries
import pandas as pd
import numpy as np
import glob,os, pathlib
from sklearn import svm 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from scipy.stats import loguniform
import re
from sklearn.mixture import GaussianMixture

cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
HRV_pathsList = glob.glob(str(mainDir)+"\\HRV_multiSubj\\Extracted-with_tlx_labels\\30s\\*.csv")

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

# Select Features
rX_select = [9,6,5,4,1]
wX_select = [10,9,7,5,0]
rX = rX[:,rX_select]
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

# Try GMMs using different types of covariances.
n_classes = 3
estimators = {cov_type: GaussianMixture(n_components=n_classes,
              covariance_type=cov_type, max_iter=30, random_state=0)
              for cov_type in ['spherical', 'diag', 'tied', 'full']}

n_estimators = len(estimators)

X_train = wX_train
X_test = wX_test
y_train = wy_train
y_test = wy_test


for index, (name, estimator) in enumerate(estimators.items()):
    print(name)
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[ry_train == i].mean(axis=0)
                                    for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel())
    print("Train accuracy: ", train_accuracy)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) 
    print("Test accuracy: ", test_accuracy)


# 5s
# Raw
# spherical
# Train accuracy:,  0.4070435115477861
# Test accuracy:,  0.41260525644297014
# diag
# Train accuracy:,  0.4678448385861937
# Test accuracy:,  0.4679765246236285
# tied
# Train accuracy:,  0.3981115222661733
# Test accuracy:,  0.4021434039295739
# full
# Train accuracy:,  0.46790863850963377
# Test accuracy:,  0.4679765246236285
# Weighted
# spherical
# Train accuracy:,  0.20524435370677555
# Test accuracy:,  0.20719571319214086
# diag
# Train accuracy:,  0.3568967717238739
# Test accuracy:,  0.35723398826231184
# tied
# Train accuracy:,  0.13614903662115604
# Test accuracy:,  0.13600408267415157
# full
# Train accuracy:,  0.35683297180043383
# Test accuracy:,  0.3574891553967849

# 10s
# Raw
# spherical
# Train accuracy:, %f 0.387630014858841
# Test accuracy:, %f 0.39561664190193163
# diag
# Train accuracy:, %f 0.21471025260029716
# Test accuracy:, %f 0.21508172362555722
# tied
# Train accuracy:, %f 0.3639487369985141
# Test accuracy:, %f 0.3803863298662704
# full
# Train accuracy:, %f 0.21536032689450224
# Test accuracy:, %f 0.21099554234769688
# Weighted
# spherical
# Train accuracy:  0.41781203566121844
# Test accuracy:  0.4112184249628529
# diag
# Train accuracy:  0.4973997028231798
# Test accuracy:  0.4988855869242199
# tied
# Train accuracy:  0.1150631500742942
# Test accuracy:  0.1151560178306092
# full
# Train accuracy:  0.46823922734026746
# Test accuracy:  0.4684249628528975

# 30s 
# Raw
# spherical
# Train accuracy:, %f 0.49339453649798476
# Test accuracy:, %f 0.5
# diag
# Train accuracy:, %f 0.5139946260635916
# Test accuracy:, %f 0.5228290062667861
# tied
# Train accuracy:, %f 0.4965293327362293
# Test accuracy:, %f 0.4923903312444047
# full
# Train accuracy:, %f 0.4907075682937752
# Test accuracy:, %f 0.5013428827215757
# Weighted
# spherical
# Train accuracy:  0.14274518584863413
# Test accuracy:  0.13428827215756492
# diag
# Train accuracy:  0.18360949395432155
# Test accuracy:  0.16696508504923904
# tied
# Train accuracy:  0.1639050604567846
# Test accuracy:  0.15443151298119964
# full
# Train accuracy:  0.26253918495297807
# Test accuracy:  0.2430617726051925

# 60s 
# Raw
# spherical
# Train accuracy:, %f 0.5251564455569462
# Test accuracy:, %f 0.5295295295295295
# diag
# Train accuracy:, %f 0.5379224030037547
# Test accuracy:, %f 0.5380380380380381
# tied
# Train accuracy:, %f 0.5007509386733416
# Test accuracy:, %f 0.501001001001001
# full
# Train accuracy:, %f 0.4773466833541927
# Test accuracy:, %f 0.4774774774774775
# Weighted
# spherical
# Train accuracy:, %f 0.3659574468085106
# Test accuracy:, %f 0.37737737737737737
# diag
# Train accuracy:, %f 0.1246558197747184
# Test accuracy:, %f 0.12462462462462462
# tied
# Train accuracy:, %f 0.396558197747184
# Test accuracy:, %f 0.3966466466466467
# full
# Train accuracy:, %f 0.17909887359198998
# Test accuracy:, %f 0.17917917917917917
