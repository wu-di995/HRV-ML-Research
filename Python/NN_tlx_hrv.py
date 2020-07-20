# Classify TLX labels using HRV
# Dense neural networks

# Import necessary libraries 
import pandas as pd
import numpy as np
import glob, pathlib, os, re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
HRV_pathsList = glob.glob(str(mainDir)+"\\HRV_multiSubj\\Extracted-with_tlx_labels\\*.csv")

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

print("Size of Feature and Label sets")
print(len(rX),len(ry))
print(len(wX),len(wy))

# Split into training and test sets
sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
def apply_sss(X,y):
    for train_index, test_index in sss.split(X,y):
            X_train, X_test = X[train_index],X[test_index]
            y_train, y_test = y[train_index],y[test_index]
    return X_train,X_test,y_train,y_test

rX_train,rX_test,ry_train,ry_test = apply_sss(rX,ry)
wX_train,wX_test,wy_train,wy_test = apply_sss(wX,wy)

# Reshape into tensor 
# rX_train = rX_train.reshape(rX_train.shape[0],rX_train.shape[1],1)
# wX_train = wX_train.reshape(wX_train.shape[0],wX_train.shape[1],1)


# DNN Model
def DNN_model(n_features):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_dim=n_features, activation="relu"),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(10, activation="relu"), 
        tf.keras.layers.Dense(3,activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                optimizer='adam',
                metrics=["accuracy"])
    # model.summary()
    return model

def test(model,dataset,devset,batchsize,epochs):
    tf.random.set_seed(10)
    datasetb = dataset.batch(batchsize)
    devsetb = devset.batch(batchsize)
    history = model.fit(datasetb,epochs=epochs,verbose=0,validation_data=devsetb)
    return history 

DNN = DNN_model(11)

# Create tf dataset
r_dataset = tf.data.Dataset.from_tensor_slices((rX_train,ry_train))
r_testset = tf.data.Dataset.from_tensor_slices((rX_test,ry_test))

w_dataset = tf.data.Dataset.from_tensor_slices((wX_train,wy_train))
w_testset = tf.data.Dataset.from_tensor_slices((wX_test,wy_test))

# Testing
batchsize = 40
epochs = 10
r_history = test(DNN,r_dataset,r_testset,batchsize,epochs)
w_history = test(DNN,w_dataset,w_testset,batchsize,epochs)

# Max validation accuracy 
r_valacc = max(r_history.history['val_accuracy'])
w_valacc = max(w_history.history['val_accuracy'])
print(r_valacc) #0.5389130115509033
print(w_valacc) #0.6820617318153381