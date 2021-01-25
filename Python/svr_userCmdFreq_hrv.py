# Support vector regression, trajectory features as targets, HRV as features 

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score, r2_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression,mutual_info_regression
import pandas as pd
import numpy as np
import glob,os, pathlib
import matplotlib.pyplot as plt 
from sklearn import svm 
import paths

# Read event name from a path
def readEvent(path):
    filenameList = path.split(os.sep)[-1].split("_")
    event = filenameList[0].lower()+"_"+filenameList[1]+"_"+filenameList[2]
    return event

# Create dataset, grouped by interfaces, 9 features, target as the 10th column 
def mk_dataset(targetPaths, hrvPaths, userCtrlPaths, hrv_metrics_list, target_cols_idx_list):
    HA_counter = 0
    JOY_counter = 0
    SNP_counter = 0
    for i, targetPath in enumerate(targetPaths):
        print(targetPath)
        event = readEvent(targetPath)
        interface = event.split("_")[1]
        # Load target dataframe
        target_df = pd.read_csv(targetPath)
        # Load HRV dataframe
        hrv_path = [path for path in hrvPaths if event in path][0]
        print(hrv_path)
        hrv_df = pd.read_csv(hrv_path)
        # Load user controlled dataframe
        userCtrl_path = [path for path in userCtrlPaths if event in path][0]
        print(userCtrl_path)
        userCtrl_df = pd.read_csv(userCtrl_path)
        # Get indices userCtrl where "User Controlled == 1"
        userCtrl_idx = userCtrl_df[(userCtrl_df["User Controlled"] == 1)].iloc[:,0].to_list()
        target_len = target_df.shape[0]
        userCtrl_len = len(userCtrl_idx)
        # print(userCtrl_idx)
        # print("User controlled: "+ str(userCtrl_len))
        # Get indices for HRV times where "NNMean" exists 
        hrv_existIdx = hrv_df[~(hrv_df["NNmean"].isnull())].index.to_list()
        if len(hrv_existIdx) == 0:
            print("No HRV metrics")
            continue
        # print(hrv_existIdx)
        # print("HRV exist: "+str(len(hrv_existIdx)))
        # Get indices that fulfill both conditions 
        ctrlHRVIdx = list(set(userCtrl_idx) & set(hrv_existIdx))
        if len(ctrlHRVIdx) == 0:
            print("No matching userCtrl and HRV times")
            continue 
        # print(featuresIdx)
        # print("Features number: "+str(len(featuresIdx)))
        # Start time exist in target datframe
        target_existIdx = target_df[~target_df["Start Win Time"].isnull()].index.to_list()
        # Convert ctrlHRVIdx to indices matching indices in target dataframe indices, and must be non null in target df
        targetIdx = target_df[(target_df.iloc[:,0].isin(ctrlHRVIdx))].index.to_list() # These indices are aligned with target_df
        targetIdx = list(set(targetIdx) & set(target_existIdx))
        # Refine featuresIdx to only those that exist in the target dataframe and must be non null in target df
        target_existIdx_featureRef = target_df[~target_df["Start Win Time"].isnull()].iloc[:,0].to_list() # These indices are aligned with HRV
        featuresIdx = list(set(ctrlHRVIdx) & set(target_existIdx_featureRef))
        # Select Target and HRV metrics
        hrv_features = hrv_df.loc[featuresIdx, hrv_metrics_list].values
        target_sel = target_df.iloc[targetIdx, target_cols_idx_list].values
        # print("HRV/Target lengths:" +str(len(featuresIdx))+" " + str(len(targetIdx)))
        if len(featuresIdx) != len(targetIdx):
            print(featuresIdx)
            print(targetIdx)
        print("HRV shape: "+ str(hrv_features.shape))
        print("Target shape: " +str(target_sel.shape))
        if len(target_cols_idx_list) == 1:
            target_sel = target_sel.reshape(-1,1)
        data = np.hstack((hrv_features,target_sel))
        # print(data.shape)
        # Append to dataset, if already exists, otherwise, create the dataset 
        if interface == "HA":
            HA_counter+=1
            try:
                HA_dataset = np.vstack((HA_dataset,data))
            except NameError:
                HA_dataset = data
        elif interface == "JOY":
            JOY_counter+=1
            try:
                JOY_dataset = np.vstack((JOY_dataset,data))
            except NameError:
                JOY_dataset = data
        elif interface == "SNP":
            SNP_counter+=1
            try:
                SNP_dataset = np.vstack((SNP_dataset,data))
            except NameError:
                SNP_dataset = data
    print(HA_counter,JOY_counter,SNP_counter)
    # Check sizes of each dataset before returning datasets
    # Standardize features 
    sc = StandardScaler()
    if HA_counter:
        HA_dataset[:,:9] = sc.fit_transform(HA_dataset[:,:9])
    else:
        HA_dataset = np.zeros(9)
    if JOY_counter:
        JOY_dataset[:,:9] = sc.fit_transform(JOY_dataset[:,:9])
    else:
        JOY_dataset = np.zeros(9)
    if SNP_counter:
        SNP_dataset[:,:9] = sc.fit_transform(SNP_dataset[:,:9])
    else:
        SNP_dataset = np.zeros(9)
    return (HA_dataset, JOY_dataset, SNP_dataset)


# Support Vector Regression
def apply_SVR_interfaces(dataset,no_features,target_col_idx):
    interfaces = ["HA","JOY","SNP"]
    r2_scores = []
    for i in range(3):
        # print("-----"+interfaces[i]+"-----")
        print(f"-----{interfaces[i], i}-----")
        if np.array_equal(dataset[i],np.zeros(9)):
            r2_scores.append("NA")
            print("No Data")
            continue
        # Separate features and targets
        X = dataset[i][:,:9]
        y = dataset[i][:,target_col_idx]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        # sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        # for train_index, test_index in sss.split(X,y):
        #     X_train, X_test = X[train_index],X[test_index]
        #     y_train, y_test = y[train_index],y[test_index]
        # f_selector = SelectKBest(score_func=f_regression, k=3)
        f_selector = SelectKBest(score_func=mutual_info_regression,k=no_features)
        # learn relationship from training data
        f_selector.fit(X_train, y_train)
        # transform train input data
        X_train_fs = f_selector.transform(X_train)
        # transform test input data
        X_test_fs = f_selector.transform(X_test)
        # Get the features that were selected 
        features_selected_mask = f_selector.get_support(indices=True)
        features_selected = [hrv_metrics_list[i] for i in features_selected_mask]
        print(f"Features selected: {features_selected}")
        # print(X_test_fs.shape)
        # SVR models
        svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        # svr_lin = SVR(kernel='linear', C=100, gamma='auto')
        # svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
        #            coef0=1)
        # Predictions
        ypred_rbf = svr_rbf.fit(X_train_fs,y_train).predict(X_test_fs) 
        # ypred_lin = svr_lin.fit(X_train_fs,y_train).predict(X_test_fs) 
        # ypred_poly = svr_poly.fit(X_train_fs,y_train).predict(X_test_fs) 
        # R^2 values 
        r2_rbf = r2_score(y_test,ypred_rbf)
        # r2_lin = r2_score(y_test,ypred_lin)
        # r2_poly = r2_score(y_test,ypred_poly)
        # MAE
        mse_rbf = mean_squared_error(y_test,ypred_rbf)
        # mse_lin = mean_squared_error(y_test,ypred_lin)
        # mse_poly = mean_squared_error(y_test,ypred_poly)

        print("R^2 values: ", r2_rbf)
        print("MSE values: ", mse_rbf)
        r2_scores.append(r2_rbf)
    return r2_scores

def apply_SVR_multiFeat(dataset,target_col_idx,export_results=False):
    if export_results:
        # Create dataframe
        results_df = pd.DataFrame(columns=range(1,len(hrv_metrics_list)+1),index=["HA","JOY","SNP"])
    for i in range(1,len(hrv_metrics_list)+1):
        print("No of features: "+str(i))
        r2_scores = apply_SVR_interfaces(dataset,i,target_col_idx)
        # results_df.loc[:,i] = r2_scores 

if __name__ == "__main__":
    # User Command Frequencies 
    # 30s
    userCmdFreqDir = paths.UserCmdFreqs_path
    userCmdFreqs30_aft_Paths = glob.glob(userCmdFreqDir + "*aft_30.csv")
    userCmdFreqs30_close_Paths = glob.glob(userCmdFreqDir + "*close_30.csv")
    # 60s
    userCmdFreqs60_aft_Paths = glob.glob(userCmdFreqDir + "*aft_60.csv")
    userCmdFreqs60_close_Paths = glob.glob(userCmdFreqDir + "*close_60.csv")

    # User Controlled status 
    # 30s
    userCtrlDir = paths.UserCtrl_path
    userCtrl30_aft_Paths = glob.glob(userCtrlDir + "*aft_30.csv")
    userCtrl30_close_Paths = glob.glob(userCtrlDir + "*close_30.csv")
    # 60s
    userCtrl60_aft_Paths = glob.glob(userCtrlDir + "*aft_60.csv")
    userCtrl60_close_Paths = glob.glob(userCtrlDir + "*close_60.csv")

    # HRV files 
    HRVDir = paths.HRV_byEvent_path 
    hrv30Paths = glob.glob(HRVDir + "30s" + os.sep + "*.csv")
    hrv60Paths = glob.glob(HRVDir + "60s" + os.sep + "*.csv")

    # HRV metrics -- use as features (9)
    hrv_metrics_list = ['SDNN','RMSSD','lf','hf','lfhf','SD1','SD2','SD1SD2','ApEn']

    # Create datasets 
    print("Creating datasets...")
    # User Cmd Freqs (mvAvg, peakFreq, totalPower)
    print("userCmdFreqs30_aft_dataset")
    userCmdFreqs30_aft_dataset = mk_dataset(userCmdFreqs30_aft_Paths,hrv30Paths,userCtrl30_aft_Paths,hrv_metrics_list,[-3,-2,-1])
    print("userCmdFreqs30_close_dataset")
    userCmdFreqs30_close_dataset = mk_dataset(userCmdFreqs30_close_Paths,hrv30Paths,userCtrl30_close_Paths,hrv_metrics_list,[-3,-2,-1])
    print("userCmdFreqs60_aft_dataset")
    userCmdFreqs60_aft_dataset = mk_dataset(userCmdFreqs60_aft_Paths,hrv60Paths,userCtrl60_aft_Paths,hrv_metrics_list,[-3,-2,-1])
    print("userCmdFreqs60_close_dataset")
    userCmdFreqs60_close_dataset = mk_dataset(userCmdFreqs60_close_Paths,hrv60Paths,userCtrl60_close_Paths,hrv_metrics_list,[-3,-2,-1])

    # SVR Regression
    ## User Command Frequencies
    ### 30s 
    #### Moving Average  
    print("MvAvg UserCmd- 30s, aft")
    apply_SVR_multiFeat(userCmdFreqs30_aft_dataset,-3)
    print("MvAvg UserCmd- 30s, close")
    apply_SVR_multiFeat(userCmdFreqs30_close_dataset,-3)
    #### Peaks Frequency 
    print("PksFreq  UserCmd- 30s, aft")
    apply_SVR_multiFeat(userCmdFreqs30_aft_dataset,-2)
    print("PksFreq  UserCmd- 30s, close")
    apply_SVR_multiFeat(userCmdFreqs30_close_dataset,-2)
    #### Total Power
    print("TotalPower  UserCmd- 30s, aft")
    apply_SVR_multiFeat(userCmdFreqs30_aft_dataset,-1)
    print("TotalPower  UserCmd- 30s, close")
    apply_SVR_multiFeat(userCmdFreqs30_close_dataset,-1)
    ### 60s 
    #### Moving Average  
    print("MvAvg UserCmd- 60s, aft")
    apply_SVR_multiFeat(userCmdFreqs60_aft_dataset,-3)
    print("MvAvg UserCmd- 60s, close")
    apply_SVR_multiFeat(userCmdFreqs60_close_dataset,-3)
    #### Peaks Frequency 
    print("PksFreq  UserCmd- 60s, aft")
    apply_SVR_multiFeat(userCmdFreqs60_aft_dataset,-2)
    print("PksFreq  UserCmd- 60s, close")
    apply_SVR_multiFeat(userCmdFreqs60_close_dataset,-2)
    #### Total Power
    print("TotalPower  UserCmd- 60s, aft")
    apply_SVR_multiFeat(userCmdFreqs60_aft_dataset,-1)
    print("TotalPower  UserCmd- 60s, close")
    apply_SVR_multiFeat(userCmdFreqs60_close_dataset,-1)

