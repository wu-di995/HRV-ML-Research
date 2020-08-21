# Support vector regression on user input regression features 
# Types of features - Welch's method frequencies (total power), moving window frequency, find peaks frequency

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

# PSD files, Moving average files 
subjFolders = glob.glob("E:\\argall-lab-data\\Trajectory Data\\*\\")
subjFolders = [path for path in subjFolders if "U00" not in path]
teleopFolders = []
for subjFolder in subjFolders:
    trajFolders = glob.glob(subjFolder+"*\\")
    for trajFolder in trajFolders:
        if "A0" in trajFolder:
            teleopFolders.append(trajFolder)
psd30Paths = []
psd60Paths = []
mvAvg30Paths = []
mvAvg60Paths = []
pksFreq30Paths = []
pksFreq60Paths = []
for teleopFolder in teleopFolders:
    psd30Paths.append(glob.glob(teleopFolder+"*psd_30.csv")[0])
    psd60Paths.append(glob.glob(teleopFolder+"*psd_60.csv")[0])
    mvAvg30Paths.append(glob.glob(teleopFolder+"*_movingAvg_30.csv")[0])
    mvAvg60Paths.append(glob.glob(teleopFolder+"*_movingAvg_60.csv")[0])
    pksFreq30Paths.append(glob.glob(teleopFolder+"*pksFreq_30.csv")[0])
    pksFreq60Paths.append(glob.glob(teleopFolder+"*pksFreq_60.csv")[0])

# HRV files 
hrv30Paths = glob.glob("E:\\argall-lab-data\\HRV_byEvent\\30s\\*.csv")
hrv60Paths = glob.glob("E:\\argall-lab-data\\HRV_byEvent\\60s\\*.csv")

# HRV metrics -- use as features 
hrv_metrics_list = ['SDNN','RMSSD','lf','hf','lfhf','SD1','SD2','SD1SD2','ApEn']


# User controlled files 
userCtrl30Paths = glob.glob("E:\\argall-lab-data\\UserControlled_byEvent\\30s\\*.csv")
userCtrl60Paths = glob.glob("E:\\argall-lab-data\\UserControlled_byEvent\\60s\\*.csv")


# Create PSD dataset, grouped by interfaces 
def mk_PSD_dataset(psdPaths,hrvPaths,userCtrlPaths,hrv_metrics_list):
    for i,psdPath in enumerate(psdPaths):
        subj = psdPath.split("\\")[-1].split("_")[0]
        interface = psdPath.split("\\")[-1].split("_")[1]
        autonomy = psdPath.split("\\")[-1].split("_")[2]
        event = subj+"_"+interface+"_"+autonomy
        print(event)
        # Load PSD dataframe
        psd_df = pd.read_csv(psdPath)
        # Load HRV dataframe
        hrv_path = [path for path in hrvPaths if event in path][0]
        hrv_df = pd.read_csv(hrv_path)
        # Load user controlled dataframe
        userCtrl_path = [path for path in userCtrlPaths if event in path][0]
        userCtrl_df = pd.read_csv(userCtrl_path)
        # print(userCtrl_df.shape)
        # print(psd_df.shape)
        # Get indices for PSD times where "User Controlled == 1"
        userCtrl_idx = (userCtrl_df["User Controlled"] == 1).tolist()
        psd_len = psd_df.shape[0]
        userCtrl_len = len(userCtrl_idx)
        # print(userCtrl_len)
        if psd_len < userCtrl_len:
            userCtrl_idx = userCtrl_idx[:psd_len]
        elif psd_len > userCtrl_len:
            print("b")
            for i in range(psd_len-userCtrl_len):
                userCtrl_idx.append(False)
        # print(len(userCtrl_idx))
        # print(userCtrl_idx)
        # print(psd_len)
        psd_userControlIdx = psd_df.iloc[userCtrl_idx].index.to_list()
        # psd_userControlIdx= psd_df[userCtrl_df["User Controlled"] == 1].index.to_list()
        """psd_df and userCtrl_df have different number of rows """
        # Get indices for HRV times where "NNMean" exists 
        hrv_existIdx = hrv_df[~(hrv_df["NNmean"].isnull())].index.to_list()
        # Get indices that fulfill both conditions 
        featuresIdx = list(set(psd_userControlIdx) & set(hrv_existIdx))
        # If no HRV metrics exist, continue to next loop 
        if len(featuresIdx) == 0:
            continue 
        # print(len(featuresIdx))
        # Select PSD and HRV metrics 
        hrv_features = hrv_df.loc[featuresIdx, hrv_metrics_list].values
        psd_selected = psd_df.iloc[featuresIdx,1:].values
        # psd_totalPower =  np.zeros((psd_selected.shape[0],1))
        psd_totalPower = np.apply_along_axis(lambda x: np.trapz(x), axis=1, arr=psd_selected)
        psd_totalPower = psd_totalPower.reshape(-1,1)
        # print(psd_selected.shape)
        # print(psd_totalPower.shape)
        data = np.hstack((hrv_features,psd_totalPower))
        # Append to dataset, if already exists, otherwise, create the dataset 
        if interface == "HA":
            try:
                HA_dataset = np.vstack((HA_dataset,data))
            except NameError:
                HA_dataset = data
        elif interface == "JOY":
            try:
                JOY_dataset = np.vstack((JOY_dataset,data))
            except NameError:
                JOY_dataset = data
        elif interface == "SNP":
            try:
                SNP_dataset = np.vstack((SNP_dataset,data))
            except NameError:
                SNP_dataset = data
    # Standardize features 
    sc = StandardScaler()
    HA_dataset[:,:9] = sc.fit_transform(HA_dataset[:,:9])
    JOY_dataset[:,:9] = sc.fit_transform(JOY_dataset[:,:9])
    SNP_dataset[:,:9] = sc.fit_transform(SNP_dataset[:,:9])
    return HA_dataset, JOY_dataset, SNP_dataset

# Create mvAvg dataset, grouped by interfaces 
def mk_mvAvg_dataset(mvAvgPaths,hrvPaths,userCtrlPaths,hrv_metrics_list):
    for i, mvAvgPath in enumerate(mvAvgPaths):
        subj = mvAvgPath.split("\\")[-1].split("_")[0]
        interface = mvAvgPath.split("\\")[-1].split("_")[1]
        autonomy = mvAvgPath.split("\\")[-1].split("_")[2]
        event = subj+"_"+interface+"_"+autonomy
        print(event)
        # Load mvAvg dataframe
        mvAvg_df = pd.read_csv(mvAvgPath)
        # Load HRV dataframe
        hrv_path = [path for path in hrvPaths if event in path][0]
        hrv_df = pd.read_csv(hrv_path)
        # Load user controlled dataframe
        userCtrl_path = [path for path in userCtrlPaths if event in path][0]
        userCtrl_df = pd.read_csv(userCtrl_path)
        # print(userCtrl_df.shape)
        # print(psd_df.shape)
        # Get indices for PSD times where "User Controlled == 1"
        userCtrl_idx = (userCtrl_df["User Controlled"] == 1).tolist()
        mvAvg_len = mvAvg_df.shape[0]
        userCtrl_len = len(userCtrl_idx)
        # print(userCtrl_len)
        if mvAvg_len < userCtrl_len:
            userCtrl_idx = userCtrl_idx[:mvAvg_len]
        elif mvAvg_len > userCtrl_len:
            print("b")
            for i in range(mvAvg_len-userCtrl_len):
                userCtrl_idx.append(False)
        mvAvg_userControlIdx = mvAvg_df.iloc[userCtrl_idx].index.to_list()
        # psd_userControlIdx= psd_df[userCtrl_df["User Controlled"] == 1].index.to_list()
        """psd_df and userCtrl_df have different number of rows """
        # Get indices for HRV times where "NNMean" exists 
        hrv_existIdx = hrv_df[~(hrv_df["NNmean"].isnull())].index.to_list()
        # Get indices that fulfill both conditions 
        featuresIdx = list(set(mvAvg_userControlIdx) & set(hrv_existIdx))
        # If no HRV metrics exist, continue to next loop 
        if len(featuresIdx) == 0:
            continue 
        # print(len(featuresIdx))
        # Select PSD and HRV metrics 
        hrv_features = hrv_df.loc[featuresIdx, hrv_metrics_list].values
        mvAvg_selected = mvAvg_df.iloc[featuresIdx,1].values.reshape(-1,1)
        data = np.hstack((hrv_features,mvAvg_selected))
        # Append to dataset, if already exists, otherwise, create the dataset 
        if interface == "HA":
            try:
                HA_dataset = np.vstack((HA_dataset,data))
            except NameError:
                HA_dataset = data
        elif interface == "JOY":
            try:
                JOY_dataset = np.vstack((JOY_dataset,data))
            except NameError:
                JOY_dataset = data
        elif interface == "SNP":
            try:
                SNP_dataset = np.vstack((SNP_dataset,data))
            except NameError:
                SNP_dataset = data
    # Standardize features 
    sc = StandardScaler()
    HA_dataset[:,:9] = sc.fit_transform(HA_dataset[:,:9])
    JOY_dataset[:,:9] = sc.fit_transform(JOY_dataset[:,:9])
    SNP_dataset[:,:9] = sc.fit_transform(SNP_dataset[:,:9])
    return HA_dataset, JOY_dataset, SNP_dataset

# Create PSD datasets 
# psd30_HA_dataset, psd30_JOY_dataset, psd30_SNP_dataset = mk_PSD_dataset(psd30Paths,hrv30Paths,userCtrl30Paths,hrv_metrics_list)
# psd60_HA_dataset, psd60_JOY_dataset, psd60_SNP_dataset = mk_PSD_dataset(psd60Paths,hrv60Paths,userCtrl60Paths,hrv_metrics_list)

# print("PSD dataset shapes")
# print(psd30_HA_dataset.shape)
# print(psd30_JOY_dataset.shape)
# print(psd30_SNP_dataset.shape)
# print(psd60_HA_dataset.shape)
# print(psd60_JOY_dataset.shape)
# print(psd60_SNP_dataset.shape)

# Create moving average datasets 
# mvAvg30_HA_dataset, mvAvg30_JOY_dataset, mvAvg30_SNP_dataset = mk_mvAvg_dataset(mvAvg30Paths,hrv30Paths,userCtrl30Paths,hrv_metrics_list)
# mvAvg60_HA_dataset, mvAvg60_JOY_dataset, mvAvg60_SNP_dataset = mk_mvAvg_dataset(mvAvg60Paths,hrv60Paths,userCtrl60Paths,hrv_metrics_list)

# print("Moving Average dataset shapes")
# print(mvAvg30_HA_dataset.shape)
# print(mvAvg30_JOY_dataset.shape)
# print(mvAvg30_SNP_dataset.shape)
# print(mvAvg60_HA_dataset.shape)
# print(mvAvg60_JOY_dataset.shape)
# print(mvAvg60_SNP_dataset.shape)

# Create peaks Frequency datasets
pksFreq30_HA_dataset, pksFreq30_JOY_dataset, pksFreq30_SNP_dataset = mk_mvAvg_dataset(pksFreq30Paths,hrv30Paths,userCtrl30Paths,hrv_metrics_list)
pksFreq60_HA_dataset, pksFreq60_JOY_dataset, pksFreq60_SNP_dataset = mk_mvAvg_dataset(pksFreq60Paths,hrv60Paths,userCtrl60Paths,hrv_metrics_list)

print("Peaks Frequency dataset shapes")
print(pksFreq30_HA_dataset.shape)
print(pksFreq30_JOY_dataset.shape)
print(pksFreq30_SNP_dataset.shape)
print(pksFreq60_HA_dataset.shape)
print(pksFreq60_JOY_dataset.shape)
print(pksFreq60_SNP_dataset.shape)


# Support Vector Regression
def apply_SVR(dataset,no_features):
    # Separate features and targets
    X = dataset[:,:9]
    y = dataset[:,-1]
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
    # print(X_test_fs.shape)
    # SVR models
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
    # Predictions
    ypred_rbf = svr_rbf.fit(X_train_fs,y_train).predict(X_test_fs) 
    ypred_lin = svr_lin.fit(X_train_fs,y_train).predict(X_test_fs) 
    ypred_poly = svr_poly.fit(X_train_fs,y_train).predict(X_test_fs) 
    # R^2 values 
    r2_rbf = r2_score(y_test,ypred_rbf)
    r2_lin = r2_score(y_test,ypred_lin)
    r2_poly = r2_score(y_test,ypred_poly)
    # MAE
    mse_rbf = mean_squared_error(y_test,ypred_rbf)
    mse_lin = mean_squared_error(y_test,ypred_lin)
    mse_poly = mean_squared_error(y_test,ypred_poly)

    print("R^2 values: ", r2_rbf,r2_lin,r2_poly)
    print("MSE values: ", mse_rbf,mse_lin,mse_poly)

def apply_SVR_multiFeat(dataset):
    for i in range(1,len(hrv_metrics_list)+1):
        print("No of features: "+str(i))
        apply_SVR(dataset,i)

# print("--- PSD 30 HA ---")
# apply_SVR_multiFeat(psd30_HA_dataset)
# print("--- PSD 30 JOY ---")
# apply_SVR_multiFeat(psd30_JOY_dataset)
# print("--- PSD 30 SNP ---")
# apply_SVR_multiFeat(psd30_SNP_dataset)
# print("--- PSD 60 HA ---")
# apply_SVR_multiFeat(psd60_HA_dataset)
# print("--- PSD 60 JOY ---")
# apply_SVR_multiFeat(psd60_JOY_dataset)
# print("--- PSD 60 SNP ---")
# apply_SVR_multiFeat(psd60_SNP_dataset)

"""
Assumes start time of user controlled == start time of PSD
--- PSD 30 HA ---
No of features: 1
R^2 values:  0.049407413191766 0.02746019154177337 0.059863092012375274
MSE values:  0.014009699926777282 0.0143331549945001 0.01385560560199475
No of features: 2
R^2 values:  0.046403000011911666 0.03608593786128056 0.06044801640299413
MSE values:  0.014053978545914362 0.014206029957699052 0.0138469850685442
No of features: 3
R^2 values:  0.15088932057471327 0.03526706126298629 0.10685909926389259
MSE values:  0.01251407384031075 0.01421809844590143 0.013162985052994795
No of features: 4
R^2 values:  0.18983799672331658 0.044431728899189005 0.16438429449385894
MSE values:  0.01194005372595314 0.014083030862487014 0.012315186811576687
No of features: 5
R^2 values:  0.19697701149259805 0.05418977505663114 0.17579395068284442
MSE values:  0.011834839929760721 0.013939218149833297 0.0121470329024324
No of features: 6
R^2 values:  0.27025683100258713 0.054052168699535774 0.2375673365733324
MSE values:  0.01075485225021189 0.013941246172981883 0.011236625424195524
No of features: 7
R^2 values:  0.3768855026412019 0.14474494994704412 0.32548572506922824
MSE values:  0.009183373875587012 0.012604628710953681 0.009940896572565203
No of features: 8
R^2 values:  0.4454724790377489 0.14538366318915563 0.36663057031661106
MSE values:  0.008172548658206668 0.012595215444970527 0.009334509626728642
No of features: 9
R^2 values:  0.4721133980404081 0.15751721771088634 -0.01346937022016892
MSE values:  0.007779918538657746 0.012416392823951149 0.014936369122589008
--- PSD 30 JOY ---
No of features: 1
R^2 values:  0.46067507970838695 0.0028511296722347756 0.19088286670575005
MSE values:  0.00644274327561603 0.01191188082986907 0.009665664933299602
No of features: 2
R^2 values:  0.46386738047952936 0.061413475172303 0.25054266626746546
MSE values:  0.006404608241329275 0.011212298549356499 0.008952972532134435
No of features: 3
R^2 values:  0.6165599277797745 0.06718541419272406 0.34228196399494737
MSE values:  0.0045805521939591325 0.011143347310665403 0.007857060362483738
No of features: 4
R^2 values:  0.6490652921063377 0.06827800270366102 0.09488221797265961
MSE values:  0.004192245053760273 0.011130295313590896 0.010812482947467143
No of features: 5
R^2 values:  0.6195596702735109 0.13444300722738634 0.5178221293459704
MSE values:  0.004544717449349799 0.010339892122605684 0.005760068034919273
No of features: 6
R^2 values:  0.6514979900008518 0.16083260149950296 0.059939536286817696
MSE values:  0.0041631841900549365 0.010024643606088065 0.011229906135221426
No of features: 7
R^2 values:  0.6339606701842999 0.34096631396308386 -16.68093373076841
MSE values:  0.004372683964809133 0.007872777039160332 0.2112153779931542
No of features: 8
R^2 values:  0.6892899024784913 0.36739795352130944 -15.179447881483174
MSE values:  0.0037117242614902964 0.007557026252166866 0.1932787177444779
No of features: 9
R^2 values:  0.6775795451277433 0.4015013436333056 -13.871093050540305
MSE values:  0.003851615490762251 0.007149629191409928 0.177649189102228
--- PSD 30 SNP ---
No of features: 1
R^2 values:  0.31241007507232244 0.2799398315344036 0.3077613155588841
MSE values:  0.10733195628555083 0.11240052205948893 0.10805762202728038
No of features: 2
R^2 values:  0.3983032426910045 0.2918266765235976 0.37411559317803134
MSE values:  0.09392413662756856 0.11054499992267534 0.09769979948424831
No of features: 3
R^2 values:  0.40759611659503836 0.34372412444761835 0.3908952952877738
MSE values:  0.09247353024217132 0.10244387102306483 0.09508050826424262
No of features: 4
R^2 values:  0.42944370151137967 0.34708302716426853 0.42402493581754175
MSE values:  0.08906314864091058 0.10191955036844089 0.08990901141679962
No of features: 5
R^2 values:  0.46125050580053284 0.348410775564158 0.4500285237672974
MSE values:  0.08409814493189675 0.10171229044788554 0.08584988276481388
No of features: 6
R^2 values:  0.5138773566662238 0.36290654607633055 0.5128803292503545
MSE values:  0.07588315711462082 0.09944951818998557 0.07603879188926768
No of features: 7
R^2 values:  0.612994472854201 0.36659188255428465 0.5966098420908837
MSE values:  0.060411095066944934 0.09887424161973779 0.06296871612724221
No of features: 8
R^2 values:  0.6219393682968064 0.3667409576069096 0.6061908999502668
MSE values:  0.05901480769882283 0.09885097118418963 0.061473124574703414
No of features: 9
R^2 values:  0.6513684569730513 0.3720396035768768 0.6047821806736885
MSE values:  0.054420962523365146 0.09802385895202456 0.061693023951251445
--- PSD 60 HA ---
No of features: 1
R^2 values:  0.19257698647498722 0.03139269492175556 0.04762941455562886
MSE values:  0.006979523713092963 0.00837283250690903 0.00823247910130976
No of features: 2
R^2 values:  0.35915245990012135 0.0661907789124121 0.25787384259986845
MSE values:  0.005539612480299752 0.008072031008419833 0.0064150848154148135
No of features: 3
R^2 values:  0.4133734595138978 0.07197086492325999 0.3159657616769881
MSE values:  0.005070915469918794 0.008022066805392839 0.005912926814037418
No of features: 4
R^2 values:  0.4125772685752047 0.1218606812524452 0.36452409163526855
MSE values:  0.005077797901362626 0.007590809397220619 0.005493179036559049
No of features: 5
R^2 values:  0.40622470346631656 0.14308717484035338 0.38276949524912096
MSE values:  0.005132710726577854 0.007407323401824191 0.005335462170622963
No of features: 6
R^2 values:  0.41222956661809396 0.15016491602646176 0.38212232209718755
MSE values:  0.005080803505629649 0.007346142011628347 0.005341056462939052
No of features: 7
R^2 values:  0.5862414312750686 0.0745011989488108 0.5211552546122394
MSE values:  0.003576610640937103 0.00800019409921832 0.0041392283838096264
No of features: 8
R^2 values:  0.5858197893072059 0.07595570612573077 0.5633303961486238
MSE values:  0.003580255397234406 0.00798762105242361 0.0037746581455015416
No of features: 9
R^2 values:  0.4968326027814146 0.06504246845680572 0.5923353176351003
MSE values:  0.0043494781814682444 0.008081957230388546 0.003523933885825741
--- PSD 60 JOY ---
No of features: 1
R^2 values:  0.6159369852119925 -0.0055497411014377995 0.48289103274393863
MSE values:  0.003425179065182684 0.008967767761031644 0.004611719277476769
No of features: 2
R^2 values:  0.6253799717180388 0.11629500951338123 0.5806350787806951
MSE values:  0.0033409639274372136 0.007881122932086987 0.0037400111271459007
No of features: 3
R^2 values:  0.5674833222278399 0.052435658361180915 0.6052648809897594
MSE values:  0.0038573020910781697 0.008450638100850111 0.00352035581107064
No of features: 4
R^2 values:  0.6689728174327423 0.2002189902738415 0.5468918400650445
MSE values:  0.0029521910001191333 0.007132665905767517 0.004040942563889695
No of features: 5
R^2 values:  0.6188387422969431 0.18116386482627678 0.5572235811846787
MSE values:  0.0033993004014297988 0.0073026047289667636 0.003948801269291486
No of features: 6
R^2 values:  0.5895809269716403 0.4408931123138907 0.6318392184874937
MSE values:  0.0036602296049369973 0.004986268224653079 0.003283358597166614
No of features: 7
R^2 values:  0.5829000113627555 0.4285564804786932 0.632475128376562
MSE values:  0.003719811838577577 0.005096289683293424 0.0032776873787584023
No of features: 8
R^2 values:  0.5803255594368432 0.3979895506090384 0.623446405055768
MSE values:  0.003742771505354711 0.005368893928547915 0.0033582080040541746
No of features: 9
R^2 values:  0.6017735894128853 0.4037229335735344 0.6234638437706836
MSE values:  0.0035514921047493184 0.005317762050323436 0.003358052480822432
--- PSD 60 SNP ---
No of features: 1
R^2 values:  0.5023364781432209 0.39933711816420814 0.48628038616093183
MSE values:  0.06692650419315903 0.08077800584995903 0.06908575047138568
No of features: 2
R^2 values:  0.4953255988303338 0.42645066515047036 0.4855857661748104
MSE values:  0.06786933729850908 0.07713173716364426 0.06917916396337795
No of features: 3
R^2 values:  0.5111323810111228 0.46284577392087367 0.4827748176882888
MSE values:  0.06574361855996072 0.0722372707365191 0.06955718434744797
No of features: 4
R^2 values:  0.6650078946426299 0.4618824167558062 0.6418795004037352
MSE values:  0.04505021879085472 0.0723668243897723 0.048160558419998535
No of features: 5
R^2 values:  0.6832150745428001 0.46783351150643837 0.6519900064986026
MSE values:  0.042601691124233766 0.0715665126323502 0.046800883059368345
No of features: 6
R^2 values:  0.846713698706745 0.4607047668817017 0.8163656551572159
MSE values:  0.02061416164878014 0.0725251964338861 0.024695410071999404
No of features: 7
R^2 values:  0.8729199941887495 0.5340979108224868 0.8232515192180867
MSE values:  0.017089901445983324 0.06265518117261952 0.023769389197043626
No of features: 8
R^2 values:  0.8936197772470095 0.5340659681757866 0.8453589644740559
MSE values:  0.014306164931646631 0.06265947686556152 0.020796348251419607
No of features: 9
R^2 values:  0.9210282147870474 0.5420491160854721 0.8678727328446254
MSE values:  0.010620238940713364 0.061585891684837864 0.017768664390575256
"""

# Moving average SVR
# print("--- Moving Average 30 HA ---")
# apply_SVR_multiFeat(mvAvg30_HA_dataset)
# print("--- Moving Average 30 JOY ---")
# apply_SVR_multiFeat(mvAvg30_JOY_dataset)
# print("--- Moving Average 30 SNP ---")
# apply_SVR_multiFeat(mvAvg30_SNP_dataset)
# print("--- Moving Average 60 HA ---")
# apply_SVR_multiFeat(mvAvg60_HA_dataset)
# print("--- Moving Average 60 JOY ---")
# apply_SVR_multiFeat(mvAvg60_JOY_dataset)
# print("--- Moving Average 60 SNP ---")
# apply_SVR_multiFeat(mvAvg60_SNP_dataset)

"""
Assumes start time of user controlled == start time of moving Average 
--- Moving Average 30 HA ---
No of features: 1
R^2 values:  0.045554924845029654 0.026704246373804197 0.03542172837300406
MSE values:  162.79231949931128 166.0075340279925 164.52066050139086
No of features: 2
R^2 values:  0.037150230050071786 0.02763105591376369 0.0350576600666197
MSE values:  164.2258433300386 165.84945528813458 164.58275681850253
No of features: 3
R^2 values:  0.04222088744069341 0.023205410825868245 0.04065916862634256
MSE values:  163.36098049037062 166.6043033646698 163.6273507979052
No of features: 4
R^2 values:  0.0964065830143147 0.0251857158722113 0.0840658966476403
MSE values:  154.11894520124608 166.2665380388253 156.22380069281877
No of features: 5
R^2 values:  0.090608583127882 0.04913182530779969 0.08568310623445996
MSE values:  155.10786522874568 162.18223523347967 155.94796575311608
No of features: 6
R^2 values:  0.2965288576023133 0.08751041979218921 0.25089385494968286
MSE values:  119.98563558322672 155.636295003003 127.76924526969579
No of features: 7
R^2 values:  0.3318817427440607 0.0750691161261755 0.2914471034168745
MSE values:  113.95576720941449 157.75831200964618 120.85239111741876
No of features: 8
R^2 values:  0.3546590437191145 0.10594602369310668 0.30600107214537864
MSE values:  110.0708190293834 152.49187653564007 118.37003315999628
No of features: 9
R^2 values:  0.37508402222171555 0.07121489420744653 0.31114967067854227
MSE values:  106.58708831222073 158.41569685278458 117.49187650205725
--- Moving Average 30 JOY ---
No of features: 1
R^2 values:  0.40101946983796866 0.043103086101955346 0.14909975753583604
MSE values:  20.790315149668924 33.213414132683646 29.534322588038346
No of features: 2
R^2 values:  0.5325557195076787 0.06481839484103025 0.2205095316372303
MSE values:  16.22475759557106 32.45968660812522 27.055724981646964
No of features: 3
R^2 values:  0.551409598303557 -0.1445808094655594 0.3632219017497351
MSE values:  15.57034887571845 39.72782844312984 22.102249866867275
No of features: 4
R^2 values:  0.656493811384852 -0.027711059092476065 0.23123776140918428
MSE values:  11.922928304929465 35.67133775709325 26.683353482532212
No of features: 5
R^2 values:  0.7337909978021988 -0.060890145402385576 -0.07381276235342615
MSE values:  9.239981556452307 36.82296727762955 37.27150486014148
No of features: 6
R^2 values:  0.751510540230529 0.34954486077156466 -0.216154379945527
MSE values:  8.624945085578602 22.57697313070054 42.21211134004309
No of features: 7
R^2 values:  0.7713459764081136 0.3429727712092735 -11.66122988704704
MSE values:  7.93646699906789 22.805087078168786 439.4649680230974
No of features: 8
R^2 values:  0.7764284894361008 0.24303186017397482 -4.411552210912526
MSE values:  7.760055509406336 26.273986202831175 187.8322754219145
No of features: 9
R^2 values:  0.8047167464401281 0.2653668240324222 0.19343443032620344
MSE values:  6.778184232239007 25.498750758453752 27.995488230400962
--- Moving Average 30 SNP ---
No of features: 1
R^2 values:  0.5835131340559931 0.5138665553322372 0.5649090901072582
MSE values:  38.18456937754904 44.56994388667877 39.89023518588919
No of features: 2
R^2 values:  0.5855620834849129 0.5297575320669095 0.5675797896655508
MSE values:  37.99671651106707 43.113018943255256 39.64537869482299
No of features: 3
R^2 values:  0.5620602784016365 0.5152752487025345 0.5536199053496089
MSE values:  40.151421449159045 44.44079131518558 40.925256200579206
No of features: 4
R^2 values:  0.6117589093326483 0.5159303571615658 0.6054852964864781
MSE values:  35.5949252522068 44.38072931455801 36.17010594711178
No of features: 5
R^2 values:  0.6244299893286195 0.5091755527120104 0.6204933411186819
MSE values:  34.433208586549256 45.00002687284044 34.79413045222837
No of features: 6
R^2 values:  0.6282187501202906 0.5048769673684284 0.6308240012594339
MSE values:  34.08584541345953 45.39413205045579 33.846989399013374
No of features: 7
R^2 values:  0.6931344845705605 0.5013167040553211 0.6845620476222287
MSE values:  28.134206674042165 45.72054599672406 28.920149377532468
No of features: 8
R^2 values:  0.7139669155405741 0.5054755545498013 0.6633709053670712
MSE values:  26.22423670686333 45.33925206353509 30.863006902697162
No of features: 9
R^2 values:  0.7261632604482918 0.5206641544980358 0.7045125428257177
MSE values:  25.10604495494414 43.94672280056148 27.09103751229419
--- Moving Average 60 HA ---
No of features: 1
R^2 values:  0.1808353914698958 0.11081209713969198 0.16792667181982834
MSE values:  77.15473228011209 83.75002273963615 78.3705670443448
No of features: 2
R^2 values:  0.19189302964192334 0.11625166164174805 0.23479741718593183
MSE values:  76.1132454972004 83.23768597788369 72.07220600388041
No of features: 3
R^2 values:  0.2025740470941707 0.11641736795157509 0.22710846980783927
MSE values:  75.10723152464111 83.22207858244852 72.79640559733832
No of features: 4
R^2 values:  0.2886735087069996 0.12522526387989663 0.24120132289129037
MSE values:  66.99777361956917 82.39248847903798 71.46904074857159
No of features: 5
R^2 values:  0.29327631987366576 0.11808913986568492 0.23391014368680285
MSE values:  66.5642482211303 83.06461924750727 72.15577571449984
No of features: 6
R^2 values:  0.24130175389531616 0.1387140637823414 0.16474850063113178
MSE values:  71.45958144436392 81.12201764275443 78.66990974622769
No of features: 7
R^2 values:  0.2659718175790884 0.12698071500734198 0.24187676865314767
MSE values:  69.13597988854205 82.22714764233864 71.40542247651348
No of features: 8
R^2 values:  0.515570868707973 0.130056287842214 0.39011617487194605
MSE values:  45.6269711170634 81.9374683810381 57.44318389702064
No of features: 9
R^2 values:  0.6444733609060485 0.1819325636955541 0.6205213323708414
MSE values:  33.486020235863 77.05139281884344 35.741992149146164
--- Moving Average 60 JOY ---
No of features: 1
R^2 values:  0.4558021897721394 -0.6495346249573479 0.3789528613854488
MSE values:  13.575408457303066 41.14883573104423 15.492470604396079
No of features: 2
R^2 values:  0.5144685618578676 -0.6221939545632489 0.4239329570305035
MSE values:  12.111933322336213 40.46680290930065 14.370409546167084
No of features: 3
R^2 values:  0.6713844912954634 -0.39270736542071605 0.6951492395704636
MSE values:  8.197551831751428 34.74209376028967 7.604722976774901
No of features: 4
R^2 values:  0.7162289476032018 0.07333459200958603 0.7203644404409065
MSE values:  7.0788743950150925 23.116339647629154 6.975711531458942
No of features: 5
R^2 values:  0.7313869560719701 0.06497610054062608 0.7234089744390558
MSE values:  6.700746897080788 23.324848270128946 6.899763426173996
No of features: 6
R^2 values:  0.8058197905426707 0.09881574249025205 0.8121480502521454
MSE values:  4.843965940627655 22.48069389669861 4.686102919561763
No of features: 7
R^2 values:  0.8583291171708006 0.12074767032168754 0.8420040066420751
MSE values:  3.5340827632287346 21.933586074924825 3.9413244672063263
No of features: 8
R^2 values:  0.8813474825974263 0.20188605857760045 -0.37042046137731144
MSE values:  2.959872968898495 19.909530223468792 34.18613079985207
No of features: 9
R^2 values:  0.9213711494517991 0.19454292937125917 0.8915177763118269
MSE values:  1.9614536160539242 20.092709899052373 2.7061676273685094
--- Moving Average 60 SNP ---
No of features: 1
R^2 values:  0.5532827588985147 0.511984913814696 0.5382368909475354
MSE values:  39.41697042957023 43.06096665066984 40.74457204765115
No of features: 2
R^2 values:  0.5563714712121894 0.5131676285704893 0.5388922254997062
MSE values:  39.144431850953104 42.95660749846706 40.68674732031199
No of features: 3
R^2 values:  0.5459861837693678 0.49734286910459513 0.5369803440409233
MSE values:  40.06079802259878 44.35293613441622 40.8554459242429
No of features: 4
R^2 values:  0.686173974209165 0.49868161761386287 0.6795551001443598
MSE values:  27.691053849020022 44.2348090384655 28.275083161714946
No of features: 5
R^2 values:  0.7036786321298869 0.5013390724722306 0.6867226815291174
MSE values:  26.146496083710733 44.000323305808486 27.642637584289055
No of features: 6
R^2 values:  0.8087547116160064 0.5024449519958238 0.7950636785446636
MSE values:  16.874902473965523 43.9027438607603 18.082957583709504
No of features: 7
R^2 values:  0.8594715503993493 0.5099702930526928 0.8039842702531963
MSE values:  12.399802901638623 43.23873065818133 17.29583170801576
No of features: 8
R^2 values:  0.868774325295245 0.5072698384527288 0.7853504494235652
MSE values:  11.578954344102957 43.4770105572305 18.940023373464857
No of features: 9
R^2 values:  0.890543033121964 0.5068806442248425 0.7969159443215855
MSE values:  9.658149786436898 43.51135187196043 17.919519286193594
"""

# Peaks Frequency SVR
print("--- Moving Average 30 HA ---")
apply_SVR_multiFeat(pksFreq30_HA_dataset)
print("--- Moving Average 30 JOY ---")
apply_SVR_multiFeat(pksFreq30_JOY_dataset)
print("--- Moving Average 30 SNP ---")
apply_SVR_multiFeat(pksFreq30_SNP_dataset)
print("--- Moving Average 60 HA ---")
apply_SVR_multiFeat(pksFreq60_HA_dataset)
print("--- Moving Average 60 JOY ---")
apply_SVR_multiFeat(pksFreq60_JOY_dataset)
print("--- Moving Average 60 SNP ---")
apply_SVR_multiFeat(pksFreq60_SNP_dataset)

"""
Peaks Frequency dataset shapes
(942, 10)
(454, 10)
(1867, 10)
(635, 10)
(274, 10)
(1413, 10)
--- Moving Average 30 HA ---
No of features: 1
R^2 values:  0.19196568285331028 0.07876641825386332 0.19414262513446157
MSE values:  0.14099522600705547 0.1607475503296053 0.1406153677975439
No of features: 2
R^2 values:  0.38271636389778474 0.08475713775432092 0.39308417201150714
MSE values:  0.10771082853265594 0.15970221991233474 0.10590173278366104
No of features: 3
R^2 values:  0.39551659183663457 0.11854386605022948 0.39366656768596975
MSE values:  0.10547729588078428 0.15380671858147257 0.10580010961246164
No of features: 4
R^2 values:  0.4097573099134184 0.12433777431668713 0.3711726065142871
MSE values:  0.10299240975511942 0.15279573007744712 0.10972511758785877
No of features: 5
R^2 values:  0.43522875485615686 0.12008431180374446 0.40895733093152364
MSE values:  0.09854785577985092 0.15353792368927616 0.10313199939253907
No of features: 6
R^2 values:  0.5395615162933927 0.17475372830768776 0.4378854888546472
MSE values:  0.08034266205648491 0.14399856803063768 0.09808427793097872
No of features: 7
R^2 values:  0.5556590657719045 0.17844764930704937 0.4296389616331687
MSE values:  0.07753377439080113 0.14335400973019777 0.0995232279170034
No of features: 8
R^2 values:  0.5954950991283466 0.24982334081980584 0.49570900011764185
MSE values:  0.07058271995273015 0.13089954889518346 0.08799455913309828
No of features: 9
R^2 values:  0.6568401661365457 0.24655874404564448 0.46065209569651067
MSE values:  0.05987851914875762 0.13146919371128504 0.09411169556787152
--- Moving Average 30 JOY ---
No of features: 1
R^2 values:  0.1419941464345531 -0.14386322383521777 -0.10095502423875202
MSE values:  0.11170435182316224 0.14892031267840225 0.14333406568031404
No of features: 2
R^2 values:  0.18897196159064955 -0.06396955460992726 -0.08117755867854037
MSE values:  0.10558827887299117 0.13851890282962379 0.14075922430606477
No of features: 3
R^2 values:  0.3971487382106139 -0.07083096114647458 -0.19865483360537306
MSE values:  0.07848560608779369 0.13941219390284346 0.15605366873802495
No of features: 4
R^2 values:  0.4894894230864909 -0.0964526178937204 0.21831780038566218
MSE values:  0.06646371100618872 0.14274789440849078 0.10176772463355105
No of features: 5
R^2 values:  0.4688006849174058 0.02808141807610065 -1.5683691957069819
MSE values:  0.06915719156650578 0.12653472557039483 0.33437769108079984
No of features: 6
R^2 values:  0.493520567225469 0.03227939644177713 -0.06265278450083933
MSE values:  0.06593889367390705 0.1259881879793552 0.1383474716547378
No of features: 7
R^2 values:  0.6010990096655331 0.19434923404262905 -0.35969995727293447
MSE values:  0.05193318481658852 0.10488820820176106 0.17702024032819622
No of features: 8
R^2 values:  0.6536265151380833 0.18978029895674464 -25.513603549906964
MSE values:  0.04509459399891961 0.10548304089453348 3.4518236521713193
No of features: 9
R^2 values:  0.6778910907326572 0.19039945750888987 -39.0968440605428
MSE values:  0.041935572789690484 0.1054024322314856 5.220234754022804
--- Moving Average 30 SNP ---
No of features: 1
R^2 values:  0.08548468099556883 -0.013499810658945899 0.03361543256506816
MSE values:  0.02306782156885281 0.025564615820539788 0.024376176434912186
No of features: 2
R^2 values:  0.09215413098461511 -0.013180501664269428 0.04033694154988132
MSE values:  0.02289959072666507 0.025556561540024787 0.024206632451653488
No of features: 3
R^2 values:  0.13928445687683 0.005607810233151045 0.05811456624058298
MSE values:  0.02171077089437724 0.02508264337001374 0.02375820795207312
No of features: 4
R^2 values:  0.2203971751860393 0.018823087220782964 0.09468480277966707
MSE values:  0.019664775956907642 0.024749299963732117 0.02283575682010882
No of features: 5
R^2 values:  0.2885672320424396 0.054447726952388864 0.15240463730519038
MSE values:  0.017945247945486357 0.0238507006557643 0.021379826212770323
No of features: 6
R^2 values:  0.2865389351027876 0.0741452945370501 0.16481811520181355
MSE values:  0.017996409900808745 0.023353847333637302 0.021066707463179627
No of features: 7
R^2 values:  0.16921722140814544 0.0710777008104796 0.18384622505398485
MSE values:  0.020955743988953116 0.023431224610168104 0.020586740606702847
No of features: 8
R^2 values:  0.1983607492529993 0.07751622373688127 0.1980493971174968
MSE values:  0.0202206248649303 0.023268818694218128 0.020228478440871132
No of features: 9
R^2 values:  0.21860946165024453 0.08048836209024113 0.23978220536835704
MSE values:  0.019709869413521177 0.023193849193119023 0.019175806108005237
--- Moving Average 60 HA ---
No of features: 1
R^2 values:  0.0030252587704759915 -0.05254079314060234 0.01827089184934183
MSE values:  0.11579731023182875 0.12225123437394873 0.114026549920352
No of features: 2
R^2 values:  0.19817420402198416 -0.12641176438978352 0.09043721067361099
MSE values:  0.09313101587131738 0.1308312509191208 0.10564452652136692
No of features: 3
R^2 values:  0.3516415060519784 -0.08728177289392924 0.33145915626752953
MSE values:  0.07530598977116491 0.12628635366422522 0.0776501432612673
No of features: 4
R^2 values:  0.43399013199795866 -0.08892788085973735 0.3748637156721297
MSE values:  0.06574130473805015 0.12647754694817614 0.0726087605431336
No of features: 5
R^2 values:  0.3936114743721024 -0.09799771973576887 0.3878522300033713
MSE values:  0.07043123292828664 0.12753099685281993 0.07110016161753747
No of features: 6
R^2 values:  0.40984739687997596 -0.07804633939973371 0.37791978710652696
MSE values:  0.06854545179683535 0.12521367016159818 0.07225380184925226
No of features: 7
R^2 values:  0.5077734416846611 -0.12888983737998894 0.4020729251328812
MSE values:  0.057171469968529946 0.13111907585084376 0.06944844650628675
No of features: 8
R^2 values:  0.5395716997502901 0.10407370275330807 0.4262323695507704
MSE values:  0.05347814394753531 0.10406066582908984 0.06664235868420532
No of features: 9
R^2 values:  0.736819251620261 0.3846384259027237 0.6252817162138912
MSE values:  0.030568099177306438 0.07147344075398587 0.04352303780896783
--- Moving Average 60 JOY ---
No of features: 1
R^2 values:  -0.2598684894921024 -0.27446670424341435 -0.2648385484443294
MSE values:  0.15901576488484462 0.16085829551720365 0.15964306664880076
No of features: 2
R^2 values:  -0.2757492987229453 -0.2644609512672018 -0.24615547343249644
MSE values:  0.1610201796693201 0.15959540778246664 0.15728496063362038
No of features: 3
R^2 values:  0.3600814064262362 -0.48749861776406345 0.2824002858516117
MSE values:  0.08076806862769302 0.18774636594352606 0.09057268149666438
No of features: 4
R^2 values:  0.40142881749556303 -0.49094763476725345 0.29404651393635706
MSE values:  0.075549357109757 0.18818168763101759 0.08910273929050147
No of features: 5
R^2 values:  0.4301072589976329 -0.5006577432674408 0.3061677245963902
MSE values:  0.07192967430223211 0.18940726025478768 0.08757284660685118
No of features: 6
R^2 values:  0.626071096849945 0.2240620517331825 0.48892617814017303
MSE values:  0.04719587087294139 0.0979359094825575 0.06450577609764932
No of features: 7
R^2 values:  0.7123273378001583 0.15159779323812117 -84.79363391126289
MSE values:  0.03630893922476669 0.10708207004416548 10.828543163381761
No of features: 8
R^2 values:  0.8442660740058592 0.11259609860722919 -51.93642940378039
MSE values:  0.019656138372395833 0.11200471423700233 6.681432929010076
No of features: 9
R^2 values:  0.8599680593103745 0.10739291126050243 0.8204229984064204
MSE values:  0.017674294057506534 0.11266144057207596 0.022665519855680962
--- Moving Average 60 SNP ---
No of features: 1
R^2 values:  0.2933341460562572 -0.006971034838489265 0.20490042325221636
MSE values:  0.005372594210555962 0.007655735340512862 0.00604493249392337
No of features: 2
R^2 values:  0.40362568777161967 -0.005238037956547759 0.22730948182660693
MSE values:  0.004534076691722484 0.007642559822037094 0.005874562328605655
No of features: 3
R^2 values:  0.454890998965343 0.053712864565026885 0.3083605032858906
MSE values:  0.004144320044242423 0.007194371649611722 0.005258352777483831
No of features: 4
R^2 values:  0.5092468547895082 0.24740903296102912 0.385113517060261
MSE values:  0.0037310668006040612 0.0057217507395683175 0.004674819845836433
No of features: 5
R^2 values:  0.5568480601496721 0.2487000607265577 0.39795340856119454
MSE values:  0.0033691673839189844 0.005711935395781666 0.004577201535347676
No of features: 6
R^2 values:  0.5695229548898884 0.26786889410704817 0.4410369500754847
MSE values:  0.0032728035003088485 0.005566199808490458 0.004249648726027488
No of features: 7
R^2 values:  0.5940359183364535 0.29034655212239147 0.49442761725035456
MSE values:  0.0030864379008368947 0.005395308099705977 0.0038437335572661835
No of features: 8
R^2 values:  0.6051662134265352 0.2930234438979046 0.48064728309410076
MSE values:  0.0030018172012105716 0.00537495639716489 0.003948501805362843
No of features: 9
R^2 values:  0.5795187386707825 0.29763293980327166 0.5232499754393494
MSE values:  0.0031968081911093234 0.005339911614858585 0.003624604765523132
"""