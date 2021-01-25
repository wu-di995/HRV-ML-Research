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


# SPARC files
lin_30_sparc_paths = glob.glob("E:\\argall-lab-data\\SPARC_userCmd_byEvent\\lin\\30s\\*.csv")
lin_60_sparc_paths = glob.glob("E:\\argall-lab-data\\SPARC_userCmd_byEvent\\lin\\60s\\*.csv")
ang_30_sparc_paths = glob.glob("E:\\argall-lab-data\\SPARC_userCmd_byEvent\\ang\\30s\\*.csv")
ang_60_sparc_paths = glob.glob("E:\\argall-lab-data\\SPARC_userCmd_byEvent\\ang\\60s\\*.csv")

# HRV files 
hrv_30_paths = glob.glob("E:\\argall-lab-data\\HRV_byEvent\\30s\\*.csv")
hrv_60_paths = glob.glob("E:\\argall-lab-data\\HRV_byEvent\\60s\\*.csv")

# HRV metrics 
HRV_metrics_list = ['SDNN','RMSSD','lf','hf','lfhf','SD1','SD2','SD1SD2','ApEn']

# Generate dataset separated by interfaces, first 9 columns are features, 10th column is target 
def mk_interfaceDatasets(SPARC_paths, HRV_paths):
    for i,SPARC_path in enumerate(SPARC_paths):
        subj = SPARC_path.split("\\")[-1].split("_")[0]
        interface = SPARC_path.split("\\")[-1].split("_")[1]
        autonomy = SPARC_path.split("\\")[-1].split("_")[2]
        event = subj+"_"+interface+"_"+autonomy
        # Load SPARC dataframe
        SPARC_df = pd.read_csv(SPARC_path)
        # Load HRV dataframe
        HRV_path = [path for path in HRV_paths if event in path][0]
        HRV_df = pd.read_csv(HRV_path)
        # Get indices for SPARC times where "User Controlled == 1"
        sparc_userControlIdx= SPARC_df[SPARC_df["User Controlled"] == 1].index.to_list()
        # Get indices for HRV times where "NNMean" exists 
        hrv_existIdx = HRV_df[~(HRV_df["NNmean"].isnull())].index.to_list()
        # If no HRV metrics exist, continue to next loop 
        if len(hrv_existIdx) == 0:
            continue 
        # Get indices that fulfill both conditions 
        featuresIdx = list(set(sparc_userControlIdx) & set(hrv_existIdx))
        # If no common indices, continue to next loop
        if len(featuresIdx) == 0:
            continue 
        # Select SPARC and HRV metrics 
        HRV_features = HRV_df.loc[featuresIdx,HRV_metrics_list].values
        SPARC_targets = SPARC_df.loc[featuresIdx,"sal"].values.reshape(-1,1)
        data = np.hstack((HRV_features,SPARC_targets))
        # print(HRV_features.shape,SPARC_targets.shape, data.shape)
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

lin_HA_30_dataset, lin_JOY_30_dataset ,lin_SNP_30_dataset = mk_interfaceDatasets(lin_30_sparc_paths,hrv_30_paths)
lin_HA_60_dataset, lin_JOY_60_dataset ,lin_SNP_60_dataset = mk_interfaceDatasets(lin_60_sparc_paths,hrv_60_paths)
ang_HA_30_dataset, ang_JOY_30_dataset ,ang_SNP_30_dataset = mk_interfaceDatasets(ang_30_sparc_paths,hrv_30_paths)
ang_HA_60_dataset, ang_JOY_60_dataset ,ang_SNP_60_dataset = mk_interfaceDatasets(ang_60_sparc_paths,hrv_60_paths)

# Generate dataset, first 9 columns are features, 10th column is target 
def mk_dataset(SPARC_paths, HRV_paths):
    for i,SPARC_path in enumerate(SPARC_paths):
        subj = SPARC_path.split("\\")[-1].split("_")[0]
        interface = SPARC_path.split("\\")[-1].split("_")[1]
        autonomy = SPARC_path.split("\\")[-1].split("_")[2]
        event = subj+"_"+interface+"_"+autonomy
        # Load SPARC dataframe
        SPARC_df = pd.read_csv(SPARC_path)
        # Load HRV dataframe
        HRV_path = [path for path in HRV_paths if event in path][0]
        HRV_df = pd.read_csv(HRV_path)
        # Get indices for SPARC times where "User Controlled == 1"
        sparc_userControlIdx= SPARC_df[SPARC_df["User Controlled"] == 1].index.to_list()
        # Get indices for HRV times where "NNMean" exists 
        hrv_existIdx = HRV_df[~(HRV_df["NNmean"].isnull())].index.to_list()
        # If no HRV metrics exist, continue to next loop 
        if len(hrv_existIdx) == 0:
            continue 
        # Get indices that fulfill both conditions 
        featuresIdx = list(set(sparc_userControlIdx) & set(hrv_existIdx))
        # If no common indices, continue to next loop
        if len(featuresIdx) == 0:
            continue 
        # Select SPARC and HRV metrics 
        HRV_features = HRV_df.loc[featuresIdx,HRV_metrics_list].values
        SPARC_targets = SPARC_df.loc[featuresIdx,"sal"].values.reshape(-1,1)
        data = np.hstack((HRV_features,SPARC_targets))
        # print(HRV_features.shape,SPARC_targets.shape, data.shape)
        # Append to dataset, if already exists, otherwise, create the dataset 
        try:
            dataset = np.vstack((dataset,data))
        except NameError:
            dataset = data
    # Standardize features 
    sc = StandardScaler()
    dataset[:,:9] = sc.fit_transform(dataset[:,:9] )
    return dataset 

# Create datasets
lin_30_dataset = mk_dataset(lin_30_sparc_paths,hrv_30_paths)
lin_60_dataset = mk_dataset(lin_60_sparc_paths,hrv_60_paths)
ang_30_dataset = mk_dataset(ang_30_sparc_paths,hrv_30_paths)
ang_60_dataset = mk_dataset(ang_60_sparc_paths,hrv_60_paths)

# Support Vector Regression
def apply_SVR(dataset):
    # Separate features and targets
    X = dataset[:,:9]
    y = dataset[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    # for train_index, test_index in sss.split(X,y):
    #     X_train, X_test = X[train_index],X[test_index]
    #     y_train, y_test = y[train_index],y[test_index]
    # f_selector = SelectKBest(score_func=f_regression, k=3)
    f_selector = SelectKBest(score_func=mutual_info_regression,k=5)
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
    # Data Visualization
    # lw = 2
    # svrs = [svr_rbf, svr_lin, svr_poly]
    # kernel_label = ['RBF', 'Linear', 'Polynomial']
    # model_color = ['m', 'c', 'g']

    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    # for ix, svr in enumerate(svrs):
    #     axes[ix].scatter(X[:,0], svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
    #                 label='{} model'.format(kernel_label[ix]))
    #     axes[ix].scatter(X[svr.support_][:,0], y[svr.support_], facecolor="none",
    #                     edgecolor=model_color[ix], s=50,
    #                     label='{} support vectors'.format(kernel_label[ix]))
        # axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        #                 y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        #                 facecolor="none", edgecolor="k", s=50,
        #                 label='other training data')
    #     axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
    #                     ncol=1, fancybox=True, shadow=True)

    # fig.text(0.5, 0.04, 'data', ha='center', va='center')
    # fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    # fig.suptitle("Support Vector Regression", fontsize=14)
    # plt.show()

# apply_SVR(lin_30_dataset)
# apply_SVR(lin_60_dataset)
# apply_SVR(ang_30_dataset)
# apply_SVR(ang_60_dataset)
# lin_30_dataset
# lin_60_dataset
# ang_30_dataset
# ang_60_dataset

""" ALL INTERFACES TOGETHER"""
""" Using all features -- f_regression"""
# lin_30_dataset
# R^2 values:  0.320454840156037 0.01950054421905567 0.23931073211883902
# MSE values:  18.941114221603005 27.329680621124417 21.202862092922665

# lin_60_dataset
# R^2 values:  0.48404989748823735 0.01922624966154196 0.30947110281741097
# MSE values:  13.686469316806122 26.01672094915447 18.3174739527104

# ang_30_dataset
# R^2 values:  0.2724863609227318 0.0136732008501812 0.16339758012272476
# MSE values:  37.1132095233794 50.31624314535157 42.678238907042505

# ang_60_dataset
# R^2 values:  0.6023388339964628 0.13204204895168137 0.5255628121019462
# MSE values:  26.63824956398395 58.14216344892882 31.781268311099524

"""Using 5 features -- f_regression"""
# lin_30_dataset
# R^2 values:  0.14088797192620028 0.024340837612768285 0.09423629147610191
# MSE values:  26.167884668378754 29.717819798470952 27.58885859693766

# lin_60_dataset
# R^2 values:  0.1890978973353663 0.03462464014440714 0.16330528313275627
# MSE values:  23.04846697292673 27.439097796138082 23.781576696826605

# ang_30_dataset
# R^2 values:  0.18290725936031638 -0.06732263736038058 0.12595696170530457
# MSE values:  44.30966491440049 57.87924193276795 47.39799073148386

# ang_60_dataset
# R^2 values:  0.3914348244934248 0.12030049239410867 0.338127560080443
# MSE values:  35.76965423905908 51.706125305543836 38.90289697661014
"""Using 4 features -- f_regression"""
# R^2 values:  0.11016356306518216 0.015419823000624167 0.08753153432044736
# MSE values:  27.10372628310024 29.98954697009362 27.79308028892354
# R^2 values:  0.18811118241036617 0.03466766681366584 0.13817626781801473
# MSE values:  23.076512610354264 27.43787483869093 24.495824788724235
# R^2 values:  0.10759507132943646 -0.06331880481366658 0.016542205269117538
# MSE values:  48.393727407730324 57.662120338492876 53.3313823200356
# R^2 values:  0.382652318794796 0.11908730320508043 0.3365787983312464
# MSE values:  36.285863849526876 51.77743296422145 38.99392859408862

"""Using 3 features-- f_regression"""
# R^2 values:  0.0812610957414982 0.014665835518022696 0.06221829271267609
# MSE values:  27.984072974617842 30.012512842810306 28.564101954701062
# R^2 values:  0.15549510098669828 0.032800930351917224 0.08929986605330431
# MSE values:  24.003567396634185 27.490933541516327 25.885050600480852
# R^2 values:  0.004496785501606482 -0.07305875236112436 -0.03031578031561377
# MSE values:  53.98458664692005 58.1903025027033 55.87241779441492
# R^2 values:  0.25971414502901824 0.1179787559223513 0.2397112661444719
# MSE values:  43.51180471070539 51.842590082319546 44.687514544726724

"""Using 8 features -- mutual_info_regression"""
# R^2 values:  0.16872803999581787 0.02076458881614862 0.12007025200877819
# MSE values:  25.319897832437277 29.826749557335102 26.801976237413236
# R^2 values:  0.34497477462412307 -0.007435704752182026 0.24727282536500672
# MSE values:  18.617940715531255 28.634589171771953 21.394946895795925
# R^2 values:  0.22275795409227905 0.011583619156920966 0.1386595155537086
# MSE values:  42.14862389377296 53.60027871104823 46.70915104831733
# R^2 values:  0.43742820602348764 0.12396935662932573 0.40295611015197963
# MSE values:  33.066299823082154 51.49048035833744 35.092467273746955
"""Using 7 features -- mutual_info_regression"""
# R^2 values:  0.1521747014318099 0.022191535838576915 0.1372446309454285
# MSE values:  25.824099659748182 29.78328586006327 26.27885800302775
# R^2 values:  0.2644034246130077 -0.004815643134610248 0.15356642841251222
# MSE values:  20.908039722048855 28.560118525486544 24.058386524592596
# R^2 values:  0.18610746125386868 0.01171419852982103 0.14386875083175943
# MSE values:  44.13612295703257 53.59319759531907 46.42666234397515
# R^2 values:  0.3937828823766759 0.12049315889917855 0.3669119871975913
# MSE values:  35.63164236786175 51.69480094038597 37.21103380913142
"""Using 6 features -- mutual_info_regression"""
# R^2 values:  0.14167231760892773 0.02227736597575103 0.11730851665604813
# MSE values:  26.143994108480793 29.780671540792614 26.886096549820987
# R^2 values:  0.23491332358818362 -0.004155811279631827 0.1668898346984945
# MSE values:  21.746244009922105 28.541363964773012 23.67969211901606
# R^2 values:  0.12149234975636047 9.676165597716224e-05 0.11423360072723177
# MSE values:  47.64009967406933 54.22319307740011 48.03373031252859
# R^2 values:  0.34018644828531486 0.07520255603716586 0.31238094234401637
# MSE values:  38.781881640588246 54.35684811274382 40.416206727682564
"""Using 5 features -- mutual_info_regression"""
# R^2 values:  0.10656096530035686 0.014768752683261055 0.09988526784383356
# MSE values:  27.21345861105737 30.00937806594495 27.416795167193076
# R^2 values:  0.1890978973353663 0.03462464014440714 0.16330528313275627
# MSE values:  23.04846697292673 27.439097796138082 23.781576696826605
# R^2 values:  0.11612033986151005 -0.00041781305717880635 0.10957289797408587
# MSE values:  47.93141539200279 54.25109766151809 48.28647295358528
# R^2 values:  0.30684730589894427 0.06677171979041252 0.2398984220195497
# MSE values:  40.7414574490373 54.85238763690647 44.676514078039204
"""Using 4 features -- mutual_info_regression"""
# R^2 values:  0.08424251688737305 0.017456683678560903 0.0637267065723307
# MSE values:  27.89326120369195 29.92750577690336 28.518156840883805
# R^2 values:  0.18811118241036617 0.03466766681366584 0.13817626781801473
# MSE values:  23.076512610354264 27.43787483869093 24.495824788724235
# R^2 values:  0.07295688198177952 -0.06912712607335658 0.04513001988087617
# MSE values:  50.27210239125382 57.97709654123158 51.78110972173747
# R^2 values:  0.3005577847832168 0.06384777598596347 0.25010033253200714
# MSE values:  41.111136827177134 55.024248372796464 44.07687606670433
"""Using 3 features -- mutual_info_regression"""
# R^2 values:  0.04900716830019003 0.017226238088467682 0.046667178980135726
# MSE values:  28.966502536544436 29.934524970473976 29.037776792580292
# R^2 values:  0.15549510098669828 0.032800930351917224 0.08929986605330431
# MSE values:  24.003567396634185 27.490933541516327 25.885050600480852
# R^2 values:  0.06690087816966117 -0.08432811951037622 0.051391913306321535
# MSE values:  50.60051003250299 58.801422706500205 51.44153700788133
# R^2 values:  0.2465772262474567 0.016064659689268845 0.22070470695046496
# MSE values:  44.28395379431325 57.832798084788735 45.80466366532489

"""Regression by interfaces"""
# Headarray
print("Headarray: lin 30s") 
apply_SVR(lin_HA_30_dataset)
print("Headarray: lin 60s") 
apply_SVR(lin_HA_60_dataset)
print("Headarray: ang 30s") 
apply_SVR(ang_HA_30_dataset)
print("Headarray: ang 60s") 
apply_SVR(ang_HA_60_dataset)
# Joystick
print("Joystick: lin 30s") 
apply_SVR(lin_JOY_30_dataset)
print("Joystick: lin 60s") 
apply_SVR(lin_JOY_60_dataset)
print("Joystick: ang 30s") 
apply_SVR(ang_JOY_30_dataset)
print("Joystick: ang 60s") 
apply_SVR(ang_JOY_60_dataset)
# SNP
print("SNP: lin 30s") 
apply_SVR(lin_SNP_30_dataset)
print("SNP: lin 60s") 
apply_SVR(lin_SNP_60_dataset)
print("SNP: ang 30s") 
apply_SVR(ang_SNP_30_dataset)
print("SNP: ang 60s") 
apply_SVR(ang_SNP_60_dataset)

"""All features -- mutual_info_regression"""
# Headarray: lin 30s
# R^2 values:  0.2409906504752728 -0.003560308662593936 0.1342231104695476
# MSE values:  7.420445419403434 9.811294814976236 8.464256939870937
# Headarray: lin 60s
# R^2 values:  0.6267189483327006 0.09188675971614968 0.3265734869361101
# MSE values:  2.617048738864259 6.366721802822712 4.721348696537654
# Headarray: ang 30s
# R^2 values:  0.20006871260055137 0.06788060648780725 0.16157308379220292
# MSE values:  34.19107354300926 39.84112540221982 35.836473462231794
# Headarray: ang 60s
# R^2 values:  0.6078308443961444 0.2125367415837247 0.5917714172608283
# MSE values:  36.49289638645415 73.2765815130877 37.9872898187283
# Joystick: lin 30s
# R^2 values:  0.5929624971108352 0.19731475269428966 0.3509219662027844
# MSE values:  0.8284284897223031 1.6336758220750769 1.3210446984217261
# Joystick: lin 60s
# R^2 values:  0.5831460088366397 0.17211229327087774 -13.32349722199925
# MSE values:  2.3355021328602454 4.638395087542564 80.25006122320507
# Joystick: ang 30s
# R^2 values:  0.29044285512011725 0.03974703900438137 0.011812637002992399
# MSE values:  19.74109489319903 26.715881816254406 27.49306471783387
# Joystick: ang 60s
# R^2 values:  0.8203650246638455 0.020329203976260368 -0.2640458481977508
# MSE values:  5.454575331092898 29.7474817840094 38.38246582018569
# SNP: lin 30s
# R^2 values:  0.32344875387101824 -0.016983174639655152 0.19373007865108904
# MSE values:  19.666069305118608 29.561783691977812 23.436746650906766
# SNP: lin 60s
# R^2 values:  0.4308520879033114 0.012782536478429862 0.3467730654031389
# MSE values:  10.722382836969148 18.59854593549248 12.306377873485838
# SNP: ang 30s
# R^2 values:  0.3328298459575849 0.03113180214243383 0.2287505748415254
# MSE values:  33.27676470775144 48.32470226310469 38.46797626734887
# SNP: ang 60s
# R^2 values:  0.590558353646673 0.3847949075726932 0.513361441563905
# MSE values:  20.932440486018944 31.451964153195963 24.879082900802807
"""8 features -- mutual_info_regression"""
# Headarray: lin 30s
# R^2 values:  0.2113434180282482 -0.019147139041299388 0.1242816707412453
# MSE values:  7.710291216885729 9.963679267366851 8.561449301124918
# Headarray: lin 60s
# R^2 values:  0.4865344580102132 0.10059104296220889 0.15978658610828123
# MSE values:  3.599872919111642 6.3056966492820505 5.890680615532528
# Headarray: ang 30s
# R^2 values:  0.14931890105741663 0.06615691130609558 0.10367670315735167
# MSE values:  36.36024802849067 39.914800466130906 38.31111027084529
# Headarray: ang 60s
# R^2 values:  0.5069408846069962 0.21260161160067126 0.3933717430704835
# MSE values:  45.881107561169785 73.27054510055197 56.44916691914994
# Joystick: lin 30s
# R^2 values:  0.5872186435046629 0.1400067060375617 0.3295587958885797
# MSE values:  0.8401187441444932 1.7503127860004768 1.3645243748482396
# Joystick: lin 60s
# R^2 values:  0.5819398162651288 0.17082331484352697 -0.5749247046629449
# MSE values:  2.342260051419552 4.645616829279731 8.823809019002866
# Joystick: ang 30s
# R^2 values:  0.28757259116220135 0.050590080601142784 -0.028921589393399838
# MSE values:  19.82095055185967 26.414209830231623 28.62636065389038
# Joystick: ang 60s
# R^2 values:  0.8214740165557608 0.013638704574950977 0.6792097613768049
# MSE values:  5.420901043528891 29.950637282646465 9.740722923111695
# SNP: lin 30s
# R^2 values:  0.3197314632745982 -0.003249451975061879 0.19829841987111385
# MSE values:  19.774123935000297 29.162570264636525 23.303953583779208
# SNP: lin 60s
# R^2 values:  0.379508684326336 0.014958192920692337 0.32267434647682014
# MSE values:  11.68965973916714 18.557557958905306 12.760382332988462
# SNP: ang 30s
# R^2 values:  0.3129304819028791 0.032190910516521565 0.22620317136890644
# MSE values:  34.26929480141662 48.27187660843326 38.595034328115574
# SNP: ang 60s
# R^2 values:  0.5252961218394843 0.37511237210555715 0.4397816459664159
# MSE values:  24.26893005774628 31.94697754332065 28.640802564729622
"""7 features -- mutual_info_regression"""
# Headarray: lin 30s
# R^2 values:  0.2113434180282482 -0.019147139041299388 0.1242816707412453
# MSE values:  7.710291216885729 9.963679267366851 8.561449301124918
# Headarray: lin 60s
# R^2 values:  0.4865344580102132 0.10059104296220889 0.15978658610828123
# MSE values:  3.599872919111642 6.3056966492820505 5.890680615532528
# Headarray: ang 30s
# R^2 values:  0.14931890105741663 0.06615691130609558 0.10367670315735167
# MSE values:  36.36024802849067 39.914800466130906 38.31111027084529
# Headarray: ang 60s
# R^2 values:  0.5069408846069962 0.21260161160067126 0.3933717430704835
# MSE values:  45.881107561169785 73.27054510055197 56.44916691914994
# Joystick: lin 30s
# R^2 values:  0.5872186435046629 0.1400067060375617 0.3295587958885797
# MSE values:  0.8401187441444932 1.7503127860004768 1.3645243748482396
# Joystick: lin 60s
# R^2 values:  0.5819398162651288 0.17082331484352697 -0.5749247046629449
# MSE values:  2.342260051419552 4.645616829279731 8.823809019002866
# Joystick: ang 30s
# R^2 values:  0.28757259116220135 0.050590080601142784 -0.028921589393399838
# MSE values:  19.82095055185967 26.414209830231623 28.62636065389038
# Joystick: ang 60s
# R^2 values:  0.8214740165557608 0.013638704574950977 0.6792097613768049
# MSE values:  5.420901043528891 29.950637282646465 9.740722923111695
# SNP: lin 30s
# R^2 values:  0.3197314632745982 -0.003249451975061879 0.19829841987111385
# MSE values:  19.774123935000297 29.162570264636525 23.303953583779208
# SNP: lin 60s
# R^2 values:  0.379508684326336 0.014958192920692337 0.32267434647682014
# MSE values:  11.68965973916714 18.557557958905306 12.760382332988462
# SNP: ang 30s
# R^2 values:  0.3129304819028791 0.032190910516521565 0.22620317136890644
# MSE values:  34.26929480141662 48.27187660843326 38.595034328115574
# SNP: ang 60s
# R^2 values:  0.5252961218394843 0.37511237210555715 0.4397816459664159
# MSE values:  24.26893005774628 31.94697754332065 28.640802564729622
"""6 features -- mutual_info_regression"""
# Headarray: lin 30s
# R^2 values:  0.14877532924048265 -0.016426598535123293 0.11191693738425479
# MSE values:  8.321987354932942 9.937081937109989 8.682332962253042
# Headarray: lin 60s
# R^2 values:  0.3389405721240455 -0.026760648050875302 0.18283341477891868
# MSE values:  4.634643881091088 7.198550923211487 5.729100825618438
# Headarray: ang 30s
# R^2 values:  0.09847591011852952 -0.03115119562140789 0.11287574299177083
# MSE values:  38.5334052355171 44.07399350270468 37.91792019007447
# Headarray: ang 60s
# R^2 values:  0.4891666819243562 0.17185959731183464 0.4731665077338367
# MSE values:  47.53506766379607 77.06175122875602 49.0239473743913
# Joystick: lin 30s
# R^2 values:  0.49790399395100216 0.07786302657785982 0.40703435922230125
# MSE values:  1.021897572175392 1.8767915358826617 1.2068412044588606
# Joystick: lin 60s
# R^2 values:  0.5416642601671126 0.1903293767963794 0.17935532074137583
# MSE values:  2.5679113565840663 4.536330483795904 4.597814676980814
# Joystick: ang 30s
# R^2 values:  0.07291645540624991 0.03466776124073201 -0.06424928786660677
# MSE values:  25.79305184904695 26.85719602194547 29.609237724398863
# Joystick: ang 60s
# R^2 values:  0.6892067199329478 0.11071017061463806 0.4961002514430808
# MSE values:  9.437167541292238 27.00308420718181 15.300801710129404
# SNP: lin 30s
# R^2 values:  0.18544617320564039 0.02831485506543885 0.13402433968082494
# MSE values:  23.67754416556567 28.245055363321587 25.172279926798012
# SNP: lin 60s
# R^2 values:  0.2252209471354616 0.006409310696244619 0.17450957023529434
# MSE values:  14.596342079643154 18.718613435154015 15.55171200327217
# SNP: ang 30s
# R^2 values:  0.20687379862993915 0.03746432237585984 0.11639749552965861
# MSE values:  39.559134692446534 48.00885212421883 44.071864513546764
# SNP: ang 60s
# R^2 values:  0.4943057948535784 0.32109897162337986 0.530387206319172
# MSE values:  25.853290566874712 34.70837785789686 24.00865164242261
"""5 features -- mutual_info_regression"""
# Headarray: lin 30s
# R^2 values:  0.12907425195501254 -0.02382295880516061 0.10740067102366924
# MSE values:  8.514594690787 10.009392360849066 8.726486183882258
# Headarray: lin 60s
# R^2 values:  0.06822757976726146 -0.05999094390242177 0.05958358325487112
# MSE values:  6.5325947470057635 7.431526327298933 6.593197234248166
# Headarray: ang 30s
# R^2 values:  0.07628360021341285 -0.1481432150751183 0.08809515170330773
# MSE values:  39.48196033269534 49.07452642859026 38.977104938226475
# Headarray: ang 60s
# R^2 values:  0.44697627740566004 0.17780760717889266 0.44082274323476023
# MSE values:  51.4610522513209 76.50826530391339 52.033663028329556
# Joystick: lin 30s
# R^2 values:  0.4646648999053865 0.09035491333043011 0.321011712044062
# MSE values:  1.089547880278439 1.8513672572774744 1.3819199408847476
# Joystick: lin 60s
# R^2 values:  0.529943218463939 0.10764812321583717 0.2161888476262177
# MSE values:  2.63358067600382 4.999567608012955 4.39144798162968
# Joystick: ang 30s
# R^2 values:  0.06010167663349453 0.0003632029087231592 -0.062285849455405495
# MSE values:  26.149580939922558 27.81160758158963 29.55461150539428
# Joystick: ang 60s
# R^2 values:  0.6919845132758675 0.02586421593644972 0.5655261975989503
# MSE values:  9.35282047572324 29.57941239975471 13.1926985830466
# SNP: lin 30s
# R^2 values:  0.14991817965760368 0.01715934705167399 0.11107880092524725
# MSE values:  24.710275961398168 28.569324951158073 25.83926348198674
# SNP: lin 60s
# R^2 values:  0.13018463704405092 0.0035417635424841443 0.10260936601448523
# MSE values:  16.386765409949433 18.772636190455728 16.906265888696616
# SNP: ang 30s
# R^2 values:  0.18611652533679302 0.040655998631925816 0.12321132768960297
# MSE values:  40.594455135313055 47.84965936184345 43.7320077496152
# SNP: ang 60s
# R^2 values:  0.4789405669538902 0.1746923564618852 0.4483100318104837
# MSE values:  26.638827947912866 42.193321770961425 28.204794330802432





# Support Vector Classification

# Convert SPARC values to labels
def SPARC_to_Labels(dataset):
    sparc = dataset[:,-1].copy()
    cutoff = np.percentile(sparc,50.0)
    # High = 1, low = 0
    for i in range(len(sparc)):
        if sparc[i]>=cutoff:
            sparc[i] = 1
        else:
            sparc[i] = 0
    dataset[:,-1] = sparc
    return dataset

lin_30_dataset_class = SPARC_to_Labels(lin_30_dataset)
lin_60_dataset_class = SPARC_to_Labels(lin_60_dataset)
ang_30_dataset_class = SPARC_to_Labels(ang_30_dataset)
ang_60_dataset_class = SPARC_to_Labels(ang_60_dataset)



def apply_SVM(dataset):
    clf = svm.SVC()
    # Separate features and targets
    X = dataset[:,:9]
    y = dataset[:,-1]
    clf = svm.SVC()
    sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for train_index, test_index in sss.split(X,y):
            X_train, X_test = X[train_index],X[test_index]
            y_train, y_test = y[train_index],y[test_index]
    # Classification 
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    print(acc)
    # print(cm)

# apply_SVM(lin_30_dataset_class) # Acc: 0.660828025477707
# apply_SVM(lin_60_dataset_class) # Acc: 0.7757847533632287
# apply_SVM(ang_30_dataset_class) # Acc: 0.6401273885350318
# apply_SVM(ang_60_dataset_class) # Acc: 0.7309417040358744

""" 
Look at interfaces individually 
(time and frequency domain) Frequency and number of interactions -> smoothness
Velocity, smoothness 

"""

