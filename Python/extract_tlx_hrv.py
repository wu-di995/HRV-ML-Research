# Extract HRV metrics and label them with subject, interface, autonomy level and TLX labels
import pandas as pd
import numpy as np
import glob 
from sklearn import svm 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import re

win = "60s"

# Create HRV Feature Set 
savedir = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\HRV_multiSubj\\Extracted-with_tlx_labels\\"+win+"\\"
## Read TLX label csv
tlxLabels_df = pd.read_csv("C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\TLX\\tlxLabels.csv")
## Select subjects 
subjectsMask = tlxLabels_df["id"].str.contains("s01|s03|s05|u03|u04|u09|u13|u14",flags=re.IGNORECASE)
sub_tlxLabels_df = tlxLabels_df.loc[subjectsMask,:] ## tlx Labels with best subjects only 
## Loop through each id and interface in sub_tlxLabels, extract HRV metrics from corresponding csv file
HRV_dir = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\HRV_multiSubj\\"+win+"\\"
HRV_filesList = glob.glob(HRV_dir+"*.csv")
for idx, row in sub_tlxLabels_df.iterrows():
    # Get labels 
    rLabel = row["Raw Label"]
    wLabel = row["Weighted Label"]
    # Subject 
    subject = row["id"]
    # Get interface
    if row["interface"] == "Sip N Puff":
        interface = "Sip-n-puff"
    elif row["interface"] == "Joystick":
        interface = "Joystick"
    elif row["interface"] == "Head Array":
        interface = "Headarray"
    # Get autonomy level 
    if row["autonomy"] == "A0":
        autonomy = "Teleoperation"
        alevel = "A0"
    elif row["autonomy"] == "A1":
        autonomy = "Low level"
        alevel = "A1"
    elif row["autonomy"] == "A2":
        autonomy = "Mid level"
        alevel = "A2"
    HRVpath = [path for path in HRV_filesList if ((subject.lower() in path.lower()) & (interface.lower() in path.lower())   
    &  (autonomy.lower() in path.lower()) )]
    # print(HRVpath)
    # Read HRV metrices csv 
    HRV_df = pd.read_csv(HRVpath[0])
    # Select non-empty rows 
    HRV_mask = ~(HRV_df["NNmean"].isnull())
    # print(HRV_mask.sum())
    # Get HRV metrics 
    HRV_metrics_df= HRV_df.loc[HRV_mask,["SDNN","RMSSD","ulf","vlf","lf","hf","lfhf","SD1","SD2","SD1SD2",
    "ApEn"]]
    print(HRV_metrics_df.shape)
    # Check if any HRV_metrics_df are empty 
    if HRV_metrics_df.shape[0] == 0:
        HRV_metrics_df.to_csv(savedir+subject+interface+alevel+"NONE.txt")
    else: 
        for col in HRV_metrics_df:
            # print(col,subject,interface,autonomy)
            # print(HRV_metrics_df[col].values[0])
            if type(HRV_metrics_df[col].values[0]) == str:
                print(col,subject,interface,alevel)
                # If the number is complex, extract the real part and replace it
                HRV_metrics_df[col] = HRV_metrics_df[col].apply(lambda x: complex(x.replace("i","j")).real)
        # Append subject, interface, autonomy level, raw Label, weighted Label to dataframe 
        HRV_metrics_df["Subject"] = [subject]*HRV_metrics_df.shape[0]
        HRV_metrics_df["Interface"] = [interface]*HRV_metrics_df.shape[0]
        HRV_metrics_df["Autonomy Level"] = [alevel]*HRV_metrics_df.shape[0]
        HRV_metrics_df["Raw Label"] = [rLabel]*HRV_metrics_df.shape[0]
        HRV_metrics_df["Weighted Label"] = [wLabel]*HRV_metrics_df.shape[0]
        # Export HRV_metrics_df as csv
        HRV_metrics_df.to_csv(savedir+subject+"_"+interface+"_"+alevel+".csv")