# Extract HRV metrics and label them with subject, interface, autonomy level and TLX labels
import pandas as pd
import numpy as np
import os, glob
from pathlib import Path
from sklearn import svm 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import re

win = "60s"

# Change paths to match local
# Create HRV Feature Set
# savedir = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\HRV_allSubj\\Extracted-with_tlx_labels\\"+win+"\\"
savedir = "/home/skrdown/Documents/argall-lab-data/HRV_newgen/Extracted_with_tlx_labels/" + win + os.sep
# Read TLX Label csv
# tlxLabels_df = pd.read_csv("C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\TLX\\tlxLabels.csv")
tlxLabels_2_path = "/home/skrdown/HRV-ML-Research/TLX/"+"tlxLabels_2.csv"
tlxLabels_3_path = "/home/skrdown/HRV-ML-Research/TLX/"+"tlxLabels_3.csv"
tlxLabels_2_df = pd.read_csv(tlxLabels_2_path)
tlxLabels_3_df = pd.read_csv(tlxLabels_3_path)

## Loop through each id and interface in sub_tlxLabels, extract HRV metrics from corresponding csv file
# HRV_dir = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\HRV_allSubj\\"+win+"\\"
HRV_dir = "/home/skrdown/Documents/argall-lab-data/HRV_newgen/HRV_byEvent/"+ win + os.sep
HRV_filesList = glob.glob(HRV_dir+"*.csv")
print(HRV_filesList)

def mk_tlx_hrv_dataframe(tlxLabels_df, savedir, TLX_levels):
    for idx, row in tlxLabels_df.iterrows():
        # Get labels 
        rLabel = row["Raw Label"]
        wLabel = row["Weighted Label"]
        # Subject 
        subject = row["id"]
        # Skip over subject u00
        if subject == "u00":
            continue
        # Get interface
        if row["interface"] == "Sip N Puff":
            # interface = "Sip-n-puff"
            interface = "SNP"
        elif row["interface"] == "Joystick":
            # interface = "Joystick"
            interface = "JOY"
        elif row["interface"] == "Head Array":
            # interface = "Headarray"
            interface = "HA"
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
        print(subject,interface,autonomy)
        HRVpath = [path for path in HRV_filesList if ((subject.lower() in path.lower()) & (interface.lower() in path.lower())   
        &  (alevel.lower() in path.lower()) )]
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
        # Directory to save to 
        sub_savedir = savedir + TLX_levels + os.sep
        Path(sub_savedir).mkdir(parents=True,exist_ok=True)
        # Check if any HRV_metrics_df are empty 
        if HRV_metrics_df.shape[0] == 0:
            HRV_metrics_df.to_csv(sub_savedir+subject+interface+alevel+"NONE.txt")
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
            HRV_metrics_df.to_csv(sub_savedir+subject+"_"+interface+"_"+alevel+".csv")

mk_tlx_hrv_dataframe(tlxLabels_2_df, savedir, "2")
mk_tlx_hrv_dataframe(tlxLabels_3_df, savedir, "3")