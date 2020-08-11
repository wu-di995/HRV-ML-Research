from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import glob,os, pathlib

# SPARC files
sparc_paths = glob.glob("E:\\argall-lab-data\\SPARC_linVel_byEvent\\60s\\*.csv")

# HRV files 
hrv_paths = glob.glob("E:\\argall-lab-data\\HRV_byEvent\\60s\\*.csv")

for sparc_path in sparc_paths:
    subj = sparc_path.split("\\")[-1].split("_")[0]
    interface = sparc_path.split("\\")[-1].split("_")[1]
    autonomy = sparc_path.split("\\")[-1].split("_")[2]
    event = subj+"_"+interface+"_"+autonomy
    print(event)
    # SPARC dataframe
    sparc_df = pd.read_csv(sparc_path)
    # HRV datafraem
    hrv_path = [path for path in hrv_paths if event in path][0]
    hrv_df = pd.read_csv(hrv_path)
    # Indices where there are HRV metrics 
    hrv_mask = ~(hrv_df["NNmean"].isnull)
    hrv_df = hrv_df[hrv_mask]
    