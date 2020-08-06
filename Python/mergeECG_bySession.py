# Merge task raw ECG csv files by session
import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 

# Load directory 
cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
HRV_byTask_dirs = glob.glob(str(mainDir)+"\\RawECG_byTasks\\*\\")

# Save directory 
savedir = str(mainDir)+"\\RawECG_bySession\\"

# For every subject folder, remove the last number in each csv filename, find unique filenames 
# Combine files of the same unique filename, check that there are 7 tassks

for subjFolder in HRV_byTask_dirs:
    csvfilesList_o = glob.glob(subjFolder+"*.csv")
    csvfilesList = [filename[::-1] for filename in csvfilesList_o] # Reverse every string in list 
    csvfilesList = [filename.split("_",1)[-1] for filename in csvfilesList] # Split strings into two segments at first "_"
    csvfilesList = [filename[::-1] for filename in csvfilesList] # Reverse back every string in list 
    sessions = list(set(csvfilesList)) # Get unique filename headers 
    for session in sessions:
        csvfiles = [filename for filename in csvfilesList_o if session in filename]
        csvfiles = sorted(csvfiles)
        sessionName = session.split("\\")[-1]
        for i,csv in enumerate(csvfiles):
            if i == 0:
                session_df = pd.read_csv(csv,header=None)
                # print(session_df.shape)
            else:
                df = pd.read_csv(csv,header=None)
                # print(df.shape)
                session_df = pd.concat([session_df,df],axis=0)
                session_df.to_csv(savedir+sessionName+".csv",index=None,header=None)
        print(session_df.shape)
