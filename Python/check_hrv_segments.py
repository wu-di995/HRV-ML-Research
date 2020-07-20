# Checking that HRV segments generated correspond to "good segments"

# Import necessary libraries 
import matplotlib.pyplot as plt
import numpy as np
import os,glob,pathlib
import pandas as pd
import pickle

cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
ECGdir = str(mainDir)+"\\RawECG\\"
HRVpathsListAll = glob.glob(str(mainDir)+"\\HRV_multiSubj\\*.csv")
savedir = str(mainDir)+"\\Plots\\HRV_rawECG_segments_check\\"
ignoreCases = ["Command","WST","Trajectory"] # Ignore these cases
HRVpathsList = [path for path in HRVpathsListAll if not any(case in path for case in ignoreCases)]

def hrv_checker(ECGdir,HRVpathsList):
    for i,path in enumerate(HRVpathsList):
        # Read the HRV file
        HRV_df = pd.read_csv(path)
        subj = path.split("\\")[-1].split("_")[0]
        session = path.split("\\")[-1].split("_")[1]
        # Read Raw ECG file
        ECG_df = pd.read_csv(ECGdir+subj+"\\"+session+".csv",header=None)
        ECG = ECG_df.values
        print("Read",i)
        # Find the segments where no HRV metrics are generated 
        t_start = HRV_df["t_start"]
        null_list = HRV_df[t_start.isnull()].index.tolist()
        # Plot Raw ECG with background as yellow if no HRV metrics are generated 
        fig,ax = plt.subplots()
        time = np.linspace(0,(len(ECG)-1)*0.004,len(ECG)) #4ms persample
        ax.plot(time,ECG)
        for nulltime in null_list:
            ax.axvspan(nulltime,nulltime+1,facecolor="yellow")
        ax.set_title(subj+" "+session)
        # plt.show()
        pickle.dump(fig, open(savedir+subj+"_"+session+'.fig.pickle', 'wb'))
        plt.savefig(savedir+subj+"_"+session+".png")
        plt.close()
    print("done")

# hrv_checker(ECGdir,HRVpathsList)

# To open interactive figures
figures = glob.glob(savedir+"*.pickle")
fig = pickle.load(open(figures[0],'rb'))
print(figures[0])
plt.show()
