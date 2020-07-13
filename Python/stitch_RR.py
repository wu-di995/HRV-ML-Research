# Stitch together RR intervals that are considered "good"
# "good" means ... 

import pandas as pd 
import glob 
import numpy as np

RRpaths = glob.glob("C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\RR\\*.csv")
beatspaths = glob.glob("C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\Beats\\*.csv")
times_df_path = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\RawECG\\HRV_times_for_best_subjs.csv"
savedir = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\RR_edited\\"
times_df = pd.read_csv(times_df_path)

for i,path in enumerate(RRpaths):
    # RR intervals (s)
    RR_df = pd.read_csv(path,header=None)
    RR_int = RR_df.iloc[:,1].values
    # Time of R peaks (s)
    beats_df = pd.read_csv(beatspaths[i],header=None)
    # RR_t = beats_df.columns.values.astype(float)[1:] # Start from the second beat
    RR_t = beats_df.iloc[0,1:].values.astype(float) # Start from the second beat
    if len(RR_int) - len(RR_t) != 0:
        print("RR lengths do not match!")

    subjID = path.split("\\")[-1].split("_")[0]
    print(subjID)
    na_list_str = times_df.loc[times_df['SubjID']==subjID]["Null List"].values[0]
    
    # If there is nothing to delete, copy over RR
    if na_list_str=="[]":
        newRR_t = np.zeros(len(RR_t))
        newRR_int = np.copy(RR_int)
        newRR_t[0] = RR_t[0]
        for i in range(1,len(newRR_int)): # Current RR_t = Prev RR_t + Prev RR_int 
            newRR_t[i] = newRR_t[i-1] + newRR_int[i-1]
    else:
        newRR_t_p = np.copy(RR_t)
        newRR_int = np.copy(RR_int) # Placeholder
        na_list = list(map(int,na_list_str.strip("[]").split(",")))
        for i,time in enumerate(na_list):
            # For each null time, eliminate RR ints from null time (inclusive) to null time +1 (exlcusive)
            indices_to_keep = np.where(~((newRR_t_p>=time) & (newRR_t_p<time+1)))
            newRR_t_p = newRR_t_p[indices_to_keep]
            newRR_int = newRR_int[indices_to_keep]
        
            if len(newRR_int) - len(newRR_t_p):
                print("New RR array lengths are not equal (1) !")

            # Make new RR_t array
            newRR_t = np.zeros(len(newRR_t_p))
            newRR_t[0] = newRR_t_p[0] # First time is the same 
            for i in range(1,len(newRR_int)): # Current RR_t = Prev RR_t + Prev RR_int 
                newRR_t[i] = newRR_t[i-1] + newRR_int[i-1]
            if len(newRR_int) - len(newRR_t):
                print("New RR array lengths are not equal (2) !")

    # Create dataframe for export 
    index = range(len(newRR_t))
    columns = ['Time','RR_int']
    newRR_df = pd.DataFrame(index=index,columns=columns)
    newRR_df['Time'] = newRR_t
    newRR_df['RR_int'] = newRR_int
    newRR_df.to_csv(savedir+subjID+"_edit-RR.csv")
        
print("end")