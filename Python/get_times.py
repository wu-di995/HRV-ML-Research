# Get the total times and amount of time that can be used for HRV

# Total task time
# Total HRV time 
# Total consecutive HRV time 

import pandas as pd 
import glob

filepaths = glob.glob("C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\HRV_multiSubj\\*.csv")
# print(len(filepaths))

index = range(len(filepaths))
columns = ['SubjID','Total Task Time', 'Total HRV Time', 'Longest Consec. HRV Time', "Null List"]
times_df = pd.DataFrame(index=index,columns=columns)


for row, path in enumerate(filepaths):
    df = pd.read_csv(path)
    # Subject ID and task 
    subj = path.split("\\")[-1].split("_")[0] + "-" + path.split("\\")[-1].split("_")[1]
    # Get the "t_start" column
    t_start = df['t_start']
    total_task_time = df.shape[0]-1+5
    total_HRV_time = t_start.count()-1
    na_list = df[t_start.isnull()].index.tolist()
    consec_times = []
    # print(na_list)
    if len(na_list) == 0:
        longest_consec_HRV = total_HRV_time
    elif na_list[-1] ==0:
        longest_consec_HRV = total_HRV_time - 1
    else: 
        for i,na in enumerate(na_list):
            if i == 0:
                if na == 0:
                    continue 
                else: 
                    consec_times.append(na-1)
            elif i == len(na_list)-1:
                if na == len(t_start)-1:
                    continue
                else:
                    consec_times.append(na-na_list[i-1]-1)
                    consec_times.append(len(t_start)-na-1)
            else:
                consec_times.append(na-na_list[i-1]-1)
        # print(consec_times)
        longest_consec_HRV = max(consec_times)
    
    # Amend dataframe 
    times_df.loc[row,"SubjID"] = subj
    times_df.loc[row,"Total Task Time"] = total_task_time
    times_df.loc[row,"Total HRV Time"] = total_HRV_time
    times_df.loc[row,"Longest Consec. HRV Time"] = longest_consec_HRV
    times_df.loc[row,"Null List"] = na_list

# Save times_df 
times_df.to_csv("C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\RawECG\\HRV_times_for_best_subjs.csv")