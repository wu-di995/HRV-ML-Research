# Perform ANOVA test across all tasks for each weighted HRV metric 

import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 
from scipy import stats 
import matplotlib.pyplot as plt 

# Weighted HRV metrics directory 
cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
weighted_HRV_df = pd.read_csv(str(mainDir)+"\\HRV_weightedWindows\\HRV_weightedAvg.csv")

# For each HRV metric, conduct an ANOVA test
HRV_metrics_list = ["SDNN","RMSSD","ulf","vlf","lf","hf","lfhf","SD1","SD2","SD1SD2","ApEn"]


# Get HRV_metrics by task and metric
def get_task_metric_df(weighted_HRV_df,task,metric):
    task_mask = weighted_HRV_df["Task"].str.contains("_"+str(task))
    task_df = weighted_HRV_df[task_mask]
    task_metric_df = task_df[metric]
    return task_metric_df

# task_df = get_task_metric_df(weighted_HRV_df,1,"SDNN")
# task_df.to_csv(str(mainDir)+"\\HRV_weightedWindows\\task_df.csv")

for metric in HRV_metrics_list:
    print(metric)
    stat, p = stats.f_oneway(get_task_metric_df(weighted_HRV_df,1,metric),
                    get_task_metric_df(weighted_HRV_df,2,metric),
                    get_task_metric_df(weighted_HRV_df,3,metric),
                    get_task_metric_df(weighted_HRV_df,4,metric),
                    get_task_metric_df(weighted_HRV_df,5,metric),
                    get_task_metric_df(weighted_HRV_df,6,metric),
                    get_task_metric_df(weighted_HRV_df,7,metric)
    )
    print("Stat: ", stat)
    print("p:", p)
    print("--------------")


# Box plot
for metric in HRV_metrics_list:
    fig,ax = plt.subplots()
    tasks_list = []
    ax.set_title(metric)
    for i in range(7):
        tasks_list.append(get_task_metric_df(weighted_HRV_df,i+1,metric))
    ax.set_xlabel("Task")
    plt.boxplot(tasks_list)
    # plt.show()
    plt.savefig(str(mainDir)+"\\HRV_weightedWindows\\"+metric+"_boxplot.png")
    plt.close()

# SDNN
# Stat:  0.5601784326082074
# p: 0.7614678870022267
# --------------
# RMSSD
# Stat:  0.14797678795232877
# p: 0.9892269832309286
# --------------
# ulf
# Stat:  2.039690197191007
# p: 0.06356248382374596
# --------------
# vlf
# Stat:  2.073249115283789
# p: 0.05938173161874236
# --------------
# lf
# Stat:  0.6877215494782158
# p: 0.6598232102002427
# --------------
# hf
# Stat:  0.08558933939885015
# p: 0.997596008676386
# --------------
# lfhf
# Stat:  0.4416710994643901
# p: 0.8499920815506625
# --------------
# SD1
# Stat:  0.14729957543593505
# p: 0.9893578583407339
# --------------
# SD2
# Stat:  0.8660503220272804
# p: 0.5213723126308875
# --------------
# SD1SD2
# Stat:  0.2913344690472694
# p: 0.940352748403852
# --------------
# ApEn
# Stat:  1.7969647581617207
# p: 0.10308923201540694
# --------------