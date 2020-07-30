# Breakdown valid HRV segments by subject and task 
import pandas as pd
import numpy as np
import glob,os, pathlib
import re
from pathlib import Path 
import matplotlib.pyplot as plt 

win = "30s"
cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
HRV_pathsList = glob.glob(str(mainDir)+"\\HRV_allSubj_byTasks\\"+win+"\\*.csv")

HA_event_dict = {}
JOY_event_dict = {}
SNP_event_dict = {}


for path in HRV_pathsList:
    pathname = path.split("\\")[-1].split("_")
    event = pathname[1]+"_"+pathname[2]+"_"+pathname[3]
    if "HA" in event:
        if event in HA_event_dict.keys():
            HA_event_dict[event] +=1
        else:
            HA_event_dict[event] = 1
    elif "JOY" in event:
        if event in JOY_event_dict.keys():
            JOY_event_dict[event] +=1
        else:
            JOY_event_dict[event] = 1
    elif "SNP" in event:
        if event in SNP_event_dict.keys():
            SNP_event_dict[event] +=1
        else:
            SNP_event_dict[event] = 1

fig, ax = plt.subplots()
ax.bar(range(len(HA_event_dict)),list(HA_event_dict.values()))
ax.set_xticks(range(len(HA_event_dict)))
ax.set_xticklabels(list(HA_event_dict.keys()))
plt.show()



