# Analyze HRV metrics by tasks
# For each task, find windows that are overlapping with task
# Get weighted average of 

import pandas as pd
import numpy as np
import glob,os, pathlib
from pathlib import Path 

# Load directory 
win = "30s"
cwd = os.getcwd()
mainDir = pathlib.Path(cwd).parent
HRV_pathsList = glob.glob(str(mainDir)+"\\HRV_allSubj\\"+win+"\\*.csv")
HRV_pathsList = [path for path in HRV_pathsList if (("Teleoperation" in path) and ("u10" not in path))]
print((HRV_pathsList))

# Task time directory 

# Read task 1 start time, which is time = 0s for HRV metrics 
# For each task, select overlapping HRV segments:
    # overlap means start or end time HRV segment is within task start end time 
# For each overlapping 

