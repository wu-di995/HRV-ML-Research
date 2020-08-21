# Get list of window start and end times
# Imports
import pandas as pd
import numpy as np
import glob,os, pathlib

# ECG directory
ECGDirs = glob.glob("E:\\argall-lab-data\\ECG_byEventNew\\*\\")
teleopPaths = [path for ECGDir in ECGDirs for path in glob.glob(ECGDir+"*Teleoperation.csv") ]
# print(len(teleopPaths)) #60

# Save directory 
savedir = "E:\\argall-lab-data\\ECG_eventStartEndTimes\\"

def readEvent(path):
    subject = path.split("\\")[-2]
    if "Headarray" in path:
        interface = "HA"
    elif "Joystick" in path:
        interface = "JOY"
    elif "Sip-n-puff" in path:
        interface = "SNP"
    if "Teleoperation" in path:
        autonomy = "A0"
    elif "Low level autonomy" in path:
        autonomy = "A1"
    elif "Mid level autonomy" in path:
        autonomy = "A1"
    event = subject+"_"+interface+"_"+autonomy
    return event

# Generator for sliding windows
def slidingWin_gen(timeSeries,windowSize,stepSize):
    for i in range(0,len(timeSeries)-windowSize+1,stepSize):
        window = timeSeries[i:i+windowSize]
        yield window

def read_WinStartEnd(ECGPath,windowSizeSecs,stepSizeSecs):
    # Sampling frequency = 250Hz
    samp_freq = 250 
    ECG_df = pd.read_csv(ECGPath,header=None)
    no_samples = ECG_df.shape[0]
    windowSize = windowSizeSecs*samp_freq
    stepSize = stepSizeSecs*samp_freq
    # Get timestamps series
    timestamps = ECG_df.iloc[:,0].values
    # Start times list 
    startTimes = []
    # End times list
    endTimes = []
    for window in slidingWin_gen(timestamps,windowSize,stepSize):
        startTimes.append(window[0])
        endTimes.append(window[-1])
    return startTimes,endTimes

for path in teleopPaths:
    event = readEvent(path)
    startTimes30, endTimes30 = read_WinStartEnd(path,30,1)
    startTimes60, endTimes60 = read_WinStartEnd(path,60,1)
    startEnd30_df = pd.DataFrame({"Start Times":startTimes30, "End Times":endTimes30})
    startEnd60_df = pd.DataFrame({"Start Times":startTimes60, "End Times":endTimes60})
    startEnd30_df.to_csv(savedir+event+"_startEnd30.csv")
    startEnd60_df.to_csv(savedir+event+"_startEnd60.csv")