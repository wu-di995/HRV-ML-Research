# All paths should end with "/"

# Raw ECG data 
RawECG_path = "/home/skrdown/Documents/argall-lab-data/ECG Data/"
# ECG combined by subject 
ECG_bySubj_path = "/home/skrdown/Documents/argall-lab-data/ECG_bySubj/"
# ECG start and end times 
ECG_startEnd_path = "/home/skrdown/Documents/argall-lab-data/ECG_eventStartEndTimes/"
# ECG by event, ECG values in volts 
ECG_byEvent_V_path = "/home/skrdown/Documents/argall-lab-data/ECG_byEvent/"
# ECG by event, for HRV generation 
ECG_byEvent_forHRV_path = "/home/skrdown/Documents/argall-lab-data/ECG_byEvent_forHRV/"
# ECG by tasks, for HRV generation
ECG_byTasks_forHRV_path = "/home/skrdown/Documents/argall-lab-data/ECG_byTask_forHRV/" 

# HRV by event
HRV_byEvent_path = "/home/skrdown/Documents/argall-lab-data/HRV_newgen/HRV_byEvent/"
# HRV by tasks
HRV_byTasks_path = "/home/skrdown/Documents/argall-lab-data/HRV_newgen/HRV_byTask/"
# HRV by event, labelled with TLX
HRV_byEvent_TLX_path = "/home/skrdown/Documents/argall-lab-data/HRV_newgen/Extracted_with_tlx_labels/"

# TLX labels
TLX_label_path = "/home/skrdown/HRV-ML-Research/TLX/"

# Trajectory Data
Traj_path = "/home/skrdown/Documents/argall-lab-data/Trajectory Data/"

# User command frequencies by event
UserCmdFreqs_path = "/home/skrdown/Documents/argall-lab-data/HRV_newgen/UserCmdFreq_byEvent/"

# User controlled by event
UserCtrl_path = "/home/skrdown/Documents/argall-lab-data/HRV_newgen/UserControlled_byEvent/"

# Plots 
Plots_featImpt_path = "/home/skrdown/Documents/argall-lab-data/HRV_newgen/Plots/Feature Importance/"


# Results - feature_impt, svm_results, svr_results 

"""
    Files that are needed 
    1. extract_tlx_hrv.py 
    2. feature_impt.py
    3. get_ECGwinStartEndTimes.py
    4. get_userCmdFreqWin.py
    5. mk_tlx_labels.py
    6. splitECG_byEvent.py
"""