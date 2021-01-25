# HRV-ML-Research

## Overview 
This project investigates the use of heart-rate variability (HRV) as indicators of cognitive load for users performing navigational tasks on an assistive robotic wheelchair.

Our main results are as follows: <br>
1. HRV can be used to classify different levels of TLX scores 
2. HRV correlate well with frequency features derived from user commands 

The excel sheets in Results folder contains all results.

## Usage instructions 
### Setup 
1. Clone the Physionet Cardiovascular Signal Toolbox <br>
   `git clone https://github.com/cliffordlab/PhysioNet-Cardiovascular-Signal-Toolbox.git`
2. Clone this repo 
3. Ensure that you also have the raw ECG and raw trajectory data 
4. Replace paths in paths.py for your local machine 
5. Copy and paste MATLAB scripts from the MATLAB to the Physionet Toolbox folder

### Generating HRV Data 
1. Ensure that paths have been properly set in paths.py 
2. Run the following python scripts:<br>
   a) get_ECG.py <br>
   b) splitECG_byEvent.py <br>
   c) splitECG_byTask.py <br>
3. Run the following MATLAB scripts from the Physionet Toolbox <br>
   a) HRV_all_subjs_byEvent.m <br>
   b) HRV_all_subjs_byTask.m <br>

### SVM Classification of TLX using HRV 
1. Ensure that paths have been properly set in paths.py 
2. Run the following python scripts: <br>
   a) mk_tlx_labels.py <br>
   b) extract_tlx_hrv.py <br>
   c) feature_impt.py <br>
   d) svm_tlx_hrv.py <br>

### SVR of User Command Frequency FEatures using HRV
1. Ensure that paths ahve been properly set in paths.py 
2. Run export_userCmdFreq.m from Physionet Toolbox 
3. Run the following python scripts: <br>
   a) get_ECGwinStartEndTimes.py <br>
   b) get_userControlledTimeWin.py <br>
   c) get_userCmdFreqWin.py <br>
   d) svr_userCmdFreq_hrv.py <br>