import pandas as pd
df = pd.read_csv("F:\\argall-lab-data\\ECG Data\\laa_wc_multi_session_s00\\LAA_WC_Multi_Session\\s00\\ecg_lead_i\\d14spkb8\\2018-09-28T17-00-16-592Z\\elec.csv")
print("Read done")
print(df.columns[0])
# Joystick - teleop 
# startIndex = df.index[df[df.columns[0]]==1537817422990].values[0]
# endIndex = df.index[df[df.columns[0]]==1537817596038].values[0]

# Joystick - mid level autonomy 
# startIndex = df.index[df[df.columns[0]]==1537814963852].values[0]
# endIndex = df.index[df[df.columns[0]]==1537816166452].values[0]

# Sip puff - teleop 
startIndex = df.index[df[df.columns[0]]==1538158795354].values[0]
endIndex = df.index[df[df.columns[0]]==1538159568423].values[0]

# Test with some other data 
# startIndex = 5000
# endIndex = 25000
print(startIndex)
print(endIndex)
print(str((endIndex-startIndex)/250)+"s")
new_df= df.iloc[startIndex:endIndex+1,1]
#Convert to mV
new_df = new_df.apply(lambda x:x*1000)

#Save mV values 
savedir = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\"
new_df.to_csv(savedir+"testData.csv",index=False,header=False) #do not save the index values 
