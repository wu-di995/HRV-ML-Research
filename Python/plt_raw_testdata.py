import pandas as pd
import matplotlib.pyplot as plt 

path = "C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\RawECG\\"

filepath = path + "u14" + "\\"  + "Sip-n-puff - WST" +".csv"

rawECG = pd.read_csv(filepath).values
plt.plot(rawECG)
plt.show()