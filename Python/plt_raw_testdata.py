import pandas as pd
import matplotlib.pyplot as plt 

rawECG = pd.read_csv("C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\testData.csv").values
plt.plot(rawECG)
plt.show()