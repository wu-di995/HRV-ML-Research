# Create plots for tlx scores 

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

tlx_df = pd.read_csv("C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\TLX\\tlx.csv")

# Exclude Pilot subjects
pilot_mask = tlx_df["id"].str.contains(r'PILOT',na=False)
non_pilots = ~pilot_mask

raw_scores = tlx_df.loc[non_pilots,"Raw Score"]
weighted_scores = tlx_df.loc[non_pilots,"Weighted Score"]

# print(tlx_df.shape)
# print(pilot_mask.sum())
# print(len(raw_scores))
# print(len(weighted_scores))

# Histogram plots of Raw and Weighted scores 

def plot_histograms(scores,scType): 
    # scType == score type (raw or weighted)
    fig1,ax1 = plt.subplots(1,2,tight_layout=True)
    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = ax1[0].hist(scores)
    print(N)
    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()
    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())
    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    # Normalize by total number of sessions
    N1, bins1, patches1 = ax1[1].hist(scores, weights=np.ones(len(scores))/len(scores))
    print(N1)
    # Labels
    if scType =="Raw":
        xlabel = "Raw Score"
    elif scType == "Weighted":
        xlabel = "Weighted Score"

    ax1[0].set_xlabel(xlabel)
    ax1[0].set_ylabel("No. of Sessions")
    ax1[1].set_xlabel(xlabel)
    ax1[1].set_ylabel("Percentage of total sessions")
    ax1[1].yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

# Convert pd series to np arrays
raw_scores_ar = raw_scores.values
weighted_scores_ar = weighted_scores.values

# Get 1/3 and 2/3 percentiles 
raw_lowcut = np.percentile(raw_scores_ar,33.3)
raw_highcut = np.percentile(raw_scores_ar,66.7)
print("Raw score 33.3% and 66.7% percentiles:",round(raw_lowcut,2),round(raw_highcut,2))

weighted_lowcut = np.percentile(weighted_scores_ar,33.3)
weighted_highcut = np.percentile(weighted_scores_ar,66.7)
print("Raw score 33.3% and 66.7% percentiles:",round(weighted_lowcut,2),round(weighted_highcut,2))

# Produce labels for each session and save as csv 
columns = ["id","interface","Raw Label", "Weighted Label"]
index = range(non_pilots.sum())
tlxLabels_df = pd.DataFrame(index=index,columns=columns)
tlxLabels_df["id"] = tlx_df.loc[non_pilots,"id"].values
tlxLabels_df["interface"] = tlx_df.loc[non_pilots,"interface"].values

for idx in index:
    rScore = raw_scores_ar[idx]
    wScore = weighted_scores_ar[idx]
    # Create raw score label
    if rScore< raw_lowcut:
        rLabel = "Low"
    elif rScore< raw_highcut:
        rLabel = "Med"
    else:
        rLabel = "High"
    # Create weighted score label
    if wScore< weighted_lowcut:
        wLabel = "Low"
    elif rScore< weighted_highcut:
        wLabel = "Med"
    else:
        wLabel = "High"
    tlxLabels_df.loc[idx,"Raw Label"] = rLabel
    tlxLabels_df.loc[idx,"Weighted Label"] = wLabel

# Save dataframe as csv 
tlxLabels_df.to_csv("C:\\Users\\Wu Di\\Documents\\HRV-ML-Research\\TLX\\tlxLabels.csv")
