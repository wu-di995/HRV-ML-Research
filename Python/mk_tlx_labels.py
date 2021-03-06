# Create plots for tlx scores and creates dataframe for tlx labels

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import paths



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



if __name__ == "__main__":
    # Change path to match local
    tlx_dir = paths.TLX_label_path
    tlx_df = pd.read_csv(tlx_dir+"tlx.csv")

    # Exclude Pilot subjects
    pilot_mask = tlx_df["id"].str.contains(r'PILOT',na=False)
    non_pilots = ~pilot_mask

    raw_scores = tlx_df.loc[non_pilots,"Raw Score"]
    weighted_scores = tlx_df.loc[non_pilots,"Weighted Score"]

    # print(tlx_df.shape)
    # print(pilot_mask.sum())
    # print(len(raw_scores))
    # print(len(weighted_scores))

    # Plotting
    # plot_histograms(raw_scores,"Raw")
    # plot_histograms(weighted_scores,"Weighted")

    # Convert pd series to np arrays
    raw_scores_ar = raw_scores.values
    weighted_scores_ar = weighted_scores.values

    # Split into two groups, high and low, which are above and below 50th percentile respectively
    raw_cut = np.percentile(raw_scores_ar,50.0)
    print("Raw score 50th percentile:",round(raw_cut,2))

    weighted_cut = np.percentile(weighted_scores_ar,50.0)
    print("Weighted score 50th percentile:",round(weighted_cut,2))

    # Split into low, med, high, which are based on 1/3 and 2/3 percentile cutoffs
    raw_lm_cut = np.percentile(raw_scores_ar, 33.3)
    raw_mh_cut = np.percentile(raw_scores_ar, 66.7)
    weighted_lm_cut = np.percentile(weighted_scores_ar, 33.3)
    weighted_mh_cut = np.percentile(weighted_scores_ar, 66.7)

    # Produce labels for each session and save as csv
    # Binary dataframe 
    columns = ["id","interface","autonomy","Raw Label", "Weighted Label"]
    index = range(non_pilots.sum())
    tlxLabels_2_df = pd.DataFrame(index=index,columns=columns)
    tlxLabels_2_df["id"] = tlx_df.loc[non_pilots,"id"].map(lambda x: x.lower()).values
    tlxLabels_2_df["interface"] = tlx_df.loc[non_pilots,"interface"].values
    tlxLabels_2_df["autonomy"] = tlx_df.loc[non_pilots,"assistance"].values

    for idx in index:
        rScore = raw_scores_ar[idx]
        wScore = weighted_scores_ar[idx]
        # Create raw score label
        if rScore< raw_cut:
            rLabel = "Low"
        else:
            rLabel = "High"
        # Create weighted score label
        if wScore< weighted_cut:
            wLabel = "Low"
        else:
            wLabel = "High"
        tlxLabels_2_df.loc[idx,"Raw Label"] = rLabel
        tlxLabels_2_df.loc[idx,"Weighted Label"] = wLabel


    # 3 Levels dataframe 
    tlxLabels_3_df = pd.DataFrame(index=index,columns=columns)
    tlxLabels_3_df["id"] = tlx_df.loc[non_pilots,"id"].map(lambda x: x.lower()).values
    tlxLabels_3_df["interface"] = tlx_df.loc[non_pilots,"interface"].values
    tlxLabels_3_df["autonomy"] = tlx_df.loc[non_pilots,"assistance"].values

    for idx in index:
        rScore = raw_scores_ar[idx]
        wScore = weighted_scores_ar[idx]
        # Create raw score label
        if rScore< raw_lm_cut:
            rLabel = "Low"
        elif rScore< raw_mh_cut:
            rLabel = "Med"
        else:
            rLabel = "High"
        # Create weighted score label
        if wScore< weighted_lm_cut:
            wLabel = "Low"
        elif wScore<weighted_mh_cut:
            wLabel = "Med"
        else:
            wLabel = "High"
        tlxLabels_3_df.loc[idx,"Raw Label"] = rLabel
        tlxLabels_3_df.loc[idx,"Weighted Label"] = wLabel

    # Save dataframe as csv 
    tlxLabels_2_df.to_csv(tlx_dir+"tlxLabels_2.csv")
    tlxLabels_3_df.to_csv(tlx_dir+"tlxLabels_3.csv")
