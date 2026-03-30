import os
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt

def find_distinct_suffixes(strings):
    if not strings:
        return []

    # 1. Find the longest common initial substring
    common_prefix = strings[0]
    for s in strings[1:]:
        nchars = min(len(common_prefix), len(s))
        i = 0
        while i < nchars and common_prefix[i] == s[i]:
            i += 1
        common_prefix = common_prefix[:i]
        if not common_prefix: # Optimization: no common prefix found, break early
            break

    # 2. Find the distinct suffixes for each string
    suffixes = []
    for s in strings:
        suffixes.append(s[len(common_prefix):])
        
    return common_prefix, suffixes

this_path = os.path.dirname(os.path.abspath(__file__))

all_csvs = natsorted(glob(os.path.join(this_path,'*.csv')))
all_csvs = [ x for x in all_csvs if  'worm_segmenter' in x ]

for this_csv in all_csvs:
    print(this_csv)
common_prefix, csv_labels = find_distinct_suffixes(all_csvs)

df_list = [pd.read_csv(this_csv,index_col = False, names = ['loss','val_loss','idx'],skiprows=1) for this_csv in all_csvs]

colormap_loss = plt.colormaps['autumn'](np.linspace(0,1,len(df_list)))
colormap_vloss = plt.colormaps['winter'](np.linspace(0,1,len(df_list)))

plt.figure(1,figsize=(15,15),dpi=300)

for i,this_df in enumerate(df_list):

    this_label = csv_labels[i]

    losses = this_df.values

    this_loss = this_df['loss'].values
    this_vloss = this_df['val_loss'].values

    min_vloss = round(min(this_vloss),4)

    x = np.arange(len(this_vloss))

    plt.plot(x,this_vloss, label = this_label+' val_loss'+ str(min_vloss), color = colormap_vloss[i])
    plt.plot(x,this_loss,  label = this_label+' loss: ' ,                  color = colormap_loss[i])

    plt.plot(np.argmin(this_vloss),this_vloss[np.argmin(this_vloss)], 'go')#, markersize = 15)

plt.legend()
plt.savefig(os.path.join(this_path,"_combined_output_losses.png"))
plt.close('all')


