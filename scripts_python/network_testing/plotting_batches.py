import os
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt


this_path = os.path.dirname(os.path.abspath(__file__))

all_csvs = natsorted(glob(os.path.join(this_path,'*.csv')))
all_csvs = [ x for x in all_csvs if  'worm_segmenter' in x ]

print(all_csvs)

df_list = [pd.read_csv(this_csv,index_col = False, names = ['loss','val_loss','idx'],skiprows=1) for this_csv in all_csvs]

plt.figure(1)

for i,this_df in enumerate(df_list):

    losses = this_df.values

    plt.plot(np.arange(len(losses)),np.asarray(losses)[:,1], label = str(i)+'val_loss')
    plt.plot(np.argmin(np.asarray(losses)[:,1]),np.asarray(losses)[:,1][np.argmin(np.asarray(losses)[:,1])], 'go')#, markersize = 15)
plt.legend()
plt.savefig(r"C:\Users\LabPC2\Documents\GitHub\LightSaver\scripts_python\network_testing\output_losses.png")
plt.close('all')


