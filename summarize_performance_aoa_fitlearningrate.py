#%% 

import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import math
import numpy as np

from scipy.optimize import curve_fit


def learningcurve(x,A,b):
    y=A*(np.exp(-x*b))
    return y

# Colormap for AoA
cmap = plt.cm.get_cmap('inferno')
colors = cmap(np.arange(cmap.N))
print(cmap.N)


df_aoa=pd.read_json('linearclass_v3_aoa.json')

print('Dropping epoch 0')
df_aoa=df_aoa[df_aoa.stage != 0]
df_aoa['aoa_rank']=df_aoa['aoa'].rank()

aoamin=df_aoa['aoa_rank'].min()
aoarange=df_aoa['aoa_rank'].max()-aoamin

from scipy.stats import pearsonr

fig,ax=plt.subplots(ncols=4)


lc={}
for convkey,convgrp in df_aoa.groupby('conv'):
    ax[convkey].set_title('Conv layer %d'%convkey)
    lc[convkey]={}
    for nodekey,nodegrp in convgrp.groupby('node'):
        A0=float(nodegrp.loc[nodegrp['stage']==35]['loss']) # starting estimate for A
        stage=np.array([float(s) for s in nodegrp['stage']])
        loss=np.array([float(l) for l in nodegrp['loss']])
        lc[convkey][nodekey]=curve_fit(learningcurve,stage,loss,p0=[A0,0])
        colind=int(((nodegrp['aoa_rank'].iloc[0]-aoamin)/aoarange)*255)
        ax[convkey].plot(stage,loss,color=colors[colind],alpha=0.2)
plt.show()
print(lc)






            

#%%
