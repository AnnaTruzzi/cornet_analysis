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
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def learningcurve(x,A,b):
    y=A*(np.exp(-x*b))
    return y

# Colormap for AoA
cmap = plt.cm.get_cmap('RdBu')
colors = cmap(np.arange(cmap.N))
print(cmap.N)


# All 1000 items
df=pd.read_json('/home/rhodricusack/cornet/linearclass_v3.json')
print(df)

fig,ax=plt.subplots(ncols=3,sharex=True)

for axcol,measure in enumerate(['prec1','prec5','loss_log']): 
    for key, convgrp in df.groupby(['conv']):
        convgrp.plot(ax=ax[axcol],kind='line', x='stage', y=measure,  label=key)
    ax[axcol].set_xlabel('epoch')
    ax[axcol].set_ylabel('%s (validation)'%measure)
ax[2].set_ylim([0,10])


# Birbeck sets
df=pd.read_json('/home/rhodricusack/cornet/linearclass_v3_val_birbeck.json')
print(df)

fig,ax=plt.subplots(nrows=2,ncols=3,sharex=True)
for axrow,grp in enumerate(df.groupby(['group'])):
    for axcol,measure in enumerate(['prec1','prec5','loss']): 
        for key, convgrp in grp[1].groupby(['conv']):
            convgrp.plot(ax=ax[axrow][axcol],kind='line', x='stage', y=measure,  label=key)
        ax[axrow][axcol].set_xlabel('epoch')
        ax[axrow][axcol].set_ylabel('%s (validation)'%measure)
    ax[axrow][2].set_ylim([0,10])
plt.show()
