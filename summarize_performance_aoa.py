#%% 

import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns

df_aoa=pd.read_json('linearclass_v3_aoa.json')


df_aoa['aoa_binned']=pd.qcut(df_aoa['aoa'],5)
print(df_aoa)

#%%
# from scipy.stats import pearsonr

# fig,allax=plt.subplots(5)

# for ind, g in enumerate(df_aoa.groupby('aoa_binned')):

#     ax=allax[ind]
#     ax.set_title("AOA (%f-%f]"%(g[0].left,g[0].right))
#     lossbylayer={}

#     for convkey, convgrp in g[1].groupby('conv'):
#         stagegrp=convgrp.groupby('stage').mean().reset_index()
#         stagegrp.plot(ax=ax,x='stage', y='loss',label=convkey)
#         ax.set_xlabel('epoch')
#         ax.set_ylim([0,50])
#         ax.set_ylabel('loss (validation)')
#     plt.legend()


from scipy.stats import pearsonr

fig,ax=plt.subplots()

epoch=2
convgrp=df_aoa.groupby('stage').get_group(epoch)
ind=0


ax.set_title("Epoch %d"%epoch)
lossbylayer={}
means = convgrp.groupby('conv').transform('mean')
convgrp['loss']=(convgrp['loss']-means['loss'])/means['loss']

#    convgrp['loss']=(convgrp['loss']-convgrp['loss'].mean())/convgrp['loss'].mean()
sns.violinplot(ax=ax,x='conv',y='loss',hue='aoa_binned',data=convgrp)
ax.set_xlabel('layer')
ax.set_ylabel('loss (validation)')
plt.legend()
plt.show()


            

#%%
