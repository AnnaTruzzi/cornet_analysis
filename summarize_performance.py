#%% 

import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib

df=pd.DataFrame(columns=['stage','conv','prec1','prec5','loss_log'])
df_aoa=pd.DataFrame(columns=['stage','conv','aoa','loss','node'])

for stage in [0,5,10,20,30]:
    for conv in range(0,4):
        lcpth='/home/rhodricusack/linearclass_v3/linearclass_time_%02d_conv_%d_v3'%(stage,conv)
        d={'stage':[stage],'conv':[conv]}
        for item in ['prec1','prec5','loss_log']:
            itpth=path.join(lcpth,'log',item)
            if path.exists(itpth):
                with open(itpth,'rb') as f:
                    it=pickle.load(f)
                    d[item]=float(it[0])
        df=df.append(pd.DataFrame.from_dict(d),ignore_index=True)
        aoapth=path.join(lcpth,'aoaresults.json')  
        with open(aoapth,'r') as f:
            aoa=json.load(f)
            for key,val in aoa.items():
                d={'stage':[stage],'conv':[conv],'node':key,'aoa':val['aoa'],'loss':val['loss']}
                df_aoa=df_aoa.append(pd.DataFrame.from_dict(d),ignore_index=True)



print(df)

plt.figure()
fig,ax=plt.subplots()
for key, grp in df.groupby(['conv']):
    ax = grp.plot(ax=ax,kind='line', x='stage', y='prec1',  label=key)
ax.set_xlabel('epoch')
ax.set_ylabel('prec1 (validation)')

plt.figure()
fig,ax=plt.subplots()
for key, grp in df.groupby(['conv']):
    grp.plot(ax=ax,kind='line', x='stage', y='loss_log',  label=key)
ax.set_ylim([0,7])
ax.set_xlabel('epoch')
ax.set_ylabel('loss (validation)')
plt.show()

#%%
from scipy.stats import pearsonr

for key, grp in df_aoa.groupby(['stage']):
    lossbylayer={}
    plt.figure()
    for convkey, convgrp in grp.groupby(['conv']):
        plt.scatter(x='aoa', y='loss', data=convgrp,  label=convkey)
        c=pearsonr(convgrp['aoa'],convgrp['loss'])
        print("epoch %d conv %d corr r=%3.2f p<%.3f"%(key,convkey,c[0],c[1]))
        lossbylayer[convkey]=convgrp['loss'].to_numpy()
        lossbylayer_aoa=convgrp['aoa'].to_numpy()
    plt.legend()
    loss3m0=lossbylayer[3]-lossbylayer[0]
    c=pearsonr(convgrp['aoa'],loss3m0)
    print("epoch %d conv 3 minus 0 corr r=%3.2f p<%.3f"%(key,c[0],c[1]))
ax.set_xlabel('epoch')
ax.set_ylim([0,10])
ax.set_ylabel('loss (validation)')
plt.show()


            

#%%
