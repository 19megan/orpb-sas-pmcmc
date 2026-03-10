# This script will use the optimized params from pMCMC to manually build 
# and run a SAS model for the ORPB tracer data
# 
# Date: 01/27/2026

#%% 
# ------------------Import dataset-------------------
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from mesas.sas.model import Model
from permetrics.regression import RegressionMetric
from ORPB_cases import *
from call_SAS import run_SAS

data_df = pd.read_csv(f'/Users/simon/Desktop/SAS/ORPB_isotope_data.csv', index_col=0, parse_dates=[0])
data_df = data_df.loc[pd.Timestamp('2014-08-01'): pd.Timestamp('2014-08-31')] #subset to Putnam's data range 2014-08-01 - 2016-08-31
issample = np.logical_not(np.isnan(data_df['ORPB 18O']))
data_df['influx (mm/hr)'] = data_df[['rainfall (mm/hr)','snowmelt (mm/hr)']].sum(axis=1)
data_df.loc[data_df['influx (mm/hr)']==0, 'precip 18O'] = 0.0

# separate Q into quickflow and baseflow 1
data_df['quickflow (mm/hr)'] = data_df['discharge (mm/hr)'] - data_df['baseflow 1 (mm/hr)']
data_df['bf1_weight'] = data_df['baseflow 1 (mm/hr)'] / data_df['discharge (mm/hr)']
data_df['qf_weight'] = data_df['quickflow (mm/hr)'] / data_df['discharge (mm/hr)']

# fill nans with mean
mean = data_df['precip 18O'].mean()
df= data_df.copy() #make a copy of the data_df
df.loc[df['precip 18O'].isna()==True, 'precip 18O']=mean

case_name = 'storage_q_gg_et_u'

# make sure pMCMC and this script use the same observed_ind
obs_made = df['ORPB 18O'].notna().to_list()
observed_ind = np.arange(len(df))[obs_made]

#%% 
# -------------------Set parameters ----------------------------
checkparams = {}
checkparams['C_old'] = -7.70623965251686 # (per mil)
checkparams['scale'] =46.33924896891153 # ET scale
checkparams['qf_a'] = 0.5406241295147112 # qf shape parameter
checkparams['qf_scale'] = 0.5021168023432488 # qf scale parameter
checkparams['a'] =  1.2251846594723816 #1.26399
checkparams['S_c'] = -129.2516881115114 #min(df['storage (mm)'])-50 # this is solid, should be less than dynamic storage
checkparams['lambda'] =0.28866859021656743 #0.3563 #1.1017 # 10/(df['storage (mm)'].median()-S_c) #median of storage is s_ref
scale = 1547.119
df['S_scale'] = checkparams['lambda']*(df['storage (mm)']-checkparams['S_c'])
#%%
#-------------------Decide what lamda should be -----------------------
# scale = df['storage (mm)']
# sscale=lamda*(scale-S_c)
# plt.plot(scale,sscale) #increase monotonically
# plt.plot(data_df['storage (mm)'], sscale) # check that they increase
# # hard to tell if sscale is good for discharge, try binning by sscale
# data_df['sscale']=sscale
# df = data_df[['sscale', 'discharge (mm/hr)']].dropna()
# df['Q'] = df['discharge (mm/hr)']
# df['S_bin'] = pd.qcut(df['sscale'], q=10, duplicates='drop')
# binned = (
#     df
#     .groupby('S_bin')
#     .agg(
#         S_scale_mid=('sscale', 'median'),
#         Q_mean=('Q', 'mean'),
#         Q_std=('Q', 'std'),
#         Q_p90=('Q', lambda x: np.percentile(x, 90)),
#         Q_p95=('Q', lambda x: np.percentile(x, 95)),
#         n=('Q', 'size')
#     )
#     .reset_index()
# )
# # variance test: should increase monotonically
# plt.figure()
# plt.plot(binned['S_scale_mid'], binned['Q_std'], marker='o')
# plt.xlabel('S_scale')
# plt.ylabel('std(Q)')
# plt.title('Discharge variability vs storage scale')
# plt.show()
# # upper-tail test: confirms scale behavior
# plt.figure()
# plt.plot(binned['S_scale_mid'], binned['Q_p90'], marker='o')
# plt.xlabel('S_scale')
# plt.ylabel('Q 90th percentile')
# plt.title('Discharge upper-tail vs storage scale')
# plt.show()



#%%
# ------------------Build the model --------------------
# Now build a model with parameters
model = run_SAS(df, checkparams)
# from mesas.sas.model import Model
# model =Model(df, config={'sas_specs': sas_specs_storage_q_g_et_u, 'solute_parameters': solute_parameters, 'options': options})
# model.run()
#%%
# ------------------Visualize the output-------------------
# data_df = model.data_df #don't forget to do this
from mesas.utils import vis
fig = plt.figure(figsize=[12,4])
st,et = observed_ind[0], observed_ind[-1]
plt.plot(model.data_df.index[st:et], model.data_df['ORPB 18O'].backfill().iloc[st:et], color='grey', label='Observed 18O outflow')

plt.plot(model.data_df.index[st:et], model.data_df['precip 18O --> discharge (mm/hr)'].iloc[st:et], color='orange', label='direct SAS Predicted 18O outflow')
# plt.axvspan(pd.Timestamp('2014-10-01'), pd.Timestamp('2015-09-30'), color='lightgrey', alpha=0.5)
# plt.axvspan(pd.Timestamp('2016-10-01'), pd.Timestamp('2017-09-30'), color='lightgrey', alpha=0.5)
# plt.axvspan(pd.Timestamp('2018-10-01'), pd.Timestamp('2019-09-30'), color='lightgrey', alpha=0.5)
plt.legend()
plt.title('Isotope outflow at ORPB')

# %%
