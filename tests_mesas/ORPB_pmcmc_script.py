# This is the main script to run pMCMC for ORPB data using mesas SAS model
# Created on: 11/10/2025
#%%
import os
import sys

script_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_folder) #sys.path is the list of directories python searches for modules/packages

REPO_ROOT = r"C:\Users\simon\Desktop\mesas" #ensure mesas and mesas.sas are importable
SAS_FOLDER = os.path.join(REPO_ROOT, "mesas", "sas") 
STOCHASTIC_FOLDER = os.path.join(REPO_ROOT, "mesas.stochastic") #ensure mesas.stochastic and submodules are importable
os.environ['PATH'] = SAS_FOLDER + os.pathsep + os.environ.get('PATH', '') #ensures windows can find the dlls in sas for compiled .pyd extentions
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, STOCHASTIC_FOLDER)

# Clear previously imported modules to ensure fresh import of edited files
modules_to_clear = [
    "ORPB_mesas_interface",
    "mesas.sas.model",
    "mesas.sas.utils_chain",
    "mesas.sas.ssm_model",
    "mesas.sas.specs",
    "utils"
]
for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]


# from functions.get_dataset import get_different_input_scenarios
import pandas as pd

from ORPB_mesas_interface import ModelInterfaceMesas
import matplotlib.pyplot as plt
import numpy as np
from model.utils_chain import Chain
from functions.utils import plot_MAP

from model.ssm_model import SSModel
from mesas.sas.model import Model as SAS_Model
import argparse
from mesas.sas.specs import Component

from ORPB_cases import *
import seaborn as sns

#%%
# SET FOLDERS
# ================================================================
data_root = "/Users/simon/Desktop/SAS"
data_resolution_root = "/Users/simon/Desktop/ORPB_resolution_datasets"
result_root = "/Users/simon/Desktop/ORPB_results"

if not os.path.exists(result_root):
    os.makedirs(result_root)
#%%
# READ DATA AND PREPROCESS
# ================================================================

# ORPB isotope data
# data_df = pd.read_csv(f'{data_root}/ORPB_isotope_data.csv', index_col=0, parse_dates=[0])

#or use resolution data
res = 'W' #'D'
resolution = 'weekly' #'daily'
data_df = pd.read_csv(f"/Users/simon/Desktop/ORPB_resolution_datasets/ORPB_isotope_data_isoMAP_precip 18O_{resolution}.csv", index_col=0, parse_dates=[0])
data_df['precip 18O'] = data_df['mean_c']
data_df['discharge (mm/hr)'] = data_df[f'discharge (mm/{res})'] # for convenience now
data_df['baseflow 1 (mm/hr)'] = data_df[f'baseflow 1 (mm/{res})']
data_df['snowmelt (mm/hr)'] = data_df[f'snowmelt (mm/{res})']
data_df['rainfall (mm/hr)'] = data_df[f'rainfall (mm/{res})']
data_df['ET (mm/hr)'] = data_df[f'ET (mm/{res})']


data_df = data_df.loc[pd.Timestamp('2014-01-01'): pd.Timestamp('2014-12-31')] #2014-08-01 - 2016-08-31subset to Putnam's data range


issample = np.logical_not(np.isnan(data_df['ORPB 18O']))
data_df['influx (mm/hr)'] = data_df[['rainfall (mm/hr)','snowmelt (mm/hr)']].sum(axis=1)

# separate Q into quickflow and baseflow 1
data_df['quickflow (mm/hr)'] = data_df['discharge (mm/hr)'] - data_df['baseflow 1 (mm/hr)']
data_df['bf1_weight'] = data_df['baseflow 1 (mm/hr)'] / data_df['discharge (mm/hr)']
data_df['qf_weight'] = data_df['quickflow (mm/hr)'] / data_df['discharge (mm/hr)']
# isbaseflow = data_df.loc[(data_df['quickflow (mm/hr)']<0.001) & issample].index
# isquickflow = data_df.loc[(data_df['quickflow (mm/hr)']>=0.001) & issample].index

# data_df['weekly_obs'] = ~data_df.loc[data_df['rainfall (mm/hr)']>0, 'Sample Name'].duplicated(keep='last') # make sure weekly values are only for when rain was actually observed (1066 obs values vs only 235 when observed without rain restriction)
# data_df['weekly_obs'] = data_df['weekly_obs'].fillna(False) #set non-masked values to False i.e. where no rain observed
# data_df.loc[data_df['weekly_obs']==False, 'precip 18O'] = np.nan # set non-weekly observed isotope values to nan so they don't influence GP fit

data_df['is_obs_input'] = data_df['precip 18O'].notna()
data_df['is_obs_input_filled'] = False
data_df.loc[data_df['precip 18O'].isna()==True, 'is_obs_input_filled']=True
# fill nans in precip 18O with mean
mean = data_df['precip 18O'].mean()
df= data_df.copy() #make a copy of the data_df
# df.loc[df['precip 18O'].isna()==True, 'precip 18O']=mean
df['precip 18O'] = df['precip 18O'].ffill().bfill()


df['is_obs_output'] = df['ORPB 18O'].notna()

case_name = 'storage_q_ug_et_u'#********************

num_input_scenarios = 3# 15 #N particles
num_parameter_samples = 3 #15 #D #may need to change this as I change isotope timeseries configurations (fewer obs mean fewer timesteps where particles get reweighted which affect the variance of the likelihood estimate)
len_parameter_MCMC = 2#5 #L

#%%
# RUN MODEL 
# ================================================================
# initialize model settings
output_obs = df['ORPB 18O'].notna().to_list()
config = {
    'dt': 1, # in hours
    'observed_made_each_step': output_obs,
    'influx': ['influx (mm/hr)'],
    'outflux': ['discharge (mm/hr)', 'ET (mm/hr)'], #['quickflow (mm/hr)', 'baseflow 1 (mm/hr)', 'ET (mm/hr)'],
    'use_MAP_AS_weight': True,
    'use_MAP_ref_traj': True,
    'use_MAP_MCMC': True,
    'update_theta_dist': True, #False,
}

model_interface_class = ModelInterfaceMesas

if case_name == 'storage_q_gg_et_u':
    model_interface = model_interface_class(
        df=df,
        customized_model=SAS_Model,
        num_input_scenarios=num_input_scenarios,
        config=config,
        theta_init=theta_storage_q_gg_et_u
    )
elif case_name == 'storage_q_ug_et_u':
    model_interface = model_interface_class(
        df=df,
        customized_model=SAS_Model,
        num_input_scenarios=num_input_scenarios,
        config=config,
        theta_init=theta_storage_q_ug_et_u
    )
elif case_name == 'storage_q_u_et_u':
    model_interface = model_interface_class(
        df=df,
        customized_model=SAS_Model,
        num_input_scenarios=num_input_scenarios,
        config=config,
        theta_init=theta_storage_q_u_et_u
    )
elif case_name == 'storage_q_g_et_u':
    model_interface = model_interface_class(
        df=df,
        customized_model=SAS_Model,
        num_input_scenarios=num_input_scenarios,
        config=config,
        theta_init=theta_storage_q_g_et_u
    )
else:
    raise ValueError("Case name not found.")

# #%%
# # CHECK INPUT SCENARIOS GENERATION
# #  ================================================================

# # check input scenarios generation
# model_interface._bulk_input_preprocess()

# plt.figure(figsize=(12, 4))
# r = model_interface.R_prime
# for i in range(num_input_scenarios):
#     plt.scatter(
#         df.index, r[i], marker=".", s=10, c="gray", alpha=0.5, label=f"Simulated inputs"
#     )
# obs = model_interface.df[model_interface.in_sol].to_numpy()
# plt.plot(df.index, obs, ".", markersize=5, label="Observed inputs")
# # plt.yscale("log")
# # plt.xticks(time[1::90], rotation=30) 
# # ax = plt.gca()
# # ax.set_xticks(ax.get_xticks()[1::90])
# plt.xlim(df.index[0], df.index[-1])
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(handles[-2:], labels[-2:], loc="upper right", fontsize=12, ncol=2)
# plt.ylabel("Concentration")
# plt.tight_layout()
# plt.show() # for command line pop-up window
# plt.savefig(f"{result_root}/input_scenarios_{case_name}.pdf")

# # %%
# # RUN pMCMC CHAIN 
# # ================================================================

# chain = Chain(model_interface=model_interface)
# chain.run_particle_filter_SIR()
# # %% Plot check

# plt.figure()
# plt.plot(model_interface.df["precip 18O"], "*")
# plt.plot(model_interface.df.index, chain.state.R[:, :, 0].T, ".", markersize=0.7) #chain.state.R[:,:,0] where 0 is quickflow (mm/hr) from outflux
# plt.xticks(rotation=30)
# plt.ylabel("Concentration")
# plt.tight_layout()
# plt.title("Input concentration scenarios from SIR")
# plt.show()

# # %%
# # for i in range(25):
# #     plt.figure()
# #     plt.plot(model_interface.df['C out'], "*")
# #     plt.plot(model_interface.df.index, chain.state.Y[i,:,0].T)#, ".", markersize=0.7)
# plt.figure()
# plt.plot(model_interface.df["ORPB 18O"], "*")
# plt.plot(model_interface.df.index, chain.state.Y[:, :, 0].T, ".", markersize=0.7)
# plt.plot(
#     model_interface.df.index,
#     chain.state.Y[np.argmax(chain.state.W), :, 0].T,
#     ".",
#     markersize=10,
# )
# plt.xticks(rotation=30)
# plt.ylabel("Concentration")
# plt.tight_layout()
# plt.title("Output concentration scenarios from SIR")
# plt.show()
# print("done SIR check")


# # %%

# chain.run_particle_filter_AS()

# plt.figure()
# sns.boxplot(chain.state.R[:, :, 0])
# plt.plot(model_interface.df["precip 18O"].values, "*")
# plt.xticks(rotation=30)
# plt.ylabel("Concentration")
# plt.tight_layout()
# plt.title("Input concentration scenarios from AS")
# r = chain.state.R[:, :, 0].T
# q = chain.state.Y[:, :, 0].T
# plt.figure()
# plt.plot(model_interface.df["ORPB 18O"], "*")
# plt.plot(model_interface.df.index, chain.state.Y[:, :, 0].T, ".", markersize=0.7)
# plt.xticks(rotation=30)
# plt.ylabel("Concentration")
# plt.tight_layout()
# plt.title("Output concentration scenarios from AS")
# plt.show()
# print("done AS check")

# # %%

# run actual particle Gibbs
model = SSModel(
    model_interface=model_interface,
    num_parameter_samples=num_parameter_samples,
    len_parameter_MCMC=len_parameter_MCMC,
)

model.run_particle_Gibbs()


# %%

# SAVE RESULTS 
# ================================================================
# get estimated parameters
#
theta = model.theta_record
theta_name = model_interface._theta_to_estimate
theta_df = pd.DataFrame(theta, columns=theta_name)
theta_std = model.theta_std
theta_std_df = pd.DataFrame(theta_std, columns=theta_name)
state_record = model.state_record #used in ORPB_SAS_check.py

input_scenarios = model.input_record
output_scenarios = model.output_record
df = model_interface.df

# MLE SAS model output from the last MCMC iteration's best run
pQ_mle = model.pQ_mle['discharge (mm/hr)']  # shape: (max_age, T)
sT_mle = model.sT_mle                        # shape: (max_age, T)
ST_mle = model.ST_mle                        # shape: (max_age, T)
# CDF over age = cumulative TTD in discharge at each time t
PQ_mle = np.cumsum(pQ_mle, axis=0) * config['dt']

#%%

# ANALYSIS AND PLOTTING
# ===============================================================
# Plot distributions of theta parameters
from scipy.stats import gaussian_kde
def kde_mode(samples):
    """Estimate mode of continuous samples using KDE."""
    samples = samples[~np.isnan(samples)]
    kde = gaussian_kde(samples)
    x = np.linspace(samples.min(), samples.max(), 1000)
    return x[np.argmax(kde(x))]

means = theta_df.iloc[1]
stds = theta_std_df.iloc[1]
ncols = 3
nrows = int(np.ceil((len(means)-3)/ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()
for i in range(len(means)-3):
    mean = means[i]
    std = stds[i]
    qfa_hist = np.random.normal(loc=mean, scale=std, size=10000)
    # mode = kde_mode(qfa_hist)
    # Plot histogram
    ax = axes[i]
    ax.hist(qfa_hist, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

    ax.set_title(f'{theta_df.columns[i]}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean}')
    # ax.axvline(mode, color='green', linestyle='--', label=f'Mode = {mode}')
    ax.legend()
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.tight_layout()
plt.show()


#%% plot each MCMC iteration output against observed

st, et = model_interface.observed_ind[0], model_interface.observed_ind[-1]
time = model_interface.df.index

# plt.figure(figsize=(12,4))
# plt.plot(model_interface.df["discharge (mm/hr)"], "grey")

# plt.plot(time[st:et], output_scenarios[-1:, st:et].T,'.', alpha = 0.1, color="grey",lw=0.5, label="predicted")
# # make a step plot
# plt.step(time[st:et], model_interface.df["ORPB 18O"].backfill().iloc[st:et], label= "observed")
# plt.plot(time[st:et],model_interface.df["ORPB 18O"].iloc[st:et], "*", label= "observed")

for i in range(len_parameter_MCMC + 1):
    plt.figure(figsize=(12, 4))

    plt.step(
        time[st:et],
        model_interface.df["ORPB 18O"].backfill().iloc[st:et],
        label="observed",
    )
    plt.plot(
        time[st:et], output_scenarios[i, st:et].T, color="orange", alpha=0.4, label="predicted"
    )
    plt.title(f'pMCMC iteration {i}')
    plt.xlabel('Date')
    plt.ylabel('delta 18O in stream (per mil)')
    # plt.xlim([time[0], time[-1]])
#     plt.plot(model_interface.df.index[st:et], output_scenarios[i, st:et].T,   label=f"output {i}", lw=0.5)
plt.legend(frameon=False)
# np.save(f"output.npy", output_scenarios)
plt.show()


#%%
# Plot TTD
ST = (ST_mle[:,:-1] + ST_mle[:,1:])/2
T = 1*np.arange(pQ_mle.shape[0]) # model.options['dt'] *np.arange(model.options['max_age']) #max age is 1st dim of pq and ST
import matplotlib.cm as cm
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0,1,len(df))]
fig, ax = plt.subplots()
for i in range(0, len(df)):
    # plt.plot(ST[:,i],pQ_mle[:,i], color=colors[i]) #density
    # plt.plot(ST[:,i],sas_age_cdf_mle[:,i], color=colors[i]) #cumulative
    plt.plot(T, pQ_mle[:,i], color=colors[i]) #TTD
plt.xlabel('Age (days)')
plt.ylabel('$p_Q$')
plt.title('Discharge SAS function over time')
# plt.xlim([0, 25])
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0,
vmax=len(df)-1))
plt.colorbar(sm, ax=ax, label='Time Index')

# %% Cumulative TTD
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0,1,len(df))]
fig, ax = plt.subplots()
for i in range(0, len(df)):
    plt.plot(T, PQ_mle[:,i], color=colors[i]) #TTD
plt.xlabel('Age (days)')
plt.ylabel('P_Q$')
plt.title('Cumulative Discharge SAS function over time')
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0,
vmax=len(df)-1))
plt.colorbar(sm, ax=ax, label='Time Index')



# %%
# # save data as csv files
theta_df.to_csv(f"{result_root}/theta_{case_name}_3Y.csv")
theta_std_df.to_csv(f"{result_root}/theta_std_{case_name}_3Y.csv")
np.savetxt(f"{result_root}/input_scenarios_{case_name}_3Y.csv", input_scenarios, delimiter=",")
np.savetxt(f"{result_root}/output_scenarios_{case_name}_3Y.csv", output_scenarios, delimiter=",")

np.save(f"{result_root}/pQ_mle_{case_name}.npy", pQ_mle)
np.save(f"{result_root}/sT_mle_{case_name}.npy", sT_mle)
np.save(f"{result_root}/ST_mle_{case_name}.npy", ST_mle)

# %% RELOAD AND PLOT SAVED RESULTS
# ================================================================

input_scenarios = pd.read_table(f"{result_root}/input_scenarios_{case_name}_W1Y_job25102382.csv", delimiter=",", header=None)
output_scenarios = pd.read_table(f"{result_root}/output_scenarios_{case_name}_W1Y_job25102382.csv", delimiter=",", header=None)
time = df.index
st, et = df.index.get_loc(df['ORPB 18O'].first_valid_index()), df.index.get_loc(df['ORPB 18O'].last_valid_index())

plt.figure()
for i in range(len_parameter_MCMC + 1):
    plt.figure(figsize=(12, 4))

    plt.step(
        time[st:et],
        df["ORPB 18O"].backfill().iloc[st:et],
        label="observed",
    )
    plt.plot(
        time[st:et], output_scenarios.iloc[i, st:et].T, color="orange", label="predicted"
    )
    plt.title(f'pMCMC iteration {i}')
    # plt.xlim([time[0], time[-1]])
#     plt.plot(model_interface.df.index[st:et], output_scenarios[i, st:et].T,   label=f"output {i}", lw=0.5)
plt.legend(frameon=False)
plt.show()
# %%

# Plot residuals between observed and predicted at each MCMC iteration
#======================================================================
plt.figure()
for i in range(len_parameter_MCMC + 1):
    plt.figure(figsize=(12, 4))

    plt.step(
        time[st:et],
        (df["ORPB 18O"].backfill().iloc[st:et]).to_numpy() - (output_scenarios[i, st:et].T),
        label="residuals",
    )
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'pMCMC iteration {i} residuals')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    # print(f'Mean residuals at iteration {i}: {np.mean((df["ORPB 18O"].backfill().iloc[st:et]).to_numpy() - (output_scenarios.iloc[i, st:et].T).to_numpy())}')
plt.legend(frameon=False)
# plt.show()
# %%
