# This is the main script to run pMCMC for ORPB data using mesas SAS model
# Created on: 11/10/2025
#%%
import os
import sys

current_path = os.getcwd()

# Make sure local directory is searched BEFORE install packages
sys.path.insert(0, '.') #ensure the current folder comes first

if current_path[-11:] != "tests_mesas":
    os.chdir("tests_mesas")
    print("Current working directory changed to 'tests_mesas'.")


# Force a fresh import
# Remove ANY cached copies
mods = [m for m in sys.modules if "ORPB_mesas_interface" in m or "tests_mesas" in m]
for m in mods:
    del sys.modules[m]
    print("Deleted cached module:", m)

# sys.path.append("../") # this makes tests_mesas in the parent directory of the import path which makes ModelInterfaceMesas not import as my edited file/

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
data_root = "/Users/simon/OneDrive/Documents/JHU/SAS"
result_root = "/Users/simon/OneDrive/Documents/JHU/SAS/ORPB_results"

if not os.path.exists(result_root):
    os.makedirs(result_root)
#%%
# READ DATA AND PREPROCESS
# ================================================================

# ORPB isotope data
data_df = pd.read_csv(f'{data_root}/ORPB_isotope_data.csv', index_col=0, parse_dates=[0])
data_df = data_df.loc[pd.Timestamp('2014-08-01'): pd.Timestamp('2014-08-31')] #2014-08-01 - 2016-08-31subset to Putnam's data range
issample = np.logical_not(np.isnan(data_df['ORPB 18O']))
data_df['influx (mm/hr)'] = data_df[['rainfall (mm/hr)','snowmelt (mm/hr)']].sum(axis=1)

# separate Q into quickflow and baseflow 1
data_df['quickflow (mm/hr)'] = data_df['discharge (mm/hr)'] - data_df['baseflow 1 (mm/hr)']
data_df['bf1_weight'] = data_df['baseflow 1 (mm/hr)'] / data_df['discharge (mm/hr)']
isbaseflow = data_df.loc[(data_df['quickflow (mm/hr)']<0.001) & issample].index
isquickflow = data_df.loc[(data_df['quickflow (mm/hr)']>=0.001) & issample].index

data_df['is_obs_input'] = data_df['precip 18O'].notna()
data_df['is_obs_input_filled'] = False
data_df.loc[data_df['precip 18O'].isna()==True, 'is_obs_input_filled']=True
# fill nans in precip 18O with mean
mean = data_df['precip 18O'].mean()
df= data_df.copy() #make a copy of the data_df
df.loc[df['precip 18O'].isna()==True, 'precip 18O']=mean

df['is_obs_output'] = df['ORPB 18O'].notna()

case_name = 'storage_q_ug_et_u'

num_input_scenarios = 15 #N
num_parameter_samples = 15 #D
len_parameter_MCMC = 5 #L

#%%
# RUN MODEL 
# ================================================================
# initialize model settings
output_obs = df['ORPB 18O'].notna().to_list()
config = {
    'dt': 24, # in hours
    'observed_made_each_step': output_obs,
    'influx': ['influx (mm/hr)'],
    'outflux': ['quickflow (mm/hr)', 'baseflow 1 (mm/hr)', 'ET (mm/hr)'],
    'use_MAP_AS_weight': True,
    'use_MAP_ref_traj': True,
    'use_MAP_MCMC': True,
    'update_theta_dist': False,
}

model_interface_class = ModelInterfaceMesas

if case_name == 'storage_q_ug_et_u':
    model_interface = model_interface_class(
        df=df,
        customized_model=SAS_Model,
        num_input_scenarios=num_input_scenarios,
        config=config,
        theta_init=theta_storage_q_ug_et_u
    )
else:
    raise ValueError("Case name not found.")

#%%
# CHECK INPUT SCENARIOS GENERATION
#  ================================================================

# check input scenarios generation
model_interface._bulk_input_preprocess()

plt.figure(figsize=(12, 4))
r = model_interface.R_prime
for i in range(num_input_scenarios):
    plt.scatter(
        df.index, r[i], marker=".", s=10, c="gray", alpha=0.5, label=f"Simulated inputs"
    )
obs = model_interface.df[model_interface.in_sol].to_numpy()
plt.plot(df.index, obs, ".", markersize=5, label="Observed inputs")
# plt.yscale("log")
# plt.xticks(time[1::90], rotation=30) 
# ax = plt.gca()
# ax.set_xticks(ax.get_xticks()[1::90])
plt.xlim(df.index[0], df.index[-1])
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[-2:], labels[-2:], loc="upper right", fontsize=12, ncol=2)
plt.ylabel("Concentration")
plt.tight_layout()
plt.savefig(f"{result_root}/input_scenarios_{case_name}.pdf")

# %%
# RUN pMCMC CHAIN 
# ================================================================

chain = Chain(model_interface=model_interface)
chain.run_particle_filter_SIR()
# %% Plot check

plt.figure()
plt.plot(model_interface.df["precip 18O"], "*")
plt.plot(model_interface.df.index, chain.state.R[:, :, 0].T, ".", markersize=0.7)
plt.xticks(rotation=30)
plt.ylabel("Concentration")
plt.tight_layout()
plt.title("Input concentration scenarios from SIR")

# %%
# for i in range(25):
#     plt.figure()
#     plt.plot(model_interface.df['C out'], "*")
#     plt.plot(model_interface.df.index, chain.state.Y[i,:,0].T)#, ".", markersize=0.7)
plt.figure()
plt.plot(model_interface.df["ORPB 18O"], "*")
plt.plot(model_interface.df.index, chain.state.Y[:, :, 0].T, ".", markersize=0.7)
plt.plot(
    model_interface.df.index,
    chain.state.Y[np.argmax(chain.state.W), :, 0].T,
    ".",
    markersize=10,
)
plt.xticks(rotation=30)
plt.ylabel("Concentration")
plt.tight_layout()
plt.title("Output concentration scenarios from SIR")
print("done SIR check")


# %%

chain.run_particle_filter_AS()

plt.figure()
sns.boxplot(chain.state.R[:, :, 0])
plt.plot(model_interface.df["precip 18O"].values, "*")
plt.xticks(rotation=30)
plt.ylabel("Concentration")
plt.tight_layout()
plt.title("Input concentration scenarios from AS")
r = chain.state.R[:, :, 0].T
q = chain.state.Y[:, :, 0].T
plt.figure()
plt.plot(model_interface.df["ORPB 18O"], "*")
plt.plot(model_interface.df.index, chain.state.Y[:, :, 0].T, ".", markersize=0.7)
plt.xticks(rotation=30)
plt.ylabel("Concentration")
plt.tight_layout()
plt.title("Output concentration scenarios from AS")
print("done AS check")

# %%

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

input_scenarios = model.input_record
output_scenarios = model.output_record
df = model_interface.df


#%% plot each MCMC iteration output against observed

plt.figure()
st, et = model_interface.observed_ind[0], model_interface.observed_ind[-1]
# plt.plot(model_interface.df["Q"], "grey")

time = model_interface.df.index
# plt.plot(time[st:et], output_scenarios[:, st:et].T,'.', alpha = 0.1, color="grey",lw=0.5, label="predicted")
# make a step plot
# plt.step(time[st:et], model_interface.df["C out"].backfill().iloc[st:et], label= "observed")
# plt.plot(time[st:et],model_interface.df["C out"].iloc[st:et], "*", label= "observed")
for i in range(len_parameter_MCMC + 1):
    plt.figure(figsize=(12, 4))

    plt.step(
        time[st:et],
        model_interface.df["ORPB 18O"].backfill().iloc[st:et],
        label="observed",
    )
    plt.plot(
        time[st:et], output_scenarios[i, st:et].T, color="orange", label="predicted"
    )
    plt.title(f'pMCMC iteration {i}')
    # plt.xlim([time[0], time[-1]])
#     plt.plot(model_interface.df.index[st:et], output_scenarios[i, st:et].T,   label=f"output {i}", lw=0.5)
plt.legend(frameon=False)
# np.save(f"output.npy", output_scenarios)

# %%
# # save data as csv files
theta_df.to_csv(f"{result_root}/theta_{case_name}.csv")

np.savetxt(f"{result_root}/input_scenarios_{case_name}.csv", input_scenarios, delimiter=",")
np.savetxt(f"{result_root}/output_scenarios_{case_name}.csv", output_scenarios, delimiter=",")


# %% RELOAD AND PLOT SAVED RESULTS
# ================================================================

input_scenarios = pd.read_table(f"{result_root}/input_scenarios_{case_name}.csv", delimiter=",", header=None)
output_scenarios = pd.read_table(f"{result_root}/output_scenarios_{case_name}.csv", delimiter=",", header=None)
time = model_interface.df.index
st, et = model_interface.observed_ind[0], model_interface.observed_ind[-1]

plt.figure()
for i in range(len_parameter_MCMC + 1):
    plt.figure(figsize=(12, 4))

    plt.step(
        time[st:et],
        model_interface.df["ORPB 18O"].backfill().iloc[st:et],
        label="observed",
    )
    plt.plot(
        time[st:et], output_scenarios[i, st:et].T, color="orange", label="predicted"
    )
    plt.title(f'pMCMC iteration {i}')
    # plt.xlim([time[0], time[-1]])
#     plt.plot(model_interface.df.index[st:et], output_scenarios[i, st:et].T,   label=f"output {i}", lw=0.5)
plt.legend(frameon=False)