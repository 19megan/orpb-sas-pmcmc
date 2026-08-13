# This is the main script to run pMCMC for ORPB data using mesas SAS model
# Created on: 11/10/2025
#%%
import os
import sys


# Clear previously imported modules to ensure fresh import of edited files
modules_to_clear = [
    "ORPB_mesas_interface",
    "mesas.sas.model",
    "mesas.sas.specs",
    "orpb_stochastic.model.utils_chain",
    "orpb_stochastic.model.ssm_model",
    "orpb_stochastic.functions.utils"
]
for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]


import pandas as pd

from ORPB_mesas_interface import ModelInterfaceMesas
import matplotlib.pyplot as plt
import numpy as np
from orpb_stochastic.model.utils_chain import Chain
from orpb_stochastic.functions.utils import plot_MAP

from orpb_stochastic.model.ssm_model import SSModel
from mesas.sas.model import Model as SAS_Model
import argparse
from mesas.sas.specs import Component

from ORPB_cases import *
from orpb_stochastic.functions.run_config import save_run_config, set_run_seed
import seaborn as sns

# Reproducibility: set a fixed seed for this run. Change per-run if desired,
# but always log it (save_run_config picks it up below).
RUN_SEED = 42
set_run_seed(RUN_SEED)

#ignore warnings on division by zero in normalize_over_interval
np.seterr(divide='ignore', invalid='ignore')

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
res = 'D' #'D'
resolution = 'daily' #'daily'
data_df = pd.read_csv(f"{data_resolution_root}/ORPB_isotope_data_bfill_precip 18O_{resolution}.csv", index_col=0, parse_dates=[0])

data_df['precip 18O'] = data_df['mean_c']

data_df = data_df.loc[pd.Timestamp('2015-01-01'): pd.Timestamp('2015-03-31 23:00:00')] #2014-08-01 - 2016-08-31subset to Putnam's data range
tag='D3M' #NOTE: must change this in ORPB_cases.py too

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
df['precip 18O'] = df['precip 18O'].bfill().ffill()


df['is_obs_output'] = df['ORPB 18O'].notna()

case_name = 'storage_q_ug_et_u_cp'#********************

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
    'update_theta_dist': False, # we want to keep prior param dists fixed at every iteration, but data-driven if set to True
}

model_interface_class = ModelInterfaceMesas

if case_name == 'storage_q_ug_et_u_cp':
    model_interface = model_interface_class(
        df=df,
        customized_model=SAS_Model,
        num_input_scenarios=num_input_scenarios,
        config=config,
        theta_init=theta_storage_q_ug_et_u_cp
    )
elif case_name == 'storage_q_gg_et_u':
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

#%%

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

#for plotting observed vs predicted outputs
st, et = model_interface.observed_ind[0], model_interface.observed_ind[-1]
time = model_interface.df.index
# %%
# # save data as csv files
tag = 'h1Y'
theta_df.to_csv(f"{result_root}/theta_{case_name}_{tag}.csv")
theta_std_df.to_csv(f"{result_root}/theta_std_{case_name}_{tag}.csv")
np.savetxt(f"{result_root}/input_scenarios_{case_name}_{tag}.csv", input_scenarios, delimiter=",")
np.savetxt(f"{result_root}/output_scenarios_{case_name}_{tag}.csv", output_scenarios, delimiter=",")

np.savetxt(f"{result_root}/pQ_mle_{case_name}_{tag}.csv", pQ_mle, delimiter=",")
np.save(f"{result_root}/sT_mle_{case_name}_{tag}.npy", sT_mle)
np.save(f"{result_root}/ST_mle_{case_name}_{tag}.npy", ST_mle)

# Save run configuration snapshot (for reproducibility / auditing).
# Looks up theta_{case_name} from globals so it matches what was actually passed
# to model_interface above.
theta_init_snapshot = globals()[f"theta_{case_name}"]
save_run_config(
    out_path=f"{result_root}/run_config_{case_name}_{tag}.json",
    case_name=case_name,
    config=config,
    N=num_input_scenarios,
    D=num_parameter_samples,
    L=len_parameter_MCMC,
    date_start=df.index[0],
    date_end=df.index[-1],
    resolution=res,
    data_source=f"ORPB_isotope_data_isoMAP_precip 18O_{resolution}.csv",
    theta_to_estimate=model_interface._theta_to_estimate,
    theta_init=theta_init_snapshot,
    seed=RUN_SEED,
)

# %% RELOAD AND PLOT SAVED RESULTS
# ================================================================
job_id = 29559083 #29549979 #29558156 #29536818 #29533975 #29530372 #29529496 #29402285 #27418806 #27228056 #27188083 #26141032 #26305810 #26135647 #25498196 #25481520 #25479958 #25439135 #25437692 #25421159 #25359163 #25357231 #25351802 #25251898
tag = 'W6M' #'D3M' #'W3M' #'D2Y' #'D1Y' #'D6M' #'D6M' #'h6M' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D5Y'
pQ_mle = pd.read_table(f"{result_root}/pQ_mle_{case_name}_{tag}_job{job_id}.csv", delimiter=",", header=None).to_numpy()
PQ_mle = np.cumsum(pQ_mle, axis=0)*1 # * config['dt']
input_scenarios = pd.read_table(f"{result_root}/input_scenarios_{case_name}_{tag}_job{job_id}.csv", delimiter=",", header=None)
output_scenarios = pd.read_table(f"{result_root}/output_scenarios_{case_name}_{tag}_job{job_id}.csv", delimiter=",", header=None)
time = df.index
st, et = df.index.get_loc(df['ORPB 18O'].first_valid_index()), df.index.get_loc(df['ORPB 18O'].last_valid_index())

theta_df = pd.read_csv(f"{result_root}/theta_{case_name}_{tag}_job{job_id}.csv", index_col=0)
theta_std_df = pd.read_csv(f"{result_root}/theta_std_{case_name}_{tag}_job{job_id}.csv", index_col=0)

#%%

# ANALYSIS AND PLOTTING
# ===============================================================
# Check convergence of theta parameters
for i in range(len(theta_df.columns)-3):
    # plot cumulative running mean of each parameter
    run_mean = np.cumsum(theta_df.iloc[:, i]) / np.arange(1, len(theta_df)+1)
    plt.figure(figsize=(12,4))
    plt.plot(run_mean)
    plt.title(f'{theta_df.columns[i]}')
    plt.xlabel('MCMC iteration')
    plt.ylabel('Cumulative Mean')
    plt.show()
    # check autocorrelation: number of independent draws
    x = theta_df.iloc[25:, i] - theta_df.iloc[25:, i].mean()   # drop first ~25 as burn-in
    acf = np.correlate(x, x, 'full')[len(x)-1:] / (x.var()*len(x))
    tau = 1 + 2*np.sum(acf[1:np.argmax(acf<0.05)])         # integrated autocorr time
    ess = len(x) / tau #want close to MC length for convergence, but if se is good then it's fine.
    se = x.std()/np.sqrt(ess)
    post_std = theta_std_df.iloc[25:, i].mean() # mean of std of posterior draws
    print(f'Posterior parameter mean = {theta_df.iloc[25:, i].mean():.3f}, std = {post_std:.3f}')
    print(f"ESS ≈ {ess:.0f} of {len(x)}, SE of mean ≈ {se:.3f}")
    print(f"SE/post_std = {se/post_std:.3f} (want << 1)") #if all seeds have <<1, convergence across seeds is solved
    # Geweke split test: compare mean of first 10% vs last 50% after burn-in
    a, b = theta_df.iloc[25:35, i], theta_df.iloc[75:, i]
    z = (a.mean() - b.mean()) / np.sqrt(a.var()/len(a) + b.var()/len(b))
    print(f'Z score: {z} (should be within +-2) - stationary')

#%%
# Plot distributions of theta parameters
from scipy.stats import gaussian_kde
def kde_mode(samples):
    """Estimate mode of continuous samples using KDE."""
    samples = samples[~np.isnan(samples)]
    kde = gaussian_kde(samples)
    x = np.linspace(samples.min(), samples.max(), 1000)
    return x[np.argmax(kde(x))]

means = theta_df.iloc[25:, :].mean(axis=0)
stds = theta_std_df.iloc[25:, :].mean(axis=0)
ncols = 3
nrows = int(np.ceil((len(means)-3)/ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()
for i in range(len(means)-3):
    mean = means.iloc[i]
    std = stds.iloc[i]
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

# plt.figure(figsize=(12,4))
# plt.plot(model_interface.df["discharge (mm/hr)"], "grey")

# plt.plot(time[st:et], output_scenarios[-1:, st:et].T,'.', alpha = 0.1, color="grey",lw=0.5, label="predicted")
# # make a step plot
# plt.step(time[st:et], model_interface.df["ORPB 18O"].backfill().iloc[st:et], label= "observed")
# plt.plot(time[st:et],model_interface.df["ORPB 18O"].iloc[st:et], "*", label= "observed")

for i in range(len_parameter_MCMC - 3, len_parameter_MCMC + 1):
    plt.figure(figsize=(12, 4))

    plt.step(
        time[st:et],
        df["ORPB 18O"].bfill().iloc[st:et],
        label="observed",
    )
    # plt.plot(# for normal runs - no iloc
    #     time[st:et], output_scenarios[i, st:et].T, color="orange", alpha=0.5, label="predicted"
    # )
    plt.plot(#for Rockfish imported runs--needs iloc
        time[st:et], output_scenarios.iloc[i, st:et].T, color="orange", alpha=0.5, label="predicted"
    )
    plt.title(f'pMCMC iteration {i}')
    plt.xlabel('Date')
    plt.ylabel('delta 18O in stream (per mil)')
    # plt.xlim([time[0], time[-1]])
#     plt.plot(model_interface.df.index[st:et], output_scenarios[i, st:et].T,   label=f"output {i}", lw=0.5)
plt.legend(frameon=False)
# np.save(f"output.npy", output_scenarios)
plt.show()

#or can plot average predictions with "predicted = output_scenarios.iloc[25:, st:et].mean(axis=0)""

#%%
# check accuracy
from permetrics.regression import RegressionMetric
import hydroeval as he
# obs = df['ORPB 18O'].bfill()[st:et].to_numpy()
obs = df['ORPB 18O'].iloc[st:et].to_numpy()#for comparing with Rockfish imported results have to set df to same length and use that to compare
mask = df['is_obs_output'].iloc[st:et]
obs = obs[mask]
for i in range(len_parameter_MCMC - 3, len_parameter_MCMC + 1):
    # pred = output_scenarios[i, st:et]
    pred = output_scenarios.iloc[i, st:et].to_numpy() #for Rockfish runs
    pred = pred[mask]
    # evaluator = RegressionMetric(obs, pred)
    # kge = evaluator.kling_gupta_efficiency()
    print(f'pMCMC iteration {i}')
    # print(f'KGE = {kge}')
    nse = he.evaluator(he.nse, pred, obs)
    print(f'NSE = {nse[0]}')
    RMSE = np.sqrt(np.mean((pred-obs)**2))
    print(f'RMSE = {RMSE}')

#%% #plot histogram of residuals for last MCMC iteration
error = obs-pred
plt.figure(figsize=(12,4))
plt.hist(error, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

#%%
# Plot TTD for last MC iteration
# ST = (ST_mle[:,:-1] + ST_mle[:,1:])/2
T = 1*np.arange(pQ_mle.shape[0]) # model.options['dt'] *np.arange(model.options['max_age']) #max age is 1st dim of pq and ST
import matplotlib.cm as cm
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0,1,len(df))]
fig, ax = plt.subplots()
for i in range(0, len(df)):
    plt.plot(T, pQ_mle[:,i], color=colors[i]) #TTD
plt.xlabel(f'Age ({res})')
plt.ylabel('$p_Q$')
plt.title('Discharge SAS function over time')
# plt.xlim([0, 5])
plt.ylim([0, 1])
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0,
vmax=len(df)-1))
plt.colorbar(sm, ax=ax, label='Time Index')

# %% Cumulative TTD
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0,1,len(df))]
fig, ax = plt.subplots()
for i in range(0, len(df)):
    plt.plot(T, PQ_mle[:,i], color=colors[i]) #TTD
plt.xlabel(f'Age ({res})')
plt.ylabel('$P_Q$')
plt.title('Cumulative Discharge SAS function over time')
plt.ylim([0, 1.1])
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0,
vmax=len(df)-1))
plt.colorbar(sm, ax=ax, label='Time Index')


#%%
# Plot residuals between observed and predicted at each MCMC iteration
#======================================================================
plt.figure()
for i in range(len_parameter_MCMC + 1):
    plt.figure(figsize=(12, 4))

    plt.step(
        time[st:et],
        (df["ORPB 18O"].backfill().iloc[st:et]).to_numpy() - (output_scenarios.iloc[i, st:et].T),
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


# Compute and plot KLD distributions of theta parameters
# ======================================================
from scipy.stats import gaussian_kde
from astropy.stats import knuth_bin_width # maximizes posterior probability of the histogram - closely related to Shannon entropy
from scipy.stats import entropy
prior_means = [0.51, 3.0, 1800, 52.5, -7.28, 0.08, 0.08, 0.08]
prior_stds = [0.249, 1.02, 200, 24.23, 0.728, 0.01, 1.17, 0.01]
post_means = theta_df.iloc[25:, :].mean(axis=0)
post_stds = theta_std_df.iloc[25:, :].mean(axis=0)

ncols = 3
nrows = int(np.ceil((len(prior_means)-3)/ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()
np.random.seed(1)
for i in range(len(prior_means)-3):
    prior_mean = prior_means[i]
    prior_std = prior_stds[i]
    post_mean = post_means.iloc[i]
    post_std = post_stds.iloc[i]
    
    prior_hist = np.random.normal(loc=prior_mean, scale=prior_std, size=10000)
    post_hist = np.random.normal(loc=post_mean, scale=post_std, size=10000)

    P_bin_width = knuth_bin_width(prior_hist) #prior=P
    P_bins = np.arange(prior_hist.min(), prior_hist.max() + P_bin_width, P_bin_width)
    Q_bin_width = knuth_bin_width(post_hist) #posterior=Q
    Q_bins = np.arange(post_hist.min(), post_hist.max() + Q_bin_width, Q_bin_width)

    counts, bin_edges = np.histogram(prior_hist, bins=P_bins)
    P_probabilities = counts / counts.sum()
    counts, bin_edges = np.histogram(post_hist, bins=P_bins) #has to be same bins for KLD
    Q_probabilities = counts / counts.sum()

    kld_bits = entropy(Q_probabilities, P_probabilities, base=2)      # log₂ ⇒ bits
    kld_nats = entropy(Q_probabilities, P_probabilities)              # natural log ⇒ nats
    # Plot histogram
    ax = axes[i]
    ax.hist(prior_hist, bins=P_bins, density=True, color='grey', edgecolor='black', alpha=0.4, label='prior dist')
    ax.hist(post_hist, bins=P_bins, density=True, color='steelblue', edgecolor='black', alpha=0.7, label='posterior dist')
    ax.plot(prior_mean,0, label=f'KLD = {kld_nats} nats')
    ax.set_title(f'{theta_df.columns[i]}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.axvline(prior_mean, color='black', linestyle='--', label=f'Prior Mean = {prior_mean}')
    ax.axvline(post_mean, color='red', linestyle='--', label=f'Posterior Mean = {post_mean}')
    labels = ax.get_legend_handles_labels()
    ax.legend(labels[0][2:], labels[1][2:])
axes[i+1].axis('off')
axes[i+1].legend(labels[0][0:2], labels[1][0:2])
# for j in range(i+1, len(axes)):
#     axes[j].set_visible(False)
fig.tight_layout()
plt.show()

