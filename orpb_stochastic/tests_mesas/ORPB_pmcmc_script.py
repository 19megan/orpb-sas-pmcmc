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
data_df = pd.read_csv(f"{data_resolution_root}/ORPB_isotope_data_precip 18O_{resolution}.csv", index_col=0, parse_dates=[0])

data_df['precip 18O'] = data_df['mean_c']

# calibration window with the preceding year of fluxes prepended (prepend_spinup, ORPB_cases.py);
# both isotopes are NaN in the spinup and water that entered then takes C_old
data_df, n_spinup = prepend_spinup(data_df, '2015-01-01', '2015-03-31 23:00:00')
options['max_age'] = n_spinup # one year of resolved ages at every observation (shared by all theta_* cases)
mask_holdout(data_df, 'daily') # held-out validation months (VALIDATION_PERIODS, ORPB_cases.py) never enter calibration
tag='D3M'

issample = np.logical_not(np.isnan(data_df['ORPB 18O']))
data_df['influx (mm/hr)'] = data_df[['rainfall (mm/hr)','snowmelt (mm/hr)']].sum(axis=1)

# separate Q into quickflow and baseflow 1
data_df['quickflow (mm/hr)'] = data_df['discharge (mm/hr)'] - data_df['baseflow 1 (mm/hr)']
data_df['bf1_weight'] = data_df['baseflow 1 (mm/hr)'] / data_df['discharge (mm/hr)']
data_df['qf_weight'] = data_df['quickflow (mm/hr)'] / data_df['discharge (mm/hr)']

# fill nans in precip 18O with mean
mean = data_df['precip 18O'].mean()
df= data_df.copy() #make a copy of the data_df
# df.loc[df['precip 18O'].isna()==True, 'precip 18O']=mean
df['precip 18O'] = df['precip 18O'].bfill().ffill()

df['is_obs_input'] = df['precip 18O'].notna()
df['is_obs_output'] = df['ORPB 18O'].notna()

case_name = 'storage_q_ug_et_u_cp'#********************

num_input_scenarios = 3# 15 #N particles
num_parameter_samples = 3 #15 #D #may need to change this as I change isotope timeseries configurations (fewer obs mean fewer timesteps where particles get reweighted which affect the variance of the likelihood estimate)
len_parameter_MCMC = 4000#5 #L

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
from pytictoc import TicToc
t = TicToc()
t.tic()
model.run_particle_Gibbs()
t.toc()

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
# importance-weight ESS per (iteration, parameter); row 0 is unfilled by design
np.savetxt(f"{result_root}/ess_{case_name}_{tag}.csv", model.ess_record, delimiter=",")
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
    date_start=df.index[n_spinup], # calibration window; spinup is df.index[:n_spinup]
    date_end=df.index[-1],
    resolution=res,
    data_source=f"ORPB_isotope_data_isoMAP_precip 18O_{resolution}.csv",
    theta_to_estimate=model_interface._theta_to_estimate,
    theta_init=theta_init_snapshot,
    seed=RUN_SEED,
)

# %% RELOAD AND PLOT SAVED RESULTS
# ================================================================
job_id = 32551260 #32522132 #32489531 #32353996 #32361285 ###31124028 #30318146 #30284266 #30283655 #30129037 #30127986 #30124944 #30103937 #30082173 #30059682 #30020017 #30012536 #30010587 #29974181 #29958090 #29939048 #29933214 #29932998 #29871348 #29855297 #29823188 #29559083 #29549979 #29558156 #29536818 #29533975 #29530372 #29529496 #29402285 #27418806 #27228056 #27188083 #26141032 #26305810 #26135647 #25498196 #25481520 #25479958 #25439135 #25437692 #25421159 #25359163 #25357231 #25351802 #25251898
tag = '2W6M' #'W6M' #'W2Y' #'D2Y' #'D6M' ###'D3M' #'D_3MF' #'ME_3M' #'D_3M' #'D_3M' #'ME_3ystd' #'D_3ystd' #'D4Y' #'ME4Y' #'ME2Y' #'ME1Y' #'ME6M' #'ME3M' #'2W4Y' #'2W2Y' #'2W1Y' #'2W6M' #'2W3M' #'W4Y' #'W2Y' #'W1Y' #'W6M' #'D3M' #'W3M' #'D2Y' #'D1Y' #'D6M' #'D6M' #'h6M' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D5Y'
pQ_mle = pd.read_table(f"{result_root}/pQ_mle_{case_name}_{tag}_job{job_id}.csv", delimiter=",", header=None).to_numpy()
PQ_mle = np.cumsum(pQ_mle, axis=0)*1 # * config['dt']
input_scenarios = pd.read_table(f"{result_root}/input_scenarios_{case_name}_{tag}_job{job_id}.csv", delimiter=",", header=None)
output_scenarios = pd.read_table(f"{result_root}/output_scenarios_{case_name}_{tag}_job{job_id}.csv", delimiter=",", header=None)
time = df.index
st, et = df.index.get_loc(df['ORPB 18O'].first_valid_index()), df.index.get_loc(df['ORPB 18O'].last_valid_index())

theta_df = pd.read_csv(f"{result_root}/theta_{case_name}_{tag}_job{job_id}.csv", index_col=0)
# theta_std_df = pd.read_csv(f"{result_root}/theta_std_{case_name}_{tag}_job{job_id}.csv", index_col=0)

#%%

# ANALYSIS AND PLOTTING
# ===============================================================
# Check convergence of theta parameters
burn_in = 1000 # drop first ~25 as burn-in
for i in range(len(theta_df.columns)-2): # skip the 2 sigmas at the end (3 before sigma filled C in was removed)
    # plot cumulative running mean of each parameter
    run_mean = np.cumsum(theta_df.iloc[:, i]) / np.arange(1, len(theta_df)+1)
    plt.figure(figsize=(12,4))
    plt.plot(run_mean)
    plt.title(f'{theta_df.columns[i]}')
    plt.xlabel('MCMC iteration')
    plt.ylabel('Cumulative Mean')
    plt.show()
    # check autocorrelation: number of independent draws
    x = theta_df.iloc[burn_in:, i] - theta_df.iloc[burn_in:, i].mean()   # drop first ~25 as burn-in
    acf = np.correlate(x, x, 'full')[len(x)-1:] / (x.var()*len(x))
    tau = 1 + 2*np.sum(acf[1:np.argmax(acf<0.05)])         # integrated autocorr time
    ess = len(x) / tau #want close to MC length for convergence, but if se is good then it's fine.
    se = x.std()/np.sqrt(ess)
    # post_std = theta_std_df.iloc[burn_in:, i].mean() # mean of std of posterior draws
    post_std = theta_df.iloc[burn_in:, i].std()
    print(f'Posterior parameter mean = {theta_df.iloc[burn_in:, i].mean():.3f}, std = {post_std:.3f}')
    print(f"ESS ≈ {ess:.0f} of {len(x)}, SE of mean ≈ {se:.3f}")
    print(f"SE/post_std = {se/post_std:.3f} (want << 1)") #if all seeds have <<1, convergence across seeds is solved
    # Geweke split test: compare mean of first 10% vs last 50% after burn-in
    a, b = theta_df.iloc[burn_in:int(burn_in*1.25), i], theta_df.iloc[int(burn_in*2):int(burn_in*2.25), i]
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

means = theta_df.iloc[burn_in:, :].mean(axis=0)
stds = theta_std_df.iloc[burn_in:, :].mean(axis=0) #for PG runs
ncols = 3
nrows = int(np.ceil((len(means)-2)/ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()
for i in range(len(means)-2):
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

for i in range(len_parameter_MCMC - 1, len_parameter_MCMC + 1):
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
for i in range(len_parameter_MCMC - 1, len_parameter_MCMC + 1):
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
# Priors as [lower, upper, shape], in the order of _theta_to_estimate, where shape is
# 'lin' for uniform and 'log' for log-uniform. The two sigmas (sigma C in, sigma C out) are
# left out of this list, so they are not plotted; add them here ([0.08, 3.5, 'log'] each) if wanted.
# NOTE: must match the priors in ORPB_cases.py for the case being plotted.
prior_bounds = [
    [0.1, 200.0, 'log'],       # qf_weight scale
    [0.1, 10.0, 'lin'],        # bf1_weight a
    [10.0, 10000.0, 'log'],    # bf1_weight M (MEAN age-ranked storage; scale = M / a)
    [0.1, 500.0, 'log'],       # ET scale
    [-19.0, 2.0, 'lin'],       # C_old
]
# --- OLD VERSION (kept for reference) -------------------------------------------------------
# Binning BOTH histograms over the posterior's range (Q_bins) renormalises the prior to that
# range, which discards the narrowing KLD is meant to measure. Replaced by the block below.
# post_means = theta_df.iloc[25:, :].mean(axis=0)
# post_stds = theta_std_df.iloc[25:, :].mean(axis=0)
#
# ncols = 3
# nrows = int(np.ceil(len(prior_bounds)/ncols))
# fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
# axes = axes.flatten()
# np.random.seed(1)
# for i in range(len(prior_bounds)):
#     lo, hi, shape = prior_bounds[i]
#     prior_mean = np.exp((np.log(lo) + np.log(hi)) / 2) if shape == 'log' else (lo + hi) / 2
#     post_mean = post_means.iloc[i]
#     post_std = post_stds.iloc[i]
#
#     if shape == 'log':  # draw flat in log space, to match the loguniform prior
#         prior_hist = np.exp(np.random.uniform(np.log(lo), np.log(hi), size=10000))
#     else:
#         prior_hist = np.random.uniform(lo, hi, size=10000)
#     post_hist = np.random.normal(loc=post_mean, scale=post_std, size=10000)
#
#     P_bin_width = knuth_bin_width(prior_hist) #prior=P
#     P_bins = np.arange(prior_hist.min(), prior_hist.max() + P_bin_width, P_bin_width)
#     Q_bin_width = knuth_bin_width(post_hist) #posterior=Q
#     Q_bins = np.arange(post_hist.min(), post_hist.max() + Q_bin_width, Q_bin_width)
#
#     counts, bin_edges = np.histogram(prior_hist, bins=Q_bins)
#     P_probabilities = counts / counts.sum()
#     counts, bin_edges = np.histogram(post_hist, bins=Q_bins) #has to be same bins for KLD
#     Q_probabilities = counts / counts.sum()
#
#     kld_bits = entropy(Q_probabilities, P_probabilities, base=2)      # log₂ ⇒ bits
#     kld_nats = entropy(Q_probabilities, P_probabilities)              # natural log ⇒ nats
#     # Plot histogram
#     ax = axes[i]
#     ax.hist(prior_hist, bins=P_bins, density=True, color='grey', edgecolor='black', alpha=0.4, label='prior dist')
#     ax.hist(post_hist, bins=Q_bins, density=True, color='steelblue', edgecolor='black', alpha=0.7, label='posterior dist')
#     ax.plot(prior_mean,0, label=f'KLD = {kld_nats:.2f} nats')
#     ax.set_title(f'{theta_df.columns[i]}')
#     ax.set_xlabel('Value')
#     ax.set_ylabel('Frequency')
#     ax.axvline(prior_mean, color='black', linestyle='--', label=f'Prior range = [{lo:.2f}, {hi:.2f}]')
#     ax.axvline(post_mean, color='red', linestyle='--', label=f'Posterior [Mean, std] = [{post_mean:.2f}, {post_std:.2f}]')
#     labels = ax.get_legend_handles_labels()
#     ax.legend(labels[0][2:], labels[1][2:])
# axes[i+1].axis('off')
# axes[i+1].legend(labels[0][0:2], labels[1][0:2])
# # for j in range(i+1, len(axes)):
# #     axes[j].set_visible(False)
# fig.tight_layout()
# plt.show()
# --- END OLD VERSION ------------------------------------------------------------------------

# Posterior draws. From a pmmh run these are the real post-burn-in samples pooled over chains
# (pmmh_<case><tag>.npz). Otherwise fall back to the old normal(mean, std) approximation from
# theta/theta_std csvs, which is only meaningful for runs made with Xu's original
# weighted-mean code (before 2026-09-24).
pmmh_file = f"{result_root}/pmmh_{case_name}_{tag}_job{job_id}.npz"
if os.path.exists(pmmh_file):
    _pm = np.load(pmmh_file)
    _b = int(_pm["num_burnin"])
    post_draws = _pm["theta_chains"][:, _b + 1:, :].reshape(-1, _pm["theta_chains"].shape[-1])
else:
    post_draws = None
    post_means = theta_df.iloc[25:, :].mean(axis=0)
    post_stds = theta_df.iloc[25:,:].std(axis=0) #theta_std should be variation in L thetas, theta_std_df is null, theta_std_df.iloc[25:, :].mean(axis=0)

# KLD(posterior || prior): information gained over the prior.
# OLD (kept for reference): binned over the PRIOR's support with nbins = 40, in log space for
# log-uniform priors. A binned KLD can never exceed ln(nbins) = 3.69 nats, reached when the whole
# posterior falls in one bin, so it understated well-constrained parameters (C_old: 3.68 binned
# vs about 5.6 actual) and could not show further gains between windows.
# NEW: bin-free. In u = position within the prior range in the prior's flat coordinate, the
# prior is Uniform(0, 1), so KLD = -H(posterior of u). H is estimated with a Gaussian KDE
# whose draws are mirrored across 0 and 1, so no density is placed outside the prior's bounds.
# Checked against exact values (normal, two-mode, and piled-against-a-bound posteriors, with
# MCMC-style repeated values): within about 0.04 nats.
def kld_reflect(x, lo, hi, shape, n_fit=20000, n_eval=5000, seed=0):
    f = np.log if shape == 'log' else (lambda v: v)
    u = (f(np.asarray(x, float)) - f(lo)) / (f(hi) - f(lo))
    rng = np.random.default_rng(seed)
    kde = gaussian_kde(rng.choice(u, size=min(n_fit, u.size), replace=False))
    ev = rng.choice(u, size=min(n_eval, u.size), replace=False)
    return float(np.mean(np.log(kde(ev) + kde(-ev) + kde(2.0 - ev))))

nbins = 40  # histogram bins for the plots only
ncols = 3
nrows = int(np.ceil(len(prior_bounds)/ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()
np.random.seed(1)
for i in range(len(prior_bounds)):
    lo, hi, shape = prior_bounds[i]
    prior_mean = np.exp((np.log(lo) + np.log(hi)) / 2) if shape == 'log' else (lo + hi) / 2
    if post_draws is not None:
        post_hist = post_draws[:, i]
    else:
        post_hist = np.random.normal(loc=post_means.iloc[i], scale=post_stds.iloc[i], size=10000)
        post_hist = post_hist[(post_hist >= lo) & (post_hist <= hi)]  # prior has no mass outside
    post_mean, post_std = post_hist.mean(), post_hist.std()

    fwd = np.log if shape == 'log' else (lambda v: v)
    edges = np.linspace(fwd(lo), fwd(hi), nbins + 1)
    # OLD binned KLD (capped at ln(nbins), see above):
    # P_probabilities = np.full(nbins, 1.0 / nbins)
    # counts, _ = np.histogram(fwd(post_hist), bins=edges)
    # Q_probabilities = counts / counts.sum()
    # kld_bits = entropy(Q_probabilities, P_probabilities, base=2)      # log₂ ⇒ bits
    # kld_nats = entropy(Q_probabilities, P_probabilities)              # natural log ⇒ nats
    kld_nats = kld_reflect(post_hist, lo, hi, shape)                    # natural log ⇒ nats
    kld_bits = kld_nats / np.log(2)                                     # log₂ ⇒ bits
    # Plot histogram (x axis in the prior's own scale)
    plot_bins = np.exp(edges) if shape == 'log' else edges
    prior_hist = (np.exp(np.random.uniform(np.log(lo), np.log(hi), size=10000)) if shape == 'log'
                  else np.random.uniform(lo, hi, size=10000))
    ax = axes[i]
    ax.hist(prior_hist, bins=plot_bins, density=True, color='grey', edgecolor='black', alpha=0.4, label='prior dist')
    ax.hist(post_hist, bins=plot_bins, density=True, color='steelblue', edgecolor='black', alpha=0.7, label='posterior dist')
    if shape == 'log':
        ax.set_xscale('log')
    ax.plot(prior_mean,0, label=f'KLD = {kld_nats:.2f} nats ({kld_bits:.2f} bits)')
    ax.set_title(f'{theta_df.columns[i]}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.axvline(prior_mean, color='black', linestyle='--', label=f'Prior range = [{lo:.2f}, {hi:.2f}]')
    ax.axvline(post_mean, color='red', linestyle='--', label=f'Posterior [Mean, std] = [{post_mean:.2f}, {post_std:.2f}]')
    labels = ax.get_legend_handles_labels()
    ax.legend(labels[0][2:], labels[1][2:])
axes[i+1].axis('off')
axes[i+1].legend(labels[0][0:2], labels[1][0:2])
fig.tight_layout()
plt.show()

