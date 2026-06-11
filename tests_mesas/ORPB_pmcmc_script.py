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
from functions.run_config import save_run_config, set_run_seed
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
data_df = pd.read_csv(f"{data_resolution_root}/ORPB_isotope_data_isoMAP_precip 18O_{resolution}.csv", index_col=0, parse_dates=[0])
data_df['precip 18O'] = data_df['mean_c']
data_df['discharge (mm/hr)'] = data_df[f'discharge (mm/{res})'] # for convenience now
data_df['baseflow 1 (mm/hr)'] = data_df[f'baseflow 1 (mm/{res})']
data_df['snowmelt (mm/hr)'] = data_df[f'snowmelt (mm/{res})']
data_df['rainfall (mm/hr)'] = data_df[f'rainfall (mm/{res})']
data_df['ET (mm/hr)'] = data_df[f'ET (mm/{res})']


data_df = data_df.loc[pd.Timestamp('2015-01-01'): pd.Timestamp('2016-12-31')] #2014-08-01 - 2016-08-31subset to Putnam's data range
tag='D2Y' #NOTE: must change this in ORPB_cases.py too

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

sTmT = pd.read_csv(f'{data_resolution_root}/sT_mT_init_{tag}.csv')
df['mT_spinup'] = sTmT['mT_init'].values 
sT_init = sTmT['sT_init'].values

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
tag = 'D1Y'
theta_df.to_csv(f"{result_root}/theta_{case_name}_{tag}.csv")
theta_std_df.to_csv(f"{result_root}/theta_std_{case_name}_{tag}.csv")
np.savetxt(f"{result_root}/input_scenarios_{case_name}_{tag}.csv", input_scenarios, delimiter=",")
np.savetxt(f"{result_root}/output_scenarios_{case_name}_{tag}.csv", output_scenarios, delimiter=",")

np.save(f"{result_root}/pQ_mle_{case_name}_{tag}.npy", pQ_mle)
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
job_id = 25481520 #25479958 #25439135 #25437692 #25421159 #25359163 #25357231 #25351802 #25251898
tag = 'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D2Y' #'D5Y'
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
# Plot distributions of theta parameters
from scipy.stats import gaussian_kde
def kde_mode(samples):
    """Estimate mode of continuous samples using KDE."""
    samples = samples[~np.isnan(samples)]
    kde = gaussian_kde(samples)
    x = np.linspace(samples.min(), samples.max(), 1000)
    return x[np.argmax(kde(x))]

MC_idx=len_parameter_MCMC - 1 # index of MCMC iteration to plot
means = theta_df.iloc[MC_idx]
stds = theta_std_df.iloc[MC_idx]
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

for i in range(len_parameter_MCMC - 5, len_parameter_MCMC + 1):
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

#%%
# check accuracy
from permetrics.regression import RegressionMetric
import hydroeval as he
# obs = df['ORPB 18O'].bfill()[st:et].to_numpy()
obs = df['ORPB 18O'].bfill().iloc[st:et].to_numpy()#for comparing with Rockfish imported results have to set df to same length and use that to compare
for i in range(len_parameter_MCMC - 5, len_parameter_MCMC + 1):
    # pred = output_scenarios[i, st:et]
    pred = output_scenarios.iloc[i, st:et].to_numpy() #for Rockfish runs
    # evaluator = RegressionMetric(obs, pred)
    # kge = evaluator.kling_gupta_efficiency()
    print(f'pMCMC iteration {i}')
    # print(f'KGE = {kge}')
    nse = he.evaluator(he.nse, pred, obs)
    print(f'NSE = {nse[0]}')
    RMSE = np.sqrt(np.mean((pred-obs)**2))
    print(f'RMSE = {RMSE}')

#%% #plot histogram of residuals for last MCMC iteration
mask = df['is_obs_output'].iloc[st:et]
obs = obs[mask]
pred = pred[mask]
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




#%%
# Compute KLD between prior and posterior residuals
# =====================================================================
# Builds two pooled residual (obs - pred) distributions and computes the
# information-gain KLD between them: D_KL(posterior_error || prior_error).
#
# CAVEAT (see papers/codebase_evaluation_ORPB.md sec 3.4): output_scenarios
# (= output_record) are DATA-CONDITIONED filtered estimates p(C_Q | y, theta),
# whereas the saved 200 prior runs are UNCONDITIONAL forward predictions
# p(C_Q | theta). The KLD below therefore reflects the COMBINED effect of
# (a) parameter learning and (b) particle-filter data conditioning -- NOT pure
# parameter information gain. For an apples-to-apples prior-vs-posterior
# predictive KLD, regenerate the posterior side with the SAME forward machinery
# used for the prior (ORPB_SAS_check.py) evaluated at posterior theta draws.
from scipy.stats import gaussian_kde

# --- observation series + real-obs mask (exclude backfilled points) ---
obs_full = df['ORPB 18O'].bfill().iloc[st:et].to_numpy()
obs_mask = df['is_obs_output'].iloc[st:et].to_numpy()          # True = real obs
obs_vals = obs_full[obs_mask]                                  # (n_obs,)

# --- PRIOR error pool: M=200 saved prior prediction time series ---
PRIOR_PRED_PATH = f"{data_resolution_root}/residuals_{tag}.csv"  # <-- set to your file
PRIOR_FILE_IS_RESIDUALS = True   # True if the file already stores (obs - pred) instead of predictions

prior_raw = pd.read_csv(PRIOR_PRED_PATH).to_numpy()  # expected (M, T_full), same cols as df
prior_sub = prior_raw[:, st:et][:, obs_mask]                      # (M, n_obs)
prior_resid = prior_sub.ravel() if PRIOR_FILE_IS_RESIDUALS else (obs_vals[None, :] - prior_sub).ravel()
prior_resid = prior_resid[np.isfinite(prior_resid)]

# --- POSTERIOR error pool: post-burn-in MCMC iterations (NOT just the last) ---
out = np.asarray(output_scenarios)                  # (L+1, T_full); handles DataFrame or ndarray
burn_in   = (len_parameter_MCMC + 1) // 2           # discard first half of the chain
post_rows = np.arange(burn_in, len_parameter_MCMC + 1)
post_pred = out[post_rows][:, st:et][:, obs_mask]   # (n_iter, n_obs)
post_resid = (obs_vals[None, :] - post_pred).ravel()
post_resid = post_resid[np.isfinite(post_resid)]

# --- KLD estimators (direction: posterior || prior = information gain) ---
def kl_gaussian(p, q):
    """Closed-form KL(P||Q) for two fitted Gaussians. Headline number."""
    mp, sp = p.mean(), p.std(ddof=1)
    mq, sq = q.mean(), q.std(ddof=1)
    return np.log(sq / sp) + (sp**2 + (mp - mq)**2) / (2 * sq**2) - 0.5

def kl_kde(p, q, n=2000, eps=1e-12):
    """KDE + numerical integration KL(P||Q). Robustness check for non-Gaussian residuals."""
    lo, hi = min(p.min(), q.min()), max(p.max(), q.max())
    pad = 0.1 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, n)
    fp = gaussian_kde(p)(grid); fq = gaussian_kde(q)(grid)
    fp /= np.trapz(fp, grid); fq /= np.trapz(fq, grid)
    fq = np.clip(fq, eps, None)                      # guard log where posterior tail -> 0
    integrand = np.where(fp > eps, fp * np.log(fp / fq), 0.0)
    return np.trapz(integrand, grid)

kld_gauss = kl_gaussian(post_resid, prior_resid)
kld_kde   = kl_kde(post_resid, prior_resid)

print(f"prior:     n={prior_resid.size:6d}  mean={prior_resid.mean():+.4f}  std={prior_resid.std(ddof=1):.4f}")
print(f"posterior: n={post_resid.size:6d}  mean={post_resid.mean():+.4f}  std={post_resid.std(ddof=1):.4f}")
print(f"D_KL(posterior || prior)  Gaussian = {kld_gauss:.4f} nats")
print(f"D_KL(posterior || prior)  KDE      = {kld_kde:.4f} nats")

# --- overlay plot ---
plt.figure(figsize=(8, 5))
lo = min(prior_resid.min(), post_resid.min())
hi = max(prior_resid.max(), post_resid.max())
bins = np.linspace(lo, hi, 80)
plt.hist(prior_resid, bins=bins, density=True, alpha=0.45, color='grey',      label='prior error')
plt.hist(post_resid,  bins=bins, density=True, alpha=0.45, color='steelblue', label='posterior error')
xx = np.linspace(lo, hi, 400)
plt.plot(xx, gaussian_kde(prior_resid)(xx), color='black', lw=1)
plt.plot(xx, gaussian_kde(post_resid)(xx),  color='navy',  lw=1)
plt.axvline(0, color='red', ls='--', lw=1)
plt.xlabel('Residual  (obs - pred),  per mil')
plt.ylabel('Density')
plt.title(f'Error distributions   |   D_KL(post||prior) = {kld_kde:.3f} nats')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

#%%
# FORWARD POSTERIOR-PREDICTIVE ERROR KLD  (apples-to-apples vs the prior)
# =====================================================================
# Unlike the cell above (which uses the FILTERED output_scenarios), this builds
# the posterior error pool from UNCONDITIONAL FORWARD runs at posterior theta
# draws -- the same kind of prediction as the saved 200 prior runs. This removes
# the particle-filter "more-reweighting-steps-at-higher-resolution" confound, so
# it is the cleaner metric for comparing data resolutions.
#
# Forward runs reuse the interface's own theta->model mapping
# (model_interface.update_theta -> _init_sas_model), which handles the active
# case (incl. *_cp) correctly. The deterministic mesas convolution runs on the
# (filled) observed input, so the prediction depends on the SAS-shape params and
# C_old; the sigma_* uncertainty params don't change the forward output.
#
# Requires the cell above to have run (defines prior_resid, obs_vals, obs_mask,
# kl_gaussian, kl_kde, st, et) and a live `model_interface` (from the RUN MODEL cell).

PRED_COL = 'precip 18O --> discharge (mm/hr)'   # mesas forward-output column ('{solute} --> {flux}')
USE_WITHIN_ITER_SPREAD = True  # False: one forward run per post-burn-in iteration mean (robust, ~rows/2 runs)
                                # True : mixture-sample theta ~ N(mean_l, std_l), K_FWD draws total
K_FWD = 200                     # number of mixture draws (used only if USE_WITHIN_ITER_SPREAD)
REGENERATE_PRIOR_FORWARD = False  # True: redraw M_PRIOR prior thetas through the SAME forward_pred()
M_PRIOR = 200                     #       machinery (fully self-consistent prior); False: reuse prior_resid

theta_cols = list(theta_df.columns)             # == model_interface._theta_to_estimate order
n_rows = len(theta_df)
post_rows_theta = np.arange(n_rows // 2, n_rows)  # discard first half as burn-in

def forward_pred(theta_vec):
    """Deterministic mesas forward prediction of stream 18O for a theta vector."""
    model_interface.update_theta(np.asarray(theta_vec, dtype=float))
    return model_interface.model.data_df[PRED_COL].to_numpy()

def pool_resid(theta_list, label):
    """Forward-run each theta, return pooled (obs - pred) residuals at real-obs timesteps."""
    out = []
    for k, th in enumerate(theta_list):
        pred = forward_pred(th)[st:et][obs_mask]
        out.append(obs_vals - pred)
        print(f"  {label} forward run {k+1}/{len(theta_list)} done", end='\r')
    print()
    r = np.concatenate(out)
    return r[np.isfinite(r)]

rng = np.random.default_rng(RUN_SEED)

# --- assemble posterior theta draws ---
if USE_WITHIN_ITER_SPREAD:
    sigma_mask = np.array(['sigma' in c for c in theta_cols])     # keep these >= eps (variances)
    post_theta = []
    for _ in range(K_FWD):
        l = int(rng.choice(post_rows_theta))
        draw = rng.normal(theta_df.iloc[l].to_numpy(float), np.abs(theta_std_df.iloc[l].to_numpy(float)))
        draw[sigma_mask] = np.clip(draw[sigma_mask], 1e-6, None)
        post_theta.append(draw)
else:
    post_theta = [theta_df.iloc[l].to_numpy(float) for l in post_rows_theta]

# --- posterior forward residual pool ---
post_fwd_resid = pool_resid(post_theta, "posterior")

# --- prior forward residual pool ---
if REGENERATE_PRIOR_FORWARD:
    # draw theta from the interface's prior distributions (same machinery as update_theta(None))
    prior_theta = [np.array([model_interface.dist_model[k].rvs() for k in theta_cols]) for _ in range(M_PRIOR)]
    prior_fwd_resid = pool_resid(prior_theta, "prior")
else:
    prior_fwd_resid = prior_resid   # reuse the saved-file forward prior from the cell above

# --- KLD: forward posterior vs forward prior ---
kld_fwd_gauss = kl_gaussian(post_fwd_resid, prior_fwd_resid)
kld_fwd_kde   = kl_kde(post_fwd_resid, prior_fwd_resid)
print(f"prior (forward):     n={prior_fwd_resid.size:6d}  mean={prior_fwd_resid.mean():+.4f}  std={prior_fwd_resid.std(ddof=1):.4f}")
print(f"posterior (forward): n={post_fwd_resid.size:6d}  mean={post_fwd_resid.mean():+.4f}  std={post_fwd_resid.std(ddof=1):.4f}")
print(f"D_KL(forward posterior || forward prior)  Gaussian = {kld_fwd_gauss:.4f} nats")
print(f"D_KL(forward posterior || forward prior)  KDE      = {kld_fwd_kde:.4f} nats")

# --- overlay plot ---
plt.figure(figsize=(8, 5))
lo = min(prior_fwd_resid.min(), post_fwd_resid.min())
hi = max(prior_fwd_resid.max(), post_fwd_resid.max())
bins = np.linspace(lo, hi, 80)
plt.hist(prior_fwd_resid, bins=bins, density=True, alpha=0.45, color='grey',     label='prior error (forward)')
plt.hist(post_fwd_resid,  bins=bins, density=True, alpha=0.45, color='seagreen', label='posterior error (forward)')
xx = np.linspace(lo, hi, 400)
plt.plot(xx, gaussian_kde(prior_fwd_resid)(xx), color='black',     lw=1)
plt.plot(xx, gaussian_kde(post_fwd_resid)(xx),  color='darkgreen', lw=1)
plt.axvline(0, color='red', ls='--', lw=1)
plt.xlabel('Residual  (obs - pred),  per mil')
plt.ylabel('Density')
plt.title(f'Forward error distributions   |   D_KL(post||prior) = {kld_fwd_kde:.3f} nats')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
# %%
