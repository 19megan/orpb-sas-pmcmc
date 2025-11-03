# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("/Users/esthersida/Documents/Code/particle/SAS/PythonEquivalent/Data/ORPB_isotope_data.csv", parse_dates=True, index_col=0)
start_ind, end_ind = 0, int(len(df)/20)

data = df.iloc[start_ind:end_ind]

data["snowfall SWE (mm/hr)"][data["snowfall SWE (mm/hr)"] == 0] = np.nan

# %%
fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

ax1 = ax[0]
ax2 = ax[1]
ax3 = ax[2]
ax4 = ax[3]

J1 = ax1.bar(data.index, data["rainfall (mm/hr)"], label="Rainfall")
J2 = ax1.bar(data.index, data["snowfall SWE (mm/hr)"],color="red", label="Snowfall")

ax1.set_yscale("log")  
ax1.set_ylim(ax1.get_ylim()[::-1]) 
ax1.set_ylabel("Precipiration (mm/hr)")

ax2.plot(data.index, data["discharge (mm/hr)"], label="Streamflow")
ax2.plot(data.index, data["snowmelt (mm/hr)"], label="Snowmelt", color="red")
ax2.plot(data.index, data["baseflow 1 (mm/hr)"], label="Baseflow", color="green")


ax2.set_ylabel("Discharge (mm/hr)")
ax2.set_yscale("log")

ax3.plot(data.index, data["storage (mm)"], label="Storage")
ax3.set_ylabel("Storage (mm)")

ax4.plot(data.index, data["ET (mm/hr)"], label="C in")
ax4.set_ylabel("ET (mm/hr)")
# ax0p = ax[0].twinx()
# CJ = ax0p.scatter(
#     data.index[data["C in"] > 0],
#     data["C in"][data["C in"] > 0],
#     color="r",
#     marker=".",
#     label=r"$Observed\ C_J$",
#     s=8,
# )
# temp_ind = np.logical_and(data["is_obs_input_filled"].values, data["C in"].values > 0)
# CJ1 = ax0p.scatter(
#     data.index[temp_ind],
#     data["J 2H"][temp_ind],
#     color="k",
#     marker=".",
#     label=r"$Filled\ C_J$",
#     s=8,
# )
ax1.legend()
ax2.legend()
fig.tight_layout()


# %%
def mystep(x,y, ax=None, where='post', **kwargs):
    assert where in ['post', 'pre']
    x = np.array(x)
    y = np.array(y)
    if where=='post': y_slice = y[:-1]
    if where=='pre': y_slice = y[1:]
    X = np.c_[x[:-1],x[1:],x[1:]]
    Y = np.c_[y_slice, y_slice, np.zeros_like(x[:-1])*np.nan]
    if not ax: ax=plt.gca()
    return ax.plot(X.flatten(), Y.flatten(), **kwargs)

fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

ax1 = ax[0, 0]
ax2 = ax[0, 1]

ax3 = ax[1, 0]
ax4 = ax[1, 1]



# mystep(np.arange(0, len(data)), data["precip 2H"].values, ax=ax1, where="pre", label="Rainfall")
ax1.scatter(data.index, data["precip 2H"], label="Rainfall", marker=".")    
ax1.fill_between(data.index, data["precip 2H"] - 3 * data["precip 2H StDev"], data["precip 2H"] + 3 * data["precip 2H StDev"], alpha=0.5)

ax2.scatter(data.index, data["precip 18O"], label="18O", marker=".")
ax2.scatter(data.index, data["precip 17O"], label="17O", marker=".")

ax3.scatter(data.index, data["ORPB 2H"], label="Rainfall", marker=".")

ax4.scatter(data.index, data["ORPB 18O"], label="18O", marker=".")
ax4.scatter(data.index, data["ORPB 17O"], label="17O", marker=".")



# %%
