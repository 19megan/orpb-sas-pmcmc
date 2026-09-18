# This script will contain all of the test cases for sas_specs, solute parameters, options, obs_uncertainty, scale_parameters, and theta dictionaries for SAS model using pMCMC
# Created on: 11/10/2025

sas_specs_storage_q_ug_et_u_cp = { #cp is for constant params (no string passed to param values)
    'discharge (mm/hr)':
        {'qf_weight': # this column in df will be the weight for the quickflow SAS function, and 1 - this column will be the weight for the baseflow 1 SAS function:
            {'func':'kumaraswamy', #only scale gets estimated, a,b,loc are fixed at 1,1,0
             'args':{
                 'a': 1.0,
                 'b': 1.0,
                 'loc': 0.0,
                 'scale': 0.254,
             },
             'prior':{ #prior for scale parameter
                 # uninformative: flat over the plausible range instead of a normal
                 # centred on the range midpoint. prior_params is [lower, upper].
                 'prior_dis': 'uniform',
                 'prior_params': [0.025, 1.0],
                 'is_nonnegative': True}
            },
         'bf1_weight':
             {'func':'gamma',
              'args': {
                  'a': 3.06, #4.433, #1.26,
                  'loc': 0.0,
                  'scale': 2100
                  },
              'priors': { #prior for multiple params
                    'a': {'prior_dis': 'uniform',
                          'prior_params': [1.0, 5.0], # [lower, upper]
                          'is_nonnegative': True},
                    'scale': {'prior_dis': 'uniform',
                              # bounds are a guess -- watch for the trace piling up
                              # against 1500 or 2200 and widen if it does
                              'prior_params': [1500.0, 2200.0], # [lower, upper]
                              'is_nonnegative': True}
              },
              'nsegment': 200}
        },        
    'ET (mm/hr)': 
        {'ET (mm/hr) SAS function':
            {'func':'kumaraswamy',
             'args':{
                 'a': 1.0,
                 'b': 1.0,
                 'loc': 0.0,
                 'scale': 54.4, #66.502, #43.19
             },
            'prior': {
                 'prior_dis': 'uniform',
                 'prior_params': [5.0, 100.0], # [lower, upper]
                 'is_nonnegative': True}
            }
        }
}

sas_specs_storage_q_gg_et_u = {
    'discharge (mm/hr)':
        {'qf_weight': # this column in df will be the weight for the quickflow SAS function, and 1 - this column will be the weight for the baseflow 1 SAS function:
            {'func':'gamma',
             'args': {
                 'a': .566, #0.178, #.5,
                 'loc': 0.0,
                 'scale': 0.59, #0.910, #.5
                 }},
         'bf1_weight':
             {'func':'gamma',
              'args': {
                  'a': 3.06, #4.433, #1.26,
                  'loc': 0.0,
                  'scale': 'S_scale'
                  }}
        },        
    'ET (mm/hr)': 
        {'ET (mm/hr) SAS function':
            {'func':'kumaraswamy',
             'args':{
                 'a': 1.0,
                 'b': 1.0,
                 'loc': 0.0,
                 'scale': 54.4, #66.502, #43.19 #just estimating scale when it's not a string
             }}
        }
}


sas_specs_storage_q_ug_et_u = {
    'discharge (mm/hr)':
        {'qf_weight': # this column in df will be the weight for the quickflow SAS function, and 1 - this column will be the weight for the baseflow 1 SAS function:
            {'func':'kumaraswamy', #only scale gets estimated, a,b,loc are fixed at 1,1,0
             'args':{
                 'a': 1.0,
                 'b': 1.0,
                 'loc': 0.0,
                 'scale': 0.254,
             },
             'prior':{ #prior for scale parameter
                 'prior_dis': 'normal',
                 'prior_params': [0.51, 0.249],
                 'is_nonnegative': True}
            },
         'bf1_weight':
             {'func':'gamma',
              'args': {
                  'a': 3.06, #4.433, #1.26,
                  'loc': 0.0,
                  'scale': 'S_scale'
                  },
              'prior': { #prior for a
                    'prior_dis': 'normal', # this only works because scale is a string and uses prior dists from scale params
                    'prior_params': [3.0, 1.02], #prior a
                    'is_nonnegative': True},
              'nsegment': 200}
        },        
    'ET (mm/hr)': 
        {'ET (mm/hr) SAS function':
            {'func':'kumaraswamy',
             'args':{
                 'a': 1.0,
                 'b': 1.0,
                 'loc': 0.0,
                 'scale': 54.4, #66.502, #43.19
             },
            'prior': {
                 'prior_dis': 'normal',
                 'prior_params': [52.5, 24.23],
                 'is_nonnegative': True}
            }
        }
}

sas_specs_storage_q_u_et_u = {
    'discharge (mm/hr)':
        {'discharge (mm/hr) SAS function':
            {'func':'kumaraswamy', #uniform dist
                'args': {
                    'a': 1.0,
                    'b': 1.0,
                    'loc': 0.0,
                    'scale': 'storage (mm)'
                    }}
        },
    'ET (mm/hr)':
        {'ET (mm/hr) SAS function':
            {'func':'kumaraswamy',
             'args':{
                 'a': 1.0,
                 'b': 1.0,
                 'loc': 0.0,
                 'scale': 43.19 #just estimating scale when it's not a string
             }}
        }

}

sas_specs_storage_q_g_et_u = {
    'discharge (mm/hr)':
        {'discharge (mm/hr) SAS function':
            {'func':'gamma',
                'args': {
                    'a': 1.26,
                    'loc': 0.0,
                    'scale': 'S_scale' #'storage (mm)'
                    }}
        },
    'ET (mm/hr)':
        {'ET (mm/hr) SAS function':
            {'func':'kumaraswamy',
             'args':{
                 'a': 1.0,
                 'b': 1.0,
                 'loc': 0.0,
                 'scale': 43.19 #just estimating scale when it's not a string
             }}
        }

}

import numpy as np
import pandas as pd

# Spinup: one year of forcing prepended to every run, excluded from the likelihood.
# max_age is set to the spinup length (one year in timesteps at any resolution), so every
# observation sees a full year of resolved input history and C_old always means "water
# older than one year" -- comparable across runs of different lengths. Whole years matter:
# truncating mid-season leaves a seasonal bias that C_old cannot absorb.
SPINUP_YEAR = '2014'

def prepend_spinup(data_df, start_date, end_date, spinup_year=SPINUP_YEAR, obs_col='ORPB 18O'):
    """Return the calibration window with one year of spinup forcing prepended.

    The spinup rows are the spinup_year rows of data_df, re-dated to end immediately before
    start_date, with obs_col set to NaN so they are never evaluated in the likelihood.

    Returns:
        (pd.DataFrame, int): the combined dataframe (with a boolean 'is_spinup' column) and
        the number of spinup timesteps, which is the max_age to use.
    """
    window = data_df.loc[pd.Timestamp(start_date): pd.Timestamp(end_date)]
    spinup = data_df.loc[spinup_year].copy()
    if window.empty:
        raise ValueError(f"No data between {start_date} and {end_date}")
    if window.index[0] <= spinup.index[-1]:
        raise ValueError(f"Calibration window starting {window.index[0].date()} overlaps the "
                         f"{spinup_year} spinup year; start after {spinup.index[-1].date()}")

    # re-date so the spinup ends one timestep before the window, at the data's own frequency
    freq = pd.infer_freq(spinup.index)
    if freq is not None:
        spinup.index = pd.date_range(end=window.index[0] - pd.tseries.frequencies.to_offset(freq),
                                     periods=len(spinup), freq=freq)
    else:  # irregular index: shift by a constant so the last spinup step sits one step before the window
        step = spinup.index[-1] - spinup.index[-2]
        spinup.index = spinup.index + (window.index[0] - step - spinup.index[-1])
    spinup[obs_col] = np.nan
    spinup['is_spinup'] = True

    combined = pd.concat([spinup, window.assign(is_spinup=False)])
    return combined, len(spinup)

#c_old=-7.6
solute_parameters = {'precip 18O': {'C_old': -7.28, # placeholder only; replaced each iteration by the sampled value
                                    'observations': 'ORPB 18O',
                                    # uninformative prior spanning the observed d18O record.
                                    # This is by far the widest of the five ranges -- expect the
                                    # lowest ESS here, and check theta_std against 3.113
                                    # (= range/sqrt(12)) to see whether the data constrained it.
                                    'prior': {
                                        'prior_dis': 'uniform',
                                        'prior_params': [-10.78, 0.0025], # [lower, upper]
                                        'is_nonnegative': False}
                                    }
                     }

options = {'influx': 'influx (mm/hr)', 'dt': 1, 'verbose': True, 'n_substeps': 1, 'record_state': True, 'validate_inputs': False,
           'max_age': None} # set at run time to the spinup length returned by prepend_spinup (None = full timeseries)

obs_uncertainty = {
    # sig_u
    'sigma observed C in': { #precip 18O': {
        'prior_dis': 'normal',
        'prior_params': [0.08, 0.01], # mean, std
        'is_nonnegative': True # sig_u becomes the scale for normal dist so can't be neg
    },
    'sigma filled C in': { #precip 18O':{
        'prior_dis': 'normal',
        'prior_params': [0.08, 0.97], # mean, std
        'is_nonnegative': True
    },
    'sigma C out': { #ORPB 18O':{
        'prior_dis': 'normal',
        'prior_params':[0.08, 0.01], # mean, std
        'is_nonnegative': True
    }
}                  

scale_parameters = {
    'lambda':{
        'prior_dis': 'normal',
        'prior_params': [1.005, 0.51], # mean, std
        'is_nonnegative': True
    },
    'S_c':{
        'prior_dis': 'normal',
        'prior_params': [-2011.85, 270.79], # mean, std
        'is_nonnegative': False
    }
}

theta_storage_q_ug_et_u_cp = {
    'sas_specs': sas_specs_storage_q_ug_et_u_cp,
    'solute_parameters': solute_parameters,
    'options': options,
    'obs_uncertainty': obs_uncertainty
}

theta_storage_q_gg_et_u = {
    'sas_specs': sas_specs_storage_q_gg_et_u,
    'solute_parameters': solute_parameters,
    'options': options,
    'obs_uncertainty': obs_uncertainty,
    'scale_parameters': scale_parameters
}

theta_storage_q_ug_et_u = {
    'sas_specs': sas_specs_storage_q_ug_et_u,
    'solute_parameters': solute_parameters,
    'options': options,
    'obs_uncertainty': obs_uncertainty,
    'scale_parameters': scale_parameters
}

theta_storage_q_u_et_u = {
    'sas_specs': sas_specs_storage_q_u_et_u,
    'solute_parameters': solute_parameters,
    'options': options,
    'obs_uncertainty': obs_uncertainty,
    'scale_parameters': scale_parameters
}

theta_storage_q_g_et_u = {
    'sas_specs': sas_specs_storage_q_g_et_u,
    'solute_parameters': solute_parameters,
    'options': options,
    'obs_uncertainty': obs_uncertainty,
    'scale_parameters': scale_parameters
}