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
                 # Uninformative over a physically plausible range, set from FORCING only
                 # (never from the isotope observations being fitted -- that would use the
                 # data twice). Quickflow samples event-scale storage; 200 mm is about twice
                 # the largest daily influx on record (101 mm/d). log-uniform because this is
                 # a positive scale spanning orders of magnitude: a linear uniform would put
                 # most of its mass in the top decade. prior_params is [lower, upper].
                 'prior_dis': 'loguniform',
                 'prior_params': [0.1, 200.0],
                 'is_nonnegative': True}
            },
         'bf1_weight':
             {'func':'gamma',
              'args': {
                  'a': 3.06, #4.433, #1.26,
                  'loc': 0.0,
                  'scale': 2100  # overwritten each iteration with M / a (see 'M' prior below)
                  },
              'priors': { #prior for multiple params
                    'a': {'prior_dis': 'uniform',
                          # shape only. a < 1 is the young-water-preference regime; the old
                          # [1.0, 5.0] excluded it outright, which is a real assumption
                          # (a >= 1 forces zero density at age 0, i.e. no young baseflow).
                          'prior_params': [0.1, 10.0], # [lower, upper]
                          'is_nonnegative': True},
                    # Parameterised by the MEAN age-ranked storage M = a * scale rather than
                    # by scale, because the physics constrains the product: bounding scale
                    # alone cannot express it while a roams over two orders of magnitude
                    # (a=10, scale=10000 would imply 100 m of water). The interface derives
                    # scale = M / a. Range: saprolite tens of metres thick (St Clair et al.
                    # 2015; Cleaves et al. 1970) at 21% specific yield (Nutter & Otton 1969)
                    # gives 2100 mm at 10 m to 10500 mm at 50 m; tracer mixes with total
                    # water-filled porosity (~35-45%), not just the drainable fraction, so
                    # the upper bound is ~50 m at Sy or ~25 m at total porosity.
                    'M': {'prior_dis': 'loguniform',
                          'prior_params': [10.0, 10000.0], # [lower, upper], mm
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
                 # ET draws from rooting-zone storage; 500 mm is generous for that, and
                 # Putnam's ~48 mm sits mid-range. log-uniform, as for the other scales.
                 'prior_dis': 'loguniform',
                 'prior_params': [0.1, 500.0], # [lower, upper]
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
                                    # C_old is the concentration of water older than max_age,
                                    # so it should sit near the long-term flux-weighted mean
                                    # precipitation d18O (-6.62). Bounded by the full observed
                                    # PRECIPITATION range [-18.92, 1.29], which is forcing, not
                                    # the stream observations being fitted -- the previous
                                    # [-10.78, 0.0025] looks like it came from the stream record.
                                    'prior': {
                                        'prior_dis': 'uniform',
                                        'prior_params': [-19.0, 2.0], # [lower, upper]
                                        'is_nonnegative': False}
                                    }
                     }

options = {'influx': 'influx (mm/hr)', 'dt': 1, 'verbose': True, 'n_substeps': 1, 'record_state': True, 'validate_inputs': False,
           'max_age': None} # set at run time to the spinup length returned by prepend_spinup (None = full timeseries)

# All three are positive scale parameters, so they get log-uniform priors over the same
# wide range: 0.01 permil is well below analyzer precision (0.08) and 5 permil is well above
# any plausible structural error. Pinning them near analyzer precision is what collapsed the
# filter -- 0.08 is the ONLY error term in the observation model, so it must also absorb
# model structural error, and the best achievable fit is ~0.82 permil RMSE. At 0.08 the
# filter treats an 11-sigma misfit as noise: particle ESS falls to ~2 of 25, every chain
# degenerates equally, and the chain weights stop telling theta draws apart.
obs_uncertainty = {
    # sig_u
    'sigma observed C in': { #precip 18O': {
        'prior_dis': 'loguniform',
        'prior_params': [0.01, 5.0], # [lower, upper]
        'is_nonnegative': True # sig_u becomes the scale for normal dist so can't be neg
    },
    'sigma filled C in': { #precip 18O':{
        'prior_dis': 'loguniform',
        'prior_params': [0.01, 5.0], # [lower, upper]
        'is_nonnegative': True
    },
    'sigma C out': { #ORPB 18O':{
        'prior_dis': 'loguniform',
        'prior_params': [0.01, 5.0], # [lower, upper]
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