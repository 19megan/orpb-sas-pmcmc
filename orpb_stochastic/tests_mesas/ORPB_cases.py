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

# Spinup: the real year of water fluxes before each calibration window, with NO isotope data.
# Its only job is to build a realistic age-ranked storage before the window starts, so pQ is
# right from the first observation. Isotopes in the spinup would give a short test (e.g. 3M)
# a year of input data it is not supposed to have, so they are blanked, and water that
# entered during the spinup takes C_old in the convolution (_precompute_convolution in
# ORPB_mesas_interface.py). C_old therefore means "water already in the catchment when the
# window starts, or older than max_age" -- the same for every run length and resolution.
# max_age is set to the spinup length (one year in timesteps).

def prepend_spinup(data_df, start_date, end_date, isotope_cols=('ORPB 18O', 'precip 18O')):
    """Return the calibration window with the year of data before it prepended as spinup.

    The spinup rows are the data_df rows in [start_date - 1 year, start_date), with every
    column in isotope_cols set to NaN: fluxes only, never isotopes.

    Returns:
        (pd.DataFrame, int): the combined dataframe (with a boolean 'is_spinup' column) and
        the number of spinup timesteps, which is the max_age to use.
    """
    start = pd.Timestamp(start_date)
    window = data_df.loc[start: pd.Timestamp(end_date)]
    if window.empty:
        raise ValueError(f"No data between {start_date} and {end_date}")
    spinup_start = start - pd.DateOffset(years=1)
    if data_df.index[0] > spinup_start:
        raise ValueError(f"Calibration window starting {start.date()} needs a year of fluxes "
                         f"before it, but the data start {data_df.index[0].date()}; start on or "
                         f"after {(data_df.index[0] + pd.DateOffset(years=1)).date()}")

    spinup = data_df.loc[(data_df.index >= spinup_start) & (data_df.index < start)].copy()
    spinup[list(isotope_cols)] = np.nan
    spinup['is_spinup'] = True

    combined = pd.concat([spinup, window.assign(is_spinup=False)])
    return combined, len(spinup)

# Held-out validation data, identical for every run in both scenarios. Stream isotopes in
# HOLDOUT_MONTHS are removed from every calibration dataset (mask_holdout), so no run, whatever
# its window, ever fits them: one month per season and per year, covering the drought onset
# (2016-10), the lowest flows (2017-01) and the wettest month (2018-07). 2014 Jul-Dec needs no
# mask: no window starts before 2015 and spinup isotopes are blanked.
HOLDOUT_MONTHS = ['2015-04', '2016-10', '2017-01', '2018-07']
VALIDATION_PERIODS = {'2014 Jul-Dec': ('2014-07-01', '2014-12-31'),
                      **{m: (m, m) for m in HOLDOUT_MONTHS}}

# Coarse stream values are means of the grab samples over the period ENDING at their stamp
# (resample labels right; monthly values sit at month end). A value stamped up to this many
# days after a held-out month still averages some of its grabs, so it is masked too.
OUTPUT_SPILL_DAYS = {'daily': 0, 'weekly': 6, 'biweekly': 13, 'monthly': 0}

def mask_holdout(df, output_resolution, obs_col='ORPB 18O', months=HOLDOUT_MONTHS):
    """Set obs_col to NaN for every stream value that contains a held-out month's samples."""
    spill = pd.Timedelta(days=OUTPUT_SPILL_DAYS[output_resolution])
    for m in months:
        month = pd.Period(m, freq='M')
        last = month.end_time.normalize() + spill
        df.loc[(df.index >= month.start_time) & (df.index <= last), obs_col] = np.nan
    return df

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

# Both are positive scale parameters, so they get log-uniform priors over the same
# wide range: 0.08 permil is analyzer precision (0.08) and 3.5 permil is std of precip 18O -
# any plausible structural error. Pinning them near analyzer precision is what collapsed the
# filter -- 0.08 is the ONLY error term in the observation model, so it must also absorb
# model structural error, and the best achievable fit is ~0.82 permil RMSE. At 0.08 the
# filter treats an 11-sigma misfit as noise: particle ESS falls to ~2 of 25, every chain
# degenerates equally, and the chain weights stop telling theta draws apart.
# One input sigma for every precipitation sample: each value is the composite for its whole
# sampling interval, so there is no separate "filled" uncertainty (removed 2026-10-21).
obs_uncertainty = {
    # sig_u
    'sigma C in': { #precip 18O': {
        'prior_dis': 'loguniform',
        'prior_params': [0.08, 3.5], # [lower, upper]
        'is_nonnegative': True # sig_u becomes the scale for normal dist so can't be neg
    },
    'sigma C out': { #ORPB 18O':{
        'prior_dis': 'loguniform',
        'prior_params': [0.08, 3.5], # [lower, upper]
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