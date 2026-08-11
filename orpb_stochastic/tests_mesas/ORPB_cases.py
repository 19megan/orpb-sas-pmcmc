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
                 'prior_dis': 'normal',
                 'prior_params': [0.51, 0.249],
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
                    'a': {'prior_dis': 'normal', # this only works because scale is a string and uses prior dists from scale params
                          'prior_params': [3.0, 1.02], #prior a
                          'is_nonnegative': True},
                    'scale': {'prior_dis': 'normal',
                              'prior_params': [1800, 200],####CHANGE THIS**************
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
                 'prior_dis': 'normal',
                 'prior_params': [52.5, 24.23],
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

import pandas as pd
tag='D_std' #NOTE: must change this depending on run
sTmT = pd.read_csv(f'/Users/simon/Desktop/ORPB_resolution_datasets/sT_mT_init_{tag}.csv')
sT_init = sTmT['sT_init'].values
mT_init = sTmT['mT_init'].values

#c_old=-7.6
solute_parameters = {'precip 18O': {'C_old': -7.28, 'observations': 'ORPB 18O', 'mT_init': mT_init}
                     }

options = {'influx': 'influx (mm/hr)', 'dt': 1, 'verbose': True, 'n_substeps': 1, 'record_state': True, 'sT_init': sT_init}#, 'max_age': 2160} #8760/12 hours ~1month or up to 4380 for 6months #set max age to reduce memory errors

obs_uncertainty = {
    # sig_u
    'sigma observed C in': { #precip 18O': {
        'prior_dis': 'normal',
        'prior_params': [0.08, 0.01], #[0.01, 0.01],#[-6.706, 3.293], # mean, std
        'is_nonnegative': True #False # sig_u becomes the scale for normal dist so can't be neg
    },
    'sigma filled C in': { #precip 18O':{
        'prior_dis': 'normal',
        'prior_params': [0.08, 1.17], #[0.02, 0.02], #[-6.706, 3.293], # mean, std
        'is_nonnegative': True #False
    },
    'sigma C out': { #ORPB 18O':{
        'prior_dis': 'normal',
        'prior_params':[0.08, 0.01], #[0.01, .01], #[5.0, 5.0], #[-7.325, 0.581], # mean, std
        'is_nonnegative': True #False
    }
}                  

scale_parameters = {
    'lambda':{
        'prior_dis': 'normal',
        'prior_params': [1.005, 0.51], #[.77, .2],#[.5, .2], #[11.54, 1.5],#[0.0099, 1.5], # mean, std
        'is_nonnegative': True
    },
    'S_c':{
        'prior_dis': 'normal',
        'prior_params': [-2011.85, 270.79], #[-295.02, 30],# [-1649.36, 10],#[-131, 10], #[-320.03, 10.0], # mean, std
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