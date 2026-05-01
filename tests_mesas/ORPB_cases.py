# This script will contain all of the test cases for sas_specs, solute parameters, options, obs_uncertainty, scale_parameters, and theta dictionaries for SAS model using pMCMC
# Created on: 11/10/2025

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
            {'func':'kumaraswamy',
             'args':{
                 'a': 1.0,
                 'b': 1.0,
                 'loc': 0.0,
                 'scale': 0.254,
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
                 'scale': 54.4, #66.502, #43.19
             }}
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
#c_old=-7.6
solute_parameters = {'precip 18O': {'C_old': -7.28, 'observations': 'ORPB 18O'}
                     }

options = {'influx': 'influx (mm/hr)', 'dt': 1, 'verbose': True, 'n_substeps': 1, 'record_state': True}#, 'max_age': 365} #set max age to reduce memory errors

obs_uncertainty = {
    # sig_u
    'sigma observed C in': { #precip 18O': {
        'prior_dis': 'normal',
        'prior_params': [0.01, 0.01],#[-6.706, 3.293], # mean, std
        'is_nonnegative': True #False # sig_u becomes the scale for normal dist so can't be neg
    },
    'sigma filled C in': { #precip 18O':{
        'prior_dis': 'normal',
        'prior_params': [0.02, 0.02], #[-6.706, 3.293], # mean, std
        'is_nonnegative': True #False
    },
    'sigma C out': { #ORPB 18O':{
        'prior_dis': 'normal',
        'prior_params':[0.01, .01], #[5.0, 5.0], #[-7.325, 0.581], # mean, std
        'is_nonnegative': True #False
    }
}                  

scale_parameters = {
    'lambda':{
        'prior_dis': 'normal',
        'prior_params': [1.21, 0.51], #[.77, .2],#[.5, .2], #[11.54, 1.5],#[0.0099, 1.5], # mean, std
        'is_nonnegative': True
    },
    'S_c':{
        'prior_dis': 'normal',
        'prior_params': [-1102.2, 166.6], #[-295.02, 30],# [-1649.36, 10],#[-131, 10], #[-320.03, 10.0], # mean, std
        'is_nonnegative': False
    }
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