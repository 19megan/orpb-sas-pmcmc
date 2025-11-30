# This script will contain all of the test cases for sas_specs, solute parameters, options, obs_uncertainty, scale_parameters, and theta dictionaries for SAS model using pMCMC
# Created on: 11/10/2025

sas_specs_storage_q_ug_et_u = {
    'quickflow (mm/hr)':
        {'quickflow (mm/hr) SAS function':
            {'func':'kumaraswamy', #uniform dist
             'args': {
                 'a': 1.0,
                 'b': 1.0,
                 'loc': 0.0,
                 'scale': 0.254
                 }}
        }, #if >1 dict for 'quickflow (mm/hr)', then mesas looks for column named 'ORPB qf' and other dict key(s) -- >1 dict will allow for weighted SAS functions, these columns in df will be the weights [0,1]
    'baseflow 1 (mm/hr)':
        {'baseflow 1 (mm/hr) SAS function':
                {'func': 'gamma',
                 'args': { 'a': 1.26,
                           'loc': 0,
                           'scale':'storage (mm)' #2106.1584
                            }}
        },
    'ET (mm/hr)': 
        {'ET (mm/hr) SAS function':
            {'func':'kumaraswamy',
             'args':{
                 'a': 1.0,
                 'b': 1.0,
                 'loc': 0.0,
                 'scale': 43.19
             }}
        }
}
#C_old = -7.6 but I'm getting errors for a negative scale in param_dist so I'll change it
solute_parameters = {'precip 18O': {'C_old': 1, 'observations': 'ORPB 18O'}
                     }

options = {'influx': 'influx (mm/hr)', 'dt': 1, 'verbose': True, 'n_substeps': 1, 'record_state': True}#, 'max_age': 365} #set max age to reduce memory errors

obs_uncertainty = {
    # sig_u
    'sigma observed C in': { #precip 18O': {
        'prior_dis': 'normal',
        'prior_params': [0.05, 0.02],#[-6.706, 3.293], # mean, std
        'is_nonnegative': True #False # sig_u becomes the scale for normal dist so can't be neg
    },
    'sigma filled C in': { #precip 18O':{
        'prior_dis': 'normal',
        'prior_params': [1.0, 0.5], #[-6.706, 3.293], # mean, std
        'is_nonnegative': True #False
    },
    'sigma C out': { #ORPB 18O':{
        'prior_dis': 'normal',
        'prior_params': [5.0, 5.0], #[-7.325, 0.581], # mean, std
        'is_nonnegative': True #False
    }
}                  

scale_parameters = {
    'lambda':{
        'prior_dis': 'normal',
        'prior_params': [-103.0, 10.3], # mean, std
        'is_nonnegative': False
    },
    'S_c':{
        'prior_dis': 'normal',
        'prior_params': [48.0, 4.8], # mean, std
        'is_nonnegative': True
    }
}


theta_storage_q_ug_et_u = {
    'sas_specs': sas_specs_storage_q_ug_et_u,
    'solute_parameters': solute_parameters,
    'options': options,
    'obs_uncertainty': obs_uncertainty,
    'scale_parameters': scale_parameters
}