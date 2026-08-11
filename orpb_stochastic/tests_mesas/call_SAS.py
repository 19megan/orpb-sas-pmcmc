# This script will be used to verify that pMCMC and SAS_check get the same predictions
# This will contain the SAS function that will be called by both scripts to run the SAS function
# Date: 02/11/2026

from copy import deepcopy
from mesas.sas.model import Model
from ORPB_cases import *
import matplotlib.pyplot as plt

def run_SAS(df, params):
    print(params)
    
    if 'qf_a' in params.keys():
        C_old = params['C_old']
        et_scale = params['et_scale']
        qf_a = params['qf_a']
        qf_scale = params['qf_scale']
        bf_a = params['bf_a']
        S_c=params['S_c']
        lamda=params['lambda']
        df['S_scale'] = lamda*(df['storage (mm)']-S_c)
        # sas_specs_storage_q_g_et_u['discharge (mm/hr)']['discharge (mm/hr) SAS function']['args']['a'] = a
        # sas_specs_storage_q_g_et_u['discharge (mm/hr)']['discharge (mm/hr) SAS function']['args']['scale'] = 'S_scale'
        # sas_specs_storage_q_g_et_u['ET (mm/hr)']['ET (mm/hr) SAS function']['args']['scale'] = et_scale
        # solute_parameters['precip 18O']['C_old'] = C_old

        sas_specs_storage_q_gg_et_u['discharge (mm/hr)']['qf_weight']['args']['a'] = qf_a
        sas_specs_storage_q_gg_et_u['discharge (mm/hr)']['qf_weight']['args']['scale'] = qf_scale
        sas_specs_storage_q_gg_et_u['discharge (mm/hr)']['bf1_weight']['args']['a'] = bf_a
        sas_specs_storage_q_gg_et_u['discharge (mm/hr)']['bf1_weight']['args']['scale'] = 'S_scale'
        sas_specs_storage_q_gg_et_u['ET (mm/hr)']['ET (mm/hr) SAS function']['args']['scale'] = et_scale
        solute_parameters['precip 18O']['C_old'] = C_old
        model = Model(df, config={'sas_specs': deepcopy(sas_specs_storage_q_gg_et_u), 'solute_parameters': deepcopy(solute_parameters), 'options': deepcopy(options)})
    else:
        C_old = params['C_old']
        et_scale = params['et_scale']
        qf_scale = params['qf_scale']
        bf_a = params['bf_a']
        S_c=params['S_c']
        lamda=params['lambda']
        df['S_scale'] = lamda*(df['storage (mm)']-S_c)

        sas_specs_storage_q_ug_et_u['discharge (mm/hr)']['qf_weight']['args']['scale'] = qf_scale
        sas_specs_storage_q_ug_et_u['discharge (mm/hr)']['bf1_weight']['args']['a'] = bf_a
        sas_specs_storage_q_ug_et_u['discharge (mm/hr)']['bf1_weight']['args']['scale'] = 'S_scale'
        sas_specs_storage_q_ug_et_u['ET (mm/hr)']['ET (mm/hr) SAS function']['args']['scale'] = et_scale
        solute_parameters['precip 18O']['C_old'] = C_old
        model = Model(df, config={'sas_specs': deepcopy(sas_specs_storage_q_ug_et_u), 'solute_parameters': deepcopy(solute_parameters), 'options': deepcopy(options)})
    
    model.run()

    # plt.figure(figsize=[12,4])
    # plt.plot(model.data_df.index, model.data_df['ORPB 18O'].backfill(), color='grey', label='Observed 18O outflow')

    # plt.plot(model.data_df.index, model.data_df['precip 18O --> discharge (mm/hr)'], color='orange', label='Predicted 18O outflow')
    # plt.legend()
    # plt.title('Isotope outflow at ORPB')
    # plt.show()

    return model


# for use of this with pMCMC script, replace
# ORPB_mesas_interface.py _init_sas_model function lines:
# "self.model = self.model_type(..)....self.model.run()"
# with
# "self.model = run_SAS(self.df, checkparams)"
# where checkparams is a dictionary with parameters for SAS model (which
# also need to be added within the function def)
# also add 'from call_SAS import run_SAS' at the top of the file