In the tests_mesas folder mesas compatability and integration with pMCMC is established:

**Files:**

mesas_cases.py : contains different cases of sas_specs, solute_parameters, options, obs_uncertainty, scale parameters, and combined theta dictionaries for mesas SAS model

mesas_dataset_preprocess.py : 

mesas_interface.py : modified version of model_interface.py in mesas.model which allows SAS via MESAS to be used in pMCMC

mesas_original.py : compares C_T calculated from convolution, new method, and SAS model

mesas_script_developing.py : similar to mesas_original.py to make sure her model matches mesas and convolution

mesas_script_debug.py : Actual implementation of pMCMC with mesas

ORPB_pmcmc_script.py : My implementation of pMCMC with mesas using ORPB dataset

output.npy : 

post_processing_varying.py :

post_processing.py : 

