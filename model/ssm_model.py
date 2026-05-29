# %%
from typing import Optional
from tqdm import tqdm
from model.model_interface import ModelInterface
from model.utils_chain import Chain
import numpy as np
import multiprocessing as mp
from functions.utils import _inverse_pmf
from copy import deepcopy

# %%
# Workers for parallel particle Gibbs. Each worker runs one chain's particle
# filter on a separate process and returns the log-weight, the trajectories
# the parent needs, AND the resulting Chain.state.
#
# Returning state is required because the parent's `chains[d]` never sees the
# worker-side mutation (the worker mutates its own pickled copy, which is then
# discarded). The next iteration's AS call needs the previous run's
# state.R/W/X/A/Y as its reference trajectory (see utils_chain.py line 138),
# so the parent threads state[d] back into the next worker call.
def worker_function_SIR(chain_d, theta_new):
    chain_d.model_interface.update_theta(theta_new)
    chain_d.run_particle_filter_SIR()

    model_W = chain_d.state.W
    B = chain_d._find_traj(chain_d.state.A, model_W, max=True)
    traj_X = chain_d._get_X_traj(chain_d.state.X, B)

    state_W = chain_d.model_interface.transition_model_probability(traj_X)
    WW = np.log(state_W) + np.log(model_W.max())

    R_traj = chain_d._get_R_traj(chain_d.state.R, B)
    Y_traj = chain_d._get_Y_traj(chain_d.state.Y, B)
    return WW, B, R_traj, traj_X, Y_traj, chain_d.state


def worker_function_AS(chain_d, theta_new, prev_state):
    # Restore the chain's filter state from the previous iteration before AS
    # reads it. Without this the chain comes in fresh from __init__ and the
    # AS run crashes on `self.state.R`.
    chain_d.state = prev_state

    chain_d.model_interface.update_theta(theta_new)
    chain_d.run_particle_filter_AS()

    model_W = chain_d.state.W
    B = chain_d._find_traj(chain_d.state.A, model_W, max=True)
    traj_X = chain_d._get_X_traj(chain_d.state.X, B)

    state_W = chain_d.model_interface.transition_model_probability(traj_X)
    WW = np.log(state_W) + np.log(model_W.max())

    R_traj = chain_d._get_R_traj(chain_d.state.R, B)
    Y_traj = chain_d._get_Y_traj(chain_d.state.Y, B)
    return WW, B, R_traj, traj_X, Y_traj, chain_d.state

#%%
class SSModel:
    def __init__(
        self,
        model_interface: ModelInterface,
        num_parameter_samples: Optional[int] = 15,
        len_parameter_MCMC: Optional[int] = 20
    ) -> None:      
        """Initialize the SSM model taking essential inputs

        Args:
            model_interface (ModelInterface): *initialized* model interface
            num_parameter_samples (int): number of parameter samples, D
            len_parameter_MCMC (int): length of parameter MCMC chain, L
            fast_convergence_phase_length (int): length of fast convergence phase

        """
        # set up input metrics
        self.D = num_parameter_samples
        self.L = len_parameter_MCMC
        
        # get info from model_interface
        self._num_theta_to_estimate = model_interface._num_theta_to_estimate
        self._theta_to_estimate = model_interface._theta_to_estimate
        self.T = model_interface.T
        self.N = model_interface.N
        self.K = model_interface.K
        self.is_MAP_MCMC = model_interface.config["use_MAP_MCMC"]
        self.update_theta_dist = model_interface.config["update_theta_dist"]

        # pass model structure for each chain
        self.models_for_each_chain = [deepcopy(model_interface) for d in range(self.D)]
        # store necessary models
        self.dist_model = model_interface.dist_model

        # initialize record for parameter and input
        self.theta_record = np.zeros(
            (self.L+1, self._num_theta_to_estimate)
            )
        self.theta_std = np.zeros(
            (self.L+1, self._num_theta_to_estimate)
            )
        
        # TODO: assume one input for now
        self.input_record = np.zeros(
            (self.L+1, self.T)
            )

        self.state_record = np.zeros(
            (self.L+1, self.T)
            )

        self.output_record = np.zeros(
            (self.L+1, self.T)
            )

        # MLE SAS model state, populated at end of run_particle_Gibbs[_parallel]
        self.pQ_mle = None
        self.sT_mle = None
        self.ST_mle = None


    def _save_mle_sas_state(self) -> None:
        """Re-run the SAS model with the final (MLE) theta and store pQ/sT/ST.

        Why: in the parallel path the parent's chain copies never see the
        worker's update_theta, so we can't read pQ off any in-memory chain.
        Re-running once with theta_modeled[-1] is cheap relative to MCMC and
        gives a single consistent MLE for both serial and parallel paths.
        """
        mle_interface = self.models_for_each_chain[0]
        # Use weighted posterior means, not the hybrid in theta_modeled
        mle_interface.update_theta(self.theta_record[-1, :])
        self.pQ_mle = {flux: mle_interface.model.get_pQ(flux)
                       for flux in mle_interface.model.fluxorder}
        self.sT_mle = mle_interface.model.get_sT()
        self.ST_mle = mle_interface.model.get_ST()


    def _sample_theta_init(self) -> np.ndarray:
        """Sample theta from given initial prior distribution

        Returns:
            np.ndarray: sampled new theta candidates
        """
        theta_new = np.zeros(self._num_theta_to_estimate)
        for i, key in enumerate(self._theta_to_estimate):
             theta_new[i] = self.dist_model[key].rvs()
        return theta_new
    
    
    def run_particle_Gibbs(self) -> None:
        """ Run particle Gibbs for parameter estimation 
        """
        # initialize likelihood, theta storage space, and chains
        WW = np.zeros(self.D)
        BB = np.zeros((self.D, self.K)).astype(int)

        theta_new = np.zeros((self.D, self._num_theta_to_estimate)) 
        chains = [Chain(model_interface=self.models_for_each_chain[d]) for d in range(self.D)]

        # draw D theta candidates for each chain, and run sMC
        for d in range(self.D):
            theta_new[d,:] = self._sample_theta_init()   
            chains[d].model_interface.update_theta(theta_new[d,:])  
            chains[d].run_particle_filter_SIR()
            chain_d = chains[d]

            model_W = chain_d.state.W
            B = chain_d._find_traj(chain_d.state.A, model_W, max=True)
            traj_X = chains[d]._get_X_traj(chain_d.state.X, B)
            BB[d,:] = B
            state_W = chain_d.model_interface.transition_model_probability(traj_X)
            WW[d] = np.log(state_W) + np.log(model_W.max()) #log on both terms
        # find optimal parameters
        W_theta = np.exp((WW - WW.max()) )#/WW.max()) #standard log-sum-exp, corrects weighted statistics - subtraction for stability, exp to get raw weights
        W_theta = W_theta / W_theta.sum() #normalize sum to 1

        save_std = np.zeros(self._num_theta_to_estimate)
        for p, key in enumerate(self._theta_to_estimate):
            theta_dist_mean = (theta_new[:,p]*W_theta).sum()
            theta_dist_std = np.sqrt(sum((theta_new[:,p] - theta_dist_mean)**2 * W_theta)/(self.D-1.))
            save_std[p] = theta_dist_std

            # find optimal trajectory of each chain
            self.theta_record[0,p] = theta_dist_mean
            self.theta_std[0,p] = theta_dist_std


        if self.is_MAP_MCMC:
            ind_best_param = np.argmax(W_theta)
        else:

            ind_best_param = _inverse_pmf(theta_new[:,-1], W_theta, num=1)[0]

        best_model = chains[ind_best_param]
        B = BB[ind_best_param,:] 
        self.input_record[0,:] = best_model._get_R_traj(best_model.state.R, B)
        self.state_record[0,:] = best_model._get_X_traj(best_model.state.X, B)
        self.output_record[0,:] = best_model._get_Y_traj(best_model.state.Y, B)

        if self.update_theta_dist:
            best_theta = {}
            for p, key in enumerate(self._theta_to_estimate):
                best_theta[key] = [self.theta_record[0,p], save_std[p]]

            for d in range(self.D):
                chains[d].model_interface._set_parameter_distribution(update=True, theta_new=best_theta)


        # for each MCMC iteration
        for l in tqdm(range(self.L)): # progress bar for long simulations
            # for each theta
            for p, key in enumerate(self._theta_to_estimate):
                theta_new = self._update_theta_at_p(
                        p=p,
                        l=l,
                        key=key
                    )

                # update chains
                for d in range(self.D):
                    # initialize a chain using this theta
                    chains[d].model_interface.update_theta(theta_new[d,:])
                    chains[d].run_particle_filter_AS()   
                    model_W = chains[d].state.W
                    B = chains[d]._find_traj(chains[d].state.A, model_W, max=True)
                    traj_X = chains[d]._get_X_traj(chains[d].state.X, B)
                    BB[d,:] = B
                    state_W = chains[d].model_interface.transition_model_probability(traj_X)
                    WW[d] = np.log(state_W) + np.log(model_W.max()) #log on both terms
                #record values over D chains and evolution of theta distributions
                W_theta = np.exp((WW - WW.max()) )#/WW.max()) #standard log-sum-exp, corrects weighted statistics
                W_theta = W_theta / W_theta.sum()
                theta_dist_mean = (theta_new[:,p]*W_theta).sum()
                theta_dist_std = np.sqrt(sum((theta_new[:,p] - theta_dist_mean)**2 * W_theta)/(self.D-1.))
                save_std[p] = theta_dist_std


                # find optimal trajectory of each chain
                self.theta_record[l+1,p] = theta_dist_mean 
                self.theta_std[l+1,p] = theta_dist_std

            if self.is_MAP_MCMC:
                ind_best_param = np.argmax(W_theta) # max likelihood of p(y|Y)
            else:
                ind_best_param = _inverse_pmf(theta_new[:,p],W_theta, num=1)[0]

            best_model = chains[ind_best_param]
            B = BB[ind_best_param,:]
            self.input_record[l+1,:] = best_model._get_R_traj(best_model.state.R, B)
            self.state_record[l+1,:] = best_model._get_X_traj(best_model.state.X, B)
            self.output_record[l+1,:] = best_model._get_Y_traj(best_model.state.Y, B)

            if self.update_theta_dist:
                best_theta = {}
                for p, key in enumerate(self._theta_to_estimate):
                    best_theta[key] = [self.theta_record[l+1,p], save_std[p]]

                for d in range(self.D):
                    chains[d].model_interface._set_parameter_distribution(update=True, theta_new=best_theta)

        self._save_mle_sas_state()


    def run_particle_Gibbs_parallel(self, num_cores: Optional[int] = None) -> None:
        """Parallel particle Gibbs. Mirrors run_particle_Gibbs but distributes
        the D chains across processes. Each MCMC iteration's D particle-filter
        evaluations run concurrently in a persistent process pool.

        Args:
            num_cores (int, optional): worker count. Defaults to min(D, cpu_count()).
        """
        if num_cores is None:
            num_cores = min(self.D, mp.cpu_count())

        # initialize likelihood, theta storage space, and chains
        WW = np.zeros(self.D)
        BB = np.zeros((self.D, self.K)).astype(int)
        R_trajs = np.zeros((self.D, self.T))
        X_trajs = np.zeros((self.D, self.T))
        Y_trajs = np.zeros((self.D, self.T))

        theta_new = np.zeros((self.D, self._num_theta_to_estimate))
        chains = [Chain(model_interface=self.models_for_each_chain[d]) for d in range(self.D)]
        # Per-chain Chain.state objects threaded across iterations. Populated
        # by the SIR pass below; each AS call consumes the previous step's
        # state and returns the new one.
        chain_states = [None] * self.D

        for d in range(self.D):
            theta_new[d, :] = self._sample_theta_init()

        # Persistent pool reused across all MCMC iterations to avoid the cost
        # of spawning Python workers each time.
        with mp.Pool(processes=num_cores) as pool:
            # Initial SIR pass for each of D theta candidates
            args_list = [(chains[d], theta_new[d, :]) for d in range(self.D)]
            results = pool.starmap(worker_function_SIR, args_list)
            for d in range(self.D):
                (WW[d], BB[d, :], R_trajs[d], X_trajs[d], Y_trajs[d],
                 chain_states[d]) = results[d]

            # weight, mean/std for each theta dimension (matches serial)
            W_theta = np.exp(WW - WW.max())
            W_theta = W_theta / W_theta.sum()

            save_std = np.zeros(self._num_theta_to_estimate)
            for p, key in enumerate(self._theta_to_estimate):
                theta_dist_mean = (theta_new[:, p] * W_theta).sum()
                theta_dist_std = np.sqrt(sum((theta_new[:, p] - theta_dist_mean) ** 2 * W_theta) / (self.D - 1.))
                save_std[p] = theta_dist_std
                self.theta_record[0, p] = theta_dist_mean
                self.theta_std[0, p] = theta_dist_std

            if self.is_MAP_MCMC:
                ind_best_param = np.argmax(W_theta)
            else:
                ind_best_param = _inverse_pmf(theta_new[:, -1], W_theta, num=1)[0]

            self.input_record[0, :] = R_trajs[ind_best_param]
            self.state_record[0, :] = X_trajs[ind_best_param]
            self.output_record[0, :] = Y_trajs[ind_best_param]

            if self.update_theta_dist:
                best_theta = {key: [self.theta_record[0, p], save_std[p]]
                              for p, key in enumerate(self._theta_to_estimate)}
                for d in range(self.D):
                    chains[d].model_interface._set_parameter_distribution(update=True, theta_new=best_theta)

            # MCMC iterations
            for l in tqdm(range(self.L)):
                for p, key in enumerate(self._theta_to_estimate):
                    theta_new = self._update_theta_at_p(p=p, l=l, key=key)

                    args_list = [(chains[d], theta_new[d, :], chain_states[d])
                                 for d in range(self.D)]
                    results = pool.starmap(worker_function_AS, args_list)
                    for d in range(self.D):
                        (WW[d], BB[d, :], R_trajs[d], X_trajs[d], Y_trajs[d],
                         chain_states[d]) = results[d]

                    W_theta = np.exp(WW - WW.max())
                    W_theta = W_theta / W_theta.sum()
                    theta_dist_mean = (theta_new[:, p] * W_theta).sum()
                    theta_dist_std = np.sqrt(sum((theta_new[:, p] - theta_dist_mean) ** 2 * W_theta) / (self.D - 1.))
                    save_std[p] = theta_dist_std

                    self.theta_record[l + 1, p] = theta_dist_mean
                    self.theta_std[l + 1, p] = theta_dist_std

                if self.is_MAP_MCMC:
                    ind_best_param = np.argmax(W_theta)
                else:
                    ind_best_param = _inverse_pmf(theta_new[:, p], W_theta, num=1)[0]

                self.input_record[l + 1, :] = R_trajs[ind_best_param]
                self.state_record[l + 1, :] = X_trajs[ind_best_param]
                self.output_record[l + 1, :] = Y_trajs[ind_best_param]

                if self.update_theta_dist:
                    best_theta = {key: [self.theta_record[l + 1, p], save_std[p]]
                                  for p, key in enumerate(self._theta_to_estimate)}
                    for d in range(self.D):
                        chains[d].model_interface._set_parameter_distribution(update=True, theta_new=best_theta)

        self._save_mle_sas_state()

    def _update_theta_at_p(
            self,
            p: int,
            l: int,
            key: str,
        ) -> np.ndarray:
        """Generate random p-th theta

        Args:
            p (int): index of theta to update
            l (int): index of current MCMC chain
            key (str): key of theta to update
        
        Returns:
            np.ndarray: new theta candidates
        """
        theta_temp = np.ones((self.D,self._num_theta_to_estimate)) * self.theta_record[l,:]
        theta_temp[:,:p] = self.theta_record[l+1,:p]
        # add random noise to the p-th theta
        theta_temp[:,p] = self.dist_model[key].rvs(self.D)
        return theta_temp 
 
# %%
