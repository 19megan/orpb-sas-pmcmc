# %%
from typing import Optional
from tqdm import tqdm
from orpb_stochastic.model.model_interface import ModelInterface
from orpb_stochastic.model.utils_chain import Chain
import numpy as np
import scipy.stats as ss
import multiprocessing as mp
from orpb_stochastic.functions.utils import _inverse_pmf
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
# The chain is the same object for every d apart from its theta and its filter state, and
# update_theta rebuilds everything derived from theta. Sending it on every dispatch made the
# parent pickle D x ~4 MB per parameter round, single-threaded, while the workers waited --
# wall time then grew linearly with D even with cores to spare. Instead each worker process
# receives one chain at start-up via the pool initializer and keeps it; a dispatch then
# carries only theta and the previous State (~0.3 MB).
_WORKER_CHAIN = None


def _init_worker(chain):
    global _WORKER_CHAIN
    _WORKER_CHAIN = chain


def _score_algorithm_3(chain_d):
    """Algorithm 3's selection score for one chain, from its finished filter state.

    WW = log prod_t p_f(X*_t|X*_{t-1}) + log prod_k p_g(y_tk|Y_tk)  [+ log pi - log q]

    the first term being Eq. 3.16's process factor (1 for the deterministic MESAS
    convolution) and the second the observation likelihood over all K observations along the
    reference trajectory. The bracketed term is 0 unless the proposal has been adapted away
    from the prior.
    """
    model_W = chain_d.state.W
    B = chain_d._find_traj(chain_d.state.A, model_W, max=True)
    traj_X = chain_d._get_X_traj(chain_d.state.X, B)
    Y_traj = chain_d._get_Y_traj(chain_d.state.Y, B)
    R_traj = chain_d._get_R_traj(chain_d.state.R, B)

    state_W = chain_d.model_interface.transition_model_probability(traj_X)
    WW = (np.log(state_W)
          + chain_d.model_interface.trajectory_loglik(Y_traj)
          + chain_d.model_interface.log_prior_over_proposal())
    return WW, B, R_traj, traj_X, Y_traj, chain_d.state


def worker_SIR(theta_new):
    _WORKER_CHAIN.model_interface.update_theta(theta_new)
    _WORKER_CHAIN.run_particle_filter_SIR()
    return _score_algorithm_3(_WORKER_CHAIN)


def worker_AS(theta_new, prev_state):
    # restore the previous iteration's filter state; AS reads it as its reference
    _WORKER_CHAIN.state = prev_state
    _WORKER_CHAIN.model_interface.update_theta(theta_new)
    _WORKER_CHAIN.run_particle_filter_AS()
    return _score_algorithm_3(_WORKER_CHAIN)


def _pmmh_evaluate(chain_d, theta_new):
    """One PMMH likelihood evaluation: fresh SIR filter at theta_new.

    Returns log Z_hat (the unbiased marginal-likelihood estimate the MH ratio needs) and one
    trajectory drawn from the final weights. Drawing it, rather than taking the arg max, is
    what makes (theta, trajectory) a sample from the joint posterior (ADH 2010, Sec. 2.4.2).
    """
    chain_d.model_interface.update_theta(theta_new) #theta_new is a list of all parameter values in order of self._theta_to_estimate
    chain_d.run_particle_filter_SIR()
    B = chain_d._find_traj(chain_d.state.A, chain_d.state.W, max=False)
    return (chain_d.log_Z,
            chain_d._get_R_traj(chain_d.state.R, B),
            chain_d._get_X_traj(chain_d.state.X, B),
            chain_d._get_Y_traj(chain_d.state.Y, B))


def worker_pmmh(theta_new):
    return _pmmh_evaluate(_WORKER_CHAIN, theta_new)


def _weighted_std(theta_p, theta_mean, W_theta):
    """Posterior spread of theta_p over the D weighted chain proposals.

    W_theta is already normalised to sum to 1, so sum(W*(theta-mean)^2) IS the weighted
    variance. The previous form divided that again by (D-1), which shrank every reported
    std by sqrt(D-1) (~4.8 at D=24) and made a flat likelihood look like a tight posterior:
    with equal weights it returns prior_std/sqrt(D-1) regardless of the data.

    The 1/(1 - sum(W^2)) factor is the standard reliability-weight correction; it reduces to
    the usual D/(D-1) when the weights are uniform, and grows as the effective sample size
    falls, which is the honest behaviour when only a couple of chains carry weight.
    """
    W_theta = np.asarray(W_theta, dtype=float)
    denom = 1.0 - np.sum(W_theta ** 2)
    var = np.sum((np.asarray(theta_p, dtype=float) - theta_mean) ** 2 * W_theta)
    if denom <= 0:
        # Total degeneracy (ESS = 1): one chain holds all the weight, so there is no second
        # point to measure spread from. The spread is unknown, not zero -- returning 0.0
        # would report a perfectly determined parameter. NaN keeps it visibly missing.
        return np.nan
    return np.sqrt(var / denom)


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
        self.ess_record = np.zeros(
            (self.L+1, self._num_theta_to_estimate)
            )
        self.varll_record = np.zeros(
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

        # The prior itself, frozen before anything can adapt dist_model. PMMH samples in
        # z = Phi^-1(F_prior(theta)), where every prior is N(0, 1).
        self._prior = dict(model_interface.dist_model)


    def _save_mle_sas_state(self, theta: Optional[np.ndarray] = None) -> None:
        """Re-run the SAS model at one theta and store pQ/sT/ST.

        Why: in the parallel path the parent's chain copies never see the
        worker's update_theta, so we can't read pQ off any in-memory chain.
        Re-running once is cheap relative to MCMC and gives a single consistent
        answer for both serial and parallel paths. Defaults to theta_record[-1].
        """
        mle_interface = self.models_for_each_chain[0]
        mle_interface.update_theta(self.theta_record[-1, :] if theta is None else theta)
        mle_interface.model.run()   # results were stripped by _init_sas_model(); regenerate for the final theta
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
    
    
    def _sample_theta_candidates(self) -> np.ndarray:
        """D whole parameter vectors from the current proposal (the prior, unless adapted).

        Algorithm 3 draws a complete theta_d(l) per chain each iteration, not one coordinate
        at a time; the coordinate-wise scan in Xu's original code was not part of the method.
        """
        theta_new = np.zeros((self.D, self._num_theta_to_estimate))
        for d in range(self.D):
            theta_new[d, :] = self._sample_theta_init()
        return theta_new

    def _record_iteration(self, l, theta_new, WW, trajectories) -> int:
        """Select theta*(l) per Algorithm 3 and store it, plus the usual diagnostics.

        theta_record holds the SELECTED candidate, which is Algorithm 3's output
        theta*(1:L) -- the sequence over l is the sample. Xu's original code stored a
        per-iteration weighted mean instead, which averages the chain away.
        """
        W_theta = np.exp(WW - WW.max())
        W_theta = W_theta / W_theta.sum()

        if self.is_MAP_MCMC:            # "Set theta*(l) <- arg max ..."
            ind_best = int(np.argmax(W_theta))
        else:                           # "(or draw theta*(l) ~ prod_k p_g ...)"
            ind_best = _inverse_pmf(theta_new[:, -1], W_theta, num=1)[0]

        self.theta_record[l, :] = theta_new[ind_best, :]
        # Spread of the D candidates under their likelihood weights. A diagnostic, not the
        # posterior width: with a score summed over K observations the weights are often
        # near one-hot, which is harmless for arg max selection but collapses this number.
        for p in range(self._num_theta_to_estimate):
            self.theta_std[l, p] = _weighted_std(theta_new[:, p],
                                                 float((theta_new[:, p] * W_theta).sum()),
                                                 W_theta)
        # one W_theta per iteration now, so these are per-iteration values broadcast across
        # the parameter columns to keep the saved file shapes unchanged
        self.ess_record[l, :] = 1.0 / np.sum(W_theta ** 2)
        self.varll_record[l, :] = np.var(WW)

        R_trajs, X_trajs, Y_trajs = trajectories
        self.input_record[l, :] = R_trajs[ind_best]
        self.state_record[l, :] = X_trajs[ind_best]
        self.output_record[l, :] = Y_trajs[ind_best]

        if self.update_theta_dist:
            best_theta = {key: [self.theta_record[l, p], self.theta_std[l, p]]
                          for p, key in enumerate(self._theta_to_estimate)}
            for d in range(self.D):
                self.models_for_each_chain[d]._set_parameter_distribution(
                    update=True, theta_new=best_theta)
            # keep the sampler's own proposal in step: self.dist_model aliases the original
            # interface, so without this the adaptation never reaches _sample_theta_init
            self.models_for_each_chain[0]._set_parameter_distribution(
                update=True, theta_new=best_theta)
            self.dist_model = self.models_for_each_chain[0].dist_model
        return ind_best

    def run_particle_Gibbs(self) -> None:
        """Particle Gibbs sampler, Algorithm 3 (Xu 2025, p. 62).

        Each iteration draws D complete parameter vectors from the prior, runs one particle
        filter per vector (SIR at l=1, AS after), scores each by the observation likelihood
        over all K observations along its reference trajectory, and selects theta*(l).
        """
        chains = [Chain(model_interface=self.models_for_each_chain[d]) for d in range(self.D)]
        WW = np.zeros(self.D)
        R_trajs = np.zeros((self.D, self.T))
        X_trajs = np.zeros((self.D, self.T))
        Y_trajs = np.zeros((self.D, self.T))

        for l in tqdm(range(self.L + 1)):
            theta_new = self._sample_theta_candidates()
            for d in range(self.D):
                chains[d].model_interface.update_theta(theta_new[d, :])
                if l == 0:
                    chains[d].run_particle_filter_SIR()
                else:
                    chains[d].run_particle_filter_AS()
                WW[d], _, R_trajs[d], X_trajs[d], Y_trajs[d], _ = _score_algorithm_3(chains[d])
            self._record_iteration(l, theta_new, WW, (R_trajs, X_trajs, Y_trajs))

        self._save_mle_sas_state()

    def run_particle_Gibbs_parallel(self, num_cores: Optional[int] = None) -> None:
        """Algorithm 3 with the D chains distributed across processes. See run_particle_Gibbs."""
        if num_cores is None:
            num_cores = min(self.D, mp.cpu_count())

        chains = [Chain(model_interface=self.models_for_each_chain[d]) for d in range(self.D)]
        chain_states = [None] * self.D
        WW = np.zeros(self.D)
        R_trajs = np.zeros((self.D, self.T))
        X_trajs = np.zeros((self.D, self.T))
        Y_trajs = np.zeros((self.D, self.T))

        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        ctx = mp.get_context("spawn")  # avoid fork() issues with PyTorch and CUDA
        with ProcessPoolExecutor(max_workers=num_cores, mp_context=ctx,
                                 initializer=_init_worker, initargs=(chains[0],)) as pool:
            for l in tqdm(range(self.L + 1)):
                theta_new = self._sample_theta_candidates()
                if l == 0:
                    futures = [pool.submit(worker_SIR, theta_new[d, :])
                               for d in range(self.D)]
                else:
                    futures = [pool.submit(worker_AS, theta_new[d, :], chain_states[d])
                               for d in range(self.D)]
                try:
                    results = [f.result() for f in futures]
                except BrokenProcessPool:
                    print('[fatal] a worker died (likely OOM) - aborting', flush=True)
                    raise
                if l == 0:
                    print('[info] initial SIR pass complete, entering MCMC loop', flush=True)
                for d in range(self.D):
                    (WW[d], _, R_trajs[d], X_trajs[d], Y_trajs[d],
                     chain_states[d]) = results[d]
                self._record_iteration(l, theta_new, WW, (R_trajs, X_trajs, Y_trajs))

        self._save_mle_sas_state()

    # ------------------------------------------------------------------------------------
    # Particle marginal Metropolis-Hastings (Andrieu, Doucet & Holenstein 2010, Sec. 2.4.2)
    # ------------------------------------------------------------------------------------
    def _to_theta(self, z: np.ndarray) -> np.ndarray:
        """Prior-whitened z -> theta. Each prior maps to N(0, 1) under z = Phi^-1(F(theta)). (F(theta) is prior CDF, phi is standard normal CDF)"""
        u = np.clip(ss.norm.cdf(z), 1e-15, 1 - 1e-15)
        return np.array([self._prior[key].ppf(u[i]) # ppf is percent point function - inverse CDF to find value given cum prob in normal dist.
                         for i, key in enumerate(self._theta_to_estimate)])

    def _to_z(self, theta: np.ndarray) -> np.ndarray:
        u = np.array([self._prior[key].cdf(theta[i])
                      for i, key in enumerate(self._theta_to_estimate)])
        return ss.norm.ppf(np.clip(u, 1e-15, 1 - 1e-15))

    def run_pmmh(self, num_burnin: Optional[int] = None, num_cores: Optional[int] = None,
                 parallel: bool = False, traj_thin: int = 10,
                 target_accept: float = 0.2) -> None:
        """Particle marginal Metropolis-Hastings, D independent chains of L+1 states each.

        Each chain proposes theta' by a random walk, runs a fresh SIR filter at theta' to get
        log Z_hat(theta'), and accepts with probability
            min(1, Z_hat(theta') pi(theta') / Z_hat(theta) pi(theta)).
        Because Z_hat is unbiased, the chain targets the exact posterior p(theta | y) for any
        N (N only affects mixing). The prior enters through the ratio and is never replaced,
        so uninformative priors stay uninformative in the answer, while the proposal still
        learns where the posterior is: that learning is what update_theta_dist was reaching
        for, done here without biasing the target.

        The walk runs in z = Phi^-1(F_prior(theta)), where every prior is N(0, 1), so bounds
        and log scales need no special cases. During burn-in the proposal covariance adapts
        to the pooled chain history (Haario et al. 2001) and its size to target_accept; it is
        frozen afterwards, so post-burn-in draws come from a fixed, valid MH kernel.

        Results:
            theta_chains [D, L+1, p]   every state, burn-in included; posterior = post-burn-in
            logZ_chains  [D, L+1]      log Z_hat of each state
            accept_chains [D, L+1]     1 where the state was a newly accepted proposal
            output_chains / state_chains / input_chains [D, n_saved, T]
                                       trajectory drawn with each state, every traj_thin steps
            theta_record [L+1, p]      chain 0, kept so existing plotting code still runs
            pQ_mle etc.                SAS state at the pooled posterior median
        """
        if (any("C_old" in key for key in self._theta_to_estimate)
                and not self.models_for_each_chain[0].config.get("C_old_as_parameter", False)):
            raise ValueError("PMMH needs config['C_old_as_parameter'] = True; otherwise each "
                             "particle draws its own C_old and theta's C_old has no effect.")
        if num_burnin is None:
            num_burnin = self.L // 2
        if num_cores is None:
            num_cores = min(self.D, mp.cpu_count())
        self.num_burnin = num_burnin
        p, D, L = self._num_theta_to_estimate, self.D, self.L

        self.theta_chains = np.zeros((D, L + 1, p))
        self.logZ_chains = np.full((D, L + 1), -np.inf)
        self.accept_chains = np.zeros((D, L + 1))
        self.traj_iters = np.arange(0, L + 1, traj_thin)
        n_saved = len(self.traj_iters)
        self.output_chains = np.zeros((D, n_saved, self.T))
        self.state_chains = np.zeros((D, n_saved, self.T))
        self.input_chains = np.zeros((D, n_saved, self.T))
        self.prop_scale_record = np.zeros(L + 1)
        self._z_history = np.zeros((D, L + 1, p))

        chains =[Chain(model_interface=self.models_for_each_chain[d]) for d in range(D)]
        pool = None
        if parallel:
            from concurrent.futures import ProcessPoolExecutor
            ctx = mp.get_context("spawn")
            pool = ProcessPoolExecutor(max_workers=num_cores, mp_context=ctx,
                                       initializer=_init_worker, initargs=(chains[0],))

        def evaluate(thetas):
            if pool is not None:
                return [f.result() for f in [pool.submit(worker_pmmh, th) for th in thetas]]
            return [_pmmh_evaluate(chains[d], th) for d, th in enumerate(thetas)]

        try:
            # initial states: one prior draw per chain, so chains start overdispersed
            z = np.random.standard_normal((D, p)) # z(theta) is the marginal likelihood of the data given the parameters: p(y_1:k | theta)
            theta = np.array([self._to_theta(z[d]) for d in range(D)])
            results = evaluate(theta)
            logZ = np.array([r[0] for r in results])
            trajs = [r[1:] for r in results]

            # proposal: log_scale^2 * cov; cov starts at a small isotropic step
            cov = np.eye(p) * 0.1 ** 2 #shape of step i.e. how far to move in each parameter
            log_scale = np.log(2.38 / np.sqrt(p)) #overall size of step (standard optimal scaling for random-walk Metropolis. Stored as log so changes act multiplicatively)
            accepted_since = np.zeros(D)

            for l in tqdm(range(L + 1)):
                if l > 0:
                    chol = np.linalg.cholesky(np.exp(2 * log_scale) * cov) #epsilon added to z. epsilon~N(0,(exp(log_scale)^2*cov)
                    z_prop = z + np.random.standard_normal((D, p)) @ chol.T #proposed new point
                    theta_prop = np.array([self._to_theta(z_prop[d]) for d in range(D)]) #mapped back to get theta_prop (which is the proposed theta the particle filter runs on)
                    results = evaluate(theta_prop)
                    logZ_prop = np.array([r[0] for r in results])
                    # log prior ratio in z space is the N(0,1) density ratio
                    log_alpha = (logZ_prop - logZ # equivalent to Zhat(theta')/Zhat(theta)
                                 - 0.5 * np.sum(z_prop ** 2, axis=1) + 0.5 * np.sum(z ** 2, axis=1)) #equivalent to prior z with density~exp(-1/2|z|^2)
                    log_alpha = np.where(np.isfinite(logZ_prop), log_alpha, -np.inf)
                    accept = np.log(np.random.uniform(size=D)) < log_alpha #same as accepting min(1,alpha)
                    for d in np.flatnonzero(accept):
                        z[d], theta[d], logZ[d] = z_prop[d], theta_prop[d], logZ_prop[d]
                        trajs[d] = results[d][1:]
                    self.accept_chains[:, l] = accept
                    accepted_since += accept

                    # adapt during burn-in only; frozen afterwards
                    if l <= num_burnin and l % 25 == 0:
                        rate = accepted_since.mean() / 25.0
                        log_scale += (rate - target_accept) / np.sqrt(l / 25.0) #Robbins-Monro update
                        accepted_since[:] = 0
                        if l >= 100:
                            hist = self._z_history[:, l // 2 : l, :].reshape(-1, p)
                            cov = np.cov(hist, rowvar=False) + 1e-6 * np.eye(p) #eye*1e-6 keeps pos defininte for Cholesky factorization to work

                self.theta_chains[:, l, :] = theta
                self.logZ_chains[:, l] = logZ
                self.prop_scale_record[l] = np.exp(log_scale)
                self._z_history[:, l, :] = z
                if l % traj_thin == 0:
                    j = l // traj_thin
                    for d in range(D):
                        R_traj, X_traj, Y_traj = trajs[d]
                        self.input_chains[d, j], self.state_chains[d, j], \
                            self.output_chains[d, j] = R_traj, X_traj, Y_traj
                if l == 0:
                    print('[info] initial states evaluated, entering PMMH loop', flush=True)
        finally:
            if pool is not None:
                pool.shutdown()

        # legacy-shaped records (chain 0) so existing save/plot code keeps working
        self.theta_record = self.theta_chains[0].copy()
        self.theta_std = np.zeros_like(self.theta_record)
        j0 = self.traj_iters
        self.output_record[:] = np.nan
        self.state_record[:] = np.nan
        self.input_record[:] = np.nan
        self.output_record[j0] = self.output_chains[0]
        self.state_record[j0] = self.state_chains[0]
        self.input_record[j0] = self.input_chains[0]

        self.theta_posterior_median = np.median(
            self.theta_chains[:, num_burnin + 1:, :].reshape(-1, p), axis=0)
        self._save_mle_sas_state(self.theta_posterior_median)

    def estimate_logZ_sd(self, theta: np.ndarray, reps: int = 20) -> float:
        """Spread of log Z_hat at one theta over repeated filters. Use it to choose N.

        PMMH mixes well when this is ~1-1.5 at a theta near the posterior (e.g. a pilot run's
        median). It grows roughly with sqrt(number of observations), so a 4-year window needs
        more particles than a 3-month one; the cost is small because mesas, not the filter,
        dominates each step.
        """
        chain = Chain(model_interface=self.models_for_each_chain[0])
        return float(np.std([_pmmh_evaluate(chain, theta)[0] for _ in range(reps)]))

    def pmmh_summary(self, num_burnin: Optional[int] = None):
        """Posterior table from post-burn-in draws pooled over chains, with split R-hat.

        R-hat well above 1.01-1.05 means the chains disagree: run longer, not wider.
        """
        import pandas as pd
        b = self.num_burnin if num_burnin is None else num_burnin
        draws = self.theta_chains[:, b + 1:, :]
        D, n, p = draws.shape
        half = n // 2
        split = np.concatenate([draws[:, :half], draws[:, half:2 * half]], axis=0)
        W = split.var(axis=1, ddof=1).mean(axis=0)
        B = half * split.mean(axis=1).var(axis=0, ddof=1)
        rhat = np.sqrt(((half - 1) / half * W + B / half) / W)
        flat = draws.reshape(-1, p)
        prior_sd = np.array([self._prior[k].std() for k in self._theta_to_estimate])
        return pd.DataFrame({
            "mean": flat.mean(axis=0),
            "sd": flat.std(axis=0),
            "q05": np.percentile(flat, 5, axis=0),
            "q50": np.percentile(flat, 50, axis=0),
            "q95": np.percentile(flat, 95, axis=0),
            "sd / prior sd": flat.std(axis=0) / prior_sd,
            "rhat": rhat,
            "accept": self.accept_chains[:, b + 1:].mean(),
        }, index=self._theta_to_estimate)

# %%
