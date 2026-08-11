"""Run-configuration snapshot helpers.

Goal: when a pMCMC run finishes, save a JSON file next to the result .npy/.csv
files that captures everything needed to reproduce or audit the run later.
"""
import json
import os
import subprocess
from datetime import datetime
from typing import Iterable, Optional


def _get_git_commit(repo_path: Optional[str] = None) -> str:
    """Return the short git commit hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=repo_path,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _extract_priors(theta_init: dict, estimated_keys: Iterable[str]) -> dict:
    """Pull prior_dis / prior_params for each estimated key from theta_init."""
    priors = {}
    estimated = set(estimated_keys)
    for section in ('obs_uncertainty', 'scale_parameters'):
        if section not in theta_init:
            continue
        for key, spec in theta_init[section].items():
            if key in estimated:
                priors[key] = {
                    'section': section,
                    'distribution': spec.get('prior_dis'),
                    'params': spec.get('prior_params'),
                    'is_nonnegative': spec.get('is_nonnegative'),
                }
    return priors


def _sanitize(value):
    """Best-effort JSON serializer for non-native types (numpy, datetime, etc.)."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def save_run_config(
    out_path: str,
    *,
    case_name: str,
    config: dict,
    N: int,
    D: int,
    L: int,
    date_start,
    date_end,
    resolution: str,
    data_source: str,
    theta_to_estimate: Iterable[str],
    theta_init: dict,
    seed: Optional[int] = None,
    extra: Optional[dict] = None,
    notes: Optional[str] = None
) -> dict:
    """Write a JSON snapshot of the run configuration.

    Args:
        out_path: where to write the JSON file.
        case_name: the SAS specs case (e.g. 'storage_q_ug_et_u').
        config: the SSM config dict (dt, MAP flags, update_theta_dist, etc.).
        N, D, L: particle filter and Gibbs sizing.
        date_start, date_end: data window (pd.Timestamp or str).
        resolution: 'H', 'D', 'W' etc.
        data_source: path or filename of the input CSV.
        theta_to_estimate: list of estimated parameter keys (from model_interface).
        theta_init: the full theta dict (sas_specs, solute_parameters, etc.).
        seed: the numpy random seed used for the run (if set).
        extra: any extra metadata you want to attach (e.g. job ID, hostname).
        notes: any additional notes about the run.

    Returns:
        The snapshot dict (also written to disk).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_commit = _get_git_commit(repo_path=repo_root)

    estimated_keys = list(theta_to_estimate)

    config_serializable = {
        k: _sanitize(v) for k, v in config.items()
        if k != 'observed_made_each_step'  # too long; reconstructable from data
    }

    snapshot = {
        'metadata': {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'git_commit': git_commit,
            'random_seed': seed,
            'extra': extra or {},
        },
        'case': {
            'name': case_name,
            'sas_specs': _sanitize(theta_init.get('sas_specs')),
            'solute_parameters': _sanitize(theta_init.get('solute_parameters')),
        },
        'data': {
            'source': str(data_source),
            'date_start': str(date_start),
            'date_end': str(date_end),
            'resolution': resolution,
        },
        'mcmc': {
            'N_particles': int(N),
            'D_chains': int(D),
            'L_iterations': int(L),
            'options': config_serializable,
        },
        'theta_to_estimate': estimated_keys,
        'priors': _extract_priors(theta_init, estimated_keys),
        'notes': notes,
    }

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    return snapshot


def set_run_seed(seed: int) -> None:
    """Set the numpy random seed for the run.

    Call this once at the top of the script, before any sampling. Keep the
    seed value in sync with what you pass to save_run_config so the saved
    snapshot reflects what actually ran.
    """
    import numpy as np
    np.random.seed(seed)
