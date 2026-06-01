"""Equilibration / burn-in detection for HMC time series."""

import numpy as np


def suggest_burnin(df, observable='plaquette', window=20, tolerance=0.02):
    """
    Suggest a burn-in cut: the first trajectory after which the rolling mean
    of *observable* stays within *tolerance* of the long-run mean.

    The long-run mean is estimated from the last 50 % of the data, which is
    expected to be thermalised.

    Parameters
    ----------
    df : pd.DataFrame
        Output of extract.load_run().
    observable : str
        Column to analyse (default 'plaquette').
    window : int
        Rolling-mean window length in trajectories.
    tolerance : float
        Fractional tolerance: stable when |rolling_mean / long_mean - 1| < tol.

    Returns
    -------
    burnin_traj : int or None
        Trajectory label of the suggested burn-in point, or None if the series
        appears always stable (or too short to judge).
    """
    if observable not in df.columns:
        return None
    col = df[observable].dropna()
    if len(col) < window * 2:
        return None

    vals       = col.values
    traj_vals  = df.loc[col.index, 'traj'].values
    long_mean  = vals[len(vals) // 2:].mean()

    if long_mean == 0.0:
        return None

    # Rolling mean via convolution (valid region only)
    kernel  = np.ones(window) / window
    rolling = np.convolve(vals, kernel, mode='valid')
    # rolling[i] corresponds to traj_vals[i + window - 1]

    rel_dev   = np.abs(rolling / long_mean - 1.0)
    stable    = rel_dev < tolerance

    # Find the first index where stability is maintained for at least `window`
    # consecutive steps — simple run-length check.
    run = 0
    for i, s in enumerate(stable):
        run = run + 1 if s else 0
        if run >= window:
            traj_idx = i - window + 1 + (window - 1)   # start of stable run
            return int(traj_vals[min(traj_idx, len(traj_vals) - 1)])

    return None
