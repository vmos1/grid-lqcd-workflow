"""Integrated autocorrelation time via the Gamma / UW method.

Reference:
    U. Wolff (ALPHA Collaboration),
    "Monte Carlo errors with less errors",
    Comput. Phys. Commun. 156 (2004) 143-153,
    hep-lat/0306017, Eqs. (26), (38), (41).
"""

import math
import numpy as np


def gamma_method(data, S=1.5):
    """
    Estimate the integrated autocorrelation time tau_int.

    Uses the automatic-windowing procedure of Madras & Sokal (1988) as
    described by Wolff (hep-lat/0306017).  The window W_opt is the smallest W
    satisfying W >= S * tau_int(W), balancing truncation bias against
    statistical noise in the tail of the autocorrelation function.

    Parameters
    ----------
    data : array-like, shape (N,)
        Evenly-spaced Monte Carlo time series of a single observable.
    S : float
        Window parameter.  Wolff recommends S = 1.5.  Increase if tau_int
        appears underestimated (window closes too early); decrease to reduce
        noise at the cost of larger truncation bias.

    Returns
    -------
    dict with keys:
        mean        : float    sample mean
        sigma       : float    statistical error on the mean
        tau_int     : float    integrated autocorrelation time
        tau_int_err : float    error on tau_int (Madras-Sokal Eq. 3.4)
        window      : int      automatic window W_opt
        rho         : ndarray  normalised ACF rho(t) for t = 0 .. W_opt
        gamma       : ndarray  unnormalised ACF Gamma(t) for t = 0 .. W_opt
    """
    x = np.asarray(data, dtype=float)
    N = len(x)
    if N < 4:
        raise ValueError("Need at least 4 data points.")

    mean  = float(x.mean())
    xc    = x - mean                  # centred series
    max_lag = N // 2

    # --- Unnormalised ACF Gamma(t) with 1/N normalisation ---
    # Direct summation: O(N * max_lag).  Fine for typical HMC run lengths.
    gamma = np.array([np.dot(xc[:N - t], xc[t:]) / N
                      for t in range(max_lag + 1)])

    if gamma[0] == 0.0:
        raise ValueError("Variance is zero — is the observable constant?")

    rho = gamma / gamma[0]            # normalised ACF rho(0) = 1

    # --- Automatic windowing ---
    # tau_int(W) = 1/2 + sum_{t=1}^{W} rho(t)
    # Advance W until the Madras-Sokal condition W >= S * tau_int(W) is met.
    tau_int = 0.5
    W_opt   = max_lag                 # fallback if window never closes
    for W in range(1, max_lag + 1):
        tau_int += rho[W]
        if W >= S * tau_int:
            W_opt = W
            break
    else:
        # Window did not close — series may be very long or very correlated.
        tau_int = 0.5 + float(rho[1:].sum())

    # --- Error estimates ---
    # sigma^2(mean) = 2 * tau_int * Gamma(0) / N    (Wolff Eq. 14)
    sigma = math.sqrt(2.0 * tau_int * gamma[0] / N)

    # sigma^2(tau_int) ~ (4*W+2)/N * tau_int^2      (Madras-Sokal Eq. 3.4)
    tau_int_err = tau_int * math.sqrt((4 * W_opt + 2) / N)

    return {
        'mean'       : mean,
        'sigma'      : sigma,
        'tau_int'    : tau_int,
        'tau_int_err': tau_int_err,
        'window'     : W_opt,
        'rho'        : rho[:W_opt + 1],
        'gamma'      : gamma[:W_opt + 1],
    }
