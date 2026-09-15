"""
models/double_heston.py
=========================
The Double Heston (two-factor stochastic volatility) model, following
Christoffersen, Heston & Jacobs (2009), "The Shape and Term Structure of
the Index Option Smirk". Motivation: a SINGLE mean-reverting variance
factor forces one mean-reversion speed (kappa) to explain volatility
persistence at every horizon simultaneously -- but real volatility often
shows both a fast-decaying component (news/earnings-driven bursts that
fade in days-weeks) and a slower, more persistent component (macro/regime-
level shifts over months-years). Two independent CIR factors let the
model represent both at once.

SDE system (risk-neutral / physical measure, r replaced by mu for
physical-measure use in calibration -- see calibration/double_heston_calibration.py):
    dS/S  = (r-q) dt + sqrt(v1) dW1_S + sqrt(v2) dW2_S
    dv1   = kappa1 (theta1 - v1) dt + xi1 sqrt(v1) dW1_v
    dv2   = kappa2 (theta2 - v2) dt + xi2 sqrt(v2) dW2_v
    Corr(dW1_S, dW1_v) = rho1,   Corr(dW2_S, dW2_v) = rho2
    Factor 1 and factor 2's Brownian motions are ALL mutually independent
    across factors (dW1_S, dW1_v) independent of (dW2_S, dW2_v).

Total instantaneous variance of returns is v1 + v2 -- the two factors add.

FACTORIZED CHARACTERISTIC FUNCTION (the key structural result that makes
this model tractable): because the two factors are independent and each
only interacts with S through its own separate Brownian motion, the
characteristic function of ln(S_T) factorizes as

    phi(u) = exp(i*u*(ln S0 + (r-q)T)) * f1(u, v1_0, T) * f2(u, v2_0, T)

where f1, f2 are EXACTLY the same "little trap" C(u,T)+D(u,T)*v0 building
blocks used in single Heston (models/heston.py), one per factor, each
using that factor's own (kappa_i, theta_i, xi_i, rho_i, vi_0) -- just
without a separate drift term (the drift i*u*(r-q)*T is applied once, at
the top level, not once per factor). This directly follows from the
independence of the two factors: E[e^{iu ln S_T}] with ln S_T's randomness
coming from two independent sources factorizes into a product of
expectations, one per source. We implement it by literally reusing
models/heston.py's C, D formulas with each factor's own parameters.
"""

import numpy as np
from scipy.integrate import quad

from models.heston import feller_condition as _single_feller_condition


def _factor_C_D(u, kappa, theta, xi, rho, T):
    """
    The little-trap C(u,T) (WITHOUT the drift term i*u*(r-q)*T -- that's
    added once at the top level in double_heston_char_func, not per
    factor) and D(u,T), for one CIR variance factor. Identical math to
    models/heston.py's heston_char_func, factored out here so both
    single-factor building blocks can be reused without duplicating the
    little-trap derivation.
    """
    u = np.asarray(u, dtype=complex)
    i = 1j
    c1 = kappa - rho * xi * i * u
    d = np.sqrt(c1**2 + xi**2 * (i * u + u**2))
    g = (c1 - d) / (c1 + d)
    exp_dT = np.exp(-d * T)

    C0 = (kappa * theta / xi**2) * ((c1 - d) * T - 2.0 * np.log((1 - g * exp_dT) / (1 - g)))
    D = (c1 - d) / xi**2 * ((1 - exp_dT) / (1 - g * exp_dT))
    return C0, D


def double_heston_char_func(u, S0, v1_0, v2_0, kappa1, theta1, xi1, rho1,
                             kappa2, theta2, xi2, rho2, r, q, T):
    """
    Factorized characteristic function of ln(S_T) under Double Heston:
        phi(u) = exp(i*u*(ln S0 + (r-q)T)) * f1(u) * f2(u)
    where f_i(u) = exp(C_i(u,T) + D_i(u,T)*vi_0), using the little-trap
    building blocks from _factor_C_D for each factor independently. See
    module docstring for why this factorization is valid.
    """
    u = np.asarray(u, dtype=complex)
    i = 1j

    C1, D1 = _factor_C_D(u, kappa1, theta1, xi1, rho1, T)
    C2, D2 = _factor_C_D(u, kappa2, theta2, xi2, rho2, T)

    f1 = np.exp(C1 + D1 * v1_0)
    f2 = np.exp(C2 + D2 * v2_0)
    drift_term = np.exp(i * u * (np.log(S0) + (r - q) * T))

    return drift_term * f1 * f2


def double_heston_feller_conditions(kappa1, theta1, xi1, kappa2, theta2, xi2):
    """
    Feller condition applied to EACH factor separately (each v_i is its
    own independent CIR process, so each needs its own condition
    2*kappa_i*theta_i >= xi_i^2 to stay strictly positive).

    Returns
    -------
    (both_satisfied: bool, slack1: float, slack2: float)
    """
    sat1, slack1 = _single_feller_condition(kappa1, theta1, xi1)
    sat2, slack2 = _single_feller_condition(kappa2, theta2, xi2)
    return (sat1 and sat2), slack1, slack2


def double_heston_price(S0, K, T, r, q, kappa1, theta1, xi1, rho1, v1_0,
                         kappa2, theta2, xi2, rho2, v2_0, option_type="call",
                         u_max=200.0, limit=200):
    """
    European option price under Double Heston, via the same P1/P2
    (Gil-Pelaez) integration used for single Heston (models/heston.py),
    just with the factorized two-factor characteristic function plugged
    in instead. The P1/P2 formulas themselves are model-agnostic -- they
    only need SOME valid characteristic function of ln(S_T), which is
    exactly what double_heston_char_func provides.

        Call = S0*e^{-qT}*P1 - K*e^{-rT}*P2
        P2 = 1/2 + (1/pi) Integral_0^inf Re[e^{-iu ln K} phi(u)/(iu)] du
        P1 = 1/2 + (1/pi) Integral_0^inf Re[e^{-iu ln K} phi(u-i)/(iu*forward)] du

    See models/heston.py's heston_price docstring for the full derivation
    and the u_max truncation judgment call, which applies identically here.
    """
    log_K = np.log(K)
    forward = S0 * np.exp((r - q) * T)

    def integrand_P2(u):
        if u == 0:
            return 0.0
        phi = double_heston_char_func(u, S0, v1_0, v2_0, kappa1, theta1, xi1, rho1,
                                       kappa2, theta2, xi2, rho2, r, q, T)
        val = np.exp(-1j * u * log_K) * phi / (1j * u)
        return val.real

    def integrand_P1(u):
        if u == 0:
            return 0.0
        phi_shifted = double_heston_char_func(u - 1j, S0, v1_0, v2_0, kappa1, theta1, xi1, rho1,
                                               kappa2, theta2, xi2, rho2, r, q, T)
        val = np.exp(-1j * u * log_K) * phi_shifted / (1j * u * forward)
        return val.real

    I_P2, _ = quad(integrand_P2, 0, u_max, limit=limit)
    I_P1, _ = quad(integrand_P1, 0, u_max, limit=limit)

    P2 = 0.5 + I_P2 / np.pi
    P1 = 0.5 + I_P1 / np.pi

    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

    if option_type == "call":
        return call
    elif option_type == "put":
        return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
