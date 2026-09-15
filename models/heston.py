"""
models/heston.py
==================
The Heston (1993) stochastic volatility model.

SDE system (risk-neutral measure):
    dS(t) = (r - q) S(t) dt + sqrt(v(t)) S(t) dW1(t)
    dv(t) = kappa (theta - v(t)) dt + xi sqrt(v(t)) dW2(t)
    Corr(dW1(t), dW2(t)) = rho dt

Parameters:
    kappa : speed of mean reversion of variance
    theta : long-run mean variance
    xi    : volatility of variance ("vol-of-vol")
    rho   : correlation between asset and variance shocks (typically < 0
            for equities -- the "leverage effect": price drops tend to
            coincide with volatility increases)
    v0    : initial (current) variance

Unlike Black-Scholes, variance itself is a random, mean-reverting process
here instead of a constant -- this is what lets Heston generate a
volatility smile/skew and capture volatility clustering.

FELLER CONDITION: 2*kappa*theta >= xi^2 guarantees the variance process
never reaches exactly zero (it stays strictly positive a.s.). In practice,
calibrated parameters on real equity data very often VIOLATE this
condition (real markets show more "vol-of-vol" than the condition allows)
-- the model still works numerically (the CIR variance process is well
defined and non-negative even when Feller is violated, it just touches
zero occasionally), but this is exactly why the Monte Carlo simulator
(models/heston_mc.py) cannot use naive Euler discretization: a scheme that
doesn't explicitly handle v hitting/crossing zero will blow up or go
complex-valued when Feller is violated, which is the empirically common
case for real calibrated parameters. See models/heston_mc.py.
"""

import numpy as np
from scipy.integrate import quad


def heston_char_func(u, S0, v0, kappa, theta, xi, rho, r, q, T):
    """
    Characteristic function of ln(S_T) under the Heston model, evaluated
    at (possibly complex) argument u:
        phi(u) = E[ exp(i * u * ln(S_T)) ]

    "LITTLE TRAP" FORMULATION (Albrecher, Mayer, Schoutens, Tistaert 2007,
    "The Little Heston Trap"): the ORIGINAL Heston (1993) formula computes
    a complex logarithm term that, when evaluated naively along a path as
    T grows (or for certain u), can cross a branch cut of the complex log
    and produce a DISCONTINUOUS characteristic function -- which silently
    corrupts the pricing integral (small errors in T can cause large,
    unpredictable jumps in the computed price). The fix is purely a
    reparametrization: define

        d  = sqrt( (rho*xi*i*u - kappa)^2 + xi^2*(i*u + u^2) )
        c1 = kappa - rho*xi*i*u
        g  = (c1 - d) / (c1 + d)          <-- the "little trap" ratio
                                               (reciprocal of the original
                                               1993 formula's g)

    Using THIS g (rather than its reciprocal, which the 1993 paper used)
    keeps the argument of the log term inside the unit disk for all
    practical parameter ranges, avoiding the branch-cut crossing. This is
    a well-documented fix, not a novel derivation -- we implement it
    exactly as in Albrecher et al. (2007), eq. 12-14, since getting this
    formula right (not the naive 1993 one) was explicitly required.

    Then:
        C(u,T) = i*u*(r-q)*T
                 + (kappa*theta/xi^2) * [ (c1-d)*T - 2*ln( (1-g*e^{-dT}) / (1-g) ) ]
        D(u,T) = ((c1-d)/xi^2) * ( (1-e^{-dT}) / (1-g*e^{-dT}) )
        phi(u) = exp( C(u,T) + D(u,T)*v0 + i*u*ln(S0) )
    """
    u = np.asarray(u, dtype=complex)
    i = 1j

    c1 = kappa - rho * xi * i * u
    d = np.sqrt(c1**2 + xi**2 * (i * u + u**2))

    # Sign convention judgment call: some references take d with a
    # negative sign / choose the branch of sqrt differently. We follow
    # Albrecher et al. (2007) directly: with g defined as (c1-d)/(c1+d),
    # this specific sign choice for d is the one that avoids the
    # discontinuity (verified empirically below with the continuity test
    # in tests/test_heston.py, which checks the CF varies smoothly as T
    # increases across a long horizon -- exactly where the naive 1993
    # formula is known to break).
    g = (c1 - d) / (c1 + d)

    exp_dT = np.exp(-d * T)
    C = i * u * (r - q) * T + (kappa * theta / xi**2) * (
        (c1 - d) * T - 2.0 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = (c1 - d) / xi**2 * ((1 - exp_dT) / (1 - g * exp_dT))

    return np.exp(C + D * v0 + i * u * np.log(S0))


def feller_condition(kappa, theta, xi):
    """
    Feller condition: 2*kappa*theta - xi^2 >= 0 keeps variance strictly
    positive. Returns (is_satisfied: bool, slack: float) where slack =
    2*kappa*theta - xi^2 (positive means satisfied, magnitude indicates
    how comfortably). Used both as a diagnostic and as a penalty term in
    calibration (see calibration/heston_calibration.py).
    """
    slack = 2 * kappa * theta - xi**2
    return slack >= 0, slack


def heston_price(S0, K, T, r, q, kappa, theta, xi, rho, v0, option_type="call",
                  u_max=200.0, limit=200):
    """
    European option price under Heston, via direct numerical integration
    of the standard P1/P2 (Gil-Pelaez inversion) representation:

        Call = S0 * e^{-qT} * P1 - K * e^{-rT} * P2
        Put  = Call - S0*e^{-qT} + K*e^{-rT}      (put-call parity)

    where
        P2 = 1/2 + (1/pi) * Integral_0^inf  Re[ e^{-iu ln K} * phi(u) / (iu) ] du
        P1 = 1/2 + (1/pi) * Integral_0^inf  Re[ e^{-iu ln K} * phi(u-i) / (iu * phi(-i)) ] du

    This uses a SINGLE characteristic function phi (evaluated at u and at
    the shifted point u-i for P1) rather than Heston's original two
    separate characteristic functions phi1, phi2 -- this is the standard
    modern reformulation (see e.g. Gatheral, "The Volatility Surface",
    ch. 2) that is mathematically equivalent to the 1993 paper but only
    needs one (little-trap-safe) CF implementation, halving the places a
    branch-cut bug could hide.

    phi(-i) is the forward price E[S_T] = S0*e^{(r-q)T} by definition of
    the characteristic function (setting u=-i makes phi(u) = E[S_T]); we
    use this identity directly rather than re-evaluating phi at u=-i
    numerically, both as a minor efficiency gain and as a free internal
    consistency check (see tests/test_heston.py).

    JUDGMENT CALL -- integration truncation (u_max=200): the P1/P2
    integrand decays because phi(u) decays (the characteristic function of
    any well-behaved distribution with finite variance) combined with the
    1/u factor, but it's also oscillatory (from the e^{-iu ln K} term). We
    integrate to a large-but-finite u_max=200 rather than literal infinity
    -- in practice the integrand is negligible well before u=200 for
    realistic equity parameters (v0, theta of order 0.01-0.25, i.e.
    vol of order 10-50%), and using a finite bound makes scipy's adaptive
    quadrature (quad) more numerically reliable than an infinite-range
    integral, which can under-resolve slowly-decaying oscillatory tails.
    We verify this choice doesn't matter by testing that the price is
    insensitive to further increasing u_max (see tests/test_heston.py).

    Returns
    -------
    float, the theoretical option price.
    """
    log_K = np.log(K)
    forward = S0 * np.exp((r - q) * T)  # = phi(-i), the risk-neutral forward price

    def integrand_P2(u):
        if u == 0:
            return 0.0  # removable singularity at u=0; integrand -> 0 in the limit
        phi = heston_char_func(u, S0, v0, kappa, theta, xi, rho, r, q, T)
        val = np.exp(-1j * u * log_K) * phi / (1j * u)
        return val.real

    def integrand_P1(u):
        if u == 0:
            return 0.0
        phi_shifted = heston_char_func(u - 1j, S0, v0, kappa, theta, xi, rho, r, q, T)
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
        # Put-call parity: C - P = S*e^{-qT} - K*e^{-rT}
        return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
