"""
trading/scanner.py
=====================
Live mispricing scanner: compares TODAY's real option chain (market
implied vol) against the Heston model's own volatility surface, implied
from a model FITTED PURELY TO HISTORICAL RETURNS (not to the chain
itself -- see README's methodology note, unchanged here).

WHAT THIS IS AND ISN'T:
  - IS: a demonstration that a returns-only Heston fit and the market's
    own option-implied vol disagree in economically interpretable ways
    (e.g. the market pricing in more skew than a symmetric-ish return
    history implies), and a concrete, inspectable list of which contracts
    disagree most.
  - IS NOT: a trading signal ready to run capital against. The comparison
    is between two DIFFERENT information sets (backward-looking realized
    dynamics vs. forward-looking option-market pricing), so a large gap
    can just as easily mean "the market knows something about the future
    that trailing returns don't capture" as "the market is mispricing
    this contract." Flagged explicitly in every output of this module,
    and again in the dashboard tab that surfaces it.

METHOD, per contract:
  1. market_iv  <- Black-Scholes IV implied from the real bid/ask mid
     (models/black_scholes.implied_volatility), via the SAME cleaning
     pipeline (data/loader.clean_option_chain) and vega filter used
     elsewhere in this project (analysis/implied_vol_smile.py) -- no
     lastPrice-fallback quotes are used here (quote_source == 'bidask'
     only), since a mispricing signal built on a stale trade print is not
     a real signal (see clean_option_chain's docstring for why fallback
     quotes exist and why they're unsuitable for this specific use).
  2. model_price <- heston_price() at that contract's (K, T) using the
     supplied calibrated Heston parameters.
  3. model_iv <- Black-Scholes IV implied FROM the Heston model price
     (puts both prices on the same, interpretable vol-point scale instead
     of comparing raw dollar prices across very different strikes/
     expiries).
  4. iv_gap = market_iv - model_iv, in vol points. Positive = the market
     is pricing MORE volatility into this contract than the (returns-
     only) Heston model expects ("rich" vs. the model); negative = the
     market is pricing LESS ("cheap" vs. the model).

A z-score of iv_gap across that day's whole surviving chain is also
reported, since a contract's raw gap is easiest to interpret relative to
how dispersed that day's gaps are overall (a lone contract at +2 vol
points means something different on a day where every contract is +1.8
vs a day where the typical gap is +0.1).
"""

import numpy as np
import pandas as pd

from models.black_scholes import implied_volatility, bs_vega
from models.heston import heston_price
from analysis.implied_vol_smile import time_to_expiry


def scan_chain_vs_heston(clean_chain, spot, r, q, heston_params, min_vega=1e-3):
    """
    Compare every liquid (bid/ask-quoted) contract in `clean_chain`
    (output of data.loader.clean_option_chain) against the Heston model
    implied by `heston_params` (a dict with kappa/theta/xi/rho/v0, e.g.
    straight from calibrate_heston_gmm).

    Returns
    -------
    pd.DataFrame, one row per surviving contract, columns:
        expiry, T, strike, option_type, moneyness, market_mid, market_iv,
        model_price, model_iv, iv_gap, iv_gap_z, quote_source (== 'bidask'
        always here), rich_or_cheap ('rich' if iv_gap > 0 else 'cheap')
    Sorted by |iv_gap| descending (biggest disagreements first).
    """
    rows = []
    for expiry, sides in clean_chain.items():
        T = time_to_expiry(expiry)
        if T <= 0:
            continue
        for option_type_plural, df in sides.items():
            option_type = "call" if option_type_plural == "calls" else "put"
            bidask_only = df[df["quote_source"] == "bidask"]
            for _, row in bidask_only.iterrows():
                K = float(row["strike"])
                mkt_price = float(row["mid"])

                market_iv = implied_volatility(mkt_price, spot, K, T, r, q, option_type)
                if np.isnan(market_iv):
                    continue
                mkt_vega = bs_vega(spot, K, T, r, market_iv, q)
                if mkt_vega < min_vega:
                    continue  # same unidentifiable-IV reasoning as implied_vol_smile.py

                model_price = heston_price(spot, K, T, r, q, heston_params["kappa"],
                                            heston_params["theta"], heston_params["xi"],
                                            heston_params["rho"], heston_params["v0"], option_type)
                model_iv = implied_volatility(model_price, spot, K, T, r, q, option_type)
                if np.isnan(model_iv):
                    continue

                rows.append({
                    "expiry": expiry, "T": T, "strike": K, "option_type": option_type,
                    "moneyness": K / spot, "market_mid": mkt_price, "market_iv": market_iv,
                    "model_price": model_price, "model_iv": model_iv,
                    "iv_gap": market_iv - model_iv, "quote_source": "bidask",
                })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    gap_mean, gap_std = result["iv_gap"].mean(), result["iv_gap"].std(ddof=1)
    result["iv_gap_z"] = (result["iv_gap"] - gap_mean) / gap_std if gap_std > 1e-8 else 0.0
    result["rich_or_cheap"] = np.where(result["iv_gap"] > 0, "rich", "cheap")

    return result.sort_values("iv_gap", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def top_mispricings(scan_df, n=15, min_abs_z=1.0):
    """
    The `n` largest |iv_gap| contracts with |z-score| >= min_abs_z (i.e.
    outliers relative to that day's OWN dispersion, not just an arbitrary
    fixed vol-point threshold that would mean different things on calm vs
    turbulent days).
    """
    if scan_df.empty:
        return scan_df
    filtered = scan_df[scan_df["iv_gap_z"].abs() >= min_abs_z]
    return filtered.head(n)
