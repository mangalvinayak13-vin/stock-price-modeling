"""
analysis/implied_vol_smile.py
================================
Computes implied volatility across the real market option chain and
compares Black-Scholes theoretical prices to market prices.

If Black-Scholes were literally true (constant sigma), the implied vol
computed from every strike at a given expiry would be the SAME number --
a flat line. In real markets it is not flat: it curves ("smile") or slopes
("skew"), which is direct empirical evidence that a single constant-sigma
model misprices away from the money -- the core motivation for Heston.
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from models.black_scholes import bs_price, bs_vega, implied_volatility


def time_to_expiry(expiry_str, ref_date=None):
    """
    Years between `ref_date` (default: today) and the expiry date string
    ("YYYY-MM-DD", as returned by yfinance). Uses ACT/365 day count (a
    standard convention; other conventions like ACT/360 exist but the
    difference is immaterial at this project's precision).
    """
    if ref_date is None:
        ref_date = dt.date.today()
    expiry_date = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
    days = (expiry_date - ref_date).days
    return max(days, 1) / 365.0  # floor at 1 day to avoid T=0 for same-day expiries


def compute_iv_for_chain(clean_chain, spot, r, q, min_vega=1e-3, verbose=True):
    """
    For every (expiry, strike, option_type) in the cleaned option chain,
    compute the Black-Scholes implied volatility from the market mid price.

    VEGA FILTER (important, not optional): rows where the Black-Scholes
    vega at the solved IV is below `min_vega` are dropped. This is the
    SAME reasoning validated in tests/test_black_scholes.py
    (test_iv_roundtrip_...): when vega is near zero (deep ITM/OTM combined
    with short time to expiry), option price is numerically insensitive to
    sigma, so ANY sigma in a wide range reprices correctly -- implied vol
    is mathematically unidentifiable from price in that regime, not just
    "noisy". In practice, on real yfinance data this filter is essential:
    contracts priced via the `lastPrice` fallback (see clean_option_chain)
    can reflect a trade from days ago, and for a near-zero-vega contract
    even a stale price maps to a wildly wrong "implied vol" (we observed
    IVs of 300%+ on near-expiry deep-ITM AAPL calls before adding this
    filter) -- this is a real data artifact, not a model or solver bug,
    and dropping these points is standard practice, not cherry-picking:
    no options desk trades or quotes IV on a contract with negligible vega.

    Returns
    -------
    pd.DataFrame with columns:
        expiry, T, strike, option_type, market_mid, quote_source,
        moneyness (K/S), implied_vol, vega
    Rows where IV could not be computed, or vega < min_vega, are dropped.
    """
    rows = []
    n_no_solution = 0
    n_low_vega = 0
    for expiry, sides in clean_chain.items():
        T = time_to_expiry(expiry)
        for option_type, df in sides.items():
            opt_type_singular = "call" if option_type == "calls" else "put"
            for _, row in df.iterrows():
                K = float(row["strike"])
                price = float(row["mid"])
                iv = implied_volatility(price, spot, K, T, r, q, opt_type_singular)
                if np.isnan(iv):
                    n_no_solution += 1
                    continue
                vega = bs_vega(spot, K, T, r, iv, q)
                if vega < min_vega:
                    n_low_vega += 1
                    continue
                rows.append({
                    "expiry": expiry,
                    "T": T,
                    "strike": K,
                    "option_type": opt_type_singular,
                    "market_mid": price,
                    "quote_source": row.get("quote_source", "unknown"),
                    "moneyness": K / spot,
                    "implied_vol": iv,
                    "vega": vega,
                })

    if verbose:
        print(f"[iv] computed {len(rows)} usable IV points "
              f"(dropped {n_no_solution} with no valid solution, "
              f"{n_low_vega} with vega < {min_vega} -- unidentifiable regime, see docstring)")

    return pd.DataFrame(rows)


def plot_smile(iv_df, expiries=None, ax=None, option_type="call", title=None):
    """
    Plot implied vol (y) vs strike (x), one line per expiry -- the key
    "volatility smile/skew" figure for the report.

    JUDGMENT CALL: we plot IV vs raw strike (not moneyness K/S or log-
    moneyness ln(K/S)) by default since that's the most directly
    interpretable axis for a single ticker on a single day; moneyness is
    more standard when overlaying multiple tickers/dates on one axis, but
    that's not what this figure is for here.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))

    df = iv_df[iv_df["option_type"] == option_type]
    if expiries is None:
        expiries = sorted(df["expiry"].unique())

    for expiry in expiries:
        sub = df[df["expiry"] == expiry].sort_values("strike")
        if len(sub) < 2:
            continue
        T_years = sub["T"].iloc[0]
        ax.plot(sub["strike"], sub["implied_vol"] * 100, marker="o", ms=3,
                label=f"{expiry} (T={T_years:.2f}y)")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(title or f"Implied Volatility Smile/Skew ({option_type}s)")
    ax.legend(fontsize=8)
    return ax


def bs_pricing_error(iv_df, spot, r, q, atm_sigma_by_expiry=None):
    """
    Compute Black-Scholes theoretical prices across the same strikes as the
    real chain, using a SINGLE flat sigma per expiry (the ATM implied vol
    for that expiry -- the "best single guess" a constant-vol model could
    make), then report RMSE/MAE against real market mid-prices.

    This quantifies exactly how much a constant-sigma model misprices away
    from the money -- the number Heston/Double Heston should improve on.

    Parameters
    ----------
    atm_sigma_by_expiry : dict[expiry -> sigma], optional. If not given,
        it's estimated per expiry as the IV of the strike closest to spot.

    Returns
    -------
    pd.DataFrame with per-expiry RMSE, MAE, and n_options, plus an overall row.
    """
    if atm_sigma_by_expiry is None:
        atm_sigma_by_expiry = {}
        for expiry, sub in iv_df.groupby("expiry"):
            idx = (sub["strike"] - spot).abs().idxmin()
            atm_sigma_by_expiry[expiry] = sub.loc[idx, "implied_vol"]

    results = []
    all_errors = []
    for expiry, sub in iv_df.groupby("expiry"):
        sigma_atm = atm_sigma_by_expiry[expiry]
        model_prices = sub.apply(
            lambda row: bs_price(spot, row["strike"], row["T"], r, sigma_atm, q, row["option_type"]),
            axis=1,
        )
        errors = model_prices.values - sub["market_mid"].values
        all_errors.extend(errors)
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        results.append({"expiry": expiry, "T": sub["T"].iloc[0], "n_options": len(sub),
                         "sigma_used": sigma_atm, "RMSE": rmse, "MAE": mae})

    all_errors = np.array(all_errors)
    results.append({"expiry": "ALL", "T": np.nan, "n_options": len(all_errors),
                     "sigma_used": np.nan,
                     "RMSE": np.sqrt(np.mean(all_errors**2)), "MAE": np.mean(np.abs(all_errors))})

    return pd.DataFrame(results)
