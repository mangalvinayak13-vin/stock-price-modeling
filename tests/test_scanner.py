"""
tests/test_scanner.py
========================
Correctness checks for trading/scanner.py, run on a hand-built synthetic
option chain (no live network fetch needed) so these run fast and
deterministically. Confirms the core invariant: a contract PRICED
EXACTLY AT the Heston model's own theoretical price must come back with
iv_gap == 0 (the scanner shouldn't invent a mispricing that isn't there),
and a contract priced away from it shows a gap of the right sign.
"""

import datetime as dt
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.heston import heston_price
from models.black_scholes import implied_volatility
from analysis.implied_vol_smile import time_to_expiry
from trading.scanner import scan_chain_vs_heston, top_mispricings

HESTON_PARAMS = {"kappa": 3.0, "theta": 0.04, "xi": 0.5, "rho": -0.4, "v0": 0.045}
S0, R, Q = 100.0, 0.03, 0.0


def _expiry_and_T(days_ahead):
    """scan_chain_vs_heston derives T from the expiry LABEL via
    time_to_expiry() (real calendar distance from today), not from a T we
    pass in directly -- so build both together here and price legs with
    the SAME T that scan_chain_vs_heston will independently recompute,
    or the two would silently disagree by a fraction of a day."""
    label = (dt.date.today() + dt.timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    return label, time_to_expiry(label)


def _make_chain(strikes, T, expiry_label, price_fn):
    """Build a clean_option_chain-shaped dict for one expiry: for each
    strike, both a call and a put priced by `price_fn(K, option_type)`,
    with a tight synthetic bid/ask straddling that price so quote_source
    is always 'bidask' (matching what clean_option_chain would tag a
    liquid, well-quoted contract as)."""
    rows_calls, rows_puts = [], []
    for K in strikes:
        for rows, opt in [(rows_calls, "call"), (rows_puts, "put")]:
            mid = price_fn(K, opt)
            rows.append({"strike": K, "mid": mid, "quote_source": "bidask"})
    return {expiry_label: {"calls": pd.DataFrame(rows_calls), "puts": pd.DataFrame(rows_puts)}}


def test_contract_priced_exactly_at_model_price_has_zero_gap():
    """A chain priced EXACTLY by the Heston model itself must scan as
    iv_gap == 0 everywhere -- the scanner must not manufacture a phantom
    mispricing when market price == model price by construction."""
    expiry_label, T = _expiry_and_T(45)
    strikes = np.linspace(80, 120, 9)

    def model_price_fn(K, opt):
        return heston_price(S0, K, T, R, Q, HESTON_PARAMS["kappa"], HESTON_PARAMS["theta"],
                             HESTON_PARAMS["xi"], HESTON_PARAMS["rho"], HESTON_PARAMS["v0"], opt)

    chain = _make_chain(strikes, T, expiry_label, model_price_fn)
    result = scan_chain_vs_heston(chain, S0, R, Q, HESTON_PARAMS)

    assert len(result) > 0
    assert result["iv_gap"].abs().max() < 1e-4
    assert (result["quote_source"] == "bidask").all()
    print(f"[OK] Chain priced exactly at Heston model price: max|iv_gap| = "
          f"{result['iv_gap'].abs().max():.2e} (~0, as required)")


def test_richly_priced_contract_flagged_rich_with_correct_sign():
    """A contract deliberately priced with HIGHER implied vol than the
    model's own surface must show a positive iv_gap and be tagged 'rich'."""
    expiry_label, T = _expiry_and_T(30)
    strikes = [100.0]
    market_sigma = 0.35  # deliberately above the model's ATM vol (~sqrt(v0)=0.212)

    from models.black_scholes import bs_price

    def rich_price_fn(K, opt):
        return bs_price(S0, K, T, R, market_sigma, Q, opt)

    chain = _make_chain(strikes, T, expiry_label, rich_price_fn)
    result = scan_chain_vs_heston(chain, S0, R, Q, HESTON_PARAMS)

    assert len(result) > 0
    assert (result["iv_gap"] > 0).all()
    assert (result["rich_or_cheap"] == "rich").all()
    print(f"[OK] Deliberately overpriced (35% vol vs ~21% model) contracts correctly "
          f"flagged 'rich', iv_gap={result['iv_gap'].iloc[0]:.4f}")


def test_top_mispricings_filters_by_zscore_and_limit():
    expiry_label, T = _expiry_and_T(30)
    strikes = np.linspace(70, 130, 13)

    def model_price_fn(K, opt):
        return heston_price(S0, K, T, R, Q, HESTON_PARAMS["kappa"], HESTON_PARAMS["theta"],
                             HESTON_PARAMS["xi"], HESTON_PARAMS["rho"], HESTON_PARAMS["v0"], opt)

    # Perturb a couple of strikes to create real outliers against an
    # otherwise "fairly priced" (model-matching) chain.
    def perturbed_fn(K, opt):
        base = model_price_fn(K, opt)
        if K in (90.0, 110.0):
            return base * 1.15
        return base

    chain = _make_chain(strikes, T, expiry_label, perturbed_fn)
    result = scan_chain_vs_heston(chain, S0, R, Q, HESTON_PARAMS)
    top = top_mispricings(result, n=5, min_abs_z=1.0)

    assert len(top) <= 5
    assert (top["iv_gap"].abs() >= result["iv_gap"].abs().median()).all()
    print(f"[OK] top_mispricings() returns {len(top)} outlier contracts, "
          f"largest |iv_gap|={top['iv_gap'].abs().max():.4f}")


if __name__ == "__main__":
    test_contract_priced_exactly_at_model_price_has_zero_gap()
    test_richly_priced_contract_flagged_rich_with_correct_sign()
    test_top_mispricings_filters_by_zscore_and_limit()
    print("\nAll scanner tests passed.")
