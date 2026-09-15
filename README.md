# Stock Price Modeling: Brownian Motion → Black-Scholes → Heston → Double Heston

A semester project implementing and rigorously validating four models of increasing
sophistication for equity price dynamics, calibrated to real historical stock data, with a
30-year walk-forward backtest and a real-time Streamlit dashboard.

## Important methodology note (read this first)

This project calibrates every model **purely from historical stock RETURNS** — no options
market data is used anywhere. This was a deliberate decision made partway through
development, after discovering that `yfinance`'s free option-chain data has severe quality
issues (near-empty bid/ask/open interest for most contracts) and, more fundamentally, never
provides *historical* option chains (only today's live snapshot) — making a genuine 30-year
options-based backtest infeasible without a paid data vendor.

**Consequence:** Black-Scholes and Heston/Double Heston are, by design, option-pricing
models. Here they're calibrated from the return series instead (moment-matching on rolling
realized variance and its autocorrelation structure — see `calibration/heston_calibration.py`
for the full derivation), and any option prices/smiles they produce are illustrative
demonstrations of the pricing formulas, not fits to real market quotes. This is stated
explicitly, and repeatedly, throughout the code and the notebook — it's a legitimate,
citable methodological choice, and being upfront about it is exactly what should survive
viva scrutiny.

## Project structure

```
data/           Data fetching & caching: yfinance historical OHLC, Finnhub real-time quotes,
                option-chain fetching/cleaning (built early on, kept for completeness, not
                used in the current equity-only calibration pipeline)
models/         One file per model: gbm.py, black_scholes.py, heston.py, heston_mc.py,
                double_heston.py, double_heston_mc.py
calibration/    scipy.optimize-based calibration: heston_calibration.py,
                double_heston_calibration.py (return-based GMM, multi-start Levenberg-Marquardt)
analysis/       realized_variance.py, gbm_diagnostics.py, implied_vol_smile.py (illustrative),
                backtest.py (30-year walk-forward), model_comparison.py (summary tables/plots)
tests/          Correctness tests for every module (see "Testing" below)
notebooks/      full_pipeline.ipynb — runs the entire pipeline end-to-end with inline plots;
                doubles as the report appendix
app.py          Streamlit real-time dashboard (deployable)
```

## Setup

Requires Python 3.10+.

```bash
cd "Physics project SEM 1"
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Real-time quotes (optional but needed for the "Live Price" dashboard tab)

Real-time prices use [Finnhub](https://finnhub.io)'s free tier (`yfinance` is NOT used for
this — it's an unofficial scraper with no SLA, unsuitable for a deployed site; see
`data/realtime.py` for the full reasoning).

1. Register for a free API key at https://finnhub.io/register
2. Copy `.env.example` to `.env` and fill in your key:
   ```
   FINNHUB_API_KEY=your_key_here
   ```

Everything else (historical data, calibration, backtesting) works without this key.

## How to run

**Run the test suite** (does this survive scrutiny — run this first):
```bash
for f in tests/test_*.py; do python3 "$f"; done
```
All tests print `[OK] ...` lines explaining what each check validates, and end with
`All ... tests passed.` Every test is a genuine correctness check (put-call parity, Heston
characteristic-function vs Monte Carlo agreement, parameter recovery on synthetic data with
known ground truth, no-lookahead backtest window validation, etc.) — see each test file's
docstring for what it's actually testing and why.

**Run the notebook** (full pipeline, report appendix):
```bash
jupyter notebook notebooks/full_pipeline.ipynb
# or, to re-execute headlessly:
jupyter nbconvert --to notebook --execute --inplace notebooks/full_pipeline.ipynb
```
Change the `TICKER` variable in the second cell to run on a different stock.

**Run the dashboard:**
```bash
streamlit run app.py
```

**Deploy the dashboard** (free): push this repo to GitHub, then deploy at
[share.streamlit.io](https://share.streamlit.io) pointing at `app.py`. Add `FINNHUB_API_KEY`
as a secret in the Streamlit Cloud app settings (not committed to the repo).

## Key findings (see the notebook's Section 8 for full detail)

- GBM's constant-volatility assumption is visibly wrong: rolling realized volatility
  clusters and swings substantially rather than staying flat (motivates everything after it).
- Heston's characteristic-function pricer (little-trap formulation, avoiding the well-known
  1993-formula discontinuity bug) and its Monte Carlo simulator (Full Truncation Euler,
  which correctly handles the variance process even when the Feller condition is violated)
  agree within Monte Carlo sampling error — validating both independently.
- Calibrating Heston on the **full 30-year history in one shot gives implausible parameters**
  (long-run vol ~49%, leverage correlation ≈ 0) because 30 years spans multiple structurally
  different regimes (dot-com, 2008 GFC, COVID) that violate the calibration's stationarity
  assumption. Shorter (5-10 year) windows give sensible numbers. This is *why* the backtest
  uses rolling windows rather than one static calibration.
- **Double Heston's second factor consistently collapses toward negligible** when calibrated
  from equity returns alone (checked across AAPL, SPY, TSLA) — the classic two-timescale
  motivation for Double Heston is an *options-market* phenomenon (fitting short- and
  long-dated implied-vol smiles together), and doesn't automatically carry over to a
  returns-only recalibration. Reported honestly rather than forced.
- The 30-year **walk-forward backtest** (strict no-lookahead: calibrate on a rolling window,
  judge only on the year immediately after) shows GBM is competitive with — and on the
  overall/calm-regime RMSE, slightly better than — Heston at pure volatility point-forecasting,
  and **none of the three models passes the Kupiec VaR coverage test** pooled across 24 years.
  Neither result is flattering to the more complex models, and both are reported anyway,
  because that's what a genuine backtest is for.

## Judgment calls (flagged throughout the code, summarized here)

Every non-obvious implementation/numerical decision is flagged with a comment at the point
it's made, explaining *why*, so it can be defended in a viva. Highlights:
- 252 trading days/year annualization convention (`models/gbm.py`)
- Newton-Raphson + bisection fallback for implied vol, with an explicit note on when implied
  vol is mathematically unidentifiable from price (near-zero vega) rather than just noisy
  (`models/black_scholes.py`)
- Little-trap Heston characteristic function (`models/heston.py`), P1/P2 Gil-Pelaez
  integration with a truncated-but-verified-insensitive upper bound
- Full Truncation Euler (not naive Euler, not the more complex Andersen QE scheme — a
  deliberate complexity/accuracy tradeoff) for Monte Carlo simulation (`models/heston_mc.py`)
- Return-based GMM calibration methodology, including two bugs found and fixed via testing
  (a sign error from mechanical window-overlap contamination, and a since-corrected
  attenuation bias in the leverage-correlation estimator) — see
  `calibration/heston_calibration.py`
- Walk-forward backtest window sizing and the Kupiec VaR coverage test methodology
  (`analysis/backtest.py`)

## Dependencies

numpy, scipy, pandas, matplotlib, yfinance, requests, python-dotenv, streamlit, jupyter,
pytest. See `requirements.txt` for versions.
