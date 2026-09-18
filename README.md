# Stock Price Modeling: Brownian Motion → Black-Scholes → Heston → Double Heston

A semester project implementing and rigorously validating four models of increasing
sophistication for equity price dynamics, calibrated to real historical stock data, with a
30-year walk-forward backtest and a real-time Streamlit dashboard. **Extended with an
options-trading layer** (Greeks, multi-leg strategy payoffs, a live model-vs-market
mispricing scanner, and a 33-year backtested volatility-selling strategy) — see
[Options Trading Extension](#options-trading-extension) below.

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
trading/        Options-trading layer, built on the models above: greeks.py (BS closed-form +
                Heston finite-difference Greeks), strategies.py (straddle/strangle/spread/
                condor payoffs, breakevens, aggregated Greeks), scanner.py (live chain vs.
                Heston-model mispricing scan), vrp_backtest.py (33-year SPY volatility-risk-
                premium backtest using VIX as the historical implied-vol input — see
                "Options Trading Extension" below)
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

## Options Trading Extension

Everything above calibrates and validates the four *price* models. This extension asks a
different question — **can these models be used to actually trade options?** — and adds a
`trading/` package (Greeks, multi-leg strategies, a live mispricing scanner) plus a real
33-year backtest of an options-selling strategy.

### The same data limitation, handled differently for the backtest

The core methodology note at the top of this README still applies: `yfinance` never serves a
*historical* option chain, only today's live one. `trading/greeks.py`, `trading/strategies.py`,
and `trading/scanner.py` work against **today's real, live chain** (via
`data/loader.get_option_chain`/`clean_option_chain`, unused elsewhere in the project until now),
so they don't need historical option data at all.

`trading/vrp_backtest.py` does need a multi-decade history, so it uses a different, standard
trick from the volatility-risk-premium literature (Carr & Wu 2009; the same approach behind
CBOE's own PUT/BXM benchmark indices): the **CBOE VIX index has a genuine free daily history
back to 1990** (`data/loader.get_vix_history`), and every VIX value *is* real option-market
information — it's computed by the CBOE each day directly from that day's live SPX chain, just
compressed into one number instead of a full per-strike surface. Using VIX as the
Black-Scholes volatility input lets the backtest use real, contemporaneous options pricing for
every trading day since 1993 (SPY's inception), without needing a paid data vendor. Full
reasoning and limitations are in the module's docstring.

### What's in `trading/`

- **`greeks.py`** — Black-Scholes delta/gamma/vega/theta/rho in closed form; Heston's
  equivalents via central finite differences on the validated `heston_price()` (Heston has no
  simple closed-form Greeks without a second layer of numerical integration). Cross-checked:
  the finite-difference Heston Greeks converge to the Black-Scholes closed form in the
  constant-vol limit (`xi→0`, `v0=theta`) — see `tests/test_greeks.py`.
- **`strategies.py`** — straddles, strangles, vertical spreads, iron condors, built from either
  pricer (Black-Scholes or Heston, via `bs_pricer`/`heston_pricer` closures). Computes entry
  cost/credit, breakevens, capped vs. uncapped max profit/loss, and aggregated Greeks. Checked
  against hand-derivable textbook results (a straddle's breakevens are exactly `K ± premium`; a
  vertical spread's max loss is exactly the debit paid) — see `tests/test_strategies.py`.
- **`scanner.py`** — compares every liquid contract in today's live chain against the
  ticker's Heston model (calibrated from returns only, **not** fit to this chain), on a common
  implied-vol scale, flagged by a z-score against that day's own dispersion. Explicitly
  documented as a **diagnostic, not a trading signal** — the two things being compared use
  different information (trailing returns vs. forward-looking option pricing), so a gap can
  just as easily mean "the market knows something the return history doesn't" as "the market
  is wrong." See `tests/test_scanner.py` (checked against a chain priced exactly at the model's
  own price, which must scan as zero gap everywhere).
- **`vrp_backtest.py`** — the 33-year backtest below.

### The VRP backtest: selling SPY straddles, 1993–2026

**Strategy:** each month, sell an at-the-money 1-month SPY straddle priced at that day's real
VIX level, held unhedged to expiry; entry costs a flat 1 vol-point haircut off VIX. A
**VRP signal** — trade only when `VIX − (Heston's own physical-measure forecast vol)` is
positive, i.e. the market is pricing in more volatility than a returns-only model itself
expects — is compared against blindly selling every month, and against buy-and-hold SPY.
Heston is re-calibrated once a year on a trailing 5-year window, strictly before the year it
trades (identical no-lookahead discipline to `analysis/backtest.py`); only the current
instantaneous variance `v0` is refreshed monthly from trailing realized variance. Full run:
`for f in tests/test_vrp_backtest.py; do python3 "$f"; done`, or the **Volatility Risk Premium
Backtest** dashboard tab.

**Result** (SPY, Sep 1993–Sep 2026, 27 traded years / 324 months, 5-year calibration, 1 vol
point round-trip cost):

| Strategy | CAGR | Annualized vol | Sharpe | Max drawdown | Win rate | Active |
|---|---|---|---|---|---|---|
| Always sell the straddle | 11.86% | 10.07% | **1.17** | −24.37% | 71.0% | 100% |
| VRP-signal-filtered | 10.18% | 8.85% | 1.15 | **−18.04%** | 72.3% | 65.7% |
| Buy & hold SPY | 10.88% | 18.67% | 0.55 | −55.19% | — | — |

Both straddle-selling variants matched buy-and-hold's CAGR with roughly **half the volatility
and drawdown** — a real, historically persistent volatility risk premium (VIX has, on average,
overpriced realized SPY volatility across this 27-year sample). The VRP signal filter mainly
traded *risk* for *return*: sitting out a third of months cut the max drawdown further (from
−24% to −18%) at a modest CAGR cost, rather than clearly beating the always-sell baseline —
another honest, not-flattering-to-the-fancier-approach finding, consistent with this project's
Heston/Double Heston results above.

**Read this before trusting those numbers for anything real:**
- The straddle is struck **exactly at spot** and priced with **one flat VIX-implied vol**, not
  a real chain's discrete strikes or skew — an index-overlay-style approximation, not a
  claim about a real fill.
- **No delta hedging.** The reported P&L includes pure directional short-straddle exposure
  held to expiry, not a variance-isolated position.
- One month (e.g. Mar 2020) can dominate several years of collected premium; a monthly Sharpe
  ratio understates this fat-tail/negative-skew risk (the classic critique of short-vol
  strategies — "picking up nickels in front of a steamroller").
- 1 vol-point of round-trip cost is a judgment call, not a measured number — SPX options are
  very liquid, but a real fill depends on size and market conditions.

### Running the extension

```bash
for f in tests/test_greeks.py tests/test_strategies.py tests/test_scanner.py tests/test_vrp_backtest.py; do
    python3 "$f"
done
```
Then use the dashboard's **Options Strategy Builder**, **Live Mispricing Scanner**, and
**Volatility Risk Premium Backtest** tabs (`streamlit run app.py`).

## Dependencies

numpy, scipy, pandas, matplotlib, yfinance, requests, python-dotenv, streamlit, jupyter,
pytest. See `requirements.txt` for versions.
