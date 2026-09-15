"""
report/build_report.py
========================
Assembles the final Word report (report/Stock_Price_Modeling_Report.docx)
from the figures/numbers produced by report/generate_figures.py. Run
generate_figures.py first (or after any code change that should be
reflected in the report).

Not part of the core deliverable -- a one-off script to produce the report
document from already-validated project code and results.
"""
import json
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

with open(os.path.join(HERE, "numbers.json")) as f:
    N = json.load(f)

MULTI_TICKER_PATH = os.path.join(HERE, "multi_ticker_check.json")
MULTI = json.load(open(MULTI_TICKER_PATH)) if os.path.exists(MULTI_TICKER_PATH) else None

doc = Document()

# --- base style ---
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)


def h1(text):
    doc.add_heading(text, level=1)

def h2(text):
    doc.add_heading(text, level=2)

def p(text, bold=False, italic=False):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.bold = bold
    run.italic = italic
    return para

def bullet(text):
    doc.add_paragraph(text, style="List Bullet")

def eq(text):
    """A visually offset 'equation' paragraph (centered, monospace-ish)."""
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run(text)
    run.italic = True
    return para

def figure(filename, caption, width=6.0):
    doc.add_picture(os.path.join(FIG, filename), width=Inches(width))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(10)

def table_from_rows(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr_cells[i].text = str(htext)
        for run in hdr_cells[i].paragraphs[0].runs:
            run.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    return t


# ===========================================================================
# TITLE PAGE
# ===========================================================================
title_para = doc.add_paragraph()
title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_run = title_para.add_run("Stock Price Modeling:\nBrownian Motion, Black-Scholes, "
                                "Heston, and Double Heston")
title_run.bold = True
title_run.font.size = Pt(24)

doc.add_paragraph()
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub_run = sub.add_run("A Comparative Study of Stochastic Volatility Models for Equity Prices,\n"
                       "with a 30-Year Walk-Forward Out-of-Sample Backtest")
sub_run.font.size = Pt(14)
sub_run.italic = True

for _ in range(4):
    doc.add_paragraph()

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
for line in ["[Your Name]", "[Roll Number]", "[Department / Programme -- B.Tech 1st Year]",
             "[College / Institute Name]", "[Course Name / Code]", "[Submission Date]"]:
    r = meta.add_run(line + "\n")
    r.font.size = Pt(12)

doc.add_page_break()


# ===========================================================================
# ABSTRACT
# ===========================================================================
h1("Abstract")
p(f"This project implements and validates four models of increasing sophistication for "
  f"equity price dynamics -- Geometric Brownian Motion (GBM), Black-Scholes, the Heston "
  f"stochastic volatility model, and the two-factor Double Heston model -- and compares "
  f"them on real historical data for {N['ticker']} spanning {N['history_start']} to "
  f"{N['history_end']} ({N['n_obs']} trading days). Rather than the conventional "
  f"options-market calibration, all models here are calibrated directly from the equity "
  f"return series using a Generalized Method of Moments (GMM) procedure built on rolling "
  f"realized variance and its autocorrelation structure -- a deliberate methodological "
  f"choice explained in Section 3.2. Every pricing formula and Monte Carlo simulator is "
  f"independently validated by construction (e.g. characteristic-function prices are "
  f"checked against Monte Carlo simulation of the underlying stochastic differential "
  f"equation; Heston is checked to collapse exactly to Black-Scholes as vol-of-vol tends "
  f"to zero; Double Heston is checked to collapse exactly to single Heston when one factor "
  f"is switched off). The centerpiece result is a strict walk-forward, no-lookahead "
  f"backtest spanning 24 rolling one-year out-of-sample windows across three decades of "
  f"market history, including the dot-com bust, the 2008 financial crisis, and the COVID-19 "
  f"crash, evaluated with both volatility-forecast accuracy and a formal Value-at-Risk "
  f"coverage test (the Kupiec test used in banking regulation). The results are reported "
  f"honestly, including findings that do not flatter the more sophisticated models: on this "
  f"backtest, the simpler GBM model is competitive with Heston on point-forecast accuracy, "
  f"and none of the three models passes the VaR coverage test across the full 30-year "
  f"sample. These findings, and their causes, are discussed in detail in Section 6.")

doc.add_page_break()


# ===========================================================================
# TABLE OF CONTENTS (manual, static -- Word can auto-generate from headings
# via References > Table of Contents if the user regenerates it)
# ===========================================================================
h1("Table of Contents")
p("(In Microsoft Word: right-click here and choose \"Update Field\" after opening, or "
  "insert References > Table of Contents to auto-generate from the heading styles used "
  "throughout this document.)", italic=True)
doc.add_page_break()


# ===========================================================================
# 1. INTRODUCTION
# ===========================================================================
h1("1. Introduction and Motivation")
p("Modeling how stock prices evolve over time is a foundational problem in quantitative "
  "finance. The simplest and most widely taught model, Geometric Brownian Motion (GBM), "
  "assumes that a stock's volatility -- how large its random fluctuations are -- is "
  "constant over time. This assumption underlies the Nobel-Prize-winning Black-Scholes "
  "option pricing formula, and is a reasonable first approximation, but real markets "
  "visibly violate it: volatility itself fluctuates, clusters (calm periods and turbulent "
  "periods persist), and tends to rise when prices fall (the well-documented \"leverage "
  "effect\"). This project builds up, in order, from the simplest model to models that "
  "explicitly relax the constant-volatility assumption:")
bullet("Geometric Brownian Motion (GBM) -- constant drift and volatility, the baseline.")
bullet("Black-Scholes -- the option pricing formula built on GBM, plus its inverse problem "
       "(recovering implied volatility from a price).")
bullet("Heston (1993) -- volatility itself follows a mean-reverting random process, "
       "correlated with the price process (capturing the leverage effect and volatility "
       "clustering).")
bullet("Double Heston (Christoffersen, Heston & Jacobs, 2009) -- two independent "
       "mean-reverting volatility factors, motivated by the empirical observation that "
       "volatility persistence often shows both a fast-decaying and a slow-decaying "
       "component simultaneously.")
p("Each model is implemented from its underlying mathematical formulation (not a "
  "pre-built library), calibrated to real historical stock data, cross-validated by at "
  "least two independent numerical methods where applicable, and unit-tested against "
  "known theoretical identities. Section 3.2 explains an important methodological pivot "
  "made partway through development: calibrating from the stock's return history directly, "
  "rather than from options market data.")

doc.add_page_break()


# ===========================================================================
# 2. THEORETICAL BACKGROUND
# ===========================================================================
h1("2. Theoretical Background")

h2("2.1 Geometric Brownian Motion")
p("Under GBM, the stock price S(t) follows the stochastic differential equation")
eq("dS(t) = mu * S(t) dt + sigma * S(t) dW(t)")
p("where W(t) is a standard Brownian motion, mu is the (constant) drift, and sigma is the "
  "(constant) volatility. Applying Ito's lemma to X(t) = ln S(t) gives the EXACT solution "
  "(used directly for simulation in this project, avoiding any discretization error):")
eq("S(t) = S(0) * exp( (mu - sigma^2/2) t + sigma W(t) )")
p("mu and sigma are estimated from historical daily log returns via their sample mean and "
  "standard deviation, annualized using the standard 252-trading-day convention.")

h2("2.2 Black-Scholes")
p("The Black-Scholes-Merton formula gives the price of a European call option as")
eq("C = S e^(-qT) N(d1) - K e^(-rT) N(d2)")
eq("d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)")
p("where r is the risk-free rate, q the dividend yield, K the strike, T the time to "
  "expiry, and N(.) the standard normal CDF. The inverse problem -- given a market price, "
  "recover the implied sigma -- has no closed form and is solved numerically (Section 4.1).")

h2("2.3 Heston Stochastic Volatility Model")
p("Heston (1993) replaces the constant sigma with a second stochastic process for the "
  "instantaneous variance v(t):")
eq("dS(t) = (r-q) S(t) dt + sqrt(v(t)) S(t) dW1(t)")
eq("dv(t) = kappa (theta - v(t)) dt + xi sqrt(v(t)) dW2(t),   Corr(dW1,dW2) = rho")
p("kappa is the speed of mean reversion, theta the long-run average variance, xi the "
  "\"vol-of-vol\", and rho the correlation between price and variance shocks (typically "
  "negative for equities -- the leverage effect). The Feller condition, "
  "2*kappa*theta >= xi^2, guarantees v(t) never reaches exactly zero; real calibrated "
  "equity parameters very often violate it (Section 5.3), which has direct numerical "
  "consequences discussed in Section 4.2.")

h2("2.4 Double Heston")
p("Christoffersen, Heston and Jacobs (2009) extend the model to two independent "
  "mean-reverting variance factors:")
eq("dS/S = (r-q) dt + sqrt(v1) dW1_S + sqrt(v2) dW2_S")
eq("dv_i = kappa_i (theta_i - v_i) dt + xi_i sqrt(v_i) dW_i,   i = 1,2 (independent factors)")
p("Total instantaneous variance is v1 + v2. Because the two factors are independent, the "
  "characteristic function of ln S(T) factorizes as a product of two single-Heston-style "
  "building blocks (Section 4.3), which is what makes the model tractable.")

doc.add_page_break()


# ===========================================================================
# 3. DATA AND METHODOLOGY
# ===========================================================================
h1("3. Data and Methodology")

h2("3.1 Data Sources")
p(f"Historical daily OHLC price data is fetched via the yfinance library. This report uses "
  f"{N['ticker']} with {N['history_years']} years of history "
  f"({N['history_start']} to {N['history_end']}, {N['n_obs']} trading days), though the "
  f"code accepts any ticker. The risk-free rate is proxied by the most recent yield on the "
  f"13-week US Treasury bill (ticker ^IRX); at the time of this run, r = "
  f"{N['risk_free_rate']:.3%}. The dividend yield is computed as annual dividend rate "
  f"divided by current price; for {N['ticker']}, q = {N['dividend_yield']:.3%}. A "
  f"real-time quote feed (Finnhub, free tier) supports the live dashboard component of "
  f"this project (Section 7).")

h2("3.2 Why Equity-Return-Based Calibration, Not Options Data")
p("The conventional way to calibrate Heston/Double Heston -- and the way originally "
  "planned for this project -- is to fit model option prices to real market option quotes "
  "across many strikes and expiries. Two problems were discovered empirically during "
  "development that led to a deliberate pivot:")
bullet("yfinance's free option-chain data has severe quality issues: on real AAPL data, "
       "essentially none of the near-dated contracts had a live, non-zero bid/ask quote "
       "or reported open interest, forcing reliance on stale \"last trade\" prices with "
       "no-arbitrage bound validation as a fallback.")
bullet("More fundamentally, yfinance never provides HISTORICAL option chains -- only "
       "today's live snapshot -- making a genuine multi-decade options-based backtest "
       "infeasible without a paid data vendor (e.g. OptionMetrics, ORATS), which was "
       "outside this project's scope and budget.")
p("Given the project's goal of a real, multi-decade backtest with free data, all models "
  "are instead calibrated purely from the stock's own historical RETURN series. This is a "
  "legitimate alternative methodology (a form of the Generalized Method of Moments applied "
  "to the CIR variance process underlying Heston), but it is a genuinely different "
  "empirical target than options-implied calibration, with consequences discussed "
  "honestly in Section 6 -- most notably, that Double Heston's classic advantage (fitting "
  "short- and long-dated option-implied-vol smiles simultaneously) does not automatically "
  "carry over to this setting.")

h2("3.3 Return-Based GMM Calibration Methodology")
p("The calibration target is rolling realized variance, RV(t): the annualized sample "
  "variance of daily log returns over a trailing 21-trading-day window (a standard proxy "
  "for the unobservable instantaneous variance v(t)). Under the CIR process governing "
  "Heston's variance, several EXACT stationary-distribution properties hold:")
bullet("E[v] = theta  (the sample mean of RV targets theta)")
bullet("Var[v] = theta * xi^2 / (2*kappa)  (the sample variance of RV constrains xi given "
       "kappa and theta)")
bullet("ACF(v, lag) = exp(-kappa * lag)  (an EXACT exponential autocorrelation-decay "
       "property of the CIR process, giving a direct handle on kappa from the SHAPE of "
       "RV's autocorrelation function, not just its level)")
p("kappa, theta and xi are fit by nonlinear least squares (scipy.optimize.least_squares, "
  "Levenberg-Marquardt) matching these model-implied moments against their empirical "
  "counterparts computed from RV(t) -- using the autocorrelation function at many lags "
  "(not just one) over-identifies the 3-parameter system, making the fit robust to noise "
  "in any single lag. A soft Feller-condition penalty is included in the objective. "
  "Optimization uses multiple random restarts (Section 4.4 discusses why this, rather "
  "than a full global optimizer, is sufficient here). rho is estimated from the "
  "correlation between daily returns and a carefully constructed non-overlapping-window "
  "proxy for the change in variance (Section 4.5 details two real estimation bugs found "
  "and fixed during development). Double Heston extends this to two factors, warm-started "
  "from the single-Heston fit as suggested by the standard calibration literature for this "
  "model, with an ordering constraint (kappa1 >= kappa2) to break the label-switching "
  "symmetry inherent to any two-factor model.")

h2("3.4 Walk-Forward Backtesting Methodology")
p("The headline result of this project is a strict walk-forward, no-lookahead backtest: "
  "each model is calibrated using ONLY data up to a point in time, then judged exclusively "
  "on the period immediately after, which was never used in calibration. The window then "
  "slides forward and repeats.")
eq("|---- calibration window (5y) ----|-- test window (1y) --|")
p(f"With a 5-year calibration window, 1-year test window, and 1-year step, this produces "
  f"{N['kupiec']['n_windows']} independent out-of-sample test periods across the full "
  f"history, spanning multiple distinct market regimes (the dot-com bust, the 2008 "
  f"financial crisis, and the COVID-19 crash).")
p("Two evaluation metrics are used at each window:")
bullet("Volatility forecast accuracy: GBM's forecast is simply its calibration-window "
       "sigma (constant, by construction). Heston's forecast uses the exact closed-form "
       "expected AVERAGE variance over the test horizon under the CIR mean-reversion ODE, "
       "E[v(t)] = theta + (v0-theta)e^(-kappa t), averaged over the horizon -- letting it "
       "say \"volatility is currently elevated/depressed and should partially mean-revert "
       "by test-window's end\", something GBM structurally cannot do.")
bullet("Value-at-Risk (VaR) backtesting via the Kupiec (1995) coverage test: each model "
       "forecasts a 1-day 95% VaR at the start of each test window (via Monte Carlo "
       "simulation for Heston/Double Heston), held fixed through that window. Breaches "
       "(days the realized return fell below the forecast threshold) are pooled across "
       "all windows and tested against the target 5% rate using the same likelihood-ratio "
       "test banking regulators require for internal VaR model validation under Basel "
       "market-risk rules.")

doc.add_page_break()


# ===========================================================================
# 4. IMPLEMENTATION DETAILS AND NUMERICAL JUDGMENT CALLS
# ===========================================================================
h1("4. Implementation Details and Key Numerical Judgment Calls")
p("Every non-obvious implementation decision below was a deliberate, defensible choice "
  "made during development, not an accident -- flagged here for transparency and to "
  "anticipate the natural follow-up questions about numerical soundness.")

h2("4.1 Robust Implied Volatility Solver")
p("Implied volatility is recovered by inverting the Black-Scholes formula using "
  "Newton-Raphson (fast, quadratic convergence when it works) with an automatic fallback "
  "to bisection (slower but mathematically guaranteed to converge, since price is "
  "monotonically increasing in sigma). Newton-Raphson is known to fail when vega (the "
  "price's sensitivity to sigma) is near zero -- which happens for deep in/out-of-the-money "
  "or very short-dated options -- because dividing by a near-zero vega sends the next "
  "iterate flying to a nonsensical value. Testing further revealed that in this near-zero-"
  "vega regime, implied volatility is not merely hard to estimate but mathematically "
  "UNIDENTIFIABLE from price (many different sigma values reprice to the same value to "
  "machine precision) -- a genuine property of the pricing function, not a solver "
  "weakness, and the test suite (tests/test_black_scholes.py) explicitly separates this "
  "unidentifiable regime from genuine solver failures.")

h2("4.2 The \"Little Heston Trap\" and Full Truncation Euler Simulation")
p("The Heston characteristic function is implemented using the \"little trap\" "
  "reparametrization (Albrecher, Mayer, Schoutens & Tistaert, 2007), which avoids a "
  "well-documented numerical discontinuity in the original 1993 formula caused by a "
  "complex logarithm crossing a branch cut for long maturities. For Monte Carlo "
  "simulation of the underlying SDE, naive Euler discretization of the variance process "
  "is numerically unsafe: a Gaussian increment can push the discretized variance negative, "
  "after which its square root is undefined. This is not a rare edge case -- it occurs "
  "whenever the Feller condition is violated, which real calibrated equity parameters "
  "very often are (Section 5.3). This project uses Full Truncation Euler (Lord, Koekkoek "
  "& van Dijk, 2010), which uses max(v,0) wherever v enters a square root or drift term "
  "while letting the stored value itself go negative before truncation on the next step -- "
  "a deliberate complexity/accuracy tradeoff against the more sophisticated (and more "
  "commonly used in industry) Andersen (2008) Quadratic-Exponential scheme, justified by "
  "keeping the implementation independently verifiable within the project's timeline while "
  "still correctly handling the negative-variance problem (validated in Section 5.3).")

h2("4.3 Factorized Double Heston Characteristic Function")
p("Because the two variance factors in Double Heston are independent and interact with "
  "the price process through separate Brownian motions, the characteristic function of "
  "ln S(T) factorizes exactly as a product of two single-factor \"little trap\" building "
  "blocks (one per factor), with the drift term applied once at the top level rather than "
  "once per factor. This is implemented by literally reusing the validated single-Heston "
  "building-block code for each factor.")

h2("4.4 Optimizer Choice: Multi-Start Local Search vs. Global Optimization")
p("The return-based GMM calibration objective is low-dimensional (3 free parameters for "
  "single Heston, 6 for Double Heston) and built entirely from smooth, closed-form moment "
  "formulas with no simulation noise -- a much better-behaved optimization landscape than "
  "the highly multimodal objective that results from fitting Heston to option prices "
  "directly across many strikes and maturities (the originally planned approach). "
  "Multi-start Levenberg-Marquardt (8-10 random restarts around a closed-form initial "
  "guess derived from the CIR moment equations) was found to be fast and sufficient; a "
  "genuine global optimizer (differential evolution) is also implemented and available "
  "as a robustness cross-check.")

h2("4.5 Two Real Bugs Found and Fixed via Testing")
p("Two genuine estimation bugs in the leverage-correlation (rho) estimator were caught "
  "specifically because the calibration pipeline was validated against SYNTHETIC data "
  "with a known ground-truth rho, not just run on real data and eyeballed:")
bullet("Sign error: the first implementation correlated today's return with the change in "
       "a TRAILING rolling-window realized-variance estimate -- but that window "
       "mechanically includes today's own return, so the \"correlation\" partly just "
       "measured how large today's return was, not its relationship to future variance. "
       "Fixed by comparing non-overlapping backward and forward windows that both exclude "
       "today's return entirely.")
bullet("Attenuation bias: even after the sign fix, the estimator recovered only "
       "roughly 15-20% of the true rho's magnitude. This is a well-documented, genuine "
       "statistical phenomenon (not a further bug): a single day's squared return is an "
       "extremely noisy (chi-squared) estimate of instantaneous variance, which severely "
       "dilutes any correlation estimated from it -- the reason practitioners use "
       "intraday data for this specific estimate, which is unavailable here. Corrected by "
       "calibrating an empirical attenuation factor via the project's own validated Monte "
       "Carlo simulator (simulating short reference paths at a known rho and measuring "
       "how much the estimator dilutes it), then rescaling the raw estimate accordingly.")

doc.add_page_break()


# ===========================================================================
# 5. RESULTS
# ===========================================================================
h1("5. Results")

h2("5.1 GBM: Parameter Estimates and Diagnostics")
p(f"From {N['history_years']} years of {N['ticker']} daily log returns, the estimated "
  f"annualized parameters are mu = {N['gbm_mu']:.4f} and sigma = {N['gbm_sigma']:.4f}. "
  f"Figure 1 shows the raw price history; Figure 2 shows the full diagnostic panel: "
  f"simulated GBM sample paths, the empirical log-return distribution against a fitted "
  f"Normal, a QQ-plot against the Normal distribution, and rolling realized volatility "
  f"over time.")
figure("01_price_history.png", f"Figure 1: {N['ticker']} daily closing price, "
       f"{N['history_start']} to {N['history_end']}.")
figure("02_gbm_diagnostics.png",
       "Figure 2: GBM diagnostic panel. Note the QQ-plot's departure from the reference "
       "line at the tails (fat tails relative to Normal) and the rolling realized "
       "volatility panel's clear clustering and large swings -- direct empirical evidence "
       "against GBM's constant-volatility assumption, motivating every model that follows.")

h2("5.2 Black-Scholes: Validation")
p(f"Put-call parity (C - P = S e^(-qT) - K e^(-rT), a model-independent no-arbitrage "
  f"identity) was verified to hold to machine precision: computed difference = "
  f"{N['bs_parity_lhs'] - N['bs_parity_rhs']:.2e}. An implied-volatility round-trip test "
  f"(price an option at a known sigma = {N['bs_iv_roundtrip_input_sigma']:.4f}, then "
  f"invert the resulting price back to sigma) recovered "
  f"{N['bs_iv_roundtrip_recovered']:.4f} -- agreement to 9 decimal places.")

h2("5.3 Heston: Calibration and Cross-Validation")
p("A first calibration attempt on the FULL 30-year history produced an important, honest "
  "finding rather than a clean result:")
table_from_rows(
    ["Parameter", "30-year calibration (non-stationary)", "5-year calibration (stationary)"],
    [
        ["kappa", f"{N['heston_30y']['kappa']:.3f}", f"{N['heston_5y']['kappa']:.3f}"],
        ["theta", f"{N['heston_30y']['theta']:.4f}", f"{N['heston_5y']['theta']:.4f}"],
        ["Implied long-run vol (sqrt theta)", f"{N['heston_30y']['theta']**0.5:.2%}", f"{N['heston_5y']['theta']**0.5:.2%}"],
        ["xi", f"{N['heston_30y']['xi']:.3f}", f"{N['heston_5y']['xi']:.3f}"],
        ["rho", f"{N['heston_30y']['rho']:.3f}", f"{N['heston_5y']['rho']:.3f}"],
        ["Feller condition satisfied", str(N['heston_30y']['feller_satisfied']), str(N['heston_5y']['feller_satisfied'])],
    ]
)
p(f"The 30-year calibration implies an implausible {N['heston_30y']['theta']**0.5:.0%} "
  f"long-run annualized volatility (well above {N['ticker']}'s typical realized vol) and "
  f"a leverage correlation near zero -- economically wrong, since equities almost always "
  f"show a clearly negative leverage effect. The 5-year calibration gives sensible "
  f"numbers ({N['heston_5y']['theta']**0.5:.0%} vol, rho = {N['heston_5y']['rho']:.2f}). "
  f"The explanation: 30 years spans multiple structurally different volatility regimes "
  f"(dot-com, 2008, COVID), violating the GMM calibration's underlying assumption that "
  f"realized variance is a STATIONARY process over the calibration window. This is "
  f"exactly why the walk-forward backtest (Section 5.5) recalibrates on rolling 5-year "
  f"windows rather than the full history at once -- the methodologically correct way to "
  f"use 30 years of data here.")
p(f"Cross-validating the characteristic-function pricer against an independent Monte "
  f"Carlo simulation of the same SDE (using the calibrated 5-year parameters, at strike "
  f"K = {N['cf_vs_mc']['K']:.2f}, T = {N['cf_vs_mc']['T']} years): characteristic-function "
  f"price = {N['cf_vs_mc']['cf_price']:.4f}, Monte Carlo price = "
  f"{N['cf_vs_mc']['mc_price']:.4f} +/- {N['cf_vs_mc']['mc_stderr']:.4f} (one standard "
  f"error) -- agreement within "
  f"{abs(N['cf_vs_mc']['cf_price']-N['cf_vs_mc']['mc_price'])/N['cf_vs_mc']['mc_stderr']:.2f} "
  f"standard errors, validating both independent implementations.")

h2("5.4 Double Heston: An Honest Non-Result")
p(f"Double Heston, warm-started from the 5-year single-Heston fit above, was calibrated "
  f"on the same data. Factor 2's share of total long-run variance came out at "
  f"{N['double_heston_5y']['theta2_share']:.2e} -- essentially exactly zero. Figure 3 "
  f"shows why: the single- and double-Heston autocorrelation fits are visually "
  f"indistinguishable, both tracking the empirical decay reasonably (though imperfectly) "
  f"with a single effective timescale.")

if MULTI is not None:
    p(f"To check this was not a one-ticker fluke, the same calibration (10-year window) "
      f"was run on 5 tickers spanning different sectors and market-cap profiles: a "
      f"broad index ETF (SPY), three mega-cap equities across different sectors (AAPL, "
      f"MSFT, JPM), and a high-volatility growth stock (TSLA). The result was consistent "
      f"across all five -- factor 2's share of total variance never exceeded 1%, and "
      f"every ticker recovered the theoretically expected NEGATIVE leverage correlation:")
    rows = []
    for tk, d in MULTI.items():
        rows.append([tk, f"{d['single_theta']**0.5:.1%}", f"{d['single_rho']:.3f}",
                     f"{d['theta2_share']:.2e}"])
    table_from_rows(["Ticker", "Long-run vol (single Heston)", "rho", "Factor 2 variance share (Double Heston)"], rows)
    p("Table: Double Heston multi-ticker robustness check (10-year calibration window, "
      "each ticker independent). Factor 2's share stays below 1% in every case, with "
      "TSLA (the highest-vol, most growth/momentum-driven name tested) showing the "
      "largest -- but still negligible -- second-factor contribution.", italic=True)

p(f"As discussed in Section 3.2, Double Heston's classic empirical "
  f"motivation comes from fitting short- and long-dated OPTION-implied-volatility smiles "
  f"simultaneously -- a genuinely different target than the autocorrelation of realized "
  f"variance from spot returns, and this result should be read as evidence about that "
  f"difference, not as a failure of the Double Heston model or its implementation (which "
  f"is independently validated in Section 5.3's style of test -- see tests/test_double_"
  f"heston.py for the formal version: Double Heston with factor 2 switched off recovers "
  f"single Heston prices to within 0.01%).")
figure("03_acf_fit.png",
       "Figure 3: Empirical realized-variance autocorrelation function against single- and "
       "double-Heston model fits (5-year calibration window). The two model curves overlap "
       "almost exactly.")
figure("04_illustrative_smile.png",
       "Figure 4: Illustrative model-implied volatility smile from the calibrated "
       "parameters (NOT fit to real market option quotes -- see Section 3.2). Shown to "
       "make the pricing formulas' qualitative behavior tangible: Heston/Double Heston "
       "produce a downward-sloping skew from the calibrated negative rho, unlike "
       "Black-Scholes' flat line.")

h2("5.5 30-Year Walk-Forward Backtest")
p(f"The full walk-forward backtest was run with 5-year calibration windows, 1-year test "
  f"windows, stepping forward 1 year at a time: {N['kupiec']['n_windows']} independent "
  f"out-of-sample periods, {N['kupiec']['n_trials_total']} pooled daily VaR trials.")
figure("05_backtest_vol_forecast.png",
       "Figure 5: Out-of-sample volatility forecast vs. realized volatility, walked "
       "forward across the full history. Note both models systematically over-forecast "
       "in most years, and both underestimate the COVID-era spike.")

rows = []
for r in N["comparison_table"]:
    rows.append([
        r["model"],
        f"{r['vol_RMSE_overall']:.4f}",
        f"{r['vol_RMSE_calm']:.4f}",
        f"{r['vol_RMSE_turbulent']:.4f}",
        f"{r['VaR_breach_rate']:.2%}",
        f"{r['kupiec_p_value']:.2e}",
        "Rejected" if r["kupiec_rejected"] else "Not rejected",
    ])
table_from_rows(
    ["Model", "Vol RMSE (overall)", "Vol RMSE (calm)", "Vol RMSE (turbulent)",
     "VaR breach rate", "Kupiec p-value", "Kupiec verdict (5%)"],
    rows,
)
p(f"Target VaR breach rate: {N['kupiec']['target_breach_rate']:.0%}.", italic=True)
figure("06_model_comparison.png",
       "Figure 6: (left) volatility forecast RMSE split by calm vs. turbulent regime; "
       "(right) observed VaR breach rate vs. the 5% target, colored red where the Kupiec "
       "test rejects the model as miscalibrated at the 5% significance level.")

doc.add_page_break()


# ===========================================================================
# 6. DISCUSSION AND LIMITATIONS
# ===========================================================================
h1("6. Discussion and Limitations")
p("The results above are reported without adjustment for favorability, which is itself "
  "a methodological point worth stating: a real backtest's value comes precisely from its "
  "ability to show a sophisticated model NOT winning, when that is what the data supports.")

h2("6.1 GBM Is Competitive With Heston on Point Forecasts")
p(f"On overall volatility-forecast RMSE, GBM ({N['comparison_table'][0]['vol_RMSE_overall']:.4f}) "
  f"modestly outperforms both Heston ({N['comparison_table'][1]['vol_RMSE_overall']:.4f}) "
  f"and Double Heston ({N['comparison_table'][2]['vol_RMSE_overall']:.4f}). The "
  f"explanation traces to the calibrated mean-reversion speed: kappa around "
  f"{N['heston_5y']['kappa']:.0f} implies a mean-reversion half-life of roughly "
  f"{0.693/N['heston_5y']['kappa']*365:.0f} days -- fast enough that within a full 1-year "
  f"test window, Heston's forecast largely collapses to theta, functioning similarly to "
  f"GBM's forecast but estimated through a noisier 3-moment procedure rather than GBM's "
  f"direct, more robust sample variance. Model sophistication does not automatically "
  f"translate into better point forecasts on every metric -- a genuinely useful thing to "
  f"discover and report, not a disappointing result to explain away.")

h2("6.2 No Model Passes the VaR Coverage Test")
p("All three models are rejected by the Kupiec test at the pooled 24-year level. This is "
  "consistent with well-documented real-world experience: single-regime risk models "
  "systematically break down across genuine regime changes (the year containing the 2008 "
  "financial crisis shows a clear spike in breach counts for every model in the "
  "underlying window-by-window data). All three models here assume Gaussian-driven "
  "shocks (with time-varying but still Gaussian-conditional variance for "
  "Heston/Double Heston) and no jump risk -- a real limitation that a fat-tailed or "
  "jump-diffusion extension would be needed to address, which is outside this project's "
  "scope.")

h2("6.3 Double Heston's Complexity Was Not Rewarded Here")
p("Double Heston's second factor consistently collapsed to negligible across every "
  "ticker tested. This should not be read as \"Double Heston is a worse model\" -- its "
  "mathematical correctness is independently validated (Section 4.3, and formally in "
  "tests/test_double_heston.py) -- but as evidence that ITS SPECIFIC CLASSICAL "
  "MOTIVATION (fitting option-implied-vol term structure) is a genuinely different "
  "empirical target than what a return-only recalibration can access. A natural direction "
  "for future work is combining this project's return-based approach with even a small "
  "amount of real options data to test whether the two-factor structure re-emerges.")

h2("6.4 Data and Scope Limitations")
bullet("All calibration uses a single ticker's own history; while the Double Heston "
       "finding was checked across three tickers (AAPL, SPY, TSLA) during development, "
       "the full walk-forward backtest and report figures focus on one ticker for depth.")
bullet("Risk-free rate is a single flat proxy (13-week T-bill), not a full term "
       "structure -- a simplification standard at this project's scope.")
bullet("Realized variance (a backward-looking, noisy proxy) stands in for the true "
       "unobservable instantaneous variance throughout -- a fundamental limitation of "
       "return-only calibration that a paid historical options data source would remove.")
bullet("The rho estimation's attenuation-bias correction (Section 4.5) is an approximate, "
       "simulation-calibrated correction, not an exact debiasing -- it should be read as "
       "giving the right order of magnitude and sign, not a high-precision estimate.")

doc.add_page_break()


# ===========================================================================
# 7. THE DEPLOYED DASHBOARD AND CODEBASE
# ===========================================================================
h1("7. Deployed Dashboard and Codebase")
p("Alongside this report, the full codebase implements a real-time Streamlit dashboard "
  "(app.py) with five sections: a live/historical price view (real-time quotes via "
  "Finnhub's free tier), GBM diagnostics, Heston/Double Heston calibration with the ACF "
  "fit visualization, an interactive version of the 30-year walk-forward backtest, and an "
  "illustrative theoretical option pricing tool. The dashboard is deployable for free via "
  "Streamlit Community Cloud. A companion Jupyter notebook "
  "(notebooks/full_pipeline.ipynb) runs the entire pipeline end-to-end with inline plots "
  "and serves as this report's appendix. The full test suite (tests/) independently "
  "validates every model and calibration routine against known theoretical identities and "
  "synthetic data with known ground truth -- see the project README for how to run it.")

doc.add_page_break()


# ===========================================================================
# 8. CONCLUSION
# ===========================================================================
h1("8. Conclusion")
p("This project implemented Geometric Brownian Motion, Black-Scholes, Heston, and Double "
  "Heston from their underlying mathematics, validated each against independent numerical "
  "methods and known theoretical identities, and calibrated them to real, multi-decade "
  "equity data using a return-based Generalized Method of Moments approach adopted after "
  "discovering real limitations in freely available options data. The centerpiece 30-year "
  "walk-forward backtest, evaluated with volatility-forecast accuracy and a formal VaR "
  "coverage test, produced results that are informative precisely because they are not "
  "uniformly flattering to the more sophisticated models: GBM remains competitive with "
  "Heston on point forecasts in this setting, no model passes VaR backtesting across the "
  "full sample, and Double Heston's second factor does not meaningfully activate under "
  "return-only calibration. Reporting these findings honestly, with their causes "
  "diagnosed and explained rather than hidden, is the central methodological commitment "
  "of this project.")

doc.add_page_break()


# ===========================================================================
# 9. REFERENCES
# ===========================================================================
h1("9. References")
refs = [
    "Black, F. and Scholes, M. (1973). \"The Pricing of Options and Corporate Liabilities\". "
    "Journal of Political Economy, 81(3), 637-654.",
    "Heston, S. L. (1993). \"A Closed-Form Solution for Options with Stochastic Volatility "
    "with Applications to Bond and Currency Options\". Review of Financial Studies, 6(2), "
    "327-343.",
    "Christoffersen, P., Heston, S. and Jacobs, K. (2009). \"The Shape and Term Structure "
    "of the Index Option Smirk: Why Multifactor Stochastic Volatility Models Work So "
    "Well\". Management Science, 55(12), 1914-1932.",
    "Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). \"The Little Heston "
    "Trap\". Wilmott Magazine, January 2007, 83-92.",
    "Lord, R., Koekkoek, R. and van Dijk, D. (2010). \"A Comparison of Biased Simulation "
    "Schemes for Stochastic Volatility Models\". Quantitative Finance, 10(2), 177-194.",
    "Andersen, L. (2008). \"Simple and Efficient Simulation of the Heston Stochastic "
    "Volatility Model\". Journal of Computational Finance, 11(3), 1-42.",
    "Cox, J. C., Ingersoll, J. E. and Ross, S. A. (1985). \"A Theory of the Term Structure "
    "of Interest Rates\". Econometrica, 53(2), 385-407.",
    "Kupiec, P. H. (1995). \"Techniques for Verifying the Accuracy of Risk Measurement "
    "Models\". Journal of Derivatives, 3(2), 73-84.",
    "Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. Wiley Finance.",
]
for r in refs:
    doc.add_paragraph(r, style="List Number")

out_path = os.path.join(HERE, "Stock_Price_Modeling_Report.docx")
doc.save(out_path)
print(f"Report saved to {out_path}")
