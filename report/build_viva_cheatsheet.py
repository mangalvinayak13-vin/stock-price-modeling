"""
report/build_viva_cheatsheet.py
==================================
Builds a compact, quick-reference Word document (report/Viva_Cheat_Sheet.docx)
for oral exam prep: every formula, every calibrated number, and the "why"
behind every judgment call, condensed to what you'd actually want to glance
at 5 minutes before walking in. Run report/generate_figures.py first (this
reads report/numbers.json and report/multi_ticker_check.json).

Not part of the core deliverable.
"""
import json
import os
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "numbers.json")) as f:
    N = json.load(f)
MULTI = json.load(open(os.path.join(HERE, "multi_ticker_check.json"))) \
    if os.path.exists(os.path.join(HERE, "multi_ticker_check.json")) else None

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)


def h1(t): doc.add_heading(t, level=1)
def h2(t): doc.add_heading(t, level=2)
def p(t, bold=False):
    para = doc.add_paragraph()
    r = para.add_run(t)
    r.bold = bold
    return para
def bullet(t): doc.add_paragraph(t, style="List Bullet")
def qa(question, answer):
    para = doc.add_paragraph()
    r1 = para.add_run("Q: " + question + "\n")
    r1.bold = True
    r2 = para.add_run("A: " + answer)


title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Viva Cheat Sheet")
r.bold = True
r.font.size = Pt(20)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run("Stock Price Modeling: GBM -> Black-Scholes -> Heston -> Double Heston").italic = True
doc.add_paragraph()

# ---------------------------------------------------------------------------
h1("1. Formulas at a Glance")

h2("GBM")
p("SDE:  dS = mu*S*dt + sigma*S*dW", bold=True)
p("Exact solution:  S(t) = S(0) * exp[(mu - sigma^2/2)t + sigma*W(t)]")
p("Why exact solution, not Euler: zero discretization error at simulated points -- the "
  "only randomness is genuine Brownian sampling, not numerical approximation.")

h2("Black-Scholes")
p("Call = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)")
p("d1 = [ln(S/K) + (r-q+sigma^2/2)T] / (sigma*sqrt(T)),   d2 = d1 - sigma*sqrt(T)")
p("Put-call parity: C - P = S*e^(-qT) - K*e^(-rT)  (verified to machine precision in tests)")

h2("Heston (1993)")
p("dS = (r-q)*S*dt + sqrt(v)*S*dW1")
p("dv = kappa*(theta-v)*dt + xi*sqrt(v)*dW2,   Corr(dW1,dW2) = rho")
p("Feller condition: 2*kappa*theta >= xi^2  (keeps v > 0 always; often violated by real "
  "calibrated equity data -- model still valid, but naive Euler MC breaks, hence Full "
  "Truncation Euler)")
p("Price via P1/P2 (Gil-Pelaez): Call = S*e^(-qT)*P1 - K*e^(-rT)*P2, each P_j a Fourier "
  "inversion integral of the (little-trap) characteristic function.")

h2("Double Heston (Christoffersen-Heston-Jacobs 2009)")
p("Two independent CIR factors v1, v2; total variance = v1+v2.")
p("Characteristic function factorizes: phi(u) = e^(iu(lnS0+(r-q)T)) * f1(u) * f2(u)")
p("(f_i is the single-Heston little-trap building block using factor i's own params)")

doc.add_page_break()

# ---------------------------------------------------------------------------
h1("2. Calibrated Numbers (This Report's Run)")

h2(f"GBM ({N['ticker']}, {N['history_years']}y)")
p(f"mu = {N['gbm_mu']:.4f}    sigma = {N['gbm_sigma']:.4f}")

h2(f"Heston, 5-year window (the CORRECT one to quote)")
h5 = N["heston_5y"]
p(f"kappa={h5['kappa']:.2f}  theta={h5['theta']:.4f} (vol={h5['theta']**0.5:.1%})  "
  f"xi={h5['xi']:.2f}  rho={h5['rho']:.2f}  v0={h5['v0']:.4f}  "
  f"Feller satisfied: {h5['feller_satisfied']}")

h2("Heston, 30-year window (deliberately shown to be WRONG -- know why)")
h30 = N["heston_30y"]
p(f"theta={h30['theta']:.4f} (implies {h30['theta']**0.5:.0%} vol -- too high), "
  f"rho={h30['rho']:.3f} (should be clearly negative, isn't)")
p("WHY: 30y spans dot-com + 2008 GFC + COVID -- non-stationary, violates the GMM "
  "calibration's stationarity assumption. This is WHY the backtest uses rolling 5y "
  "windows, not one static fit.")

h2("Double Heston (5-year window)")
dh = N["double_heston_5y"]
p(f"Factor 1: kappa1={dh['kappa1']:.2f} theta1={dh['theta1']:.4f} xi1={dh['xi1']:.2f}")
p(f"Factor 2: kappa2={dh['kappa2']:.2e} theta2={dh['theta2']:.2e} xi2={dh['xi2']:.2e}  "
  f"(essentially zero -- factor 2 share = {dh['theta2_share']:.2e})")

if MULTI:
    h2("Multi-ticker check (Double Heston factor 2 share, all negligible)")
    line = "  |  ".join(f"{tk}: {d['theta2_share']:.1e}" for tk, d in MULTI.items())
    p(line)

h2("CF vs Monte Carlo cross-validation")
cv = N["cf_vs_mc"]
p(f"K={cv['K']:.1f}, T={cv['T']}y: CF price={cv['cf_price']:.4f}, "
  f"MC price={cv['mc_price']:.4f} +/- {cv['mc_stderr']:.4f} "
  f"(agree within {abs(cv['cf_price']-cv['mc_price'])/cv['mc_stderr']:.2f} std errors)")

h2("30-Year Walk-Forward Backtest Headline Numbers")
for row in N["comparison_table"]:
    p(f"{row['model']}: vol RMSE={row['vol_RMSE_overall']:.4f} "
      f"(calm={row['vol_RMSE_calm']:.4f}, turbulent={row['vol_RMSE_turbulent']:.4f})  |  "
      f"VaR breach rate={row['VaR_breach_rate']:.2%} (target 5%), "
      f"Kupiec p={row['kupiec_p_value']:.1e} -> "
      f"{'REJECTED' if row['kupiec_rejected'] else 'not rejected'}")
p(f"n_windows={N['kupiec']['n_windows']}, pooled VaR trials={N['kupiec']['n_trials_total']}")

doc.add_page_break()

# ---------------------------------------------------------------------------
h1("3. Every Judgment Call, One Line Each")
bullet("252 trading days/year annualization (standard equity convention, not 365).")
bullet("Newton-Raphson + bisection fallback for implied vol -- Newton fails when vega~0 "
       "(deep ITM/OTM, near-expiry); bisection guaranteed to converge since price is "
       "monotone in sigma.")
bullet("Near-zero-vega implied vol is mathematically UNIDENTIFIABLE from price, not just "
       "hard to estimate -- many sigmas reprice identically to machine precision.")
bullet("Little-trap Heston CF (Albrecher et al. 2007): avoids branch-cut discontinuity in "
       "the original 1993 formula for long maturities. Uses g=(c1-d)/(c1+d), NOT its "
       "reciprocal.")
bullet("u_max=200 truncation for the P1/P2 Fourier integral: verified price is insensitive "
       "to pushing this further (checked against u_max=500).")
bullet("Full Truncation Euler (not naive Euler, not Andersen QE) for Heston/Double Heston "
       "MC: naive Euler breaks when v goes negative (common when Feller violated); QE is "
       "more accurate but substantially more complex to implement correctly. Chose the "
       "middle ground.")
bullet("Pivoted from options-market calibration to equity-return-based GMM calibration: "
       "yfinance option chains have near-zero real bid/ask/OI coverage and NEVER provide "
       "historical chains -- infeasible for a 30-year backtest without paid data.")
bullet("Realized variance (21-day rolling window) proxies the unobservable v(t).")
bullet("kappa identified from EXACT CIR property: ACF(v,lag) = exp(-kappa*lag). Fit via "
       "least squares across MANY lags (over-identified, robust to single-lag noise).")
bullet("rho estimated from Corr(r_t, d(v1+v2)) -- derived via Ito's isometry to leading "
       "order in dt: Corr(dr,dv) = rho EXACTLY to leading order. Two bugs found fixing "
       "this (see below).")
bullet("BUG 1 (sign error): naive trailing-window RV overlap contaminated the correlation "
       "-- fixed using non-overlapping forward/backward windows.")
bullet("BUG 2 (attenuation bias): daily squared returns are noisy (chi-sq) proxies for "
       "variance, diluting the correlation to ~15-20% of true magnitude -- fixed via a "
       "simulation-calibrated correction factor using the project's own validated MC engine.")
bullet("Double Heston: rho1=rho2 (can't separately identify 2 leverage correlations from "
       "1 aggregate moment); v1_0,v2_0 split by each factor's long-run variance share; "
       "kappa1>=kappa2 ordering enforced to break label-switching symmetry.")
bullet("Multi-start Levenberg-Marquardt (not full global optimizer): the return-based "
       "objective is low-dimensional (3-6 params) and smooth (no simulation noise) -- much "
       "better-behaved than the original options-price objective would have been.")
bullet("Walk-forward: 5y calibration / 1y test / 1y step -- balances stationarity (short "
       "enough window) against statistical power (~24 independent regimes over 30y).")
bullet("VaR held fixed within each test window (periodic, not continuous, recalibration) "
       "-- matches real-world practice and keeps the backtest tractable.")
bullet("Kupiec test: the SAME likelihood-ratio coverage test used in Basel banking "
       "regulation to validate internal VaR models.")

doc.add_page_break()

# ---------------------------------------------------------------------------
h1("4. Anticipated Questions and Answers")

qa("Why not just use options data like everyone else?",
   "Tried it first. yfinance's free option-chain feed has near-zero real bid/ask/open-"
   "interest coverage (verified empirically), and never exposes HISTORICAL chains -- only "
   "today's snapshot. A 30-year options-based backtest needs a paid vendor "
   "(OptionMetrics/ORATS), out of scope. Pivoted to return-based GMM calibration, which "
   "is a legitimate, citable alternative methodology.")

qa("Why does the 30-year Heston calibration give a weird 49% volatility number?",
   "Non-stationarity: 30 years spans dot-com, 2008 GFC, and COVID -- structurally "
   "different regimes that violate the calibration's stationary-process assumption. "
   "5-10 year windows give sensible numbers. This is exactly why the backtest uses "
   "rolling 5-year windows, not one static fit.")

qa("Why does Double Heston's second factor disappear?",
   "Checked across 5 tickers (AAPL, SPY, TSLA, MSFT, JPM) -- consistently negligible "
   "(<1% variance share). Double Heston's classic advantage is fitting SHORT and LONG "
   "option-maturity implied-vol smiles simultaneously; the autocorrelation of realized "
   "variance from spot returns doesn't show that same two-timescale signature. Honest "
   "finding about the methodology difference, not a bug -- verified independently that "
   "the model collapses EXACTLY to single Heston when factor 2 is switched off by hand.")

qa("How do you know your Heston pricer is actually correct?",
   "Two independent checks: (1) as xi->0 with v0=theta, Heston must equal Black-Scholes "
   "exactly -- verified to 1e-9 relative error. (2) The characteristic-function price "
   "agrees with an independent Monte Carlo simulation of the same SDE within Monte Carlo "
   "sampling error, including under Feller-violating parameters that break naive Euler.")

qa("Why Full Truncation Euler instead of the Andersen QE scheme?",
   "QE is more accurate for large time steps and is the industry standard, but "
   "substantially more complex to implement correctly (switches between two sampling "
   "regimes based on a threshold test). Full Truncation Euler still correctly handles "
   "negative variance (unlike naive Euler) and its bias is small at daily time steps -- "
   "a deliberate complexity/accuracy tradeoff, stated explicitly rather than hidden.")

qa("Why did GBM beat Heston on volatility forecasting in your backtest?",
   "The calibrated Heston mean-reversion speed (kappa ~12-14/year) is fast -- half-life "
   "of ~2-4 weeks. Within a full 1-year test window, Heston's forecast collapses close to "
   "theta almost immediately, similar in spirit to GBM's forecast but estimated through a "
   "noisier 3-moment GMM procedure instead of GBM's direct, more robust sample variance. "
   "Model sophistication doesn't automatically win on every metric -- reported honestly.")

qa("Why does no model pass the VaR coverage test?",
   "All three assume Gaussian-driven shocks (Heston/Double Heston have time-varying but "
   "still conditionally Gaussian variance) with no jump risk. Real markets have jump risk "
   "(earnings surprises, macro shocks) these models don't capture. Also matches "
   "well-documented real-world experience: single-regime VaR models break down across "
   "genuine crises (breach counts spike specifically in the 2008 window for every model).")

qa("What's the Feller condition and why is it violated?",
   "2*kappa*theta >= xi^2 guarantees the CIR variance process never hits exactly zero. "
   "Real calibrated equity data typically has more vol-of-vol (xi) than this allows -- "
   "the model is still mathematically valid (CIR is well-defined even at the boundary), "
   "but naive Euler MC breaks because it can push discretized variance negative. That's "
   "exactly why Full Truncation Euler is necessary, not optional.")

qa("What would you do differently with more time/budget?",
   "Get access to real historical options data (paid vendor) to do the classical "
   "options-implied calibration and directly test whether Double Heston's two-factor "
   "structure re-emerges there. Also consider a particle-filter/Kalman-filter MLE "
   "approach for parameter estimation, which is more statistically efficient than "
   "moment-matching but substantially more complex to implement correctly.")

out_path = os.path.join(HERE, "Viva_Cheat_Sheet.docx")
doc.save(out_path)
print(f"Saved {out_path}")
