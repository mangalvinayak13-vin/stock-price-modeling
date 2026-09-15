"""
data/realtime.py
==================
Real-time (or near-real-time) stock quotes for the deployed website, using
Finnhub's free tier (https://finnhub.io).

WHY FINNHUB AND NOT yfinance FOR THE LIVE WEBSITE:
`yfinance` scrapes an undocumented, unofficial Yahoo endpoint. It has no
SLA, can rate-limit or block an IP with no warning, and Yahoo has changed
its response format multiple times in ways that broke yfinance overnight.
That's an acceptable risk for a research notebook you re-run yourself, but
not for a public website other people will load. Finnhub's free tier gives
a documented, supported REST API (60 requests/minute) with a real quote
endpoint, which is enough for a dashboard that refreshes every few seconds
for a handful of tickers -- the actual load pattern of a student project
demo site.

SETUP: sign up for a free API key at https://finnhub.io/register, then set
it as an environment variable:
    export FINNHUB_API_KEY="your_key_here"
(or put it in a `.env` file in the project root -- see README).

JUDGMENT CALL: "real-time" on Finnhub's free tier means live trade data for
US exchanges (not delayed 15-20 min like many free feeds) but is rate
limited. For a dashboard showing one ticker at a time with a few-second
refresh, this is well within the free tier's 60 calls/minute limit. If this
project needs to show many tickers simultaneously or sub-second updates,
that requires a paid tier -- flagged here rather than silently degrading.
"""

import os
import time
import requests
from dotenv import load_dotenv

# Load FINNHUB_API_KEY (and any other secrets) from a .env file in the
# project root if one exists, so the key doesn't need to be exported in
# every shell session. Safe no-op if no .env file is present.
load_dotenv()

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


def _get_api_key():
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError(
            "FINNHUB_API_KEY environment variable is not set. "
            "Get a free key at https://finnhub.io/register and run:\n"
            "    export FINNHUB_API_KEY='your_key_here'\n"
            "(See README.md for details.)"
        )
    return key


def get_live_quote(ticker, timeout=5):
    """
    Fetch the current real-time quote for `ticker` from Finnhub's /quote
    endpoint.

    Returns
    -------
    dict with keys:
        'price'          : float, current price (last trade)
        'change'         : float, change from previous close
        'percent_change' : float, percent change from previous close
        'high'           : float, day's high
        'low'            : float, day's low
        'open'           : float, day's open
        'prev_close'     : float, previous close
        'timestamp'      : int, unix timestamp of the quote

    Raises
    ------
    RuntimeError if the API key is missing or the request fails.
    """
    key = _get_api_key()
    resp = requests.get(
        f"{FINNHUB_BASE_URL}/quote",
        params={"symbol": ticker, "token": key},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    # Finnhub returns all-zero fields for an invalid/unrecognized symbol
    # rather than an HTTP error -- check for that explicitly.
    if data.get("c", 0) == 0 and data.get("pc", 0) == 0:
        raise ValueError(f"Finnhub returned no data for ticker '{ticker}' "
                          "(symbol may be invalid, or free-tier coverage doesn't include it).")

    return {
        "price": data["c"],
        "change": data["d"],
        "percent_change": data["dp"],
        "high": data["h"],
        "low": data["l"],
        "open": data["o"],
        "prev_close": data["pc"],
        "timestamp": data["t"],
    }


def poll_live_quote(ticker, interval_seconds=5, n_updates=None, callback=None):
    """
    Simple polling loop for a live-updating dashboard: repeatedly calls
    get_live_quote() every `interval_seconds`, respecting Finnhub's free
    60 calls/minute rate limit by construction (interval_seconds >= 1
    keeps a single ticker well under that limit).

    Parameters
    ----------
    interval_seconds : float, seconds between polls (>=1 recommended)
    n_updates : int or None, stop after this many polls (None = forever;
                use a finite number for scripts/tests, None inside the
                Streamlit app which manages its own refresh loop)
    callback : callable(dict) or None, called with each quote dict; if
               None, the quote is just returned as the last value

    Returns
    -------
    The last quote dict fetched (useful when n_updates is finite).

    NOTE: for the actual Streamlit app we do NOT use this blocking loop --
    Streamlit has its own rerun/auto-refresh mechanism (st.autorefresh /
    a fragment on a timer) that calls get_live_quote() once per rerun
    instead. This function is provided for scripts/tests and for any
    non-Streamlit use of the data layer.
    """
    count = 0
    quote = None
    while n_updates is None or count < n_updates:
        quote = get_live_quote(ticker)
        if callback is not None:
            callback(quote)
        count += 1
        if n_updates is None or count < n_updates:
            time.sleep(interval_seconds)
    return quote
