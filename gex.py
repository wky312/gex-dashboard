"""
GEX (Gamma Exposure) calculation engine.
Vectorized Black-Scholes Greeks + dealer gamma aggregation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy.stats import norm
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Risk-free rate from FRED ──────────────────────────────────────────────

_RF_CACHE = {}

def get_risk_free_rate() -> float:
    if 'rate' in _RF_CACHE:
        ts, val = _RF_CACHE['rate']
        if (datetime.now(timezone.utc) - ts).total_seconds() < 3600:
            return val
    try:
        url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO&cosd=2025-01-01'
        text = requests.get(url, timeout=5).text
        lines = [l for l in text.strip().split('\n')[1:] if '.' in l.split(',')[-1]]
        val = float(lines[-1].split(',')[1]) / 100
        _RF_CACHE['rate'] = (datetime.now(timezone.utc), val)
        return val
    except Exception:
        return 0.043  # fallback


# ── Vectorized Black-Scholes Greeks ───────────────────────────────────────

def bs_greeks(S: float, K: np.ndarray, T: np.ndarray, r: float,
              sigma: np.ndarray, is_call: np.ndarray) -> dict:
    """Compute delta and gamma for arrays of options (vectorized)."""
    T = np.maximum(T, 1e-6)
    sigma = np.maximum(sigma, 1e-6)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    delta = np.where(is_call, call_delta, put_delta)

    return {'delta': delta, 'gamma': gamma}


# ── Fetch + compute GEX for one symbol ────────────────────────────────────

def fetch_gex(symbol: str, r: float = None, max_expirations: int = 12) -> dict:
    """
    Fetch options chain and compute dealer GEX for a symbol.

    Returns dict with:
      - spot: current price
      - gex_by_strike: DataFrame with strike, call_gex, put_gex, net_gex
      - gex_by_expiry: DataFrame with expiry, net_gex
      - total_gex: float
      - gamma_flip: float or None
      - key_levels: dict with max_gamma_strike, gamma_flip, put_wall, call_wall
      - regime: 'positive' | 'negative' | 'neutral'
      - timestamp: str
    """
    if r is None:
        r = get_risk_free_rate()

    ticker = yf.Ticker(symbol)
    spot = ticker.info.get('regularMarketPrice') or ticker.fast_info.get('lastPrice')
    if not spot:
        hist = ticker.history(period='1d')
        spot = float(hist['Close'].iloc[-1])

    expirations = list(ticker.options)
    if not expirations:
        raise ValueError(f"No options data for {symbol}")

    # Limit expirations to keep it fast
    expirations = expirations[:max_expirations]

    now = datetime.now(timezone.utc)
    all_rows = []

    for exp_str in expirations:
        try:
            chain = ticker.option_chain(exp_str)
        except Exception:
            continue

        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        T = max((exp_date - now).total_seconds() / (365.25 * 86400), 1e-6)

        for side, df in [('call', chain.calls), ('put', chain.puts)]:
            if df.empty:
                continue
            mask = (df['openInterest'] > 0) & (df['impliedVolatility'] > 0.001)
            df = df[mask].copy()
            if df.empty:
                continue

            K = df['strike'].values.astype(float)
            sigma = df['impliedVolatility'].values.astype(float)
            oi = df['openInterest'].values.astype(float)
            is_call = np.ones(len(K), dtype=bool) if side == 'call' else np.zeros(len(K), dtype=bool)
            T_arr = np.full(len(K), T)

            greeks = bs_greeks(spot, K, T_arr, r, sigma, is_call)

            # Dealer GEX (per contract = 100 shares)
            # Dealers are short calls (negative gamma) and short puts (positive gamma)
            # GEX_dealer = OI * Gamma * 100 * S^2 * 0.01
            raw_gex = oi * greeks['gamma'] * 100 * spot**2 * 0.01
            if side == 'call':
                dealer_gex = -raw_gex  # dealers short calls → negative gamma
            else:
                dealer_gex = raw_gex   # dealers short puts → positive gamma

            for i in range(len(K)):
                all_rows.append({
                    'strike': K[i],
                    'expiry': exp_str,
                    'T': T,
                    'side': side,
                    'oi': oi[i],
                    'iv': sigma[i],
                    'delta': greeks['delta'][i],
                    'gamma': greeks['gamma'][i],
                    'dealer_gex': dealer_gex[i],
                })

    if not all_rows:
        raise ValueError(f"No valid options data for {symbol}")

    df = pd.DataFrame(all_rows)

    # ── Aggregate by strike ──
    call_gex = df[df['side'] == 'call'].groupby('strike')['dealer_gex'].sum().rename('call_gex')
    put_gex = df[df['side'] == 'put'].groupby('strike')['dealer_gex'].sum().rename('put_gex')
    total_oi = df.groupby('strike')['oi'].sum().rename('total_oi')
    gex_by_strike = pd.concat([call_gex, put_gex, total_oi], axis=1).fillna(0)
    gex_by_strike['net_gex'] = gex_by_strike['call_gex'] + gex_by_strike['put_gex']
    gex_by_strike = gex_by_strike.reset_index()

    # Filter to relevant strike range (within ±15% of spot)
    strike_lo = spot * 0.85
    strike_hi = spot * 1.15
    gex_by_strike = gex_by_strike[
        (gex_by_strike['strike'] >= strike_lo) &
        (gex_by_strike['strike'] <= strike_hi)
    ].sort_values('strike').reset_index(drop=True)

    # ── Aggregate by expiry ──
    gex_by_expiry = df.groupby('expiry').agg(
        net_gex=('dealer_gex', 'sum'),
        total_oi=('oi', 'sum'),
    ).reset_index().sort_values('expiry')

    # ── Key levels ──
    total_gex = gex_by_strike['net_gex'].sum()

    # Max gamma strike (absolute)
    max_idx = gex_by_strike['net_gex'].abs().idxmax()
    max_gamma_strike = gex_by_strike.loc[max_idx, 'strike']

    # Put wall (largest negative GEX strike = biggest put gamma concentration)
    put_data = gex_by_strike[gex_by_strike['put_gex'] > 0]
    put_wall = put_data.loc[put_data['put_gex'].idxmax(), 'strike'] if not put_data.empty else None

    # Call wall (largest positive call gamma = most negative dealer call GEX...
    # actually call wall = strike with most call OI gamma, acts as resistance)
    call_data = gex_by_strike[gex_by_strike['call_gex'] < 0]
    call_wall = call_data.loc[call_data['call_gex'].abs().idxmax(), 'strike'] if not call_data.empty else None

    # ── Gamma flip (linear interpolation of zero crossing) ──
    gamma_flip = find_gamma_flip(gex_by_strike['strike'].values,
                                  gex_by_strike['net_gex'].values, spot)

    # ── Regime ──
    if total_gex > 0:
        regime = 'positive'
    elif total_gex < 0:
        regime = 'negative'
    else:
        regime = 'neutral'

    return {
        'symbol': symbol,
        'spot': spot,
        'gex_by_strike': gex_by_strike,
        'gex_by_expiry': gex_by_expiry,
        'raw_options': df,
        'total_gex': total_gex,
        'gamma_flip': gamma_flip,
        'key_levels': {
            'max_gamma_strike': max_gamma_strike,
            'gamma_flip': gamma_flip,
            'put_wall': put_wall,
            'call_wall': call_wall,
        },
        'regime': regime,
        'risk_free_rate': r,
        'timestamp': now.isoformat(),
    }


def find_gamma_flip(strikes: np.ndarray, net_gex: np.ndarray,
                    spot: float) -> float | None:
    """Find gamma flip level via linear interpolation of zero crossings.
    If multiple crossings exist, return the one closest to spot."""
    flips = []
    for i in range(len(net_gex) - 1):
        if net_gex[i] * net_gex[i + 1] < 0:  # sign change
            # Linear interpolation
            x0, x1 = strikes[i], strikes[i + 1]
            y0, y1 = net_gex[i], net_gex[i + 1]
            flip = x0 - y0 * (x1 - x0) / (y1 - y0)
            flips.append(flip)
    if not flips:
        return None
    # Return flip closest to spot
    return min(flips, key=lambda f: abs(f - spot))


# ── Multi-symbol parallel fetch ───────────────────────────────────────────

def fetch_multi_gex(symbols: list[str], max_expirations: int = 12) -> dict:
    """Fetch GEX for multiple symbols in parallel."""
    r = get_risk_free_rate()
    results = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fetch_gex, sym, r, max_expirations): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                results[sym] = future.result()
            except Exception as e:
                results[sym] = {'error': str(e), 'symbol': sym}

    return results
