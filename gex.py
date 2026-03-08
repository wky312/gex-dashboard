"""
GEX (Gamma Exposure) calculation engine.
Vectorized Black-Scholes Greeks + dealer gamma aggregation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from scipy.stats import norm
from scipy.optimize import brentq
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

    # ── Implied move (1σ) per expiry ──
    implied_moves = []
    for exp_str in expirations:
        exp_rows = df[df['expiry'] == exp_str]
        if exp_rows.empty:
            continue
        T_val = exp_rows['T'].iloc[0]
        # Use ATM IV as proxy (nearest strike to spot)
        atm = exp_rows.iloc[(exp_rows['strike'] - spot).abs().argsort()[:4]]
        avg_iv = atm['iv'].mean()
        move_1s = spot * avg_iv * np.sqrt(T_val)
        dte = int(round(T_val * 365.25))
        total_exp_gex = exp_rows['dealer_gex'].sum()
        implied_moves.append({
            'expiry': exp_str,
            'dte': dte,
            'iv': avg_iv,
            'move_1s': move_1s,
            'upper': spot + move_1s,
            'lower': spot - move_1s,
            'net_gex': total_exp_gex,
        })
    implied_moves_df = pd.DataFrame(implied_moves)

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
        'implied_moves': implied_moves_df,
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


# ── Black-Scholes price (for implied vol extraction) ─────────────────────

def bs_price(S, K, T, r, sigma, is_call):
    """Scalar Black-Scholes price."""
    T = max(T, 1e-6)
    sigma = max(sigma, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, is_call):
    """Extract implied volatility from option price using Brent's method."""
    if price <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
    if price < intrinsic:
        return None
    try:
        iv = brentq(lambda sigma: bs_price(S, K, T, r, sigma, is_call) - price,
                     0.01, 5.0, xtol=1e-6, maxiter=100)
        return iv
    except (ValueError, RuntimeError):
        return None


# ── TAIFEX (Taiwan) options fetcher ──────────────────────────────────────

_TW_RF_RATE = 0.018  # Taiwan 3-month rate (~1.8%)
_TXO_MULTIPLIER = 50  # TXO contract multiplier is 50 TWD per point

def _parse_taifex_date(date_str: str) -> str:
    """Convert TAIFEX date format (20260318) to YYYY-MM-DD."""
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def fetch_gex_taifex(query_date: str = None, max_expirations: int = 8) -> dict:
    """
    Fetch TXO options data from TAIFEX and compute dealer GEX.

    Args:
        query_date: Date to query in YYYY/MM/DD format. Defaults to today (Taiwan time).
        max_expirations: Max number of expirations to include.

    Returns same structure as fetch_gex().
    """
    from datetime import timedelta

    # Fetch TAIEX spot price from Yahoo Finance
    ticker = yf.Ticker('^TWII')
    try:
        spot = ticker.info.get('regularMarketPrice') or ticker.fast_info.get('lastPrice')
        if not spot:
            hist = ticker.history(period='1d')
            spot = float(hist['Close'].iloc[-1])
    except Exception:
        spot = None

    # Fetch options data from TAIFEX
    url = 'https://www.taifex.com.tw/cht/3/optDailyMarketReport'
    hdrs = {'User-Agent': 'Mozilla/5.0'}
    rows = None

    if query_date is None:
        # Try recent business days (today backwards)
        tw_now = datetime.now(timezone.utc) + timedelta(hours=8)
        for offset in range(5):
            d = tw_now - timedelta(days=offset)
            query_date = d.strftime('%Y/%m/%d')
            resp = requests.post(url, data={
                'queryType': '2', 'marketCode': '0',
                'commodity_id': 'TXO', 'queryDate': query_date,
            }, headers=hdrs, timeout=15)
            soup = BeautifulSoup(resp.text, 'html.parser')
            tables = soup.find_all('table')
            if tables and len(tables[0].find_all('tr')) > 10:
                rows = tables[0].find_all('tr')
                break
    else:
        resp = requests.post(url, data={
            'queryType': '2', 'marketCode': '0',
            'commodity_id': 'TXO', 'queryDate': query_date,
        }, headers=hdrs, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        tables = soup.find_all('table')
        if tables:
            rows = tables[0].find_all('tr')

    if not rows or len(rows) < 2:
        raise ValueError("No TAIFEX options data found")

    # Parse all rows
    raw_data = []
    for row in rows[1:]:
        cells = row.find_all(['th', 'td'])
        if len(cells) < 16:
            continue
        texts = [c.get_text(strip=True) for c in cells]

        exp_code = texts[1]   # e.g. '202603', '202603F1', '202603W2'
        exp_date = texts[2]   # e.g. '20260318'
        strike_str = texts[3]
        side_str = texts[4]   # 'Call' or 'Put'
        settle_str = texts[9]  # 結算價
        last_str = texts[8]   # 最後成交價
        oi_str = texts[15]    # 未沖銷契約量

        # Skip empty rows
        if not exp_code or not strike_str.replace(',', '').replace('.', '').isdigit():
            continue

        # Parse OI
        oi_clean = oi_str.replace(',', '')
        if oi_clean in ['-', ''] or not oi_clean.isdigit():
            continue
        oi = int(oi_clean)
        if oi == 0:
            continue

        # Parse strike
        strike = float(strike_str.replace(',', ''))

        # Parse price (prefer settlement, fallback to last)
        price = None
        for p_str in [settle_str, last_str]:
            p_clean = p_str.replace(',', '')
            if p_clean not in ['-', ''] and p_clean.replace('.', '').isdigit():
                price = float(p_clean)
                break

        if price is None or price <= 0:
            continue

        is_call = side_str.strip().lower() == 'call'

        raw_data.append({
            'exp_code': exp_code,
            'exp_date': exp_date,
            'strike': strike,
            'is_call': is_call,
            'side': 'call' if is_call else 'put',
            'price': price,
            'oi': oi,
        })

    if not raw_data:
        raise ValueError("No valid TXO options data found")

    raw_df = pd.DataFrame(raw_data)

    # If we couldn't get spot from Yahoo, estimate from put-call parity at ATM
    if spot is None:
        # Use the monthly expiry with most data
        main_exp = raw_df.groupby('exp_code').size().idxmax()
        exp_data = raw_df[raw_df['exp_code'] == main_exp]
        # Find strike where call and put prices are closest
        strikes_with_both = set(
            exp_data[exp_data['side'] == 'call']['strike']
        ) & set(
            exp_data[exp_data['side'] == 'put']['strike']
        )
        if strikes_with_both:
            min_diff = float('inf')
            for s in strikes_with_both:
                c = exp_data[(exp_data['strike'] == s) & (exp_data['side'] == 'call')]['price'].iloc[0]
                p = exp_data[(exp_data['strike'] == s) & (exp_data['side'] == 'put')]['price'].iloc[0]
                diff = abs(c - p)
                if diff < min_diff:
                    min_diff = diff
                    spot = s + c - p  # put-call parity approximation
        if spot is None:
            raise ValueError("Cannot determine TAIEX spot price")

    # Get unique expirations sorted by date
    exp_map = raw_df[['exp_code', 'exp_date']].drop_duplicates().sort_values('exp_date')
    # Filter to only monthly + nearest weekly (skip deep weekly)
    expirations = list(exp_map.itertuples(index=False))[:max_expirations]

    now = datetime.now(timezone.utc)
    r = _TW_RF_RATE
    all_rows = []

    for exp_code, exp_date_str in expirations:
        exp_date = datetime.strptime(exp_date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
        T = (exp_date - now).total_seconds() / (365.25 * 86400)
        if T <= 0:
            continue  # skip expired
        T = max(T, 1e-6)

        exp_opts = raw_df[raw_df['exp_code'] == exp_code].copy()
        if exp_opts.empty:
            continue

        exp_str = _parse_taifex_date(exp_date_str)

        for _, opt in exp_opts.iterrows():
            K = opt['strike']
            price = opt['price']
            oi = opt['oi']
            is_call = opt['is_call']

            # Calculate implied volatility from settlement/last price
            iv = implied_vol(price, spot, K, T, r, is_call)
            if iv is None or iv < 0.01:
                continue

            # Calculate Greeks
            K_arr = np.array([K])
            T_arr = np.array([T])
            sigma_arr = np.array([iv])
            is_call_arr = np.array([is_call])
            greeks = bs_greeks(spot, K_arr, T_arr, r, sigma_arr, is_call_arr)

            # Dealer GEX: OI * Gamma * Multiplier * S^2 * 0.01
            raw_gex = oi * greeks['gamma'][0] * _TXO_MULTIPLIER * spot**2 * 0.01
            if is_call:
                dealer_gex = -raw_gex
            else:
                dealer_gex = raw_gex

            all_rows.append({
                'strike': K,
                'expiry': exp_str,
                'T': T,
                'side': 'call' if is_call else 'put',
                'oi': oi,
                'iv': iv,
                'delta': greeks['delta'][0],
                'gamma': greeks['gamma'][0],
                'dealer_gex': dealer_gex,
            })

    if not all_rows:
        raise ValueError("No valid options data after IV calculation for TAIEX")

    df = pd.DataFrame(all_rows)

    # ── From here, same aggregation logic as fetch_gex ──
    call_gex = df[df['side'] == 'call'].groupby('strike')['dealer_gex'].sum().rename('call_gex')
    put_gex = df[df['side'] == 'put'].groupby('strike')['dealer_gex'].sum().rename('put_gex')
    total_oi = df.groupby('strike')['oi'].sum().rename('total_oi')
    gex_by_strike = pd.concat([call_gex, put_gex, total_oi], axis=1).fillna(0)
    gex_by_strike['net_gex'] = gex_by_strike['call_gex'] + gex_by_strike['put_gex']
    gex_by_strike = gex_by_strike.reset_index()

    # Filter to ±15% of spot
    strike_lo = spot * 0.85
    strike_hi = spot * 1.15
    gex_by_strike = gex_by_strike[
        (gex_by_strike['strike'] >= strike_lo) &
        (gex_by_strike['strike'] <= strike_hi)
    ].sort_values('strike').reset_index(drop=True)

    # Aggregate by expiry
    gex_by_expiry = df.groupby('expiry').agg(
        net_gex=('dealer_gex', 'sum'),
        total_oi=('oi', 'sum'),
    ).reset_index().sort_values('expiry')

    # Key levels
    total_gex = gex_by_strike['net_gex'].sum()

    max_idx = gex_by_strike['net_gex'].abs().idxmax()
    max_gamma_strike = gex_by_strike.loc[max_idx, 'strike']

    put_data = gex_by_strike[gex_by_strike['put_gex'] > 0]
    put_wall = put_data.loc[put_data['put_gex'].idxmax(), 'strike'] if not put_data.empty else None

    call_data = gex_by_strike[gex_by_strike['call_gex'] < 0]
    call_wall = call_data.loc[call_data['call_gex'].abs().idxmax(), 'strike'] if not call_data.empty else None

    gamma_flip = find_gamma_flip(gex_by_strike['strike'].values,
                                  gex_by_strike['net_gex'].values, spot)

    regime = 'positive' if total_gex > 0 else ('negative' if total_gex < 0 else 'neutral')

    # Implied moves per expiry
    unique_expiries = df['expiry'].unique()
    implied_moves = []
    for exp_str in sorted(unique_expiries):
        exp_rows = df[df['expiry'] == exp_str]
        if exp_rows.empty:
            continue
        T_val = exp_rows['T'].iloc[0]
        atm = exp_rows.iloc[(exp_rows['strike'] - spot).abs().argsort()[:4]]
        avg_iv = atm['iv'].mean()
        move_1s = spot * avg_iv * np.sqrt(T_val)
        dte = int(round(T_val * 365.25))
        total_exp_gex = exp_rows['dealer_gex'].sum()
        implied_moves.append({
            'expiry': exp_str,
            'dte': dte,
            'iv': avg_iv,
            'move_1s': move_1s,
            'upper': spot + move_1s,
            'lower': spot - move_1s,
            'net_gex': total_exp_gex,
        })
    implied_moves_df = pd.DataFrame(implied_moves)

    return {
        'symbol': 'TAIEX',
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
        'implied_moves': implied_moves_df,
        'timestamp': now.isoformat(),
    }


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
