"""
GEX Dashboard — Professional Dealer Gamma Exposure visualization.
Inspired by Lieta Research / GEX field charts.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from gex import fetch_gex, fetch_multi_gex, fetch_gex_taifex
from interpret import interpret_gex
import colorsys

st.set_page_config(
    page_title="GEX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp { background: #0a0a0f; }
    .block-container { padding-top: 1rem; max-width: 100%; }
    .top-bar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 0; border-bottom: 1px solid #1e1e2e; margin-bottom: 8px;
    }
    .top-bar h2 { color: #c0c0c0; font-size: 1rem; font-weight: 400; margin: 0; }
    .top-bar .ts { color: #555; font-size: 0.8rem; }
    .interp-box {
        background: #111118; border: 1px solid #1e1e2e; border-radius: 8px;
        padding: 14px 18px; margin-bottom: 10px; font-size: 0.88rem;
        color: #b0b0c0; line-height: 1.6;
    }
    .interp-box h4 { color: #e0a040; margin: 0 0 6px; font-size: 0.85rem; }
    .interp-box strong { color: #e0e0f0; }
    .summary-bar {
        background: #111118; border-left: 3px solid #e0a040;
        padding: 10px 16px; margin-bottom: 10px; border-radius: 0 8px 8px 0;
        font-size: 1rem; color: #e0e0f0;
    }
    .legend-table { font-size: 0.72rem; color: #888; }
    .legend-table td, .legend-table th { padding: 2px 8px; }
    .legend-table th { color: #666; font-weight: 400; text-align: left; border-bottom: 1px solid #1e1e2e; }
</style>
""", unsafe_allow_html=True)

# ── Color palette for expirations ─────────────────────────────────────────

def expiry_colors(n):
    """Generate n distinct colors for expiration dates."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
        colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
    return colors


# ══════════════════════════════════════════════════════════════════════════
#  Main GEX field chart (horizontal, strike on Y-axis)
# ══════════════════════════════════════════════════════════════════════════

def make_gex_field_chart(data: dict) -> go.Figure:
    """Build the professional horizontal GEX field chart."""
    spot = data['spot']
    kl = data['key_levels']
    raw = data['raw_options']
    gamma_flip = kl.get('gamma_flip')
    implied_moves = data.get('implied_moves', pd.DataFrame())
    is_taiex = data['symbol'] == 'TAIEX'

    # Filter to ±8% of spot for display
    lo, hi = spot * 0.92, spot * 1.08
    df = raw[(raw['strike'] >= lo) & (raw['strike'] <= hi)].copy()

    expirations = sorted(df['expiry'].unique())
    n_exp = len(expirations)
    colors = expiry_colors(n_exp)
    exp_color = dict(zip(expirations, colors))

    # Create figure with 3 subplots sharing Y-axis:
    # [Put Delta curves | GEX bars | Call Delta curves]
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.18, 0.64, 0.18],
        shared_yaxes=True,
        horizontal_spacing=0.005,
    )

    # ── Center: GEX horizontal bars by expiry (stacked) ──
    gex_pivot = df.pivot_table(index='strike', columns='expiry',
                                values='dealer_gex', aggfunc='sum').fillna(0)
    gex_pivot = gex_pivot.sort_index()
    strikes = gex_pivot.index.values

    for exp in expirations:
        if exp not in gex_pivot.columns:
            continue
        vals = gex_pivot[exp].values / 1e6
        dte = df[df['expiry'] == exp]['T'].iloc[0] * 365.25
        label = f"{exp} ({int(dte)}d)"

        fig.add_trace(go.Bar(
            y=strikes, x=vals, orientation='h',
            name=label,
            marker_color=exp_color[exp],
            marker_line_width=0,
            opacity=0.85,
            hovertemplate=f'{exp}<br>Strike: %{{y:,.0f}}<br>GEX: %{{x:,.1f}}M<extra></extra>',
        ), row=1, col=2)

    # ── Left: Put Delta curves ──
    for exp in expirations:
        puts = df[(df['expiry'] == exp) & (df['side'] == 'put')].sort_values('strike')
        if puts.empty:
            continue
        fig.add_trace(go.Scatter(
            y=puts['strike'], x=puts['delta'],
            mode='lines', line=dict(color=exp_color[exp], width=1),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=1)

    # ── Right: Call Delta curves ──
    for exp in expirations:
        calls = df[(df['expiry'] == exp) & (df['side'] == 'call')].sort_values('strike')
        if calls.empty:
            continue
        fig.add_trace(go.Scatter(
            y=calls['strike'], x=calls['delta'],
            mode='lines', line=dict(color=exp_color[exp], width=1),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=3)

    # ── Implied move ranges ──
    if not implied_moves.empty:
        max_gex = abs(gex_pivot.values).max() / 1e6 if gex_pivot.values.size > 0 else 1
        for _, row in implied_moves.iterrows():
            exp = row['expiry']
            if exp not in exp_color:
                continue
            color = exp_color[exp]
            upper = min(row['upper'], hi)
            lower = max(row['lower'], lo)
            x_pos = max_gex * 0.7
            fig.add_trace(go.Scatter(
                y=[lower, upper], x=[x_pos, x_pos],
                mode='lines+text',
                line=dict(color=color, width=3),
                text=[None, f"{exp} ±{row['move_1s']:,.0f}"],
                textposition='middle right',
                textfont=dict(size=8, color=color),
                showlegend=False, hoverinfo='skip',
            ), row=1, col=2)

    # ── Horizontal lines: Spot, Gamma Flip ──
    for col in [1, 2, 3]:
        fig.add_hline(y=spot, line_dash="solid", line_color="#58a6ff", line_width=1.5,
                      row=1, col=col)
    spot_fmt = f"{spot:,.0f}" if is_taiex else f"${spot:,.2f}"
    fig.add_annotation(
        y=spot, x=0, text=f"  Spot ({spot_fmt})", showarrow=False,
        font=dict(color="#58a6ff", size=10), xref="x2", yref="y",
        xanchor="left",
    )

    if gamma_flip:
        for col in [1, 2, 3]:
            fig.add_hline(y=gamma_flip, line_dash="dot", line_color="#f85149",
                          line_width=1, row=1, col=col)
        flip_fmt = f"{gamma_flip:,.0f}" if is_taiex else f"${gamma_flip:,.1f}"
        fig.add_annotation(
            y=gamma_flip, x=0, text=f"  Γ Flip ({flip_fmt})", showarrow=False,
            font=dict(color="#f85149", size=10), xref="x2", yref="y",
            xanchor="left",
        )

    # ── 25-delta line ──
    nearest_exp = expirations[0] if expirations else None
    if nearest_exp:
        ne_calls = df[(df['expiry'] == nearest_exp) & (df['side'] == 'call')]
        ne_puts = df[(df['expiry'] == nearest_exp) & (df['side'] == 'put')]
        if not ne_calls.empty:
            c25 = ne_calls.iloc[(ne_calls['delta'] - 0.25).abs().argsort()[:1]]
            if not c25.empty:
                fig.add_hline(y=float(c25['strike'].iloc[0]), line_dash="dash",
                              line_color="rgba(255,255,255,0.15)", line_width=0.5,
                              row=1, col=2)
        if not ne_puts.empty:
            p25 = ne_puts.iloc[(ne_puts['delta'] + 0.25).abs().argsort()[:1]]
            if not p25.empty:
                fig.add_hline(y=float(p25['strike'].iloc[0]), line_dash="dash",
                              line_color="rgba(255,255,255,0.15)", line_width=0.5,
                              row=1, col=2)

    # ── Layout ──
    fig.update_layout(
        barmode='relative',
        template='plotly_dark',
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='#0a0a0f',
        font=dict(color='#888', size=10),
        height=max(700, len(strikes) * 4 + 200),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="v", x=0.01, y=0.01, xanchor="left", yanchor="bottom",
            bgcolor="rgba(10,10,15,0.9)", bordercolor="#1e1e2e", borderwidth=1,
            font=dict(size=9, color='#aaa'),
            traceorder="normal",
        ),
        title=dict(
            text=f"{data['symbol']} Dealers Gamma Hedging | 造市商Γ分佈",
            font=dict(size=13, color='#888'),
            x=0.01, xanchor='left',
        ),
    )

    y_tickformat = ",.0f" if is_taiex else "$,.0f"
    y_dtick = 200 if is_taiex else 5
    fig.update_yaxes(
        tickformat=y_tickformat, dtick=y_dtick, gridcolor='#1a1a2a', gridwidth=0.5,
        zeroline=False, range=[lo, hi],
    )

    fig.update_xaxes(title_text="Put Δ", row=1, col=1,
                     range=[-1, 0], gridcolor='#1a1a2a', zeroline=True,
                     zerolinecolor='#2a2a3a')
    fig.update_xaxes(title_text="Dealers Gamma | 造市商Γ", row=1, col=2,
                     gridcolor='#1a1a2a', zeroline=True,
                     zerolinecolor='#444', zerolinewidth=1.5)
    fig.update_xaxes(title_text="Call Δ", row=1, col=3,
                     range=[0, 1], gridcolor='#1a1a2a', zeroline=True,
                     zerolinecolor='#2a2a3a')

    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Candlestick chart with GEX overlays
# ══════════════════════════════════════════════════════════════════════════

def make_kline_chart(data: dict) -> go.Figure | None:
    """Build 60-min candlestick chart with GEX levels overlay."""
    symbol = data['symbol']
    is_taiex = symbol == 'TAIEX'

    # Determine ticker and fetch OHLC
    if is_taiex:
        ticker_sym = '^TWII'
        title = 'TAIEX 加權指數 60min K線 + GEX 水位'
    else:
        ticker_sym = symbol
        title = f'{symbol} 60min K-line + GEX Levels'

    try:
        tk = yf.Ticker(ticker_sym)
        ohlc = tk.history(period='1mo', interval='60m')
        if ohlc.empty or len(ohlc) < 5:
            return None
    except Exception:
        return None

    spot = data['spot']
    kl = data['key_levels']
    gamma_flip = kl.get('gamma_flip')
    put_wall = kl.get('put_wall')
    call_wall = kl.get('call_wall')
    max_gamma = kl.get('max_gamma_strike')
    regime = data['regime']
    implied_moves = data.get('implied_moves', pd.DataFrame())

    # Get nearest expiry implied move for ±1σ range
    im_upper = im_lower = None
    nearest_exp_label = None
    if not implied_moves.empty:
        nearest = implied_moves.iloc[0]
        im_upper = nearest['upper']
        im_lower = nearest['lower']
        nearest_exp_label = f"{nearest['expiry']} ({int(nearest['dte'])}d)"

    # Build candlestick figure
    fig = go.Figure()

    # ── Implied move shaded range (±1σ) ──
    if im_upper and im_lower:
        fig.add_hrect(
            y0=im_lower, y1=im_upper,
            fillcolor="rgba(88,166,255,0.06)", line_width=0,
            annotation_text=f"±1σ Implied Move ({nearest_exp_label})",
            annotation_position="top left",
            annotation_font=dict(size=9, color='#58a6ff'),
        )
        # Upper and lower bounds as dashed lines
        fig.add_hline(y=im_upper, line_dash="dash", line_color="rgba(88,166,255,0.3)",
                      line_width=0.8)
        fig.add_hline(y=im_lower, line_dash="dash", line_color="rgba(88,166,255,0.3)",
                      line_width=0.8)
        # Labels on right
        fig.add_annotation(
            y=im_upper, x=1, xref='paper', yref='y',
            text=f" +1σ {im_upper:,.0f}", showarrow=False,
            font=dict(color='#58a6ff', size=9), xanchor='left',
        )
        fig.add_annotation(
            y=im_lower, x=1, xref='paper', yref='y',
            text=f" -1σ {im_lower:,.0f}", showarrow=False,
            font=dict(color='#58a6ff', size=9), xanchor='left',
        )

    # ── Gamma Flip line ──
    if gamma_flip:
        fig.add_hline(y=gamma_flip, line_dash="dot", line_color="#f85149", line_width=1.5)
        fig.add_annotation(
            y=gamma_flip, x=0, xref='paper', yref='y',
            text=f"Γ Flip {gamma_flip:,.0f} ", showarrow=False,
            font=dict(color='#f85149', size=9), xanchor='right',
        )

    # ── Call Wall line ──
    if call_wall:
        fig.add_hline(y=call_wall, line_dash="dash", line_color="#da3633", line_width=1)
        fig.add_annotation(
            y=call_wall, x=1, xref='paper', yref='y',
            text=f" Call Wall {call_wall:,.0f}", showarrow=False,
            font=dict(color='#da3633', size=9), xanchor='left',
        )

    # ── Put Wall line ──
    if put_wall:
        fig.add_hline(y=put_wall, line_dash="dash", line_color="#3fb950", line_width=1)
        fig.add_annotation(
            y=put_wall, x=1, xref='paper', yref='y',
            text=f" Put Wall {put_wall:,.0f}", showarrow=False,
            font=dict(color='#3fb950', size=9), xanchor='left',
        )

    # ── Max Gamma (Gamma Field center) ──
    if max_gamma:
        fig.add_hline(y=max_gamma, line_dash="dashdot", line_color="#e0a040",
                      line_width=1)
        fig.add_annotation(
            y=max_gamma, x=0, xref='paper', yref='y',
            text=f"Max Γ {max_gamma:,.0f} ", showarrow=False,
            font=dict(color='#e0a040', size=9), xanchor='right',
        )

    # ── Gamma Field shaded zone (around max gamma, ±0.5% as proxy) ──
    if max_gamma:
        gf_lo = max_gamma * 0.995
        gf_hi = max_gamma * 1.005
        fig.add_hrect(
            y0=gf_lo, y1=gf_hi,
            fillcolor="rgba(224,160,64,0.08)", line_width=0,
        )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=ohlc.index,
        open=ohlc['Open'], high=ohlc['High'],
        low=ohlc['Low'], close=ohlc['Close'],
        increasing=dict(line=dict(color='#3fb950'), fillcolor='#23803a'),
        decreasing=dict(line=dict(color='#da3633'), fillcolor='#8b1a1a'),
        name='K-line',
    ))

    # ── SMA 20 ──
    if len(ohlc) >= 20:
        sma20 = ohlc['Close'].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=ohlc.index, y=sma20,
            mode='lines', line=dict(color='rgba(200,200,200,0.4)', width=1),
            name='SMA 20',
        ))

    # ── Call/Put Dominate indicator (regime background) ──
    regime_color = 'rgba(63,185,80,0.04)' if regime == 'positive' else 'rgba(218,54,51,0.04)'
    regime_label = '正 Gamma (Call Dominate)' if regime == 'positive' else '負 Gamma (Put Dominate)'
    # Add subtle background tint
    y_min = ohlc['Low'].min() * 0.998
    y_max = ohlc['High'].max() * 1.002
    fig.add_hrect(y0=y_min, y1=y_max, fillcolor=regime_color, line_width=0)

    # ── Key Delta strikes (25-delta call/put from nearest expiry) ──
    raw = data['raw_options']
    nearest_exp_list = sorted(raw['expiry'].unique())
    if nearest_exp_list:
        ne = nearest_exp_list[0]
        ne_calls = raw[(raw['expiry'] == ne) & (raw['side'] == 'call')]
        ne_puts = raw[(raw['expiry'] == ne) & (raw['side'] == 'put')]
        if not ne_calls.empty:
            c25 = ne_calls.iloc[(ne_calls['delta'] - 0.25).abs().argsort()[:1]]
            c25_strike = float(c25['strike'].iloc[0])
            if y_min < c25_strike < y_max:
                fig.add_hline(y=c25_strike, line_dash="dot",
                              line_color="rgba(255,255,255,0.15)", line_width=0.7)
                fig.add_annotation(
                    y=c25_strike, x=1, xref='paper', yref='y',
                    text=f" 25Δ Call {c25_strike:,.0f}", showarrow=False,
                    font=dict(color='rgba(255,255,255,0.35)', size=8), xanchor='left',
                )
        if not ne_puts.empty:
            p25 = ne_puts.iloc[(ne_puts['delta'] + 0.25).abs().argsort()[:1]]
            p25_strike = float(p25['strike'].iloc[0])
            if y_min < p25_strike < y_max:
                fig.add_hline(y=p25_strike, line_dash="dot",
                              line_color="rgba(255,255,255,0.15)", line_width=0.7)
                fig.add_annotation(
                    y=p25_strike, x=1, xref='paper', yref='y',
                    text=f" 25Δ Put {p25_strike:,.0f}", showarrow=False,
                    font=dict(color='rgba(255,255,255,0.35)', size=8), xanchor='left',
                )

    # ── Layout ──
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='#0a0a0f',
        font=dict(color='#888', size=10),
        height=500,
        margin=dict(l=10, r=80, t=40, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", x=0.01, y=1.02, xanchor="left", yanchor="bottom",
            bgcolor="rgba(10,10,15,0.9)", bordercolor="#1e1e2e", borderwidth=1,
            font=dict(size=9, color='#aaa'),
        ),
        title=dict(
            text=title,
            font=dict(size=13, color='#888'),
            x=0.01, xanchor='left',
        ),
    )

    tick_fmt = ",.0f" if is_taiex else "$,.0f"
    fig.update_yaxes(tickformat=tick_fmt, gridcolor='#1a1a2a', gridwidth=0.5,
                     zeroline=False, side='right')
    fig.update_xaxes(gridcolor='#1a1a2a', gridwidth=0.5, rangeslider_visible=False)

    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Expiration legend table
# ══════════════════════════════════════════════════════════════════════════

def make_legend_table(data: dict) -> str:
    im = data.get('implied_moves', pd.DataFrame())
    if im.empty:
        return ""

    is_taiex = data['symbol'] == 'TAIEX'
    cur = '' if is_taiex else '$'

    rows_html = ""
    colors = expiry_colors(len(im))
    for i, (_, row) in enumerate(im.iterrows()):
        color = colors[i]
        gex_val = row['net_gex'] / 1e6
        gex_sign = '+' if gex_val >= 0 else ''
        rows_html += f"""<tr>
            <td><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;margin-right:4px;vertical-align:middle;"></span>{row['expiry']}</td>
            <td>{int(row['dte'])}d</td>
            <td>GEX: {gex_sign}{gex_val:,.1f}M</td>
            <td>IV: {row['iv']*100:.1f}%</td>
            <td>±{cur}{row['move_1s']:,.0f} (1σ)</td>
        </tr>"""

    return f"""<table class="legend-table">
        <tr><th>Expiry</th><th>DTE</th><th>GEX</th><th>ATM IV</th><th>Implied Move</th></tr>
        {rows_html}
    </table>"""


# ══════════════════════════════════════════════════════════════════════════
#  GEX levels summary card (for below candlestick)
# ══════════════════════════════════════════════════════════════════════════

def make_levels_card(data: dict) -> str:
    """HTML card summarizing all GEX levels shown on the K-line chart."""
    kl = data['key_levels']
    spot = data['spot']
    regime = data['regime']
    is_taiex = data['symbol'] == 'TAIEX'
    im = data.get('implied_moves', pd.DataFrame())

    items = []

    # Implied Move
    if not im.empty:
        nearest = im.iloc[0]
        items.append(f"""<div class="lv-item">
            <span class="lv-label" style="color:#58a6ff;">Implied Move (±1σ)</span>
            <span class="lv-val">{nearest['lower']:,.0f} ~ {nearest['upper']:,.0f}
            <span class="lv-sub">({nearest['expiry']}, {int(nearest['dte'])}d, IV {nearest['iv']*100:.1f}%)</span></span>
        </div>""")

    # Gamma Flip
    gf = kl.get('gamma_flip')
    if gf:
        dist = (gf / spot - 1) * 100
        items.append(f"""<div class="lv-item">
            <span class="lv-label" style="color:#f85149;">Gamma Flip | 波動分界線</span>
            <span class="lv-val">{gf:,.0f} <span class="lv-sub">({dist:+.1f}%)</span></span>
        </div>""")

    # Gamma Field (Max Gamma)
    mg = kl.get('max_gamma_strike')
    if mg:
        dist = (mg / spot - 1) * 100
        items.append(f"""<div class="lv-item">
            <span class="lv-label" style="color:#e0a040;">Gamma Field | Gamma 磁場</span>
            <span class="lv-val">{mg:,.0f} <span class="lv-sub">({dist:+.1f}%)</span></span>
        </div>""")

    # Call Wall
    cw = kl.get('call_wall')
    if cw:
        dist = (cw / spot - 1) * 100
        items.append(f"""<div class="lv-item">
            <span class="lv-label" style="color:#da3633;">Call Wall</span>
            <span class="lv-val">{cw:,.0f} <span class="lv-sub">({dist:+.1f}%)</span></span>
        </div>""")

    # Put Wall
    pw = kl.get('put_wall')
    if pw:
        dist = (pw / spot - 1) * 100
        items.append(f"""<div class="lv-item">
            <span class="lv-label" style="color:#3fb950;">Put Wall</span>
            <span class="lv-val">{pw:,.0f} <span class="lv-sub">({dist:+.1f}%)</span></span>
        </div>""")

    # Call/Put Dominate
    regime_color = '#3fb950' if regime == 'positive' else '#da3633'
    regime_text = '正 Gamma — Call Dominate (造市商壓縮波動)' if regime == 'positive' else '負 Gamma — Put Dominate (造市商放大波動)'
    items.append(f"""<div class="lv-item">
        <span class="lv-label" style="color:{regime_color};">Call/Put Dominate</span>
        <span class="lv-val">{regime_text}</span>
    </div>""")

    # CE (nearest expiry)
    if not im.empty:
        nearest = im.iloc[0]
        items.append(f"""<div class="lv-item">
            <span class="lv-label" style="color:#888;">CE = 最近到期日</span>
            <span class="lv-val">{nearest['expiry']} ({int(nearest['dte'])}d) · GEX: {nearest['net_gex']/1e6:+,.1f}M</span>
        </div>""")

    # Key Delta
    raw = data['raw_options']
    nearest_exp_list = sorted(raw['expiry'].unique())
    if nearest_exp_list:
        ne = nearest_exp_list[0]
        ne_calls = raw[(raw['expiry'] == ne) & (raw['side'] == 'call')]
        ne_puts = raw[(raw['expiry'] == ne) & (raw['side'] == 'put')]
        delta_parts = []
        if not ne_calls.empty:
            c25 = ne_calls.iloc[(ne_calls['delta'] - 0.25).abs().argsort()[:1]]
            delta_parts.append(f"25Δ Call: {float(c25['strike'].iloc[0]):,.0f}")
        if not ne_puts.empty:
            p25 = ne_puts.iloc[(ne_puts['delta'] + 0.25).abs().argsort()[:1]]
            delta_parts.append(f"25Δ Put: {float(p25['strike'].iloc[0]):,.0f}")
        if delta_parts:
            items.append(f"""<div class="lv-item">
                <span class="lv-label" style="color:#888;">Key Delta</span>
                <span class="lv-val">{' · '.join(delta_parts)}</span>
            </div>""")

    return f"""
    <style>
        .lv-card {{ background: #111118; border: 1px solid #1e1e2e; border-radius: 8px; padding: 8px 16px; }}
        .lv-item {{ display: flex; justify-content: space-between; align-items: center;
                    padding: 8px 0; border-bottom: 1px solid #1a1a2a; font-size: 0.88rem; }}
        .lv-item:last-child {{ border-bottom: none; }}
        .lv-label {{ font-weight: 500; min-width: 200px; }}
        .lv-val {{ color: #e0e0f0; text-align: right; }}
        .lv-sub {{ color: #666; font-size: 0.78rem; }}
    </style>
    <div class="lv-card">{''.join(items)}</div>
    """


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

SYMBOLS = {'TAIEX': '台灣加權指數', 'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', 'IWM': 'Russell 2000', 'DIA': 'Dow Jones'}

# Controls
c1, c2, c3, c4 = st.columns([1.5, 2, 1, 1])
with c1:
    symbol = c1.selectbox("標的", list(SYMBOLS.keys()),
                           format_func=lambda s: f"{s} — {SYMBOLS[s]}",
                           label_visibility="collapsed")
with c2:
    custom = c2.text_input("自訂（如 AAPL, TSLA）", "", label_visibility="collapsed",
                            placeholder="自訂標的（留空用左邊選擇）")
    if custom.strip():
        symbol = custom.strip().upper()
with c3:
    max_exp = c3.selectbox("到期日", [6, 10, 15, 20], index=1,
                            format_func=lambda x: f"{x} 個到期日",
                            label_visibility="collapsed")
with c4:
    run = c4.button("🔍 計算", type="primary", use_container_width=True)

if not run:
    st.markdown("""
    <div style="text-align:center; color:#555; padding: 6rem 0;">
        <p style="font-size: 1.3rem; color: #777;">📊 Dealer Gamma Exposure Dashboard</p>
        <p>選擇標的後按「計算」，查看造市商 Gamma 分布、Flip 水位、支撐壓力位</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

with st.spinner(f"正在分析 {symbol}..."):
    try:
        if symbol == 'TAIEX':
            data = fetch_gex_taifex(max_expirations=max_exp)
        else:
            data = fetch_gex(symbol, max_expirations=max_exp)
    except Exception as e:
        st.error(f"分析失敗: {e}")
        st.stop()

interp = interpret_gex(data)

# Top bar
st.markdown(f"""<div class="top-bar">
    <h2>{data['symbol']} Dealers Gamma Hedging | 造市商Γ分佈</h2>
    <span class="ts">{data['timestamp'][:19]} UTC · {'TAIFEX' if data['symbol'] == 'TAIEX' else 'Yahoo Finance (delayed ~15 min)'}</span>
</div>""", unsafe_allow_html=True)

# Summary
st.markdown(f'<div class="summary-bar">{interp["summary"]}</div>', unsafe_allow_html=True)

# Main layout: Chart (left 70%) + Interpretation (right 30%)
col_chart, col_info = st.columns([7, 3])

with col_chart:
    fig = make_gex_field_chart(data)
    st.plotly_chart(fig, use_container_width=True)

    # Legend table
    legend_html = make_legend_table(data)
    if legend_html:
        st.markdown(legend_html, unsafe_allow_html=True)

with col_info:
    # Key metrics
    spot = data['spot']
    kl = data['key_levels']
    regime = data['regime']

    regime_emoji = '🟢' if regime == 'positive' else '🔴'
    is_taiex = data['symbol'] == 'TAIEX'
    cur = '' if is_taiex else '$'
    spot_display = f"{spot:,.0f}" if is_taiex else f"${spot:,.2f}"
    st.markdown(f"""<div class="interp-box">
        <h4>📍 關鍵水位</h4>
        <strong>現價:</strong> {spot_display}<br>
        <strong>Regime:</strong> {regime_emoji} {'正 Gamma' if regime == 'positive' else '負 Gamma'}<br>
        <strong>Gamma Flip:</strong> {f"{cur}{kl['gamma_flip']:,.0f} ({(kl['gamma_flip']/spot-1)*100:+.1f}%)" if kl['gamma_flip'] else 'N/A'}<br>
        <strong>Put Wall:</strong> {f"{cur}{kl['put_wall']:,.0f}" if kl['put_wall'] else 'N/A'}<br>
        <strong>Call Wall:</strong> {f"{cur}{kl['call_wall']:,.0f}" if kl['call_wall'] else 'N/A'}<br>
        <strong>Max Gamma:</strong> {f"{cur}{kl['max_gamma_strike']:,.0f}" if kl['max_gamma_strike'] else 'N/A'}
    </div>""", unsafe_allow_html=True)

    # Regime analysis
    st.markdown(f"""<div class="interp-box">
        <h4>📡 Gamma 環境</h4>
        {interp['regime_analysis']}
    </div>""", unsafe_allow_html=True)

    # Support / Resistance
    st.markdown(f"""<div class="interp-box">
        <h4>🎯 支撐壓力位</h4>
        {interp['support_resistance']}
    </div>""", unsafe_allow_html=True)

    # Volatility
    st.markdown(f"""<div class="interp-box">
        <h4>📈 波動率展望</h4>
        {interp['volatility_outlook']}
    </div>""", unsafe_allow_html=True)

    # Scenarios
    st.markdown(f"""<div class="interp-box">
        <h4>⚡ 關鍵情境</h4>
        {interp['key_scenarios']}
    </div>""", unsafe_allow_html=True)

    multiplier = 50 if data['symbol'] == 'TAIEX' else 100
    data_src = 'TAIFEX' if data['symbol'] == 'TAIEX' else 'Yahoo Finance (delayed ~15 min)'
    st.markdown(f"""<div style="color:#444; font-size:0.7rem; margin-top:8px;">
        Risk-free rate: {data['risk_free_rate']*100:.2f}% · GEX = OI × Γ × {multiplier} × S² × 0.01 · {data_src}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  K-line chart with GEX overlays (full width below)
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")

kline_fig = make_kline_chart(data)
if kline_fig:
    st.plotly_chart(kline_fig, use_container_width=True)

    # Levels summary card
    levels_html = make_levels_card(data)
    st.markdown(levels_html, unsafe_allow_html=True)
else:
    st.info("無法取得 K 線資料")
