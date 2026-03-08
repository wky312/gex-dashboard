"""
GEX Dashboard — Dealer Gamma Exposure web dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from gex import fetch_gex, fetch_multi_gex
from interpret import interpret_gex

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GEX Dashboard · Dealer Gamma Exposure",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for web-page feel ──────────────────────────────────────────

st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dark theme overrides */
    .stApp { background: #0d1117; }

    .hero {
        text-align: center;
        padding: 1.5rem 0 1rem;
        border-bottom: 1px solid #21262d;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e6edf3;
        margin: 0;
    }
    .hero .sub {
        color: #7d8590;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex; gap: 12px; flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        flex: 1; min-width: 150px;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
    }
    .metric-card .label { font-size: 0.72rem; color: #7d8590; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 1.5rem; font-weight: 700; color: #e6edf3; margin-top: 4px; }
    .metric-card .delta { font-size: 0.82rem; margin-top: 2px; }
    .metric-card .delta.green { color: #3fb950; }
    .metric-card .delta.red { color: #f85149; }
    .metric-card .delta.yellow { color: #d29922; }

    /* Interpretation panel */
    .interp-panel {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 1rem;
    }
    .interp-panel h3 {
        color: #f0883e;
        font-size: 1rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.5rem;
    }

    /* Summary banner */
    .summary-banner {
        background: linear-gradient(135deg, #161b22, #1c2333);
        border: 1px solid #30363d;
        border-left: 4px solid #f0883e;
        border-radius: 10px;
        padding: 18px 24px;
        font-size: 1.1rem;
        color: #e6edf3;
        margin-bottom: 1.2rem;
    }

    /* Level tags */
    .level-row {
        display: flex; align-items: center; gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid #21262d;
    }
    .level-row:last-child { border-bottom: none; }
    .level-tag {
        display: inline-block;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .level-tag.resistance { background: rgba(248,81,73,0.15); color: #f85149; }
    .level-tag.support { background: rgba(63,185,80,0.15); color: #3fb950; }

    /* Timestamp */
    .ts { color: #484f58; font-size: 0.75rem; text-align: right; margin-top: 0.5rem; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 8px 20px; color: #7d8590;
    }
    .stTabs [aria-selected="true"] {
        background: #f0883e !important; color: #fff !important;
        border-color: #f0883e !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  Chart builders
# ══════════════════════════════════════════════════════════════════════════

CHART_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,17,23,1)',
    font=dict(family='Inter, sans-serif', color='#e6edf3'),
    margin=dict(l=50, r=20, t=30, b=50),
)


def make_gex_strike_chart(df, spot, kl, compact=False):
    fig = go.Figure()

    pos = df['net_gex'].clip(lower=0) / 1e6
    neg = df['net_gex'].clip(upper=0) / 1e6

    fig.add_trace(go.Bar(
        x=df['strike'], y=pos, name='正 GEX（穩定）',
        marker_color='rgba(63,185,80,0.7)',
        hovertemplate='Strike: $%{x:,.0f}<br>GEX: +%{y:,.0f}M<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        x=df['strike'], y=neg, name='負 GEX（不穩定）',
        marker_color='rgba(248,81,73,0.7)',
        hovertemplate='Strike: $%{x:,.0f}<br>GEX: %{y:,.0f}M<extra></extra>',
    ))

    fig.add_vline(x=spot, line_dash="dash", line_color="#58a6ff", line_width=2,
                  annotation_text=f"Spot ${spot:,.1f}" if not compact else None,
                  annotation_font_color="#58a6ff")

    if kl.get('gamma_flip'):
        fig.add_vline(x=kl['gamma_flip'], line_dash="dot", line_color="#d29922", line_width=2,
                      annotation_text=f"Flip ${kl['gamma_flip']:,.0f}" if not compact else None,
                      annotation_font_color="#d29922")

    if kl.get('put_wall') and not compact:
        fig.add_vline(x=kl['put_wall'], line_dash="dot", line_color="#3fb950",
                      annotation_text=f"Put Wall ${kl['put_wall']:,.0f}",
                      annotation_font_color="#3fb950", annotation_position="bottom left")
    if kl.get('call_wall') and not compact:
        fig.add_vline(x=kl['call_wall'], line_dash="dot", line_color="#f85149",
                      annotation_text=f"Call Wall ${kl['call_wall']:,.0f}",
                      annotation_font_color="#f85149")

    fig.update_layout(
        **CHART_LAYOUT,
        barmode='relative',
        height=420 if not compact else 320,
        xaxis_title="Strike Price",
        yaxis_title="Dealer GEX ($ millions)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center") if not compact else dict(visible=False),
    )
    return fig


def make_gex_expiry_chart(df):
    fig = go.Figure()
    colors = ['#3fb950' if v >= 0 else '#f85149' for v in df['net_gex']]
    fig.add_trace(go.Bar(
        x=df['expiry'], y=df['net_gex'] / 1e6,
        marker_color=colors,
        hovertemplate='%{x}<br>GEX: %{y:,.0f}M<extra></extra>',
    ))
    fig.update_layout(**CHART_LAYOUT, height=320,
                      xaxis_title="Expiration", yaxis_title="Net GEX ($M)")
    return fig


def make_gex_heatmap(raw_df, spot):
    mask = (raw_df['strike'] >= spot * 0.92) & (raw_df['strike'] <= spot * 1.08)
    df = raw_df[mask].copy()
    pivot = df.pivot_table(index='strike', columns='expiry',
                           values='dealer_gex', aggfunc='sum') / 1e6
    pivot = pivot.sort_index(ascending=False)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=[f"${s:,.0f}" for s in pivot.index],
        colorscale=[[0, '#f85149'], [0.5, '#0d1117'], [1, '#3fb950']],
        zmid=0, colorbar_title="GEX ($M)",
        hovertemplate='Strike: %{y}<br>Expiry: %{x}<br>GEX: %{z:,.1f}M<extra></extra>',
    ))
    fig.update_layout(**CHART_LAYOUT, height=400,
                      xaxis_title="Expiration", yaxis_title="Strike")
    return fig


def make_call_put_chart(df, spot):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['strike'], y=df['call_gex'] / 1e6, name='Call GEX',
        marker_color='rgba(248,81,73,0.6)',
    ))
    fig.add_trace(go.Bar(
        x=df['strike'], y=df['put_gex'] / 1e6, name='Put GEX',
        marker_color='rgba(63,185,80,0.6)',
    ))
    fig.add_vline(x=spot, line_dash="dash", line_color="#58a6ff", line_width=1)
    fig.update_layout(**CHART_LAYOUT, barmode='group', height=350,
                      xaxis_title="Strike", yaxis_title="GEX ($M)",
                      legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Metric card HTML helpers
# ══════════════════════════════════════════════════════════════════════════

def metric_card(label, value, delta=None, delta_color='green'):
    delta_html = f'<div class="delta {delta_color}">{delta}</div>' if delta else ''
    return f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>"""


def render_metrics_row(data, interp):
    spot = data['spot']
    regime = data['regime']
    kl = data['key_levels']

    regime_color = 'green' if regime == 'positive' else 'red'
    regime_label = '正 GAMMA' if regime == 'positive' else '負 GAMMA'
    regime_desc = '低波動 · 均值回歸' if regime == 'positive' else '高波動 · 趨勢加速'

    flip_val = f"${kl['gamma_flip']:,.1f}" if kl['gamma_flip'] else 'N/A'
    flip_delta = ''
    if kl['gamma_flip']:
        pct = (kl['gamma_flip'] / spot - 1) * 100
        flip_delta = f"距現價 {pct:+.1f}%"

    pw = f"${kl['put_wall']:,.0f}" if kl['put_wall'] else 'N/A'
    cw = f"${kl['call_wall']:,.0f}" if kl['call_wall'] else 'N/A'

    html = f"""<div class="metric-row">
        {metric_card('現價', f'${spot:,.2f}')}
        {metric_card('Gamma Regime', regime_label, regime_desc, regime_color)}
        {metric_card('Gamma Flip', flip_val, flip_delta, 'yellow')}
        {metric_card('Put Wall（支撐）', pw, '', 'green')}
        {metric_card('Call Wall（壓力）', cw, '', 'red')}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  Render single symbol
# ══════════════════════════════════════════════════════════════════════════

def render_single(data):
    interp = interpret_gex(data)

    # Summary banner
    st.markdown(f'<div class="summary-banner">{interp["summary"]}</div>', unsafe_allow_html=True)

    # Metrics
    render_metrics_row(data, interp)

    # Two-column layout: Chart (left) + Interpretation (right)
    col_chart, col_interp = st.columns([3, 2])

    with col_chart:
        fig = make_gex_strike_chart(data['gex_by_strike'], data['spot'], data['key_levels'])
        st.plotly_chart(fig, use_container_width=True)

    with col_interp:
        # Regime analysis
        st.markdown(f"""<div class="interp-panel">
            <h3>📡 Gamma 環境分析</h3>
            {interp['regime_analysis']}
        </div>""", unsafe_allow_html=True)

        # Volatility outlook
        st.markdown(f"""<div class="interp-panel">
            <h3>📈 波動率展望</h3>
            {interp['volatility_outlook']}
        </div>""", unsafe_allow_html=True)

    # Support / Resistance + Scenarios
    col_sr, col_sc = st.columns([1, 1])

    with col_sr:
        st.markdown(f"""<div class="interp-panel">
            <h3>🎯 支撐與壓力位</h3>
            {interp['support_resistance']}
        </div>""", unsafe_allow_html=True)

    with col_sc:
        st.markdown(f"""<div class="interp-panel">
            <h3>⚡ 關鍵情境</h3>
            {interp['key_scenarios']}
        </div>""", unsafe_allow_html=True)

    # Detailed charts in tabs
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Call vs Put GEX", "到期日分布", "GEX Heatmap"])

    with tab1:
        fig = make_call_put_chart(data['gex_by_strike'], data['spot'])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = make_gex_expiry_chart(data['gex_by_expiry'])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = make_gex_heatmap(data['raw_options'], data['spot'])
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown(f"""<div class="ts">
        Risk-free rate: {data['risk_free_rate']*100:.2f}% (FRED 3M T-Bill) ·
        Data: {data['timestamp'][:19]} UTC ·
        Source: Yahoo Finance (delayed ~15 min) ·
        GEX = OI × Γ × 100 × S² × 0.01
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  Render multi-symbol comparison
# ══════════════════════════════════════════════════════════════════════════

def render_multi(results):
    st.markdown('<div class="hero"><h1>📊 多標的 GEX 比較</h1></div>', unsafe_allow_html=True)

    # Summary table
    rows = []
    for sym, data in results.items():
        if 'error' in data:
            rows.append({'標的': sym, 'Regime': f"❌ {data['error']}"})
            continue
        kl = data['key_levels']
        r = data['regime']
        rows.append({
            '標的': sym,
            '現價': f"${data['spot']:,.2f}",
            'Regime': f"{'🟢' if r=='positive' else '🔴'} {r.upper()}",
            'Total GEX': f"{data['total_gex']/1e9:+.2f}B",
            'Gamma Flip': f"${kl['gamma_flip']:,.0f}" if kl['gamma_flip'] else "N/A",
            'Put Wall': f"${kl['put_wall']:,.0f}" if kl['put_wall'] else "—",
            'Call Wall': f"${kl['call_wall']:,.0f}" if kl['call_wall'] else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Charts grid
    valid = {s: d for s, d in results.items() if 'error' not in d}
    if not valid:
        return

    cols = st.columns(min(len(valid), 3))
    for i, (sym, data) in enumerate(valid.items()):
        with cols[i % len(cols)]:
            interp = interpret_gex(data)
            regime = data['regime']
            emoji = '🟢' if regime == 'positive' else '🔴'
            st.markdown(f"### {emoji} {sym} · ${data['spot']:,.2f}")
            fig = make_gex_strike_chart(data['gex_by_strike'], data['spot'],
                                         data['key_levels'], compact=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"<small>{interp['summary']}</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

SYMBOLS = {'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', 'IWM': 'Russell 2000', 'DIA': 'Dow Jones'}

# Top bar as tabs instead of sidebar
st.markdown("""<div class="hero">
    <h1>📊 Dealer Gamma Exposure Dashboard</h1>
    <div class="sub">造市商 Gamma 曝險分析 · 支撐壓力位識別 · 市場環境解讀</div>
</div>""", unsafe_allow_html=True)

# Controls in columns
c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
with c1:
    mode = st.radio("模式", ["單一標的分析", "多標的比較"], horizontal=True, label_visibility="collapsed")
with c2:
    if mode == "單一標的分析":
        symbol = st.selectbox("標的", list(SYMBOLS.keys()),
                               format_func=lambda s: f"{s} — {SYMBOLS[s]}",
                               label_visibility="collapsed")
    else:
        symbols = st.multiselect("標的", list(SYMBOLS.keys()),
                                  default=["SPY", "QQQ", "IWM"],
                                  format_func=lambda s: f"{s}",
                                  label_visibility="collapsed")
with c3:
    max_exp = st.selectbox("到期日數", [6, 10, 15, 20], index=1, label_visibility="collapsed",
                            format_func=lambda x: f"{x} 個到期日")
with c4:
    run = st.button("🔍 計算 GEX", type="primary", use_container_width=True)

st.markdown("---")

if not run:
    st.markdown("""
    <div style="text-align:center; color:#7d8590; padding: 3rem 0;">
        <p style="font-size: 1.2rem;">👆 選擇標的後點擊「計算 GEX」</p>
        <p style="font-size: 0.9rem; max-width: 600px; margin: 1rem auto;">
            本工具分析造市商的 Gamma Exposure 分布，識別 Gamma Flip（正負翻轉水位）、
            Put Wall（支撐）和 Call Wall（壓力），並提供市場環境解讀。
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if mode == "單一標的分析":
    with st.spinner(f"正在分析 {symbol}..."):
        try:
            data = fetch_gex(symbol, max_expirations=max_exp)
        except Exception as e:
            st.error(f"分析失敗: {e}")
            st.stop()
    render_single(data)
else:
    if not symbols:
        st.warning("請至少選擇一個標的")
        st.stop()
    with st.spinner(f"正在分析 {', '.join(symbols)}..."):
        results = fetch_multi_gex(symbols, max_expirations=max_exp)
    render_multi(results)
