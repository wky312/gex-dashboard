"""
GEX Dashboard — Streamlit app for Dealer Gamma Exposure analysis.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from gex import fetch_gex, fetch_multi_gex

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GEX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYMBOLS = {
    'SPY': 'S&P 500 ETF',
    'QQQ': 'Nasdaq 100 ETF',
    'IWM': 'Russell 2000 ETF',
    'DIA': 'Dow Jones ETF',
}

REGIME_COLORS = {
    'positive': '#22c55e',
    'negative': '#ef4444',
    'neutral': '#94a3b8',
}
REGIME_LABELS = {
    'positive': '正 Gamma（低波動，均值回歸）',
    'negative': '負 Gamma（高波動，趨勢加速）',
    'neutral': '中性',
}


# ══════════════════════════════════════════════════════════════════════════
#  Chart builders
# ══════════════════════════════════════════════════════════════════════════

def make_gex_strike_chart(df: pd.DataFrame, spot: float, kl: dict,
                           compact: bool = False) -> go.Figure:
    fig = go.Figure()

    pos = df['net_gex'].clip(lower=0)
    neg = df['net_gex'].clip(upper=0)

    fig.add_trace(go.Bar(
        x=df['strike'], y=pos, name='正 GEX',
        marker_color='rgba(34,197,94,0.7)',
    ))
    fig.add_trace(go.Bar(
        x=df['strike'], y=neg, name='負 GEX',
        marker_color='rgba(239,68,68,0.7)',
    ))

    fig.add_vline(x=spot, line_dash="dash", line_color="#60a5fa",
                  annotation_text=f"Spot ${spot:,.1f}" if not compact else None)

    if kl.get('gamma_flip'):
        fig.add_vline(x=kl['gamma_flip'], line_dash="dot", line_color="#f59e0b",
                      annotation_text=f"Flip ${kl['gamma_flip']:,.1f}" if not compact else None)

    if kl.get('put_wall') and not compact:
        fig.add_vline(x=kl['put_wall'], line_dash="dot", line_color="#22c55e",
                      annotation_text=f"Put Wall ${kl['put_wall']:,.0f}")
    if kl.get('call_wall') and not compact:
        fig.add_vline(x=kl['call_wall'], line_dash="dot", line_color="#ef4444",
                      annotation_text=f"Call Wall ${kl['call_wall']:,.0f}")

    height = 350 if compact else 500
    fig.update_layout(
        barmode='relative',
        template='plotly_dark',
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Strike",
        yaxis_title="Dealer GEX ($)",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        showlegend=not compact,
    )
    return fig


def make_gex_expiry_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ['#22c55e' if v >= 0 else '#ef4444' for v in df['net_gex']]
    fig.add_trace(go.Bar(
        x=df['expiry'], y=df['net_gex'],
        marker_color=colors,
    ))
    fig.update_layout(
        template='plotly_dark',
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="到期日",
        yaxis_title="Net GEX ($)",
    )
    return fig


def make_gex_heatmap(raw_df: pd.DataFrame, spot: float) -> go.Figure:
    mask = (raw_df['strike'] >= spot * 0.90) & (raw_df['strike'] <= spot * 1.10)
    df = raw_df[mask].copy()

    pivot = df.pivot_table(index='strike', columns='expiry',
                           values='dealer_gex', aggfunc='sum')
    pivot = pivot.sort_index(ascending=False)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        colorbar_title="GEX",
    ))
    fig.update_layout(
        template='plotly_dark',
        height=400,
        margin=dict(l=60, r=20, t=20, b=60),
        xaxis_title="到期日",
        yaxis_title="Strike",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Rendering functions
# ══════════════════════════════════════════════════════════════════════════

def render_single(data: dict):
    symbol = data['symbol']
    spot = data['spot']
    regime = data['regime']
    kl = data['key_levels']
    gex_df = data['gex_by_strike']

    st.title(f"📊 {symbol} Gamma Exposure")

    cols = st.columns(5)
    cols[0].metric("現價", f"${spot:,.2f}")
    cols[1].metric("Gamma Regime",
                   regime.upper(),
                   delta=REGIME_LABELS[regime],
                   delta_color="normal" if regime == 'positive' else "inverse")
    cols[2].metric("Gamma Flip",
                   f"${kl['gamma_flip']:,.1f}" if kl['gamma_flip'] else "N/A",
                   delta=f"{(kl['gamma_flip']/spot - 1)*100:+.2f}% from spot" if kl['gamma_flip'] else None)
    cols[3].metric("Put Wall（支撐）",
                   f"${kl['put_wall']:,.0f}" if kl['put_wall'] else "N/A")
    cols[4].metric("Call Wall（壓力）",
                   f"${kl['call_wall']:,.0f}" if kl['call_wall'] else "N/A")

    st.markdown(f"<small>Risk-free rate: {data['risk_free_rate']*100:.2f}% · "
                f"Data: {data['timestamp'][:19]} UTC · "
                f"Source: Yahoo Finance (delayed ~15 min)</small>",
                unsafe_allow_html=True)

    # GEX by Strike
    st.subheader("GEX 分布（by Strike）")
    fig = make_gex_strike_chart(gex_df, spot, kl)
    st.plotly_chart(fig, use_container_width=True)

    # GEX by Expiry + Heatmap
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("GEX by 到期日")
        fig_exp = make_gex_expiry_chart(data['gex_by_expiry'])
        st.plotly_chart(fig_exp, use_container_width=True)
    with col2:
        st.subheader("GEX Heatmap（Strike × Expiry）")
        fig_heat = make_gex_heatmap(data['raw_options'], spot)
        st.plotly_chart(fig_heat, use_container_width=True)

    # Key levels table
    st.subheader("關鍵水位")
    levels = []
    for name, key in [("Gamma Flip", 'gamma_flip'), ("Max Gamma Strike", 'max_gamma_strike'),
                      ("Put Wall（支撐）", 'put_wall'), ("Call Wall（壓力）", 'call_wall')]:
        val = kl.get(key)
        levels.append({
            "水位": name,
            "價格": f"${val:,.1f}" if val else "N/A",
            "距現價": f"{(val/spot-1)*100:+.2f}%" if val else "",
        })
    st.dataframe(pd.DataFrame(levels), hide_index=True, use_container_width=True)

    # Interpretation
    with st.expander("解讀說明"):
        if regime == 'positive':
            st.markdown("""
            **正 Gamma 環境** — 造市商會在價格上漲時賣出、下跌時買入，
            形成「均值回歸」效果，壓低盤中波動。價格傾向被吸引到 Max Gamma Strike 附近。
            """)
        else:
            st.markdown("""
            **負 Gamma 環境** — 造市商被迫追高殺低（上漲時買、下跌時賣），
            形成正回饋循環，放大價格波動。突破 Gamma Flip 水位後波動可能急劇增加。
            """)


def render_multi(results: dict):
    st.title("📊 多標的 GEX 比較")

    rows = []
    for sym, data in results.items():
        if 'error' in data:
            rows.append({'標的': sym, '狀態': f"Error: {data['error']}"})
            continue
        kl = data['key_levels']
        rows.append({
            '標的': sym,
            '現價': f"${data['spot']:,.2f}",
            'Regime': data['regime'].upper(),
            'Total GEX': f"{data['total_gex']:,.0f}",
            'Gamma Flip': f"${kl['gamma_flip']:,.1f}" if kl['gamma_flip'] else "N/A",
            'Put Wall': f"${kl['put_wall']:,.0f}" if kl['put_wall'] else "N/A",
            'Call Wall': f"${kl['call_wall']:,.0f}" if kl['call_wall'] else "N/A",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    valid = {s: d for s, d in results.items() if 'error' not in d}
    if not valid:
        return

    n = len(valid)
    cols = st.columns(min(n, 3))
    for i, (sym, data) in enumerate(valid.items()):
        with cols[i % len(cols)]:
            st.subheader(f"{sym}")
            regime = data['regime']
            color = REGIME_COLORS[regime]
            st.markdown(f"<span style='color:{color};font-weight:bold'>{regime.upper()}</span> · "
                        f"Spot ${data['spot']:,.2f}",
                        unsafe_allow_html=True)
            fig = make_gex_strike_chart(data['gex_by_strike'], data['spot'],
                                         data['key_levels'], compact=True)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  Main flow
# ══════════════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────────────

st.sidebar.title("GEX Dashboard")
st.sidebar.markdown("Dealer Gamma Exposure Analysis")

mode = st.sidebar.radio("模式", ["單一標的", "多標的比較"], index=0)

if mode == "單一標的":
    symbol = st.sidebar.selectbox("標的", list(SYMBOLS.keys()), index=0,
                                   format_func=lambda s: f"{s} — {SYMBOLS[s]}")
    custom = st.sidebar.text_input("自訂標的（留空用上方選擇）", "")
    if custom.strip():
        symbol = custom.strip().upper()
    symbols = None
else:
    symbols = st.sidebar.multiselect("標的", list(SYMBOLS.keys()),
                                      default=["SPY", "QQQ", "IWM"],
                                      format_func=lambda s: f"{s} — {SYMBOLS[s]}")
    symbol = None

max_exp = st.sidebar.slider("最大到期日數量", 4, 20, 10,
                             help="越多越精確但越慢")

run = st.sidebar.button("計算 GEX", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**公式**
`GEX = OI × Gamma × 100 × S² × 0.01`

Dealer 賣出 Call → 負 Gamma
Dealer 賣出 Put → 正 Gamma

**正 Gamma 環境**: 造市商賣高買低，壓縮波動
**負 Gamma 環境**: 造市商追高殺低，放大波動
**Gamma Flip**: 正負翻轉的價格水位
""")

# ── Main content ──

if not run:
    st.title("📊 Dealer Gamma Exposure (GEX) Dashboard")
    st.markdown("""
    點擊左側 **「計算 GEX」** 開始分析。

    本工具從 Yahoo Finance 取得選擇權資料，計算造市商（Dealer）的 Gamma Exposure 分布，
    識別 **Gamma Flip**（正負翻轉水位）、**Put Wall**（支撐）和 **Call Wall**（壓力）。

    > **資料延遲約 15-20 分鐘**（Yahoo Finance 免費數據限制）
    """)
    st.stop()

if mode == "單一標的":
    with st.spinner(f"正在計算 {symbol} 的 GEX..."):
        try:
            data = fetch_gex(symbol, max_expirations=max_exp)
        except Exception as e:
            st.error(f"計算失敗: {e}")
            st.stop()
    render_single(data)

elif mode == "多標的比較":
    if not symbols:
        st.warning("請至少選擇一個標的")
        st.stop()
    with st.spinner(f"正在計算 {', '.join(symbols)} 的 GEX..."):
        results = fetch_multi_gex(symbols, max_expirations=max_exp)
    render_multi(results)
