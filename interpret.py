"""
Market interpretation engine for GEX data.
Generates human-readable analysis of gamma exposure distribution.
"""

import numpy as np
import pandas as pd


def interpret_gex(data: dict) -> dict:
    """
    Generate comprehensive market interpretation from GEX data.

    Returns dict with:
      - summary: one-line verdict
      - regime_analysis: gamma regime explanation
      - support_resistance: support/resistance level analysis
      - volatility_outlook: expected volatility behavior
      - key_scenarios: what-if scenarios
      - trading_implications: actionable insights
    """
    spot = data['spot']
    regime = data['regime']
    kl = data['key_levels']
    gex_df = data['gex_by_strike']
    total_gex = data['total_gex']
    gamma_flip = kl.get('gamma_flip')
    put_wall = kl.get('put_wall')
    call_wall = kl.get('call_wall')
    max_gamma = kl.get('max_gamma_strike')

    # ── Spot vs Gamma Flip position ──
    if gamma_flip:
        flip_dist_pct = (gamma_flip / spot - 1) * 100
        above_flip = spot > gamma_flip
    else:
        flip_dist_pct = None
        above_flip = None

    # ── GEX concentration analysis ──
    top_strikes = gex_df.nlargest(5, 'net_gex')[['strike', 'net_gex']]
    bottom_strikes = gex_df.nsmallest(5, 'net_gex')[['strike', 'net_gex']]

    # Find where most gamma is concentrated (within what % range of spot)
    near_spot = gex_df[(gex_df['strike'] >= spot * 0.97) & (gex_df['strike'] <= spot * 1.03)]
    near_gex_pct = near_spot['net_gex'].sum() / total_gex * 100 if total_gex != 0 else 0

    # ── Build interpretations ──
    result = {}

    # 1. Summary
    if regime == 'positive':
        if gamma_flip and flip_dist_pct is not None and abs(flip_dist_pct) < 2:
            result['summary'] = f"⚠️ 正 Gamma 但接近 Flip 水位（距 {flip_dist_pct:+.1f}%），需警惕轉負風險"
        else:
            result['summary'] = "🟢 正 Gamma 環境 — 造市商壓縮波動，市場傾向區間震盪"
    else:
        if gamma_flip and flip_dist_pct is not None and abs(flip_dist_pct) < 2:
            result['summary'] = f"⚠️ 負 Gamma 但接近 Flip 水位（距 {flip_dist_pct:+.1f}%），可能即將轉正"
        else:
            result['summary'] = "🔴 負 Gamma 環境 — 造市商放大波動，趨勢可能加速"

    # 2. Regime analysis
    regime_text = []
    if regime == 'positive':
        regime_text.append(
            f"**當前為正 Gamma 環境**（Total GEX: {total_gex/1e9:+.2f}B）。"
            "造市商持有正 Gamma 部位，當價格上漲時他們需要賣出避險、下跌時買入，"
            "形成「均值回歸」效果。這通常會壓低盤中波動，價格傾向在特定區間內震盪。"
        )
        if max_gamma:
            regime_text.append(
                f"\n\n**Max Gamma 集中在 ${max_gamma:,.0f}**（距現價 {(max_gamma/spot-1)*100:+.1f}%），"
                "這個價位具有「磁吸效應」，價格會傾向被拉回此處。"
            )
    else:
        regime_text.append(
            f"**當前為負 Gamma 環境**（Total GEX: {total_gex/1e9:+.2f}B）。"
            "造市商持有負 Gamma 部位，被迫順勢避險（漲時買、跌時賣），"
            "形成正回饋循環，放大價格波動。市場容易出現急漲急跌。"
        )

    if gamma_flip:
        if above_flip:
            regime_text.append(
                f"\n\n現價在 Gamma Flip（${gamma_flip:,.1f}）**之上** {abs(flip_dist_pct):.1f}%。"
            )
            if regime == 'positive':
                regime_text.append(
                    "只要維持在 Flip 上方，正 Gamma 的穩定效果持續。"
                    f"若跌破 ${gamma_flip:,.0f} 則可能進入負 Gamma，波動急增。"
                )
        else:
            regime_text.append(
                f"\n\n現價在 Gamma Flip（${gamma_flip:,.1f}）**之下** {abs(flip_dist_pct):.1f}%。"
            )
            if regime == 'negative':
                regime_text.append(
                    "在 Flip 下方的負 Gamma 區域，下跌趨勢可能自我強化。"
                    f"需回到 ${gamma_flip:,.0f} 上方才能恢復穩定。"
                )

    result['regime_analysis'] = ''.join(regime_text)

    # 3. Support / Resistance
    sr_text = []
    sr_text.append("### 支撐與壓力水位\n")

    # Resistance levels (above spot)
    sr_text.append("**壓力位（Resistance）：**\n")
    if call_wall and call_wall > spot:
        dist = (call_wall / spot - 1) * 100
        sr_text.append(
            f"- **Call Wall ${call_wall:,.0f}**（距現價 +{dist:.1f}%）— "
            "最大 Call Gamma 集中處，造市商在此大量避險賣壓，價格難以突破。"
            "若突破此位可能觸發 short squeeze / gamma squeeze。\n"
        )
    if gamma_flip and gamma_flip > spot:
        dist = (gamma_flip / spot - 1) * 100
        sr_text.append(
            f"- **Gamma Flip ${gamma_flip:,.1f}**（距現價 +{dist:.1f}%）— "
            "正負 Gamma 翻轉水位，突破後市場行為模式會改變。\n"
        )

    # Add top positive GEX strikes above spot as resistance
    above_spot = gex_df[(gex_df['strike'] > spot) & (gex_df['net_gex'] > 0)]
    if not above_spot.empty:
        top_above = above_spot.nlargest(2, 'net_gex')
        for _, row in top_above.iterrows():
            if row['strike'] != call_wall:
                dist = (row['strike'] / spot - 1) * 100
                sr_text.append(
                    f"- **${row['strike']:,.0f}**（+{dist:.1f}%）— "
                    f"正 GEX 集中（{row['net_gex']/1e6:,.0f}M），造市商賣壓區。\n"
                )

    sr_text.append("\n**支撐位（Support）：**\n")
    if put_wall and put_wall < spot:
        dist = (put_wall / spot - 1) * 100
        sr_text.append(
            f"- **Put Wall ${put_wall:,.0f}**（距現價 {dist:.1f}%）— "
            "最大 Put Gamma 集中處，造市商在此大量買入避險，形成強支撐。"
            "若跌破可能引發 put-driven 加速下跌。\n"
        )
    if gamma_flip and gamma_flip < spot:
        dist = (gamma_flip / spot - 1) * 100
        sr_text.append(
            f"- **Gamma Flip ${gamma_flip:,.1f}**（距現價 {dist:.1f}%）— "
            "跌破此位將進入負 Gamma 環境，波動可能急劇放大。\n"
        )

    # Add top positive GEX strikes below spot as support
    below_spot = gex_df[(gex_df['strike'] < spot) & (gex_df['net_gex'] > 0)]
    if not below_spot.empty:
        top_below = below_spot.nlargest(2, 'net_gex')
        for _, row in top_below.iterrows():
            if row['strike'] != put_wall:
                dist = (row['strike'] / spot - 1) * 100
                sr_text.append(
                    f"- **${row['strike']:,.0f}**（{dist:.1f}%）— "
                    f"正 GEX 集中（{row['net_gex']/1e6:,.0f}M），造市商買盤區。\n"
                )

    result['support_resistance'] = ''.join(sr_text)

    # 4. Volatility outlook
    vol_text = []
    if regime == 'positive':
        vol_text.append(
            "**預期低波動** — 正 Gamma 壓縮盤中波幅，"
            "適合賣方策略（short straddle/strangle）。"
        )
        if near_gex_pct > 50:
            vol_text.append(
                f"\n\nGamma 高度集中在現價附近（±3% 範圍佔 {near_gex_pct:.0f}%），"
                "磁吸效應強烈，價格可能被釘在這個區間。"
            )
    else:
        vol_text.append(
            "**預期高波動** — 負 Gamma 放大價格波動，"
            "盤中可能出現較大幅度的單邊走勢。適合買方策略或趨勢跟蹤。"
        )

    # Expiry concentration
    exp_df = data['gex_by_expiry']
    if not exp_df.empty:
        nearest_exp = exp_df.iloc[0]
        nearest_pct = nearest_exp['net_gex'] / total_gex * 100 if total_gex != 0 else 0
        if abs(nearest_pct) > 30:
            vol_text.append(
                f"\n\n⚠️ **最近到期日（{nearest_exp['expiry']}）佔 Total GEX 的 {nearest_pct:.0f}%**。"
                "到期日當天 gamma 會急劇衰減（charm effect），可能導致 GEX 分布大幅改變。"
            )

    result['volatility_outlook'] = ''.join(vol_text)

    # 5. Key scenarios
    scenarios = []
    if gamma_flip:
        if above_flip:
            scenarios.append(
                f"**若跌破 Gamma Flip（${gamma_flip:,.0f}）：** "
                "正 → 負 Gamma 翻轉，造市商從「穩定器」變成「放大器」，"
                "下跌可能加速，波動急增。"
            )
            if call_wall and call_wall > spot:
                scenarios.append(
                    f"\n\n**若突破 Call Wall（${call_wall:,.0f}）：** "
                    "大量 Call 被迫 delta hedge 買入，可能觸發向上 gamma squeeze，"
                    "短期快速上漲。"
                )
        else:
            scenarios.append(
                f"**若漲回 Gamma Flip（${gamma_flip:,.0f}）上方：** "
                "負 → 正 Gamma 翻轉，造市商開始壓縮波動，市場可能穩定。"
            )
            if put_wall and put_wall < spot:
                scenarios.append(
                    f"\n\n**若跌破 Put Wall（${put_wall:,.0f}）：** "
                    "大量 Put 被迫 delta hedge 賣出，可能觸發向下加速，"
                    "形成 put-driven crash 風險。"
                )

    result['key_scenarios'] = '\n'.join(scenarios) if scenarios else "目前無特殊情境需要關注。"

    # 6. Levels table for display
    levels = []
    if call_wall and call_wall > spot:
        levels.append({
            'type': 'resistance', 'name': 'Call Wall',
            'price': call_wall, 'dist_pct': (call_wall/spot-1)*100,
            'strength': '🔴🔴🔴',
        })
    if gamma_flip and gamma_flip > spot:
        levels.append({
            'type': 'resistance', 'name': 'Gamma Flip',
            'price': gamma_flip, 'dist_pct': (gamma_flip/spot-1)*100,
            'strength': '🟡🟡🟡',
        })
    if gamma_flip and gamma_flip <= spot:
        levels.append({
            'type': 'support', 'name': 'Gamma Flip',
            'price': gamma_flip, 'dist_pct': (gamma_flip/spot-1)*100,
            'strength': '🟡🟡🟡',
        })
    if put_wall and put_wall < spot:
        levels.append({
            'type': 'support', 'name': 'Put Wall',
            'price': put_wall, 'dist_pct': (put_wall/spot-1)*100,
            'strength': '🟢🟢🟢',
        })
    if max_gamma:
        mg_type = 'resistance' if max_gamma > spot else 'support'
        levels.append({
            'type': mg_type, 'name': 'Max Gamma',
            'price': max_gamma, 'dist_pct': (max_gamma/spot-1)*100,
            'strength': '⚡⚡⚡',
        })

    # Sort: resistance descending, then support descending
    levels.sort(key=lambda x: -x['price'])
    result['levels'] = levels

    return result
