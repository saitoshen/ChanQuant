# -*- coding: utf-8 -*-
"""验证两个 GUI bug 修复：截面阶段分析量纲、组合报告切策略腿交易记录。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    from ttrade import cross as CX
    from ttrade.backtest import BacktestConfig, combine_trades, stage_table

    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2016-01-04", periods=600)
    frames = {}
    for k in range(4):
        px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
        frames[f"S{k}_SHFE"] = pd.DataFrame(
            {"open": px, "high": px * 1.01, "low": px * 0.99, "close": px,
             "volume": np.full(len(idx), 1e5), "open_interest": np.full(len(idx), 1e4)},
            index=idx)
    cfg = CX.CrossConfig(mode="hedge", factor_mode="single",
                         factor_keys=["mom_20"], top_pct=40, bot_pct=40,
                         weight="equal", rebalance=10)
    bt = BacktestConfig(initial_cash=100_000.0, fee_rate=0.0003, slippage=0.0)
    out = CX.run(frames, cfg, bt)
    assert out.ok, out.error
    res = CX.to_result(out, label="截面", step=out.step, bpy=252.0,
                       initial_cash=bt.initial_cash)

    # --- 验收 1：主 equity 首值 = 真金本金，阶段分析不再是 -100%
    first = float(res.equity.iloc[0])
    print(f"主 equity 首值 = {first:,.2f}（期望 {bt.initial_cash:,.0f}）")
    assert abs(first - bt.initial_cash) < 1e-6, "主 equity 量纲没修"
    for sym, lr in res.legs.items():
        lf = float(lr.equity.iloc[0])
        assert abs(lf - bt.initial_cash) < 1e-6, f"腿 {sym} 首值 {lf} 不对"
    yrs = stage_table(res.equity, "YE", base=bt.initial_cash)
    print("年度收益（前 3 年）：")
    print(yrs[["period", "ret"]].head(3).to_string(index=False))
    assert all(abs(r) < 2.0 for r in yrs["ret"]), "年度收益还有 -100% 这种量纲错"
    assert abs(float(yrs["ret"].sum()) - res.metrics["total_return"]) < 1e-6, \
        "各年相加 ≠ 累计收益"
    print(f"各年相加 = {yrs['ret'].sum():.6f}，累计收益 = {res.metrics['total_return']:.6f} ✓")

    # --- 验收 2：combine_trades 的 nt[0] 是策略名，按策略名能切出交易
    trades = combine_trades([("有色截面对冲", res), ("油脂截面对冲", res)])
    leg = "有色截面对冲"
    rows = [nt for nt in trades if str(nt[0]) == leg]
    print(f"combine_trades 共 {len(trades)} 笔；切「{leg}」得 {len(rows)} 笔")
    assert len(rows) > 0, "按策略名切交易仍为空"
    old = [nt for nt in trades
           if str(getattr(nt[1], "symbol", "") or "") == leg]
    print(f"旧逻辑按 Trade.symbol 切得 {len(old)} 笔（复现原 bug）✓")
    print("两个修复都通过")


if __name__ == "__main__":
    main()
