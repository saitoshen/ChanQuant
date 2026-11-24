# -*- coding: utf-8 -*-
"""HL-Point 36 因子的非空 + PIT 验证。

用法：C:/ProgramData/anaconda3/python.exe _tools/hlpoint_verify.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttrade import factors as F  # noqa: E402
from ttrade import factors_hlpoint as HP  # noqa: E402

NEW33 = [
    "dev_z_20", "dev_z_60", "dev_z_120", "dev_z_agg", "dev_spread",
    "up_sh", "dn_sh", "up_sh_atr", "dn_sh_atr", "up_sh_int", "dn_sh_int",
    "shadow_high_z", "shadow_low_z", "slope_atr", "r2", "slope_chg_atr",
    "dist_hi", "dist_lo", "ext_hi", "ext_lo", "ch_slope_atr",
    "hit_high", "hit_low", "clus_high", "clus_low", "level_high", "level_low",
    "oi_chg_5", "oi_chg_20", "oi_z_60", "oi_ratio_60", "oi_px_sync_5",
    "roll_chg_20",
]


def make_bars(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """合成 1h bar（含夜盘），只为验证算式可跑 + 覆盖率，不代表真实行情。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n, freq="h")
    idx = pd.DatetimeIndex([t for t in idx if 9 <= t.hour <= 23 or t.hour >= 21][:n])
    n = len(idx)
    ret = rng.normal(0, 0.004, n)
    close = 3000.0 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 0.003, n)) + 0.001
    high = close * (1 + spread)
    low = close * (1 - spread)
    openp = np.r_[close[0], close[:-1]]
    vol = rng.integers(500, 5000, n).astype(float)
    oi = np.cumsum(rng.normal(0, 300, n)) + 200000
    # 展期三腿：让价差随时间漂，否则 roll_chg_20 恒 0、压不到差分逻辑。
    # 🔴 必须是「**一天一个快照**、铺满当天每根 bar」—— 真实 aux 就是这样
    # （`auxdata.attach_intraday` 把日频快照 shift(1) 后广播到当天）。
    # 做成逐根随机游走的话，`_day_diff` 里 `groupby(day).last()` 会取到当天
    # 最后一根（也就是未来）的值，测试会误报 PIT 失败。
    day = pd.DatetimeIndex(idx).normalize()
    udays = pd.unique(day)
    ry_daily = pd.Series(np.cumsum(rng.normal(0, 0.0008, len(udays))), index=udays)
    ry = ry_daily.reindex(day).to_numpy()
    df = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close,
                       "volume": vol, "open_interest": oi,
                       # 展期三腿（模拟 aux 列）
                       "o6_1_close": close * (1.0 + ry),
                       "o6_2_close": close * (0.998 + ry),
                       "o6_3_close": close * (0.995 + ry),
                       "o6_1_dte": 20.0, "o6_2_dte": 140.0, "o6_3_dte": 260.0},
                      index=idx)
    df["trading_day"] = pd.DatetimeIndex(df.index).normalize()
    return df


def main() -> int:
    print("因子库总数:", len(F.FACTORS))
    miss = [k for k in NEW33 if k not in F.FACTORS]
    print("新因子缺失:", miss if miss else "无")

    bars = make_bars()
    print(f"样本: {len(bars)} 根 1h bar  {bars.index[0]} ~ {bars.index[-1]}")
    print()

    hdr = f"{'因子':<16}{'有效':>7}{'占比':>8}{'均值':>12}{'标准差':>12}{'最小':>12}{'最大':>12}"
    print(hdr)
    print("-" * len(hdr))
    bad = []
    for k in NEW33:
        s = F.FACTORS[k].func(bars)
        v = pd.to_numeric(s, errors="coerce")
        ok = int(v.notna().sum())
        pct = ok / len(bars)
        if ok == 0:
            bad.append((k, "全 NaN"))
            print(f"{k:<16}{ok:>7}{pct:>7.1%}   <<< 全 NaN")
            continue
        if pct < 0.5:
            bad.append((k, f"覆盖率仅 {pct:.1%}"))
        print(f"{k:<16}{ok:>7}{pct:>7.1%}{v.mean():>12.4f}{v.std():>12.4f}"
              f"{v.min():>12.4f}{v.max():>12.4f}")
    print()

    # ---- PIT 硬校验 1：尾部截断重算 ----
    print("PIT 校验（一）截断尾根重算，取重叠段末 200 根比对")
    for k in ["hit_high", "hit_low", "clus_high", "clus_low", "level_high", "level_low",
              "dev_z_agg", "slope_chg_atr", "oi_ratio_60"]:
        full = pd.to_numeric(F.FACTORS[k].func(bars), errors="coerce")
        cut = len(bars) - 1
        part = pd.to_numeric(F.FACTORS[k].func(bars.iloc[:cut]), errors="coerce")
        a = full.iloc[:cut].tail(200).values
        b = part.tail(200).values
        m = np.isfinite(a) & np.isfinite(b)
        d = float(np.max(np.abs(a[m] - b[m]))) if m.any() else float("nan")
        flag = "OK" if (not np.isfinite(d) or d < 1e-9) else "🔴 变了"
        print(f"  {k:<16} maxdiff={d:.3e}  {flag}")

    # ---- PIT 硬校验 2：逐点扩历史重算（**这条才能抓 OLS 错位**）----
    #
    # 上一条只截掉最后一根，对「窗口整体前移」这类错误完全无感：被比较的两份
    # 数据用的是同一套错位下标，值自然一致。实测就是这样漏过了 `slope_atr`
    # 的下标 bug（每个斜率偷了未来 60 根，横截面 IC 0.64）。
    # 正确做法：第 t 行的值只能由 `df[:t+1]` 算出来。
    print()
    print("PIT 校验（二）逐点扩历史重算（第 t 行只准用 df[:t+1]）")
    pts = [600, 1200, 1800]
    bad_pts = []
    for k in NEW33:
        full = pd.to_numeric(F.FACTORS[k].func(bars), errors="coerce")
        worst = 0.0
        for t in pts:
            if t >= len(bars):
                continue
            one = pd.to_numeric(F.FACTORS[k].func(bars.iloc[:t + 1]), errors="coerce")
            v1, v2 = full.iloc[t], one.iloc[t]
            if pd.isna(v1) and pd.isna(v2):
                continue
            if pd.isna(v1) != pd.isna(v2):
                worst = float("inf")
                break
            worst = max(worst, abs(float(v1) - float(v2)))
        if worst > 1e-8:
            bad_pts.append((k, worst))
            print(f"  {k:<16} 🔴 maxdiff={worst:.3e}")
    if bad_pts:
        print(f"  ⇒ {len(bad_pts)} 个因子看了未来：{[b[0] for b in bad_pts]}")
    else:
        print(f"  全部 {len(NEW33)} 个因子在 {pts} 三个时点上都只用了过去数据 ✓")

    print()
    if bad:
        print("问题项:")
        for k, why in bad:
            print("  ", k, why)
        return 1
    if bad_pts:
        return 1
    print("全部 33 个新因子有效且 PIT 干净。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
