# -*- coding: utf-8 -*-
"""2×2 归因：**因子实现** 还是 **数据/品种池** 造成的 IC 差异？

三列对照（同一份 IC 管线：日盘快照 + 每 5 交易日 + Spearman）：

| 列 | 因子函数 | 数据 |
|---|---|---|
| `原报表` | future_quant 原函数 | 原项目自己的 MySQL 面板 |
| `原fn`  | **future_quant 原函数** | **本机 tq 1h（和 CQ 同一份）** |
| `CQ`    | ChanQuant 库函数 | 本机 tq 1h |

- `原fn ≈ 原报表`：数据/池不是问题 ⇒ 差异出在库的因子实现；
- `原fn ≈ CQ`：因子实现没问题 ⇒ 差异出在数据源 / 品种池。

用法：C:/ProgramData/anaconda3/python.exe _tools/hlpoint_ic_origfn.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "D:/ChanQuant-Dev")
sys.path.insert(0, "D:/ChanQuant-Dev/_tools")
sys.path.insert(0, "D:/CozeTask/future_quant")

from scipy.stats import spearmanr  # noqa: E402

from strategy.hl_point import indicators as I2, levels as L2  # noqa: E402
from strategy.hl_point.config import HLConfig  # noqa: E402
from ttrade import auxdata, tqdata  # noqa: E402

from hlpoint_ic_cmp import HS, KEYS, START, END, snap_positions, xs_ic_table  # noqa: E402
from hlpoint_full import all_symbols  # noqa: E402

ROLL_CSV = "D:/CozeTask/future_quant/data/roll_yield_panel_all87.csv"

#: 原项目列名 → 本脚本统一用的键名（只有展期 3 个不同）
ORIG2KEY = {"roll": "roll_main_oi", "roll_ann": "roll_main_annual",
            "curv": "roll_main_curv"}
CFG = HLConfig(timeframe="1h", use_mtf=False, use_corr=False)
_ROLL = None


def roll_panel() -> pd.DataFrame:
    global _ROLL
    if _ROLL is None:
        p = pd.read_csv(ROLL_CSV, parse_dates=["trade_date"])
        p["variety"] = p["variety"].astype(str).str.upper()
        _ROLL = p
    return _ROLL


def oi_features(close: np.ndarray, oi: np.ndarray) -> dict:
    """照抄 `tools_ml._oi_features`（5 个持仓量因子，全部无量纲）。"""
    s = pd.Series(np.asarray(oi, float))
    c = pd.Series(np.asarray(close, float))
    out = {
        "oi_chg_5": (s / s.shift(5) - 1.0).values,
        "oi_chg_20": (s / s.shift(20) - 1.0).values,
    }
    r1 = s.pct_change()
    mu = r1.rolling(60, min_periods=60).mean()
    sd = r1.rolling(60, min_periods=60).std()
    out["oi_z_60"] = ((r1 - mu) / sd).values
    out["oi_ratio_60"] = (s / s.rolling(60, min_periods=60).mean()).values
    out["oi_px_sync_5"] = (np.sign(c.pct_change(5)) * np.sign(s.pct_change(5))).values
    return out


CACHE = "reports/_origfn_store.pkl"


def main() -> int:
    rows = all_symbols()
    if os.environ.get("HP_REUSE", "1") == "1" and os.path.exists(CACHE):
        import pickle
        store = pickle.load(open(CACHE, "rb"))
        print(f"复用缓存 {CACHE}（{len(store)} 个字段）")
        return report(store)
    store: dict = {}
    n_ok = 0
    for i, (sym, ex, _n) in enumerate(rows, 1):
        try:
            df = tqdata.load_main(sym, ex, "1h", adjust="qfq")
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 500:
            continue
        df = auxdata.attach_for_period(df, sym, ex, "1h")
        idx = df.index
        pos = snap_positions(idx)
        if len(pos) < 50:
            continue
        sday = idx.normalize().to_numpy()[pos]
        m = (sday >= START.to_datetime64()) & (sday <= END.to_datetime64())
        if m.sum() < 50:
            continue
        pos, sday = pos[m], pd.DatetimeIndex(sday[m])

        feats = I2.build_features(df, CFG)
        feats = feats.join(L2.level_features(df, feats, CFG))
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
        oi = pd.to_numeric(df.get("open_oi", df.get("open_interest")),
                           errors="coerce").to_numpy(float)
        oifs = oi_features(close, oi)

        # 展期：原项目 `_roll_features` 口径 —— 按 trade_date 取面板，**shift(1)**
        # 用上一交易日（PIT 保守），再 reindex 到 bar 的交易日
        sub = roll_panel()
        sub = sub[sub["variety"] == sym.upper()].sort_values("trade_date")
        sub = sub.set_index("trade_date")[["roll", "roll_ann", "curv"]].copy()
        sub["roll_chg_20"] = sub["roll"] - sub["roll"].shift(20)
        sub = sub.shift(1)
        day_key = pd.DatetimeIndex(idx.normalize())

        for c in KEYS:
            src = next((o for o, k in ORIG2KEY.items() if k == c), c)
            if c in ("roll_main_oi", "roll_main_annual", "roll_main_curv", "roll_chg_20"):
                v = sub[src].reindex(day_key).to_numpy(float) if len(sub) else np.full(len(df), np.nan)
            elif c in oifs:
                v = np.asarray(oifs[c], float)
            elif c in feats.columns:
                v = pd.to_numeric(feats[c], errors="coerce").to_numpy(float)
            else:
                continue
            store.setdefault(c, {})[sym] = pd.Series(v[pos], index=sday)

        for h in HS:
            p2 = pos + h
            ok = p2 < len(close)
            fwd = np.full(len(pos), np.nan)
            fwd[ok] = close[p2[ok]] / close[pos[ok]] - 1.0
            store.setdefault(f"__y{h}__", {})[sym] = pd.Series(fwd, index=sday)
        n_ok += 1
        if i % 10 == 0:
            print(f"  [{i}/{len(rows)}] {sym}_{ex} ({n_ok} 个可用)", flush=True)
    print(f"载入 {n_ok} 个品种")

    import pickle
    pickle.dump(store, open(CACHE, "wb"), protocol=4)
    return report(store)


def report(store: dict) -> int:
    res = {h: xs_ic_table(store, h) for h in HS}
    orig = pd.read_csv("D:/CozeTask/future_quant/backtest/record/"
                       "hl_point_factor_report_20261015d.csv")
    # 原报表里叫 roll/roll_ann/curv，本脚本统一用库里的键名
    orig["key"] = orig["factor"].map(lambda f: ORIG2KEY.get(f, f))
    o = orig.set_index("key")

    out = pd.DataFrame(index=KEYS)
    out["原IC"] = o["ic"].reindex(KEYS)
    out["原t"] = o["t"].reindex(KEYS)
    for h in HS:
        out[f"原fn{h}"] = res[h]["ic"].reindex(KEYS)
        out[f"原fn{h}_t"] = res[h]["t"].reindex(KEYS)
    cq = pd.read_csv("reports/hlpoint_ic_cmp.csv", index_col=0)
    out["CQ46"] = cq["CQ46_IC"].reindex(KEYS)
    out["CQ30"] = cq["CQ30_IC"].reindex(KEYS)
    out.to_csv("reports/hlpoint_ic_origfn.csv", encoding="utf-8-sig")

    print("\n" + "=" * 92)
    for h in HS:
        a = out["原IC"].to_numpy(float)
        b = out[f"原fn{h}"].to_numpy(float)
        c2 = out[f"CQ{h}"].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        m2 = np.isfinite(b) & np.isfinite(c2)
        print(f"H={h}  原报表 vs 原fn（同函数、换数据）: 秩相关 {spearmanr(a[m], b[m]).statistic:+.3f} "
              f"同号 {int((np.sign(a[m]) == np.sign(b[m])).sum())}/{int(m.sum())}")
        print(f"H={h}  原fn  vs CQ  （同数据、换函数）: 秩相关 {spearmanr(b[m2], c2[m2]).statistic:+.3f} "
              f"同号 {int((np.sign(b[m2]) == np.sign(c2[m2])).sum())}/{int(m2.sum())}")
    print("=" * 92)
    print(f"\n{'因子':<18}{'原IC':>9}{'原t':>7}{'原fn46':>9}{'t':>7}{'CQ46':>9}{'差(fn-CQ)':>11}")
    print("-" * 70)
    for k in KEYS:
        r = out.loc[k]
        print(f"{k:<18}{r['原IC']:>+9.4f}{r['原t']:>7.2f}{r['原fn46']:>+9.4f}"
              f"{r['原fn46_t']:>7.2f}{r['CQ46']:>+9.4f}"
              f"{(r['原fn46'] - r['CQ46']):>+11.4f}")
    print("\n写出 reports/hlpoint_ic_origfn.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
