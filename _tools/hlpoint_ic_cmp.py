# -*- coding: utf-8 -*-
"""逐因子 IC 对照：ChanQuant（本机移植版）vs 原项目（future_quant/strategy/hl_point）。

## 口径怎么对齐的

原项目的锚点在 `strategy/hl_point/tools_factor_report.py::ic_ir()`：

- 面板 `HL_ML_TF=1h`、`HL_ML_H_DAYS=5`、`HL_ML_EXTRA=oi,roll`，2016-01-01 ~ 2026-10-14；
- 截面取 **日盘快照**：只留 `8 <= hour < 16` 的 bar，对 (日历日, 品种) 取**最后一根**
  （`tools_wf_v2.daily_snap`）；
- 步长 **每 5 个交易日**一次截面（`STEP_D=5`），持有期 `H = 5 × 6 = 30` 根；
- IC = 每个截面 `Spearman(factor, 未来 H 根原始收益)` 的均值，t = mean/(std/√n)。

本脚本**照抄这套口径**，只在「持有期根数」上给两档：

- `h46`：5 天 × 实测 9.28 根/天 ⇒ 46 根。**经济持有期与原项目一致**（5 个交易日）；
- `h30`：原项目写死的 `BPD=6` ⇒ 30 根。**根数一致、经济持有期偏短**（≈3.2 天）。

两档一起看，就能把「持有期口径差异」和「因子实现 / 数据差异」分开。

## 差异来源（对照时心里要有数）

| 项 | 原项目 | 本机 ChanQuant |
|---|---|---|
| 品种池 | DB 内全 888（含金融，金融截面宽度 <10 成不了组） | 68 个商品，无金融 |
| 数据库 | 自己的 MySQL 面板 | `tqdata` 1h（qfq 复权） |
| 展期 3 因子 | 未平滑日快照 | 库里多一道 5 日均值 |

用法：
    C:/ProgramData/anaconda3/python.exe _tools/hlpoint_ic_cmp.py
    HP_MAXN=20 ...        # 只跑前 N 个品种（调试 / 省内存）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.stats import spearmanr  # noqa: E402

from ttrade import auxdata, factors as F, tqdata  # noqa: E402
from hlpoint_full import all_symbols  # noqa: E402

ORIG_CSV = ("D:/CozeTask/future_quant/backtest/record/"
            "hl_point_factor_report_20261015d.csv")
START = pd.Timestamp("2016-01-01")
END = pd.Timestamp("2026-10-14")
STEP_D = 5          # 每 5 个交易日一次截面（原项目 STEP_D）
HS = (30, 46)       # 持有期根数：原项目写死 30 / 本机实测密度 46
MINW = 10           # 截面最少品种数

#: ChanQuant 因子键 → 原项目因子名（只有展期 3 个是映射过去的）
KEY2ORIG = {
    "roll_main_oi": "roll",
    "roll_main_annual": "roll_ann",
    "roll_main_curv": "curv",
}

#: 36 个因子（默认 34 个 + ext_hi/ext_lo 两个价格水平量；对照原项目 36 行要全开）
KEYS = ["dev_z_20", "dev_z_60", "dev_z_120", "dev_z_agg", "dev_spread",
        "up_sh", "dn_sh", "up_sh_atr", "dn_sh_atr", "up_sh_int", "dn_sh_int",
        "shadow_high_z", "shadow_low_z", "slope_atr", "r2", "slope_chg_atr",
        "dist_hi", "dist_lo", "ext_hi", "ext_lo", "ch_slope_atr",
        "hit_high", "hit_low", "clus_high", "clus_low", "level_high", "level_low",
        "oi_chg_5", "oi_chg_20", "oi_z_60", "oi_ratio_60", "oi_px_sync_5",
        "roll_main_oi", "roll_main_annual", "roll_main_curv", "roll_chg_20"]


def snap_positions(idx: pd.DatetimeIndex) -> np.ndarray:
    """日盘快照：`8 <= hour < 16` 的 bar 里，取每(日历日)最后一根的位置。"""
    hour = idx.hour.to_numpy()
    day = idx.normalize().to_numpy()
    keep = (hour >= 8) & (hour < 16)
    pos = np.flatnonzero(keep)
    if not len(pos):
        return pos
    d = day[pos]
    # 每个交易日的最后一根：日期变了的下一根就是分界
    last = np.append(d[1:] != d[:-1], True)
    return pos[last]


def one_symbol(sym: str, ex: str, store: dict, days: dict) -> bool:
    """跑一个品种：36 个因子只保留日盘快照那几行，攒进 store。"""
    try:
        df = tqdata.load_main(sym, ex, "1h", adjust="qfq")
    except Exception as exc:  # noqa: BLE001
        print(f"  {sym}_{ex} 读取失败: {exc}")
        return False
    if df is None or len(df) < 500:
        return False
    df = auxdata.attach_for_period(df, sym, ex, "1h")
    idx = df.index
    pos = snap_positions(idx)
    if len(pos) < 50:
        return False
    day = idx.normalize()
    sday = day[pos]
    m = (sday >= START) & (sday <= END)
    if m.sum() < 50:
        return False
    pos, sday = pos[m], sday[m]
    store.setdefault("__day__", {})[sym] = pd.DatetimeIndex(sday)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    for k in KEYS:
        spec = F.FACTORS.get(k)
        if spec is None:
            continue
        v = pd.to_numeric(spec.func(df), errors="coerce").to_numpy(float)
        store.setdefault(k, {})[sym] = pd.Series(v[pos], index=sday)
    for h in HS:
        p2 = pos + h
        ok = p2 < len(close)
        fwd = np.full(len(pos), np.nan)
        fwd[ok] = close[p2[ok]] / close[pos[ok]] - 1.0
        store.setdefault(f"__y{h}__", {})[sym] = pd.Series(fwd, index=sday)
    store.setdefault("__bars__", {})[sym] = len(df)
    days[sym] = sday
    return True


def neff(a: np.ndarray, k: int = 5) -> float:
    """IC 序列的**有效样本量**（自相关修正）。

    🔴 逐因子 IC 的 t 值很容易被高估：`roll` / `curv` 这类慢变量在品种间的排序
    几个月都不变，490 个截面里的「独立信息」远少于 490。原项目的 `t` 也是
    `mean/(std/√n)`，同样没修正 —— 所以拿它的 t 当「强因子」的证据会误导。
    """
    n = len(a)
    x = a - a.mean()
    v = float((x * x).sum())
    if n < 20 or v <= 0:
        return float(n)
    rho = 0.0
    for j in range(1, min(k, n - 1) + 1):
        rho += float((x[:-j] * x[j:]).sum()) / v
    rho = max(-0.95, rho / k)
    return float(n / (1.0 + 2.0 * rho * k))


def xs_ic_table(store: dict, h: int) -> pd.DataFrame:
    """按每 STEP_D 个交易日取截面，逐因子算 Spearman IC / t。"""
    ymap = store[f"__y{h}__"]
    # 宽表：行 = 交易日（全池并集，和原项目 `dates = sorted(S['date'].unique())` 一致），
    # 列 = 品种。`DataFrame(dict)` 会自动按索引并集对齐。
    Y = pd.DataFrame(ymap).sort_index()
    pick = Y.index[::STEP_D]
    Y = Y.reindex(pick).to_numpy(float)
    out = {}
    for k in KEYS:
        if k not in store:
            continue
        Xm = pd.DataFrame(store[k]).reindex(index=pick).to_numpy(float)
        ics = []
        for i in range(len(pick)):
            xs, ys = Xm[i], Y[i]
            m = np.isfinite(xs) & np.isfinite(ys)
            if m.sum() < MINW or np.ptp(xs[m]) == 0:
                continue
            ic = spearmanr(xs[m], ys[m]).statistic
            if np.isfinite(ic):
                ics.append(float(ic))
        a = np.asarray(ics, float)
        if len(a) < 10:
            out[k] = dict(ic=np.nan, ir=np.nan, t=np.nan, n=len(a), pos=np.nan,
                          neff=np.nan, tadj=np.nan)
            continue
        mu, sd = float(a.mean()), float(a.std(ddof=1))
        ne = neff(a)
        out[k] = dict(ic=mu, ir=mu / sd if sd else np.nan,
                      t=mu / (sd / np.sqrt(len(a))) if sd else np.nan,
                      n=len(a), pos=float((a > 0).mean()),
                      neff=ne,
                      tadj=mu / (sd / np.sqrt(ne)) if sd else np.nan)
    return pd.DataFrame(out).T


def main() -> int:
    maxn = int(os.environ.get("HP_MAXN", "0") or 0)
    rows = all_symbols()
    if maxn:
        rows = rows[:maxn]
    store: dict = {}
    days: dict = {}
    t0 = pd.Timestamp.now()
    for i, (sym, ex, _n) in enumerate(rows, 1):
        ok = one_symbol(sym, ex, store, days)
        if i % 5 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] {sym}_{ex} {'ok' if ok else '-'} "
                  f"({(pd.Timestamp.now() - t0).total_seconds():.0f}s)", flush=True)
    print(f"载入 {len(days)} 个品种")

    res = {h: xs_ic_table(store, h) for h in HS}
    orig = pd.read_csv(ORIG_CSV)
    orig["key"] = orig["factor"].map(lambda f: {v: k for k, v in KEY2ORIG.items()}.get(f, f))

    out = pd.DataFrame(index=KEYS)
    out["group"] = orig.set_index("key")["group"].reindex(KEYS)
    out["name"] = orig.set_index("key")["name"].reindex(KEYS)
    out["原_IC"] = orig.set_index("key")["ic"].reindex(KEYS)
    out["原_t"] = orig.set_index("key")["t"].reindex(KEYS)
    out["原_n"] = orig.set_index("key")["n"].reindex(KEYS)
    for h in HS:
        out[f"CQ{h}_IC"] = res[h]["ic"].reindex(KEYS)
        out[f"CQ{h}_t"] = res[h]["t"].reindex(KEYS)
        out[f"CQ{h}_tadj"] = res[h]["tadj"].reindex(KEYS)
        out[f"CQ{h}_neff"] = res[h]["neff"].reindex(KEYS)
        out[f"CQ{h}_n"] = res[h]["n"].reindex(KEYS)
    out["ΔIC(46)"] = out["CQ46_IC"] - out["原_IC"]
    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/hlpoint_ic_cmp.csv", encoding="utf-8-sig")

    # ---- 一致性统计 ----
    print("\n" + "=" * 88)
    for h in HS:
        a = out["原_IC"].to_numpy(float)
        b = out[f"CQ{h}_IC"].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        r = spearmanr(a[m], b[m]).statistic
        same = int((np.sign(a[m]) == np.sign(b[m])).sum())
        print(f"H={h}: 可比因子 {int(m.sum())}/36 | IC 秩相关 {r:+.3f} | "
              f"同号 {same}/{int(m.sum())} | "
              f"|IC| 均值 原 {np.abs(a[m]).mean():.4f} vs CQ {np.abs(b[m]).mean():.4f}")
    a = out["原_IC"].to_numpy(float)
    print(f"原项目 |t|>2 : {int((np.abs(out['原_t'].to_numpy(float)) > 2).sum())} 个")
    for h in HS:
        t = out[f"CQ{h}_t"].to_numpy(float)
        ta = out[f"CQ{h}_tadj"].to_numpy(float)
        print(f"CQ{h} |t|>2 : {int((np.abs(t) > 2).sum())} 个 "
              f"| 自相关修正后 |t_adj|>2 : {int((np.abs(ta) > 2).sum())} 个 "
              f"| 有效样本量中位 {np.nanmedian(out[f'CQ{h}_neff'].to_numpy(float)):.0f}")
    print("=" * 88)

    print(f"\n{'因子':<18}{'原IC':>9}{'原t':>7}{'CQ46_IC':>9}{'t':>7}{'t_adj':>7}"
          f"{'ΔIC':>9}{'CQ30_IC':>9}{'t30':>7}")
    print("-" * 84)
    for k in KEYS:
        r = out.loc[k]
        print(f"{k:<18}{r['原_IC']:>+9.4f}{r['原_t']:>7.2f}"
              f"{r['CQ46_IC']:>+9.4f}{r['CQ46_t']:>7.2f}{r['CQ46_tadj']:>7.2f}"
              f"{r['ΔIC(46)']:>+9.4f}{r['CQ30_IC']:>+9.4f}{r['CQ30_t']:>7.2f}")
    print(f"\n写出 reports/hlpoint_ic_cmp.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
