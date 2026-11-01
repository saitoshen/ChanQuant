"""期权 T 型报价：把一条链压成「行权价 × 认购/认沽」的一张表。

期权库里存的是**长表**（每个合约一行一天），T 型报价要的是**横截面**：同一行
权价上，认购和认沽并排。这个模块只做这一件事 —— 取数、对齐、反解 IV。

为什么 IV 要现场反解而不是读 `option_iv` 表：那张表是**按品种聚合**的（平值 IV、
偏度、曲率），拿来画波动率曲面够用，但报价表要的是**每一档行权价**各自的 IV。
真要看微笑，必须逐合约反解。

**反解要小心三件事**：

1. 价格低于内在价值（深度实值、临近到期）时二分法会一路走到边界，得到的
   `IV_LO` 是噪声不是信号 —— 这类档位标成 NaN，界面上显示「—」
2. 没有成交量、持仓量也是 0 的档位，`close` 往往是上一笔成交价或者交易所给
   的参考价，反解出来的 IV 全是毛刺 —— 保留数值但标记出来
3. `T` 用**自然日 / 365**，不是交易日。期权定价的时间是日历时间，按 252 算
   会把两个月的期权算成 2.4 个月

标的价格优先取 `option_underlying` 的当日收盘；没有就退回品种级 IV 表里最近
一天的记录（那条记录反解时用的 F 也是对的）。两边都没有就整张表不出 IV。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ttrade import optstore as OS
from ttrade import options as OPT

# 反解出来贴边界的 IV 一律当噪声。二分法的搜索区间就是这两个值，落在端点上
# 说明「按这个价格，真实波动率在区间外」—— 要么是深度实值，要么价格是脏数据
_IV_EDGE = (OPT.IV_LO + 1e-3, OPT.IV_HI - 1e-3)

# 一条链最多取多少档。股指期权一个月能有 200 多档，全画出来表格要滚很久，
# 而且离平值太远的档位根本没流动性
MAX_STRIKES = 400


def chains(variety: str | None = None) -> pd.DataFrame:
    """可选的期权链清单。`variety` 为空就是全库。

    和 `optstore.list_months` 的区别只在**出错时的表达**：那边返回 None 表示
    「库被占着」，界面要分得清「正在下载」和「没有数据」。
    """
    return OS.list_months(variety)


def varieties() -> list[str]:
    """库里有期权数据的品种。"""
    return list(OS.varieties())


def _last_two(df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    """每个合约最后两行的 `col`，用来算涨跌。

    不能用 `df.groupby(...).tail(1)` 再去找前一行：一条链上不同合约的最后一天
    可能不一样（有的档位三个月没成交），按整表取「最后两天」会把没成交的合约
    整段丢掉。逐合约取尾部两行才对。
    """
    if df.empty:
        return df
    d = df.sort_values(["contract", "dt"])
    g = d.groupby("contract", sort=False)[col]
    d["prev"] = g.shift(1)
    return d.groupby("contract", sort=False).tail(1)


def snapshot(variety: str, underlying: str, expiry=None,
             period: str = "1d", max_strikes: int = MAX_STRIKES) -> dict:
    """一条链的 T 型报价。

    返回字典：

        day         数据日期（最后一个有数据的交易日）
        spot        标的价格
        expiry      到期日
        dte         剩余自然日
        rows        表：strike / c_last / c_chg / c_vol / c_oi / c_iv /
                        p_last / p_chg / p_vol / p_oi / p_iv
        atm         平值行权价（离 spot 最近的那一档）
        note        给界面显示的一句话（数据缺什么、IV 为什么是空的）
        error       出错原因，非空时 rows 为空

    不抛异常：界面直接拿 `error` 显示，比 try/except 到处包一层干净。
    """
    out = {"day": None, "spot": None, "expiry": None, "dte": None,
           "rows": pd.DataFrame(), "atm": None, "note": "", "error": ""}
    if not OS.exists():
        out["error"] = ("本地还没有期权库 —— 去「数据管理 → 天勤」勾上"
                        "「一并更新期权」，或命令行跑 download_options.py")
        return out

    strikes = OS.list_strikes(variety, underlying, expiry)
    if strikes is None or strikes.empty:
        out["error"] = f"{variety} / {underlying} 这条链上没有合约"
        return out

    contracts = [str(c) for c in strikes["contract"]]
    bars = OS.load_bars(contracts, period)
    if bars.empty:
        out["error"] = (f"{variety} / {underlying} 有 {len(contracts)} 个合约，"
                        f"但一条 {period} K 线都没有")
        return out

    last = _last_two(bars, "close")
    last = last[np.isfinite(pd.to_numeric(last["close"], errors="coerce"))]
    if last.empty:
        out["error"] = "这条链上所有合约的收盘价都是空的"
        return out
    day = pd.Timestamp(last["dt"].max())
    out["day"] = day

    exp = pd.Timestamp(expiry) if expiry is not None else None
    if exp is None:
        # 没指定到期日就取合约元信息里的那一个。一条链只有一个月，随便拿一行
        # 都是对的；带错的情况是「标的自带月份但传了别的 expiry」，那也不该到这里
        meta = OS.load_meta(varieties=[variety])
        hit = meta[meta["underlying"] == underlying]
        exp = pd.Timestamp(hit["expiry"].iloc[0]) if len(hit) else None
    out["expiry"] = exp
    dte = int((exp - day).days) if exp is not None else None
    out["dte"] = dte

    spot = OS.underlying_close(underlying, period)
    out["spot"] = float(spot) if spot else None

    # ---- 宽表：一行一个行权价，C / P 各几列
    m = last.merge(strikes[["contract", "strike", "cp"]], on="contract", how="left")
    m["strike"] = pd.to_numeric(m["strike"], errors="coerce")
    m = m[np.isfinite(m["strike"])]
    m["cp"] = m["cp"].astype(str).str.upper().str[:1]
    m = m.sort_values("strike")
    if max_strikes and len(m) > max_strikes:
        # 太多档就只留离平值最近的这些。按「离 spot 的距离」截，而不是按
        # 行权价排序截 —— 后者会只留下一边
        mid = float(spot) if spot else float(m["strike"].median())
        keep = (m["strike"] - mid).abs().sort_values().index[:max_strikes]
        m = m.loc[keep].sort_values("strike")

    # ---- IV：一次向量化反解
    iv = np.full(len(m), np.nan)
    if spot and exp is not None and dte is not None and dte > 0:
        px = pd.to_numeric(m["close"], errors="coerce").to_numpy(dtype=float)
        kk = m["strike"].to_numpy(dtype=float)
        call = (m["cp"] == "C").to_numpy()
        t = max(dte, 1) / 365.0
        ok = np.isfinite(px) & (px > 0) & np.isfinite(kk)
        if ok.any():
            got = OPT.implied_vol(px[ok], float(spot), kk[ok], t, call[ok])
            got = np.where((got <= _IV_EDGE[0]) | (got >= _IV_EDGE[1]), np.nan, got)
            iv[ok] = got
    m["iv"] = iv

    def _side(cp: str, name: str, col: str) -> pd.Series:
        sub = m[m["cp"] == cp].set_index("strike")[col]
        sub.name = name
        return sub

    rows = pd.DataFrame(index=pd.Index(sorted(m["strike"].unique()), name="strike"))
    for cp, tag in (("C", "c"), ("P", "p")):
        for col, name in (("close", "last"), ("prev", "prev"), ("volume", "vol"),
                          ("close_oi", "oi"), ("iv", "iv")):
            rows[f"{tag}_{name}"] = _side(cp, f"{tag}_{name}", col)
    for tag in ("c", "p"):
        prev = rows[f"{tag}_prev"]
        last_px = rows[f"{tag}_last"]
        rows[f"{tag}_chg"] = (last_px - prev).where(prev > 0)
        rows[f"{tag}_chg_pct"] = (last_px / prev - 1.0).where(prev > 0)
    rows = rows.drop(columns=[c for c in rows.columns if c.endswith("_prev")])
    rows = rows.reset_index()

    if spot:
        i = (rows["strike"] - float(spot)).abs().idxmin()
        out["atm"] = float(rows.loc[i, "strike"])

    out["rows"] = rows
    if not spot:
        out["note"] = "缺标的价格，算不出 IV —— 期权库里没有这条标的的 K 线"
    elif not dte or dte <= 0:
        out["note"] = f"这条链已于 {exp:%Y-%m-%d} 到期，没有剩余期限，算不出 IV"
    else:
        n_iv = int(rows["c_iv"].notna().sum() + rows["p_iv"].notna().sum())
        if n_iv == 0:
            out["note"] = "全部档位的 IV 都贴到搜索边界（深度实值或价格异常），已置空"
    return out


def contract_bars(contract: str, period: str = "1d") -> pd.DataFrame:
    """单个期权合约的 K 线（`dt` 作索引）。给「双击某档看 K 线」用。"""
    from ttrade import optfetch as OF

    df = OF.load_contract(contract, period)
    if df is None or df.empty:
        return pd.DataFrame()
    if "dt" in df.columns:
        df = df.set_index("dt")
    return df.sort_index()
