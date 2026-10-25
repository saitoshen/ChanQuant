"""行情数据质量修复。

东方财富美股后复权数据存在三类已知瑕疵，会污染回测：
1. 复权因子在某日整体跳变（如 NKE 2007-03-19 价格凭空翻倍）；
2. 单日脏数据（如 NKE 2012-06-29、AXP 2008-01-02 单日腰斩次日复原）；
3. 休市日被塞入 volume=0、OHLC 全相等的占位行（如 MSFT 2014-04-18 耶稣受难日）。

处理原则：只做「可判定为数据错误」的修复，绝不抹掉真实的大幅波动。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COLS = ["open", "high", "low", "close"]


def drop_placeholder_rows(df: pd.DataFrame) -> pd.DataFrame:
    """剔除休市日占位行：成交量为 0 且四价完全相同。"""
    if df.empty:
        return df
    flat = (
        (df["open"] == df["high"])
        & (df["high"] == df["low"])
        & (df["low"] == df["close"])
    )
    zero_vol = df["volume"].fillna(0) <= 0
    return df[~(flat & zero_vol)]


def repair_jumps(df: pd.DataFrame, jump: float = 0.35, report: list | None = None) -> pd.DataFrame:
    """修复复权因子跳变与单日脏数据。

    jump: 单日对数收益阈值，超过即视为可疑
    判定：
      - 次日反向且幅度相当 -> 单日脏数据，删除该行
      - 之后走势恢复常态     -> 复权因子跳变，对后续价格整体等比修正
    """
    d = df.copy()
    fixed = 0
    for _ in range(30):
        c = d["close"].to_numpy(dtype=float)
        if len(c) < 5:
            break
        lr = np.diff(np.log(np.where(c > 0, c, np.nan)))
        abs_lr = np.abs(lr)
        cand = np.where(abs_lr > jump)[0]
        if len(cand) == 0:
            break

        k = int(cand[0])  # 跳变发生在第 k+1 行
        date_bad = d.index[k + 1]

        # 单日脏数据：次日反向且幅度接近
        if k + 1 < len(lr) and lr[k] * lr[k + 1] < 0 and abs_lr[k + 1] > jump * 0.6:
            d = d.drop(index=date_bad)
            fixed += 1
            if report is not None:
                report.append(f"删除单日脏数据 {date_bad:%Y-%m-%d} (日变动 {np.expm1(lr[k])*100:+.0f}%)")
            continue

        # 复权因子跳变：后续价格整体除以跳变因子，使序列连续
        factor = float(np.exp(lr[k]))
        if not np.isfinite(factor) or factor <= 0:
            break
        for col in PRICE_COLS:
            arr = d[col].to_numpy(dtype=float).copy()
            arr[k + 1:] = arr[k + 1:] / factor
            d[col] = arr
        fixed += 1
        if report is not None:
            report.append(f"修正复权跳变 {date_bad:%Y-%m-%d} (因子 {factor:.3f})")

        # 防御：若修正后该点仍超阈值（浮点或连续异常），强制删除该行避免死循环
        c2 = d["close"].to_numpy(dtype=float)
        if k + 1 < len(c2) and abs(np.log(c2[k + 1] / c2[k])) > jump:
            d = d.drop(index=date_bad)
            if report is not None:
                report.append(f"强制剔除异常点 {date_bad:%Y-%m-%d}")
        if fixed >= 30:
            break
    return d


def clean_prices(df: pd.DataFrame, report: list | None = None) -> pd.DataFrame:
    """完整清洗流程。"""
    out = drop_placeholder_rows(df)
    out = repair_jumps(out, report=report)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out
