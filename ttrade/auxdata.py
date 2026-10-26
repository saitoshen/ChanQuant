"""衍生数据（期限结构 / 期权波动率）的读取与挂载。

## 为什么单独一层

`data_futures/` 里是主力连续合约，只有一条价格曲线。展期因子需要**不同到期日
的多个合约**，期权因子需要**隐含波动率**——这两类信息不在主力连续里，必须另外
从真实合约和期权明细聚合出来。

聚合结果放在 `data_futures_aux/`：

- `{SYM}_{EX}.csv`：期限结构，每日一行，近/次近/第三近合约的收盘价、到期天数、持仓量
- `opt_{SYM}_{EX}.csv`：期权波动率，每日一行，近/次月平值 IV、偏度、看跌看涨持仓比

## 为什么挂在 df 上，而不是给因子函数加参数

因子函数的签名是 `f(df) -> Series`，`factors.raw()` / `factor_frame()` /
`standardize()` 全部按这个约定工作。如果给展期因子单独加一个 `aux=` 参数，
就要在策略、机器学习、因子分析页三条链路上各加一次分支——分支越多越容易漏。

改成「加载时把衍生列并进 df」，因子函数退化成读列，其它代码一行都不用改。
美股没有这些列，读出来是 NaN，`filter_available()` 会自动把它们从美股因子
列表里剔掉，不需要额外的市场判断。

本模块只依赖 pandas 和标准库，**不导入 tushare**——GUI 启动时就会用到它。
"""

from __future__ import annotations

import os

import pandas as pd

# 期限结构列：近月 c1 / 次近 c2 / 第三近 c3
CURVE_COLS = [
    "c1_close", "c1_dte", "c1_oi",
    "c2_close", "c2_dte", "c2_oi",
    "c3_close", "c3_dte", "c3_oi",
]
# 期权列
OPT_COLS = ["iv_near", "iv_next", "dte_near", "dte_next", "skew", "curv", "pcr", "n_opt"]

_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def curve_path(aux_dir: str, symbol: str, exchange: str) -> str:
    return os.path.join(aux_dir, f"{symbol}_{exchange}.csv")


def opt_path(aux_dir: str, symbol: str, exchange: str) -> str:
    return os.path.join(aux_dir, f"opt_{symbol}_{exchange}.csv")


def _read(path: str, cols: list[str]) -> pd.DataFrame:
    """带 mtime 校验的读取缓存。更新数据后文件会变，缓存必须失效。"""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return pd.DataFrame()
    hit = _cache.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "date" not in df.columns or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("date").set_index("date").sort_index()
    _cache[path] = (mtime, df)
    return df


def load_curve(aux_dir: str, symbol: str, exchange: str) -> pd.DataFrame:
    return _read(curve_path(aux_dir, symbol, exchange), CURVE_COLS)


def load_opt(aux_dir: str, symbol: str, exchange: str) -> pd.DataFrame:
    return _read(opt_path(aux_dir, symbol, exchange), OPT_COLS)


def attach(
    df: pd.DataFrame,
    symbol: str,
    exchange: str,
    aux_dir: str | None = None,
) -> pd.DataFrame:
    """把衍生列按日期并到行情表上。缺文件时原样返回（不报错）。

    用 `join` 而不是 `merge`：df 已经按 date 建了索引，join 能保证不改变行数
    和顺序——K 线渲染、信号对齐都依赖「第 i 行就是第 i 根 K 线」。
    万一衍生文件里有行情表没有的日期，`how="left"` 会直接丢掉，不会多出行。
    """
    if aux_dir is None:
        aux_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data_futures_aux"
        )
    if not os.path.isdir(aux_dir) or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    parts = [load_curve(aux_dir, symbol, exchange), load_opt(aux_dir, symbol, exchange)]
    out = df
    for part in parts:
        if part.empty:
            continue
        cols = [c for c in part.columns if c not in out.columns]
        if not cols:
            continue
        out = out.join(part[cols], how="left")
    return out


def clear_cache() -> None:
    _cache.clear()


def summary(aux_dir: str) -> dict:
    """统计衍生数据覆盖情况，给界面和自检用。"""
    import json

    out = {"curve": 0, "opt": 0, "dates": 0, "first": "", "last": ""}
    if not os.path.isdir(aux_dir):
        return out
    for fn in os.listdir(aux_dir):
        if fn.startswith("opt_") and fn.endswith(".csv"):
            out["opt"] += 1
        elif fn.endswith(".csv") and not fn.startswith("_"):
            out["curve"] += 1
    man = os.path.join(aux_dir, "_manifest.json")
    if os.path.exists(man):
        try:
            with open(man, encoding="utf-8") as fh:
                dates = json.load(fh).get("dates") or []
            if dates:
                out["dates"] = len(dates)
                out["first"], out["last"] = dates[0], dates[-1]
        except Exception:
            pass
    return out
