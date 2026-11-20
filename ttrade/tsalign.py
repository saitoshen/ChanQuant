"""时间戳精度归一：`merge_asof` 之前两侧必须先同一个精度。

🔴 为什么单独一个模块：pandas 2.x 的时间精度是**跟着数据来源走**的，
同一个平台上并存着两套 ——

* DuckDB 读出来的表是 `datetime64[us]`（`tqdata.load_main` / 研究页 / 回测）；
* 天勤推上来的 K 线是 `datetime64[ns]`（`tqfetch.local_time`，「ns → 时区」）；
* `pd.Timedelta(0)`、`.normalize()` 在旧代码里曾把 ns 悄悄降成 us。

`pd.merge_asof(left, right, on="t")` 要求两侧那一列**精度完全一致**，否则
不是「差一点」而是直接抛 `MergeError: incompatible merge keys [0]
dtype('<M8[ns]') and dtype('<M8[us]')`。

2026-11-20 现场（白银 / 铅 1d_tq「理论持仓还有空单」）：实盘帧是
`merge_live(库内 us, 实时 ns)` → 整帧变 **ns**；基本面因子走
`factors_ext._asof` → `merge_asof` 抛异常 → 该因子被静默剔除 → 少一票 →
分数从 +0.46 掉到 −0.55 → `sig` 0 变 −1。研究页只用库内数据（us），
所以**同一个策略、同一段行情，研究页空仓、实盘 −2 手**。

判据只能有一处，所以四个 `merge_asof` 调用点（`factors_ext` /
`factors_fx5` ×2 / `factors_macro`）全部改走 `merge_asof_ns`，别再各写一遍
`astype("datetime64[ns]")`。
"""

from __future__ import annotations

import pandas as pd

#: 归一目标精度。选 ns 而不是 us：ns 是 pandas 的默认精度，往上转永不丢信息，
#: 往 us 转在亚微秒数据上会静默截断（天勤的 bar 时间戳就是 ns）。
NS = "datetime64[ns]"


def to_ns(values) -> pd.DatetimeIndex:
    """任何时间序列 → `datetime64[ns]` 的 DatetimeIndex。

    接受 `Series` / `Index` / `ndarray` / 列表。`NaT` 原样保留（调用方自己
    `dropna`）—— 这里不做清洗，隐藏缺值会让「少了一天」变成「悄悄用前一天」。
    """
    return pd.DatetimeIndex(values).astype(NS)


def ns_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """把 `df[col]` 就地归一到 ns（返回同一个对象，省一次拷贝）。"""
    if col in df.columns:
        df[col] = to_ns(df[col])
    return df


def merge_asof_ns(left: pd.DataFrame, right: pd.DataFrame, on: str = "t",
                  **kwargs) -> pd.DataFrame:
    """两侧 `on` 列先归一到 ns 再 `merge_asof`（其余参数原样透传）。

    约定：**右表是源**（外部序列，一天一个值），**左表是目标**（bar 时间戳）。
    结果按左表原有的行序返回吗？—— 不。`merge_asof` 本来就要求左表按键升序，
    这里统一 `sort_values(on)`，调用方要原序就自己带一列 `_i` 再排回来
    （`factors_fx5` 就是这么做的）。
    """
    left = left.copy()
    right = right.copy()
    if on not in left.columns or on not in right.columns:
        raise KeyError(f"merge_asof_ns: 缺列 {on!r}（左 {list(left.columns)}）")
    ns_column(left, on)
    ns_column(right, on)
    return pd.merge_asof(left.sort_values(on), right.sort_values(on), on=on,
                         **kwargs)
