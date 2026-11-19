"""汇率 5m 因子：Dukascopy 的 USDCNH / DXY **分钟线**，与日频 `macro_*` 完全分开。

## 为什么要单独一套

`macro_usdcnh` / `macro_dxy` 走的是 Tushare 日线：一天一个值，日内恒定，所以
5m 上一整天的信号只有一个方向，且 D+1 08:00 才更新（见 `factors_macro.AVAIL_08`）。
Dukascopy 的分钟线是**实时报价**，5 分钟一个值 —— 日内有信息，这是它唯一
但有决定性的优势。两者必须并存对比，不能互相替代：日频那套的样本从 2016 年
就有，分钟线这套只有 2020-01 起。

## 🔴🔴 时间对齐：三层「时差」，都来自数据源本身

这里不是「代码写错了」，而是三种来源不同的时间差叠加。混在一起看，很容易
把其中一层误当成 bug 去「修」，把另一层当成小事放过。

### ① 到达时差（所有品种都有，固定 1 根）

判据只有一句（与 #164 全库口径一致）：

    **源 bar 的「结束时刻」≤ 基准 bar 的时间戳**

本系统的 bar 时间戳是**开始时刻**（天勤口径），bar `t` 覆盖 `[t, t+5m)`，
所以「基准 bar 的时间戳」= 上一根 bar 的结束时刻。

Dukascopy 的 bar 时间戳**也是开始时刻**，实测可以确认：外汇周五 21:00 UTC
（夏令时）/ 22:00 UTC（冬令时）收盘 ⇒ 北京 05:00 / 06:00，而本地数据里每周
最后一根 bar 的时间戳众数是 **04:55 / 05:55** —— 正好是「覆盖到 05:00 为止」
的那一根。若是结束时刻标注，众数就会是 05:00 / 06:00。

**⚠️ 注意：时间戳本身没有错位。** 用 lead-lag 实测（2026-11-19，USDCNH 5m
收益 vs AU/AG/CU/SC 5m 收益，k=−4..+4）：相关峰值**全部落在 k=0**
（AU −0.0725、AG −0.1375、CU −0.1416），k=±1 时几乎为 0。也就是说同一时间戳
的 5m 收益是**物理同期**的，没有任何平移。

真正的问题是**到达时间**：`r_fx[t]` 覆盖 `[t, t+5m)`，但它在 `t` 时刻还没
生成出来 —— 要到 `t+5m` 那根开始，这根 bar 才闭合、才可读。

    bar 09:00 若用 fx 09:00 的 close：那根 09:05 才收完 → 在 09:00 拿不到

所以「滞后一根」不是给数据打补丁，而是**让因子只使用当时真正还在手里的
信息**。用「事件时间」说这叫前视，用「到达时间」说这叫时差，两种说法指向
同一个操作。实测代价：不做这一步，h=1 的 IC 会从 +0.0061 虚增到 +0.0155
（AU `chg_6`，2.5 倍），信号层面 30 分钟因子有 14.2% 的仓位是「用还没到的
价格算出来的」。

### ② 日历时差（只有 DXY 有，会随美国夏令时移动 1 小时）

`DOLLAR.IDX-USD` 是 CFD，**每天只交易 21 小时**，休市 3 小时。关键在
**休市窗口不是固定在 UTC，而是跟着纽约收盘走**（实测）：

| 季节 | 北京休市窗口 | 恢复的第一根 |
|---|---|---|
| 美国夏令时（3 月第 2 周日 ~ 11 月第 1 周日） | 05:00 – 07:59 | 08:00 |
| 美国冬令时 | 06:00 – 08:59 | 09:00 |

对中国期货的直接影响只有一个格子：**商品日盘 09:00 的开盘第一根**。

- 夏天：09:00 时手上是刚收完的 08:55 → 时差 5 分钟，正常
- 冬天：09:00 时手上只有 **05:55** → **时差 3 小时**

实测 2020-12-15：基准 bar 09:00 的 `chg_48` = +0.000762，与该日 05:55 那根
完全相等。这是数据源在该时段确实不报价，**无法修正，也不该硬补** —— 补了
就是编数据。

（USDCNH 没有这个问题：它 24 小时连续，实测夏冬两季都没有日内休市窗口。）

### ③ 跨缺口（周末 / 休市累计，靠掩码挡掉）

`chg_n` 若跨越缺口（第 i 根与第 i−n 根间隔 > 1.6×n×5min）置 NaN。否则周一
第一根算出来的是 60 小时的累计变动，周一开盘后的 `chg_48` 算的是「跨了一个
周末的 4 小时变动」。

这条规则对 DXY 还有一个副作用：**每天开盘后的头 48 根（4 小时）`chg_48`
全是 NaN**，因为窗口一定跨过那 3 小时休市。2020 年 dxy 的 `chg_48` 覆盖率
只有 70.7%，就是这条造成的；`chg_6` / `chg_12` / `z_48` / `rv_48` 都在
94~99%。这不是缺陷，是「4 小时前」在休市日确实没有对应的价格。

## 其它已知口径

- 跨缺口掩码：`chg_n` 若跨越周末（第 i 根与第 i−n 根间隔 > 1.6×n×5min）置 NaN，
  否则周一第一根算出来的是 60 小时的累计变动。
- 陈旧值保护：最近一根可用 bar 距基准超过 `MAX_STALE_BARS`（4 小时）就置 NaN。
  dxy 的 2022~2025 整段缺失，没有这条保护会把 2025 的基准 bar 接上 2021 的
  汇率值，看着「有数」其实是死值。
- 时区：Dukascopy 原生 UTC，固定 +8 到北京时间（中国无夏令时，不受伦敦夏令时
  影响；只有数据本身是伦敦墙上时间才需要按 `Europe/London` 换算，见
  `_tools/duka_fx/duka_fx.py::to_beijing`）。
- 数据文件缺失时全部返回 NaN，`factors.filter_available()` 会把它们从下拉里
  剔掉 —— 与 `lpr_*` 的处理一致，不报错。
- ⚠️ Dukascopy 的 `DOLLAR.IDX-USD` 是 **CFD 报价**，不是 ICE 的 DXY 结算价，
  水平可能差几十个基点、每天只交易 21 小时；做方向可以，做「精确点位」不行。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

BAR = "5min"
BAR_TD = pd.Timedelta(BAR)
BARS_PER_DAY = 288

# 🔴 日线在「北京几点」切开（见 `_daily_pct` 的说明）。
# 0 = 北京自然日（旧口径）；6 = 纽约日（外汇真实的交易日边界，冬令时准确）。
DAY_CUT_HOUR = 6

# 源 bar 从「时间戳」到「真正收完可读」的间隔。正常必须等于一根 bar
# （见 `_asof` 的说明）；置 0 只用于做「越界版」对照实验，**不要在生产里改**。
AVAIL_SHIFT = BAR_TD

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data_duka")

# code -> (文件名, 中文名)
_SYMS = {
    "usdcnh": ("USDCNH_5m.parquet", "美元兑离岸人民币"),
    "dxy": ("DOLLARIDX_5m.parquet", "美元指数"),
}

# 最近一根可用 bar 距基准超过这个根数就置 NaN（见模块说明的「陈旧值保护」）。
MAX_STALE_BARS = 48

_cache: dict = {}          # code -> {"bars", "sig", "fac", "daily"}


# ── 数据 ────────────────────────────────────────────────────────────────────
def _load(code: str) -> pd.DataFrame | None:
    """读本地 5m 行情（带 mtime 缓存，补数据后不用重启进程）。"""
    fn = _SYMS.get(code, (None, None))[0]
    if not fn:
        return None
    path = os.path.join(DATA_DIR, fn)
    try:
        st = os.stat(path)
    except OSError:
        return None
    sig = (st.st_mtime, st.st_size)
    c = _cache.setdefault(code, {"bars": None, "sig": None, "fac": None, "daily": None})
    if c["bars"] is not None and c["sig"] == sig:
        return c["bars"]
    try:
        bars = pd.read_parquet(path)
    except Exception:
        return None
    if not len(bars) or "close" not in bars:
        return None
    bars = bars[["open", "high", "low", "close"]].sort_index()
    # ⚠️ 这里必须把 daily 一起置空：补数据后 mtime 变了，5m 因子和日线分位
    # 都要重算，只清 fac 会让日频那份继续用旧数据（静默不更新）。
    c.update(bars=bars, sig=sig, fac=None, daily=None, daily_avail=None)
    return bars


def _span_ok(idx: pd.DatetimeIndex, n: int, tol: float = 1.6) -> pd.Series:
    """第 i 根往前 n 根的时间跨度是否正常（没跨周末/长假）。"""
    d = pd.Series(idx, index=idx).diff(n)
    return (d <= BAR_TD * n * tol) & d.notna()


def _build(code: str) -> pd.DataFrame | None:
    """算一次全部候选因子并缓存（按文件 mtime 失效）。"""
    c = _cache.setdefault(code, {"bars": None, "sig": None, "fac": None, "daily": None})
    if c["fac"] is not None:
        return c["fac"]
    bars = _load(code)
    if bars is None:
        return None
    px = bars["close"].astype(float)
    idx = bars.index
    out = pd.DataFrame(index=idx)

    for n in (1, 3, 6, 12, 48):
        out["chg_%d" % n] = (px / px.shift(n) - 1.0).where(_span_ok(idx, n))

    # 相对 48 根（4 小时）的 z：均值/标准差只用**当根之前**的 48 根
    m = px.rolling(48, min_periods=36).mean().shift(1)
    s = px.rolling(48, min_periods=36).std().shift(1)
    out["z_48"] = ((px - m) / s.replace(0, np.nan)).where(_span_ok(idx, 1))

    r1 = (px / px.shift(1) - 1.0).where(_span_ok(idx, 1))
    out["rv_48"] = r1.rolling(48, min_periods=36).std() * np.sqrt(BARS_PER_DAY * 252)

    # 扩张分位：与 factors_macro 同口径（只用过去），但这个在 5m 上是分钟分位，
    # 变化极慢，主要作为「汇率水平」的代理
    out["pct"] = px.expanding(min_periods=BARS_PER_DAY * 20).rank(pct=True)

    c["fac"] = out
    return out


# ── 日线口径（5m 聚合） ─────────────────────────────────────────────────────
def _daily_pct(code: str) -> pd.Series | None:
    """把 5m 聚合成**日线**，再算扩张分位 —— 口径与 `macro_<code>` 一模一样。

    口径与 `macro_<code>` 一致（都是扩张历史分位），差别在**日线怎么切**：这里按
    `DAY_CUT_HOUR` 切（默认 **6**，即外汇真实的交易日边界 —— 纽约 17:00 收盘），
    Tushare 按它自己的日历（实测相对北京自然日整体晚约 1 天，见下）。

    🔴 为什么不在北京 00:00 切（旧口径）：外汇在北京时间周一 05:00/06:00 才开盘，
    00:00 切会把「北京周六 00:00~04:55」单独切成一条**只有 60 根的伪日线**
    （6 年 356 条，占 16.6%），它的收盘价≈周五收盘价，是重复样本。按 06:00 切之后
    周六那一条消失、归到周五，一周 5~6 条里有几条是残的这个问题就没了。

    ⚠️ 与 `macro_usdcnh` 的差别，2026-11-19 实测（10 品种 / 2020 起）：
    日末值相关 **0.99836**、差值中位 −0.9bp；**分位逐位差均值 0.0130、中位
    0.0073、>0.2 占 0.0%、相关 0.9981**。也就是说两边给出的分位**几乎相同**
    —— 早期版本这里写过「分位平均差 0.2134、48% 交易日差 >0.2」，那是拿
    `fx5_usdcnh_pct`（**分钟粒度**的分位，1877 个唯一值）去比日频分位，
    口径不同，**结论不成立，已作废**。

    🔴🔴 真正需要注意的是**下游的标准化窗口**：`FactorConfig.window` 的单位是
    bar 根数（`F.win_bars` 不做周期换算）。**换天数不能除以 288** —— 5m 一天
    只有 AU/AG **111 根**、CU/SN **93 根**、IM **48 根**（商品日盘 4h + 夜盘 5.5h
    ≈ 9.5 小时，288 那是 24 小时连续外汇的密度）。所以 `window=545` 是
    **AU 4.9 天 / IM 11.4 天**，不是 1.9 天。

    而分位是慢变量，窗口内 std 小得离谱，**数据源之间那点差就被放大成信号**：

        window 根   AU 天数   窗口 std 中位   0.0130 的分位差 = 几个 σ
        252         2.3       0.00471         **2.8 σ**
        545         4.9       0.00722         1.8 σ
        1440        13.0      0.01197         1.1 σ

    滞回带 `entry_z=1.0 / exit_z=0.3` 才 0.7σ 宽，2.8σ 的输入差当然把信号全打乱。
    实测信号不一致率随窗口单调收敛：252 根 **41%** → 545 根 **30.6%** →
    2880 根 **6.5~12%** → 14400 根 **2.2~2.9%**。所以「换数据源后夏普从 +1.18
    掉到 +0.05」是**窗口放大**，不是数据源的锅 —— 同一个坑把 `macro_usdcnh`
    自己也坑了（e1.0/x0.3、Tushare·08:00）：

        window      252     545     1440    2880    5760
        AU 天数     2.3     4.9     13.0    25.9    51.9
        组合夏普   **+0.75** +0.53  +0.39   +0.20   +0.17
        单品种     1.36~2.62（252）→ 全线下滑

    **夏普随窗口单调下降、短窗口最高 —— 这正是可疑的地方，不是优点。**
    日频因子铺到分钟线上，window 按品种给 **每天根数 × N**，别照抄日线的 252/545。

    🔴 **那 `macro_usdcnh` 在短窗口下的 +1.x 可信吗？四个品种里三个不可信。**
    按**自然日**给它加扰动（σ=0.0065 / 0.0130，5 个随机种子取中位）：

        品种  window   原值   σ=0.0065  σ=0.0130   结论
        AU    252     +1.36    +0.56     **+0.26**   掉 1.10，基本清零
        AU    545     +1.11    +0.63     +0.24      同上
        AG    252     +1.90    +1.19     **+0.63**   掉 1.27
        IM    252     +2.62    +2.36     **+1.95**   只掉 0.67
        IM    1440    +1.44    +1.44     **+1.45**   完全不掉（min 1.10）← 真信号

    ⇒ **AU/AG 上的高夏普是「短窗口 + 精确数据源 + 精确阈值」凑出来的**；
    **IM 是真信号，而且在 30 天窗口（1440 根）下最稳**，短窗口对它只是虚增。
    参数也极敏感：`entry_z` 3.0→1.0 就让 AU 从 +0.54 变 +1.11。

    🔴🔴 **「提前 8 小时」这个卖点不成立（2026-11-19 复核，作废前一版结论）**。
    同一份 Duka 数据只改可用时刻：window=545 组合 **00:00 +0.37 vs 08:00 +0.40**；
    window=2880 **+0.08 vs +0.03**。提前 8 小时没有价值，甚至略差。
    前一版写的「+0.50 vs +0.44」是拿 **Tushare 数据假装 00:00 可用**测的 ——
    Tushare 的 D 行要 D+1 08:00 才发布，提前 8 小时用就是**偷未来**。同样地，
    真把它提前（Tushare·00:00）也只值 **+1.19 vs +1.03**（window=545 组合），
    **偷 8 小时 ≈ +0.16，不是 +0.79**（那个 +0.79 是同时换了数据源 + 分位起点，
    不是单变量）。

    ⚠️ **两边日历不一样，但不是夏普差异的主因**。四种切分点实测（2020 起，5 品种
    等权，e1.0/x0.3 / window=545）：

        A 北京00:00（旧口径）  组合 +0.38   周六 356 条伪日线
        B 北京08:00（=UTC 日） 组合 +0.43   伪日线挪到周日 358 条，比 A 更短（24~36 根）
        C 北京06:00（纽约冬）  组合 +0.44   周六 0 条、周日 195 条 ← **当前默认**
        D 北京05:00（纽约夏）  组合 +0.47   一周只有 5 条，但冬令时切错 1 小时

    改切分点只捞回 **+0.05~0.09**，而 Tushare 与 Duka 的整体差距是 **0.38 vs 1.03**，
    所以它**不是主要矛盾**。B 在 AG/CU/SN 上都更好、AU 反而从 +0.12 变 −0.20
    —— 改善本身也在噪声范围内，别指望换个切点就把两个数据源对齐。

    另外 Tushare 的日期标签相对北京自然日**整体晚约 1 天**：2026-06-06（周六）
    6.789500 ≈ Duka 2026-06-05（周五）6.789070。只留周一~周五重算，分位差反而
    **变大**（0.0130 → 0.0195），信号差几乎不变（30.6% → 28.7%）⇒ 剔周末不解决问题。
    """
    c = _cache.setdefault(code, {"bars": None, "sig": None, "fac": None,
                                 "daily": None, "daily_avail": None})
    if c.get("daily") is not None:
        return c["daily"]
    bars = _load(code)
    if bars is None:
        return None
    # 🔴 按 DAY_CUT_HOUR 切（0=北京自然日，6=纽约日）。北京 06:00 之后算下一天，
    # 于是「北京周一 05:00~05:55」「北京周六 05:00~05:55」各有归处，不会再长出
    # 一条只有 60 根的伪日线（旧口径 weekly 6 条里有 1 条是残的）。
    day = (bars.index - pd.Timedelta(hours=DAY_CUT_HOUR)).normalize()
    px = bars["close"].astype(float)
    daily = px.groupby(day).last().dropna()
    if len(daily) < 60:
        return None
    # 可用时刻 = 该组**最后一根 bar 收完**（+5min）。不再假设「次日 00:00 可用」：
    # 切分点一改，收盘落点就跟着变（06:00 切 ⇒ 收盘在次日 05:55/07:55），
    # 拿固定 00:00 当可用时刻就会在夏令时多推 2 小时。
    avail = (bars.index.to_series().groupby(day).last() + BAR_TD)
    avail = avail.reindex(daily.index)
    pct = daily.expanding(min_periods=60).rank(pct=True)
    pct.index = pd.DatetimeIndex(pct.index).astype("datetime64[ns]")
    c["daily"] = pct
    c["daily_avail"] = pd.DatetimeIndex(avail.to_numpy()).astype("datetime64[ns]")
    return pct


def _asof_daily(series: pd.Series, index, avail=None) -> pd.Series:
    """日频序列 → bar：只在源行的**可用时刻之后**才铺上去。

    `avail` 是逐行的可用时刻（= 该日最后一根 bar 收完）。传 None 时退回旧口径
    「索引次日 00:00 可用」，**只用于对照实验**，不要在生产里用。

    ⚠️ 这里不能用「索引日期 + 1 天」：`DAY_CUT_HOUR=6` 时索引是「北京 06:00 所属
    的自然日」，收盘落在次日 05:55~07:55，固定加一天会在夏令时早 2 小时。
    """
    if series is None or not len(series):
        return pd.Series(np.nan, index=index, dtype=float)
    if avail is not None:
        t_src = pd.DatetimeIndex(avail).astype("datetime64[ns]")
    else:
        t_src = (pd.DatetimeIndex(series.index).astype("datetime64[ns]")
                 + pd.Timedelta(days=1))
    right = pd.DataFrame({
        "t": t_src,
        "v": series.to_numpy(dtype=float),
    }).dropna(subset=["t"]).sort_values("t")
    left = pd.DataFrame({"t": pd.DatetimeIndex(index).astype("datetime64[ns]")})
    left["_i"] = np.arange(len(left))
    out = pd.merge_asof(left.sort_values("t"), right, on="t",
                        direction="backward")
    return pd.Series(out.sort_values("_i")["v"].to_numpy(dtype=float), index=index)


# ── 对齐 ────────────────────────────────────────────────────────────────────
def _asof(series: pd.Series, index, max_stale: int = MAX_STALE_BARS) -> pd.Series:
    """源因子（标注在 bar **开始时刻**）→ 基准 bar。

    🔴 可用时刻 = 源时间戳 + 5m（即该 bar 的结束时刻），再取
    `merge_asof(backward)`：基准 bar `T` 只拿「结束时刻 ≤ T」的最后一根。
    这就是「到达时差」那一层：同一时间戳的收益是物理同期的（lead-lag 实测
    峰值在 k=0），只是它在 `T` 时刻还没生成完，不能在 `T` 用。

    ⚠️ 不要用 `shift(1)` 再同索引对齐 —— fx 连续时两者等价，但 fx 有周末缺口
    时不等价：`shift` 是「上一行」，`merge_asof` 是「上一根已收完的」。
    """
    if series is None or not len(series):
        return pd.Series(np.nan, index=index, dtype=float)
    # 🔴 统一到 ns 精度再算：`pd.Timedelta(0)` 是 us 精度，加到 DatetimeIndex 上
    # 会把 `M8[ns]` 变成 `M8[us]`，与基准索引对不上，merge_asof 直接抛
    # MergeError（实测 2026-11-19 踩过）。
    avail = (pd.DatetimeIndex(series.index).astype("datetime64[ns]") + AVAIL_SHIFT)
    avail = avail.astype("datetime64[ns]")
    right = (pd.DataFrame({"t": avail, "v": series.to_numpy(dtype=float)})
             .dropna(subset=["t"]).sort_values("t"))
    left = pd.DataFrame(
        {"t": pd.DatetimeIndex(index).astype("datetime64[ns]")}).sort_index()
    left["_i"] = np.arange(len(left))
    tol = BAR_TD * int(max_stale) if max_stale else None
    out = pd.merge_asof(left.sort_values("t"), right, on="t",
                        direction="backward", tolerance=tol)
    out = out.sort_values("_i")
    return pd.Series(out["v"].to_numpy(dtype=float), index=index)


def _make(code: str, col: str):
    def f(df: pd.DataFrame) -> pd.Series:
        fac = _build(code)
        if fac is None or col not in fac:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return _asof(fac[col], df.index)

    f.__name__ = "f_fx5_%s_%s" % (code, col)
    return f


def _make_daily(code: str):
    def f(df: pd.DataFrame) -> pd.Series:
        pct = _daily_pct(code)
        if pct is None:
            return pd.Series(np.nan, index=df.index, dtype=float)
        avail = (_cache.get(code) or {}).get("daily_avail")
        return _asof_daily(pct, df.index, avail=avail)

    f.__name__ = "f_fx5d_%s_pct" % code
    return f


# key 后缀 -> 中文标签 / 说明
_SPECS = {
    "chg_6": ("30分钟变动", "近 30 分钟（6 根）涨跌幅，分钟级汇率脉冲。"),
    "chg_12": ("1小时变动", "近 1 小时（12 根）涨跌幅。"),
    "chg_48": ("4小时变动", "近 4 小时（48 根）涨跌幅。"),
    "z_48": ("4小时偏离", "相对前 4 小时均值的标准差倍数（z）。"),
    "rv_48": ("4小时已实现波动", "近 4 小时收益标准差，年化。"),
    "pct": ("汇率分位", "收盘价的扩张历史分位（分钟粒度，变化慢）。"),
}

CATEGORY = "汇率5m"


def register_into(FACTORS, FactorSpec) -> int:
    """注册 2 个品种 × 6 个 = 12 个 5m 汇率因子。

    数据缺失时仍注册，值全 NaN（`filter_available` 自动剔掉）。
    """
    formula = ("chg_n = close_t / close_{t-n} - 1（跨缺口置 NaN）；"
               "z = (close_t - MA48) / SD48（统计量只用 t 之前）；"
               "可用时刻 = bar 时间戳 + 5min（结束时刻）")
    n = 0
    for code, (_fn, cname) in _SYMS.items():
        for col, (label, desc) in _SPECS.items():
            key = "fx5_%s_%s" % (code, col)
            FACTORS[key] = FactorSpec(
                key, "汇率5m·%s%s" % (cname, label), _make(code, col),
                CATEGORY, desc, formula=formula,
            )
            n += 1
        # 日线口径：5m 聚合成日线再算扩张分位。与 `macro_<code>` **同口径**
        # （都是 rank(pct=True)），但可用时刻早 8 小时（次日 00:00 vs 08:00）。
        key = "fx5d_%s_pct" % code
        FACTORS[key] = FactorSpec(
            key, "汇率日·%s分位(5m聚合)" % cname, _make_daily(code),
            CATEGORY,
            "5m 行情按纽约日（北京 %02d:00 切）聚合成日线后的扩张历史分位；"
            "口径同 macro_%s，但日线由行情自己聚合。" % (DAY_CUT_HOUR, code),
            formula=("日收盘 = 该组最后一根 5m 的 close；"
                     "Pct_t = rank(x_1..x_t)/t；可用时刻 = 该根收完（+5min）"),
        )
        n += 1
    return n
