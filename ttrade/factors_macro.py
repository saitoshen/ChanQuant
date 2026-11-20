"""宏观因子：全天候（All Weather）那套 23 个宏观变量 + 2 条东财独立口径。

## 为什么这些能当因子用

单品种的价格因子回答「这个品种自己现在什么状态」，宏观因子回答「整个市场
处在什么环境」。后者对商品的解释力集中在**长周期**（利率、通胀、美元）和
**风险偏好**（美股、波动率、汇率）两个维度上 —— 这也是桥水全天候把资产按
「增长 / 通胀」两个轴分类的原因。

## 25 个变量分五组

- **利率（8）**：SHIBOR 隔夜/3M、LPR 1Y/5Y、美债 2Y/10Y、10Y−2Y 利差、美债实际 10Y
- **汇率与风险偏好（5）**：美元指数、美元兑离岸人民币、标普 500、纳斯达克、
  标普 20 日波动
- **外盘权益（2）**：恒生、日经 225
- **国内增长与通胀（8）**：制造业 PMI、CPI 同比、PPI 同比、M1 同比、M2 同比、
  社融增量、社融存量、GDP 同比
- **东财独立口径（2）**：美元兑离岸人民币·东财、美元指数·东财 —— 与主源
  （Tushare FXCM）**并存**，不是替代。主源的 `trade_date` 比真实交易日早一天，
  只能靠 `_LABEL_LAG` 手工补一天；东财的标签就是纽约交易日（2026-11-20 用
  dukascopy 5m 反推，相关 0.988@k=0），所以这两条是「口径核对的裁判」：
  同一时点上两者分位差得远，说明主源那条的偏移假设又变了。

## 因子值是**扩张分位**，不是水平

利率 2019 年是 2.5%、2023 年是 5%，直接拿水平做因子，模型学到的其实是
「这是哪一年」。所以每个序列先转成**扩张分位**（`expanding().rank(pct=True)`，
∈[0,1]）：0 表示「这是历史上最低的 1%」，1 表示最高。

扩张窗口保证 PIT：t 日的分位只由 ≤t 的样本算出来，不含未来。

分位在**序列自己的日频索引**上算一次并缓存，再 as-of 铺到 bar 上。不能反过来
「先铺到 bar 再算分位」—— 分钟线的 bar 一天有几十根，同一段历史会被重复计入，
分位会被采样频率带偏。

## 数据不全怎么办

`lpr_1y` / `lpr_5y` 目前没有数据源（见 `macro.py` 的说明）。它们照样注册，
读出来是 NaN，`filter_available()` 会自动从可用列表里剔掉 —— 界面上不会
出现两条永远空着的因子。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# code -> 中文标签
_LABEL = {
    "shibor_on": "SHIBOR隔夜",
    "shibor_3m": "SHIBOR3M",
    "lpr_1y": "LPR1年",
    "lpr_5y": "LPR5年",
    "us_y2": "美债2年",
    "us_y10": "美债10年",
    "us_10y2y": "美债10Y-2Y",
    "us_real10": "美债实际10年",
    "dxy": "美元指数",
    "usdcnh": "美元兑离岸人民币",
    "dxy_em": "美元指数·东财",
    "usdcnh_em": "美元兑离岸人民币·东财",
    "spx": "标普500",
    "ixic": "纳斯达克",
    "hsi": "恒生指数",
    "n225": "日经225",
    "spx_vol20": "标普20日波动",
    "cn_pmi_mfg": "制造业PMI",
    "cn_cpi_yoy": "CPI同比",
    "cn_ppi_yoy": "PPI同比",
    "cn_m1_yoy": "M1同比",
    "cn_m2_yoy": "M2同比",
    "cn_sf_inc": "社融增量",
    "cn_sf_cum": "社融存量",
    "cn_gdp_yoy": "GDP同比",
}

_DESC = {
    "shibor_on": "银行间隔夜资金价格的历史分位，最灵敏的流动性读数。",
    "shibor_3m": "3 个月 SHIBOR 的历史分位，银行负债成本的基准。",
    "lpr_1y": "1 年期 LPR 的历史分位，企业短贷定价基准（暂无数据源）。",
    "lpr_5y": "5 年期以上 LPR 的历史分位，房贷定价基准（暂无数据源）。",
    "us_y2": "美债 2 年期收益率分位，对联储政策路径最敏感的一档。",
    "us_y10": "美债 10 年期收益率分位，全球风险资产的贴现率锚。",
    "us_10y2y": "10Y−2Y 利差分位，倒挂（分位极低）是衰退的经典先行指标。",
    "us_real10": "美债 10 年期实际收益率分位，黄金的机会成本。",
    "dxy": "美元指数分位，商品的计价货币，高位通常压制商品。",
    "usdcnh": "美元兑离岸人民币分位，高位＝人民币贬值。",
    "dxy_em": "美元指数分位（东财日线口径）。与 `macro_dxy` 并存，用来核对主源的标签偏移。",
    "usdcnh_em": "美元兑离岸人民币分位（东财日线口径）。标签即纽约交易日，无 `_LABEL_LAG` 错位。",
    "spx": "标普 500 分位，全球风险偏好的主温度计。",
    "ixic": "纳斯达克分位，成长股/科技的风险偏好。",
    "hsi": "恒生指数分位，中国资产的离岸定价。",
    "n225": "日经 225 分位，亚太风险偏好与日元套息交易的镜子。",
    "spx_vol20": "标普 20 日年化波动率分位（VIX 的现货版），高位＝恐慌。",
    "cn_pmi_mfg": "制造业 PMI 分位，50 是荣枯线，中国总需求的月度温度计。",
    "cn_cpi_yoy": "CPI 同比分位，消费端通胀。",
    "cn_ppi_yoy": "PPI 同比分位，工业品出厂价，和商品价格同源。",
    "cn_m1_yoy": "M1 同比分位，企业活期存款，领先于实体需求。",
    "cn_m2_yoy": "M2 同比分位，广义流动性。",
    "cn_sf_inc": "社融增量分位，实体融资需求的月度读数。",
    "cn_sf_cum": "社融存量分位，信用扩张的累积水平。",
    "cn_gdp_yoy": "GDP 同比分位，季频，最慢但最权威的增长读数。",
}

# 分位需要的最少样本。5 年日频约 1200 个，月频 5 年只有 60 个 ——
# 取 60 是为了让月频指标从第 5 年起就有值，日频指标前 3 个月是 NaN（预热期）。
MIN_OBS = 60

_cache: dict[str, pd.Series] = {}

# ── 「每天几点才真正取得到」的序列 ──────────────────────────────────────────
# 用户 2026-11-19 钉死：Tushare 的日频汇率 **D+1 08:00 之后才更新**，凌晨取不到。
# 后果很具体：商品夜盘 21:00 ~ 次日 02:30 里，00:00 之后那一段如果照「自然日 − 1」
# 取数，拿到的是**当时还没发布**的那一行 —— 这是真正的未来函数，而且只在凌晨
# 两三个小时里发生，日盘完全看不出来。
#
# 🔴🔴 美债实际利率（`us_real10`）同理，用户 2026-11-19 追加要求套这条。
# 原因是时区链，不是「8 点」这个数字本身：
#
#     纽约不是 UTC+0 —— 是 **UTC−5（EST 冬）/ UTC−4（EDT 夏）**
#     财政部按**美国交易日 D** 发布（美东当天约 15:30~18:00 收完盘才出）
#     ⇒ 换成北京时间 = **D+1 的 04:30~07:00**（夏）/ **05:30~07:00**（冬）
#
# 也就是说 bar 11-19 02:00（商品夜盘）绝不可能拿到「11-18」那行 —— 它在
# 北京 11-19 早上才发布。用 `avail_hour=8` 后凌晨取到的是 D−2，比实际可用
# 时刻**保守 1~3.5 小时**（宁可晚用，不会早用）。实测发布侧证据：
# 财政部 XML feed 的 `updated` 是 UTC 15:48（北京 23:48），但那是**上一交易日**
# 那批数据的写入戳；同一天 FRED 的 `DFII10` 到北京 11-19 09:30 还没有 11-18
# （财政部有），说明财政部的 D+1 更新确实集中在北京次日早晨，不在前一夜。
#
# 只作用于这三项。其余宏观量（PMI / CPI / SHIBOR 等）本来就是低频、各走各的
# 发布节奏，不套这条。
AVAIL_08 = ("dxy", "usdcnh", "us_real10",
            # 东财口径的两条（纽约日 D 的收盘在北京 D+1 06:00，见 `macro._em_fx`）
            "usdcnh_em", "dxy_em",
            # 🔴 美股指数：收盘 16:00 美东 = 北京 **D+1 04:00（夏）/ 05:00（冬）**。
            # 不在这里就意味着凌晨（夜盘 00:00~02:30）用得上**当天才收的**那一行。
            # 2026-11-20 用东财日线反推过标签：k=0、相关 **1.000**、价位一分不差 ⇒
            # 标签没问题，问题只在「几点可取」，所以归这一组而不是加 `_LABEL_LAG`。
            "spx", "ixic",
            # 派生自 spx：标签跟着 spx 走，取数时刻也必须跟着走
            "spx_vol20")


def clear_cache() -> None:
    _cache.clear()


def _pct_series(code: str) -> pd.Series:
    """宏观序列 → 扩张分位（PIT），缓存在序列自己的日频索引上。

    🔴 **只有「读成功」才写缓存。** `edb.duckdb` 是 DuckDB 单文件库，文件锁
    是排他的 —— 宏观定时任务在写库的那几十秒里，只读连接会抛异常
    （`edb.connect` 只重试 3 次共 1.2 秒，挡不住一次完整的宏观下载）。以前这里
    `except: pass` 之后**连空 Series 一起缓存了**，于是一次撞锁就让整个进程的
    宏观因子永久变 NaN：因子被 `factor_matrix` 静默剔除、参与打分的票数比回测
    少几票，实盘与回测的信号就翻到两边去（2026-11-17 两条 IM 策略就是这个
    现象：一条实盘多 1 手、回测空仓，另一条反过来）。

    两种「空」必须分开：
      - **读失败**（异常）→ 不缓存，返回空，下一次调用自己重试；
      - **真没数据**（LPR 这种抓不到的 code）→ `M.series()` 返回空 Series、
        不抛异常，照常缓存 —— 否则每分钟都去查一次库。
    """
    if code in _cache:
        return _cache[code]
    s = pd.Series(dtype=float)
    try:
        from ttrade import macro as M

        raw = M.series(code)
        if raw is not None and len(raw):
            s = raw.astype(float).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            s = s.expanding(min_periods=MIN_OBS).rank(pct=True)
    except Exception:  # noqa: BLE001
        return s                      # 读失败：不写缓存，留给下次重试
    _cache[code] = s
    return s


def _target(index, lag_days: int = 1, avail_hour: int = 0) -> pd.DatetimeIndex:
    """每根 bar 的「取数时刻」：先按可用时刻往前推，再退日历日。

    `_asof()` 和 `asof_rows()` **共用这一个函数**。以前这四行写在 `_asof` 里，
    想在界面上显示「这根取的是哪一行」就得再抄一遍 —— 抄一遍就多一份判据，
    改一处忘一处正是这个项目的第一类 bug（#145）。
    """
    target = pd.DatetimeIndex(index)
    if avail_hour:
        target = target - pd.Timedelta(hours=int(avail_hour))
    if lag_days:
        target = target - pd.Timedelta(days=int(lag_days))
    return target


def asof_rows(code: str, index, lag_days: int = 1,
              avail_hour: int = 0) -> pd.DatetimeIndex:
    """每根 bar 实际取到的是**哪一天的那一行**（取不到的写 NaT）。

    只给界面和自检用：`_asof()` 返回的是值，问「这个值是哪天的」看不出来，
    而「日频因子是不是偷了当天」恰恰只能靠日期回答 —— 日频值日内恒定，图上
    是一条台阶，看不出台阶是从哪一行搬来的（2026-11-20 用户就是这么卡的）。

    可用时刻的口径与 `_asof()` 完全一致（同一个 `_target`），所以两边的结论
    不可能对不上。
    """
    s = _pct_series(code)
    if s is None or len(s) == 0:
        return pd.DatetimeIndex([pd.NaT] * len(pd.DatetimeIndex(index)))
    rows = pd.DatetimeIndex(s.index).normalize()
    # 同一行出现多次（宏观库补过数）只认最后一次，与 `_asof` 的 groupby.last 对齐
    rows = pd.DatetimeIndex(pd.unique(rows.sort_values()))
    t = _target(index, lag_days, avail_hour)
    pos = np.searchsorted(rows.values, t.values, side="right") - 1
    out = np.where(pos >= 0, rows.values[np.clip(pos, 0, None)],
                   np.datetime64("NaT"))
    return pd.DatetimeIndex(out)


def _asof(series: pd.Series, index, lag_days: int = 1,
          avail_hour: int = 0) -> pd.Series:
    """把分位序列 as-of 铺到 bar 上。宏观数据没有品种维度，所有品种同值。

    `lag_days` 是**日历日**滞后：宏观数据存的是发布日，再退一天把「当天发布当天
    就能用」这种乐观假设也挡掉。

    `avail_hour` 是**这一天几点才真正取得到**。默认 0（只按日期算，不区分时刻）。

    🔴🔴 为什么要这个参数（用户 2026-11-19 钉死的口径）：Tushare 的日频汇率
    **D+1 08:00 之后才更新**，凌晨取不到。而商品夜盘是 21:00 ~ 次日 02:30：

        bar 11-17 02:00（夜盘）  按「自然日 − 1」→ 取 ≤ 11-16 的行
                                 而 11-16 那行要到 11-17 08:00 才发布 ⇒ 偷未来

    传 `avail_hour=8` 后，目标时刻先减 8 小时再减 1 天：

        bar 11-17 02:00  →  11-15 18:00  →  取 ≤ 11-15（即「自然日 − 2」）
        bar 11-17 09:30  →  11-16 01:30  →  取 ≤ 11-16（仍是「前一天」）
        bar 11-17 08:00  →  11-16 00:00  →  取 ≤ 11-16（刚好接上）

    两种时段用同一个公式，不需要在夜盘/日盘之间分支 —— 分支写两遍判据正是这个
    项目最容易出 bug 的地方。

    ⚠️ 实现上**不能再 `normalize()`**：一旦把时刻抹掉，08:00 这条边界就没了，
    目标时刻退化成整天，凌晨段又会退回「自然日 − 1」。
    """
    if series is None or len(series) == 0:
        return pd.Series(np.nan, index=index, dtype=float)
    target = _target(index, lag_days, avail_hour)
    left = pd.DataFrame({"t": target}, index=np.arange(len(target)))
    right = pd.DataFrame({
        "t": pd.DatetimeIndex(series.index).normalize(),
        "v": series.to_numpy(dtype=float),
    }).dropna(subset=["t"]).sort_values("t")
    right = right.groupby("t", as_index=False).last()
    # 走 `merge_asof_ns`：目标时刻是 ns（实盘帧）还是 us（库内帧）不定，
    # 右表（EDB / 东财）一律 us —— 精度不一致会直接抛 MergeError（见 tsalign）
    from ttrade.tsalign import merge_asof_ns

    out = merge_asof_ns(left, right, on="t", direction="backward")
    return pd.Series(out.reindex(left.index)["v"].to_numpy(dtype=float),
                     index=index)


def asof_row(code: str, ts) -> pd.Timestamp:
    """某一根 bar 上，这个宏观因子取自**哪一天的那一行**（取不到返回 NaT）。

    可用时刻与滞后天数都只在这里各定义一次（`_avail_hour` / `_lag_days`），
    `_make` 和界面都调它 —— 界面上显示的「取自 11-16」与实际取数用的是同一条
    判据，不可能各说一套。
    """
    got = asof_rows(code, [pd.Timestamp(ts)], lag_days=_lag_days(code),
                    avail_hour=_avail_hour(code))
    return got[0]


def _avail_hour(code: str) -> int:
    """这个宏观量「当天几点才真正取得到」。见 `AVAIL_08` 的说明。"""
    return 8 if code in AVAIL_08 else 0


# ── 行标签比真实交易日早一天的序列（Tushare FXCM 外汇表）────────────────────
# 🔴🔴 2026-11-20 用户报「IM 的美元兑离岸人民币在 11-19 09:30 就用到了 11-19
# 的收盘」，查下来**成立**，而且不是 `lag_days` 能挡的那种。
#
# 证据（用 `data_duka/USDCNH_5m.parquet` 独立算纽约日收盘，按北京 06:00 切，
# 与 `factors_fx5.DAY_CUT_HOUR=6` 同一口径，大波动日一行一行对）：
#
#     NY 2026-03-31  −0.4101%  →  宏观行 03-30(Mon) −0.4087%
#     NY 2026-05-15  +0.4096%  →  宏观行 05-14(Thu) +0.4057%（05-15 无此行）
#     NY 2026-03-13  +0.3782%  →  宏观行 03-12(Thu) +0.3842%
#     NY 2026-03-20  +0.4160%  →  宏观行 03-19(Thu) +0.3805%（03-20 无此行）
#
# 也就是说 **Tushare 的 `trade_date` 比这根 K 线真实的交易日早一天**：纽约日 D
# 的收盘被写进 D−1 那一行。旁证是星期结构 —— 这个序列 2025 年起**没有周五行**
# （Friday 1 行、Saturday 91、Sunday 98、周一到周四各 98），周五那根被记到了周四。
#
# 后果很具体：纽约日 D 的收盘发生在**北京 D+1 06:00**，写进 D−1 行之后，这一行
# 要等到**北京 D+1 06:00** 才成立（标签 + 2 天）。按「标签 − 1 天」取数，就会在
# bar D 09:30 用上北京时间 D+1 06:00 才有的值 —— 超前 20.5 小时。
#
# 所以这两条的日历日滞后是 **2**（发布 1 天 + 标签错位 1 天），不是 1。
# `us_real10` 走财政部 `us_trycr`，没这个错位，仍是 1。
# 判据由自检 `_macro_label_offset` 用真实 5m 数据反推，源换了口径就会红。
#
# ── 2026-11-20 用 AKShare（东方财富）日线复核 ──────────────────────────────
# `ak.fx_hist_em("USDCNH")` / `ak.index_global_hist_em("美元指数")` 拿到的日线，
# 标签就是**纽约交易日**（实测与 dukascopy 5m 按北京 06:00 切的日收盘相关
# **0.988@k=0**，两者同一天对齐），可以当裁判。拿它逐段反推 Tushare 的偏移：
#
#     usdcnh  vs akshare 收益相关（最佳 k）
#       2012-2014  k=0  0.944      2015-2017  k=0  0.985
#       2018-2020  k=0  0.818      2021-2023  k=+1 0.602    2024-2026  k=+1 0.983
#     dxy     vs akshare
#       2012-2014  k=+1 0.963      2015-2017  k=+1 0.983
#       2018-2020  k=+1 0.559      2021-2023  k=+1 0.534    2024-2026  k=+1 0.986
#
# 🔴 `usdcnh` 的偏移**不是全史一致**：2020 年及以前是 0，2021 年起才是 +1
# （2021-2023 是过渡段，k=+1 0.602 压过 k=0 0.403，但不够干净）。`dxy` 全史 +1。
#
# 这里仍取**常数 1**，理由：只有取大不取小才不会前视。2021 年前多退一天只是
# 「用了更旧的一天」，不是泄漏；而按日期分段写两个判据，会在过渡段（数据本身
# 就在混）制造第二套口径 —— 这正是本项目第一类 bug 的来源（#145）。要改成分段，
# 必须先有一段干净的证据说明切换发生在哪一天。
#
# 🔴 东财口径的 `usdcnh_em` / `dxy_em` **不要**加进来：东财的标签就是纽约交易日，
# 错位为零，滞后仍是 1（发布那一层，由 `AVAIL_08` 的 08:00 兜住）。
# 最近 8 行的逐值对照（2026-11-20 实测，Tushare 行 D 对东财 D+1，差 ≤0.0001）：
#     TS 11-12 6.67474 ↔ 东财 11-13 6.67470      TS 11-17 6.66925 ↔ 东财 11-18 6.66920
#     TS 11-16 6.66689 ↔ 东财 11-17 6.66700      TS 11-18 6.69335 ↔ 东财 11-19 6.69240
_LABEL_LAG = {"usdcnh": 1, "dxy": 1}      # 额外的「标签比真实交易日早」天数


def _lag_days(code: str) -> int:
    """这条序列总共要退几个日历日。

    = 1（发布日与交易日之间的一天）+ `_LABEL_LAG` 里的标签错位天数。
    `_make()` 和 `asof_row()` 都走这里，界面上显示的取数日与回测取数是同一条判据。
    """
    return 1 + int(_LABEL_LAG.get(code, 0))


def _make(code: str):
    # 汇率类要到 08:00 才可用（见 AVAIL_08），且标签比真实交易日早一天（见 _LABEL_LAG）
    avail = _avail_hour(code)
    lag = _lag_days(code)

    def f(df: pd.DataFrame) -> pd.Series:
        return _asof(_pct_series(code), df.index, lag_days=lag, avail_hour=avail)

    f.__name__ = f"f_macro_{code}"
    f.__doc__ = _DESC.get(code, "")
    return f


def register_into(FACTORS, FactorSpec) -> int:
    """注册 25 个宏观因子，值都是「历史扩张分位」。"""
    # 25 条的公式完全一样，差别只在背后是哪条序列 —— 写一遍就够了。
    # 用分位而不是水平值：利率 2.5% 在 2019 年是常态、2023 年是低位，直接
    # 拿水平做因子，模型学到的是「这是哪一年」。
    formula = ("Pct_t = rank(x_1..x_t) / t（扩张分位，只用 ≤t 的值）；"
               "x 为发布日对齐后的原始序列")
    n = 0
    for code, label in _LABEL.items():
        key = f"macro_{code}"
        FACTORS[key] = FactorSpec(
            key, f"宏观·{label}", _make(code), "宏观", _DESC.get(code, ""),
            formula=formula,
        )
        n += 1
    return n
