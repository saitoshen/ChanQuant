"""宏观因子：全天候（All Weather）那套 23 个宏观变量。

## 为什么这些能当因子用

单品种的价格因子回答「这个品种自己现在什么状态」，宏观因子回答「整个市场
处在什么环境」。后者对商品的解释力集中在**长周期**（利率、通胀、美元）和
**风险偏好**（美股、波动率、汇率）两个维度上 —— 这也是桥水全天候把资产按
「增长 / 通胀」两个轴分类的原因。

## 23 个变量分四组

- **利率（8）**：SHIBOR 隔夜/3M、LPR 1Y/5Y、美债 2Y/10Y、10Y−2Y 利差、美债实际 10Y
- **汇率与风险偏好（5）**：美元指数、美元兑离岸人民币、标普 500、纳斯达克、
  标普 20 日波动
- **外盘权益（2）**：恒生、日经 225
- **国内增长与通胀（8）**：制造业 PMI、CPI 同比、PPI 同比、M1 同比、M2 同比、
  社融增量、社融存量、GDP 同比

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
AVAIL_08 = ("dxy", "usdcnh", "us_real10")


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
    target = pd.DatetimeIndex(index)
    if avail_hour:
        target = target - pd.Timedelta(hours=int(avail_hour))
    if lag_days:
        target = target - pd.Timedelta(days=int(lag_days))
    left = pd.DataFrame({"t": target}, index=np.arange(len(target)))
    right = pd.DataFrame({
        "t": pd.DatetimeIndex(series.index).normalize(),
        "v": series.to_numpy(dtype=float),
    }).dropna(subset=["t"]).sort_values("t")
    right = right.groupby("t", as_index=False).last()
    out = pd.merge_asof(left.sort_values("t"), right, on="t", direction="backward")
    return pd.Series(out.reindex(left.index)["v"].to_numpy(dtype=float),
                     index=index)


def _make(code: str):
    # 汇率类要到 08:00 才可用（见 AVAIL_08 的说明），其余按日期算
    avail = 8 if code in AVAIL_08 else 0

    def f(df: pd.DataFrame) -> pd.Series:
        return _asof(_pct_series(code), df.index, avail_hour=avail)

    f.__name__ = f"f_macro_{code}"
    f.__doc__ = _DESC.get(code, "")
    return f


def register_into(FACTORS, FactorSpec) -> int:
    """注册 23 个宏观因子，值都是「历史扩张分位」。"""
    # 23 条的公式完全一样，差别只在背后是哪条序列 —— 写一遍就够了。
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
