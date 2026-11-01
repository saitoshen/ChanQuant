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


def clear_cache() -> None:
    _cache.clear()


def _pct_series(code: str) -> pd.Series:
    """宏观序列 → 扩张分位（PIT），缓存在序列自己的日频索引上。"""
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
        pass
    _cache[code] = s
    return s


def _asof(series: pd.Series, index, lag_days: int = 1) -> pd.Series:
    """把分位序列 as-of 铺到 bar 上。宏观数据没有品种维度，所有品种同值。

    滞后 1 个日历日：宏观数据存的是发布日，再退一天把「当天发布当天就能用」
    这种乐观假设也挡掉。
    """
    if series is None or len(series) == 0:
        return pd.Series(np.nan, index=index, dtype=float)
    left = pd.DataFrame({"t": pd.DatetimeIndex(index).normalize()})
    if lag_days:
        left["t"] = left["t"] - pd.Timedelta(days=int(lag_days))
    right = pd.DataFrame({
        "t": pd.DatetimeIndex(series.index).normalize(),
        "v": series.to_numpy(dtype=float),
    }).dropna(subset=["t"]).sort_values("t")
    right = right.groupby("t", as_index=False).last()
    out = pd.merge_asof(left.sort_values("t"), right, on="t", direction="backward")
    return pd.Series(out.reindex(left.index)["v"].to_numpy(dtype=float),
                     index=index)


def _make(code: str):
    def f(df: pd.DataFrame) -> pd.Series:
        return _asof(_pct_series(code), df.index)

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
