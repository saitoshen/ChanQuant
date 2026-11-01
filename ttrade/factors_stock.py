"""A 股专属因子：基本面 / 资金流 / 概念板块 / 除权。

## 为什么单独一个文件

期货那套因子（展期、期权、持仓量）在股票上一条都用不了；股票这套（PE、换手率、
大单净买入、概念板块热度）在期货上也一条都用不了。硬塞进 `factors.py` 只会让
那个文件里的 `markets` 判断越来越长。按市场分文件，注册时标 `markets=("cn_stock",)`，
界面层用 `factors.filter_by_market()` 一过滤就干净了。

## 数据从哪来

全部来自 `data_tq/stock.duckdb` 的三张表，由 `ttrade/stockaux.attach()` 在加载
K 线时 join 到 df 上：

    stock_basic_ts    PE / PB / PS / 总市值 / 流通市值 / 换手率 / 股息率
    stock_moneyflow   大单 / 超大单 / 小单的买卖手数，净流入额
    stock_adj         复权因子（除权除息日检测）
    stock_concept     概念板块成分（所属概念数）

列不在 df 上时一律返回全 NaN，`filter_available()` 会把这条因子从该标的的
列表里剔掉 —— 只下了日线没下基本面的用户不会看到一屏空行。

## PIT

- 基本面 / 资金流是**收盘后**发布的。`stockaux.attach()` 在日线上按当天 join
  （信号在收盘产生、下一根开盘成交，合法），在分钟线上整体后移一个交易日。
- 本文件里的因子只做 `rolling` / `shift(正数)` / `pct_change`，没有一处
  `shift(-n)`。除权天数用「上一次除权距今多少根 bar」，只看过去。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 工具


def _nan(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=df.index, dtype=float)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """取衍生列，没有就返回全 NaN（和 `factors._col` 同一个约定）。"""
    if name not in df.columns:
        return _nan(df)
    return pd.to_numeric(df[name], errors="coerce").astype(float)


def _safe_div(a, b):
    return a / b.replace(0.0, np.nan)


def _inv(s: pd.Series) -> pd.Series:
    """取倒数。0 和缺失都返回 NaN，避免 inf 污染 z-score。"""
    return _safe_div(pd.Series(1.0, index=s.index), s)


# ---------------------------------------------------------------- 基本面
def f_ep_ttm(df: pd.DataFrame) -> pd.Series:
    """盈利收益率 E/P = 1 / PE(TTM)。

    直接拿 PE 做因子会有两个问题：亏损股 PE 为负、数值右偏极严重（几百倍的
    PE 一大把）。取倒数之后分布接近正态，且「亏损」自然变成负的 E/P，
    方向和经济含义都对得上（越便宜越高）。
    """
    return _inv(_col(df, "pe_ttm"))


def f_bp(df: pd.DataFrame) -> pd.Series:
    """账面市值比 B/P = 1 / PB。经典的价值因子，比 1/PE 稳（净资产很少为负）。"""
    return _inv(_col(df, "pb"))


def f_sp_ttm(df: pd.DataFrame) -> pd.Series:
    """营收市值比 S/P = 1 / PS(TTM)。对亏损但有收入的成长股也能排序。"""
    return _inv(_col(df, "ps_ttm"))


def f_size_log(df: pd.DataFrame) -> pd.Series:
    """对数总市值。A 股上长期是小市值占优，所以 `hint_dir=-1`。

    取对数是因为市值跨了三个数量级（几十亿到两万亿），原值进 z-score
    会被几个巨无霸带偏。
    """
    mv = _col(df, "total_mv")
    return np.log(mv.where(mv > 0))


def f_turnover_20(df: pd.DataFrame) -> pd.Series:
    """20 日平均换手率。换手高＝分歧大、筹码松动，A 股上偏负面。"""
    return _col(df, "turnover_rate").rolling(20, min_periods=10).mean()


def f_turnover_chg_20(df: pd.DataFrame) -> pd.Series:
    """换手率 20 日变化（相对 20 日前）。放量是情绪启动的先行指标。"""
    t = _col(df, "turnover_rate").rolling(5, min_periods=3).mean()
    return _safe_div(t, t.shift(20)) - 1.0


def f_dv_ttm(df: pd.DataFrame) -> pd.Series:
    """股息率（TTM）。高股息在弱市里是防守因子。"""
    return _col(df, "dv_ttm")


def f_circ_ratio(df: pd.DataFrame) -> pd.Series:
    """流通市值 / 总市值。比例低＝解禁压力大、流通盘小易被炒。"""
    return _safe_div(_col(df, "circ_mv"), _col(df, "total_mv"))


# ---------------------------------------------------------------- 资金流
def f_mf_net_5(df: pd.DataFrame) -> pd.Series:
    """5 日累计净流入占成交额比例。

    用「占比」而不是绝对金额：绝对金额和市值强相关，做成因子等于又加了一个
    市值因子。占比之后才是「这只股票今天被买得凶不凶」。
    """
    return _col(df, "mf_net_ratio").rolling(5, min_periods=3).sum()


def f_mf_lg_20(df: pd.DataFrame) -> pd.Series:
    """20 日平均「大单+超大单」净买入占比。

    机构和大户的痕迹在大小单结构里。20 日均值是为了滤掉单日的对倒噪声。
    """
    return _col(df, "mf_lg_ratio").rolling(20, min_periods=10).mean()


def f_mf_rev_5(df: pd.DataFrame) -> pd.Series:
    """资金流反转：5 日大单净买入的相反数。

    A 股短周期上，散户追着大单买进去往往是接盘 —— 大单刚猛买完的五天里
    反转概率偏高。方向取负，和 `rev_5` 是同一类逻辑，只是信号源换成资金流。
    """
    return -_col(df, "mf_lg_ratio").rolling(5, min_periods=3).sum()


# ---------------------------------------------------------------- 概念板块
def f_concept_count(df: pd.DataFrame) -> pd.Series:
    """所属概念板块数量。

    挂的概念越多，说明这只股票被越多主题覆盖 —— 主题轮动时更容易被资金
    想起来。是个粗糙但便宜的「热度」代理；真正的板块动量要按板块成分股
    聚合，那是另一层的事。
    """
    return _col(df, "concept_count")


def f_board_code(df: pd.DataFrame) -> pd.Series:
    """板块哑变量：0=主板 / 1=创业板 / 2=科创板 / 3=北交所。

    不是「好因子」，是**控制变量**：创业板和科创板的波动、估值中枢、涨跌幅
    限制都和主板不同，混在一起做截面排序时它天然占优或吃亏。放进因子库让
    用户自己决定要不要用它做中性化。
    """
    return _col(df, "board_code")


# ---------------------------------------------------------------- 除权
def f_exdiv_days(df: pd.DataFrame) -> pd.Series:
    """距上一次除权除息的 bar 数。

    除权日复权因子会跳一下。刚除权完的股票价格「看起来便宜了」，填权行情
    是 A 股一个长期存在的现象；距离越近，填权预期越强。用 bar 计数而不是
    自然日，日线和分钟线上口径才一致。
    """
    adj = _col(df, "adj_factor")
    if adj.notna().sum() < 2:
        return _nan(df)
    jump = adj.pct_change().abs() > 1e-9
    pos = np.arange(len(df), dtype=float)
    last = pd.Series(np.where(jump.to_numpy(dtype=bool), pos, np.nan)).ffill()
    return pd.Series(pos - last.to_numpy(dtype=float), index=df.index)


# ---------------------------------------------------------------- 量价（A 股口径）
#
# 这几条不用任何辅助表，只吃 OHLCV —— 所以它们在**分钟线**上也能算，而基本面
# 那批在分钟线上只能读到「上一个交易日」的快照（见 `stockaux.attach`）。

def f_mom_20(df: pd.DataFrame) -> pd.Series:
    """20 日动量。A 股一个月级别的动量方向不稳，但作为**反转的对立面**必须
    有一个：不同板块、不同年份，动量和反转会互相切换，只给一个方向等于预设答案。
    """
    c = _col(df, "close")
    return _safe_div(c, c.shift(20)) - 1.0


def f_rev_5(df: pd.DataFrame) -> pd.Series:
    """5 日反转。A 股最稳的短期异象之一：涨得最猛的五天之后往往回吐。"""
    c = _col(df, "close")
    return -(_safe_div(c, c.shift(5)) - 1.0)


def f_vol_20(df: pd.DataFrame) -> pd.Series:
    """20 日收益波动率。低波因子在 A 股长期有效（低波组合风险调整后更优）。"""
    return _col(df, "close").pct_change().rolling(20, min_periods=10).std()


def f_illq_20(df: pd.DataFrame) -> pd.Series:
    """Amihud 非流动性：20 日平均 |收益| / 成交额。

    成交额用对数先压一下量纲 —— 直接除，小盘股会大到把 z-score 拉爆。
    值越大＝同样的钱能推动更多价格＝流动性越差，所以 `hint_dir=1`
    （流动性溢价：难买的更贵）。
    """
    ret = _col(df, "close").pct_change().abs()
    amt = _col(df, "amount")
    return _safe_div(ret, np.log1p(amt.where(amt > 0))).rolling(20, min_periods=10).mean()


def f_amt_log_20(df: pd.DataFrame) -> pd.Series:
    """20 日平均成交额（对数）。规模 / 关注度的成交量口径代理。"""
    amt = _col(df, "amount")
    return np.log1p(amt.where(amt > 0).rolling(20, min_periods=10).mean())


def f_ma_gap_20(df: pd.DataFrame) -> pd.Series:
    """收盘相对 20 日均线的偏离。乖离率，短期超买超卖的直观刻度。"""
    c = _col(df, "close")
    ma = c.rolling(20, min_periods=10).mean()
    return _safe_div(c, ma) - 1.0


def f_high_52w(df: pd.DataFrame) -> pd.Series:
    """距 52 周高点的距离（收盘 / 250 日最高 − 1，负数）。

    越接近 0 越靠近前高。「创新高」在 A 股是持续的正向信号（动量），
    所以 `hint_dir=1`：越靠近前高越高分。
    """
    c = _col(df, "close")
    hi = c.rolling(250, min_periods=120).max()
    return _safe_div(c, hi) - 1.0


# ---------------------------------------------------------------- 估值分位

def f_ep_pct(df: pd.DataFrame) -> pd.Series:
    """E/P 在过去 250 根 bar 里的分位（0~1）。

    同一个 PE 在不同行业、不同年代的含义完全不同。横截面上直接比 PE 会把
    「银行天然低 PE」当成「银行便宜」，用**自身历史分位**才是「这只股票相对
    它自己贵不贵」。分位越高＝越便宜（E/P 越大）。
    """
    s = f_ep_ttm(df)
    return s.rolling(250, min_periods=120).rank(pct=True)


def f_dv_pct(df: pd.DataFrame) -> pd.Series:
    """股息率的历史分位。高股息 + 自身高分位＝真便宜，而不是行业属性。"""
    s = _col(df, "dv_ttm")
    return s.rolling(250, min_periods=120).rank(pct=True)


# ---------------------------------------------------------------- 规模变化

def f_mv_chg_20(df: pd.DataFrame) -> pd.Series:
    """总市值 20 日变化率。

    和价格动量高度相关但不完全一样：市值变化含股本变动（增发 / 解禁）。
    两者差得多的股票，是「价格没动但股本变了」的那批，值得单独看。
    """
    mv = _col(df, "total_mv")
    return _safe_div(mv, mv.shift(20)) - 1.0


def f_mf_sm_5(df: pd.DataFrame) -> pd.Series:
    """5 日小单净买入占比。

    小单是散户。散户净买得多、大单净卖得也多的时候，往往是派发阶段 ——
    所以这条的 `hint_dir=-1`，和 `mf_lg_20`（大单）方向相反。
    """
    return _col(df, "mf_sm_ratio").rolling(5, min_periods=3).mean()


# ---------------------------------------------------------------- 注册
_SPECS = [
    # (key, label, func, category, desc, hint_dir)
    ("ep_ttm", "盈利收益率 E/P", f_ep_ttm, "基本面",
     "1 / PE(TTM)。亏损股自然为负，分布比 PE 本身接近正态。", 1),
    ("bp", "账面市值比 B/P", f_bp, "基本面",
     "1 / PB。经典价值因子，净资产很少为负，比 E/P 稳。", 1),
    ("sp_ttm", "营收市值比 S/P", f_sp_ttm, "基本面",
     "1 / PS(TTM)。对亏损但有收入的成长股也能排序。", 1),
    ("size_log", "对数总市值", f_size_log, "基本面",
     "总市值取对数。A 股长期小市值占优，高分偏跌。", -1),
    ("stock_turnover_20", "换手率 20 日", f_turnover_20, "基本面",
     "20 日平均换手率。换手高＝分歧大、筹码松动。", -1),
    ("turnover_chg_20", "换手率变化 20", f_turnover_chg_20, "基本面",
     "近 5 日均换手 / 20 日前 5 日均换手 − 1，情绪启动的先行指标。", 1),
    ("dv_ttm", "股息率 TTM", f_dv_ttm, "基本面",
     "股息率（TTM）。弱市里的防守因子。", 1),
    ("circ_ratio", "流通市值占比", f_circ_ratio, "基本面",
     "流通市值 / 总市值。比例低＝解禁压力大、流通盘小。", -1),

    ("mf_net_5", "资金净流入 5 日", f_mf_net_5, "资金流",
     "5 日累计净流入额 / 成交额。用占比避开市值相关性。", 1),
    ("mf_lg_20", "大单净买入 20 日", f_mf_lg_20, "资金流",
     "20 日平均「大单+超大单」净买入手数占比，机构痕迹。", 1),
    ("mf_rev_5", "资金流反转 5 日", f_mf_rev_5, "资金流",
     "5 日大单净买入的相反数。A 股短周期上追大单容易接盘。", -1),

    ("concept_count", "所属概念数", f_concept_count, "概念板块",
     "挂了多少个概念板块，主题轮动时的热度粗代理。", 1),
    ("board_code", "板块哑变量", f_board_code, "概念板块",
     "0=主板 1=创业板 2=科创板 3=北交所。控制变量，不是选股信号。", 1),

    ("exdiv_days", "距上次除权天数", f_exdiv_days, "除权",
     "距上一次除权除息的 bar 数，填权行情的距离代理。", 1),

    # v2.4.0 扩充：量价 / 估值分位 / 规模变化
    ("stock_mom_20", "动量 20 日", f_mom_20, "量价",
     "20 日涨幅。和反转互为对手，别只看一个方向。", 1),
    ("stock_rev_5", "反转 5 日", f_rev_5, "量价",
     "近 5 日涨幅的相反数。A 股最稳的短期异象之一。", 1),
    ("stock_vol_20", "波动率 20 日", f_vol_20, "量价",
     "20 日收益标准差。低波在 A 股长期有效。", -1),
    ("stock_illiq_20", "非流动性 20 日", f_illq_20, "量价",
     "Amihud：|收益| / ln(成交额) 的 20 日均值。流动性溢价。", 1),
    ("stock_amt_20", "成交额 20 日（对数）", f_amt_log_20, "量价",
     "20 日平均成交额取对数，规模与关注度的成交量口径。", 1),
    ("stock_ma_gap_20", "乖离率 20 日", f_ma_gap_20, "量价",
     "收盘 / 20 日均线 − 1，短期超买超卖的直观刻度。", -1),
    ("stock_high_52w", "距 52 周高点", f_high_52w, "量价",
     "收盘 / 250 日最高 − 1。越接近 0 越靠近前高。", 1),
    ("stock_ep_pct", "E/P 历史分位", f_ep_pct, "估值分位",
     "E/P 在自身 250 根 bar 里的分位。避开「行业天然低 PE」的误判。", 1),
    ("stock_dv_pct", "股息率历史分位", f_dv_pct, "估值分位",
     "股息率在自身 250 根 bar 里的分位。高分位＋高股息＝真便宜。", 1),
    ("stock_mv_chg_20", "市值变化 20 日", f_mv_chg_20, "基本面",
     "总市值 20 日变化率。和价格动量之差就是股本变动。", 1),
    ("mf_sm_5", "小单净买入 5 日", f_mf_sm_5, "资金流",
     "5 日小单净买入占比。小单＝散户，追高时往往是派发。", -1),
]


def register_into(FACTORS: dict, FactorSpec) -> int:
    """把 A 股因子注册进主表。返回注册条数。

    `markets=("cn_stock",)` 是关键：界面层按当前大类过滤，期货页面看不到
    这些因子，股票页面也看不到展期 / 期权那些。
    """
    n = 0
    for key, label, func, cat, desc, hint in _SPECS:
        if key in FACTORS:
            continue
        FACTORS[key] = FactorSpec(key, label, func, cat, desc, hint,
                                  markets=("cn_stock",))
        n += 1
    return n
