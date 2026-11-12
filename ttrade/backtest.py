"""向量化友好的逐日回测引擎（支持多空双向）。

规则：
1. 策略在第 T 日收盘后给出目标仓位 signal[T] ∈ {-1, 0, 1}；
2. 实际成交发生在第 T+1 日开盘价（杜绝未来函数）；
3. 买卖均计手续费与滑点；
4. **仓位按手数下**（见 `BacktestConfig.sizing`）：期货是「分配资金 ÷ 每手保证金」，
   杠杆由品种自己的保证金比例决定；美股没有杠杆概念，走 `unit` 口径（名义敞口 =
   权益）；
5. **不做爆仓强平**。权益亏穿（为负）也照跑到底：回测是虚拟测试，杠杆按品种
   保证金率算，比实盘敢开的大得多，强平掉只会把后半段历史抹掉。要看的是「这套
   规则从头到尾发生了什么」，而不是「几几年被抬走」；
6. 空仓时资金闲置（不计利息）。

## 为什么权益是「现金 + 浮动盈亏」而不是「市值」

期货开仓只是把保证金冻住，钱还是你的，权益不会因为开仓而缩水。所以持仓期间
`equity = cash + 手数 × (现价 − 开仓价) × 乘数 × 方向`，平仓时这笔盈亏才落进
`cash`。按市值算（`手数 × 现价`）在期货上会得到一条完全错误的曲线：开仓瞬间
权益看起来就翻了几倍。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """按数据本身推断「一年有多少根 bar」，用于年化波动和夏普。

    不能写死 252：日线一年 244 根，1 小时一年约 1.4 千根，5 分钟一年约 1.2 万根。
    用 252 去年化 5 分钟数据，年化波动会被低估约 7 倍（√(11700/252)），夏普虚高
    同样倍数 —— 而且这个错误不会报错，只是数字看起来「特别好」。

    用整段数据的根数除以日历跨度，比按周期查表更稳：不同品种夜盘时长不同
    （有的到 23:00，有的到 01:00 或 02:30），查表必然对不齐。
    """
    if index is None or len(index) < 2:
        return float(TRADING_DAYS)
    days = (pd.Timestamp(index[-1]) - pd.Timestamp(index[0])).days
    if days <= 0:
        return float(TRADING_DAYS)
    return float(len(index) / (days / 365.25))


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    fee_rate: float = 0.0005      # 单边手续费：按成交名义额的比例
    slippage: float = 0.0005      # 单边滑点：按价格的比例（美股口径）
    long_only: bool = False       # True 时把 -1 裁成 0（股票市场默认）
    bars_per_year: float = float(TRADING_DAYS)   # 日线默认；分钟级由调用方按数据推断

    # ---- 期货口径的成本 ----
    # 交易所手续费有两种收法：按成交额比例（`fee_rate`）和按手固定（元/手）。
    # 郑商所、中金所多数品种是后者，用比例去套会差一个数量级，所以两个都留着，
    # 谁非零用谁（`fee_per_lot` 优先）。
    fee_per_lot: float = 0.0
    # 滑点按**跳**算：一跳的价格 = `tick_size`，滑几跳 = `slippage_ticks`。
    # 螺纹一跳 1 元、铜一跳 10 元，写成一个 bp 永远对不齐。`tick_size` 为 0
    # 时退回上面的百分比口径（`slippage`）。
    tick_size: float = 0.0
    slippage_ticks: float = 0.0

    # ---- 仓位怎么定 ----
    #   unit   ：名义敞口 = 权益，不加杠杆、不看乘数（美股、批量脚本）
    #   capital：手数 = 分配资金 ÷ 每手保证金（期货默认）
    #   risk   ：手数 = 止损金额 ÷（止损倍数 × ATR × 乘数）
    #   fixed  ：固定 `volume` 手
    # 禁止做多（只做空）。和 `long_only` 是一对互斥开关，两个都开等于不做 ——
    # 界面上不允许勾成那样。截面那条路会在权重里直接不建多头腿
    short_only: bool = False

    sizing: str = "unit"
    capital: float = 0.0          # 分配资金（0 = 用 initial_cash）
    risk_amount: float = 20_000.0
    stop_atr_n: int = 14
    stop_mult: float = 5.0
    volume: int = 1
    multiplier: float = 1.0       # 合约乘数（每手多少单位）
    # 保证金率。默认 1.0 = 全额，也就是「不认识的品种先按不加杠杆算」。
    # 写成 0.1 的话，任何忘了填乘数/保证金率的地方都会悄悄变成 10 倍杠杆
    margin_rate: float = 1.0

    # ---- K 线内成交（只有 `run_intrabar_backtest` 读它） ----
    # 🔴 **铁律：进场当根不出场**。引擎里 ① 反向突破离场 ② ATR 止损 都带
    # `i > entry_bar` 守卫，进场那一根只负责建仓，任何出场都要等到下一根。
    # 理由不是「乐观/悲观」的偏好，是**这根 bar 承载不了这个判断**：high/low
    # 分不出「进场前」还是「进场后」，而突破那根本身往往就是最宽的一根，
    # 它的 low 绝大多数是在建仓之前做出来的。拿它判止损 = 凭空多一次亏损。
    # 碳酸锂 2026-11-05 10:00 那根：open 88420、high 93460、low 88100，
    # 触发价 92640、止损 89405 —— 价格必须先从 88420 涨到 92640 才可能进场，
    # 而 88100 在开盘那一刻就已成交过，早于建仓。
    #
    # 下面这个开关是**唯一的例外，且只用于诊断**：打开 = 允许进场当根就被扫，
    # 得到的是「K 线内不可知」这件事值多少钱的**悲观上界**（因为那条路九成
    # 是假止损）。默认关。不要拿 True 的结果当回测结论。
    #
    # 实盘（`strategy_runner._intrabar_tick`）逐 tick 判止损，价格路径真实可见，
    # 本来就不会有这个问题，也不读这个开关；回测与实盘剩下的差就是「K 线粒度」
    # 本身带来的那部分。
    intrabar_samebar_stop: bool = False

    # ---- 网格（档位信号）的成交模型 ----
    # True（默认）：**挂单**成交 —— 档位价上挂着限价单，K 线内触到就按那个价
    #   成交，止盈也是提前挂好的一格。见 `run_grid_intrabar`。
    # False：回到「第 T 根收盘定档、第 T+1 根开盘成交」。留给对拍用：同一份
    #   信号在两种成交模型下的差，就是「追价 vs 挂单」值多少钱。
    # 只对 `max|signal| > 1` 且带 `grid_step` 列的（网格那一族）生效。
    grid_intrabar: bool = True

    # 🔴 同一根 K 线不许既进场又出场 —— 这是**结构性不变量**，不做成开关。
    #    一根 bar 只有 OHLC，high/low 的先后分不出来，允许同根来回等于拿路径
    #    假设编收益：实测铁矿 5m 放宽 = +25.3%，收紧 = −21.0%。想多要来回只能
    #    降周期（1m），不能放宽假设。见 `run_grid_intrabar` 的 ④。

    # 平仓当根要不要允许再开新仓（默认 **不许**）。    #
    # 「反手」在 K 线内是**假装精确**：一根 5 分钟线里「先止损、再反向开仓」
    # 和「先反向开仓、再被打止损」，盘口上长得一模一样，分不出先后 —— 挑对
    # 自己有利的那个就是凭空造 alpha。默认让出一根 bar 的空窗：这根平掉了就
    # 平掉了，下一根才重新看触发价。
    #
    # 这也和平台规则的语义对齐：`sbbox_mini` 那套是「收盘确认 → 下一根开盘」，
    # 持有多单时根本看不到空头信号（`pos` 当根还是 1），只有多单真的出场之后
    # 才可能做空。原先 K 线内那条会在**同一根**里平多开空，看着像反手，实盘
    # 做不到。想看那个乐观口径，把它设 True 再跑一遍。
    intrabar_reentry_samebar: bool = False

    @property
    def label(self) -> str:
        return "只做多" if self.long_only else "多空双向"

    def slip_amount(self, price: float, scale: float = 1.0) -> float:
        """单边滑点，**价格单位**。按跳算；没有 tick 时退回百分比。

        `scale` 是这根 K 线的复权因子（`P_adj = P_raw × scale`）。**一跳是真实
        世界里的固定金额**，换算到前复权价上要乘 scale：螺纹永远是 1 元/吨，
        但复权后历史价格被整体缩小了，不乘的话五年前那段的滑点会按 0.67 倍收。
        传 1.0（默认）就是不复权口径。

        百分比口径不用乘：它是「占价格的比例」，分子分母同时被 scale 缩放。
        """
        if self.tick_size > 0 and self.slippage_ticks > 0:
            return float(self.tick_size) * float(self.slippage_ticks) * float(scale or 1.0)
        return abs(float(price)) * float(self.slippage)

    def fee_amount(self, lots: float, mult: float, price: float) -> float:
        """单边手续费，**金额**。按手收优先，否则按成交名义额。"""
        if self.fee_per_lot > 0:
            return abs(float(lots)) * float(self.fee_per_lot)
        return abs(float(lots)) * float(mult) * float(price) * float(self.fee_rate)

    @property
    def budget(self) -> float:
        """这条策略能动用的名义资金。"""
        return float(self.capital or self.initial_cash)


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    ret: float
    bars: int
    direction: int = 1            # 1=多头，-1=空头
    open_position: bool = False
    lots: float = 0.0             # 这笔实际下了多少手（unit 口径是份额，小数）
    # ---- 金额口径 ----
    # `gross` 是**按盘口价**算的价差盈亏（带符号手数 × 乘数 × 未含滑点的价差），
    # `cost` 是**这一笔自己吃掉的双边成本**（开仓 + 平仓的手续费与滑点），
    # `pnl` 是两者之差。
    #
    # `gross` 一定要用**不含滑点**的价格：滑点已经单列进 `cost` 了，用成交价
    # （已含滑点）去算毛盈亏，滑点就被扣了两遍 —— 逐笔金额加起来会比账户实际
    # 变化少「总滑点」那么多，而且不报错。`gross − cost = 账户变化` 是这一对
    # 字段唯一说得通的验收标准。
    #
    # 为什么逐笔也要记成本：绩效页只有一个「总成本」，回答不了「这笔到底被
    # 吃掉多少」。而百分比收益（`ret`）是纯价差、不含成本 —— 一笔毛赚 0.05%、
    # 成本 0.1% 的交易在表里是「+0.05%」，看着是赚的，钱其实少了
    gross: float = 0.0
    cost: float = 0.0
    # 截面策略一笔只对应一个品种，但同一个 `Result` 里混着好几个品种的交易；
    # 单策略回测只有一只标的，留空即可
    symbol: str = ""
    # 这一笔「为什么进、为什么出」，交易记录页的备注列直接读它。
    # 引擎只填**默认值**（策略名 / 信号反向 / 信号离场），具体口径由策略自己
    # 往 df 里写 `entry_note` / `exit_note` 两列覆盖 —— 「分档浮盈回撤」「ATR
    # 止损」这类信息只有因子策略自己知道，引擎不该假装懂
    entry_note: str = ""
    exit_note: str = ""

    @property
    def pnl(self) -> float:
        """这一笔的净盈亏（**金额**，已扣双边成本）。

        交易记录页的「收益」「累计盈亏」两列读它，绩效页的胜率 / 盈利因子还是
        读 `ret`（纯价差百分比）—— 两个口径各管一处：百分比说「这段行情走了
        多少」，金额说「钱到底多了还是少了」。
        """
        return float(self.gross) - float(self.cost)


@dataclass
class Result:
    equity: pd.Series
    position: pd.Series
    trades: list[Trade]
    metrics: dict = field(default_factory=dict)
    label: str = ""
    # 逐品种拆账（只有截面策略有）：品种 → 该品种自己的 Result。绩效页的
    # 「品种」下拉框读它，回答「组合里到底是哪个在赚」
    legs: dict = field(default_factory=dict)


_METRIC_KEYS = (
    "total_return", "cagr", "ann_vol", "sharpe", "max_dd", "calmar",
    "exposure", "long_exposure", "short_exposure",
    "avg_notional", "peak_notional", "avg_margin", "peak_margin", "util",
    "n_trades", "n_long", "n_short",
    "win_rate", "avg_trade", "profit_factor", "payoff", "avg_win", "avg_loss",
    "avg_bars", "best_trade", "worst_trade",
)


def _metrics(equity: pd.Series, position: pd.Series, trades: list[Trade],
             bars_per_year: float = float(TRADING_DAYS),
             base: float | None = None,
             price: pd.Series | None = None,
             mult: float = 1.0, margin_rate: float = 1.0) -> dict:
    """绩效指标。**全套单利口径**：本金（= 期初权益）固定不变。

    系统的回测框架是单利框架 —— 本金和使用的资金是给定的，赚到的钱不会自动
    变成下一笔的仓位（`capital` / `fixed` / `risk` 三种手数口径都是固定手数，
    `unit` 也只是「名义敞口=当时的本金」）。所以比例型指标统一拿**本金**当
    分母，而不是「上一根权益」：

        累计收益 = (期末权益 − 本金) ÷ 本金
        年化收益 = 累计收益 ÷ 年数
        最大回撤 = 权益 ÷ 历史峰值 − 1（回撤本来就该相对峰值看，见下）
        卡玛     = 年化收益 ÷ |最大回撤|

        逐根收益 = 当日盈亏 ÷ 本金
        夏普     = 逐根收益的日均 × 年根数 ÷ 年化波动

    分母全是本金，所以「日均 × 年根数」和「累计 ÷ 年数」是同一个量级的数，
    夏普、卡玛、年化摆放着能互相验算。买了一只二十年涨百倍的股票，本金口径下
    后期的日盈亏就是本金的几十倍，波动自然很大 —— 那不是算错，是「仓位市值
    已经远超初始本金」这件事本身。
    """
    equity = equity.dropna()
    if len(equity) < 2:
        return {k: 0.0 for k in _METRIC_KEYS}

    # 本金 = 期初权益。`run_backtest` 第 0 根一定是空仓，所以它就是 `initial_cash`。
    # 调用方可以显式传进来（基准第一根就已经建了仓，`eq.iloc[0]` 不等于本金）
    base = float(base if base else equity.iloc[0])
    total = (float(equity.iloc[-1]) - base) / base if base > 0 else 0.0

    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    # 单利年化 = 累计收益 ÷ 年数。没有 `(1+累计)^(1/年数)−1` 那一步 —— 那是
    # 假设每年把赚到的钱再押进去，而本金固定时那笔钱并没有变成仓位
    cagr = float(total / years) if years > 0 else 0.0

    # 逐根收益 = 当日盈亏 ÷ 本金（单利）。`pct_change()` 是「相对上一根权益」，
    # 那是复利口径：权益涨上去之后同样的盈亏被算成更小的波动，夏普跟着虚高
    ret = (equity.diff().fillna(0.0) / base if base > 0
           else pd.Series(0.0, index=equity.index))

    bpy = float(bars_per_year or TRADING_DAYS)
    vol = ret.std(ddof=1) * np.sqrt(bpy) if ret.std(ddof=1) > 0 else 0.0
    sharpe = (ret.mean() * bpy) / vol if vol > 0 else 0.0

    # 回撤也按**本金**算：(权益 − 历史峰值) ÷ 本金。单利框架里本金是唯一的分母，
    # 换成「相对峰值」的话，赚了一半之后同样的回落会被算成更小的百分比，
    # 而且和卡玛的分子（年化，相对本金）不是一个基准 —— 两个数摆在一起没法比
    dd_series = (equity - equity.cummax()) / base if base > 0 else equity * 0.0
    max_dd = float(dd_series.min())
    # 卡玛：分子分母都以本金为基准
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    # 权益一旦被亏穿，波动 / 夏普 / 卡玛都失去意义（净值翻号后的「反弹」不是
    # 收益），标 NaN，界面显示「—」；累计收益、最大回撤、期末资金照常给
    if float(equity.min()) <= 0:
        vol = sharpe = calmar = float("nan")

    pos = position.reindex(equity.index).fillna(0)
    long_expo = float((pos > 0).mean())
    short_expo = float((pos < 0).mean())

    closed = [t for t in trades if not t.open_position]
    n = len(closed)
    if n:
        rets = np.array([t.ret for t in closed])
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        win_rate = len(wins) / n
        avg_trade = float(rets.mean())
        gross_win = wins.sum() if len(wins) else 0.0
        gross_loss = abs(losses.sum()) if len(losses) else 0.0
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        # 平均盈亏比（payoff）= 平均盈利 ÷ 平均亏损。它和盈利因子是**一对**：
        #
        #     盈利因子 = 总盈利 ÷ 总亏损
        #             = (盈利笔数 × 平均盈利) ÷ (亏损笔数 × 平均亏损)
        #             = 胜率 / (1 − 胜率) × 平均盈亏比
        #
        # 只看盈利因子，看不出这 2.0 是「六成胜率、每次小赚」还是「三成胜率、
        # 一次赚三倍」，两种打法要盯的东西完全不同（前者怕胜率掉，后者怕单笔
        # 尾部）。补上这一项才拆得开
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else (999.0 if avg_win > 0 else 0.0)
        avg_bars = float(np.mean([t.bars for t in closed]))
        best, worst = float(rets.max()), float(rets.min())
    else:
        win_rate = avg_trade = profit_factor = avg_bars = best = worst = 0.0
        avg_win = avg_loss = payoff = 0.0

    # ---- 资金占用诊断（绩效页的三列）----
    #
    # 和「分配了多少本金」是两回事：`capital_pct=30%` 只说「开仓那一刻用 30%」，
    # 而日内策略大部分时间空仓 —— 实测一个 24 条策略的组合，名义本金 500 万，
    # 平均占用只有 8.3 万（1.66%）、峰值 83 万（16.6%）。不把这两个数摆出来，
    # 用户会以为 500 万都在干活。
    #
    # 口径：`|手数| × 收盘 × 乘数 × 保证金率`。**不按开仓价** —— 持仓期间保证金
    # 跟着结算价走，用开仓价会在趋势行情里系统性低估。
    avg_notional = peak_notional = avg_margin = peak_margin = util = 0.0
    if price is not None and float(margin_rate or 0) > 0:
        px = pd.to_numeric(price, errors="coerce").reindex(equity.index).ffill()
        notional = pos.abs().to_numpy(dtype=float) * px.to_numpy(dtype=float) * float(mult or 1.0)
        ok = np.isfinite(notional)
        if ok.any():
            avg_notional = float(notional[ok].mean())
            peak_notional = float(notional[ok].max())
            mr = float(margin_rate)
            avg_margin = avg_notional * mr
            peak_margin = peak_notional * mr
            util = avg_margin / base if base > 0 else 0.0

    return {
        "total_return": float(total),
        "cagr": float(cagr),
        # 本金：所有比例型指标的分母，也是绩效表里「本金」那一行读的
        "initial_cash": float(base),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "calmar": float(calmar),
        "exposure": long_expo + short_expo,
        "long_exposure": long_expo,
        "short_exposure": short_expo,
        # 资金占用：平均 / 峰值 / 利用率（平均占用 ÷ 本金）
        "avg_notional": avg_notional,
        "peak_notional": peak_notional,
        "avg_margin": avg_margin,
        "peak_margin": peak_margin,
        "util": float(util),
        "n_trades": n,
        "n_long": sum(1 for t in closed if t.direction == 1),
        "n_short": sum(1 for t in closed if t.direction == -1),
        "win_rate": float(win_rate),
        "avg_trade": avg_trade,
        "profit_factor": profit_factor,
        "payoff": payoff,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_bars": avg_bars,
        "best_trade": best,
        "worst_trade": worst,
    }


def _trade_ret(direction: int, entry: float, exit_: float) -> float:
    return float(exit_ / entry - 1.0) if direction == 1 else float(1.0 - exit_ / entry)


def _atr_array(close: np.ndarray, high: np.ndarray, low: np.ndarray, n: int) -> np.ndarray:
    """真实波幅均值的整段序列，给「按止损金额定仓」用。"""
    n = max(1, int(n or 14))
    prev = np.concatenate(([np.nan], close[:-1]))
    tr = np.maximum(np.maximum(high - low, np.abs(high - prev)), np.abs(low - prev))
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy(dtype=float)


def size_lots(cfg: BacktestConfig, price: float, equity: float, atr_value: float = 0.0,
              share: float = 1.0) -> float:
    """这一笔该下多少手。

    `share`：这笔单子要跟几个方向**分**同一份预算（网格的中间震荡带里多空各挂
    一张网时是 2）。只对「按钱定手数」的口径（`budget` / `capital` / `risk`）
    生效 —— `fixed` 是「每格固定 N 手」，本来就是按份数定的，不摊。

    `unit` 口径返回的是**手数**（可以是小数，名义敞口 = 权益），其余口径返回整数手。
    手数为 0 表示这笔不做（钱不够一手保证金）；例外是 `budget`（按分配资金定手数，
    四舍五入但**最少 1 手**，见下）。

    **五种口径都是单利**：手数只看「本金（或分配的那部分）/ 止损金额 / 固定
    手数」，不看账户当前权益。赚了不加大仓位，亏了也不缩 —— 回测出来的是
    「这套规则在固定本金下能赚多少」，而不是「复利滚了二十年之后的奇迹」。
    """
    if price <= 0:
        return 0.0
    mode = (cfg.sizing or "unit").lower()
    mult = float(cfg.multiplier or 1.0)
    if mode == "fixed":
        return float(max(1, int(cfg.volume or 1)))
    if mode == "risk":
        per = float(cfg.stop_mult or 0) * float(atr_value or 0) * mult
        if per <= 0:
            return 1.0                      # ATR 还没暖起来：先 1 手，别裸奔
        return float(max(1, int(float(cfg.risk_amount or 0) / max(share, 1e-9) / per)))
    if mode == "budget":
        # 「按分配资金定手数」：手数 = 分配的钱 ÷ 这一手要花的钱（价格×乘数），
        # **四舍五入**，且**不足一手按 1 手**。
        # 为什么必须留 1 手这个下限：股指一手名义就 80~120 万，100 万本金分三档，
        # 每档 43 万 ⇒ 算出来 0.36 手，四舍五入是 0 手 ⇒ 整段不交易，回测空跑。
        # 期货不能下 0.36 手，所以「买不起也至少下 1 手」是这个口径的定义，
        # 代价是低资金账户的实际杠杆会**高于**设定杠杆（43 万预算买 120 万货）。
        # 要按保证金口径（整手、钱不够就不做）请用 `capital`。
        per = price * mult
        if per <= 0:
            return 0.0
        return float(max(1, int(round(float(cfg.budget or 0.0) / max(share, 1e-9) / per))))
    if mode == "capital":
        per = price * mult * float(cfg.margin_rate or 0)
        if per <= 0:
            return 1.0
        # 分配资金 ÷ 每手保证金，**固定用分配的那笔钱**，不随盈亏变。
        # 按当前权益收口就变成复利了：50 万赚到 80 万会自己加到 80 万的仓，
        # 亏到 30 万又自动减到 30 万 —— 那是另一套策略，不是用户要的「单利」
        return float(max(int(cfg.budget / max(share, 1e-9) / per), 0))
    # 名义敞口 = **本金**（固定）。原来是「当前权益」，权益从 10 万涨到 34 万，
    # 手数跟着放大 3.4 倍 —— 那是复利，和其余三种口径不是一套东西：双均线 /
    # 唐奇安 / 海龟 / 通道突破 / MACD 这几条老策略因此会跑出「越赚越猛」的曲线，
    # 和固定手数的结果没法比。改成盯本金之后，四种口径统一是单利。
    # 期货一手是 price × 乘数，不除乘数的话「不加杠杆」会变成 10 倍杠杆
    return float(max(float(cfg.initial_cash or equity or 0.0), 0.0) / (price * mult))


def size_lots_intrabar(cfg: BacktestConfig, price: float, equity: float,
                       stop_dist: float, mult: float) -> float:
    """K 线内这一族的手数。

    `risk`（按止损金额定仓）口径**直接用策略给的 `stop_dist`**：

        手数 = risk_amount ÷ (stop_dist × 合约乘数)

    这样「打到止损就亏 `risk_amount`」是字面意义上的成立 —— 因为引擎用的止损
    距离和算手数用的止损距离是同一个数。

    老路径的 `risk` 是「倍数 × ATR(stop_atr_n)」反推出来的，而这一族本来就有
    `stop_dist`。反推要多绕一层：先除 `cfg.stop_mult` 还原 ATR，再乘回去。
    `cfg.stop_mult` 填得和策略参数不一致（策略自己的 stop_mult 是可调参数），
    仓位就按另一个风险在走，而且不会报错 —— 这正是「止损金额分配仓位」最容易
    出错的地方，所以这里绕开反推。
    """
    mode = (cfg.sizing or "").lower()
    if mode == "risk" and _finite(stop_dist) and float(stop_dist) > 0:
        per = float(stop_dist) * float(mult or 1.0)
        if per > 0:
            # 和 `size_lots` 同一个下限：至少 1 手，凑不出 1 手就不做
            return float(max(1, int(float(cfg.risk_amount or 0) / per)))
    a = 0.0
    if _finite(stop_dist) and float(cfg.stop_mult or 0) > 0:
        a = float(stop_dist) / float(cfg.stop_mult)
    return size_lots(cfg, price, equity, a)


def _note_col(data: pd.DataFrame, name: str) -> np.ndarray | None:
    """取一列逐根的原因说明，转成 object 数组；没有这一列返回 None。

    非字符串（NaN / None / 数字）一律当空串：pandas 的字符串列里缺值一定是
    NaN，直接拼进界面会显示成「nan」。
    """
    if name not in data.columns:
        return None
    out = np.empty(len(data), dtype=object)
    for i, v in enumerate(data[name].to_numpy(dtype=object)):
        out[i] = "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()
    return out


def _num_col(data: pd.DataFrame, name: str, n: int) -> np.ndarray:
    """取一列数值，缺列或非数值一律 NaN。

    K 线内进出场那一族的列（触发价 / 止损距离 / 出场标志）都可以缺：策略
    只做多就不写 `entry_short`，不设止损就不写 `stop_dist`。缺列按 NaN
    处理，逐根的判断自然跳过它 —— 不需要为每种组合写一遍分支。
    """
    if name not in data.columns:
        return np.full(n, np.nan, dtype=float)
    return pd.to_numeric(data[name], errors="coerce").to_numpy(dtype=float)


def _finite(x: float) -> bool:
    """`NaN` 和 `inf` 都不算有效价（`x == x` 判不出 inf）。"""
    return bool(np.isfinite(x))


def run_intrabar_backtest(data: pd.DataFrame, cfg: BacktestConfig,
                          label: str = "") -> Result:
    """**K 线内部进出场**：按「触发价」成交，而不是等下一根开盘。

    老路径（`signal` → T+1 开盘成交）是「收盘后才做决定」的模型，适合日线、
    也适合任何「收盘确认」的规则。但突破类策略的真实成交发生在 K 线走的
    过程中：价格碰到 20 根箱体上沿的那一刻就该进场，等收盘再说等于白送了
    一段行情。这条分支就是为那种规则准备的。

    ## 策略要提供哪些列（全部可选，缺列 = 该功能不用）

    | 列名 | 含义 | 单位 |
    |---|---|---|
    | `entry_long`  | 本根内做多的触发价 | 价格 |
    | `entry_short` | 本根内做空的触发价 | 价格 |
    | `exit_long`   | 1 = 多头下一根开盘平 | 0/1 |
    | `exit_short`  | 1 = 空头下一根开盘平 | 0/1 |
    | `stop_dist`   | 止损距离（入场价 ∓ 它） | 价格 |

    `stop_dist` 是**距离**不是价位：止损要绑在「实际成交价」上，策略在写这一列
    时还不知道引擎会用哪个价成交（含滑点、含跳空）。给距离，引擎自己减。

    ## 成交价的判定

    一律取「触发价」，但被开盘跳空打断时**按开盘价成交** —— 开盘价已经越过
    触发价，说明那一刻市场没有给你触发价：

        做多入场  fill = max(open, entry_long)
        做空入场  fill = min(open, entry_short)
        多头止损  fill = min(open, stop)      ← 低开直接穿，比 stop 更差
        空头止损  fill = max(open, stop)

    滑点在成交价上再叠一层（买加卖减），和 `run_backtest` 同一套 `slip_amount`。

    ## 同一根 K 线里的先后顺序（无 tick 数据，只能定约定）

    1. 开盘：上一根收盘给的 `exit_*` 先执行 —— 那是**已知**的、开盘就该做的；
    2. 盘中：已有仓位的止损（可能跳空成交在开盘价）；
    3. 盘中：空仓才看入场；
    4. 盘中：刚进的仓，同一根就打到止损 —— **按先入场后止损处理**。
    5. 盘中：本根平过仓（止损或开盘出场）⇒ 本根**不再开新仓**，下一根才重新
       看触发价 —— 也就是「不反手」，见 `intrabar_reentry_samebar`。

    第 4 条是保守假设。没有分笔数据时，「这根 K 线先冲到高点再砸到低点」
    和「先砸后冲」看起来一模一样，猜对自己有利的一边等于自己给自己送 alpha。
    反过来（假设先止损后入场）会让回测好得离谱，实盘永远做不到。
    """
    open_ = data["open"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    high = (data["high"].to_numpy(dtype=float) if "high" in data.columns
            else open_.copy())
    low = (data["low"].to_numpy(dtype=float) if "low" in data.columns
           else open_.copy())
    scale = (pd.to_numeric(data["price_scale"], errors="coerce").fillna(1.0)
             .to_numpy(dtype=float) if "price_scale" in data.columns else None)

    idx = data.index
    n = len(data)
    mult = float(cfg.multiplier or 1.0)

    e_long = _num_col(data, "entry_long", n)
    e_short = _num_col(data, "entry_short", n)
    if cfg.long_only:
        # 「只做多」在老路径上靠裁 `signal` 实现，这里没有 signal 可裁 —— 只能
        # 把做空触发价整列抹掉。不抹的话，界面上勾了「只做多」，回测还是会做空
        e_short = np.full(n, np.nan, dtype=float)
    x_long = _num_col(data, "exit_long", n)
    x_short = _num_col(data, "exit_short", n)
    stop_d = _num_col(data, "stop_dist", n)
    note_in = _note_col(data, "entry_note")
    note_out = _note_col(data, "exit_note")

    cash = float(cfg.initial_cash)
    cost_total = 0.0
    notional_total = 0.0
    lots = 0.0
    pos = 0
    entry_price = 0.0        # 含滑点的成交价（账户口径）
    entry_raw = 0.0          # 不含滑点的盘口价（毛盈亏口径，见 Trade.gross）
    entry_cost = 0.0
    entry_date = None
    entry_bar = 0
    entry_dir = 1
    entry_why = ""
    stop_px = float("nan")
    closed_i = -1            # 最近一次平仓发生在哪一根（防「平仓当根又开仓」）
    pos_state = np.zeros(n, dtype=int)
    equity = np.zeros(n, dtype=float)
    trades: list[Trade] = []

    def _slip(price: float, i: int) -> float:
        return cfg.slip_amount(price, scale[i] if scale is not None else 1.0)

    def do_close(i: int, raw_px: float, why: str) -> None:
        """平掉当前仓位。`raw_px` 是**不含滑点**的成交基准价。"""
        nonlocal cash, lots, pos, entry_price, entry_date, entry_cost
        nonlocal cost_total, notional_total, stop_px, closed_i
        if entry_price <= 0 or pos == 0:
            return
        sp = _slip(raw_px, i)
        # 平多 = 卖出（滑点往下）；平空 = 买入（滑点往上）
        fill = raw_px - sp if pos == 1 else raw_px + sp
        cash += lots * mult * (fill - entry_price)
        fee = cfg.fee_amount(lots, mult, fill)
        cash -= fee
        close_cost = fee + sp * mult * abs(lots)
        cost_total += close_cost
        notional_total += abs(lots) * mult * abs(fill)
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=idx[i],
            entry_price=float(entry_price),
            exit_price=float(fill),
            ret=_trade_ret(entry_dir, entry_price, fill),
            bars=i - entry_bar,
            direction=entry_dir,
            lots=abs(float(lots)),
            gross=float(lots) * mult * (float(raw_px) - float(entry_raw)),
            cost=float(entry_cost) + float(close_cost),
            entry_note=entry_why,
            exit_note=why,
        ))
        lots = 0.0
        pos = 0
        entry_price = 0.0
        entry_date = None
        entry_cost = 0.0
        stop_px = float("nan")
        closed_i = i

    def do_open(i: int, raw_px: float, direction: int, why: str) -> bool:
        """开仓。`raw_px` 是**不含滑点**的成交基准价（触发价或开盘价）。"""
        nonlocal cash, lots, pos, entry_price, entry_raw, entry_date
        nonlocal entry_bar, entry_dir, entry_cost, entry_why, stop_px
        nonlocal cost_total, notional_total
        sp = _slip(raw_px, i)
        fill = raw_px + sp if direction > 0 else raw_px - sp
        # 手数：`risk` 口径按这一笔的**实际止损距离**算（见 `size_lots_intrabar`），
        # 其余口径照旧。不能先反推 ATR 再走 `size_lots` —— 那要多绕一层，
        # 且依赖 `cfg.stop_mult` 和策略参数一致
        q = size_lots_intrabar(cfg, fill, cash, stop_d[i], mult)
        if q <= 0:
            return False
        fee = cfg.fee_amount(q, mult, fill)
        cash -= fee
        entry_cost = fee + sp * mult * q
        cost_total += entry_cost
        notional_total += q * mult * abs(fill)
        lots = q if direction > 0 else -q
        pos = direction
        entry_price = float(fill)
        entry_raw = float(raw_px)
        entry_date = idx[i]
        entry_bar = i
        entry_dir = direction
        entry_why = why
        # 止损绑**实际成交价**：策略只知道距离，成交价是引擎算出来的
        stop_px = float("nan")
        if _finite(stop_d[i]) and float(stop_d[i]) > 0:
            stop_px = fill - float(stop_d[i]) if direction > 0 else fill + float(stop_d[i])
        return True

    for i in range(n):
        o, hi, lo, cl = open_[i], high[i], low[i], close[i]

        # 🔴 **进场当根不出场**：① ② 都加了 `i > entry_bar`。
        # 这根 bar 的 high/low 分不出「进场前」还是「进场后」，拿它判任何出场
        # 都是在赌对自己有利的那个顺序 —— 而止损方向的那个赌注是系统性亏钱
        # （突破那根往往最宽，low 常常在建仓前就做出来了）。所以进场那一根
        # 只负责建仓，出场从**下一根**起才允许。
        #
        # 唯一的例外是 ③ 末尾那个诊断开关 `intrabar_samebar_stop`（默认关），
        # 它的存在只为量出「K 线内不可知」值多少钱，不能拿来出结果。
        live_pos = pos != 0 and i > entry_bar

        # ① 上一根收盘给的出场：本根开盘执行
        if live_pos and i > 0:
            want_out = (x_long[i - 1] > 0.5) if pos == 1 else (x_short[i - 1] > 0.5)
            if want_out:
                why = (note_out[i - 1] if note_out is not None else "") or "反向突破离场"
                do_close(i, o, why)

        # ② 已有仓位的止损（K 线内；开盘已穿过就按开盘价，更差）
        if live_pos and _finite(stop_px):
            if pos == 1 and lo <= stop_px:
                do_close(i, min(o, stop_px), "ATR 止损")
            elif pos == -1 and hi >= stop_px:
                do_close(i, max(o, stop_px), "ATR 止损")

        # ③ 空仓时看本根有没有触发入场
        # 本根刚平过仓就不再开新仓：持有多单时看不到空头触发价，多单出场之后
        # 也要等到下一根才重新看 —— 否则「止损 / 反向突破离场」的同一根里就能
        # 反手做空，而那一刻的先后顺序是分不出来的（见
        # `intrabar_reentry_samebar`）
        if pos == 0 and (cfg.intrabar_reentry_samebar or closed_i != i):
            up, dn = e_long[i], e_short[i]
            # 止损距离必须已知：ATR 没暖起来就进的话，这一笔没有止损，
            # 后面 300 根都靠「反向突破」兜底 —— 那不是这套规则
            ready = _finite(stop_d[i]) and float(stop_d[i]) > 0
            long_ok = ready and _finite(up) and hi >= up
            short_ok = ready and _finite(dn) and lo <= dn
            if long_ok and short_ok:
                # 同一根两头都打穿：分不出先后，宁可这单不做
                long_ok = short_ok = False
            if long_ok:
                raw = max(o, up)
                why = (note_in[i] if note_in is not None else "") or label
                if (do_open(i, raw, 1, why) and cfg.intrabar_samebar_stop
                        and lo <= stop_px):
                    # ⚠️ 只有显式打开诊断开关才会走到这里（默认关）。
                    # 这一步**故意违反**「进场当根不出场」，只为给出悲观上界，
                    # 不要拿它的结果当回测结论。
                    # 🔴 只能按 `stop_px` 成交，**不能**再取 `min(o, stop_px)`：
                    # 开仓发生在开盘之后，当根开盘价在时间上早于建仓，用它就
                    # 等于「建仓前就按开盘价止损了」—— 一分钟没持仓，却记一笔
                    # 亏损，而且亏损幅度被放大成「整根 K 线的幅度」
                    do_close(i, stop_px, "ATR 止损")
            elif short_ok:
                raw = min(o, dn)
                why = (note_in[i] if note_in is not None else "") or label
                if (do_open(i, raw, -1, why) and cfg.intrabar_samebar_stop
                        and hi >= stop_px):
                    do_close(i, stop_px, "ATR 止损")

        eq = cash + (lots * mult * (cl - entry_price) if pos != 0 else 0.0)
        pos_state[i] = pos
        equity[i] = eq

    # 未平仓的最后一笔（口径与 `run_backtest` 一致：只付开仓那一侧成本）
    if pos != 0 and entry_price > 0:
        last = close[-1]
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=idx[-1],
            entry_price=float(entry_price),
            exit_price=float(last),
            ret=_trade_ret(entry_dir, entry_price, last),
            bars=n - 1 - entry_bar,
            direction=entry_dir,
            open_position=True,
            lots=abs(float(lots)),
            gross=float(lots) * mult * (float(last) - float(entry_raw)),
            cost=float(entry_cost),
            entry_note=entry_why,
            exit_note="样本结束，仍持仓",
        ))

    eq = pd.Series(equity, index=idx, name="equity")
    ps = pd.Series(pos_state, index=idx, name="position")
    m = _metrics(eq, ps, trades, cfg.bars_per_year, base=cfg.initial_cash,
                 price=pd.Series(close, index=idx), mult=mult,
                 margin_rate=cfg.margin_rate)
    m["final_equity"] = float(eq.iloc[-1])
    m["total_cost"] = float(cost_total)
    m["notional"] = float(notional_total)
    m["avg_cost"] = float(cost_total / m["n_trades"]) if m["n_trades"] else 0.0
    m["cost_rate"] = float(cost_total / notional_total) if notional_total > 0 else 0.0
    m["intrabar"] = 1.0
    return Result(equity=eq, position=ps, trades=trades, metrics=m, label=label)


def run_level_backtest(data: pd.DataFrame, cfg: "BacktestConfig",
                       label: str = "") -> Result:
    """**分级仓位**：`signal` 是档位号（-N…N），不再是 -1/0/1。

    网格交易要「跌一格买一格」，一次持有 1 档、2 档……N 档。老路径的 `pos`
    只有 -1/0/1 三种取值，表达不了这个，硬塞会把 3 当成长、把 -3 当成平。
    这里把它推广：

    - `signal = k`（k>0）= 持有 k 份多头，`-k` = k 份空头，0 = 空仓；
    - **一份的手数 = `size_lots()` 那一次的结果**，和单仓位策略的「满仓」是
      同一份 ⇒ 实盘 `st.target = st.signal × st.lots` 与这里逐档一致；
      一份的手数在建仓那一刻定下来，之后不再随价格漂 —— 否则每根 bar 都改
      总仓，网格就没有「格」可言了。所以分级策略请用「固定手数」口径；
    - 每加一份是**一条独立的腿**，减仓按**后进先出**（网格卖的正是最近买的那
      一格），每次减仓落一笔已实现盈亏，交易记录里逐笔可见；
    - 成交时点与老路径相同：第 T 根收盘定档位，第 T+1 根开盘成交。

    只有当 `max|signal| > 1` 才走这里。全部是 ±1 的策略仍走原来那条路，
    历史回测结果可以逐位对拍 —— 多一条执行模型不等于换掉原来那一条。
    """
    open_ = data["open"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    scale = (pd.to_numeric(data["price_scale"], errors="coerce").fillna(1.0).to_numpy(float)
             if "price_scale" in data.columns else None)
    sig = data["signal"].fillna(0).astype(int).to_numpy()
    if cfg.long_only:
        sig = np.maximum(sig, 0)
    if cfg.short_only:
        sig = np.minimum(sig, 0)
    idx = data.index
    note_in = _note_col(data, "entry_note")
    note_out = _note_col(data, "exit_note")
    n = len(data)
    mult = float(cfg.multiplier or 1.0)
    mode = (cfg.sizing or "unit").lower()

    atr_arr = None
    if mode == "risk":
        atr_arr = _atr_array(close, data["high"].to_numpy(dtype=float)
                             if "high" in data else close,
                             data["low"].to_numpy(dtype=float)
                             if "low" in data else close,
                             cfg.stop_atr_n)

    target = np.roll(sig, 1)
    target[0] = 0

    cash = float(cfg.initial_cash)
    cost_total = 0.0
    notional_total = 0.0
    legs: list[dict] = []        # 每一份的建仓明细，同一方向
    pos = 0
    pos_state = np.zeros(n, dtype=int)
    equity = np.zeros(n, dtype=float)
    trades: list[Trade] = []

    def _slip(price: float, i: int) -> float:
        return cfg.slip_amount(price, scale[i] if scale is not None else 1.0)

    def _held() -> float:
        return sum(leg["lots"] for leg in legs)

    def _open_leg(i: int, direction: int, q: float, why: str) -> None:
        nonlocal cash, cost_total, notional_total
        sp = _slip(open_[i], i)
        fill = open_[i] + sp if direction > 0 else open_[i] - sp
        fee = cfg.fee_amount(q, mult, fill)
        cash -= fee
        cost = fee + sp * mult * q
        cost_total += cost
        notional_total += q * mult * abs(fill)
        legs.append({"lots": q * direction, "fill": float(fill),
                     "raw": float(open_[i]), "date": idx[i], "bar": i,
                     "why": why or label, "cost": float(cost)})

    def _close_part(i: int, amount: float, why: str) -> None:
        """从**最新的腿**开始平掉 amount 手（正数），每次落一笔交易。"""
        nonlocal cash, cost_total, notional_total
        left = float(amount)
        while left > 1e-9 and legs:
            leg = legs[-1]
            take = min(left, abs(leg["lots"]))
            d = 1 if leg["lots"] > 0 else -1
            sp = _slip(open_[i], i)
            fill = open_[i] - sp if d > 0 else open_[i] + sp
            cash += take * d * mult * (fill - leg["fill"])
            fee = cfg.fee_amount(take, mult, fill)
            cash -= fee
            close_cost = fee + sp * mult * take
            cost_total += close_cost
            notional_total += take * mult * abs(fill)
            share = take / abs(leg["lots"])
            entry_cost = leg["cost"] * share
            trades.append(Trade(
                entry_date=leg["date"],
                exit_date=idx[i],
                entry_price=float(leg["fill"]),
                exit_price=float(fill),
                ret=_trade_ret(d, leg["fill"], fill),
                bars=i - leg["bar"],
                direction=d,
                lots=float(take),
                # 毛盈亏用**不含滑点**的盘口价，滑点在 cost 里已经算过一次
                gross=take * d * mult * (float(open_[i]) - float(leg["raw"])),
                cost=float(entry_cost) + float(close_cost),
                entry_note=leg["why"],
                exit_note=why,
            ))
            if take >= abs(leg["lots"]) - 1e-9:
                legs.pop()
            else:
                leg["lots"] -= take * d
                leg["cost"] -= entry_cost
            left -= take

    unit_q = 0.0               # 一份的手数，建仓那一刻定下来
    for i in range(n):
        want = int(target[i])
        want_units = abs(want)
        want_dir = 1 if want > 0 else (-1 if want < 0 else 0)
        dec = i - 1 if i > 0 else 0
        held = _held()

        # 方向反了 / 该清仓：先全平。平的理由从**上一根**取（这一根的动作是
        # 上一根收盘定下的）
        if held != 0 and (want_dir == 0 or (held > 0) != (want_dir > 0)):
            why = (note_out[dec] if note_out is not None else "") or (
                "信号反向" if want_dir else "信号离场")
            _close_part(i, abs(held), why)
            held = _held()

        if want_dir != 0:
            if not legs:
                a = float(atr_arr[i]) if atr_arr is not None else 0.0
                sp = _slip(open_[i], i)
                fill = open_[i] + sp if want_dir > 0 else open_[i] - sp
                unit_q = size_lots(cfg, fill, cash, a if a == a else 0.0)
            want_lots = want_units * unit_q
            cur = abs(_held())
            if unit_q > 0 and want_lots > cur + 1e-9:
                add = int(round((want_lots - cur) / unit_q))
                why = (note_in[dec] if note_in is not None else "") or label
                for _ in range(max(add, 1)):
                    _open_leg(i, want_dir, unit_q, why)
            elif want_lots < cur - 1e-9:
                why = (note_out[dec] if note_out is not None else "") or "减仓"
                _close_part(i, cur - want_lots, why)

        # 注意：这句必须在 `if want_dir != 0` **之外**。放里面的话，清仓那根
        # 不进这个分支，`pos` 会一直沿用上一个非零方向 ⇒ 持仓曲线永远回不到 0，
        # 「14:55 是否空仓」「是否隔夜」这类只看持仓序列的检查全部失真。
        pos = want_dir if legs else 0

        eq = cash + sum(leg["lots"] * mult * (close[i] - leg["fill"]) for leg in legs)
        pos_state[i] = pos
        equity[i] = eq

    # 未平仓的腿：逐条入账，只算开仓那一侧的成本
    for leg in legs:
        d = 1 if leg["lots"] > 0 else -1
        last = float(close[-1])
        trades.append(Trade(
            entry_date=leg["date"],
            exit_date=idx[-1],
            entry_price=float(leg["fill"]),
            exit_price=last,
            ret=_trade_ret(d, leg["fill"], last),
            bars=n - 1 - leg["bar"],
            direction=d,
            open_position=True,
            lots=abs(float(leg["lots"])),
            gross=float(leg["lots"]) * mult * (last - float(leg["raw"])),
            cost=float(leg["cost"]),
            entry_note=leg["why"],
            exit_note="样本结束，仍持仓",
        ))

    eq_series = pd.Series(equity, index=idx, name="equity")
    ps = pd.Series(pos_state, index=idx, name="position")
    m = _metrics(eq_series, ps, trades, cfg.bars_per_year, base=cfg.initial_cash,
                 price=pd.Series(close, index=idx), mult=mult,
                 margin_rate=cfg.margin_rate)
    m["final_equity"] = float(eq_series.iloc[-1])
    m["total_cost"] = float(cost_total)
    m["notional"] = float(notional_total)
    m["avg_cost"] = float(cost_total / m["n_trades"]) if m["n_trades"] else 0.0
    m["cost_rate"] = float(cost_total / notional_total) if notional_total > 0 else 0.0
    # 仓位数（档位）上限，绩效页能看出这张网铺到多深
    m["max_level"] = int(np.abs(sig).max()) if n else 0
    return Result(equity=eq_series, position=ps, trades=trades, metrics=m, label=label)


def run_grid_intrabar(data: pd.DataFrame, cfg: "BacktestConfig",
                      label: str = "") -> Result:
    """网格的**挂单**成交模型：档位价上挂着限价单，K 线内触到就成交。

    和 `run_level_backtest`（第 T 根收盘定档、第 T+1 根**开盘**成交）的差别只有
    成交价与成交时点，但这一条差别对网格是决定性的：

    - **建仓**：第 T 根里 `low <= 中枢 − k×格距`（多头第 k 档）就在**那个价位**
      成交，而不是等 T+1 开盘追价。一根大阴线扫穿 3 档就一次成交 3 档 —— 这正
      是「先挂一排限价单，价格扫下来依次成交」的真实结果。
    - **平仓**：每条腿建仓那一刻就在**上一档**挂好止盈价（多头 = 建仓档 + 一格），
      之后哪根 `high >= 止盈价` 就在止盈价平掉。网格赚的就是这一格的差价，不是
      「等收盘信号说该减了再开盘追」。
    - 限价成交的价格口径：买单 `min(开盘, 挂单价)`、卖单 `max(开盘, 挂单价)`
      —— 跳空**只会成交得更好**，绝不会更差。这是限价单的定义，不是乐观假设。
    - 🔴**进场当根不参与止盈**（项目铁律）：一根宽幅 bar 同时触到建仓价和止盈价
      时，分不出先后，新腿从下一根才开始算止盈。
    - 🔴**尾盘强平是唯一的例外**：`grid_flat` 标记的那一根之后，下一根用**开盘价**
      一次性市价全平，不看是否触到挂单价。收盘强平本来就得出在市价上。
    - 方向反了（跳空直接跨过中枢、止盈单来不及成交）：按开盘价市价平掉反向腿。

    需要的列：`open/high/low/close`、`signal`（档位号，只用它判方向与上限）、
    `center`（中枢）、`grid_step`（一格的价格宽度）、`grid_flat`（可选）。
    这三列由 `strategies.strat_grid` 输出。
    """
    idx = data.index
    n = len(data)
    if n == 0:
        return Result(equity=pd.Series(dtype=float), position=pd.Series(dtype=int),
                      trades=[], metrics=_metrics(pd.Series(dtype=float),
                                                  pd.Series(dtype=int), [],
                                                  cfg.bars_per_year,
                                                  base=cfg.initial_cash),
                      label=label)
    open_ = data["open"].to_numpy(dtype=float)
    high = (data["high"].to_numpy(dtype=float) if "high" in data.columns else open_.copy())
    low = (data["low"].to_numpy(dtype=float) if "low" in data.columns else open_.copy())
    close = data["close"].to_numpy(dtype=float)
    center = pd.to_numeric(data["center"], errors="coerce").to_numpy(dtype=float)
    step = pd.to_numeric(data["grid_step"], errors="coerce").to_numpy(dtype=float)
    scale = (pd.to_numeric(data["price_scale"], errors="coerce").fillna(1.0).to_numpy(float)
             if "price_scale" in data.columns else None)
    sig = data["signal"].fillna(0).astype(int).to_numpy()
    if cfg.long_only:
        sig = np.maximum(sig, 0)
    if cfg.short_only:
        sig = np.minimum(sig, 0)
    flat = (data["grid_flat"].fillna(False).astype(bool).to_numpy()
            if "grid_flat" in data.columns else np.zeros(n, dtype=bool))
    note_in = _note_col(data, "entry_note")
    # 网的深度：策略给的 `grid_cap`（= grid_n，整排挂单的档数）。没有这一列才
    # 退回「信号最大档位」——那只是观测值，样本里价格没走到深处就会把网缩小。
    if "grid_cap" in data.columns:
        _capv = pd.to_numeric(data["grid_cap"], errors="coerce").dropna()
        cap = int(_capv.iloc[0]) if len(_capv) else 1
    else:
        cap = int(np.abs(sig).max()) if n else 0
    cap = max(cap, 1)
    # 破网止损：单位是**格宽**，`0` = 关闭。由策略的 `net_stop` 参数经
    # `grid_net_stop` 列传下来 —— 引擎只负责按 high/low 判触发，不自己定价位。
    net_stop = 0.0
    if "grid_net_stop" in data.columns:
        _ns = pd.to_numeric(data["grid_net_stop"], errors="coerce").dropna()
        net_stop = float(_ns.iloc[0]) if len(_ns) else 0.0
    halt_day = None            # 破网止损当天不再开新网
    # 方向几何：`grid_follow=1` 顺势 ⇒ 档位挂在**趋势那一侧**（做多挂在中心之上，
    # 突破一档买一份，止盈再走一格）；`0` 逆势 ⇒ 挂在**回调侧**（做多挂中心之下，
    # 止盈回到中枢那一格）。只翻信号不翻几何 = 档位永远挂在价格到不了的一侧。
    follow = bool(pd.to_numeric(data["grid_follow"], errors="coerce").fillna(0).iloc[0]) \
        if "grid_follow" in data.columns else False
    # 中间震荡带：这一根**多空两张网同时挂**（`strat_grid` 的 `grid_side=2`）。
    # 没有这一列 = 老的单边网，行为逐位不变。
    dual = (pd.to_numeric(data["grid_dual"], errors="coerce").fillna(0)
            .astype(bool).to_numpy()) if "grid_dual" in data.columns \
        else np.zeros(n, dtype=bool)
    # 双向时每个方向只能分到一半预算 —— 中间带里两条腿同时挂着，名义敞口是两倍。
    n_sides = 2.0 if bool(dual.any()) else 1.0
    mult = float(cfg.multiplier or 1.0)

    cash = float(cfg.initial_cash)
    cost_total = 0.0
    notional_total = 0.0
    unit_q = 0.0                      # 一份手数，建仓那一刻定死
    legs: list[dict] = []
    trades: list[Trade] = []
    equity = np.zeros(n, dtype=float)
    pos_state = np.zeros(n, dtype=int)
    max_legs = 0
    mixed_bars = 0                    # 多空同时持有的根数（中间震荡带的判据）
    net_stop_hits = 0                 # 破网止损触发次数（绩效页/自检要能看见）
    blown = False                     # 权益打穿 0 之后整段停手（见循环 ⓪）

    def _slip(price: float, i: int) -> float:
        return cfg.slip_amount(price, scale[i] if scale is not None else 1.0)

    def _close_leg(i: int, leg: dict, price: float, why: str) -> None:
        """按 price（盘口价，滑点另算）平掉整条腿，落一笔交易。"""
        nonlocal cash, cost_total, notional_total
        d = int(leg["dir"])
        q = abs(float(leg["lots"]))
        sp = _slip(price, i)
        fill = price - sp if d > 0 else price + sp
        cash += q * d * mult * (fill - leg["fill"])
        fee = cfg.fee_amount(q, mult, fill)
        cash -= fee
        close_cost = fee + sp * mult * q
        cost_total += close_cost
        notional_total += q * mult * abs(fill)
        trades.append(Trade(
            entry_date=leg["date"], exit_date=idx[i],
            entry_price=float(leg["fill"]), exit_price=float(fill),
            ret=_trade_ret(d, leg["fill"], fill), bars=i - int(leg["bar"]),
            direction=d, lots=float(q),
            # 毛盈亏用不含滑点的挂单价：滑点在 cost 里已经算过一次
            gross=q * d * mult * (float(price) - float(leg["raw"])),
            cost=float(leg["cost"]) + float(close_cost),
            entry_note=leg["why"], exit_note=why))

    for i in range(n):
        c = center[i]
        st = step[i]
        # ⓪ 爆仓线：上一根收盘时权益已经 <= 0，现实里就是被强平出场，不能再
        #    接着开仓 —— 否则回测会一路亏到 −300% 这种根本不存在的净值
        #    （AG 六档空网格遇上单边牛市就是这样）。平掉剩余腿后整段停手。
        # ⓪ 爆仓线：上一根收盘时权益已经 <= 0，现实里就是被强平出场，不能再
        #    接着开仓 —— 否则回测会一路亏到 −300% 这种根本不存在的净值
        #    （AG 六档空网格遇上单边牛市就是这样）。平掉剩余腿后整段停手。
        if not blown and i > 0 and equity[i - 1] <= 0:
            blown = True
        if blown:
            if legs:
                for leg in list(legs):
                    _close_leg(i, leg, float(open_[i]), "爆仓强平")
                legs.clear()
            pos_state[i] = 0
            equity[i] = cash
            continue

        # ① 尾盘 / 换场强平：**唯一**走「开盘市价」的一条
        if i > 0 and flat[i - 1] and legs:
            for leg in list(legs):
                _close_leg(i, leg, float(open_[i]), "尾盘/换场强平")
            legs.clear()

        # 🔴**方向必须取上一根的信号**：第 i 根收盘才算出 `sig[i]`，拿它去决定
        #    第 i 根内挂哪一侧的档位＝用当根的收盘知道该做多、再回头在开盘把单子
        #    挂进去，是标准的前视。上一根的信号在第 i 根开盘前就已经可知，才是
        #    实盘能拿到的信息。档位价用 `center[i]`（= `sma.shift(1)`，同样不含
        #    当根收盘），两者一起保证「只用开盘前已知的信息」。
        _prev_sig = int(sig[i - 1]) if i > 0 else 0
        want = 0 if not np.isfinite(c) else int(np.sign(_prev_sig))
        # ② 这一根允许几个方向：
        #    · `grid_dual` = 上一根落在中间震荡带里 ⇒ **多空两张网同时挂**
        #    · 否则只允许信号指的那一侧（老行为）
        #    · 信号为 0 且不在带里 ⇒ `allow=None` = 「不知道」，已有腿一律不动
        #      （老代码里 `want==0` 也是不动的，这里必须保住：不能把「收盘价刚好
        #      落在中枢上」当成平仓信号）
        _dual = bool(dual[i - 1]) if i > 0 else False
        if _dual:
            allow = (1, -1)
        elif want != 0:
            allow = (want,)
        else:
            allow = None
        # ②a 方向上翻 / 出了中间带（跳空跨过中枢，止盈单来不及成交）：
        #     只平掉**不再允许**的那一侧，不是无脑全平 —— 双向时另一侧本来就该留着。
        #     逆势几何下被平掉的这一侧通常正在浮盈（止盈价在中枢附近还没碰到），
        #     所以这条在震荡带边缘扮演的是「兑现」而不是「止损」。
        if legs and allow is not None:
            bad = [g for g in legs if int(g["dir"]) not in allow]
            for leg in bad:
                _close_leg(i, leg, float(open_[i]), "信号反向")
                legs.remove(leg)

        # ②b 破网止损：价格越过「最外一档再走 net_stop 个格宽」⇒ 整张网全平，
        #     当天不再开新网。挂在那个价位上的是**止损单**，所以成交价按限价单
        #     口径：跳空只会更差（多单 `min(开盘, 止损价)`、空单 `max(...)`）——
        #     和止盈的「只会更好」正好相反，这才是止损该有的方向。
        #     🔴只对**上一根及更早建的腿**生效：当根刚建的腿不参与（进场当根不
        #     出场，同 #101 那条铁律）。所以这一段放在 ③ 建仓之前。
        if legs and net_stop > 0:
            # 🔴双向时**按方向分组各判各的**：两组腿的「网外」方向相反，拿
            #    `legs[0]` 当唯一方向会把另一组当成同向（止损价挂反，触发即亏）。
            for _d in sorted({int(g["dir"]) for g in legs}):
                grp = [g for g in legs if int(g["dir"]) == _d]
                # 🔴只对**上一根及更早建的腿**生效：当根刚建的腿不参与（进场当根
                #    不出场，同 #101 那条铁律）。所以这一段放在 ③ 建仓之前。
                if not all(int(g["bar"]) < i for g in grp):
                    continue
                # 🔴止损必须挂在**亏钱那一侧**（−d0 方向），两种几何的锚点不同：
                #   · 逆势（follow=0）：网铺在中枢的**回调侧**，即 `c − d0·k·格宽`，
                #     所以「网破」＝价格再往那个方向走 → 止损在网外 `c − d0·(cap+ns)·格宽`
                #   · 顺势（follow=1）：网铺在**趋势侧**（`c + d0·k·格宽`），最外一档
                #     本来就在盈利方向，往外挂是止盈不是止损 ⇒ 顺势的「网破」定义是
                #     「价格掉回中枢另一侧」→ 止损在 `c − d0·ns·格宽`
                # 两种都是 `c − d0·…`。写成 `c + d0·…` 会让多单止损挂到价格上方，
                # 建仓下一根就「止损」且还是赚的（实测 SR 5287 建仓、5290 止损）。
                if follow:
                    stop_px = c - _d * net_stop * st
                else:
                    stop_px = c - _d * (cap + net_stop) * st
                hit = (np.isfinite(stop_px) and
                       ((_d > 0 and low[i] <= stop_px) or (_d < 0 and high[i] >= stop_px)))
                if hit:
                    px = min(float(open_[i]), float(stop_px)) if _d > 0 \
                        else max(float(open_[i]), float(stop_px))
                    for leg in grp:
                        _close_leg(i, leg, px, "破网止损")
                        legs.remove(leg)
                    halt_day = pd.Timestamp(idx[i]).date()
                    net_stop_hits += 1

        # ③ 建仓（限价挂单）：中枢已算出、当根允许开网、且不是刚强平完的那根
        flat_now = bool(i > 0 and flat[i - 1])
        halted = halt_day is not None and pd.Timestamp(idx[i]).date() == halt_day
        if (allow is not None and np.isfinite(c) and np.isfinite(st) and st > 0
                and not flat_now and not halted):
            if not legs:
                # `share=n_sides`：中间带里两个方向各占一半预算，否则两条腿叠起来
                # 的名义敞口就是设定杠杆的两倍（单边网时 n_sides=1，逐位不变）。
                unit_q = float(size_lots(cfg, c, cash, 0.0, share=n_sides))
            if unit_q > 0:
                # `taken` 必须带方向：双向时多头第 1 档和空头第 1 档是两个不同价位
                taken = {(int(leg["dir"]), int(leg["level"])) for leg in legs}
                why = (note_in[i] if note_in is not None else "") or label
                # `side` = 档位相对中枢的偏移方向：+1 挂在上方、−1 挂在下方。
                # 顺势跟着仓位方向挂（做多在上面，突破才成交），逆势反着挂
                # （做多挂在下面，跌下来才成交）。止盈也按同一条轴：
                # 顺势再走一格（k+1），逆势回中枢一格（k−1）。
                for _d in allow:
                    side = _d if follow else -_d
                    for k in range(1, cap + 1):
                        if (_d, k) in taken:
                            continue
                        lvl = c + side * k * st
                        if not np.isfinite(lvl) or lvl <= 0:
                            continue
                        if side > 0:
                            if high[i] < lvl:
                                continue      # 没扫到这一档，单子继续挂着
                            px = max(float(open_[i]), float(lvl))
                        else:
                            if low[i] > lvl:
                                continue
                            px = min(float(open_[i]), float(lvl))
                        sp = _slip(px, i)
                        fill = px + sp if _d > 0 else px - sp
                        fee = cfg.fee_amount(unit_q, mult, fill)
                        cash -= fee
                        cost = fee + sp * mult * unit_q
                        cost_total += cost
                        notional_total += unit_q * mult * abs(fill)
                        # 止盈价建仓那一刻定死：顺势再走一格、逆势回中枢那一格
                        off = (k + 1) if follow else (k - 1)
                        tp = c + side * off * st
                        legs.append({"dir": _d, "lots": unit_q * _d,
                                     "raw": float(px), "fill": float(fill),
                                     "tp": float(tp), "level": k,
                                     "cost": float(cost), "date": idx[i],
                                     "bar": i, "why": why})

        # ④ 止盈（限价）。放在建仓之后，但 🔴**同一根 K 线绝不进出**：
        #    建仓那一根从下一根才开始判止盈。理由和 `intrabar_samebar_stop`
        #    是同一条铁律 —— 一根 bar 只有 OHLC，high 完全可能出现在砸下去之前，
        #    「先成交买、再成交卖」是拿 OHLC 编出来的来回，不是盘口事实。
        #    实测这个假设值多少钱：铁矿 5m 放宽（允许同根来回）= +25.3%，
        #    一收紧 = −21.0%，**46 个百分点的差额全在路径假设上**。
        #    这条不做成开关：它不是「悲观/乐观偏好」，是「这根 bar 承载不了这个
        #    判断」。要更多来回只能降周期（1m），不能靠放宽假设。
        if legs:
            alive = []
            for leg in legs:
                if int(leg["bar"]) == i:
                    alive.append(leg)
                    continue
                d = int(leg["dir"])
                tp = float(leg["tp"])
                if d > 0 and high[i] >= tp:
                    _close_leg(i, leg, max(float(open_[i]), tp), "止盈(一格)")
                elif d < 0 and low[i] <= tp:
                    _close_leg(i, leg, min(float(open_[i]), tp), "止盈(一格)")
                else:
                    alive.append(leg)
            legs = alive

        max_legs = max(max_legs, len(legs))
        # 同一根上多空两条腿并存了几根。这是「中间震荡带真的双向在做」的判据 ——
        # 光看笔数分不出「先后做过多和空」与「同时挂着两张网」。
        if len({int(g["dir"]) for g in legs}) > 1:
            mixed_bars += 1
        # 双向时 `legs` 里多空并存，持仓方向只能按**净头寸**报（图上的多空色块
        # 只分得清一边）。这不影响绩效 —— 权益是逐腿带符号累加的。
        _net = sum(float(g["lots"]) for g in legs)
        pos_state[i] = 0 if abs(_net) < 1e-12 else int(np.sign(_net))
        equity[i] = cash + sum(leg["lots"] * mult * (close[i] - leg["fill"])
                               for leg in legs)

    # 未平仓的腿：逐条入账，只算开仓那一侧的成本（与其它路径口径一致）
    for leg in legs:
        d = int(leg["dir"])
        last = float(close[-1])
        trades.append(Trade(
            entry_date=leg["date"], exit_date=idx[-1],
            entry_price=float(leg["fill"]), exit_price=last,
            ret=_trade_ret(d, leg["fill"], last), bars=n - 1 - int(leg["bar"]),
            direction=d, open_position=True, lots=abs(float(leg["lots"])),
            gross=float(leg["lots"]) * mult * (last - float(leg["raw"])),
            cost=float(leg["cost"]), entry_note=leg["why"],
            exit_note="样本结束，仍持仓"))

    eq_series = pd.Series(equity, index=idx, name="equity")
    ps = pd.Series(pos_state, index=idx, name="position")
    m = _metrics(eq_series, ps, trades, cfg.bars_per_year, base=cfg.initial_cash,
                 price=pd.Series(close, index=idx), mult=mult,
                 margin_rate=cfg.margin_rate)
    m["final_equity"] = float(eq_series.iloc[-1])
    m["total_cost"] = float(cost_total)
    m["notional"] = float(notional_total)
    m["avg_cost"] = float(cost_total / m["n_trades"]) if m["n_trades"] else 0.0
    m["cost_rate"] = float(cost_total / notional_total) if notional_total > 0 else 0.0
    # 这张网最多同时铺了几档（信号上限是 cap，实际铺到的深度才是真的）
    m["max_level"] = int(max_legs)
    m["grid_depth_cap"] = int(cap)
    m["net_stop"] = float(net_stop)
    m["net_stop_hits"] = int(net_stop_hits)
    # 中间震荡带：同时挂两个方向的根数占比，以及每方向分到的预算份数。
    # 「两个方向都真的成交过」= 网格的震荡收益到底有没有吃到，看这个。
    m["dual_bars"] = int(dual.sum())
    m["n_sides"] = float(n_sides)
    m["mixed_bars"] = int(mixed_bars)
    return Result(equity=eq_series, position=ps, trades=trades, metrics=m, label=label)


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig | None = None, label: str = "") -> Result:
    """按 signal 列执行回测。df 需含 open/close 与 signal（取值 -1/0/1）。

    带 `entry_long` / `entry_short` 列的策略走 K 线内成交（见
    `run_intrabar_backtest`），**其余一律走原来的 T+1 开盘路径**。老路径
    一行没动，是为了让所有历史回测结果仍能逐位对拍 —— 多一条执行模型不等于
    换掉原来的那一条。
    """
    cfg = cfg or BacktestConfig()
    data = df.dropna(subset=["open", "close"]).copy()
    # 价格必须为正：期货无成交日会返回 0 价，参与运算会导致除零、净值变 inf
    data = data[(data["open"] > 0) & (data["close"] > 0)]
    if "signal" not in data:
        raise ValueError("缺少 signal 列")
    if "entry_long" in data.columns or "entry_short" in data.columns:
        return run_intrabar_backtest(data, cfg, label=label)
    # 分级仓位（网格那一族）：signal 是档位号而不是 -1/0/1。判断放在这里、
    # 而不是让老循环去兼容 3 / -3 —— 老循环里 `pos = want` 之后到处按 pos==1
    # 判多空，-3 会被当成「非多头」走进空头分支，算出来的东西是错的也不报错。
    # 全部是 ±1 时一行都不进这条分支，历史结果逐位不变
    _sig = data["signal"].fillna(0).astype(int).to_numpy()
    if len(_sig) and int(np.abs(_sig).max()) > 1:
        # 网格那一族有两套成交模型：**挂单（K 线内）** 与 收盘定档/次日开盘成交。
        # 只要策略输出了 `grid_step`（一格的价格宽度）就说明它是一张「档位挂单
        # 表」，走挂单模型；没有这一列（老的自造档位信号）仍走原来那条，历史
        # 结果逐位不变。
        if cfg.grid_intrabar and "grid_step" in data.columns:
            return run_grid_intrabar(data, cfg, label=label)
        return run_level_backtest(data, cfg, label=label)

    open_ = data["open"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    # 复权因子：数据层把前复权用过的缩放系数挂在 `price_scale` 列上，滑点要按它
    # 换回真实的一跳。没有这一列（不复权、美股、手工造的 df）就是 1.0
    scale = (pd.to_numeric(data["price_scale"], errors="coerce").fillna(1.0).to_numpy(float)
             if "price_scale" in data.columns else None)
    sig = data["signal"].fillna(0).astype(int).to_numpy()
    if cfg.long_only:
        sig = np.maximum(sig, 0)
    if cfg.short_only:
        # 禁止做多：把 +1 砍成 0，只留 -1 / 0。和 long_only 对称，两个都开等于
        # 全空仓（上层 UI 会拦住这种组合，这里只保证引擎语义明确）
        sig = np.minimum(sig, 0)
    idx = data.index
    # 逐根的「进场 / 出场原因」，可选列。策略自己写（因子策略会写「因子分
    # +2.31 越过进场阈值 2」「分档浮盈回撤」这种），没写就留空由引擎兜底
    note_in = _note_col(data, "entry_note")
    note_out = _note_col(data, "exit_note")
    n = len(data)
    mult = float(cfg.multiplier or 1.0)
    mode = (cfg.sizing or "unit").lower()

    atr_arr = None
    if mode == "risk":
        atr_arr = _atr_array(close, data["high"].to_numpy(dtype=float)
                             if "high" in data else close,
                             data["low"].to_numpy(dtype=float)
                             if "low" in data else close,
                             cfg.stop_atr_n)

    # 目标仓位：T 日收盘的信号，T+1 开盘执行
    target = np.roll(sig, 1)
    target[0] = 0

    cash = float(cfg.initial_cash)
    cost_total = 0.0             # 手续费 + 滑点，**金额**，绩效表要看它
    # 成交名义额累计。成本金额除以它才是「每笔交易被吃掉多少」，绩效页那一行
    # 用的是这个比例而不是金额 —— 金额随本金和手数变，比例才是成本本身
    notional_total = 0.0
    lots = 0.0                   # 正=多，负=空，0=空仓（unit 口径是份额）
    pos = 0                      # 1=多，-1=空，0=平
    # 这一笔**开仓时**花掉的成本（手续费 + 滑点，金额）。平仓时要把它和
    # 平仓那一侧的成本合起来记进同一笔交易，否则「这笔交易的成本」只剩一半，
    # 看上去像只收了单边
    entry_cost = 0.0
    # 开仓时的**盘口价**（不含滑点）。毛盈亏要用它算，不能用 `entry_price`
    # —— 那个已经含了滑点，两边一起算会把滑点扣两遍
    entry_raw = 0.0
    pos_state = np.zeros(n, dtype=int)
    equity = np.zeros(n, dtype=float)
    trades: list[Trade] = []
    entry_price = entry_date = None
    entry_bar = 0
    entry_dir = 1

    def close_position(i: int, price: float, held: float, why: str = "",
                       cost: float = 0.0, raw_price: float | None = None) -> None:
        nonlocal entry_price, entry_date
        if entry_price is None:
            return
        # 毛盈亏按**不含滑点**的盘口价算：滑点在 `cost` 里已经算过一次了
        raw_exit = float(price if raw_price is None else raw_price)
        trades.append(
            Trade(
                entry_date=entry_date,
                exit_date=idx[i],
                entry_price=float(entry_price),
                exit_price=float(price),
                ret=_trade_ret(entry_dir, entry_price, price),
                bars=i - entry_bar,
                direction=entry_dir,
                lots=abs(float(held)),
                gross=float(held) * mult * (raw_exit - float(entry_raw)),
                # 开仓那一侧的 `entry_cost` + 平仓这一侧的 `cost` = 双边
                cost=float(entry_cost) + float(cost),
                entry_note=entry_why,
                exit_note=why,
            )
        )
        entry_price = entry_date = None

    entry_why = ""

    for i in range(n):
        want = target[i]
        if want != pos:
            # 这一根的动作是**上一根收盘**定下的（`target = roll(sig, 1)`），
            # 所以原因也要从上一根取 —— 拿当根去解释，会写出「用今天收盘的
            # 信号解释昨天开盘的平仓」，时间轴反了
            dec = i - 1 if i > 0 else 0
            # ---- 先平掉现有仓位 ----
            if pos != 0:
                sp = cfg.slip_amount(open_[i], scale[i] if scale is not None else 1.0)
                price = open_[i] - sp if pos == 1 else open_[i] + sp
                # 平仓：浮动盈亏落地，再扣平仓手续费
                cash += lots * mult * (price - entry_price)
                fee = cfg.fee_amount(lots, mult, price)
                cash -= fee
                # 滑点按「每手多少价差 × 乘数 × 手数」折成钱，和手续费一起记
                close_cost = fee + sp * mult * abs(lots)
                cost_total += close_cost
                notional_total += abs(lots) * mult * abs(price)
                why = (note_out[dec] if note_out is not None else "") or (
                    "信号反向" if want == -pos else "信号离场")
                close_position(i, price, lots, why, cost=close_cost,
                               raw_price=open_[i])
                lots = 0.0
                pos = 0
                entry_cost = 0.0

            # ---- 再按需开新仓 ----
            if want != 0:
                sp = cfg.slip_amount(open_[i], scale[i] if scale is not None else 1.0)
                price = open_[i] + sp if want > 0 else open_[i] - sp
                a = float(atr_arr[i]) if atr_arr is not None else 0.0
                q = size_lots(cfg, price, cash, a if a == a else 0.0)
                if q > 0:
                    fee = cfg.fee_amount(q, mult, price)      # 开仓手续费
                    cash -= fee
                    # 开仓这一侧的成本先记着，等平仓时并进同一笔交易
                    entry_cost = fee + sp * mult * q
                    cost_total += entry_cost
                    notional_total += q * mult * abs(price)
                    lots = q if want > 0 else -q
                    entry_price, entry_date, entry_bar, entry_dir = price, idx[i], i, want
                    entry_raw = float(open_[i])
                    entry_why = (note_in[dec] if note_in is not None else "") or label
                    pos = want

        eq = cash + (lots * mult * (close[i] - entry_price) if pos != 0 else 0.0)

        pos_state[i] = pos
        equity[i] = eq

    # 未平仓的最后一笔
    if pos != 0 and entry_price is not None:
        last = close[-1]
        trades.append(
            Trade(
                entry_date=entry_date,
                exit_date=idx[-1],
                entry_price=float(entry_price),
                exit_price=float(last),
                ret=_trade_ret(entry_dir, entry_price, last),
                bars=n - 1 - entry_bar,
                direction=entry_dir,
                open_position=True,
                lots=abs(float(lots)),
                # 没平仓 = 只付了开仓那一侧的成本。别按「双边」估，那会把还没
                # 发生的钱先记成花掉了。估值用最后一根收盘（盘口价，不含滑点）
                gross=float(lots) * mult * (float(last) - float(entry_raw)),
                cost=float(entry_cost),
                entry_note=entry_why,
                exit_note="样本结束，仍持仓",
            )
        )

    eq = pd.Series(equity, index=idx, name="equity")
    ps = pd.Series(pos_state, index=idx, name="position")
    m = _metrics(eq, ps, trades, cfg.bars_per_year, base=cfg.initial_cash,
                 price=pd.Series(close, index=idx), mult=mult,
                 margin_rate=cfg.margin_rate)
    m["final_equity"] = float(eq.iloc[-1])
    # 成本单列出来：手续费和滑点已经扣进权益里了，但用户要能回答「这些钱花在哪」
    m["total_cost"] = float(cost_total)
    m["notional"] = float(notional_total)
    m["avg_cost"] = float(cost_total / m["n_trades"]) if m["n_trades"] else 0.0
    # 成本率 = 花掉的钱 ÷ 成交额。绩效页那一行显示的是它（百分比），不是金额 ——
    # 金额随本金和手数变，比例才是「每笔交易被吃掉多少」
    m["cost_rate"] = float(cost_total / notional_total) if notional_total > 0 else 0.0
    return Result(equity=eq, position=ps, trades=trades, metrics=m, label=label)


def buy_and_hold(df: pd.DataFrame, cfg: BacktestConfig | None = None,
                 short: bool = False) -> Result:
    """买入持有基准：首个交易日开盘全仓买入并一直持有。

    `short=True` 是它的镜像 —— 首日**卖空**并一直拿着，给「禁止做多」的策略
    当对照。不镜像的话，一条只做空的策略会去跟「一直做多」比，比出来的超额
    没有意义。
    """
    cfg = cfg or BacktestConfig(long_only=True)
    d = df.dropna(subset=["open", "close"]).copy()
    d = d[(d["open"] > 0) & (d["close"] > 0)]
    d["signal"] = 1
    # 首日就建仓（run_backtest 会延迟一天），用等价方式直接算
    open_ = d["open"].to_numpy(dtype=float)
    close = d["close"].to_numpy(dtype=float)
    price = open_[0] + cfg.slip_amount(
        open_[0],
        float(d["price_scale"].iloc[0]) if "price_scale" in d.columns else 1.0)
    mult = float(cfg.multiplier or 1.0)
    # 基准要和策略用同一套仓位口径，否则「策略 vs 买入持有」是在比杠杆，
    # 不是在比策略。期货上尤其明显：策略 10 倍杠杆、基准全额，比出来毫无意义
    q = size_lots(cfg, price, cfg.initial_cash, 0.0)
    fee0 = cfg.fee_amount(q, mult, price)
    cash = cfg.initial_cash - fee0
    # 空头就是把 q 取负：`cash + q×乘数×(收 − 开)` 对正负都成立
    held = -float(q) if short else float(q)
    equity = cash + held * mult * (close - price)
    eq = pd.Series(equity, index=d.index, name="equity")
    ps = pd.Series(-1 if short else 1, index=d.index, name="position")
    trade = Trade(
        entry_date=d.index[0],
        exit_date=d.index[-1],
        entry_price=float(price),
        exit_price=float(close[-1]),
        ret=float(1.0 - close[-1] / price) if short else float(close[-1] / price - 1.0),
        bars=len(d) - 1,
        direction=-1 if short else 1,
        open_position=True,
        lots=abs(float(q)),
        # 基准只开了仓、一直没平，所以成本只有开仓那一侧（手续费 + 滑点）。
        # 毛盈亏用**盘口价**（`open_[0]`，不含滑点），滑点在成本里已经算过
        gross=held * mult * (float(close[-1]) - float(open_[0])),
        cost=float(fee0) + abs(float(price - open_[0]) * q * mult),
    )
    m = _metrics(eq, ps, [trade], cfg.bars_per_year, base=cfg.initial_cash,
                 price=pd.Series(close, index=d.index), mult=mult,
                 margin_rate=cfg.margin_rate)
    m["final_equity"] = float(eq.iloc[-1])
    # 基准只有一次开仓（而且一直没平），成本就是这么一点 —— 正好当对照：
    # 策略那条如果成本是这个数的几十倍，说明换手才是真正的敌人
    slip0 = (price - open_[0]) * q * mult
    m["total_cost"] = float(fee0 + abs(slip0))
    m["avg_cost"] = float(m["total_cost"])
    m["notional"] = float(abs(q) * mult * price)
    m["cost_rate"] = float(m["total_cost"] / m["notional"]) if m["notional"] > 0 else 0.0
    return Result(equity=eq, position=ps, trades=[trade], metrics=m,
                  label="卖空持有" if short else "买入持有")


def _to_daily(eq: pd.Series) -> pd.Series:
    """把一个净值 / 仓位序列压成「每个交易日一个点」，取当天最后一根。

    分钟级一天有几十个点。直接按时间戳并集和日线策略对齐的话，组合的均值会在
    「只有分钟级参与」和「两条都参与」之间来回跳 —— 画出来是一条上下翻飞的
    带子，看着像净值在乱跳，其实是成分每天都在变。统一到交易日之后，每一天
    都是同一组策略的盯市净值，组合的每日盈亏才可比。
    """
    s = eq.dropna()
    if s.empty:
        return s
    return s.groupby(s.index.normalize()).last()


def combine(results: list[Result], label: str = "组合",
            initial_cash: float = 100_000.0) -> Result:
    """把多条资金曲线合成一条等权组合曲线。

    **逐日盯市**：每条曲线先压成「每个交易日一个点」（当天最后一根），再对齐、
    取均值。单条策略本身就是逐根盯市的（`cash + shares * close`），这里只是
    把它落到交易日这个共同刻度上 —— 混合周期（30 分钟 + 1 小时 + 日线）的组合
    也只有这样才说得清「今天一共赚了多少」。

    **成分的起止不能让它从均值里消失**。这一点很要紧：两条曲线覆盖的日期往往
    差一天（一个 30 分钟策略的最后一根在 14:55，一个 1 小时的在次日 11:00，
    或者本地数据更新进度不同）。如果那条先结束的直接不参与当天的均值，组合
    净值会从「两条的平均」掉成「另一条自己」—— 看起来是末端凭空跳水 15%，
    而交易记录上那天还是赚的。

    - 还没上线的日子按 **1.0（现金）** 计：钱已经分给它了，只是还没开始交易
    - 数据结束之后沿用**最后一个净值**：停手了，钱还在，只是不再赚亏

    两头都不跳，组合净值也始终等于「各策略净值之和 ÷ 条数」，和「每个策略
    10 万、加起来除以二」的直觉一致。`ffill()` 之后 `fillna(1.0)` 正好是这两件事：
    ffill 只往前带（头部没值填不了），剩下的空只可能是头部。

    **先各自归一化到 1 再平均**，不是直接平均净值：一个品种 3000 点、另一个
    80 块，初始资金也可能不同，直接平均出来的数字没有任何含义。

    平均的是**净值水平**，所以这是「初始等权、之后不再平衡」的组合：表现好的
    那条占的份额会慢慢变大，和持有一篮子标的的感觉一致。若按「每日再平衡」
    算（把每天的收益取均值再连乘），调仓成本没计进去，收益会略好看一点 ——
    两种口径都能自圆其说，这里选更保守的那种，并在标题里写明「等权」。

    某条策略还没上线的日子按 1.0 计（见上），不是不参与 —— 用 skipna 跳过的话
    均值会在它上线那天从「一条」变成「两条」，一样是假跳变。

    交易记录用 `combine_trades()` 合并，别直接拼接：拼接出来是「A 的全部交易，
    然后 B 的全部交易」，看不出两条策略其实在同一段时间里并行。
    """
    curves: list[pd.Series] = []
    poss: list[pd.Series] = []
    trades: list[Trade] = []
    # 亏穿的成分个数（见下面那条 `wiped` 的处理）
    blown = 0
    # 各成分的成本占自己初始权益的比例。组合曲线是「各自归一化后取均值 × 初始
    # 资金」，每条成分在组合里占 1/n，所以它的成本也要按这个份额折 —— 直接把
    # 金额加起来是错的：一条 10 万本金的策略和一条 100 万的，成本差十倍不代表
    # 谁更能花
    cost_pcts: list[float] = []
    # 成交额也按同样的份额折一次：成本率是「成本 ÷ 成交额」，两个比例相除
    # 正好把「每条成分占 1/n」约掉，和直接拿金额相除是同一个数
    notional_pcts: list[float] = []
    # 保证金占用同理：各成分的占用折成「占自己本金的比例」，再按 1/n 折到组合。
    # 平均占用是**精确**的（均值可加：Σ 各成分日均 ÷ n 就是组合日均），
    # 峰值占用不是 —— 各成分的峰值在不同时刻发生，求和只是**上限**
    margin_pcts: list[float] = []
    peak_pcts: list[float] = []
    for r in results:
        eq = _to_daily(r.equity)
        if len(eq) < 2:
            continue
        curve = eq / float(eq.iloc[0])
        # ---- 亏穿的成分按「份额归零」算，不是按负净值算
        #
        # 单品种回测**故意不强平**（见 `check_no_blowup`）：杠杆比实盘敢开的大，
        # 强平掉会把后半段历史抹掉，留着负净值才能看清这条路到底怎么死的。
        # 但那是「一个账户自己的事」。等权组合是**净值曲线取平均**，一条 -3.5 的
        # 曲线进去，等于说这个品种倒欠 3.5 份本金 —— 现实里它到 0 就被强平了，
        # 不可能接着亏。用户报过的现象就是这么来的：78 个品种里 20 个转负，
        # 组合被拖到 -47%、最大回撤 -133%、夏普一起失去意义。
        #
        # 所以这里一次性钉死：跌破 0 那根起，这条成分的份额就是 0（本金亏光），
        # 之后即使「负现金」又赚回来也不复活 —— 那笔钱已经不存在了
        wiped = curve.index[curve.to_numpy() <= 0]
        if len(wiped):
            curve = curve.copy()
            curve.loc[wiped[0]:] = 0.0
            blown += 1
        curves.append(curve)
        poss.append(_to_daily(r.position))
        trades += list(r.trades)
        base = float(eq.iloc[0])
        cost_pcts.append(float(r.metrics.get("total_cost") or 0.0) / base
                         if base > 0 else 0.0)
        notional_pcts.append(float(r.metrics.get("notional") or 0.0) / base
                             if base > 0 else 0.0)
        margin_pcts.append(float(r.metrics.get("avg_margin") or 0.0) / base
                           if base > 0 else 0.0)
        peak_pcts.append(float(r.metrics.get("peak_margin") or 0.0) / base
                         if base > 0 else 0.0)

    if not curves:
        blank = pd.Series(dtype=float)
        return Result(equity=blank, position=blank, trades=[], metrics={}, label=label)

    grid = pd.DatetimeIndex(sorted(set().union(*[c.index for c in curves])))
    # ffill 把「数据结束」往后带，fillna(1.0) 补的是头部「还没上线」——
    # 两种都是钱还在、只是不动，绝不能让成分从均值里掉出去
    mat = pd.concat([c.reindex(grid).ffill().fillna(1.0) for c in curves], axis=1)
    eq = mat.mean(axis=1) * float(initial_cash)

    pmat = pd.concat([p.reindex(grid).ffill().fillna(0.0) for p in poss], axis=1)
    pos = pmat.mean(axis=1)

    # 年化口径按组合自己的日频算。沿用某一条成分的 bars_per_year 会错得离谱：
    # 一条 30 分钟策略一年几千根，拿它去年化日频曲线，夏普会虚高一个量级
    bpy = bars_per_year(eq.index)
    m = _metrics(eq, pos, trades, bpy, base=float(initial_cash))
    m["final_equity"] = float(eq.iloc[-1])
    total_cost = (sum(cost_pcts) / len(cost_pcts)) * float(initial_cash) if cost_pcts else 0.0
    m["total_cost"] = float(total_cost)
    m["avg_cost"] = float(total_cost / m["n_trades"]) if m["n_trades"] else 0.0
    total_notional = (sum(notional_pcts) / len(notional_pcts)) * float(initial_cash) \
        if notional_pcts else 0.0
    m["notional"] = float(total_notional)
    m["cost_rate"] = float(total_cost / total_notional) if total_notional > 0 else 0.0
    # 组合的资金占用：平均占用可加，精确；峰值占用是各成分峰值之和，是**上限**
    n_pct = len(margin_pcts)
    if n_pct:
        m["avg_margin"] = sum(margin_pcts) / n_pct * float(initial_cash)
        m["peak_margin"] = sum(peak_pcts) / n_pct * float(initial_cash)
        m["util"] = (m["avg_margin"] / float(initial_cash)
                     if float(initial_cash) > 0 else 0.0)
        m["peak_is_bound"] = 1.0
    # 亏光的成分个数：等权组合看到 -47% 时，用户需要知道是「20 个品种本金亏光」
    # 还是「每个都亏一点」，这两种情况的处理方式完全不同
    m["n_wiped"] = float(blown)
    return Result(equity=eq, position=pos, trades=trades, metrics=m, label=label)


def combine_trades(named: list[tuple[str, Result]]) -> list[tuple[str, Trade]]:
    """把几个策略的逐笔交易合成一条时间线，按开仓时间排序。

    顺序很重要：按策略分组拼接的话，表格是「A 从 2023 年一路做到 2026 年，
    然后 B 又从 2023 年开始」—— 看起来像先后跑了两遍，而不是两条策略在同一段
    时间里并行。按开仓时间排，读起来才是真正的时间线。
    """
    out = [(name, t) for name, r in named for t in r.trades]
    out.sort(key=lambda nt: (nt[1].entry_date, nt[1].exit_date))
    return out


def drawdown_series(equity: pd.Series, base: float | None = None) -> pd.Series:
    """回撤曲线：(权益 − 历史峰值) ÷ **本金**，和绩效页的「最大回撤」同一口径。

    本金是单利框架里唯一的分母。用「相对当时权益」的话，赚了一半之后同样的
    回落只写出 −13%（而不是 −20%），回撤会随着盈利越变越好看 —— 那是复利
    口径，和卡玛的分子（年化，相对本金）也对不上。
    """
    eq = pd.to_numeric(equity, errors="coerce")
    b = float(base or (eq.dropna().iloc[0] if eq.notna().any() else 0.0) or 0.0)
    if b <= 0:
        return eq * 0.0
    return (eq - eq.cummax()) / b


def yearly_returns(equity: pd.Series, base: float | None = None) -> pd.Series:
    """按自然年计算收益率（**单利**：当年盈亏 ÷ 本金）。

    和 `stage_table` 同一个口径 —— 各年**相加**等于累计收益，而「年末 ÷ 上年末」
    那种复利算法下，各年相乘才等于累计。
    """
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if eq.empty:
        return pd.Series(dtype=float, name="return")
    base0 = float(base if base else eq.iloc[0])
    yr = eq.resample("YE").last()
    prev = yr.shift(1)
    prev.iloc[0] = base0
    return ((yr - prev) / base0).rename("return")


def stage_table(equity: pd.Series, freq: str = "YE",
                base: float | None = None) -> pd.DataFrame:
    """按自然周期切段：段收益、段内最大回撤、段内波动、根数。

    **单利口径**：段收益 = (本段末 − 上段末) ÷ 本金。原来是「本段末 ÷ 上段末 − 1」，
    那是复利 —— 十二个月得**连乘**才等于全年，而本金固定时每个月的盈亏都是直接
    加在账户上的，月度收益**相加**就该等于年度收益。用上段末当分母还有个毛病：
    赚钱之后同样的盈利会被算成更小的百分比，前后两个月没法比。

    基准取「上段末」而不是「段内第一根」：第一根只是这个月的第一天，拿它当基准
    会把「上月末到本月初」那段跳空吞掉。

    段内最大回撤的峰值在段首重置，且相对峰值而不是本金：它回答的是「这个月里
    最多回撤了多少」，不是「从历史最高点回撤了多少」。
    """
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 2:
        return pd.DataFrame(columns=["period", "ret", "max_dd", "vol", "bars"])

    base0 = float(base if base else eq.iloc[0])
    prev = base0
    rows = []
    for key, seg in eq.groupby(pd.Grouper(freq=freq)):
        seg = seg.dropna()
        if seg.empty:
            continue
        last = float(seg.iloc[-1])
        ret = (last - prev) / base0 if base0 > 0 else 0.0
        # 段内回撤同样以本金为分母（和 `_metrics` 一致），不是段内峰值的百分比
        dd = (float(((seg - seg.cummax()) / base0).min())
              if base0 > 0 else 0.0)
        # 段内波动按根数年化，和 `_metrics` 一样走「当日盈亏 ÷ 本金」
        r = (seg.diff().dropna() / base0) if base0 > 0 else seg.diff().dropna() * 0.0
        vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
        rows.append({"period": key, "ret": ret, "max_dd": dd, "vol": vol, "bars": len(seg)})
        prev = last
    return pd.DataFrame(rows)


def month_matrix(equity: pd.Series, base: float | None = None) -> pd.DataFrame:
    """月度收益矩阵：行=年，列=1..12，值=当月收益（缺失为 NaN）。

    单利口径，所以**同一行的十二个月相加就是这一年的年度收益**，不用拿年度
    数字倒推。年度和月度分开算（都走 `stage_table`），首尾不满整年的那两段
    也不会互相拖累。
    """
    st = stage_table(equity, "ME", base=base)
    if st.empty:
        return pd.DataFrame()
    df = st.assign(year=st["period"].dt.year, month=st["period"].dt.month)
    return df.pivot_table(index="year", columns="month", values="ret", aggfunc="last")

