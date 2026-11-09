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
    # 允许「同一根 K 线里先进场、再被止损」。默认 **开**：没有分笔数据时
    # 「先冲高再砸下来」和「先砸再冲高」长得一模一样，按对自己有利的那个
    # 假设等于白送 alpha。
    #
    # 但要知道代价：止损距离比这根 K 线的振幅还小时（比如 5 分钟上
    # 1.5×ATR 只有一跳半），突破那根往往就是最宽的一根，于是三成以上的仓
    # 会在**同一根**里被扫掉。那种结果不是策略烂，是**这根 K 线的信息量
    # 不够**。想看「只从下一根开始止损」的乐观口径，把它设 False 再跑一遍，
    # 两个数字的差就是「同根内不可知」这件事值多少钱。
    intrabar_samebar_stop: bool = True

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
    "n_trades", "n_long", "n_short",
    "win_rate", "avg_trade", "profit_factor", "payoff", "avg_win", "avg_loss",
    "avg_bars", "best_trade", "worst_trade",
)


def _metrics(equity: pd.Series, position: pd.Series, trades: list[Trade],
             bars_per_year: float = float(TRADING_DAYS),
             base: float | None = None) -> dict:
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


def size_lots(cfg: BacktestConfig, price: float, equity: float, atr_value: float = 0.0) -> float:
    """这一笔该下多少手。

    `unit` 口径返回的是**手数**（可以是小数，名义敞口 = 权益），其余口径返回整数手。
    手数为 0 表示这笔不做（钱不够一手保证金）。

    **四种口径都是单利**：手数只看「本金（或分配的那部分）/ 止损金额 / 固定
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
        return float(max(1, int(float(cfg.risk_amount or 0) / per)))
    if mode == "capital":
        per = price * mult * float(cfg.margin_rate or 0)
        if per <= 0:
            return 1.0
        # 分配资金 ÷ 每手保证金，**固定用分配的那笔钱**，不随盈亏变。
        # 按当前权益收口就变成复利了：50 万赚到 80 万会自己加到 80 万的仓，
        # 亏到 30 万又自动减到 30 万 —— 那是另一套策略，不是用户要的「单利」
        return float(max(int(cfg.budget / per), 0))
    # 名义敞口 = **本金**（固定）。原来是「当前权益」，权益从 10 万涨到 34 万，
    # 手数跟着放大 3.4 倍 —— 那是复利，和其余三种口径不是一套东西：双均线 /
    # 唐奇安 / 海龟 / 通道突破 / MACD 这几条老策略因此会跑出「越赚越猛」的曲线，
    # 和固定手数的结果没法比。改成盯本金之后，四种口径统一是单利。
    # 期货一手是 price × 乘数，不除乘数的话「不加杠杆」会变成 10 倍杠杆
    return float(max(float(cfg.initial_cash or equity or 0.0), 0.0) / (price * mult))


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
    pos_state = np.zeros(n, dtype=int)
    equity = np.zeros(n, dtype=float)
    trades: list[Trade] = []

    def _slip(price: float, i: int) -> float:
        return cfg.slip_amount(price, scale[i] if scale is not None else 1.0)

    def do_close(i: int, raw_px: float, why: str) -> None:
        """平掉当前仓位。`raw_px` 是**不含滑点**的成交基准价。"""
        nonlocal cash, lots, pos, entry_price, entry_date, entry_cost
        nonlocal cost_total, notional_total, stop_px
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

    def do_open(i: int, raw_px: float, direction: int, why: str) -> bool:
        """开仓。`raw_px` 是**不含滑点**的成交基准价（触发价或开盘价）。"""
        nonlocal cash, lots, pos, entry_price, entry_raw, entry_date
        nonlocal entry_bar, entry_dir, entry_cost, entry_why, stop_px
        nonlocal cost_total, notional_total
        sp = _slip(raw_px, i)
        fill = raw_px + sp if direction > 0 else raw_px - sp
        # risk 口径要的是 ATR 本身，而这一族给的是「倍数 × ATR」后的距离
        a = 0.0
        if _finite(stop_d[i]) and float(cfg.stop_mult or 0) > 0:
            a = float(stop_d[i]) / float(cfg.stop_mult)
        q = size_lots(cfg, fill, cash, a)
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

        # ① 上一根收盘给的出场：本根开盘执行
        if pos != 0 and i > 0:
            want_out = (x_long[i - 1] > 0.5) if pos == 1 else (x_short[i - 1] > 0.5)
            if want_out:
                why = (note_out[i - 1] if note_out is not None else "") or "反向突破离场"
                do_close(i, o, why)

        # ② 已有仓位的止损（K 线内；开盘已穿过就按开盘价，更差）
        if pos != 0 and _finite(stop_px):
            if pos == 1 and lo <= stop_px:
                do_close(i, min(o, stop_px), "ATR 止损")
            elif pos == -1 and hi >= stop_px:
                do_close(i, max(o, stop_px), "ATR 止损")

        # ③ 空仓时看本根有没有触发入场
        if pos == 0:
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
                    # 🔴 这里只能按 `stop_px` 成交，**不能**再取 `min(o, stop_px)`：
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
    m = _metrics(eq, ps, trades, cfg.bars_per_year, base=cfg.initial_cash)
    m["final_equity"] = float(eq.iloc[-1])
    m["total_cost"] = float(cost_total)
    m["notional"] = float(notional_total)
    m["avg_cost"] = float(cost_total / m["n_trades"]) if m["n_trades"] else 0.0
    m["cost_rate"] = float(cost_total / notional_total) if notional_total > 0 else 0.0
    m["intrabar"] = 1.0
    return Result(equity=eq, position=ps, trades=trades, metrics=m, label=label)


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

    open_ = data["open"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    # 复权因子：数据层把前复权用过的缩放系数挂在 `price_scale` 列上，滑点要按它
    # 换回真实的一跳。没有这一列（不复权、美股、手工造的 df）就是 1.0
    scale = (pd.to_numeric(data["price_scale"], errors="coerce").fillna(1.0).to_numpy(float)
             if "price_scale" in data.columns else None)
    sig = data["signal"].fillna(0).astype(int).to_numpy()
    if cfg.long_only:
        sig = np.maximum(sig, 0)
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
    m = _metrics(eq, ps, trades, cfg.bars_per_year, base=cfg.initial_cash)
    m["final_equity"] = float(eq.iloc[-1])
    # 成本单列出来：手续费和滑点已经扣进权益里了，但用户要能回答「这些钱花在哪」
    m["total_cost"] = float(cost_total)
    m["notional"] = float(notional_total)
    m["avg_cost"] = float(cost_total / m["n_trades"]) if m["n_trades"] else 0.0
    # 成本率 = 花掉的钱 ÷ 成交额。绩效页那一行显示的是它（百分比），不是金额 ——
    # 金额随本金和手数变，比例才是「每笔交易被吃掉多少」
    m["cost_rate"] = float(cost_total / notional_total) if notional_total > 0 else 0.0
    return Result(equity=eq, position=ps, trades=trades, metrics=m, label=label)


def buy_and_hold(df: pd.DataFrame, cfg: BacktestConfig | None = None) -> Result:
    """买入持有基准：首个交易日开盘全仓买入并一直持有。"""
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
    equity = cash + q * mult * (close - price)
    eq = pd.Series(equity, index=d.index, name="equity")
    ps = pd.Series(1, index=d.index, name="position")
    trade = Trade(
        entry_date=d.index[0],
        exit_date=d.index[-1],
        entry_price=float(price),
        exit_price=float(close[-1]),
        ret=float(close[-1] / price - 1.0),
        bars=len(d) - 1,
        direction=1,
        open_position=True,
        lots=float(q),
        # 基准只开了仓、一直没平，所以成本只有开仓那一侧（手续费 + 滑点）。
        # 毛盈亏用**盘口价**（`open_[0]`，不含滑点），滑点在成本里已经算过
        gross=float(q) * mult * (float(close[-1]) - float(open_[0])),
        cost=float(fee0) + abs(float(price - open_[0]) * q * mult),
    )
    m = _metrics(eq, ps, [trade], cfg.bars_per_year, base=cfg.initial_cash)
    m["final_equity"] = float(eq.iloc[-1])
    # 基准只有一次开仓（而且一直没平），成本就是这么一点 —— 正好当对照：
    # 策略那条如果成本是这个数的几十倍，说明换手才是真正的敌人
    slip0 = (price - open_[0]) * q * mult
    m["total_cost"] = float(fee0 + abs(slip0))
    m["avg_cost"] = float(m["total_cost"])
    m["notional"] = float(abs(q) * mult * price)
    m["cost_rate"] = float(m["total_cost"] / m["notional"]) if m["notional"] > 0 else 0.0
    return Result(equity=eq, position=ps, trades=[trade], metrics=m, label="买入持有")


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

