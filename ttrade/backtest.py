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
    fee_rate: float = 0.0005      # 单边手续费 5bp
    slippage: float = 0.0005      # 单边滑点 5bp
    long_only: bool = False       # True 时把 -1 裁成 0（股票市场默认）
    bars_per_year: float = float(TRADING_DAYS)   # 日线默认；分钟级由调用方按数据推断

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

    @property
    def label(self) -> str:
        return "只做多" if self.long_only else "多空双向"

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


@dataclass
class Result:
    equity: pd.Series
    position: pd.Series
    trades: list[Trade]
    metrics: dict = field(default_factory=dict)
    label: str = ""


_METRIC_KEYS = (
    "total_return", "cagr", "ann_vol", "sharpe", "max_dd", "calmar",
    "exposure", "long_exposure", "short_exposure",
    "n_trades", "n_long", "n_short",
    "win_rate", "avg_trade", "profit_factor",
    "avg_bars", "best_trade", "worst_trade",
)


def _metrics(equity: pd.Series, position: pd.Series, trades: list[Trade],
             bars_per_year: float = float(TRADING_DAYS)) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {k: 0.0 for k in _METRIC_KEYS}

    # 权益可能被亏穿成负数（回测不再强平，见 `run_backtest`）。拿负数当基数算
    # 收益率没有意义 —— -100 亏到 -50 会算出 +50% —— 所以只保留「上一根还是
    # 正的」那些，之后的收益记 0，波动和夏普不会再被翻号后的假反弹污染
    ret = equity.pct_change().fillna(0.0)
    ret = ret.where(equity.shift(1) > 0, 0.0)
    ratio = equity.iloc[-1] / equity.iloc[0]
    total = ratio - 1.0

    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    # 负数开 1/years 次方是复数，直接算「全部亏完」
    if years <= 0:
        cagr = 0.0
    elif ratio <= 0:
        cagr = -1.0
    else:
        cagr = ratio ** (1 / years) - 1

    bpy = float(bars_per_year or TRADING_DAYS)
    vol = ret.std(ddof=1) * np.sqrt(bpy) if ret.std(ddof=1) > 0 else 0.0
    sharpe = (ret.mean() * bpy) / vol if vol > 0 else 0.0

    dd_series = equity / equity.cummax() - 1.0
    max_dd = float(dd_series.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    # 权益一旦被亏穿，「相对上一根」的百分比收益会在 0 附近炸开（300 → 20000
    # 就是 +6500%），年化波动能算出六位数、夏普随之失去意义，卡玛也是。这三个
    # 是**比例型**指标，亏穿过就标 NaN，界面显示「—」；累计收益、最大回撤、
    # 期末资金这些绝对值指标照常给
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
        avg_bars = float(np.mean([t.bars for t in closed]))
        best, worst = float(rets.max()), float(rets.min())
    else:
        win_rate = avg_trade = profit_factor = avg_bars = best = worst = 0.0

    return {
        "total_return": float(total),
        "cagr": float(cagr),
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

    `capital` 和 `risk` 都是**单利**：手数只看「分配的资金 / 止损金额」，不看
    账户当前权益。赚了不加大仓位，亏了也不缩 —— 这样回测出来的是「这套规则在
    固定规模下能赚多少」，而不是「复利滚了二十年之后的奇迹」。要看复利，把
    资金口径换成 `unit`。
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
    # 名义敞口 = 权益。期货一手是 price × 乘数，不除乘数的话
    # 「不加杠杆」会变成 10 倍杠杆（RB 乘数 10）
    return float(max(equity, 0.0) / (price * mult))


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig | None = None, label: str = "") -> Result:
    """按 signal 列执行回测。df 需含 open/close 与 signal（取值 -1/0/1）。"""
    cfg = cfg or BacktestConfig()
    data = df.dropna(subset=["open", "close"]).copy()
    # 价格必须为正：期货无成交日会返回 0 价，参与运算会导致除零、净值变 inf
    data = data[(data["open"] > 0) & (data["close"] > 0)]
    if "signal" not in data:
        raise ValueError("缺少 signal 列")

    open_ = data["open"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    sig = data["signal"].fillna(0).astype(int).to_numpy()
    if cfg.long_only:
        sig = np.maximum(sig, 0)
    idx = data.index
    n = len(data)
    fee, slip = cfg.fee_rate, cfg.slippage
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
    lots = 0.0                   # 正=多，负=空，0=空仓（unit 口径是份额）
    pos = 0                      # 1=多，-1=空，0=平
    pos_state = np.zeros(n, dtype=int)
    equity = np.zeros(n, dtype=float)
    trades: list[Trade] = []
    entry_price = entry_date = None
    entry_bar = 0
    entry_dir = 1

    def close_position(i: int, price: float, held: float) -> None:
        nonlocal entry_price, entry_date
        if entry_price is None:
            return
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
            )
        )
        entry_price = entry_date = None

    for i in range(n):
        want = target[i]
        if want != pos:
            # ---- 先平掉现有仓位 ----
            if pos != 0:
                price = open_[i] * (1 - slip) if pos == 1 else open_[i] * (1 + slip)
                # 平仓：浮动盈亏落地，再扣平仓手续费（按成交名义额）
                cash += lots * mult * (price - entry_price)
                cash -= abs(lots) * mult * price * fee
                close_position(i, price, lots)
                lots = 0.0
                pos = 0

            # ---- 再按需开新仓 ----
            if want != 0:
                price = open_[i] * (1 + slip) if want > 0 else open_[i] * (1 - slip)
                a = float(atr_arr[i]) if atr_arr is not None else 0.0
                q = size_lots(cfg, price, cash, a if a == a else 0.0)
                if q > 0:
                    cash -= q * mult * price * fee        # 开仓手续费
                    lots = q if want > 0 else -q
                    entry_price, entry_date, entry_bar, entry_dir = price, idx[i], i, want
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
            )
        )

    eq = pd.Series(equity, index=idx, name="equity")
    ps = pd.Series(pos_state, index=idx, name="position")
    m = _metrics(eq, ps, trades, cfg.bars_per_year)
    m["final_equity"] = float(eq.iloc[-1])
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
    price = open_[0] * (1 + cfg.slippage)
    mult = float(cfg.multiplier or 1.0)
    # 基准要和策略用同一套仓位口径，否则「策略 vs 买入持有」是在比杠杆，
    # 不是在比策略。期货上尤其明显：策略 10 倍杠杆、基准全额，比出来毫无意义
    q = size_lots(cfg, price, cfg.initial_cash, 0.0)
    cash = cfg.initial_cash - q * mult * price * cfg.fee_rate
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
    )
    m = _metrics(eq, ps, [trade], cfg.bars_per_year)
    m["final_equity"] = float(eq.iloc[-1])
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
    for r in results:
        eq = _to_daily(r.equity)
        if len(eq) < 2:
            continue
        curves.append(eq / float(eq.iloc[0]))
        poss.append(_to_daily(r.position))
        trades += list(r.trades)

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
    m = _metrics(eq, pos, trades, bpy)
    m["final_equity"] = float(eq.iloc[-1])
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


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def yearly_returns(equity: pd.Series) -> pd.Series:
    """按自然年计算收益率。"""
    yr = equity.resample("YE").last()
    prev = yr.shift(1)
    prev.iloc[0] = equity.iloc[0]
    return (yr / prev - 1.0).rename("return")


def stage_table(equity: pd.Series, freq: str = "YE") -> pd.DataFrame:
    """按自然周期切段：段收益、段内最大回撤、段内波动、根数。

    **段收益用「上段末 → 本段末」算**，不是「段内第一根 → 段末」。第一根只是
    这个月的第一天，拿它当基准会把「上月末到本月初」那段跳空吞掉，十二个月的
    收益连乘起来对不上全年 —— 而这两个数字都会摆在页面上，对不上就是 bug。

    段内最大回撤的峰值在段首重置：它回答的是「这个月里最多回撤了多少」，
    不是「从历史最高点回撤了多少」。后者叫全局回撤，看资金曲线就够了。
    """
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 2:
        return pd.DataFrame(columns=["period", "ret", "max_dd", "vol", "bars"])

    base = float(eq.iloc[0])
    rows = []
    for key, seg in eq.groupby(pd.Grouper(freq=freq)):
        seg = seg.dropna()
        if seg.empty:
            continue
        last = float(seg.iloc[-1])
        ret = last / base - 1.0 if base else 0.0
        dd = float((seg / seg.cummax() - 1.0).min())
        # 段内波动按根数年化，和 `_metrics` 的口径一致（否则月表里的数字
        # 和绩效页的「年化波动」对不上，看的人会以为有一处算错了）
        r = seg.pct_change().dropna()
        vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
        rows.append({"period": key, "ret": ret, "max_dd": dd, "vol": vol, "bars": len(seg)})
        base = last
    return pd.DataFrame(rows)


def month_matrix(equity: pd.Series) -> pd.DataFrame:
    """月度收益矩阵：行=年，列=1..12，值=当月收益（缺失为 NaN）。

    年度和月度分开算（都走 `stage_table`），不用「十二个月连乘」倒推年度 ——
    首尾不满整年的那两段会让连乘结果和真实年度收益差出几个点。
    """
    st = stage_table(equity, "ME")
    if st.empty:
        return pd.DataFrame()
    df = st.assign(year=st["period"].dt.year, month=st["period"].dt.month)
    return df.pivot_table(index="year", columns="month", values="ret", aggfunc="last")

