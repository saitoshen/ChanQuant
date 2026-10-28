"""向量化友好的逐日回测引擎（支持多空双向）。

规则：
1. 策略在第 T 日收盘后给出目标仓位 signal[T] ∈ {-1, 0, 1}；
2. 实际成交发生在第 T+1 日开盘价（杜绝未来函数）；
3. 买卖均计手续费与滑点；
4. 名义敞口 = 账户权益（不加杠杆），多头买入全额、空头卖出等额名义；
5. 空头遇到价格暴涨导致权益归零时强制平仓（爆仓），后续不再交易；
6. 空仓时资金闲置（不计利息）。
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

    @property
    def label(self) -> str:
        return "只做多" if self.long_only else "多空双向"


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
    blowup: bool = False          # 是否因爆仓强平


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
    "n_trades", "n_long", "n_short", "n_blowup",
    "win_rate", "avg_trade", "profit_factor",
    "avg_bars", "best_trade", "worst_trade",
)


def _metrics(equity: pd.Series, position: pd.Series, trades: list[Trade],
             bars_per_year: float = float(TRADING_DAYS)) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {k: 0.0 for k in _METRIC_KEYS}

    ret = equity.pct_change().fillna(0.0)
    total = equity.iloc[-1] / equity.iloc[0] - 1.0

    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    bpy = float(bars_per_year or TRADING_DAYS)
    vol = ret.std(ddof=1) * np.sqrt(bpy) if ret.std(ddof=1) > 0 else 0.0
    sharpe = (ret.mean() * bpy) / vol if vol > 0 else 0.0

    dd_series = equity / equity.cummax() - 1.0
    max_dd = float(dd_series.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

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
        "n_blowup": sum(1 for t in trades if t.blowup),
        "win_rate": float(win_rate),
        "avg_trade": avg_trade,
        "profit_factor": profit_factor,
        "avg_bars": avg_bars,
        "best_trade": best,
        "worst_trade": worst,
    }


def _trade_ret(direction: int, entry: float, exit_: float) -> float:
    return float(exit_ / entry - 1.0) if direction == 1 else float(1.0 - exit_ / entry)


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

    # 目标仓位：T 日收盘的信号，T+1 开盘执行
    target = np.roll(sig, 1)
    target[0] = 0

    cash = cfg.initial_cash
    shares = 0.0
    pos = 0                      # 1=多，-1=空，0=平
    dead = False                 # 爆仓后停止交易
    pos_state = np.zeros(n, dtype=int)
    equity = np.zeros(n, dtype=float)
    trades: list[Trade] = []
    entry_price = entry_date = None
    entry_bar = 0
    entry_dir = 1

    def close_position(i: int, price: float, blowup: bool = False) -> None:
        nonlocal cash, shares, entry_price, entry_date
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
                blowup=blowup,
            )
        )
        entry_price = entry_date = None

    for i in range(n):
        if dead:
            equity[i] = 0.0
            continue

        want = target[i]
        if want != pos:
            # ---- 先平掉现有仓位 ----
            if pos == 1:
                price = open_[i] * (1 - slip)
                cash += shares * price * (1 - fee)
                close_position(i, price)
                shares = 0.0
            elif pos == -1:
                price = open_[i] * (1 + slip)      # 买回，滑点不利
                cash -= (-shares) * price * (1 + fee)
                close_position(i, price)
                shares = 0.0
            pos = 0

            # ---- 再按需开新仓（名义敞口 = 全部权益，不加杠杆）----
            if want == 1 and cash > 0:
                price = open_[i] * (1 + slip)
                qty = cash / (price * (1 + fee))
                shares, cash = qty, 0.0
                entry_price, entry_date, entry_bar, entry_dir = price, idx[i], i, 1
                pos = 1
            elif want == -1 and cash > 0:
                price = open_[i] * (1 - slip)
                qty = cash / price
                cash += qty * price * (1 - fee)     # 卖出所得 + 原有现金
                shares = -qty
                entry_price, entry_date, entry_bar, entry_dir = price, idx[i], i, -1
                pos = -1

        eq = cash + shares * close[i]

        # ---- 爆仓：权益归零则强平 ----
        if eq <= 0 and pos != 0:
            close_position(i, close[i], blowup=True)
            cash, shares, pos = 0.0, 0.0, 0
            dead = True
            eq = 0.0

        pos_state[i] = pos
        equity[i] = eq

    # 未平仓的最后一笔
    if shares != 0 and entry_price is not None:
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
    shares = (cfg.initial_cash * (1 - cfg.fee_rate)) / price
    equity = shares * close
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

    对齐用 `ffill(limit_area="inside")`：净值在两根 bar 之间是常数，往前填是
    精确的；但只填「自己区间内部」的空档，策略还没上线、或者数据已经结束的
    日子保持空 —— 往后一直填的话，一条 2020 年就停掉的策略会以最后那点净值
    一直参与组合，把后面的曲线压平。

    **先各自归一化到 1 再平均**，不是直接平均净值：一个品种 3000 点、另一个
    80 块，初始资金也可能不同，直接平均出来的数字没有任何含义。

    平均的是**净值水平**，所以这是「初始等权、之后不再平衡」的组合：表现好的
    那条占的份额会慢慢变大，和持有一篮子标的的感觉一致。若按「每日再平衡」
    算（把每天的收益取均值再连乘），调仓成本没计进去，收益会略好看一点 ——
    两种口径都能自圆其说，这里选更保守的那种，并在标题里写明「等权」。

    某条策略还没上线的日子**不参与**当天的均值 —— 用 1.0 去填等于凭空多出
    一份现金，会把组合净值稀释成「一半空仓」，看起来像策略失效。

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
    mat = pd.concat([c.reindex(grid).ffill(limit_area="inside") for c in curves], axis=1)
    eq = mat.mean(axis=1, skipna=True).dropna() * float(initial_cash)

    pmat = pd.concat([p.reindex(grid).ffill(limit_area="inside") for p in poss], axis=1)
    pos = pmat.mean(axis=1, skipna=True).reindex(eq.index).fillna(0.0)

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

