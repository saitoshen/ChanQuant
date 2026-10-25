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


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    fee_rate: float = 0.0005      # 单边手续费 5bp
    slippage: float = 0.0005      # 单边滑点 5bp
    long_only: bool = False       # True 时把 -1 裁成 0（股票市场默认）

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


def _metrics(equity: pd.Series, position: pd.Series, trades: list[Trade]) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {k: 0.0 for k in _METRIC_KEYS}

    ret = equity.pct_change().fillna(0.0)
    total = equity.iloc[-1] / equity.iloc[0] - 1.0

    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS) if ret.std(ddof=1) > 0 else 0.0
    sharpe = (ret.mean() * TRADING_DAYS) / vol if vol > 0 else 0.0

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
    m = _metrics(eq, ps, trades)
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
    m = _metrics(eq, ps, [trade])
    m["final_equity"] = float(eq.iloc[-1])
    return Result(equity=eq, position=ps, trades=[trade], metrics=m, label="买入持有")


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def yearly_returns(equity: pd.Series) -> pd.Series:
    """按自然年计算收益率。"""
    yr = equity.resample("YE").last()
    prev = yr.shift(1)
    prev.iloc[0] = equity.iloc[0]
    return (yr / prev - 1.0).rename("return")
