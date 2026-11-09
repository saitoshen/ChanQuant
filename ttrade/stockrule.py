"""A 股「按条件选股」的组合回测：每天筛一遍全市场，够条件的按等权买进去。

## 为什么不用 `run_backtest`

平台的引擎是**逐品种**的：给定一只票的信号序列，算它自己的净值。而这类
策略的仓位是**动态成员**——今天符合条件的是这 30 只，明天可能换成另外 25 只。
把它拆成「每只票各跑一遍再平均」是不对的：那样 5000 只票每只权重 1/5000，
一只一年只触发一次的策略贡献几乎是 0。

所以这里自己算账户级净值：一个资金池、最多同时持 `max_pos` 只、等权、按
次日开盘成交、按换手吃成本。和实盘的差距只剩「集合竞价能不能成交」。

## 时点口径

- 第 T 日**收盘**筛（用的都是 T 日已定型的数：收盘价、成交量、成交额、PE）
- 第 **T+1 日开盘**买 / 卖 —— 与平台「T 收盘出信号、T+1 开盘成交」一致
- T+1 开盘一字涨停买不进（`open >= 涨停价`）⇒ 跳过，不假装成交
- T+1 开盘一字跌停卖不出 ⇒ 顺延到下一个能卖的交易日

## 成本

按平台 A 股口径（`replay.MARKET_COST["cn_stock"]`）：手续费 8bp、滑点 10bp。
印花税单列（卖出 5bp），因为它是**单边**的，摊到两边会把买入成本算高。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttrade.strategies import _limit_pct, _round2

# 交易日最少多少根才出结论（少于这个数，年化没意义）
MIN_DAYS = 60

# 🔴 Tushare `daily.amount` 单位是**千元**，配置里的 `amt_min` 按**万元**给。
# 不换算就是差 10 倍：「成交额 ≥ 5000 万」会变成「≥ 500 万」也放进来，而且
# 不报错、只让选出来的票变多。所以换算只留一个出口 `amount_wan()`。
AMOUNT_KYUAN_PER_WAN = 10.0


@dataclass
class ScreenConfig:
    """选股条件 + 组合参数。默认值就是用户提的那一套（PE 上限 25）。"""

    start: str = ""
    end: str = ""
    # ---- 选股条件 ----
    max_price: float = 80.0
    gain_lo: float = 0.0
    gain_hi: float = 9.8
    vol_mult: float = 3.0
    amt_min: float = 5000.0          # 万元
    pe_min: float = 0.0              # 负 PE 是亏损，不算「低估值」
    pe_max: float = 25.0
    # ---- 组合 ----
    hold_days: int = 5
    max_pos: int = 10
    capital: float = 1_000_000.0
    fee_rate: float = 0.0008         # 单边（佣金 + 过户费摊薄）
    slippage: float = 0.0010         # 单边
    stamp_duty: float = 0.0005       # 印花税，只在卖出收
    # 同一天候选多于空位时按什么排：amount=成交额（默认）/ gain=涨幅 / pe=PE
    rank_by: str = "amount"

    def clone(self, **kw) -> "ScreenConfig":
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(kw)
        return ScreenConfig(**d)


# ------------------------------------------------------------------ 取数

PANEL_SQL = """
SELECT b.dt, b.ts_code, b.open, b.close, b.volume, b.amount, b.pct_chg,
       s.pe_ttm, a.adj_factor
FROM stock_bars_ts b
LEFT JOIN stock_basic_ts s ON s.ts_code = b.ts_code AND s.dt = b.dt
LEFT JOIN stock_adj      a ON a.ts_code = b.ts_code AND a.dt = b.dt
WHERE b.dt BETWEEN ? AND ?
"""


def load_panels(start: str, end: str, db: str = ""):
    """把日线 + PE + 复权因子读成「日期 × 股票」的宽表（float32，省内存）。

    6000 万格级别的矩阵，`float64` 会吃掉几个 G —— 这份数据只做阈值比较和
    净值累加，`float32` 的 7 位有效数字足够。

    🔴 **`open`/`close` 是原始价（不复权），另给一份 `open_adj`/`close_adj`**：
    - 筛选用的都是原始价 —— 「股价 < 80」「涨停价 = 昨收 × (1+幅度)」都是
      用户在大屏幕上看到的那个数，改用复权价会整体错位；
    - 但**算收益必须用复权价**：除权除息日原始价会凭空掉一大块，10 送 10
      直接腰斩，用原始价算就是把分红送股当成亏损。持有 5 天的策略里，一笔
      撞上除权日就能贡献 -5% ~ -50% 的假亏。
    """
    from ttrade import stockstore as SS

    if db:
        import duckdb
        con = duckdb.connect(db, read_only=True)
        try:
            raw = con.execute(PANEL_SQL, [start, end]).df()
        finally:
            con.close()
    else:
        with SS.connect(read_only=True) as con:
            raw = con.execute(PANEL_SQL, [start, end]).df()

    if raw.empty:
        return None, None

    # 🔴 640 万行的 `ts_code` 如果是 Python 字符串对象，光这一列就要几百 MB，
    # 加上后面七次 `pivot` 的中间结果很容易把内存顶爆。先降成 category，
    # pivot 完再把列名还原成 str（否则 `reindex` / 字典查找会在
    # CategoricalIndex 上出怪事）。
    raw["ts_code"] = raw["ts_code"].astype("category")
    raw["dt"] = pd.to_datetime(raw["dt"])
    for c in ("open", "close", "volume", "amount", "pct_chg", "pe_ttm", "adj_factor"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce").astype("float32")
    for c in ("open", "close"):
        raw.loc[raw[c] <= 0, c] = np.nan      # 停牌返回 0 价，不能当成真价格
    # 复权因子缺失（极少数新股 / 停牌段）按 1.0 处理，等价于「不复权」，
    # 至少不会因为 NaN 把整笔交易的收益算丢
    raw["adj_factor"] = raw["adj_factor"].fillna(1.0)

    def wide(col):
        w = raw.pivot(index="dt", columns="ts_code", values=col).sort_index()
        w.columns = w.columns.astype(str)
        return w

    panels = {c: wide(c) for c in
              ("open", "close", "volume", "amount", "pct_chg", "pe_ttm")}
    adj = wide("adj_factor").ffill().fillna(1.0)
    panels["adj"] = adj
    # 后复权价：只用于算收益，不参与任何阈值比较
    panels["open_adj"] = panels["open"] * adj
    panels["close_adj"] = panels["close"] * adj

    # 🔴 板块按**代号**判，不查 `stock_meta`：那张表只有在市股票，区间内退市的
    # 197 只查不到 → 被「非北交所」白名单静默剔掉。顺带也不需要 JOIN 了。
    from ttrade.stockaux import BOARD_CODE
    panels["board"] = pd.Series(
        {c: BOARD_CODE.get(SS.guess_board(c), np.nan) for c in panels["close"].columns}
    ).reindex(panels["close"].columns)
    # 板块认不出来的会被「非北交所」的白名单（0/1/2）静默剔掉，必须报出来。
    n_unknown = int(panels["board"].isna().sum())
    if n_unknown:
        print(f"⚠️ 板块未知 {n_unknown} 只（B 股等，不在主板/创业板/科创板/北交所"
              f"名单里，会被「非北交所」这条剔掉）")
    # 🔴 这里**不换算**成交额：`panels["amount"]` 保持 Tushare 原始单位（千元），
    # 谁要用万元口径一律走 `amount_wan()`。把换算藏在这里的话，`screen()` 和
    # 条件漏斗各读一遍就会漏掉，而且不会有任何报错。
    codes = panels["close"].columns
    # 🔴 ST 过滤是这套条件的第一条，但它的数据源（曾用名表）是**另外下载**的。
    # 表缺失 / 是空的时候 `st_matrix` 会安静地返回全 False ⇒ 第 1 条白写，
    # 结果只会更好看。这种情况必须直接报错，不许降级。
    try:
        nc = SS.load_namechange()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "读不到 ST 曾用名表（stock_namechange）——先跑 `download_stock_ts.py --st`；"
            "没有它「非 ST」这条会静默失效"
        ) from exc
    if nc is None or nc.empty:
        raise RuntimeError(
            "ST 曾用名表是空的（stock_namechange 0 行）——`--st` 没跑成功，"
            "不能继续：否则「非 ST」这条条件等于没写"
        )
    panels["is_st"] = st_matrix(panels["close"].index, codes)
    return panels, codes


def st_matrix(index: pd.DatetimeIndex, codes) -> pd.DataFrame:
    """`日期 × 股票` 的 ST 布尔矩阵（含 `*ST`），按**当时的**曾用名。

    逐区间标而不是逐日查表：一只票五年也就改几次名，而这里有 5000 多列。
    """
    from ttrade import stockstore as SS

    out = pd.DataFrame(False, index=index, columns=codes)
    iv = SS.st_intervals()
    if iv is None or iv.empty:
        return out
    pos = {c: i for i, c in enumerate(codes)}
    for code, grp in iv.groupby("ts_code", sort=False):
        j = pos.get(code)
        if j is None:
            continue
        for start, end in zip(grp["start"], grp["end"]):
            mask = out.index >= start
            if pd.notna(end):
                mask &= out.index <= end
            out.iloc[mask, j] = True
    return out


# ------------------------------------------------------------------ 选股

def amount_wan(panels: dict) -> pd.DataFrame:
    """成交额换成**万元**（配置里 `amt_min` 的口径）。

    唯一出口，`screen()` 和条件漏斗都走它 —— 两边各自 `/ 10` 迟早会漏一处。
    """
    return panels["amount"] / AMOUNT_KYUAN_PER_WAN


def screen(panels: dict, cfg: ScreenConfig) -> pd.DataFrame:
    """逐日筛出符合条件的股票（布尔宽表）。

    每一条都对应配置里的一个参数，顺序与策略说明一致。
    """
    close, vol = panels["close"], panels["volume"]
    amt, pct, pe = amount_wan(panels), panels["pct_chg"], panels["pe_ttm"]
    board = panels["board"].astype(float)
    is_st = panels["is_st"].astype(bool)

    prev_vol = vol.shift(1)
    prev_close = close.shift(1)
    prev2_close = close.shift(2)
    # 昨天的涨停价：前天收盘 × (1 + 昨天的幅度)。幅度用**昨天**的 ST 状态，
    # 因为 ST 是当天起效 —— 用今天的会把「今天才戴帽」那只昨天的涨停算错。
    # `board` 是一维（每只票一个板块），和 (日期 × 股票) 的布尔阵广播即可，
    # 不用 `np.repeat` 铺一张同尺寸的矩阵
    lim_prev = _limit_pct(is_st.shift(1, fill_value=False).to_numpy(),
                          board.to_numpy())
    yday_up = _round2(prev2_close.to_numpy() * (1.0 + lim_prev))
    yday_limit_up = prev_close.to_numpy() >= yday_up - 1e-6

    not_bse = board.isin([0.0, 1.0, 2.0]).to_numpy()
    with np.errstate(invalid="ignore"):
        ok = (
            (~is_st.to_numpy())
            & not_bse
            & (~yday_limit_up)
            & (close.to_numpy() < float(cfg.max_price))
            & (pct.to_numpy() >= float(cfg.gain_lo))
            & (pct.to_numpy() <= float(cfg.gain_hi))
            & (vol.to_numpy() >= float(cfg.vol_mult) * prev_vol.to_numpy())
            & (pe.to_numpy() > float(cfg.pe_min))
            & (pe.to_numpy() <= float(cfg.pe_max))
            & (amt.to_numpy() >= float(cfg.amt_min))
        )
    return pd.DataFrame(ok, index=close.index, columns=close.columns)


# ------------------------------------------------------------------ 组合回测

@dataclass
class Position:
    code: str
    lots: float
    entry_i: int
    entry_date: pd.Timestamp
    entry_px: float          # **原始价**（含滑点），用来记账面金额和手数
    exit_i: int
    entry_adj: float = np.nan  # 建仓那根的后复权价，收益一律按它折算
    # 开仓时为了「同一天多只」记下排名分，回测里没用，交易表用得上
    score: float = 0.0


@dataclass
class ScreenResult:
    equity: pd.Series = field(default_factory=pd.Series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    picks: pd.DataFrame = field(default_factory=pd.DataFrame)


def run(panels: dict, cfg: ScreenConfig) -> ScreenResult:
    """账户级回测：一个资金池，每天按筛出来的名单等权持有。

    🔴 **买多少股用原始价、赚多少用复权价**：份额必须是真实手数（原始价算），
    但两个时点之间的涨跌必须走复权口径 —— 除权日原始价会凭空掉一块，用原始
    价算等于把分红送股当成亏损。
    """
    close, open_, amt = panels["close"], panels["open"], amount_wan(panels)
    o_adj, c_adj = panels["open_adj"], panels["close_adj"]
    board = panels["board"].astype(float)
    is_st = panels["is_st"].astype(bool)

    dates = close.index
    n = len(dates)
    if n < MIN_DAYS:
        raise ValueError(f"样本只有 {n} 个交易日，不够回测（至少 {MIN_DAYS}）")

    ok = screen(panels, cfg)
    codes = list(close.columns)
    col_of = {c: i for i, c in enumerate(codes)}

    o = open_.to_numpy()
    cl = close.to_numpy()
    oa = o_adj.to_numpy()
    ca = c_adj.to_numpy()
    amt_a = amt.to_numpy()
    ok_a = ok.to_numpy()

    # 每天的涨停 / 跌停价（用于「买不进 / 卖不出」）。float32 就够 ——
    # 只做「开盘价有没有顶到涨停」的比较，省下两张 float64 的全尺寸矩阵。
    # 🔴 这里必须是**原始价**：涨停价是交易所按前收（原始）算的。
    lim_today = _limit_pct(is_st.to_numpy(), board.to_numpy())
    prev_cl = np.vstack([np.full((1, len(codes)), np.nan), cl[:-1]])
    up_px = _round2(prev_cl * (1.0 + lim_today)).astype("float32")
    dn_px = _round2(prev_cl * (1.0 - lim_today)).astype("float32")
    eps = 1e-6

    cash = float(cfg.capital)
    hold: dict[str, Position] = {}
    equity = np.full(n, np.nan)
    n_pos = np.zeros(n, dtype=int)
    trades = []
    fee_r, slip, stamp = float(cfg.fee_rate), float(cfg.slippage), float(cfg.stamp_duty)

    def mv(p: "Position", i: int) -> float:
        """持仓在 i 日收盘的市值（复权口径）。"""
        j = col_of[p.code]
        a = ca[i, j]
        if not np.isfinite(a):
            return p.lots * p.entry_px          # 停牌，按成本记
        return p.lots * p.entry_px * (a / p.entry_adj)

    for i in range(n):
        # ---- ① 到期卖出：开盘价成交 ----
        for code in [c for c, p in hold.items() if p.exit_i <= i]:
            p = hold[code]
            j = col_of[code]
            px = o[i, j]
            if not np.isfinite(px):
                p.exit_i = i + 1          # 停牌，顺延
                continue
            if np.isfinite(dn_px[i, j]) and px <= dn_px[i, j] + eps:
                p.exit_i = i + 1          # 一字跌停卖不出
                continue
            # 复权口径下的「等效出场价」：把分红送股也算进收益
            fill = p.entry_px * (oa[i, j] * (1.0 - slip) / p.entry_adj)
            gross = p.lots * (fill - p.entry_px)
            fee_out = p.lots * fill * fee_r + p.lots * fill * stamp
            fee_in = p.lots * p.entry_px * fee_r
            cash += p.lots * fill - fee_out
            trades.append({
                "code": code, "entry_date": p.entry_date, "exit_date": dates[i],
                "lots": p.lots, "entry_px": p.entry_px, "exit_px": fill,
                "pnl": gross, "ret": (fill - p.entry_px) / p.entry_px,
                "days": i - p.entry_i, "cost": fee_in + fee_out,
                "pnl_net": gross - fee_in - fee_out,
            })
            del hold[code]

        # ---- ② 新买入：用**昨天收盘**筛出来的名单，今天开盘成交 ----
        if i > 0 and len(hold) < int(cfg.max_pos):
            cand = np.flatnonzero(ok_a[i - 1])
            if len(cand):
                # 同一根有多个候选时，按成交额从大到小排（流动性优先，结果也可复现）
                order = cand[np.argsort(-np.nan_to_num(amt_a[i - 1, cand]))]
                eq_now = cash + sum(mv(p, i - 1) for p in hold.values())
                alloc = eq_now / max(int(cfg.max_pos), 1)
                for j in order:
                    if len(hold) >= int(cfg.max_pos):
                        break
                    code = codes[j]
                    if code in hold:
                        continue
                    px = o[i, j]
                    if not np.isfinite(px):
                        continue                      # 停牌
                    if np.isfinite(up_px[i, j]) and px >= up_px[i, j] - eps:
                        continue                      # 一字涨停，买不进
                    lots = np.floor(alloc / (px * (1.0 + slip) * 100.0)) * 100.0
                    if lots <= 0:
                        continue
                    fill = px * (1.0 + slip)
                    cost = lots * fill * fee_r
                    if lots * fill + cost > cash:
                        lots = np.floor((cash * 0.995) / (fill * (1 + fee_r) * 100)) * 100
                        if lots <= 0:
                            continue
                        cost = lots * fill * fee_r
                    cash -= lots * fill + cost
                    hold[code] = Position(
                        code=code, lots=lots, entry_i=i, entry_date=dates[i],
                        entry_px=fill, exit_i=i + max(int(cfg.hold_days), 1),
                        entry_adj=float(oa[i, j]) if np.isfinite(oa[i, j]) else float(o[i, j]),
                    )

        # ---- ③ 收盘估值 ----
        val = cash + sum(mv(p, i) for p in hold.values())
        equity[i] = val
        n_pos[i] = len(hold)

    eq = pd.Series(equity, index=dates, name="equity").ffill()
    tdf = pd.DataFrame(trades)
    daily = pd.DataFrame({"equity": eq, "positions": n_pos}, index=dates)
    m = metrics(eq, daily, tdf, cfg, panels)
    # 期末还拿着的仓位：净值已经按最后收盘价 mark-to-market 记账（所以收益是
    # 全的），但**没进 `trades`** —— 逐笔统计和成本合计会把这几笔漏掉，必须
    # 显式告诉读数的人，否则「胜率 60%」是少算了最后一批的下场。
    m["n_open_at_end"] = len(hold)
    m["open_at_end_cost"] = float(sum(
        p.lots * p.entry_px * cfg.fee_rate for p in hold.values()))
    return ScreenResult(equity=eq, trades=tdf, metrics=m, daily=daily, picks=ok)


# ------------------------------------------------------------------ 指标

def metrics(eq: pd.Series, daily: pd.DataFrame, trades: pd.DataFrame,
            cfg: ScreenConfig, panels: dict) -> dict:
    ret = eq.pct_change().fillna(0.0)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    days = len(eq)
    years = days / 244.0
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and total > -1 else float("nan")
    vol = float(ret.std(ddof=1) * np.sqrt(244)) if days > 2 else float("nan")
    sharpe = float(ret.mean() * 244 / vol) if vol and np.isfinite(vol) and vol > 0 else float("nan")
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())
    calmar = float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else float("nan")

    # 等权全市场基准：每日所有可交易股票涨幅的均值，等权累乘
    pct = panels["pct_chg"]
    bench_r = pct.mean(axis=1, skipna=True).reindex(eq.index).fillna(0.0) / 100.0
    bench = (1.0 + bench_r).cumprod()
    bench_total = float(bench.iloc[-1] - 1.0)

    out = {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        "days": days, "years": years,
        "final_equity": float(eq.iloc[-1]),
        "total_return": total, "cagr": cagr, "ann_vol": vol,
        "sharpe": sharpe, "max_dd": mdd, "calmar": calmar,
        "bench_total": bench_total,
        "excess": total - bench_total,
        "avg_positions": float(daily["positions"].mean()),
        "max_positions": int(daily["positions"].max()),
        "utilization": float((daily["positions"] / max(cfg.max_pos, 1)).mean()),
        "n_days_in_market": int((daily["positions"] > 0).sum()),
    }
    if len(trades):
        wins = trades[trades["pnl_net"] > 0]
        losses = trades[trades["pnl_net"] <= 0]
        out.update({
            "n_trades": int(len(trades)),
            "win_rate": float(len(wins) / len(trades)),
            "avg_win": float(wins["ret"].mean()) if len(wins) else 0.0,
            "avg_loss": float(losses["ret"].mean()) if len(losses) else 0.0,
            "profit_factor": (float(wins["pnl_net"].sum() / abs(losses["pnl_net"].sum()))
                              if len(losses) and losses["pnl_net"].sum() != 0 else float("nan")),
            "avg_days": float(trades["days"].mean()),
            "total_cost": float(trades["cost"].sum()),
            "avg_pnl": float(trades["pnl_net"].mean()),
        })
    else:
        out.update({"n_trades": 0, "win_rate": float("nan")})
    return out
