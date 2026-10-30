"""截面策略：把「谁强谁弱」当成信号，而不是「这一根该不该买」。

时序策略问的是「这个品种现在该不该持有」，截面策略问的是「这几个品种里谁最强、
谁最弱」。两个问题不一样，回测也不能共用一个引擎 —— 时序引擎一次只持有一个
品种的仓位，截面引擎每根 bar 要同时持有一篮子，多空两侧各一半。

两种玩法：

- **截面对冲**（`mode="hedge"`）：按截面分数排序，多最强的 top X%、空最弱的
  bottom X%，等权，定期调仓。净敞口约等于 0，赚的是「强弱差」。
- **截面选品种 + 时序交易**（`mode="select"`）：先用截面分数挑出 top X%，再对
  挑中的每个品种单独跑时序策略（规则或因子），由时序信号决定这一腿是多是空
  还是空仓。截面负责「买什么」，时序负责「什么时候」。

分数怎么来：每个因子先算**时序**原值（复用 `factors.raw`），再在每根 bar 上做
**横截面** Z-score —— 这就是和时序因子策略最本质的区别。时序 Z-score 是「跟
自己的历史比」，截面 Z-score 是「跟同期的其他品种比」，后者才消得掉「整个板块
一起涨」这种共同因子。

成本按「换手 × 单边成本」扣：换手是权重变化的绝对值之和，单边成本按品种查
（`symbols_info` + `costs`）。不按笔数扣是因为截面策略天天在调仓，按笔算会把
成本算成 0。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttrade import costs, factors, symbols_info
from ttrade import factor_strategy as fstrat
from ttrade.backtest import BacktestConfig
from ttrade.strategies import STRATEGIES, run_strategy

MIN_CROSS = 3          # 横截面上至少几个品种有值，才算得出一组强弱

# 每折最多喂多少行给模型。截面是「品种 × 时间」堆起来的长表，6 个品种 × 十年
# 5 分钟就是 100 万行，一折要十几秒；几十折下来是几十分钟。树模型不缺这几百万
# 行里的重复信息，按等间隔抽到 2 万行，一折 0.5 秒左右，排序结论几乎不变。
ML_MAX_TRAIN_ROWS = 20_000


@dataclass
class CrossConfig:
    """截面策略的配置。`mode` 决定后面哪一组字段有效，无效的保留着不删。"""

    mode: str = "hedge"                 # hedge / select
    rebalance: int = 5                  # 调仓间隔，单位是**根**（5m 上 48 根约一天）
    top_pct: float = 10.0               # 多头：选最强的百分之几
    bot_pct: float = 10.0               # 空头：选最弱的百分之几（可以和多头不同）
    long_only: bool = False
    weight: str = "equal"               # equal = 等权；score = 按分数加权

    # 截面因子。三种模式和时序那边一样：single / multi / ml
    factor_mode: str = "multi"
    factor_keys: list = field(default_factory=list)
    factor_reverse: dict = field(default_factory=dict)
    factor_cfg: dict = field(default_factory=dict)   # 窗口 / 模型 / 训练窗口 …

    # 选品种之后怎么交易（只对 select 生效）
    timing_family: str = "rule"         # rule / factor
    timing_rule_key: str = ""
    timing_rule_params: dict = field(default_factory=dict)
    timing_factor_mode: str = ""
    timing_factor_keys: list = field(default_factory=list)
    timing_factor_reverse: dict = field(default_factory=dict)
    timing_factor_cfg: dict = field(default_factory=dict)

    def n_legs(self, n_valid: int) -> int:
        """多头持几个品种。至少 1 个，最多不超过有效品种数的一半。"""
        return _legs(n_valid, self.top_pct)

    def n_short_legs(self, n_valid: int) -> int:
        """空头持几个品种。默认和多头一样，但可以单独调。"""
        return _legs(n_valid, self.bot_pct)

    def step(self) -> int:
        return max(1, int(self.rebalance or 1))


def _legs(n_valid: int, pct: float) -> int:
    k = int(round(n_valid * float(pct or 0) / 100.0))
    return max(1, min(k, max(1, n_valid // 2)))


@dataclass
class CrossResult:
    equity: pd.Series
    weights: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    label: str = ""
    legs: int = 0                       # 最近一次调仓持了几个品种
    n_rebalance: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.equity is not None and len(self.equity) > 1


# ------------------------------------------------------------------ 面板


def close_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """收盘价面板（index × 品种）。缺口前向填充：一个品种停牌不该让整行作废。"""
    cols = {k: df["close"] for k, df in frames.items() if df is not None and len(df)}
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index().ffill()


def open_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = {k: df["open"] for k, df in frames.items() if df is not None and len(df)}
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index().ffill()


def _row_z(row: pd.Series) -> pd.Series:
    """一根 bar 上的横截面 Z-score。样本不足就整行 NaN（不硬算）。"""
    v = row.dropna()
    if len(v) < MIN_CROSS:
        return pd.Series(np.nan, index=row.index)
    sd = float(v.std(ddof=0))
    if not sd or sd != sd:
        return pd.Series(0.0, index=row.index)
    return (row - float(v.mean())) / sd


def _cross_z(panel: pd.DataFrame) -> pd.DataFrame:
    """整块面板一次算完横截面 Z-score。

    和逐行调 `_row_z` 等价，但快两个数量级：`DataFrame.apply(axis=1)` 要走
    Python 层，6 个品种 × 8 个因子 × 3 万根就是 150 万次函数调用，一次回测
    要七十秒 —— 拖一下「调仓间隔」就卡住。pandas 的 `mean/std(axis=1)` 是
    按列向量化的，同样的数学，毫秒级。

    口径必须和 `_row_z` 完全一致：`ddof=0`、缺值跳过、有效样本不足 `MIN_CROSS`
    整行 NaN、横截面全一样（std=0）时记 0（没有强弱可分，不是 NaN）。
    """
    n = panel.notna().sum(axis=1)
    mean = panel.mean(axis=1)
    sd = panel.std(axis=1, ddof=0)
    z = panel.sub(mean, axis=0).div(sd.replace(0.0, np.nan), axis=0)
    z = z.mask(sd.eq(0.0), 0.0)
    return z.mask(n < MIN_CROSS)


def _factor_cfg(cfg: "CrossConfig"):
    """截面因子那一套参数（窗口 / 模型 / 训练窗口 …）转成 `FactorConfig`。"""
    d = dict(getattr(cfg, "factor_cfg", None) or {})
    return fstrat.FactorConfig(
        **{k: v for k, v in d.items()
           if k in fstrat.FactorConfig.__dataclass_fields__}
    )


def score_panel(frames: dict[str, pd.DataFrame], keys: list[str],
                reverse: dict | None = None, mode: str = "multi",
                fcfg=None, progress=None, cancel=None) -> pd.DataFrame:
    """截面合成分数。三种模式和时序那边对齐：单因子 / 多因子 / 机器学习。

    - `single`：取第一个因子，横截面 Z-score
    - `multi`：每个因子先横截面 Z-score，再按方向取号、等权平均
    - `ml`：把所有品种的样本堆成一张长表，滚动训练一个「相对强弱」模型

    因子在某个品种上没数据（美股没持仓量、新品种没有展期）就留 NaN，合成分数
    会自动跳过 —— 一个品种缺一个因子不该让整组排名作废。
    """
    keys = [k for k in (keys or []) if k in factors.FACTORS]
    if not keys:
        raise ValueError("没有选中任何可用因子")
    rev = set(k for k, v in (reverse or {}).items() if v)

    if mode == "ml":
        return ml_score_panel(frames, keys, fcfg, progress=progress, cancel=cancel)

    parts = []
    for key in keys:
        raw = {}
        for sym, df in frames.items():
            try:
                raw[sym] = factors.raw(df, key)
            except Exception:  # noqa: BLE001
                continue
        if len(raw) < MIN_CROSS:
            continue
        panel = pd.concat(raw, axis=1).sort_index().ffill()
        if not panel.notna().any().any():
            # 这个因子在这批品种上一个值都没有（日线库里没有展期 / 期权列）。
            # 留着它会**毒掉整个平均**：NaN + x 还是 NaN，十几个因子里只要有一个
            # 全空，合成分数就全空，界面上显示「截面分数全空」却查不出是谁
            continue
        z = _cross_z(panel)
        if key in rev:
            z = -z
        parts.append(z)
        if mode == "single":
            # 单因子就取**第一个真能算出数的**因子：勾了十几个因子时排在最前的
            # 往往是日线库里没数据的展期 / 期权因子，盲取 keys[0] 会直接报
            # 「这些因子在当前品种池上都没有足够数据」
            break
    if not parts:
        raise ValueError("这些因子在当前品种池上都没有足够数据")
    # 按时间取各因子的均值，**跳过缺值**：某个品种缺一个因子时不该让整行作废
    stacked = pd.concat(parts, axis=1, keys=list(range(len(parts))))
    out = stacked.T.groupby(level=1).mean().T
    return out.dropna(how="all")


def ml_score_panel(frames: dict[str, pd.DataFrame], keys: list[str], fcfg,
                   progress=None, cancel=None) -> pd.DataFrame:
    """横截面机器学习打分：把所有品种的样本堆成一张长表，滚动训练。

    和时序 ML 的区别在**标签**。这里预测的是「同期比别的品种强多少」：每根 bar
    上先把各品种的未来收益减去当期截面均值，模型学的才是排序。用绝对收益当
    标签的话，模型会先学会「这段行情整体在涨」—— 而截面策略做多空对冲，那个
    beta 一分钱也拿不到，学得再准也没用。

    训练样本按时间正序切折，训练集只取到 `b - horizon`：第 i 行的标签要等到
    i+h 才成立，混进训练集就是未来函数。
    """
    from ttrade import factor_strategy as fstrat
    from ttrade import mlmodels as ML
    from ttrade.factor_metrics import forward_return

    keys = [k for k in (keys or []) if k in factors.FACTORS]
    if len(keys) < 2:
        raise ValueError("机器学习至少需要 2 个因子")

    # 特征矩阵必须同构：某个品种缺这一列，整个模型就对不齐。取交集而不是各自
    # 补 0 —— 补 0 在标准化后的因子空间里是「中位数」，会让该品种凭空多出一个
    # 中性观测，排名被稀释
    avail = None
    for df in frames.values():
        got = set(factors.filter_available(df, keys))
        avail = got if avail is None else (avail & got)
    use = [k for k in keys if k in (avail or set())]
    if len(use) < 2:
        raise ValueError("这批品种上没有共同可用的 2 个因子（各自缺的数据不一样）")

    fcfg = fcfg or fstrat.FactorConfig()
    horizon = max(1, int(fcfg.horizon or 1))

    blocks = []
    for sym, df in frames.items():
        fm = factors.factor_frame(df, use, window=fcfg.window, method=fcfg.method)
        if fm.empty:
            continue
        d = fm.copy()
        d["__fwd"] = forward_return(df, horizon).reindex(fm.index)
        d["__sym"] = sym
        blocks.append(d)
    if not blocks:
        raise ValueError("这批品种算不出因子矩阵")
    long = pd.concat(blocks).sort_index()
    long.index = pd.MultiIndex.from_arrays([long.index, long["__sym"]])
    long = long.sort_index()
    # 截面去均值：留下的是「相对强弱」，共同的市场 beta 被减掉了
    long["__y"] = long["__fwd"] - long.groupby(level=0)["__fwd"].transform("mean")

    X = long[use].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    Y = long["__y"].to_numpy(dtype=float)
    ts = long.index.get_level_values(0).unique()
    pos = np.searchsorted(ts, long.index.get_level_values(0).to_numpy())
    n = len(ts)

    bpd = fstrat._bars_per_day(pd.DatetimeIndex(ts))
    train_bars = fstrat._to_bars(fcfg.train_window, bpd)
    step_bars = fstrat._to_bars(fcfg.step, bpd)
    min_bars = min(int(fcfg.min_train), train_bars)
    if n <= min_bars + horizon + 10:
        raise ValueError("样本太短，跑不了机器学习的滚动训练")

    bounds = []
    t = min_bars
    while t < n:
        bounds.append((max(0, t - train_bars), t, min(n, t + step_bars)))
        t += step_bars

    pred = np.full(len(long), np.nan)
    done = 0
    for a, b, c in bounds:
        if cancel and cancel():
            raise fstrat.Cancelled()
        tr = (pos >= a) & (pos < b - horizon) & np.isfinite(Y)
        rows = np.nonzero(tr)[0]
        if len(rows) < 60:
            continue
        stride = max(1, len(rows) // ML_MAX_TRAIN_ROWS)
        rows = rows[::stride]
        model = ML.build(fcfg.model)
        try:
            model.fit(X[rows], Y[rows])
        except Exception:  # noqa: BLE001
            continue
        want = (pos >= b) & (pos < c)
        try:
            pred[want] = model.predict(X[want])
        except Exception:  # noqa: BLE001
            continue
        done += 1
        if progress:
            progress(done, len(bounds), f"第 {done}/{len(bounds)} 折训练完成")

    if cancel and cancel():
        raise fstrat.Cancelled()

    ser = pd.Series(pred, index=long.index, name="pred").dropna()
    if ser.empty:
        raise ValueError("机器学习一折都没训出来（样本或因子不足）")
    panel = ser.unstack().sort_index()
    # 预测值本身已经同尺度，再做一次横截面标准化是为了和单/多因子那条路的
    # 「分数」口径一致（按分数加权时两者才可比）。排序不变
    return _cross_z(panel).dropna(how="all")


def cost_vector(frames: dict[str, pd.DataFrame], bt: BacktestConfig,
                profile: str | None = None) -> pd.Series:
    """每个品种「单边成本占名义额的比例」，用来乘换手。

    按手收的手续费要折成比例才能和权重相乘：`元/手 ÷ (价格 × 乘数)`。用中位价
    折，不用最新价 —— 最新价可能是个跳空，会把成本算歪。
    """
    out = {}
    for sym, df in frames.items():
        if df is None or not len(df):
            continue
        sym_code, _, ex = sym.rpartition("_")
        c = BacktestConfig()
        costs.apply(c, sym_code, ex, profile)
        price = float(pd.Series(df["close"]).median() or 0.0)
        # 乘数要查品种信息：`BacktestConfig()` 的默认值是 1，拿它去折算「元/手」
        # 会把螺纹（乘数 10）的手续费算小 10 倍
        mult = float(symbols_info.info(sym_code, ex).get("mult") or 1.0)
        if price <= 0:
            continue
        if c.fee_per_lot > 0:
            fee = c.fee_per_lot / (price * max(mult, 1e-9))
        else:
            fee = c.fee_rate
        slip = c.slip_amount(price) / price
        out[sym] = fee + slip
    return pd.Series(out, dtype=float)


# ------------------------------------------------------------------ 回测


def _metrics(equity: pd.Series, bpy: float) -> dict:
    eq = equity.dropna()
    if len(eq) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "max_dd": 0.0, "calmar": 0.0}
    ret = eq.pct_change().dropna()
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if eq.iloc[0] > 0 else 0.0
    years = max(len(eq) / max(bpy, 1.0), 1e-9)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    vol = float(ret.std(ddof=0) * np.sqrt(bpy)) if len(ret) else 0.0
    sharpe = float(ret.mean() * bpy / vol) if vol > 0 else 0.0
    peak = eq.cummax()
    dd = float((eq / peak - 1.0).min())
    calmar = float(cagr / abs(dd)) if dd < 0 else 0.0
    return {"total_return": total, "cagr": cagr, "ann_vol": vol,
            "sharpe": sharpe, "max_dd": dd, "calmar": calmar}


def run(frames: dict[str, pd.DataFrame], cfg: CrossConfig, bt: BacktestConfig,
        profile: str | None = None, label: str = "",
        progress=None, cancel=None) -> CrossResult:
    """跑一遍截面策略。出错写进 `CrossResult.error`，不抛异常。

    `progress` / `cancel` 只给机器学习模式用：walk-forward 要跑几十折，界面
    那边靠它显示进度、也靠它中途叫停。
    """
    if not frames or len(frames) < MIN_CROSS:
        return CrossResult(pd.Series(dtype=float), pd.DataFrame(),
                           error=f"截面策略至少要 {MIN_CROSS} 个品种，现在只有 {len(frames)} 个")
    try:
        score = score_panel(frames, list(cfg.factor_keys or []),
                            cfg.factor_reverse, mode=cfg.factor_mode or "multi",
                            fcfg=_factor_cfg(cfg), progress=progress, cancel=cancel)
    except fstrat.Cancelled:
        raise                     # 用户点的「取消」，不是回测失败
    except Exception as exc:  # noqa: BLE001
        return CrossResult(pd.Series(dtype=float), pd.DataFrame(), error=str(exc))
    if score.empty:
        return CrossResult(pd.Series(dtype=float), pd.DataFrame(),
                           error="截面分数全空：选中的因子在这批品种上没有数据")

    close = close_panel(frames)
    close = close.reindex(score.index).ffill()
    rets = close.pct_change().fillna(0.0)
    cvec = cost_vector(frames, bt, profile).reindex(score.columns).fillna(0.0)

    if cfg.mode == "select":
        weights = _select_weights(frames, score, cfg)
    else:
        weights = _hedge_weights(score, cfg)

    # 两个权重构造器都返回**稠密**矩阵（每一根 bar 都有值）：调仓日之间保持
    # 上一组的仓位，调仓日整体换一次。不要在这里再 ffill —— 那会把「主动平掉」
    # 和「还没调仓」混成同一个 NaN。
    weights = weights.reindex(columns=score.columns).fillna(0.0)
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    fee = (weights.diff().abs() * cvec).sum(axis=1).fillna(0.0)
    net = port_ret - fee

    equity = (1.0 + net).cumprod()
    equity.iloc[0] = 1.0
    bpy = bt.bars_per_year or 252.0
    n_reb = len(range(0, len(score), cfg.step()))
    return CrossResult(
        equity=equity, weights=weights, metrics=_metrics(equity, bpy),
        label=label, legs=int(weights.abs().gt(0).sum(axis=1).max() or 0),
        n_rebalance=n_reb,
    )


def to_result(out: "CrossResult", label: str = "", step: int = 1,
              bpy: float = 252.0):
    """把 `CrossResult` 包成 `backtest.Result`，好让现成的绩效页直接吃得下。

    截面策略没有「一笔一笔的成交」：仓位是按权重挂着的，用成交笔数去描述它
    只会得到「0 笔交易」。这里把**每次调仓区间**当成一笔 —— 区间收益为正算赢。
    这样胜率、盈亏比、最好 / 最差一笔都还有意义，表格也不会整列写 0。
    """
    from ttrade.backtest import Result, _metrics

    eq = out.equity.dropna()
    w = out.weights.reindex(eq.index).fillna(0.0)
    has_long = (w > 0).any(axis=1)
    has_short = (w < 0).any(axis=1)
    pos = pd.Series(0.0, index=eq.index)
    pos[has_long.to_numpy()] = 1.0
    pos[(has_short & ~has_long).to_numpy()] = -1.0

    m = dict(_metrics(eq, pos, [], bpy))
    ret = eq.pct_change().fillna(0.0)
    step = max(1, int(step or 1))
    seg: list[float] = []
    for i in range(0, len(ret), step):
        chunk = ret.iloc[i:i + step]
        if len(chunk):
            seg.append(float((1.0 + chunk).prod() - 1.0))
    n = len(seg)
    wins = [x for x in seg if x > 0]
    losses = [x for x in seg if x <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    m["n_trades"] = n
    m["n_long"] = n if bool(has_long.any()) else 0
    m["n_short"] = n if bool(has_short.any()) else 0
    m["win_rate"] = (len(wins) / n) if n else 0.0
    m["avg_trade"] = (sum(seg) / n) if n else 0.0
    m["profit_factor"] = ((gross_win / gross_loss) if gross_loss > 0
                          else (999.0 if gross_win > 0 else 0.0))
    m["avg_bars"] = float(step)
    m["best_trade"] = max(seg) if seg else 0.0
    m["worst_trade"] = min(seg) if seg else 0.0
    # 绩效表里「期末资金」这一行读的是 `final_equity`，`backtest._metrics` 不算
    # 它（那是 `run_backtest` 事后补的），这里得自己补上
    m["final_equity"] = float(eq.iloc[-1])
    return Result(equity=eq, position=pos, trades=[], metrics=m, label=label)


def _hedge_weights(score: pd.DataFrame, cfg: CrossConfig) -> pd.DataFrame:
    """多强空弱。多空各占一半名义，净敞口 ≈ 0。

    多空腿数可以不一样（`top_pct` / `bot_pct`）：多 10% 空 10% 是常见的对称
    写法，但「多最强的 10%、空最弱的 20%」也讲得通（下跌时弱者更弱）。两边
    各自按自己的腿数平分 0.5 的名义额，净敞口仍然是 0。
    """
    rows, top, bot, k_long, k_short, ok = _rank(score, cfg)
    inv_l = 1.0 / np.maximum(k_long, 1)
    inv_s = 1.0 / np.maximum(k_short, 1)
    if cfg.weight == "score":
        mat = score.to_numpy(dtype=float)[rows]
        pos = np.where(top, np.clip(np.nan_to_num(mat), 0.0, None), 0.0)
        s = pos.sum(axis=1)
        good = s > 0
        share = np.zeros_like(pos)
        share[good] = pos[good] / s[good, None]
        w = np.where(good[:, None], 0.5 * share, top * (0.5 * inv_l)[:, None])
    else:
        w = top * (0.5 * inv_l)[:, None]
    if not cfg.long_only:
        w = w - bot * (0.5 * inv_s)[:, None]
    return _place(score, rows, w, ok)


def _rank(score: pd.DataFrame, cfg: CrossConfig):
    """调仓日上「谁在前、谁在后」一次性排完。

    返回 `(行号, 多头掩码, 空头掩码, 多头几个, 空头几个, 有效行掩码)`。
    逐行 `nlargest` 在 5 分钟全历史上是 4 万多次 pandas 调用、七十秒；这里按轴
    排序一次就够 —— 拖一下「调仓间隔」不至于卡住半分钟。

    NaN 必须排到两端之外：多头侧按 `-inf` 参与排序，空头侧按 `+inf`，这样
    「有效品种不足」的品种永远不会被选中，和 `dropna()` 之后再排等价。
    """
    step = cfg.step()
    rows = np.arange(0, len(score), step)
    mat = score.to_numpy(dtype=float)[rows] if len(rows) else np.zeros((0, score.shape[1]))
    n_valid = np.sum(~np.isnan(mat), axis=1)
    ok = n_valid >= MIN_CROSS

    def _n(pct: float) -> np.ndarray:
        k = np.round(n_valid * float(pct or 0) / 100.0).astype(int)
        k = np.maximum(1, np.minimum(k, np.maximum(1, n_valid // 2)))
        return np.where(ok, k, 0)

    k_long = _n(cfg.top_pct)
    k_short = (np.zeros(len(rows), dtype=int) if cfg.long_only
               else _n(cfg.bot_pct))
    top = np.zeros(mat.shape, dtype=bool)
    bot = np.zeros(mat.shape, dtype=bool)
    width = int(max(k_long.max() if len(k_long) else 0,
                    k_short.max() if len(k_short) else 0))
    if width:
        hi = np.argsort(-np.where(np.isnan(mat), -np.inf, mat), axis=1, kind="stable")
        lo = np.argsort(np.where(np.isnan(mat), np.inf, mat), axis=1, kind="stable")
        for j in range(width):
            sel = np.nonzero(k_long > j)[0]
            if len(sel):
                top[sel, hi[sel, j]] = True
            sel = np.nonzero(k_short > j)[0]
            if len(sel):
                bot[sel, lo[sel, j]] = True
    return rows, top, bot, k_long, k_short, ok


def _place(score: pd.DataFrame, rows, w: np.ndarray, ok) -> pd.DataFrame:
    """把调仓日的权重放回整条时间轴，中间那段沿用上一组。

    有效品种不足的行**留空**而不是清零：`ffill` 之后就是「还没到下一个调仓日，
    先按上一组持仓挂着」，和逐行那版 `continue` 的语义一致。清零等于凭空平仓，
    会在净值和换手上多出一批假动作。
    """
    dense = np.full((len(score), score.shape[1]), np.nan)
    if len(rows):
        dense[rows[ok]] = w[ok]
    return pd.DataFrame(dense, index=score.index,
                        columns=score.columns).ffill().fillna(0.0)


def _select_weights(frames: dict[str, pd.DataFrame], score: pd.DataFrame,
                    cfg: CrossConfig) -> pd.DataFrame:
    """先选品种，再让每个品种自己的时序信号决定仓位。

    时序信号在**各自的行情表**上算（`strategies.run_strategy` / 因子策略），算完
    再对齐到截面面板的索引上。这样 5 分钟周期里各品种交易时段不同也不会串。
    """
    sigs: dict[str, pd.Series] = {}
    for sym, df in frames.items():
        sigs[sym] = _timing_signal(df, cfg)
    sig_panel = pd.concat(sigs, axis=1).sort_index().reindex(score.index).ffill().fillna(0.0)

    rows, top, _bot, k_long, _k_short, ok = _rank(score, cfg)
    inv = 1.0 / np.maximum(k_long, 1)
    held = _place(score, rows, top * inv[:, None], ok)
    # 选中的那一段里，按时序信号决定持有方向；没选中就是 0
    return held * sig_panel


def _timing_signal(df: pd.DataFrame, cfg: CrossConfig) -> pd.Series:
    """单个品种的时序信号（-1/0/1）。"""
    if cfg.timing_family == "rule":
        key = cfg.timing_rule_key
        if key not in STRATEGIES:
            return pd.Series(0.0, index=df.index)
        try:
            out = run_strategy(df, key, dict(cfg.timing_rule_params or {}))
            return pd.Series(out["signal"], index=df.index).astype(float).fillna(0.0)
        except Exception:  # noqa: BLE001
            return pd.Series(0.0, index=df.index)
    keys = [k for k in (cfg.timing_factor_keys or []) if k in factors.FACTORS]
    if not keys:
        return pd.Series(0.0, index=df.index)
    try:
        from ttrade import factor_strategy as fstrat

        fc = fstrat.FactorConfig(**{k: v for k, v in (cfg.timing_factor_cfg or {}).items()
                                    if k in fstrat.FactorConfig.__dataclass_fields__})
        fc.reverse = dict(cfg.timing_factor_reverse or {})
        out = fstrat.run_factor_strategy(df, cfg.timing_factor_mode or "single", keys, fc)
        return pd.Series(out.df["signal"], index=out.df.index).astype(float).fillna(0.0)
    except Exception:  # noqa: BLE001
        return pd.Series(0.0, index=df.index)
