"""离线回放：拿一份保存好的配置，不碰界面，直接算出回测结果。

三处需要它：

- 「策略交易和组合」里双击一条策略，看它的 K 线 / 绩效 / 因子；
- 选中多条策略，出一份组合报告；
- 交易 Agent 说「用双均线回测 RB」，结果要落在组合页而不是研究页。

这三件事都不该去动「策略研究」页那排控件 —— 那会把手上的配置冲掉，而且
用户切回来会发现界面莫名其妙变成了另一套参数。

所以这里把 `MainWindow.run_analysis` 里「选策略 → 跑因子 → 回测」那一段抽成
纯函数：输入只有行情表和 `StrategyConfig`，输出一个 `Replay`。视图层拿到
`Replay` 之后自己拼 `ViewCtx`（那是 GUI 的对象，`ttrade` 不该反向依赖 `gui`）。

## 和研究页的差别

- 机器学习模式同步算完，不起线程。研究页要线程是因为拖滑杆会反复重算，
  这里是一次性操作，阻塞几百毫秒可以接受，代码也少一半。
- 不处理实时行情。回放就是回放。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ttrade import factor_strategy as fstrat
from ttrade.backtest import BacktestConfig, bars_per_year, buy_and_hold, run_backtest
from ttrade.strategies import STRATEGIES, run_strategy

# 各市场的成本假设。`gui/app.py` 的 MARKETS 直接用这三项，保证「研究页算出来
# 的数」和「组合页回放出来的数」用同一套假设 —— 两边不一致的话，同一条策略在
# 两个页面显示不同收益，用户只会认为程序算错了。
MARKET_COST = {
    "us": {"fee": 0.0005, "slippage": 0.0005, "allow_short": False},
    "cn": {"fee": 0.0001, "slippage": 0.0002, "allow_short": True},
}


def market_of(cfg) -> str:
    """配置属于哪个市场。有交易所就是期货，没有就是美股 —— 和研究页的判据一致。"""
    return "cn" if getattr(cfg, "exchange", "") else "us"


def make_bt(market: str, allow_short: bool = False) -> BacktestConfig:
    """按市场成本建一份回测配置。`allow_short` 来自配置里的 `note == "short"`。"""
    c = MARKET_COST.get(market) or MARKET_COST["us"]
    return BacktestConfig(
        fee_rate=c["fee"],
        slippage=c["slippage"],
        long_only=not (c["allow_short"] and allow_short),
    )


def factor_config(d: dict) -> fstrat.FactorConfig:
    """把存下来的 dict 还原成 FactorConfig。

    过滤掉不认识的键：配置文件是跨版本存的，新版本删掉的字段留在 json 里
    会让 `FactorConfig(**d)` 直接 TypeError，而用户只是打开一个旧策略看看图。
    """
    known = fstrat.FactorConfig.__dataclass_fields__
    return fstrat.FactorConfig(**{k: v for k, v in (d or {}).items() if k in known})


def hold_text(bars: float, period: str, bars_per_year: float = 252.0) -> str:
    """平均持仓长度的可读写法。

    `avg_bars` 的单位是「根」，日线一根就是一天，分钟级一根只有几十分钟。
    统一写「天」会把 1 小时的 160 根说成 160 天（实际约 32 个交易日）。
    """
    if period == "1d":
        return f"{bars:.0f} 天"
    bpy = float(bars_per_year or 252)
    days = bars / bpy * 365.25 if bpy > 0 else 0.0
    return f"{bars:.0f} 根（约 {days:.0f} 天）"


def summary(rp: "Replay", head: str, period: str) -> str:
    """把一次回放摊成一段纯文本。

    两个用处：Agent 页右下角的「运行输出」，以及模型自己发起回测时喂回去的
    那段结果。所以写的是**数字**，不是「回测完成」这种话 —— 模型要的就是数字，
    少一行它就要反过来问用户，而用户手上只有这张图。

    没有基准（组合报告）时省掉「基准 / 超额」两行：那两行写「—」只会让模型
    以为算失败，不如不写。
    """
    if rp.res is None:
        return ""
    # 组合报告没有「原始行情」这一说（几条曲线拼出来的），样本跨度就看资金
    # 曲线自己 —— 拿某一条成分的行情去报跨度，会少算或多算一截。
    idx = rp.raw.index if rp.raw is not None and len(rp.raw) else rp.res.equity.index
    if idx is None or not len(idx):
        return ""
    m = rp.res.metrics
    b = rp.bh.metrics if rp.bh is not None else None
    span = f"样本 {len(idx)} 根（{idx[0]:%Y-%m-%d} → {idx[-1]:%Y-%m-%d}）"

    title = f"{head} · {period}"
    if rp.label:
        title += f" · {rp.label}"
    lines = [title, span]
    lines.append(
        f"策略  总收益 {m['total_return'] * 100:+.1f}%｜年化 {m['cagr'] * 100:+.2f}%"
        f"｜最大回撤 {m['max_dd'] * 100:.1f}%｜夏普 {m['sharpe']:.2f}"
        f"｜卡玛 {m['calmar']:.2f}"
    )
    if b is not None:
        lines.append(
            f"基准  总收益 {b['total_return'] * 100:+.1f}%｜年化 {b['cagr'] * 100:+.2f}%"
            f"｜最大回撤 {b['max_dd'] * 100:.1f}%｜夏普 {b['sharpe']:.2f}"
        )
        lines.append(f"超额  {(m['cagr'] - b['cagr']) * 100:+.2f}%/年")
    lines.append(
        f"交易  {m['n_trades']} 笔｜胜率 {m['win_rate'] * 100:.0f}%"
        f"｜盈亏比 {m['profit_factor']:.2f}｜仓位时间 {m['exposure'] * 100:.0f}%"
        f"｜平均持仓 {hold_text(m['avg_bars'], rp.period, rp.bars_per_year)}"
    )
    if not rp.long_only and m.get("n_short"):
        lines.append(
            f"方向  多头 {m['long_exposure'] * 100:.0f}% / 空头 {m['short_exposure'] * 100:.0f}%"
            f"｜空头 {m['n_short']} 笔"
            + (f"｜爆仓 {m['n_blowup']} 次" if m.get("n_blowup") else "")
        )
    return "\n".join(lines)


@dataclass
class Replay:
    """一次离线回放的全部产物。视图层要什么这里就有什么。"""

    df: pd.DataFrame | None = None          # 含 signal 列，画 K 线用
    raw: pd.DataFrame | None = None         # 原始行情，算因子评估用
    res: object | None = None               # backtest.Result
    bh: object | None = None                # 买入持有基准；组合没有
    label: str = ""
    family: str = "rule"
    rule_key: str = ""
    period: str = "1d"                      # 原始周期键（1d / 5m …），不是中文标签
    factor_cfg: object | None = None
    outcome: object | None = None           # fstrat.FactorOutcome
    keys: list = field(default_factory=list)
    filter_active: bool = False
    long_only: bool = False
    bars_per_year: float = 252.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.res is not None and not self.error


WARMUP_BARS = 250      # 回测窗口前面多留几根，让均线一进入窗口就是暖的


def window(df: pd.DataFrame | None, years: float) -> pd.DataFrame | None:
    """按「最近 N 年」截一段行情。`years` 为 0 / None 时原样返回。

    前面多留 `WARMUP_BARS` 根：直接从窗口起点开始算，MA200 之类要空转 200 根，
    日线近一年就只剩 40 根有信号 —— 图上看起来像「策略一直空仓」。
    留出预热段之后，窗口内的每一根都是有信号的；代价是窗口前那一段也进了
    指标计算，收益从预热段开始累计（预热段不开仓，对总收益几乎没有影响）。
    """
    if df is None or df.empty:
        return df
    try:
        years = float(years or 0)
    except (TypeError, ValueError):
        years = 0.0
    if years <= 0:
        return df
    start = pd.Timestamp(df.index[-1]) - pd.DateOffset(days=int(round(years * 365.25)))
    if start <= pd.Timestamp(df.index[0]):
        return df
    i = int(df.index.searchsorted(start))
    return df.iloc[max(0, i - WARMUP_BARS):]


def run(cfg, df: pd.DataFrame | None, market: str = "",
        with_bh: bool = True) -> Replay:
    """按 `cfg` 跑一遍回测。出错不抛异常，写进 `Replay.error`。

    不抛异常是有意的：调用方是「双击一行看个图」，报错该显示在页面上，而不是
    变成一个没人接的 traceback。
    """
    mk = market or market_of(cfg)
    if df is None or df.empty:
        return Replay(error=f"{cfg.key()} 没有行情数据")

    # 配置里限定了回测区间就只算这一段。5 分钟十年 26 万根，跑因子评估 /
    # 机器学习要几十秒，而用户通常只想看最近两年
    df = window(df, getattr(cfg, "range_years", 0) or 0)
    if df is None or len(df) < 2:
        return Replay(error=f"{cfg.key()} 在所选区间里没有足够的行情")

    period = getattr(cfg, "period", "") or "1d"
    bt = make_bt(mk, getattr(cfg, "note", "") == "short")
    # 年化口径跟着周期走：5 分钟一年一万多根，按 252 算夏普会虚高一个数量级
    bt.bars_per_year = bars_per_year(df.index)
    bpy = bt.bars_per_year

    family = getattr(cfg, "family", "rule") or "rule"
    rule_signal = None
    base = df
    label = ""
    rule_key = getattr(cfg, "rule_key", "") or ""

    if family == "rule":
        if rule_key not in STRATEGIES:
            return Replay(error=f"平台没有这个策略：{rule_key or '（空）'}")
        try:
            base = run_strategy(df, rule_key, dict(getattr(cfg, "rule_params", {}) or {}))
        except Exception as exc:  # noqa: BLE001
            return Replay(error=f"策略计算失败：{exc}")
        rule_signal = base["signal"].to_numpy(int)
        label = STRATEGIES[rule_key].label

    # 规则策略 + 因子过滤：`factor_mode` 非空就说明这份配置带着过滤条件。
    # 因子策略本身当然也走这一段。
    mode = getattr(cfg, "factor_mode", "") or ""
    filter_active = family == "rule" and bool(mode)
    keys = list(getattr(cfg, "factor_keys", []) or [])
    if family == "rule" and not filter_active:
        try:
            res = run_backtest(base, bt, label=label)
        except Exception as exc:  # noqa: BLE001
            return Replay(error=f"回测失败：{exc}")
        bh = buy_and_hold(df, bt) if with_bh else None
        return Replay(df=base, raw=df, res=res, bh=bh, label=label, family=family,
                      rule_key=rule_key, period=period, long_only=bt.long_only,
                      bars_per_year=bpy)

    if not keys:
        return Replay(error=f"{cfg.title()} 没有选中因子，无法回放")
    if mode == "ml" and len(keys) < 2:
        return Replay(error="机器学习多因子至少需要 2 个因子")

    fc = factor_config(getattr(cfg, "factor_cfg", {}) or {})
    try:
        if rule_signal is None:
            out = fstrat.run_factor_strategy(df, mode or "single", keys, fc)
        else:
            out = fstrat.run_rule_with_filter(df, rule_signal, mode or "single", keys, fc)
    except Exception as exc:  # noqa: BLE001
        return Replay(error=f"因子计算失败：{exc}")

    if family == "factor":
        label = out.label
    else:
        label = f"{label} + 因子过滤（{out.pack.label}）"
    try:
        res = run_backtest(out.df, bt, label=label)
    except Exception as exc:  # noqa: BLE001
        return Replay(error=f"回测失败：{exc}")
    bh = buy_and_hold(df, bt) if with_bh else None
    return Replay(
        df=out.df, raw=df, res=res, bh=bh, label=label, family=family,
        rule_key=rule_key, period=period, factor_cfg=fc, outcome=out, keys=keys,
        filter_active=filter_active, long_only=bt.long_only, bars_per_year=bpy,
    )
