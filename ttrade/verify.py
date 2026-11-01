"""策略验证：查前视偏差，同时给「逐根重放」备好数据。

## 为什么要单独一个模块

回测是「拿整段历史算一遍」，验证是「站在第 i 根上，只用那天能看到的数据算一遍」。
两件事的输出长得一样（信号、净值、交易），但假设完全不同 —— 回测跑得快是因为它
默认因子不会偷看未来；验证的全部意义就是**不相信这个默认**。

所以这里的做法是反过来的：不重算因子，而是**把未来那段数据改掉**，看第 i 根上
的信号变不变。

- 不变 → 这根 bar 的决策没用到未来（不管因子内部怎么写的）
- 变了 → 有前视偏差，而且能指出是哪一根、哪条策略/因子

比「逐根截断重算」快两个数量级（截断重算要 N 次全量计算，这里只要 N 次扰动后的
全量计算，而且 N 只有 8~12），而且**不需要知道因子用了哪张辅助表** —— 展期、IV、
基本面这些外挂数据也一并覆盖了：它们按日期 join，只要 join 的键是 bar 日期，
扰动未来就影响不到第 i 根。

## 两种检查

1. **策略级**：扰动第 i 根之后的 OHLCV，重跑整条策略，比较第 i 根信号
2. **因子级**：同样扰动，比较每个因子在第 i 根的值 —— 策略没问题但某个因子
   偷看了未来时，靠这一层定位

## 逐根重放

`Report.df` 带着完整信号和净值，界面层拿它做「播放」：每帧只画到第 i 根，
看起来就是策略一天天走过来的样子。这部分纯展示，不需要重算任何东西。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttrade import replay

# 检查点最少从第几根开始。太靠前全是预热期的 NaN，比出来的「一致」没有意义
WARMUP = 60

# 扰动倍数。3 倍足够把任何「偷看一根」暴露出来，又不至于把 rolling 里的
# 浮点误差放大成假阳性（比较用 isclose，见 `_same`）
PERTURB = 3.0


@dataclass
class Check:
    """一个检查点：第 idx 根上，扰动未来之后信号有没有变。"""

    idx: int
    date: str
    signal: int
    ok: bool
    note: str = ""


@dataclass
class FactorBound:
    """一个因子在第 idx 根上的取值，以及扰动未来之后还一不一样。"""

    key: str
    label: str
    idx: int
    value: float
    ok: bool
    note: str = ""


@dataclass
class Report:
    """验证结果。`error` 非空时其余字段都不可用。"""

    df: pd.DataFrame | None = None          # 带 signal 列的完整 K 线（重放用）
    raw: pd.DataFrame | None = None         # 原始 K 线
    res: object | None = None               # 原回测结果（净值 / 交易）
    label: str = ""
    period: str = "1d"
    market: str = ""
    checks: list[Check] = field(default_factory=list)
    factors: list[FactorBound] = field(default_factory=list)
    leaks: list[str] = field(default_factory=list)
    n_bars: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        """全部检查点一致 = 没查出前视偏差。"""
        if self.error:
            return False
        return not self.leaks

    def summary(self) -> str:
        if self.error:
            return f"验证失败：{self.error}"
        if not self.checks:
            return "没有可用的检查点（数据太短）"
        if self.leaks:
            head = "；".join(self.leaks[:3])
            more = f"… 共 {len(self.leaks)} 处" if len(self.leaks) > 3 else ""
            return f"查出 {len(self.leaks)} 处前视偏差：{head}{more}"
        tail = ""
        if self.factors:
            nk = len({f.key for f in self.factors})
            tail = f"，{nk} 个因子也没用到未来"
        return f"{len(self.checks)} 个检查点全部一致{tail}"


def _same(a, b) -> bool:
    """两个信号 / 因子值算不算「一样」。NaN 和 NaN 也算一样。"""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if fa != fa and fb != fb:
        return True
    if fa != fa or fb != fb:
        return False
    return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))


def _perturb(df: pd.DataFrame, after: int) -> pd.DataFrame:
    """把第 `after` 根之后的数据改成「另一段历史」。

    只动价格和量，不动索引和列结构 —— 策略和因子看到的东西形状不变，数值全变。
    用乘法而不是加常数：加法对「收益率 / 分位」这类因子几乎没影响（涨跌比例没变），
    乘法会把未来的涨跌幅彻底打乱。
    """
    out = df.copy()
    n = len(out)
    if after + 1 >= n:
        return out
    tail = out.iloc[after + 1:]
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            # 交替放大 / 缩小，让未来的「涨」变成「跌」，方向性因子才会现形
            scale = np.where(np.arange(len(tail)) % 2 == 0, PERTURB, 1.0 / PERTURB)
            out.loc[tail.index, col] = tail[col].to_numpy(float) * scale
    if "volume" in out.columns:
        out.loc[tail.index, "volume"] = tail["volume"].to_numpy(float) * 7.0
    if "open_interest" in out.columns:
        out.loc[tail.index, "open_interest"] = (
            tail["open_interest"].to_numpy(float) * 5.0)
    if "amount" in out.columns:
        out.loc[tail.index, "amount"] = tail["amount"].to_numpy(float) * 7.0
    return out


def _checkpoints(n: int, want: int) -> list[int]:
    """均匀撒检查点。从 `WARMUP` 之后开始，最后一根也要查（边界最容易出事）。"""
    lo = min(WARMUP, max(1, n // 4))
    hi = n - 1
    if hi <= lo:
        return []
    want = max(1, min(int(want), hi - lo))
    step = (hi - lo) / max(1, want - 1) if want > 1 else 0
    pts = sorted({int(round(lo + step * k)) for k in range(want)})
    return [p for p in pts if lo <= p <= hi]


def _fmt_date(ts) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)


def run(cfg, df: pd.DataFrame, market: str = "", n_checks: int = 8,
        factor_keys=None, progress=None) -> Report:
    """跑一遍验证。出错不抛异常，写进 `Report.error`。

    `factor_keys` 不给就按 `cfg` 自己取（规则策略不选因子时为空）。
    """
    mk = market or replay.market_of(cfg)
    period = getattr(cfg, "period", "") or "1d"
    if df is None or df.empty:
        return Report(error=f"{cfg.key()} 没有行情数据", market=mk, period=period)

    full = replay.run(cfg, df, mk)
    if full.error:
        return Report(error=full.error, market=mk, period=period)
    if full.df is None or "signal" not in full.df.columns:
        return Report(error="回测没有产出信号列，无法验证", market=mk, period=period)

    base = full.df
    sig = base["signal"].to_numpy(int)
    n = len(base)

    # 机器学习一轮训练几十折，检查点多了会等到没耐心 —— 少查几个但照查
    mode = getattr(cfg, "factor_mode", "") or ""
    if mode == "ml":
        n_checks = min(int(n_checks), 4)
    pts = _checkpoints(n, n_checks)

    rep = Report(df=base, raw=full.raw, res=full.res, label=full.label,
                 period=period, market=mk, n_bars=n)
    if not pts:
        rep.error = f"只有 {n} 根 K 线，不够做验证（至少要 {WARMUP + 20} 根）"
        return rep

    for k, i in enumerate(pts):
        if progress is not None:
            progress(k, len(pts))
        try:
            pert = _perturb(df, i)
            r2 = replay.run(cfg, pert, mk)
            if r2.error:
                rep.checks.append(Check(i, _fmt_date(base.index[i]), int(sig[i]),
                                        False, f"重跑失败：{r2.error}"))
                continue
            s2 = r2.df["signal"].to_numpy(int)
            same = _same(sig[i], s2[i])
            rep.checks.append(Check(
                i, _fmt_date(base.index[i]), int(sig[i]), same,
                "" if same else f"扰动未来后信号从 {int(sig[i])} 变成 {int(s2[i])}"))
            if not same:
                rep.leaks.append(
                    f"{_fmt_date(base.index[i])} 信号 {int(sig[i])}→{int(s2[i])}")
        except Exception as exc:  # noqa: BLE001 - 验证本身不该把界面打崩
            rep.checks.append(Check(i, _fmt_date(base.index[i]), int(sig[i]),
                                    False, f"{type(exc).__name__}: {exc}"))

    # 因子级：定位到具体是哪个因子偷看了未来
    keys = list(factor_keys if factor_keys is not None
                else (getattr(cfg, "factor_keys", []) or []))
    if keys:
        from ttrade.factors import FACTORS

        fpts = pts[:3] if len(pts) > 3 else pts
        for key in keys[:8]:
            spec = FACTORS.get(key)
            if spec is None:
                continue
            for i in fpts:
                try:
                    v1 = spec.func(df).iloc[i]
                    v2 = spec.func(_perturb(df, i)).iloc[i]
                except Exception:  # noqa: BLE001 - 因子算不出来不是验证的结论
                    continue
                same = _same(v1, v2)
                try:
                    val = float(v1)
                except (TypeError, ValueError):
                    val = float("nan")
                rep.factors.append(FactorBound(
                    key, spec.label, i, val, same,
                    "" if same else f"扰动未来后从 {v1:.6g} 变成 {v2:.6g}"))
                if not same:
                    rep.leaks.append(f"{key} @ {_fmt_date(base.index[i])} 用到未来数据")

    if progress is not None:
        progress(len(pts), len(pts))
    return rep


def boundaries(df: pd.DataFrame, keys, bars: int = 3) -> list[dict]:
    """每个因子「最后一根非空值」出现在哪，用来给界面上那张边界表填数。

    这不是前视检测（那是 `run` 干的事），而是**解释**：用户看到某个因子在
    某段之后整列空，得知道是数据没下够，还是因子本来就只有那么长。
    """
    from ttrade.factors import FACTORS

    out = []
    for key in keys or []:
        spec = FACTORS.get(key)
        if spec is None:
            continue
        try:
            s = spec.func(df).dropna()
        except Exception:  # noqa: BLE001
            out.append({"key": key, "label": getattr(spec, "label", key),
                        "last": "算不出", "bars": -1})
            continue
        if not len(s):
            out.append({"key": key, "label": spec.label, "last": "全空", "bars": -1})
            continue
        lag = int(len(df) - 1 - df.index.get_loc(s.index[-1]))
        out.append({
            "key": key, "label": spec.label,
            "last": _fmt_date(s.index[-1]), "bars": lag,
        })
    return out
