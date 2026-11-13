"""组合工作区的资金分配：一条策略「用多少钱」到底怎么算。

## 两层含义，别混

| | 是什么 | 谁在用 |
|---|---|---|
| **分配额** `principal` | 这条策略的账户规模，也是净值曲线的起点 | `capital_pct% × principal` = 开仓预算；`risk_pct% × principal` = 每笔止损金额 |
| **实际占用** | **每笔交易开仓时**的 `|手数| × 开仓价 × 合约乘数 × 保证金率`，取均值 | 回答「它出手一次压多少钱」 |

两者差得很远，而且**不能用逐根均值去量**。`capital_pct = 30%` 只是说「开仓
那一刻拿本金的 30%」；按每根 bar 摊平的话，日内策略一天只有两三根有仓，
其余时间空仓都按 0 进分母，算出来的「平均」只有本金的 1~2% —— 那量的是
「摊到全天每根 bar 用多少」，不是「出手时用多少」。用户要比的是后者：
**我给它 30 万，它一开仓实际压多少**，差多少就调多少。

所以「平均占用」= 逐笔开仓占用的均值。「峰值占用」仍然是逐根最大值 ——
持仓期间保证金跟着结算价走，开仓那一刻通常不是最高点。两者口径不同，
界面上都写清楚。

`measure()` 把实际占用量出来，`allocate()` 用它反过来把分配额调到位：
**让每条策略的平均开仓占用 = 组合给它的那份钱**。这就是「比较一下，相应
调整策略的资金仓位分配」的落地方式。

## 四种仓位口径对缩放的反应

| `sizing` | 手数怎么来 | 放大 `principal` 会怎样 |
|---|---|---|
| `capital` | 分配额 ÷ 每手保证金 | 线性放大（默认口径） |
| `risk` | 止损金额 ÷ (止损倍数 × ATR × 乘数) | 线性放大 |
| `unit` | 本金 ÷ (价格 × 乘数) | 线性放大 |
| `fixed` | 固定 `volume` 手 | **完全不变** —— 标 `scalable=False`，组合没法给它调仓 |

比率都是单利：手数只看本金（或分配的那部分），不看当前权益，所以
「占用 ∝ principal」严格成立，缩放是精确的，不需要迭代。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 分配方式的三种取值，和界面下拉框一一对应
MODES = {
    "equal": "资金等权",
    "risk": "风险平价",
    "manual": "手动设置",
}

# 校准口径：拿什么当作「这条策略实际用了多少钱」
BASES = {
    "principal": "按分配额（不校准）",
    "usage": "按平均占用",
    "peak": "按峰值占用",
}


@dataclass
class Usage:
    """一条策略的「用了多少钱」实测值。"""

    sid: str
    name: str
    sizing: str = "capital"
    principal: float = 0.0        # 分配额（当前配置）
    capital: float = 0.0          # 开仓预算 = principal × capital_pct%
    avg_notional: float = 0.0     # 平均**开仓**名义敞口（|手数| × 开仓价 × 乘数）
    peak_notional: float = 0.0    # 逐根最大名义敞口（含持仓浮动）
    avg_margin: float = 0.0       # 平均开仓占用（名义 × 保证金率）—— 主口径
    peak_margin: float = 0.0      # 逐根最大占用
    util: float = 0.0             # avg_margin / principal（出手时用了分配额的几成）
    vol: float = 0.0              # 年化波动（策略净值）
    ret: float = 0.0              # 累计收益
    max_dd: float = 0.0           # 最大回撤（正数）
    bars: int = 0                 # 回测里的交易笔数（老版本存的是 bar 数）
    note: str = ""

    @property
    def scalable(self) -> bool:
        """这个仓位口径会不会跟着 `principal` 走。

        **必须是派生属性**，不能是字段：字段和 `sizing` 一旦脱钩，`fixed`
        那条会被当成分母去放大本金 —— 而手数压根不动，缩放系数写了也白写，
        报表上却显示「已调整」。这种错不报错，只是组合实际仓位和界面不符。
        """
        return u_scalable(self.sizing)

    @property
    def ok(self) -> bool:
        return self.avg_margin > 0 and self.bars > 0


@dataclass
class Share:
    """一条策略在某个工作区里分到多少。"""

    sid: str
    name: str
    weight: float = 0.0           # 组合内的权重（和为 1）
    target: float = 0.0           # 目标平均占用（= 总资金 × weight）
    avg_margin: float = 0.0       # 实测平均占用（按当前 principal）
    scale: float = 1.0            # principal 的缩放系数
    old_principal: float = 0.0
    new_principal: float = 0.0
    est_avg: float = 0.0          # 缩放后的预计平均占用
    note: str = ""


# ---------------------------------------------------------------- 测量
def measure(cfg, frame: pd.DataFrame | None, res, mult: float = 1.0,
            margin_rate: float = 1.0) -> Usage:
    """把一条策略「实际用了多少钱」量出来。

    🔴 **占用口径只有一处实现**：`backtest._metrics()` 算完放进
    `res.metrics`，这里只读。两处各算一套的话，绩效页显示的占用和配权用的
    占用迟早会不一样，而且不会报错。

    平均占用 = **每笔交易开仓时**的保证金，取均值（`|手数| × 开仓价 × 乘数
    × 保证金率`）。不是逐根 bar 的均值 —— 那个会把空仓时间算进分母，日内
    策略一天两三根有仓，摊出来的「平均」只有本金的 1~2%，看着像「这条策略
    不用钱」，其实每次出手都用满了 `capital_pct`。

    峰值占用是逐根的最大值（持仓期间保证金跟结算价走，开仓那一刻通常不是
    最高点），所以两者口径不同，界面要写清楚。

    `frame` / `mult` / `margin_rate` 保留在签名里是为了兼容老调用方（早先这里
    自己按收盘价逐根算），现在不再使用 —— 占用完全从 `res.metrics` 来。
    """
    sid = str(getattr(cfg, "sid", "") or "")
    name = str(getattr(cfg, "title", lambda: "")() or sid)
    principal = float(getattr(cfg, "principal", 0) or 0)
    u = Usage(sid=sid, name=name,
              sizing=str(getattr(cfg, "sizing", "") or "capital"),
              principal=principal,
              capital=float(getattr(cfg, "capital", 0) or 0))
    if not u.scalable:
        u.note = "固定手数，组合无法调整"

    met = dict(getattr(res, "metrics", {}) or {})
    u.vol = abs(float(met.get("ann_vol") or 0.0))
    u.ret = float(met.get("total_return") or 0.0)
    u.max_dd = abs(float(met.get("max_dd") or 0.0))
    u.avg_notional = float(met.get("avg_notional") or 0.0)
    u.peak_notional = float(met.get("peak_notional") or 0.0)
    u.avg_margin = float(met.get("avg_margin") or 0.0)
    u.peak_margin = float(met.get("peak_margin") or 0.0)
    u.util = u.avg_margin / principal if principal > 0 else 0.0
    u.bars = int(met.get("n_trades") or 0)
    if u.avg_margin <= 0:
        u.note = u.note or "回测里没有开过仓，量不出占用（只能按波动率配权）"
    return u
    return u


def u_scalable(sizing: str) -> bool:
    """这个仓位口径会不会跟着 `principal` 走。"""
    return str(sizing or "capital").lower() != "fixed"


# ---------------------------------------------------------------- 配权
def erc_weights(cov: np.ndarray) -> np.ndarray:
    """等风险贡献（ERC）：每条策略对组合波动的贡献相等。

    用 Spinu (2013) 的凸形式：`min ½wᵀΣw − Σ bᵢ·log wᵢ`（`bᵢ = 1/N`），
    解出来归一化就是 ERC 权重。严格凸 + 对数障碍，牛顿法 5~8 步到机器精度；
    实测 n=24 也只要 8 步。

    🔴 别用「乘性迭代」`w ← w·(1/N)/RC份额` 那种写法：对角占优时收敛，
    相关性一上来就把权重推到角点（实测 5 条相关策略，风险份额会变成
    0 / 0.5 / 0.5，看着像解出来了，其实只留了两条）。
    """
    c = np.asarray(cov, dtype=float)
    n = c.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    # 数值上保证正定：加一点点对角，避免完全共线时牛顿方向失效
    tr = float(np.trace(c)) / n
    c = c + np.eye(n) * max(tr, 1.0) * 1e-10
    b = np.full(n, 1.0 / n)
    w = 1.0 / np.sqrt(np.maximum(np.diag(c), 1e-18))
    w = w / w.sum()

    def f(x):
        return 0.5 * x @ c @ x - b @ np.log(x)

    fv = f(w)
    for _ in range(100):
        g = c @ w - b / w
        h = c + np.diag(b / w ** 2)
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:               # 极端共线
            step = np.linalg.lstsq(h, g, rcond=None)[0]
        t = 1.0
        nw = None
        while t > 1e-12:
            cand = w - t * step
            if np.all(cand > 0):
                cv = f(cand)
                if cv < fv:
                    nw, fv = cand, cv
                    break
            t *= 0.5
        if nw is None:                              # 已经到底了
            break
        if np.max(np.abs(nw - w)) < 1e-14:
            w = nw
            break
        w = nw
    return w / w.sum()


def risk_parity(usages: list[Usage], rets: pd.DataFrame | None = None
                ) -> np.ndarray:
    """风险平价权重。

    有净值序列就用**真 ERC**（含相关性）；没有就退回「波动率倒数」——
    那是 ERC 在相关系数为 0 时的特例，两条高度相关的策略会被它各给一半，
    而 ERC 会把它们当成同一个风险源、合起来给一份。
    """
    n = len(usages)
    if n == 0:
        return np.zeros(0)
    if rets is not None and rets.shape[1] == n and len(rets) > 30:
        r = rets.to_numpy(dtype=float)
        r = r[np.isfinite(r).all(axis=1)]
        if len(r) > 30:
            cov = np.cov(r, rowvar=False)
            if np.all(np.isfinite(cov)):
                return erc_weights(cov)
    vol = np.array([max(u.vol, 1e-9) for u in usages], dtype=float)
    inv = 1.0 / vol
    return inv / inv.sum()


def weights(mode: str, usages: list[Usage],
            rets: pd.DataFrame | None = None,
            manual: dict | None = None) -> np.ndarray:
    """三种分配方式 → 权重（和为 1）。

    `manual`：`{sid: 金额或百分比}`。全是百分比就和为 100%，混着写也能认 ——
    有金额就按金额占总额的比例，剩下的人分余下的。
    """
    m = str(mode or "equal").lower()
    n = len(usages)
    if n == 0:
        return np.zeros(0)
    if m == "risk":
        return risk_parity(usages, rets)
    if m == "manual" and manual:
        raw = np.array([float(manual.get(u.sid, 0.0) or 0.0) for u in usages])
        tot = float(raw.sum())
        if tot > 0:
            return raw / tot
    return np.full(n, 1.0 / n)


# ---------------------------------------------------------------- 分配
def allocate(usages: list[Usage], total: float, mode: str = "equal",
             rets: pd.DataFrame | None = None,
             manual: dict | None = None,
             basis: str = "principal",
             max_scale: float | None = None) -> list[Share]:
    """把组合工作区的 `total` 分下去，给出每条策略的新 `principal`。

    `basis` 决定「用多少钱」按哪一层对齐：

    | basis | 含义 | 什么时候用 |
    |---|---|---|
    | `principal`（默认） | 分到多少就设成多少本金 | 正常组合管理。总资金 ÷ N，可预期，永远不会超 |
    | `usage` | 让**平均开仓占用**等于分到的那份 | 想让每条策略出手时真正压一样多的钱 |
    | `peak` | 同上，但对齐**峰值占用**（逐根最大值） | 比 `usage` 保守：问的是「最坏那一刻压多少钱」 |

    🔴 `usage` / `peak` 往往会让 `本金合计 > 总资金`，因为 `capital_pct=30%`
    只是「开仓那一刻用 30%」—— 平均开仓占用通常只有分配额的 20%~40%
    （满仓的策略也可能接近 100%）。对齐「平均开仓占用」意味着本金放大 2~5 倍；
    对齐「峰值占用」时倍数还可能更高（持仓期间价格涨上去，保证金跟着涨）。
    所以这两个口径要配着「本金合计 / 总资金」的告警一起看，默认用 `principal`。

    `max_scale=None` 表示**不封顶**：封顶会把几条策略压成同一个倍数，
    权重差异全被抹掉（实测 4 条策略里 3 条撞 20 倍上限，风险平价和等权
    看起来一模一样），比不封顶更容易误导人。

    缩放是精确的，不用迭代：四种仓位口径的手数都只跟 `principal` 成正比
    （`fixed` 除外，它在 `Usage.scalable` 里已经标掉）。
    """
    n = len(usages)
    out: list[Share] = []
    if n == 0 or float(total) <= 0:
        return out
    b = str(basis or "principal").lower()
    if b not in BASES:
        b = "principal"
    w = weights(mode, usages, rets, manual)
    for u, wi in zip(usages, w):
        s = Share(sid=u.sid, name=u.name, weight=float(wi),
                  target=float(total) * float(wi),
                  avg_margin=u.avg_margin,
                  old_principal=u.principal or u.capital or 0.0)
        if b == "principal":
            s.new_principal = s.target
            s.scale = (s.target / s.old_principal) if s.old_principal > 0 else 0.0
            s.est_avg = u.avg_margin * s.scale
            if not u.scalable:
                s.new_principal = s.old_principal
                s.scale = 1.0
                s.est_avg = u.avg_margin
                s.note = "固定手数，本金改了也不影响手数"
            out.append(s)
            continue
        measured = u.avg_margin if b == "usage" else u.peak_margin
        if not u.scalable:
            s.scale, s.new_principal, s.est_avg = 1.0, s.old_principal, u.avg_margin
            s.note = "固定手数，组合无法调仓"
        elif measured <= 0 or u.principal <= 0:
            # 没有实测占用（一路上空仓 / 没行情）：按分配额直接设，不放大 ——
            # 放大一个「空仓策略」等于凭空给它加杠杆
            s.scale = (s.target / u.principal) if u.principal > 0 else 1.0
            s.new_principal = u.principal * s.scale
            s.est_avg = u.avg_margin * s.scale
            s.note = "没有实测占用，按分配额直接设"
        else:
            s.scale = s.target / measured
            if max_scale is not None and s.scale > float(max_scale):
                s.note = f"缩放封顶 {float(max_scale):g} 倍"
                s.scale = float(max_scale)
            s.new_principal = u.principal * s.scale
            s.est_avg = u.avg_margin * s.scale
        out.append(s)
    return out


def apply_shares(items: list, shares: list[Share]) -> list:
    """按分配结果生成新配置（**不改原对象，也不落盘**）。

    只动 `principal`：`capital_pct` / `risk_pct` / `sizing` 是策略自己的仓位
    决定方式，组合不该覆盖 —— 一覆盖就变成「组合改了策略的性格」。
    """
    bysid = {s.sid: s for s in shares}
    out = []
    for c in items:
        s = bysid.get(str(getattr(c, "sid", "") or ""))
        if s is None or s.new_principal <= 0:
            out.append(c)
            continue
        n = c.clone() if hasattr(c, "clone") else c
        n.principal = round(float(s.new_principal), 2)
        out.append(n)
    return out


def summarize(shares: list[Share]) -> pd.DataFrame:
    """给界面用的表：权重 / 目标 / 实测占用 / 缩放 / 本金前后。"""
    rows = []
    for s in shares:
        rows.append({
            "策略": s.name,
            "权重": s.weight,
            "目标平均占用": s.target,
            "实测平均占用": s.avg_margin,
            "缩放": s.scale,
            "本金(前)": s.old_principal,
            "本金(后)": s.new_principal,
            "预计平均占用": s.est_avg,
            "说明": s.note,
        })
    return pd.DataFrame(rows)
