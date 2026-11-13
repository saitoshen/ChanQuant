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

## 风险平价不只看净值波动

`weight ∝ 1 / 标准化风险`，而

    标准化风险 = 年化波动 ÷ 资金利用率 × 回测时长系数

- **利用率折算**（`vol / util`，`util = 平均开仓占用 ÷ 分配额`）：同样 20% 的净值
  波动，一条用满 90% 本金的策略和一条只用 20% 的策略，单位资金的效率差 4.5 倍。
  不折的话杠杆小的策略会一直白拿钱 —— 它「看起来温和」只是因为钱没花出去。
- **盈利因子门槛**（`profit_factor < 1.3`）⇒ 直接落到**最低档**。`profit_factor`
  未知（没成交 / 没量测）不罚，按中性处理。
- **回测时长分档**：≥3 年 ×1.00、2~3 年 ×1.12、1~2 年 ×1.25、半年~1 年 ×1.40，
  **不足半年直接最低档**。系数乘在风险上 ⇒ 回测越短权重越小。为什么上限只到
  1.40：波动率项在组合内的跨度通常 3~10 倍，「时长的影响约等于波动率的 1/3」
  换算成乘数就是 `3^(1/3) ~ 10^(1/3) ≈ 1.44 ~ 2.15`，取 1.40 落在下界以内 ——
  意思是「回测长短最多相当于 3 倍的波动率差异」，不会盖过波动率本身。

「最低档」= 把它的风险分数抬到**有效组里最高**那一档，而不是判 0 —— 判 0 等于
静默把一条策略从组合里删掉，那是替用户做决定。全部条目都触底时自然成了等权。

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

import math
from dataclasses import dataclass, field, replace

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

# 调哪个旋钮把占用调到目标值。
#
# 🔴 「本金」和「占用百分比」是**相乘**的两个因子：
#        平均占用 ≈ 本金 × 占用百分比 × 保证金率 × （仓位节奏系数）
# 所以「把占用放大 k 倍」有无数种拆法。三种拆法的区别只在**本金会不会膨胀**：
#
#   principal：pct 不动，本金 ×k。占用对了，但本金合计往往 3~8 倍于总资金
#   pct      ：本金不动，pct ×k。钱包不膨胀，但 pct 会顶到 500% 上限
#   split    ：本金先归位到「分到的钱」，剩下交给 pct。两边都不极端（默认推荐）
KNOBS = {
    "split": "本金归位＋占用百分比补",
    "principal": "只调本金",
    "pct": "只调占用百分比",
}

# `capital_pct` 的存盘上限，和 `strategy_store._clamp_pct` 保持一致。
# 两边不一致的话，界面算出 620% 写进去，下次加载被夹回 500% —— 手数悄悄变了。
PCT_CAP = 500.0

# ---- 风险平价的三把尺子（见模块文档「风险平价不只看净值波动」） ----
#
# 盈利因子门槛：低于 1.3 的策略直接落到最低档（= 组合里风险最高那一档的权重）。
# 1.3 是用户定的线：1.0 只是不亏手续费，扣掉滑点/冲击成本后 1.3 以下基本没有
# 安全边际。
PF_FLOOR = 1.3

# 回测时长档位：(最少年数, 风险系数, 档名)。**降序**排列，第一个命中的就是它。
#
# 上限为什么只到 1.40：波动率项在组合内的跨度通常 3~10 倍，而「时长的影响约
# 等于波动率的 1/3」换算成乘数就是 3^(1/3) ~ 10^(1/3) = 1.44 ~ 2.15。取 1.40
# 落在下界**以内** —— 意思是「回测长短最多相当于 3 倍的波动率差异」，
# 绝不会盖过波动率本身。再大就会变成「按回测年份拍脑袋」。
SPAN_TIERS = (
    (3.0, 1.00, "回测≥3年"),
    (2.0, 1.12, "回测2~3年"),
    (1.0, 1.25, "回测1~2年"),
    (0.5, 1.40, "回测半年~1年"),
)
# 不足半年 ⇒ 最低档（回测太短，什么结论都可能是运气）
SPAN_MIN_YEARS = 0.5

# 资金利用率的下限。`vol / util` 在 util 极小时会放大几十上百倍，把这条策略的
# 权重压到 0.0x% —— 那已经是「静默删策略」了。2% 以下不再区分（界面上会点名）。
UTIL_FLOOR = 0.02


@dataclass
class Usage:
    """一条策略的「用了多少钱」实测值。"""

    sid: str
    name: str
    sizing: str = "capital"
    # 分配额。🔴 语义是「**量测那一刻**的配置」，不是「现在的配置」：`measure()`
    # 跑一次完整回测要几秒到几分钟，不可能每次打开预览都重量。配置后来又变了
    # 的话，`avg_margin` 这些绝对值也是按老本金量的 —— 必须用 `rebase()` 折算
    principal: float = 0.0
    capital: float = 0.0          # 开仓预算 = principal × capital_pct%
    capital_pct: float = 30.0     # 占用百分比（只有 sizing="capital" 时起作用）
    avg_notional: float = 0.0     # 平均**开仓**名义敞口（|手数| × 开仓价 × 乘数）
    peak_notional: float = 0.0    # 逐根最大名义敞口（含持仓浮动）
    avg_margin: float = 0.0       # 平均开仓占用（名义 × 保证金率）—— 主口径
    peak_margin: float = 0.0      # 逐根最大占用
    util: float = 0.0             # avg_margin / principal（出手时用了分配额的几成）
    vol: float = 0.0              # 年化波动（策略净值）
    ret: float = 0.0              # 累计收益
    max_dd: float = 0.0           # 最大回撤（正数）
    bars: int = 0                 # 回测里的交易笔数（老版本存的是 bar 数）
    # 盈利因子（`profit_factor`，毛利润 / 毛亏损）。0 = 没读到（没成交 / 老结果）
    profit_factor: float = 0.0
    # 回测区间长度（年）。0 = 不知道（没有净值曲线 / 老记录）
    span_years: float = 0.0
    note: str = ""
    # 量测时的配置基准。和 `principal` / `capital_pct` 重复是**故意的**：
    # `rebase()` 会把 `principal` 改写成当前配置，基准得留一份原值，
    # 否则第二次折算就找不到比例了（会把已折算的值再折一遍）
    meas_principal: float = 0.0
    meas_pct: float = 0.0

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
    # 目标占用（= 总资金 × weight）。**口径跟着 basis 走**：basis=usage 时是
    # 目标「平均开仓占用」，basis=peak 时是目标「峰值占用」。列名别写死成
    # 「平均」，否则按峰值校准时那一列会被当成平均值去读
    target: float = 0.0
    avg_margin: float = 0.0       # 实测平均占用（按当前 principal）
    peak_margin: float = 0.0      # 实测峰值占用（按当前 principal）
    scale: float = 1.0            # 总缩放系数（= 本金系数 × 百分比系数）
    old_principal: float = 0.0
    new_principal: float = 0.0
    old_pct: float = 30.0         # 占用百分比（前）
    new_pct: float = 30.0         # 占用百分比（后）
    est_avg: float = 0.0          # 缩放后的预计占用（口径同 target）
    # 标准化风险分数（= 年化波动 ÷ 资金利用率 × 回测时长系数）。风险平价下
    # 权重 ∝ 1/这个数；别的模式下是 0，界面按「—」显示
    risk_score: float = 0.0
    pf: float = 0.0               # 盈利因子（0 = 未知）
    span: float = 0.0             # 回测区间长度（年，0 = 未知）
    # 资质档位的一句话结论（「回测2~3年 ×1.15」「盈利因子 1.12 < 1.3 → 最低档」）。
    # 单独一列，不和 `note` 混在一起 —— `note` 讲的是本金/百分比怎么调的
    tier: str = ""
    # 波动率无效（`vol` 是 NaN 或 0）。风险平价下这些条目会被补齐，权重不是
    # 由它自己的风险算出来的 —— 预览表必须点名，否则「为什么 27 条权重一样」
    # 没人解释得清
    vol_bad: bool = False
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
    自己按收盘价逐根算），现在只用来兜底「回测区间有多长」—— 占用完全从
    `res.metrics` 来。

    除了占用，这里还顺手记下**风险平价要用的另外三个数**：`util`（占用 ÷ 分配额）、
    `profit_factor`、`span_years`。它们都必须和占用在同一次回测里量出来，否则
    「用 A 次的收益率配权、B 次的手数下单」，两边迟早对不上。
    """
    sid = str(getattr(cfg, "sid", "") or "")
    name = str(getattr(cfg, "title", lambda: "")() or sid)
    principal = float(getattr(cfg, "principal", 0) or 0)
    pct = float(getattr(cfg, "capital_pct", 0) or 0)
    u = Usage(sid=sid, name=name,
              sizing=str(getattr(cfg, "sizing", "") or "capital"),
              principal=principal,
              capital=float(getattr(cfg, "capital", 0) or 0),
              capital_pct=pct,
              # 记下这次量测的基准，`rebase()` 靠它把实测值折到后来的配置上
              meas_principal=principal, meas_pct=pct)
    if not u.scalable:
        u.note = "固定手数，组合无法调整"

    met = dict(getattr(res, "metrics", {}) or {})
    # 🔴 `ann_vol` 可能是 NaN（权益被亏穿时 `_metrics` 有意标的）。**必须原样
    # 保留 NaN，不能改写成 0**：NaN 是「风险大到把本金亏光了」，0 是「压根没
    # 量过」，两者在风险平价里该给的权重完全不同（见 `risk_parity`）。
    # 但直接往下传也不行 —— `1/vol` 会把整张表的权重全变成 nan。
    try:
        vol = abs(float(met.get("ann_vol")))
    except (TypeError, ValueError):
        vol = float("nan")
    if not math.isfinite(vol):
        vol = float("nan")
        u.note = "净值被亏穿，波动率无意义（风险平价里按最高风险计）"
    u.vol = vol
    u.ret = float(met.get("total_return") or 0.0)
    u.max_dd = abs(float(met.get("max_dd") or 0.0))
    u.avg_notional = float(met.get("avg_notional") or 0.0)
    u.peak_notional = float(met.get("peak_notional") or 0.0)
    u.avg_margin = float(met.get("avg_margin") or 0.0)
    u.peak_margin = float(met.get("peak_margin") or 0.0)
    u.util = u.avg_margin / principal if principal > 0 else 0.0
    u.bars = int(met.get("n_trades") or 0)
    try:
        u.profit_factor = float(met.get("profit_factor") or 0.0)
    except (TypeError, ValueError):
        u.profit_factor = 0.0
    if not math.isfinite(u.profit_factor) or u.profit_factor < 0:
        # `profit_factor` 在「一笔没成交」时可能是 0 或 NaN，两种都当「未知」，
        # 不能当成「盈利因子极差」去打最低档
        u.profit_factor = 0.0
    u.span_years = span_of(res, frame)
    if u.avg_margin <= 0:
        u.note = u.note or "回测里没有开过仓，量不出占用（只能按波动率配权）"
    return u


def span_of(res, frame: pd.DataFrame | None = None) -> float:
    """回测区间有多长（年）。取净值曲线的首尾，没有净值时退回行情索引。

    为什么不从 `res.metrics` 读：`_metrics` 里压根没有时间跨度字段（只有
    `cagr` / `avg_bars` 这种相对量）。而从 `equity.index` 首尾相减是**精确**的，
    也天然跟着「这次回测真正跑到哪」走 —— 用户改了区间，这里立刻跟着变。

    返回 0 表示「不知道」（没有净值曲线 / 索引不是时间），调用方按中性处理，
    不能当成「回测 0 天」去打最低档。
    """
    for obj in (getattr(res, "equity", None), frame):
        idx = getattr(obj, "index", None)
        if idx is None or len(idx) < 2:
            continue
        try:
            days = float((idx[-1] - idx[0]).days)
        except (TypeError, AttributeError, ValueError):
            continue
        if math.isfinite(days) and days > 0:
            return days / 365.25
    return 0.0


_REBASE_TAG = "实测占用已按当前配置折算"


def rebase(u: Usage, principal: float, capital_pct: float,
           capital: float = 0.0, sizing: str = "") -> Usage:
    """把一条实测记录折算到**当前配置**上（返回新的 `Usage`，不改入参）。

    🔴 这是「点了『应用分配』，下次打开又回到原来的」的根因所在。

    「量占用」是一次完整回测，不可能每次打开预览都重量，所以 `_usages` 里存的
    是**量测那一刻**的本金和占用百分比。用户点「应用分配」之后配置变了、快照
    没变，预览表就会一直显示应用前的「本金(前) / 占用%(前)」，看着像没保存；
    更糟的是 `knob=split` 的 `kprin = 目标 ÷ 本金(前)`、`knob=pct` 的顶格判断
    `room = 上限 ÷ 占用%(前)`，全都拿旧值当分母 —— 表上给的「本金(后)」也会错。

    折算关系：占用 ≈ 本金 × 占用百分比 × 保证金率 × 仓位节奏，所以
    `实测占用(当前配置) = 实测占用(量测时) × 本金比 × 百分比比`。
    折算完 `meas_*` 同步更新成当前值，所以**反复调用是幂等的**
    （第二次比值为 1，不会再折一遍）。

    `fixed` 口径不折：它按手数下单，本金改了手数一动不动，硬乘等于凭空加杠杆。
    """
    p_cur = float(principal or 0.0)
    c_cur = float(capital_pct or 0.0)
    sz = str(sizing or u.sizing or "capital")
    p_old = float(getattr(u, "meas_principal", 0.0) or 0.0) or float(u.principal or 0.0)
    c_old = float(getattr(u, "meas_pct", 0.0) or 0.0) or float(u.capital_pct or 0.0)
    rp = rc = 1.0
    if u_scalable(sz):
        if p_old > 0 and p_cur > 0:
            rp = p_cur / p_old
        if _pct_effective(sz) and c_old > 0 and c_cur > 0:
            rc = c_cur / c_old
    r = rp * rc
    out = replace(u, sizing=sz, principal=p_cur, capital=float(capital or 0.0),
                  capital_pct=c_cur, meas_principal=p_cur, meas_pct=c_cur)
    if abs(r - 1.0) <= 1e-9:
        return out
    if out.avg_margin > 0 or out.peak_margin > 0:
        out.avg_margin *= r
        out.peak_margin *= r
        out.avg_notional *= r
        out.peak_notional *= r
        out.util = out.avg_margin / p_cur if p_cur > 0 else 0.0
        # 只留**最后一次**的折算标记：每改一次配置就调一次，不去重的话
        # 说明列会攒成「折算 ×3.33；折算 ×1.07；折算 ×0.92…」
        keep = [x for x in (out.note or "").split("；")
                if x and not x.startswith(_REBASE_TAG)]
        keep.append(f"{_REBASE_TAG} ×{r:.2f}")
        out.note = "；".join(keep)
    return out


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


def _join(a: str, b: str) -> str:
    """把两段说明拼成「；」分隔的一句话（任一段为空就只留另一段）。"""
    a, b = (a or "").strip(), (b or "").strip()
    if not b:
        return a
    return f"{a}；{b}" if a else b


def span_factor(years: float) -> tuple[float, str, bool]:
    """回测时长 → (风险系数, 档名, 是否最低档)。

    系数**乘在风险上**，所以越大 = 权重越小：回测越长系数越小、分到的钱越多。
    `years <= 0`（不知道）返回 `(1.0, "", False)` —— 中性，不奖不罚。
    """
    try:
        y = float(years or 0.0)
    except (TypeError, ValueError):
        return 1.0, "", False
    if not math.isfinite(y) or y <= 0:
        return 1.0, "", False
    if y < SPAN_MIN_YEARS:
        return (SPAN_TIERS[-1][1],
                f"回测仅 {y * 12:.0f} 个月（不足半年）→ 最低档", True)
    for lo, f, name in SPAN_TIERS:
        if y >= lo:
            return f, name, False
    return SPAN_TIERS[-1][1], SPAN_TIERS[-1][2], False


def risk_scores(usages: list[Usage]) -> tuple[np.ndarray, list[str]]:
    """每条策略的**标准化风险分数**（配权用 `1/分数`）+ 每条一句说明。

        分数 = 年化波动 ÷ 资金利用率 × 回测时长系数

    分数**越小越安全**，所以权重 ∝ `1/分数`。全部有限、全部为正 ——
    调用方可以放心取倒数，不会再出现 nan。

    ## 三处折算，各自的「未知」都不一样

    | 缺什么 | 怎么补 | 为什么不判 0 / 不判最高 |
    |---|---|---|
    | 利用率没量测（`util=0`） | 组合**中位数**利用率 | 不知道它压了多少，折成中位数＝不奖不罚。判 0 会把它当「几乎不用钱 ⇒ 单位风险爆炸」直接压死 |
    | 波动没量测（`vol=0`） | 有效组**几何平均**（典型风险） | 判 0 经 `1/vol` 会吃掉全部资金（实测 4 条没数据的拿走 100%） |
    | 波动 NaN（权益亏穿） | 有效组**最大**风险 | `_metrics` 有意标 NaN ＝「风险大到没意义」，该给最小一档，但不能判 0 |

    ## 「最低档」是什么

    盈利因子 < `PF_FLOOR`、回测不足半年、权益亏穿 —— 这三种一律把分数抬到
    **有效组里最高**那一档（`s_top`），也就是拿到最小的那档权重。抬的时候取
    `max(自己的分数, s_top)`：一个本来就比 `s_top` 更危险的策略不会被这一抬
    反而变安全。全部条目都触底时分数全相等 ⇒ 自然成了等权；如果全部条目的风险
    本来就一样（刚开工作区、配置全同），「最高那一档」就是大家自己，同权是对的
    —— 外面没有更差的档可参照，硬压一个比例就等于凭空发明一个系数。

    返回的说明里会点名「为什么它在最低档」，界面直接显示，不用用户猜。
    """
    n = len(usages)
    if n == 0:
        return np.zeros(0), []
    vol = np.array([float(getattr(u, "vol", 0.0) or 0.0) for u in usages],
                   dtype=float)
    util = np.array([float(getattr(u, "util", 0.0) or 0.0) for u in usages],
                    dtype=float)
    pf = np.array([float(getattr(u, "profit_factor", 0.0) or 0.0)
                   for u in usages], dtype=float)
    yrs = np.array([float(getattr(u, "span_years", 0.0) or 0.0)
                    for u in usages], dtype=float)
    why = [""] * n

    # 注意 `float('nan') or 0.0` 还是 nan（NaN 是真值），所以上面的 `or` 只
    # 处理 None；真正的有效性判断必须走 isfinite
    good = np.isfinite(vol) & (vol > 1e-9)

    # ---- ① 资金利用率 ----
    u_ok = np.isfinite(util) & (util > 0)
    u_ref = float(np.median(util[u_ok])) if u_ok.any() else 1.0
    u_eff = np.maximum(np.where(u_ok, util, u_ref), UTIL_FLOOR)
    for i in range(n):
        if not u_ok[i]:
            why[i] = _join(why[i], (f"资金利用率没量测，按组合中位 {u_ref:.0%} 折算"
                                    if u_ok.any() else "资金利用率没量测，不折算"))
        elif util[i] < UTIL_FLOOR:
            why[i] = _join(
                why[i], f"资金利用率仅 {util[i]:.1%}，按 {UTIL_FLOOR:.0%} 折算")

    # ---- ② 回测时长 ----
    k_span = np.ones(n)
    bottom = np.zeros(n, dtype=bool)
    for i in range(n):
        f, name, bot = span_factor(yrs[i])
        k_span[i] = f
        if bot:
            bottom[i] = True
            why[i] = _join(why[i], name)
        elif name:
            why[i] = _join(why[i], f"{name} ×{f:.2f}")

    # ---- ③ 盈利因子门槛 ----
    pf_ok = np.isfinite(pf) & (pf > 0)
    for i in range(n):
        if pf_ok[i] and pf[i] < PF_FLOOR:
            bottom[i] = True
            why[i] = _join(why[i], f"盈利因子 {pf[i]:.2f} < {PF_FLOOR:g} → 最低档")

    # ---- 分数 ----
    s = np.full(n, np.nan)
    s[good] = vol[good] / u_eff[good] * k_span[good]
    if not good.any():                       # 一条都没量过 ⇒ 无从比较，等权
        return np.ones(n), why
    typical = float(np.exp(np.mean(np.log(s[good]))))     # 几何平均
    worst = float(s[good].max())
    for i in range(n):
        if good[i]:
            continue
        if math.isfinite(vol[i]):            # vol=0：风险未知
            s[i] = typical
            why[i] = _join(why[i], "没有波动实测，按组合典型风险")
        else:                                # vol=NaN：风险已知极大
            s[i] = worst
            bottom[i] = True
            why[i] = _join(why[i], "净值被亏穿，按最高风险")

    # ---- 最低档：抬到「有效组里最高风险」那一档 ----
    free = good & ~bottom
    pool = s[free] if free.any() else s[np.isfinite(s)]
    s_top = float(pool.max()) if len(pool) else worst
    for i in np.where(bottom)[0]:
        # 只往上抬，不往下压：本来就比 s_top 更危险的策略不能被「抬」变安全
        if not math.isfinite(s[i]) or s[i] < s_top:
            s[i] = s_top
        why[i] = _join(why[i], "按最低档配权")
    return s, why


def risk_parity(usages: list[Usage], rets: pd.DataFrame | None = None
                ) -> np.ndarray:
    """风险平价权重。

    有净值序列就用**真 ERC**（含相关性）；没有就退回「波动率倒数」——
    那是 ERC 在相关系数为 0 时的特例，两条高度相关的策略会被它各给一半，
    而 ERC 会把它们当成同一个风险源、合起来给一份。

    ## 🔴 两个必须挡住的坑，都会让整张分配表变成 nan / 严重错配

    **① `ann_vol` 可能是 NaN，而 NaN 会通过 `1/vol` 污染所有人。**
    `backtest._metrics` 对「权益被亏穿」（`equity.min() <= 0`）的策略**有意**
    把 `ann_vol` 标成 NaN —— 净值翻号之后的波动率确实没有意义。但
    `inv = 1/vol` 里只要混进一个 NaN，`inv.sum()` 就是 NaN，于是
    **27 条策略的权重全变 NaN**，界面上权重 / 目标占用 / 缩放 / 本金(后) /
    预计占用整列都是 nan。用户只会看到「算不出来」，看不出是其中**一条**
    爆仓策略干的。

    **② `ann_vol = 0` 比 NaN 更危险 —— 它不报错，直接吃掉全部资金。**
    没有回测实测的策略（取不到行情、因子缺数据、或者压根没点过「量占用」）
    走的是 `Usage` 默认值 `vol=0`。而 `max(vol, 1e-9)` 把 0 变成一个极小的
    正数，`1/1e-9 = 1e9` ⇒ 这些**最不该拿到钱**的策略权重趋近 100%。
    实测 23 条正常 + 4 条没数据：没数据的 4 条合计拿到 **100.00%**，正常的
    23 条拿到 0.00%，而界面上一切正常、不报错。

    ## 处理方式：无效的分两种，各按各的补

    - **`vol = 0`（没量过：取不到行情 / 因子缺数据 / 没点「量占用」）**
      ⇒ 风险**未知**，用有效组的**几何平均**（组合典型风险）。既不放大到
      垄断，也不判 0（那等于「这条不用给钱」，同样没依据）。
    - **`vol = NaN`（权益被亏穿，`_metrics` 有意标 NaN）**
      ⇒ 风险**已知是极大的**，用有效组的**最大波动**。给它最小的一档权重，
      但不能直接判 0 —— 那会静默把一条策略从组合里删掉，属于替用户做决定。

    全部无效（刚开工作区、一条都没量过）时退回等权 —— 那是唯一有依据的
    结论，也是 `weights()` 在别的模式下的一贯兜底。

    ## ERC 也吃标准化（不然相关性一开，利用率/时长/盈利因子全白算）

    有净值序列时走 ERC。但 ERC 只看协方差，看不见 `util` / 时长 / 盈利因子。
    所以喂给它的**不是原始收益，而是标准化收益**：

        x_i = r_i × (s_i / σ_i)          # σ_i = 这条收益序列自己的标准差
        ⇒ σ(x_i) = s_i                   # 标准化之后，波动正好等于风险分数

    ERC 在 `x` 上求「风险贡献相等」，等价于「按标准化风险等权」，和没有相关性
    时的 `1/s` 是同一个口径。🔴 别写成 `r_i / s_i`：那是反的（波动大的反而
    拿到更多钱），而且两种写法在图上一模一样，只能靠权重方向验。

    实测波动为 0 / 非有限的条目（一条直上的净值）没法标准化，整体退回 `1/s`
    —— 这时少了相关性修正，但权重方向仍然对。
    """
    n = len(usages)
    if n == 0:
        return np.zeros(0)
    s, _ = risk_scores(usages)
    if not (np.all(np.isfinite(s)) and np.all(s > 0)):     # 理论上到不了
        return np.full(n, 1.0 / n)
    if rets is not None and rets.shape[1] == n and len(rets) > 30:
        r = rets.to_numpy(dtype=float)
        r = r[np.isfinite(r).all(axis=1)]
        if len(r) > 30:
            sd = r.std(axis=0)
            if np.all(np.isfinite(sd)) and np.all(sd > 0):
                cov = np.cov(r * (s / sd), rowvar=False)
                if np.all(np.isfinite(cov)):
                    w = erc_weights(cov)
                    if np.all(np.isfinite(w)) and float(w.sum()) > 0:
                        return w
    inv = 1.0 / s
    tot = float(inv.sum())
    if not math.isfinite(tot) or tot <= 0:      # 理论上到不了，兜底不为 nan
        return np.full(n, 1.0 / n)
    return inv / tot


def weights(mode: str, usages: list[Usage],
            rets: pd.DataFrame | None = None,
            manual: dict | None = None) -> np.ndarray:
    """三种分配方式 → 权重（和为 1）。

    `manual`：`{sid: 金额或百分比}`。全是百分比就和为 100%，混着写也能认 ——
    有金额就按金额占总额的比例，剩下的人分余下的。

    `risk` 的口径**不是**单纯的波动率倒数，而是「标准化风险」的倒数（见
    `risk_parity` / `risk_scores`）：波动 ÷ 资金利用率 × 回测时长系数，
    盈利因子 <1.3 或回测不足半年的落最低档。
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
             max_scale: float | None = None,
             knob: str = "split",
             pct_cap: float = PCT_CAP) -> list[Share]:
    """把组合工作区的 `total` 分下去，给出每条策略的新 `principal` 与新占用百分比。

    `basis` 决定「用多少钱」按哪一层对齐：

    | basis | 含义 | 什么时候用 |
    |---|---|---|
    | `principal`（默认） | 分到多少就设成多少本金 | 正常组合管理。总资金 ÷ N，可预期，永远不会超 |
    | `usage` | 让**平均开仓占用**等于分到的那份 | 想让每条策略出手时真正压一样多的钱 |
    | `peak` | 同上，但对齐**峰值占用**（逐根最大值） | 比 `usage` 保守：问的是「最坏那一刻压多少钱」 |

    `knob` 决定这个倍数**记在哪个参数上**（见 `KNOBS`）：

        k = 目标占用 ÷ 实测占用            # 总共要放大/缩小多少
        平均占用 ≈ 本金 × 占用百分比 × 保证金率 × 仓位节奏

    - `split`（默认）：先把本金归位到「分到的那份」（`本金合计 = 总资金`，
      钱包不膨胀），剩下的差异交给占用百分比。两边的幅度都不极端。
    - `principal`：占用百分比不动，全压在**本金**上。占用能对准，但本金会
      放大 2~8 倍 —— 这时「本金合计 / 总资金」那一栏会报警。
    - `pct`：本金不动，全压在**占用百分比**上。适合「本金就是账户真实规模、
      不能动」的场景；百分比顶到 `pct_cap` 之后剩下的才转到本金。

    🔴 三种拆法给出**同一个占用**，区别只在本金和百分比怎么分。所以预览表里
    必须同时列出「本金(前→后)」和「占用%(前→后)」，否则用户只看到本金没变，
    会以为没生效 —— 而实际是百分比从 30% 变成了 90%。

    🔴 只有 `sizing="capital"` 的策略，占用百分比才有意义。`risk` 用的是
    `risk_pct`、`fixed` 用的 `volume`、`unit` 用的是本金 —— 对这几种，`knob`
    自动退回「只调本金」，并在 `note` 里写明，绝不假装调过百分比。

    `max_scale=None` 表示**不封顶**：封顶会把几条策略压成同一个倍数，
    权重差异全被抹掉（实测 4 条策略里 3 条撞 20 倍上限，风险平价和等权
    看起来一模一样），比不封顶更容易误导人。

    `mode="risk"` 时，`Share` 会额外带上配权的账本：`risk_score`（标准化风险，
    越大分得越少）、`pf`、`span`、`tier`（「回测2~3年 ×1.12」「盈利因子 1.12
    < 1.3 → 最低档」）。没有这几个字段，用户看到「波动 3% 却只分到 1.2%」时
    完全无从追问。

    缩放是精确的，不用迭代：`capital` / `risk` / `unit` 的手数都只跟
    `principal`（和百分比）成正比，`fixed` 除外（它在 `Usage.scalable` 里标掉）。
    """
    n = len(usages)
    out: list[Share] = []
    if n == 0 or float(total) <= 0:
        return out
    b = str(basis or "principal").lower()
    if b not in BASES:
        b = "principal"
    kn = str(knob or "split").lower()
    if kn not in KNOBS:
        kn = "split"
    cap = max(1.0, float(pct_cap or PCT_CAP))
    w = weights(mode, usages, rets, manual)
    # 风险平价才需要把「分数 / 档位」贴到预览表上。别的模式下不显示 0 分，
    # 否则用户会以为「标准化风险 0 = 没风险」在参与配权
    is_risk = str(mode or "").lower() == "risk"
    sc, why = risk_scores(usages) if is_risk else (np.zeros(n), [""] * n)
    for k, (u, wi) in enumerate(zip(usages, w)):
        s = Share(sid=u.sid, name=u.name, weight=float(wi),
                  target=float(total) * float(wi),
                  avg_margin=u.avg_margin, peak_margin=u.peak_margin,
                  old_principal=u.principal or u.capital or 0.0,
                  old_pct=float(u.capital_pct or 0.0),
                  new_pct=float(u.capital_pct or 0.0),
                  risk_score=float(sc[k]) if is_risk else 0.0,
                  pf=float(u.profit_factor or 0.0),
                  span=float(u.span_years or 0.0),
                  tier=str(why[k] or ""),
                  vol_bad=not (math.isfinite(float(u.vol or 0.0))
                               and float(u.vol or 0.0) > 0),
                  # 把测量阶段的结论带到预览表上：风险平价里「波动率无效」
                  # 「没有回测实测」这类原因必须让人看见，否则权重为什么是
                  # 这个数完全没有解释
                  note=str(getattr(u, "note", "") or ""))
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
            k = s.target / measured
            if max_scale is not None and k > float(max_scale):
                s.note = f"缩放封顶 {float(max_scale):g} 倍"
                k = float(max_scale)
            s.scale = k
            kprin, kpct = _split_scale(u, s, k, kn, cap)
            s.new_principal = u.principal * kprin
            if abs(kpct - 1.0) > 1e-12 and abs(u.capital_pct) > 0:
                s.new_pct = min(u.capital_pct * kpct, cap)
                if kpct > 1.0 and u.capital_pct * kpct > cap + 1e-9:
                    s.note = ((s.note + "；") if s.note else "") + \
                        f"占用百分比顶到 {cap:g}%，余下的落在本金"
            # 预计值必须按**同一口径**给：按峰值校准就要报预计峰值，拿平均值
            # 去和峰值目标比，永远差 40%，看着像没调准
            s.est_avg = measured * kprin * kpct
        out.append(s)
    return out


def _split_scale(u: Usage, s: Share, k: float, knob: str,
                 cap: float) -> tuple[float, float]:
    """把总缩放 `k` 拆成（本金系数, 百分比系数），两者相乘等于 `k`。

    - `principal`：全给本金
    - `pct`：先全给百分比，顶到 `cap` 之后剩下的转本金
    - `split`：本金先归位到目标额，剩下的给百分比，百分比再顶就还给本金

    🔴 非 `capital` 口径一律 `(k, 1)`：那几种的手数压根不看 `capital_pct`，
    硬拆给百分比会算出一个「看起来调过、实际没动」的结果。
    """
    pct = float(u.capital_pct or 0.0)
    if knob == "principal" or pct <= 0 or not _pct_effective(u.sizing):
        return k, 1.0
    room = max(1.0, cap / pct)               # 百分比还最多能放大几倍
    if knob == "pct":
        kpct = min(k, room)
        return k / kpct, kpct
    # split：本金归位到目标额（s.target），百分比承担剩余
    base = float(s.old_principal or 0.0)
    if base <= 0:
        return k, 1.0
    kprin = float(s.target) / base           # 本金先对准「分到的钱」
    if kprin <= 0:
        return k, 1.0
    kpct = k / kprin
    if kpct > room:                          # 百分比顶住，剩下的还给本金
        kpct = room
        kprin = k / kpct
    if kpct < 1e-9:
        return k, 1.0
    return kprin, kpct


def _pct_effective(sizing: str) -> bool:
    """`capital_pct` 对这个仓位口径有没有用（只有 `capital` 有用）。"""
    return str(sizing or "capital").lower() == "capital"


def apply_shares(items: list, shares: list[Share]) -> list:
    """按分配结果生成新配置（**不改原对象，也不落盘**）。

    写回两样东西：`principal` 和 `capital_pct`。

    🔴 以前这里只写 `principal`，理由是「`capital_pct` 是策略自己的仓位性格，
    组合不该覆盖」。这个说法在 `basis="principal"` 下成立，但一旦按**实际占用**
    校准就不成立了：占用 ≈ 本金 × 百分比 × 保证金率，只动本金意味着为了让
    「30% 仓位」的策略压满分到的钱，本金得放大 3 倍以上 —— 钱包凭空膨胀，
    「本金合计 = 总资金」这条基本约定就破了。所以百分比也要参与。

    到底调哪个、各调多少，由 `allocate(knob=...)` 决定，这里只负责照写；
    并且**只在 `sizing="capital"` 时才写百分比** —— 其余口径手数不看它，
    写进去只会让配置看起来变了而手数没变。
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
        if (_pct_effective(getattr(c, "sizing", "capital"))
                and float(s.new_pct or 0) > 0
                and abs(float(s.new_pct) - float(getattr(c, "capital_pct", 0) or 0)) > 1e-9):
            n.capital_pct = round(min(float(s.new_pct), PCT_CAP), 4)
        out.append(n)
    return out


def summarize(shares: list[Share]) -> pd.DataFrame:
    """给界面用的表：权重 / 目标 / 实测占用 / 缩放 / 本金前后 / 占用% 前后。"""
    rows = []
    for s in shares:
        rows.append({
            "策略": s.name,
            "权重": s.weight,
            "标准化风险": s.risk_score,
            "盈利因子": s.pf,
            "回测年数": s.span,
            "目标占用": s.target,
            "实测平均占用": s.avg_margin,
            "实测峰值占用": s.peak_margin,
            "缩放": s.scale,
            "本金(前)": s.old_principal,
            "本金(后)": s.new_principal,
            "占用%(前)": s.old_pct,
            "占用%(后)": s.new_pct,
            "预计占用": s.est_avg,
            "说明": s.note,
            "档位": s.tier,
        })
    return pd.DataFrame(rows)
