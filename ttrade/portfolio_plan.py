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
- **盈利因子门槛**（`profit_factor < 1.1`）⇒ 直接落到**最低档**。`profit_factor`
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

## 一手保底：分到的钱开不出一手，等于没分

风险平价、资金等权都可能给某条策略分到「不够一手」的钱。`capital` 口径下
那就是 **0 手**：`round(预算 ÷ 每手保证金) == 0` ⇒ 整段不交易 ⇒ 净值是直线、
占用 0、波动 0、盈利因子 0，然后以「没数据」的身份被配权规则再罚一次。
用户看到的是一条「永远不动的曲线」，却看不出原因是钱不够。

所以 `allocate()` 最后一步做一手保底（`apply_lot_floor`）：

    判据：在「历史最贵那一手」的成本下，算出来的手数要 ≥ 0.5
          （`lots_round(0.5) == 1`，0.49 就已经是 0 手）
    P_req = 0.5 × L_peak ÷ k
        其中 L_peak = 按区间最高价算的一手保证金
             k      = 开仓预算 ÷ 本金（`capital`：占用%；`unit`：保证金率）

- **为什么按最贵那一手判**：价格翻 2~5 倍后同样的本金手数会掉到 0.5 以下，
  这就是「前期能开、后期开不出来」。按均价判会漏掉后半段。
- **为什么门槛是 0.5 不是 1**：`lots_round(x) = floor(x+0.5)`，0.5 手就下 1 手。
  按 1.0 判等于要求翻倍，会把「明明开得出来」的策略也抬一遍。
- **抬出来的差额不从别人身上扣**：直接加进总本金。挤别人等于让「够一手的」
  替「不够一手的」出钱，那是另一个策略被改了仓位，用户没要求。
- **上限**：单条上调额不超过组合总资金的 `LOT_LIFT_CAP_SHARE`。到了上限还差
  就停下，在说明里写「上调到 X 仍不足一手（需 Y）」—— 否则它会以「已保底」
  的样子继续空跑。
- **只对 `capital` / `unit` 生效**：`budget` / `risk` 在 `size_lots` 里本来
  就有 `max(1, ...)`，买不起也至少 1 手；`fixed` 手数写死。
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
# 盈利因子门槛：低于这个数的策略直接落到最低档（= 组合里风险最高那一档的权重）。
# 🔴 1.1 是用户定的（3.6.1 从 1.3 降下来）：1.0 只是不亏手续费，扣掉滑点/冲击
# 成本之后 1.1 以下基本没有安全边际，但 1.3 会误伤一批「刚过盈亏平衡、还在
# 爬坡」的策略 —— 工作区实测 5 条 1.04~1.26 全被压到最低档，太狠。
PF_FLOOR = 1.1

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

# ---- 一手保底（见 `lot_floor` / `apply_lot_floor`）----
#
# 分配到的本金如果历史上连 1 手都开不出来，这条策略的回测/实盘就是空跑：
# 净值一条直线、占用 0、波动率 0、盈利因子 0，然后就以「没数据」的身份被
# 配权规则再罚一次。用户要的是：把它的本金抬到「历史上至少能开 1 手」，
# 并且**不动别人的钱**，缺的部分直接加到总本金上。

# 判「开得出一手」用的手数门槛。
# 🔴 **0.5，不是 1.0**：`lots_round(x) = floor(x + 0.5)`，算出来 0.5 手就已经
# 下 1 手了（实测 0.49→0、0.50→1、0.80→1）。按 1.0 判等于把要求翻倍，
# 会把「明明开得出来」的策略也抬一遍（v3.6.3 就是这么错的，用户当场指出）。
LOT_MIN_LOTS = 0.5

# 判「开得出一手」用的价格档：按**历史最贵那一手**判。
# 这是「怎么后期开不出来仓位」的答案 —— 同一个品种价格翻 2~5 倍，同样的本金
# 后期手数就掉到 0.5 以下变 0 手。按均价判会漏掉这一段。
LOT_PRICE_PCT = 1.0

# 一手保底**每条最多往上加多少**：不超过组合总资金的这个比例。
# 存在的意义是「别让一条小额策略把总本金抬爆」—— 比如某条分到 1 万、而它
# 一手要 30 万，硬抬会让总资金翻倍。加到这个数还差就停下并写明「仍不足一手」，
# 让人知道这条实际是「钱不够」，而不是悄悄给它空账户。
# 🔴 卡的是「上调额」，不是「最终本金」（卡最终本金的话等权 4 条时 room 恒为 0，
# 一条都抬不动）。
LOT_LIFT_CAP_SHARE = 0.5

# 这几个口径**本身就有保底 1 手**，不会出现「钱不够就整段空跑」：
#   budget：`max(1, round(预算 ÷ 每手名义))`  —— 买不起也至少 1 手
#   risk  ：`max(1, round(risk_amount ÷ 每手风险))`
# 它们的资金不足表现为「实际杠杆/止损金额超出设定」，不是开不出仓，
# 所以不参与「一手保底」上调，只在说明列写清楚。
LOT_NO_LIFT_SIZINGS = ("budget", "risk")


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
    vol: float = 0.0              # 年化波动（策略净值，**含翻负段**，见 measure）
    ret: float = 0.0              # 累计收益
    max_dd: float = 0.0           # 最大回撤（正数）
    bars: int = 0                 # 回测里的交易笔数（老版本存的是 bar 数）
    # 整段资金曲线里权益是否曾经 <= 0（爆仓 / 亏穿）。
    # 🔴 它**不参与评分**：亏穿是杠杆的账，由 `util` 那一项去罚；这里只用来在
    # 预览表上说清楚「这条为什么占用高 / 波动大」。波动率本身照常是真值
    blown: bool = False
    # ---- 一手成本（「一手保底」用）----
    # 每手保证金 = 价格 × 合约乘数 × 保证金率，在**回测区间的行情上**取：
    #   `lot_margin_avg` = 按平均价算的一手要多少钱
    #   `lot_margin_peak` = 按区间最高价算的一手要多少钱
    # 同一个品种十年里价格翻几倍很常见，只按平均价分配就会在贵的那几年
    # 悄悄开不出仓 —— 所以两个都要，目标值取两者之间。
    # 0 = 没量到（没有行情 / 乘数缺失）
    lot_margin_avg: float = 0.0
    lot_margin_peak: float = 0.0
    # 历史上平均每笔开几手（从成交记录算）。<1 就说明「大部分时候开不出整手」。
    # 0 = 一次都没开过 —— 这时只能按口径解析式反推
    avg_lots: float = 0.0
    # 保证金率（`unit` 口径反推手数要用；其余口径用不到）
    margin_rate: float = 0.0
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
    # 资质档位的一句话结论（「回测2~3年 ×1.12」「盈利因子 1.05 < 1.1 → 最低档」）。
    # 单独一列，不和 `note` 混在一起 —— `note` 讲的是本金/百分比怎么调的
    tier: str = ""
    # 波动率无效（`vol` 是 NaN 或 0）。风险平价下这些条目会被补齐，权重不是
    # 由它自己的风险算出来的 —— 预览表必须点名，否则「为什么 27 条权重一样」
    # 没人解释得清
    vol_bad: bool = False
    # 整段曲线里权益曾经 <= 0（爆仓 / 亏穿）。波动率照常是真值，所以这一栏
    # 只用来解释「为什么它波动这么大、权重这么小」
    blown: bool = False
    # ---- 一手保底 ----
    # `lot_need`：让这条策略历史上至少开得出 1 手所需的本金（0 = 不需要/算不出）
    lot_need: float = 0.0
    # 真的被上调了没有 / 上调了多少（这部分钱是**额外加进总本金**的，
    # 不从别人身上扣）
    lifted: bool = False
    lift_extra: float = 0.0
    # 一手保底的说明：一手要多少钱、当前只够几手、上调到上限还差多少
    floor_note: str = ""
    note: str = ""


# ---------------------------------------------------------------- 测量
def _join(a: str, b: str) -> str:
    """把两段说明拼成「；」分隔的一句话（任一段为空就只留另一段）。"""
    a, b = (a or "").strip(), (b or "").strip()
    if not b:
        return a
    return f"{a}；{b}" if a else b


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
    # 波动率：**整条资金曲线**上算出来的真值，直接采用。
    #
    # 🔴 3.6.2 起 `_metrics` 不再因为「权益亏穿」把 `ann_vol` 标 NaN（见那边的
    # 注释）。理由有两条，都来自用户：① 波动率要覆盖整个过程资金曲线，亏穿了
    # 照样跑到底、照样量得出来；② 会不会亏穿是**杠杆**的事 —— 杠杆低就亏不穿，
    # 而杠杆高低已经由「资金利用率（`util`）」这一项进了分数。拿波动率再罚一次
    # 等于同一个原因罚两遍，还把最有信息量的那条数据（最颠的那条）扔掉了。
    #
    # 但仍然保留 NaN 分支：3.6.1 及以前存下来的老记录里 `ann_vol` 可能就是
    # NaN，而 `1/vol` 会把整张表的权重全变成 nan，那种只能按「没量到」补。
    try:
        vol = abs(float(met.get("ann_vol")))
    except (TypeError, ValueError):
        vol = float("nan")
    if not math.isfinite(vol):
        vol = float("nan")
        u.note = _join(u.note, "没有波动实测（老记录/没跑完），按组合典型风险配权")
    u.vol = vol
    # 「曾经亏穿」：只留痕，不参与评分。老记录没有这个键 ⇒ False
    try:
        u.blown = float(met.get("blown") or 0.0) > 0.5
    except (TypeError, ValueError):
        u.blown = False
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
    # ---- 一手成本（「一手保底」的原料，见 `lot_floor`）----
    u.margin_rate = float(margin_rate or 0.0)
    u.lot_margin_avg, u.lot_margin_peak = lot_cost(frame, mult, margin_rate)
    lots = [abs(float(getattr(t, "lots", 0) or 0))
            for t in (getattr(res, "trades", None) or [])]
    u.avg_lots = float(np.mean(lots)) if lots else 0.0
    if u.avg_margin <= 0:
        u.note = u.note or "回测里没有开过仓，量不出占用（只能按波动率配权）"
    return u


def lot_cost(frame: pd.DataFrame | None, mult: float = 1.0,
             margin_rate: float = 0.0) -> tuple[float, float]:
    """历史上「开一手」要占多少钱：(按均价, 按区间最高价)，单位元。

        每手保证金 = 价格 × 合约乘数 × 保证金率

    取**全区间**而不是「实际持仓那几根」：要回答的是「这个账户在任何时候
    能不能开得起一手」，那就得看整个区间里的价格。

    🔴 为什么必须区分平均和峰值：同一个品种十年里价格翻 2~4 倍很常见。
    只按均价分配，会在贵的那几年悄悄变成 0 手 —— 而回测里那几年恰好可能
    是行情最好的时候。峰值一手就是「最贵的时候也开得起」那条线。
    """
    if frame is None or not len(frame) or mult <= 0 or margin_rate <= 0:
        return 0.0, 0.0
    if "close" not in getattr(frame, "columns", ()):
        return 0.0, 0.0
    px = pd.to_numeric(frame["close"], errors="coerce").dropna()
    px = px[px > 0]
    if not len(px):
        return 0.0, 0.0
    per = float(mult) * float(margin_rate)
    return float(px.mean()) * per, float(px.max()) * per


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


def lot_floor(u: Usage, price_pct: float = LOT_PRICE_PCT,
              principal: float | None = None,
              min_lots: float = LOT_MIN_LOTS) -> tuple[float, str]:
    """让这条策略**历史上至少开得出 1 手**所需的本金。返回 `(需要的本金, 说明)`。

    `需要的本金 = 0` 表示「不用抬」或「算不出来」，`说明` 里写为什么。

    `principal`：**要判定的那个本金**，默认用实测时的本金。配权之后必须传
    「分到的那份」—— 因为「分到的钱比实测时少」恰恰就是开不出手的直接原因，
    拿实测本金去判会出现「分配只给了 1.4 万、实测时 5 万，于是判定够用」这种
    最要命的漏判（实测 23 条里有 4 条就是这样）。

    ## 判据（两处关键，都是被用户当场纠正过的）

    1. **门槛是 0.5 手，不是 1 手**。`lots_round(x) = floor(x + 0.5)`：
       算出来 0.5 手就已经下 1 手（0.49→0、0.50→1）。按 1.0 手判等于要求
       翻倍，会把「明明开得出来」的策略也抬一遍。
    2. **按历史最贵那一手判，不是均价**。同一个品种价格翻 2~5 倍，同样的
       本金后期手数就掉到 0.5 以下变 0 手 —— 这就是「前期能开、后期开不出来」。

        P_req = LOT_MIN_LOTS × L_peak ÷ k
        k    = 开仓预算 ÷ 本金（`capital`：占用%；`unit`：保证金率）

    即「最贵那一手的一半」要多少预算，再按占用比例折回本金。价格回到均价时
    算出来是 `0.5 × L_peak ÷ L_avg` 手 —— 实测工作区这几条在 0.8~1.05 之间，
    正好落在「一手均价的钱」附近，所以用户提的「取平均一手到峰值一手中间的
    资金」这个直觉是对的，只是更准确的写法是上面这条。

    ## 为什么只对 `capital` / `unit` 生效

    `budget`（网格按分配资金定手）和 `risk`（按止损金额定仓）在
    `backtest.size_lots` 里都有 `max(1, ...)` —— 买不起也至少 1 手，不会
    整段空跑。它们资金不足的代价是「实际杠杆 / 止损金额超出设定」，那是另一
    件事，抬本金治不了也不该用它治。`fixed` 手数是写死的，与本金无关。

    `capital` 是唯一会「钱不够一手 ⇒ 0 手 ⇒ 整段不交易」的口径，也正是用户
    反馈里那批「历史上根本开不了仓」的策略。
    """
    if not u_scalable(u.sizing):
        return 0.0, ""                      # fixed：手数写死，跟本金无关
    sz = str(u.sizing or "capital").lower()
    if sz in LOT_NO_LIFT_SIZINGS:
        return 0.0, f"{sz} 口径不足一手时按 1 手保底，不会空跑（代价是实际杠杆偏高）"
    p = float(u.principal if principal is None else principal)
    L_avg = float(getattr(u, "lot_margin_avg", 0.0) or 0.0)
    L_peak = float(getattr(u, "lot_margin_peak", 0.0) or 0.0)
    if L_avg <= 0 or L_peak <= 0 or p <= 0:
        return 0.0, "没有行情或合约乘数，算不出一手要多少钱"
    # 目标价格档：`price_pct=1` 直接取峰值；调小就放宽到均价附近
    t = max(0.0, min(1.0, float(price_pct)))
    L_req = L_avg + t * max(0.0, L_peak - L_avg)
    # k = 开仓预算 ÷ 本金
    if _pct_effective(sz):
        k = float(u.capital_pct or 0.0) / 100.0
        if k <= 0:
            return 0.0, "占用百分比是 0，本金再大也开不出仓"
        src = f"占用 {k:.0%}"
    elif sz == "unit":
        k = float(getattr(u, "margin_rate", 0.0) or 0.0)
        if k <= 0:
            return 0.0, "没有保证金率，算不出 unit 口径的手数"
        src = "名义敞口＝本金"
    else:
        return 0.0, f"{sz} 口径算不出杠杆，一手门槛无法估算"
    need = float(min_lots) * L_req / k
    if need <= p * (1.0 + 1e-9):
        return 0.0, ""                      # 按最贵那一手也开得出，不用抬
    can = p * k / L_req                     # 这份本金在目标价上算出来几手
    can_avg = p * k / L_avg                 # 按均价几手（给人对照）
    return need, (f"最贵一手 {L_peak:,.0f} 元（均价 {L_avg:,.0f}），按 0.5 手算需 "
                  f"{need:,.0f} 元本金；这份 {p:,.0f} 只够 {can:.2f} 手"
                  f"（均价下 {can_avg:.2f} 手，{src}）")


def apply_lot_floor(shares: list[Share], usages: list[Usage], total: float,
                    price_pct: float = LOT_PRICE_PCT,
                    cap_share: float = LOT_LIFT_CAP_SHARE,
                    min_lots: float = LOT_MIN_LOTS) -> dict:
    """把「历史上开不出 1 手」的条目本金抬到够 1 手，**不动其他条目的钱**。

    抬出来的差额不是从别人身上扣 —— 它会直接变成「总本金」的增加：
    调用方拿返回的 `extra` 去加总资金。用户明确要求这一点
    （「不挤压其余品种的权重分配，只是增加总本金」）。

    `cap_share` 是单条上限（占组合总资金的比例）。到了上限还差，就停下并在
    `floor_note` 里写「上调到 X 仍不足一手（需 Y）」—— 用户要的正是这句注释，
    否则这条策略会以「已经保底了」的样子继续空跑。

    返回 `{"before": Σ本金, "after": Σ本金, "extra": 差额, "n": 抬起条数,
    "short": [仍不足一手的名字]}`。
    """
    by = {str(getattr(u, "sid", "")): u for u in usages}
    before = float(sum(float(s.new_principal or 0) for s in shares))
    extra = 0.0
    n = 0
    short: list[str] = []
    for s in shares:
        u = by.get(str(s.sid))
        if u is None:
            continue
        # 🔴 必须按「**分到的**本金」判，不能按实测本金 —— 分配后金额变少正是
        # 开不出手的直接原因（实测 23 条里 4 条属于这种）
        need, why = lot_floor(u, price_pct, float(s.new_principal or 0), min_lots)
        s.lot_need = float(need or 0.0)
        s.floor_note = why
        if need <= 0:
            continue
        old = float(s.new_principal or 0)
        # 🔴 上限卡的是「**往上加多少**」，不是「最终本金多少」。卡最终本金的话，
        # 只要分到的那份就已经 ≥ 总资金×比例（等权 4 条时必然如此），room 直接
        # 是 0，一条都抬不动 —— 这个坑实测踩过一次。
        room = max(0.0, float(total) * max(0.0, float(cap_share)))
        if room <= 1e-9:                    # 一点空间都没有
            s.floor_note = _join(why, f"需要 {need:,.0f} 元，没有可上调的空间")
            short.append(s.name)
            continue
        give = old + min(max(0.0, float(need) - old), room)
        s.new_principal = give
        s.lifted = give > old + 1e-9
        s.lift_extra = max(0.0, give - old)
        if old > 0:
            # 手数 ∝ 本金：预计占用、总缩放跟着同一个比例走
            k = give / old
            s.est_avg *= k
            s.scale *= k
        extra += s.lift_extra
        n += 1
        if give < float(need) - 1e-6:
            s.floor_note = _join(
                why, f"已上调到上限 {give:,.0f} 元，仍不足一手（需 {need:,.0f}）")
            short.append(s.name)
    after = float(sum(float(s.new_principal or 0) for s in shares))
    return {"before": before, "after": after, "extra": after - before,
            "n": n, "short": short}


def risk_scores(usages: list[Usage]) -> tuple[np.ndarray, list[str]]:
    """每条策略的**标准化风险分数**（配权用 `1/分数`）+ 每条一句说明。

        分数 = 年化波动 ÷ 资金利用率 × 回测时长系数

    分数**越小越安全**，所以权重 ∝ `1/分数`。全部有限、全部为正 ——
    调用方可以放心取倒数，不会再出现 nan。

    ## 三处折算，各自的「未知」都不一样

    | 缺什么 | 怎么补 | 为什么这么补 |
    |---|---|---|
    | 利用率没量测（`util=0`） | 组合**中位数**利用率 | 不知道它压了多少，折成中位数＝不奖不罚。判 0 会把它当「几乎不用钱 ⇒ 单位风险爆炸」直接压死 |
    | 波动没量测（`vol=0`） | 有效组**几何平均**（典型风险） | 判 0 经 `1/vol` 会吃掉全部资金（实测 4 条没数据的拿走 100%） |

    > **波动率不再有「因为亏穿所以是 NaN」这一说**（3.6.2 改）。波动率按
    > **整条资金曲线**算，权益翻负那段照样计入 —— 亏没亏穿是**杠杆**的事，
    > 已经由 `util` 那一项除进去了；亏穿的策略波动率通常最大，如实给高分、
    > 拿小权重才对。`Usage.blown` 只用来在这条说明里点名，不参与打分。

    ## 「最低档」是什么

    盈利因子 < `PF_FLOOR`、回测不足半年 —— 这两种把分数抬到**有效组里最高**
    那一档（`s_top`），也就是拿到最小的那档权重。抬的时候取
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
    # 曾经亏穿（只用于说明，不参与打分、不降档）
    blown = np.array([bool(getattr(u, "blown", False)) for u in usages], dtype=bool)
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
    # 「净值被亏穿」**不降档**（3.6.1 按用户要求去掉），3.6.2 起波动率也不再因此
    # 置 NaN，所以亏穿的策略现在拿的是**自己真实的波动率**（含翻负段），分数会
    # 如实偏高、权重如实偏小 —— 正是应该的。
    # 亏穿为什么还不额外降档：它是**杠杆**的账，杠杆高低已经由 ③ 里的
    # `vol / util` 除进去了（util = 平均占用 ÷ 本金，`vol/util` 化简后就是
    # 「标的价波动 ÷ 保证金率」，压多少仓一目了然）。再拿它降一档等于同一个原因
    # 罚两遍。用户原话：「杠杆比率低，也亏不穿，这就是为什么要用资金利用率」。
    #
    # 剩下真正要补「组合典型风险」的只有一种：`vol` 没量到（老记录 / vol=0）。
    for i in range(n):
        if good[i]:
            if blown[i]:
                why[i] = _join(why[i], "净值曾亏穿（波动率含翻负段，按真值配权；"
                                        "杠杆由利用率体现）")
            continue
        s[i] = typical
        why[i] = _join(why[i], "没有波动实测，按组合典型风险")

    # ---- 最低档：抬到「有效组里最高风险」那一档 ----
    free = good & ~bottom
    pool = s[free] if free.any() else s[np.isfinite(s)]
    s_top = float(pool.max()) if len(pool) else typical
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
    `inv = 1/vol` 里只要混进一个 NaN，`inv.sum()` 就是 NaN，于是
    **27 条策略的权重全变 NaN**，界面上权重 / 目标占用 / 缩放 / 本金(后) /
    预计占用整列都是 nan。用户只会看到「算不出来」，看不出是其中**一条**
    策略干的。（3.6.2 之前的主要来源是「权益被亏穿 ⇒ `_metrics` 标 NaN」，
    现在那条路已经关掉了 —— 波动率按整条曲线算。这条防线仍然要留着，
    兜的是老记录和取不到行情的情况。）

    **② `ann_vol = 0` 比 NaN 更危险 —— 它不报错，直接吃掉全部资金。**
    没有回测实测的策略（取不到行情、因子缺数据、或者压根没点过「量占用」）
    走的是 `Usage` 默认值 `vol=0`。而 `max(vol, 1e-9)` 把 0 变成一个极小的
    正数，`1/1e-9 = 1e9` ⇒ 这些**最不该拿到钱**的策略权重趋近 100%。
    实测 23 条正常 + 4 条没数据：没数据的 4 条合计拿到 **100.00%**，正常的
    23 条拿到 0.00%，而界面上一切正常、不报错。

    ## 处理方式：唯一「没有可用波动」的情况是没量到，按组合典型风险补

    - **`vol = 0`（没量过：取不到行情 / 因子缺数据 / 没点「量占用」）**
      ⇒ 风险**未知**，用有效组的**几何平均**（组合典型风险）。既不放大到
      垄断，也不判 0（那等于「这条不用给钱」，同样没依据）。
    - **`vol = NaN`（老记录里 `_metrics` 标过 NaN，或根本没有 `ann_vol`）**
      ⇒ 同样按**组合典型风险**。

    **亏穿（`equity.min() <= 0`）不再是这里的一个分支。** 3.6.2 起波动率在
    整条资金曲线（含翻负段）上照算，所以亏穿的策略拿的是自己真实的、通常
    最大的波动率 ⇒ 分数本来就高 ⇒ 权重本来就小，不需要额外规则。
    亏穿该不该发生，由 `util`（资金利用率）和 `profit_factor` 两把尺子去判：
    用户原话是「杠杆比率低，也亏不穿，这就是为什么要用资金利用率」——
    杠杆已经进了分母，再拿亏穿降一档就是同一个原因罚两遍。
    `Usage.blown` 只负责在说明里点名「这条曾经亏穿」，让人知道权重小的原因。

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
    盈利因子 <1.1 或回测不足半年的落最低档。
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
             pct_cap: float = PCT_CAP,
             lift: bool = True,
             lift_price_pct: float = LOT_PRICE_PCT,
             lift_cap: float = LOT_LIFT_CAP_SHARE,
             lift_min_lots: float = LOT_MIN_LOTS) -> list[Share]:
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
    越大分得越少）、`pf`、`span`、`tier`（「回测2~3年 ×1.12」「盈利因子 1.05
    < 1.1 → 最低档」）。没有这几个字段，用户看到「波动 3% 却只分到 1.2%」时
    完全无从追问。

    ## 一手保底（`lift=True`，默认）

    分完之后，对 `capital` / `unit` 口径逐条检查「这点本金历史上开得出一手吗」，
    开不出的抬到够一手（口径见 `lot_floor`）。抬出来的差额**不从别人身上扣**，
    直接加在总本金上 —— 用户要求「不挤压其余品种的权重分配，只是增加总本金」。
    预览表的 `lift_extra` / `floor_note` 会写明抬了多少、以及抬到上限仍不够的
    情况；`Share.lifted` 为真的条目在界面上单独标色。

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
                  blown=bool(getattr(u, "blown", False)),
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
    # 一手保底：放在最后一步，因为它改的是**最终本金**。
    # 放在前面会被后面的缩放系数覆盖掉（缩放按 target 算，不看保底）。
    if lift and out:
        apply_lot_floor(out, usages, float(total), lift_price_pct, lift_cap,
                        lift_min_lots)
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
            # 一手保底：`一手门槛` 是「抬到够 1 手要多少本金」，`上调` 是实际
            # 加了多少钱（这部分是额外进总本金的，不从别人身上扣）
            "一手门槛": s.lot_need,
            "上调": s.lift_extra if s.lifted else 0.0,
            "一手保底说明": s.floor_note,
        })
    return pd.DataFrame(rows)
