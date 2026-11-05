"""因子策略：单因子 / 多因子 / 机器学习多因子，以及「规则策略 + 因子过滤」。

## 三类策略的共同点

都先算出一个**综合分数** `score`（已做 PIT 标准化），再用带迟滞的阈值
把分数转成 -1/0/1 的持仓信号：

    分数 > entry_z          → 开多
    分数回落到 exit_z 以下   → 平多
    分数 < -entry_z         → 开空
    分数回升到 -exit_z 以上  → 平空

`entry_z > exit_z` 是关键：没有迟滞的话，分数在阈值附近抖动会让策略每天
开平仓，手续费吃光收益。默认 1.0 / 0.3。

`exit_z` 可以取负（范围 -2 ~ 2）。取 0 是「只等分数转负就平」，取负则是
「要反向到足够强才平」——`exit_z = -1` 意味着做多之后分数要跌破 -1 才走，
中间那一段「分数已经转负但还没到 -1」照样拿着。想拉长持仓、少被噪音
扫出去，就把它往负里调。注意它只放宽**离场**，进场条件一个字没动。

## 方向为什么不能写死

用「动量高就做多」这种先验方向，在 A 股商品期货上会直接亏 —— 实测
CU/RB/M 三个品种的 60 日动量 IC 分别是 -0.021 / -0.039 / 和 120 日动量
-0.036 / -0.040 / -0.041，**全是负的**。所以方向必须由数据决定。

但「用全样本 IC 定方向」是未来函数：等于提前知道这个因子在整个样本里管用。
这里用**扩张窗口的累计 IC**，并且滞后 `horizon` 期 —— 因为算 IC 用的前向
收益 `fwd[t]` 要等到 `t+h` 才知道，t 时刻不能拿来用。这是整个模块里最容易
写错、也最隐蔽的一处未来函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import ttrade.factors as F
import ttrade.factor_metrics as M
import ttrade.mlmodels as ML
from ttrade.strategies import PARAM_LABELS, STRATEGIES, run_strategy


class Cancelled(Exception):
    """训练/预测过程中被用户取消。"""


def _is_degenerate_model(model, X_probe: np.ndarray) -> bool:
    """这个模型是不是「训废了」（只输出常数）。

    稀疏采样 + 小样本会让树模型训出**单叶树**：那棵树没有任何分裂，
    预测值恒等于训练集均值。特征重要度还在（树确实存在），所以从产物
    文件上看不出任何异常 —— 但实盘会炸：预测恒定 → 滚动 z 的标准差是 0
    → z 分数永远是 NaN → 报「z 分数算不出来（历史预测不够）」。

    这里拿几行真实特征探一下，看预测值是不是只有一个取值。

    `X_probe` 要给**多行**：只给一行的话，恒定模型和一棵刚好在这行上
    落同一个叶子的树看起来一样，判不出来。
    """
    try:
        Xp = np.asarray(X_probe, dtype=float)
        if Xp.ndim != 2 or Xp.shape[0] < 2:
            return False
        p = np.asarray(model.predict(Xp), dtype=float)
    except Exception:  # noqa: BLE001
        return False
    p = p[np.isfinite(p)]
    if len(p) < 2:
        return False
    # 相对标准差：预测值本身量纲很小（前向收益大概 1e-3），必须相对比较
    scale = float(np.max(np.abs(p)))
    if scale <= 0:
        return True
    return float(np.std(p)) <= 1e-9 * max(1.0, scale)


# ---------------------------------------------------------------- 配置
def _clean_tiers(tiers) -> list[list[float]]:
    """把「分档浮盈回撤」的参数规整成 [(浮盈 ATR, 回撤 ATR), …]。

    两条约束都在这里兜住，界面和配置文件都可能是脏的：

    - 浮盈门槛按升序排、去重。乱序的话「取第一个满足的档」会取到低档，
      高浮盈反而按宽松的回撤走
    - 回撤幅度必须随浮盈**递减**（浮盈越多、允许回撤越小）。写反了不会报错，
      只会让这条规则悄悄变成「浮盈越多越松」，也就是完全相反的东西
    """
    out: list[list[float]] = []
    for pair in list(tiers or []):
        try:
            p, t = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (p == p and t == t) or t <= 0:      # NaN / 非正回撤
            continue
        out.append([max(0.0, p), t])
    out.sort(key=lambda x: x[0])
    # 递减约束：从高浮盈那档往下压，保证低档的回撤不小于高档
    for i in range(len(out) - 2, -1, -1):
        if out[i][1] < out[i + 1][1]:
            out[i][1] = out[i + 1][1]
    return out or [[2.0, 2.0], [4.0, 1.2], [6.0, 0.8]]


@dataclass
class FactorConfig:
    # 因子构造
    window: int = 252            # 标准化窗口（也是 IC 统计窗口）
    method: str = "z"            # z / rank
    horizon: int = 5             # 预测/评估的前向收益期数

    # 阈值信号
    #
    # exit_z 允许为负：0 是「分数转负就平」，负值是「要反向到足够强才平」。
    # 想持仓长一点、别被一次回抽扫出去，就往负里调（下限 -2）。上限始终压在
    # entry_z 以内 —— 迟滞一旦倒挂，分数刚过进场线就同时满足出场条件，
    # 信号会在开平之间每根抖一次。
    entry_z: float = 1.0
    exit_z: float = 0.3
    orient: str = "auto"         # auto（PIT 滚动 IC 定向）/ long_high / long_low

    # 因子反转：{因子键: True} 表示这个因子按反方向使用（等效于把该因子取负号）。
    # 多因子组合里常有「有的因子要反、有的不用反」，所以是逐因子开关而不是全局开关。
    # 只作用于单因子 / 多因子；机器学习自己学方向，不看这个。
    reverse: dict = field(default_factory=dict)

    # 机器学习
    model: str = ML.DEFAULT_MODEL
    # train_window / step 按**交易日**算，不是按 bar 根数：同样的 1000 天，日线
    # 是 1000 根、1h 约 6900 根、5m 约 68000 根。参数的意义是「用过去几年训练」，
    # 跟行情周期无关，具体换算见 `_to_bars()`。
    #
    # min_train 不一样，它是**根数**：问的是「多少个样本才够训一个模型」，这是
    # 统计问题，跟多久的历史无关。换算成交易日反而会在分钟级上把门槛抬到数据
    # 长度都够不着（5m 的 500 天是 3.4 万根），整段不出信号。
    train_window: int = 1000     # 滚动训练窗口（交易日）
    step: int = 180              # 每隔多少交易日重训一次（180≈9 个月）
    min_train: int = 500         # 首次训练所需最少样本（根）

    # 因子过滤
    filter_mode: str = "entry"   # entry（否决整笔）/ delay（延后入场）/ hold（转差离场）
    long_th: float = 0.0
    short_th: float = 0.0

    # ---- 样本稀疏性（只对 ml 生效）
    #
    # 分钟线上每根 bar 都是一个样本的话，相邻样本的标签重叠 horizon 根，
    # 等于同一段行情被数了 horizon 遍 —— 模型学到的是「这几天在涨」而不是
    # 「这种形态之后会涨」。而且 5 分钟一年 6 万根，绝大多数是横盘噪声，
    # 真正值得学的「突破那一刻」只占几个百分点。
    #
    # `trigger` 模式只在**信号触发**的 bar 上建样本：先用一组因子（或一条规则）
    # 判断「现在值不值得学」，再按触发强度分档 —— 强突破和弱突破是两回事，
    # 分成不同档各自成样本，标签才够丰富（趋势和反转都能覆盖到）。
    #
    # 触发因子**不需要**和特征因子重合，也不要求它们出现在 `keys` 里：
    # 它们只负责「决定在哪几根建样本」，训练时用到的仍是 `keys` 全部因子。
    sample_mode: str = "all"        # all / trigger
    trigger_keys: list = field(default_factory=list)
    trigger_rule: str = ""          # 规则策略 key（`STRATEGIES` 里那个）
    trigger_params: dict = field(default_factory=dict)
    trigger_side: str = "both"      # up / down / both
    trigger_z: float = 1.0          # 触发阈值（触发强度的绝对值）
    trigger_tiers: int = 0          # 分档数：0/1 = 不分档，只取触发点
    trigger_span: float = 1.0       # 每档多宽（强度单位）
    trigger_cool: int = 5           # 同一档的冷却根数

    # 触发信号反转：把「触发」这件事本身反过来用。因子那侧是**强度取负号**
    # （向上突破变成向下破位，样本方向、进场方向、出场判据全跟着翻），规则那侧是
    # **signal 取负**（规则本来做多的位置做空）。两条通道各管各的，互不影响。
    #
    # 反的是「在哪儿学、往哪边做」，不是「涨跌」的定义：标签仍是前向收益，
    # 模型学到的仍然是「这种形态之后会涨」。所以「因子反转」不等于把策略做成
    # 反手策略 —— 它是「这个因子我读反了」（动量因子在震荡市里突破即回落）。
    trigger_flip: bool = False      # 反转因子触发
    trigger_rule_flip: bool = False  # 反转规则触发

    # 多品种合并训练：把 N 个品种的样本堆成一张长表训一个模型，再各自交易。
    # 回测里界面上有个勾选框（`chk_ml_pool`），以前**没存进配置** —— 导致
    # 回测走合训、离线训练和实盘走逐品种，三者不一致。现在存进来，训练脚本
    # 和实盘就能和回测对齐。
    pooled: bool = False

    # ---- 多因子组合的权重（只对 multi 生效）
    #
    #   equal ：等权平均。老口径，也是默认 —— 权重这东西拍脑袋定，
    #           效果往往不如不设，先让人从等权起步
    #   custom：人为权重。界面上每个因子填一个数，按权重加权平均；
    #           某根缺值时按当根**有值的那些权重**重新归一，不会因为
    #           一个因子缺数就把整行分数拉低
    #   ic    ：滚动 IC 加权。每 `ic_rebal` 根，用过去一个窗口的 IC 绝对值
    #           当权重 —— 方向仍由 `orient` 决定，所以这里只取强度不取符号，
    #           不然「自动定向已经翻过一次号」再乘一次负权重，等于反两次
    weight_mode: str = "equal"       # equal / custom / ic
    weights: dict = field(default_factory=dict)   # 因子键 -> 权重（custom 用）
    ic_rebal: int = 60               # 滚动 IC 加权：多少根重算一次权重

    # ---- 跨周期因子：{因子键: 周期}
    #
    # 空 / 缺键 = 跟着策略本身的周期走（老口径）。填了别的周期，这个因子就
    # 在**那个周期**的行情上算，再对齐回基准周期。
    #
    # 为什么要单独一项：周线策略想用日线因子（「大方向按周线，具体动手看日线
    # 有没有走坏」），以前只能把策略整个降到日线，周线的过滤就没了。
    #
    # 对齐的**截断**规则见 `align_to_base`：大周期的 bar 只有在它收完之后
    # 才允许被小周期看到。写错这一处就等于在一周的开头用了周末的收盘价 ——
    # 回测漂亮得不正常，而且一个字都不会报错。
    factor_periods: dict = field(default_factory=dict)

    # ---- 出场规则（只对因子策略生效，覆盖「分数回落就平」）
    #
    #   score：老口径，分数回落到 `exit_z` 以内就平
    #   tier ：分档浮盈回撤。浮盈和回撤都按 ATR 折算，分 3 档 —— 浮盈越多，
    #          允许的回撤越小（先让利润跑，跑起来之后收紧）
    #   abs  ：绝对回撤。从持仓期间的最高浮盈回撤固定 N×ATR 就平，不分档
    #   box  ：反向箱体突破。持仓期间价格反向穿越「过去 N 根的高低点」就平
    #
    # 这几种都是**只减仓不反手**：出场之后要等信号自己回到 0 再重新给方向，
    # 否则一根大阴线平多、紧接着反手做空，那是另一条策略（趋势反手），
    # 不该藏在「出场规则」里偷偷发生。
    #
    # 另外还有一条**独立**的止损（`exit_sl`）：它和上面四种口径是**与**的关系，
    # 不是第五个选项。选了分档回撤也能同时挂止损 —— 分档管的是「赚了之后怎么
    # 收」，止损管的是「一进场就错，亏到哪认输」，两件事不该二选一。
    # 口径为 `score` 时它也一样生效：那时它是唯一一条 ATR 规则。
    exit_mode: str = "score"
    # 三档：(浮盈达到几个 ATR, 从最高点回撤几个 ATR 就平)。必须递减，
    # 否则「浮盈越多回撤越少」这句话就是假的
    exit_tiers: list = field(default_factory=lambda: [[2.0, 2.0], [4.0, 1.2], [6.0, 0.8]])
    exit_abs_atr: float = 2.0        # 绝对回撤：回撤几个 ATR
    exit_box_bars: int = 20          # 反向箱体：箱体回看根数（决定箱体宽度）
    exit_box_buf: float = 0.0        # 反向箱体：边界缓冲（ATR 倍数），0 = 触到就走
    exit_atr_n: int = 14             # 算 ATR 的根数

    # ---- 独立止损（和上面四种口径**叠加**，不是二选一）
    #
    # 从**进场价**往回走 N×ATR 就平仓。它不看浮盈最高点，所以和「绝对回撤」
    # 有区别：绝对回撤是「从最好浮盈回撤 N×ATR」，浮盈到过 +5ATR 之后，回撤
    # 到 +3ATR 就平了，永远不会亏；止损是「从进场价亏 N×ATR」，浮盈再高也
    # 不改变这条线的位置。两件事，一个管收利润、一个管认亏。
    exit_sl: bool = False
    exit_sl_atr: float = 2.0

    # ---- 禁止分数出场（离场阈值完全不参与）
    #
    # 勾了之后，分数回落到 `exit_z` 以内**不平仓**，仓位只由上面那些 ATR 规则
    # 了结。和止损一样是独立开关，不是第五种口径 —— 它管的是「要不要让离场
    # 阈值参与」，不是「用哪种 ATR 算法」。
    #
    # 进场条件一个字不动（分数仍要越过 `entry_z` 才开仓）；分数**反向**越过
    # 进场阈值时仍然平仓 —— 那是反向信号，不是离场阈值。拿着反方向的仓等
    # ATR 触发，等于让一条趋势策略硬扛整个反向趋势。
    exit_no_score: bool = False

    # ---- 分数反向也不平仓（比上面那条更彻底）
    #
    # 勾了之后，持仓期间分数**反向越过进场阈值**（+1 变 -1 这种）也不平仓，
    # 仓位只由 ATR 规则了结。它包含 `exit_no_score`：反向都不平，回落当然
    # 也不平（`normalized()` 里会顺手把那条也打开）。
    #
    # 用途是「有浮盈后尽量不出场」这一类：胜率会降（回吐的次数变多），换来
    # 更大的盈亏比 —— 截断亏损、让利润奔跑。顺带也过滤掉一批噪音反向信号：
    # 分数在正负之间来回穿，只要 ATR 没到，仓位就不动。
    #
    # 代价要清楚：**进场方向只由第一笔决定**，之后分数翻向不再换仓。一路
    # 反向的趋势里，仓位靠 ATR 止损退出，而不是靠分数。
    exit_no_flip: bool = False

    def reversed_keys(self) -> set[str]:
        """真正打开反转的因子。顺手把非真值过滤掉，调用方不用自己判。"""
        return {k for k, v in dict(self.reverse or {}).items() if v}

    def period_of(self, key: str, base: str = "") -> str:
        """这个因子实例该在哪个周期上算。

        `key` 可以是实例 id（`mom_60@1w`），也可以是老配置里的纯键 —— 后者
        去 `factor_periods` 里查。`base` 是策略本身的周期；返回空串表示
        「没有指定」，调用方自己知道基准是什么，这里不替它猜。
        """
        _k, p = F.split_spec(key)
        if p:
            return p
        p = str(dict(self.factor_periods or {}).get(str(key), "") or "")
        return p or str(base or "")

    def is_reversed(self, key: str) -> bool:
        """这个实例要不要反着用。

        先按实例 id 查，再退回纯键 —— 界面上「反」是按因子行勾的（一行里的
        几个周期一起反），老配置里存的也是纯键。两处都认，才不会出现
        「勾了反转但周线那一份没反」。
        """
        rv = self.reversed_keys()
        return key in rv or F.split_spec(key)[0] in rv

    def weight_of(self, key: str) -> float:
        """这个实例在「自定权重」下的权重。同样先按实例查、再退回纯键。"""
        w = dict(self.weights or {})
        k, _p = F.split_spec(key)
        for probe in (key, k):
            try:
                v = float(w.get(probe))
            except (TypeError, ValueError):
                continue
            if v > 0:
                return v
        return 1.0

    def normalized(self) -> "FactorConfig":
        c = FactorConfig(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        # dict 是可变对象，复制一份，免得调用方事后改原配置影响已经算好的结果
        c.reverse = {k: True for k in c.reversed_keys()}
        c.window = int(max(60, c.window))
        c.horizon = int(max(1, c.horizon))
        c.entry_z = float(max(0.0, c.entry_z))
        # 离场阈值下限 -2：负值是「反向到足够强才平」，持仓更长。上限仍压在
        # entry_z 以内 —— 迟滞倒挂的话，分数刚越过进场线就同时满足出场条件，
        # 信号每根都在开平之间抖，手续费吃光收益
        c.exit_z = float(min(max(-2.0, c.exit_z), c.entry_z))
        c.train_window = int(max(200, c.train_window))
        c.step = int(max(20, c.step))
        c.min_train = int(max(120, c.min_train))
        c.trigger_keys = [str(k) for k in (c.trigger_keys or [])]
        c.trigger_tiers = int(max(0, c.trigger_tiers))
        c.trigger_span = float(max(1e-6, c.trigger_span))
        c.trigger_cool = int(max(0, c.trigger_cool))
        c.trigger_z = float(c.trigger_z)
        c.trigger_flip = bool(c.trigger_flip)
        c.trigger_rule_flip = bool(c.trigger_rule_flip)
        # 权重：只认正数。负权重等于「这个因子反着用」，那件事该由「反转」
        # 勾选框表达 —— 两个地方都能翻方向，出问题时根本查不出是谁翻的
        c.weight_mode = str(c.weight_mode or "equal")
        if c.weight_mode not in ("equal", "custom", "ic"):
            c.weight_mode = "equal"
        c.weights = {str(k): float(v) for k, v in dict(c.weights or {}).items()
                     if float(v) > 0}
        c.ic_rebal = int(max(1, c.ic_rebal))
        # 跨周期：只留非空值。空串等于「跟基准周期走」，存进去只会让
        # 「这个因子到底是不是跨周期」这件事变得要多判一次
        c.factor_periods = {str(k): str(v) for k, v in dict(c.factor_periods or {}).items()
                            if str(v or "").strip()}
        # 出场规则
        c.exit_mode = str(c.exit_mode or "score")
        if c.exit_mode not in ("score", "tier", "abs", "box"):
            c.exit_mode = "score"
        c.exit_tiers = _clean_tiers(c.exit_tiers)
        c.exit_abs_atr = float(max(0.1, c.exit_abs_atr))
        c.exit_box_bars = int(max(2, c.exit_box_bars))
        c.exit_box_buf = float(max(0.0, c.exit_box_buf))
        c.exit_atr_n = int(max(2, c.exit_atr_n))
        # 止损：0 或负数等于关掉（界面上勾了但填 0 也算关，免得写出一条
        # 「进场即平仓」的规则）
        c.exit_sl = bool(c.exit_sl) and float(c.exit_sl_atr or 0) > 0
        c.exit_sl_atr = float(max(0.1, c.exit_sl_atr))
        # 禁止分数出场 + 口径 score + 没开止损 = 一条出场规则都没有，仓位会
        # 一直拿到回测结束。这种组合没有意义，直接当没勾（界面上也会拦住）
        c.exit_no_flip = bool(c.exit_no_flip)
        # 反向都不平，回落自然也不平：两条勾选框在界面上有包含关系，
        # 配置层面也按同一套语义归一，免得程序调用时出现「no_flip 但还在
        # 按离场阈值平仓」这种自相矛盾的组合
        c.exit_no_score = bool(c.exit_no_score) or c.exit_no_flip
        c.exit_no_score = c.exit_no_score and (c.exit_mode != "score" or c.exit_sl)
        c.exit_no_flip = c.exit_no_flip and c.exit_no_score
        return c


@dataclass
class TriggerPlan:
    """触发模式下的决策素材：在哪几根做决策、方向从哪来、出场看什么。

    - `mask`：**决策点**，也就是建样本的那几根 bar（稀疏采样挑出来的）
    - `strength`：触发强度，带符号、**密集**（每根都有值），出场靠它
    - `armed`：规则刚开仓 / 刚翻向的那几根
    - `rule_sig`：规则自己的持仓（-1 / 0 / 1），规则通道的出场靠它
    - `factor_channel`：有没有因子触发通道。没有就只能靠规则，强度那条路是空的

    单独打包是因为「什么时候允许开仓、什么时候必须平仓」和「分数怎么算」是两套
    东西，而这两套要的东西（强度、规则信号）都在 `build_score` 里已经算过一遍，
    不带走就得重算。
    """

    mask: np.ndarray
    strength: np.ndarray
    armed: np.ndarray
    rule_sig: np.ndarray
    factor_channel: bool = True
    rule_channel: bool = False


@dataclass
class ScorePack:
    """综合分数的来源与中间产物，供界面画图/出表。"""

    score: pd.Series
    fmat: pd.DataFrame
    raw: pd.DataFrame
    pred: pd.Series | None = None
    importance: pd.Series | None = None
    orient: pd.DataFrame | None = None
    folds: int = 0
    notes: list[str] = field(default_factory=list)
    label: str = ""
    # 触发模式（稀疏采样）才有：决策点 / 触发强度 / 规则信号，交给状态机用
    trigger: TriggerPlan | None = None


@dataclass
class FactorOutcome:
    df: pd.DataFrame
    pack: ScorePack
    label: str = ""
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- PIT 定向
def pit_orientation(
    factor: pd.Series,
    fwd: pd.Series,
    horizon: int,
    min_obs: int = 250,
) -> pd.Series:
    """用扩张窗口的累计 IC 决定因子方向（+1 高分做多 / -1 高分做空）。

    **必须滞后 horizon 期**：`fwd[t]` 是 t+1 开盘到 t+h 收盘的收益，t 时刻
    还没发生。不滞后就等于用未来收益决定今天的方向，回测会好得离谱。
    """
    ic = factor.expanding(min_periods=int(min_obs)).corr(fwd)
    ic = ic.shift(int(horizon))
    return np.sign(ic.fillna(0.0)).replace(0.0, 1.0).astype(float)


def _orientation_frame(
    fmat: pd.DataFrame, fwd: pd.Series, cfg: FactorConfig
) -> pd.DataFrame:
    """每个因子在每个时点的方向。`orient` 配置可覆盖为固定方向。

    反转在这里生效，而不是在最后对综合分数取负号：多因子组合时，只反一个因子
    必须只把那一列取反；整体取负会把其它因子也一起反掉。
    """
    if cfg.orient == "long_high":
        ori = pd.DataFrame(1.0, index=fmat.index, columns=fmat.columns)
    elif cfg.orient == "long_low":
        ori = pd.DataFrame(-1.0, index=fmat.index, columns=fmat.columns)
    else:
        ori = pd.DataFrame(
            {
                c: pit_orientation(fmat[c], fwd, cfg.horizon, max(120, cfg.window // 2))
                for c in fmat.columns
            },
            index=fmat.index,
        )
    # 列名是因子实例 id（`mom_60@1w`），`reversed_keys()` 里可能是纯键也可能
    # 是实例 id —— 交给 `is_reversed` 两个都认
    rev = [c for c in ori.columns if cfg.is_reversed(c)]
    if rev:
        ori = ori.copy()
        for c in rev:
            ori[c] = -ori[c]
    return ori


def _reverse_note(cfg: FactorConfig, keys: list[str]) -> str:
    """反转说明。写成「反转使用：A、B」，界面上直接看得出哪几个因子被反着用。"""
    rev = [k for k in keys if cfg.is_reversed(k)]
    if not rev:
        return ""
    names = "、".join(F.spec_label(k) for k in rev)
    return f"反转使用：{names}（等效于把该因子取负号）"


def _warmup_note(score: pd.Series | None, n_bars: int, cfg: FactorConfig) -> str:
    """分数整列 NaN 时说清原因，别让人以为策略坏了。

    标准化窗口是**根**，不是「天」。同一个 252 在 5 分钟上只占两天，在周线上
    是五年 —— 周线策略选「近 1 年」只有五十来根，分数全空、信号全 0，界面上
    就是一条不交易的策略，看不出是样本不够。
    """
    if score is None or len(score) == 0 or int(score.notna().sum()) > 0:
        return ""
    return (f"分数全空：这段行情只有 {int(n_bars)} 根，而标准化窗口要 "
            f"{int(cfg.window)} 根 —— 把回测区间拉长，或者把窗口调小")


def _orient_note(ori: pd.DataFrame | None, cfg: FactorConfig) -> str:
    """自动定向时报出**末期实际生效**的方向。

    这一行是给「为什么 ZN 和 C 上反转结果相反」这类困惑用的：自动定向的符号
    每个品种不一样、同一品种还会中途翻，光看「反转」两个字完全看不出最后落在
    哪个绝对方向上。`ori` 里已经是反转之后的值，所以这里报的就是当前真实口径。
    """
    if cfg.orient != "auto" or ori is None or ori.empty:
        return ""
    tail = ori.dropna(how="all")
    if tail.empty:
        return ""
    last = tail.iloc[-1]
    up = int((last > 0).sum())
    down = int((last < 0).sum())
    if len(last) == 1:
        return "末期方向：" + ("高分做多（+1）" if up else "高分做空（−1）")
    return f"末期方向：{up} 个高分做多 / {down} 个高分做空"


# ---------------------------------------------------------------- 多因子权重
def _weighted_mean(mat: pd.DataFrame, w) -> pd.Series:
    """按权重求每行加权平均，缺值跳过、**当根重新归一**。

    `w` 可以是 Series（每个因子一个权重），也可以是与 `mat` 同形的 DataFrame
    （权重逐根变，滚动 IC 加权用）。

    缺值必须按「当根有值的那些权重」归一，不能直接用 `w.sum()`：那样
    「A 因子这根没数」会被算成「A 因子给了个 0 分」，分数被系统性拉向 0 ——
    因子预热期和展期 / 期权这类稀疏因子正好全落在这种情形里。
    """
    if isinstance(w, pd.DataFrame):
        ww = w.reindex(index=mat.index, columns=mat.columns).astype(float)
    else:
        row = w.reindex(mat.columns).fillna(0.0).to_numpy(dtype=float)
        ww = pd.DataFrame(np.tile(row, (len(mat), 1)), index=mat.index,
                          columns=mat.columns)
    wm = ww.where(mat.notna())
    den = wm.sum(axis=1, skipna=True)
    num = (mat.fillna(0.0) * wm.fillna(0.0)).sum(axis=1)
    return (num / den.where(den > 0)).rename("score")


def rolling_ic_weights(fmat: pd.DataFrame, fwd: pd.Series, cfg: FactorConfig
                       ) -> pd.DataFrame:
    """滚动 IC 权重：每 `ic_rebal` 根，用过去一个窗口的 |IC| 当权重。

    **必须滞后 horizon**：`fwd[t]` 是 t+1 开盘到 t+h 收盘的收益，t 时刻还没发生。
    不滞后就等于用未来收益给今天配权重，回测会好得离谱 —— 和 `pit_orientation`
    是同一个坑，那边已经踩过一次。

    取绝对值不取符号：方向归 `orient` 管（自动定向已经按 IC 符号翻过号），
    这里再乘一次符号等于反两次，会变成「越没用的因子给越多权重」。

    权重不是逐根变的：每 `ic_rebal` 根更新一次，中间沿用上一次的值。IC 的日间
    噪声本来就大，每根都跟着抖会把换手率顶起来。
    """
    win = int(max(20, cfg.window))
    mp = max(20, win // 4)
    cols = {}
    for c in fmat.columns:
        ic = fmat[c].rolling(win, min_periods=mp).corr(fwd).shift(int(cfg.horizon))
        cols[c] = ic.abs()
    w = pd.DataFrame(cols, index=fmat.index).astype(float)
    step = int(max(1, cfg.ic_rebal))
    grid = pd.Series(False, index=w.index)
    grid.iloc[::step] = True
    w = w.where(grid).ffill()
    # 一个因子都没 IC（预热期，或者这批因子在这段行情里全不灵）时退回等权。
    # 权重全 0 会让分数整段 NaN，界面上就是「策略没信号」，查起来毫无线索
    dead = ~(w.sum(axis=1, skipna=True) > 0)
    if bool(dead.any()):
        w.loc[dead, :] = 1.0
    return w


# ---------------------------------------------------------------- 阈值状态机
def hysteresis(
    score: np.ndarray, entry: float, exit_: float
) -> np.ndarray:
    """带迟滞的阈值状态机。NaN 分数一律强制空仓（预热期不交易）。

    `exit_` 为负时（比如 -1.0），做多要等分数跌破 -1 才平、做空要等分数涨过
    +1 才平 —— 中间那段「方向已经不对、但强度还不够」照样拿着。所以同一个
    状态机既支持「分数转负就走」（`exit_=0`），也支持「反向坐实了才走」
    （`exit_<0`），不需要另写一套。
    """
    s = np.asarray(score, dtype=float)
    n = len(s)
    pos = np.zeros(n, dtype=int)
    cur = 0
    e, x = float(entry), float(exit_)
    for i in range(n):
        v = s[i]
        if not np.isfinite(v):
            cur = 0
            pos[i] = 0
            continue
        if cur == 1:
            if v < x:
                cur = 0
        elif cur == -1:
            if v > -x:
                cur = 0
        if cur == 0:
            if v > e:
                cur = 1
            elif v < -e:
                cur = -1
        pos[i] = cur
    return pos


def trigger_hysteresis(
    score: np.ndarray,
    plan: TriggerPlan,
    entry: float,
    exit_: float,
) -> np.ndarray:
    """触发模式（稀疏采样）的持仓状态机。

    和 `hysteresis` 的区别是**谁决定什么**：

    - **进场**：只在触发点（`plan.mask`）。方向由触发给（向上突破做多、向下破位
      做空），模型只负责「做不做」—— 分数在**触发方向那一侧**过进场阈值才进。
      模型不认同这个方向（向上突破但模型看空）就不做，也不反手：那是另一条策略。
    - **出场**：每根都判，看的是**触发强度**回落到离场阈值以内，不是分数。分数
      在触发点之间是阶梯常值，拿它出场等于永远等不到「跌破」，仓位会一路拖到
      下一次触发。
    - **规则通道**：规则只给「刚开仓 / 刚翻向」，没有强度，所以它的仓位按规则
      自己的 signal 出场（变 0 或翻向就平）。**哪条通道开的仓，就由哪条通道负责
      平** —— 混着用一个口径，总有一条是错的。

    强度缺失（因子那几根没值）时**不动**：没有信息就不要凭空白砍一刀。

    两条通道在同一根同时说话、方向却相反时**不做**。挑一条（比如「因子优先」）
    会做出「规则刚翻空、仓位却是多」这种谁也解释不清的仓位。
    """
    sc = np.asarray(score, dtype=float)
    s_v = np.asarray(plan.strength, dtype=float)
    r_v = np.asarray(plan.rule_sig, dtype=float)
    m_v = np.asarray(plan.mask, dtype=bool)
    a_v = np.asarray(plan.armed, dtype=bool)
    n = min(len(sc), len(s_v), len(r_v), len(m_v), len(a_v))
    pos = np.zeros(len(sc), dtype=int)
    cur = 0
    src = ""                      # 当前仓位是哪条通道开的：factor / rule

    for i in range(n):
        # ---- 出场：每根都判 ----
        if cur == 1:
            if src == "factor":
                if np.isfinite(s_v[i]) and s_v[i] < exit_:
                    cur = 0
            elif not (np.isfinite(r_v[i]) and r_v[i] > 0):
                cur = 0
        elif cur == -1:
            if src == "factor":
                if np.isfinite(s_v[i]) and s_v[i] > -exit_:
                    cur = 0
            elif not (np.isfinite(r_v[i]) and r_v[i] < 0):
                cur = 0

        # ---- 进场：只在触发点 ----
        if cur == 0 and m_v[i]:
            # 两条通道各自给一个方向。同时说话但方向相反 = 信号矛盾，这一根不做 ——
            # 悄悄挑一条（比如「因子优先」）会让图上出现「规则刚翻空、仓位却是多」
            # 这种谁也解释不清的仓位
            f_dir = (1 if s_v[i] > 0 else -1) if (
                plan.factor_channel and np.isfinite(s_v[i]) and s_v[i] != 0) else 0
            r_dir = (1 if r_v[i] > 0 else -1) if (
                a_v[i] and np.isfinite(r_v[i]) and r_v[i] != 0) else 0
            d, ch = 0, ""
            if f_dir and r_dir and f_dir != r_dir:
                d, ch = 0, ""
            elif f_dir:
                d, ch = f_dir, "factor"
            elif r_dir:
                d, ch = r_dir, "rule"
            v = sc[i]
            if d and np.isfinite(v) and (v > entry if d == 1 else v < -entry):
                cur, src = d, ch
        pos[i] = cur
    return pos


def _atr_array(df: pd.DataFrame, n: int) -> np.ndarray:
    """真实波幅均值（ATR）整段序列。口径和 `backtest._atr_array` 一致。

    出场规则里所有「几个 ATR」都按这个算：`exit_atr_n` 默认 14，和回测里
    「按止损金额定仓」用的那个 14 是同一个数 —— 两处用不同 ATR 的话，
    「止损 5×ATR」和「回撤 2×ATR」在图上对不上，没法解释。
    """
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    high = (pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
            if "high" in df.columns else close)
    low = (pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
           if "low" in df.columns else close)
    n = max(2, int(n or 14))
    prev = np.concatenate(([np.nan], close[:-1]))
    tr = np.maximum(np.maximum(high - low, np.abs(high - prev)), np.abs(low - prev))
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy(dtype=float)


class ExitTracker:
    """ATR 出场规则的状态机。**回测逐根推进、实盘每根推一次，共用这一份。**

    出场规则必须是同一份实现，否则「回测里 2.3 年平仓、实盘一直拿着」这种
    差异没有任何线索可查 —— 两条路各写一遍，迟早会分叉。

    三种规则都是**只减仓、不反手**：触发之后强制空仓，等信号自己回到 0
    （或者自己翻到反方向 = 一笔新交易）才重新放行。理由很实际 ——
    一根大阴线平多、紧接着反手做空，那是「趋势反手」策略，不该藏在
    「出场规则」这个开关里悄悄发生。

    浮盈的起点是**信号生效那根的收盘价**（不是下一根开盘）。回测的真实成交
    在下一根开盘，两者差一个跳空，但规则必须是「只看收盘就能判」的 ——
    实盘那一刻还不知道下一根开在哪。宁可定义得朴素一点，也要两边算得一样。

    止损（`exit_sl`）是**独立**的一条：它不看浮盈最高点，只看「离进场价亏了
    几个 ATR」，所以和上面三种口径是叠加关系而不是并列关系。分档回撤管的是
    「赚了之后怎么收」，止损管的是「一进场就错，亏到哪认输」。

    ATR 缺失（预热期前 n 根）时不动：没有尺子就不要凭空白砍一刀。
    """

    def __init__(self, cfg: FactorConfig):
        self.cfg = cfg
        self.mode = str(cfg.exit_mode or "score")
        self.tiers = _clean_tiers(cfg.exit_tiers)
        self.bars = int(max(2, cfg.exit_box_bars))
        self.buf = float(max(0.0, cfg.exit_box_buf))
        # 0 = 没开止损。`normalized()` 已经把「勾了但填 0」折算成关掉
        self.sl = float(cfg.exit_sl_atr or 0.0) if cfg.exit_sl else 0.0
        # 离场阈值完全不参与：分数回落到 exit_z 以内不平仓
        self.no_score = bool(cfg.exit_no_score)
        # 更彻底一层：分数反向越过进场阈值也不平仓（隐含 no_score）
        self.no_flip = bool(cfg.exit_no_flip)
        # 上一次被出场规则平掉时的方向。分数翻向要算「另一笔交易」，
        # 没有这个值的话，no_flip 下分数一路反向、仓位会永久空着
        self.exit_dir = 0
        self.pos = 0
        self.entry = 0.0
        self.best = 0.0        # 持仓期间的最好浮盈（ATR 倍数）
        self.blocked = False   # 已被出场规则平掉，等信号归零 / 翻向再放行
        self.n_exit = 0
        self.n_sl = 0          # 其中由止损打掉几次（单独记，摘要里要说清）
        # 上一根**为什么**平仓（没平就是空串）。`apply_exit_rules` 把它逐根
        # 收下来写进 `exit_note` 列，交易记录页的备注列直接读 —— 只给一句
        # 「分档浮盈回撤」，用户没法判断这次是该平还是被噪音扫掉的
        self.last = ""

    def active(self) -> bool:
        """这条配置会不会真的改动信号。"""
        return self.mode != "score" or self.sl > 0 or self.no_score

    def step(self, want: int, px: float, atr: float,
             box_low: float = float("nan"), box_high: float = float("nan"),
             fresh: bool = False) -> int:
        """推进一根，返回这根**实际**该持有的仓位（-1/0/1）。

        `want` 是阈值信号说的仓位；返回值是叠加出场规则之后的仓位。
        `box_low` / `box_high` 是箱体边界（调用方按「过去 N 根、不含当根」算好）。
        `fresh` 表示这根**分数刚越过进场阈值**（从阈值内穿到阈值外）。

        `fresh` 是为「出场之后还能再进」准备的。只看信号的话，出场后信号
        往往还是 +1（趋势没结束），仓位就永远空着 —— 一条 20 年的趋势策略
        会因为一次回撤退出而错过后面整段。真正的语义应该是「这一笔被止损了，
        等下一次进场信号再进」，而进场信号就是分数重新越过阈值。
        """
        raw = int(want)                # 阈值信号原样，用来判断「信号自己走平」
        want = raw
        self.last = ""
        # 禁止分数出场：离场阈值说「平」的时候不认，仓位继续拿着。
        # 只认反向 —— 分数反向越过的是**进场**阈值，不是离场阈值；拿着
        # 反方向的仓等 ATR 触发，等于让一条趋势策略硬扛整个反向趋势
        if self.no_score and raw == 0:
            want = self.pos
        # 更彻底一层：反向也不认。持仓中分数翻向时维持原仓位，一路拿到 ATR
        # 触发为止（空仓时照常跟随，否则连第一笔都进不去）
        if self.no_flip and self.pos != 0 and want != 0 and want != self.pos:
            want = self.pos
        if self.blocked:
            # 释放看 `raw` 不看 `want`：no_score 下 want 被改写成了持仓方向，
            # 拿它判「信号走平」永远成立，止损一次就会在下一根立刻重开
            if raw == 0:
                self.blocked = False       # 信号自己走平：重新放行
            elif self.no_flip and self.exit_dir and raw == -self.exit_dir:
                # 分数翻到了反方向 = 另一笔交易。no_flip 下分数可能一路反向
                # 不回头，不认这一条的话仓位会永久空着（止损一次就再也不开）。
                # 只在 no_flip 下放开：默认口径的行为一个字不改
                self.blocked = False
            elif fresh and want != 0:
                # 分数重新越过进场阈值：算一笔新交易。
                # 原来这里写的是 `want == self.pos`，可出场时 pos 已经被清成 0，
                # 那个分支永远进不去 —— 分数一直不回落到阈值内的话仓位就永久
                # 空着，一次止损废掉后面整段（趋势策略最要命的那种）
                self.blocked = False
            else:
                return 0
        if want != self.pos:
            self.pos = want
            self.entry = px if np.isfinite(px) else 0.0
            self.best = 0.0
            return self.pos
        if self.pos == 0:
            return 0
        a = float(atr) if np.isfinite(atr) else 0.0
        if a <= 0 or not np.isfinite(px) or not self.entry:
            return self.pos

        exc = self.pos * (px - self.entry) / a    # 浮盈，ATR 倍数（空头也转正）
        if exc > self.best:
            self.best = exc
        hit = False
        why = ""
        if self.mode == "tier":
            allow = self.tiers[0][1]
            for need, trail in self.tiers:
                if self.best >= need:
                    allow = trail
            hit = (self.best - exc) >= allow
            if hit:
                why = (f"分档浮盈回撤（浮盈最高 {self.best:+.1f}ATR，"
                       f"回撤 {self.best - exc:.1f}ATR ≥ {allow:g}）")
        elif self.mode == "abs":
            hit = (self.best - exc) >= float(self.cfg.exit_abs_atr)
            if hit:
                why = (f"绝对回撤（从最高浮盈 {self.best:+.1f}ATR 回撤 "
                       f"{self.best - exc:.1f}ATR ≥ {float(self.cfg.exit_abs_atr):g}）")
        elif self.mode == "box":
            if np.isfinite(box_low) and np.isfinite(box_high):
                if self.pos > 0:
                    hit = px < box_low + self.buf * a
                    if hit:
                        why = (f"反向箱体突破（收 {px:,.2f} 跌破过去 {self.bars} 根"
                               f"最低 {box_low:,.2f}）")
                else:
                    hit = px > box_high - self.buf * a
                    if hit:
                        why = (f"反向箱体突破（收 {px:,.2f} 涨过过去 {self.bars} 根"
                               f"最高 {box_high:,.2f}）")
        # 止损：**独立**于上面三种口径，任何口径下都判一次。
        # 判据是「离进场价亏了几个 ATR」，不看浮盈最高点 —— 和「绝对回撤」
        # 的区别就在这里：浮盈到过 +5ATR 之后，绝对回撤会在回落到 +3ATR 时
        # 平（保住利润），止损这条线的位置却一直钉在进场价下面 N 个 ATR。
        # 所以两者同时开是有意义的：一个收利润，一个认亏。
        if not hit and self.sl > 0 and exc <= -self.sl:
            hit = True
            self.n_sl += 1
            why = f"ATR 止损（亏 {-exc:.1f}ATR ≥ {self.sl:g}）"
        if hit:
            self.exit_dir = self.pos      # 记住被平掉的方向（no_flip 释放要用）
            self.pos = 0
            self.blocked = True
            self.n_exit += 1
            self.last = why
            return 0
        return self.pos

    def describe(self) -> str:
        mode = self.mode
        label = {"tier": "分档浮盈回撤", "abs": "绝对回撤",
                 "box": "反向箱体突破"}.get(mode, mode)
        sl = (f" ＋ 止损 {self.sl:g}ATR（其中 {self.n_sl} 次由止损打掉）"
              if self.sl > 0 else "")
        # 不按分数出场要说出来：同一套 ATR 参数、勾没勾这一项，交易笔数会差
        # 一大截，而图上只是「有些仓位拿得久」，看不出是配置还是行情
        ns = ("，反向也不平仓" if self.no_flip else
              "，不按分数出场" if self.no_score else "")
        if mode == "score":
            # 只开了止损、没选 ATR 口径：别写成「出场：score」，那是个内部键名
            if self.no_score and self.sl > 0:
                # 分数规则一条都不剩，唯一的出场线就是止损
                return (f"出场：只按止损 {self.sl:g}ATR{ns}（共平仓 "
                        f"{self.n_exit} 次，全部由止损打掉）")
            if self.sl > 0:
                return (f"出场：分数出场 ＋ 止损 {self.sl:g}ATR"
                        f"（共平仓 {self.n_exit} 次，其中止损 {self.n_sl} 次）")
            return ""
        if mode == "tier":
            return (f"出场：{label}（" + "、".join(
                f"浮盈≥{p:g}ATR 回撤{t:g}ATR" for p, t in self.tiers)
                + f"）{ns}，共平仓 {self.n_exit} 次{sl}")
        if mode == "abs":
            return (f"出场：{label}（从最高浮盈回撤 "
                    f"{self.cfg.exit_abs_atr:g}ATR）{ns}，共平仓 {self.n_exit} 次{sl}")
        return (f"出场：{label}（箱体 {self.bars} 根，边界缓冲 {self.buf:g}ATR）"
                f"{ns}，共平仓 {self.n_exit} 次{sl}")


def apply_exit_rules(
    sig: np.ndarray,
    df: pd.DataFrame,
    cfg: FactorConfig,
    score: np.ndarray | None = None,
    entry: float = 0.0,
    reasons_out: list | None = None,
) -> tuple[np.ndarray, str]:
    """在阈值信号之上叠加 ATR 口径的出场规则。返回 (新信号, 说明)。

    出场判定用**当根收盘**，而回测在**下一根开盘**执行（`run_backtest` 的
    `target = roll(sig, 1)`）—— 和进场同一个口径，没有未来信息。

    箱体边界只取**过去** N 根（`[i-bars, i)`，不含当根）：把当根算进去的话，
    当根的低点天然就是「箱体下沿」，永远触发不了。

    `score` / `entry` 传了才支持「出场后分数重新越过阈值就再进」（见
    `ExitTracker.step` 的 `fresh`）。不传就是老行为：等信号自己归零或翻向。

    止损和四种口径是**叠加**关系：口径选 `score`（等于不叠 ATR 规则）但勾了
    止损时，这里照样要跑一遍 —— 那时候止损就是唯一一条 ATR 规则。

    `reasons_out` 传了就把逐根的平仓原因填进去（长度 = len(sig)，没平是空串）。
    用**出参**而不是返回值，是为了不破坏既有的 `sig, desc = ...` 两元解包 ——
    这条函数有十几处调用，加一个返回值等于把每一处都改一遍。
    """
    s = np.asarray(sig, dtype=int)
    cfg = cfg.normalized()
    if len(s) == 0:
        return s, ""
    if not ExitTracker(cfg).active():
        return s, ""

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    high = (pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
            if "high" in df.columns else close)
    low = (pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
           if "low" in df.columns else close)
    atr = _atr_array(df, cfg.exit_atr_n)
    tracker = ExitTracker(cfg)

    fresh = None
    if score is not None and len(score) == len(s):
        sc = np.abs(np.asarray(score, dtype=float))
        e = float(entry)
        prev = np.r_[np.nan, sc[:-1]]
        # 从「阈值内」穿到「阈值外」的那一根。NaN（预热期）不算穿越
        with np.errstate(invalid="ignore"):
            fresh = (sc > e) & ~(prev > e)

    out = s.copy()
    reasons: list[str] = []
    for i in range(len(s)):
        lo = hi = float("nan")
        if i >= tracker.bars:
            lo = float(np.nanmin(low[i - tracker.bars:i]))
            hi = float(np.nanmax(high[i - tracker.bars:i]))
        out[i] = tracker.step(int(s[i]), close[i], atr[i], lo, hi,
                              bool(fresh[i]) if fresh is not None else False)
        reasons.append(tracker.last)
    if reasons_out is not None:
        reasons_out.clear()
        reasons_out.extend(reasons)
    return out, tracker.describe()


def _rolling_z(s: pd.Series, window: int) -> pd.Series:
    """对模型预测值再做一次 PIT 标准化，让 entry_z 在各模型间可比。"""
    return F.standardize(s, window=window, method="z")


# ---------------------------------------------------------------- 跨周期因子
#
# 「周线策略调日线因子」这类需求，本质是**同一时刻、两个时间尺度上的数据**。
# 难点全在一个地方：大周期的 bar 什么时候才算数。
#
# 具体的坑：周线 bar 的时间戳是那周最后一根日线的日期（`merge_days` 的约定）。
# 直接把它贴到周一，就等于在周一就看到了周五的收盘价 —— 一周里每一个周线因子
# 都是未来数据。回测会好得离谱，而且一个字都不报错。
#
# 所以这里引入「决策时刻」：
#
#   日线级（1d / 1d_tq / 2d / 1w）：时间戳是**交易日标签**（00:00），这天的
#       数据要到收盘才齐 → 决策时刻 = 当天 15:00
#   分钟 / 小时级：时间戳本来就是 bar 的**结束时刻**（`merge_bars` 取组内
#       最后一根的时间），不用换算
#
# 两边都换成决策时刻之后，对齐就退化成「只取 <= 当前时刻的最后一根」——
# 一次 `searchsorted`，晚于当前时刻的 bar 一根都进不来。这就是截断。
#
# 基准周期也要换算：周线 bar 的时间戳同样是「那周最后一天的 00:00」，不换算的
# 话，周五那根日线因子（15:00 才齐）会被判成「比周线 bar 晚」，每周白丢一天。

DAY_CLOSE = pd.Timedelta(hours=15)

# 日线级周期的名字。**不从 tqdata 导**：这个模块被 selftest / 实盘 / 打包后的
# 环境反复导入，多一个 import 就多一处能断的地方；而周期名是这里唯一的依赖，
# 写死一张表反而更稳（`tqdata.is_daily` 判的也是这几个）
_DAY_PERIODS = {"1d", "1d_tq", "2d", "1w"}


def _decision_time(index, period: str) -> pd.DatetimeIndex:
    """把某个周期的 bar 时间戳换成「这根 bar 的数据已经全部可用的时刻」。"""
    idx = pd.DatetimeIndex(index)
    if str(period or "") in _DAY_PERIODS:
        return idx.normalize() + DAY_CLOSE
    return idx


def align_to_base(
    series: pd.Series,
    base_index: pd.Index,
    base_period: str = "",
    src_period: str = "",
) -> pd.Series:
    """把源周期的序列对齐到基准周期的每一根 bar 上，**只用已收完的 bar**。

    这是整个跨周期功能的关键一步，也是唯一一处会悄悄引入未来数据的地方：
    源周期里时间戳晚于当前 bar 的那些值，一根都不许进来。用 `searchsorted` 而
    不是 `reindex(method="ffill")`：后者要求两边都单调且源索引唯一，源周期是
    合成周期时（周线 / 2 日）偶尔会出现重复时间戳，一重复就抛异常。
    """
    base_index = pd.DatetimeIndex(base_index)
    if series is None or not len(series):
        return pd.Series(np.nan, index=base_index, dtype=float)

    src_t = _decision_time(series.index, src_period)
    vals = pd.Series(np.asarray(series, dtype=float), index=src_t)
    # 同一时刻出现两根（理论上不该有）：留最后一根，别让 searchsorted 拿错
    vals = vals[~vals.index.duplicated(keep="last")].sort_index()
    # 2 日 / 周线的**最后一根可能还没走完**（合成周期是按「行序两两成对」「ISO
    # 周」分组的，最后一组随时会被新来的一天改写）。它的时间戳是组内最后一根
    # 日线的日期，看上去像「已经收完」，其实是「到现在为止」——用它等于拿
    # 「这周到今天为止」当周线因子，数值每天变一次，还和以后重算的结果不一样。
    # 所以整根丢掉：宁可让大周期因子晚一根，也不要用一根会变形的 bar。
    # 日线级里只有这两档会这样，1d / 1d_tq 每天收盘就是完整的一天。
    if str(src_period or "") in ("2d", "1w") and len(vals) > 1:
        vals = vals.iloc[:-1]
    tgt = _decision_time(base_index, base_period)

    pos = vals.index.values.searchsorted(tgt.values, side="right") - 1
    out = np.full(len(base_index), np.nan, dtype=float)
    ok = pos >= 0
    if ok.any():
        out[ok] = vals.to_numpy(dtype=float)[pos[ok]]
    return pd.Series(out, index=base_index, dtype=float)


def period_frames(
    cfg: "FactorConfig",
    keys: list[str],
    base_period: str = "",
    loader=None,
    extra_keys=(),
) -> dict[str, pd.DataFrame]:
    """把跨周期因子要用的行情一次读出来：`{周期: DataFrame}`。

    读的活交给调用方（`loader(period) -> DataFrame`）：界面用带缓存的仓库，
    实盘用 `tqdata.load_main`，各自清楚该从哪儿取数。这里只负责算「哪些周期
    真的需要」—— 一个跨周期因子都没配时返回空 dict，调用方什么都不用做。

    `extra_keys` 是触发因子（稀疏采样用的那组）：它们不参与合成，但同样可能
    被指到别的周期上。
    """
    need: list[str] = []
    for k in list(keys or []) + list(extra_keys or []):
        p = cfg.period_of(k, base_period)
        if p and p != str(base_period or "") and p not in need:
            need.append(p)
    out: dict[str, pd.DataFrame] = {}
    if loader is None:
        return out
    for p in need:
        try:
            frame = loader(p)
        except Exception:      # noqa: BLE001
            frame = None       # 读不到就留空，由 factor_matrix 报「缺哪个周期」
        if frame is not None and len(frame):
            out[p] = frame
    return out


def factor_matrix(
    df: pd.DataFrame,
    keys: list[str],
    cfg: "FactorConfig",
    extra: dict[str, pd.DataFrame] | None = None,
    base_period: str = "",
    min_obs: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, list[str]]]:
    """按**每个因子实例自己的周期**算因子矩阵，再对齐到 `df` 的每一根 bar 上。

    `keys` 里每一项都是**实例**：`mom_60`（跟策略周期）或 `mom_60@1w`（周线）。
    同一个因子出现两次、周期不同，就是两列 —— 这正是「一个因子同时用多个
    周期」的做法，不需要另开一套数据结构。

    返回 `(fmat, rawf, kept, gone, missing)`：

    - `fmat` / `rawf`：标准化值 / 原始值，列名是实例 id，顺序与 `keys` 一致
    - `kept`：真正算出来的实例（`keys` 的子集，保持原顺序）
    - `gone`：在这个标的上没有足够观测的
    - `missing`：`{周期: [实例]}`，跨周期行情根本没读到

    单周期（全是纯键、`factor_periods` 为空）时走的就是原来的 `factor_frame`
    那条路，结果与以前逐位相同 —— 加了跨周期不该动老策略的每一个数字。
    """
    cfg = cfg.normalized()
    # 去重但保序：同一个实例重复出现（配置手改过）不该算两遍
    seen: set[str] = set()
    inst: list[tuple[str, str, str]] = []      # (实例 id, 因子键, 周期)
    for k in keys:
        spec = str(k)
        if spec in seen:
            continue
        seen.add(spec)
        base, p = F.split_spec(spec)
        inst.append((spec, base, cfg.period_of(spec, base_period)))

    groups: dict[str, list[tuple[str, str]]] = {}
    for spec, base, p in inst:
        groups.setdefault(p, []).append((spec, base))

    cols: dict[str, pd.Series] = {}
    raws: dict[str, pd.Series] = {}
    gone: list[str] = []
    missing: dict[str, list[str]] = {}
    for p, items in groups.items():
        cross = bool(p) and p != str(base_period or "")
        frame = (extra or {}).get(p) if cross else df
        if frame is None or frame.empty:
            missing[p] = [spec for spec, _b in items]
            continue
        # 可用性按**因子键**判（同一个键的两个周期各判一次，因为行情不同）
        use = set(F.filter_available(frame, [b for _s, b in items], min_obs=min_obs))
        for spec, base in items:
            if base not in use:
                gone.append(spec)
                continue
            r = F.raw(frame, base)
            z = F.standardize(r, window=cfg.window, method=cfg.method)
            if cross:
                r = align_to_base(r, df.index, base_period, p)
                z = align_to_base(z, df.index, base_period, p)
            cols[spec] = z
            raws[spec] = r

    kept = [spec for spec, _b, _p in inst if spec in cols]
    fmat = pd.DataFrame({k: cols[k] for k in kept}, index=df.index) if kept \
        else pd.DataFrame(index=df.index)
    rawf = pd.DataFrame({k: raws[k] for k in kept}, index=df.index) if kept \
        else pd.DataFrame(index=df.index)
    return fmat, rawf, kept, gone, missing


def cross_period_note(
    cfg: "FactorConfig", keys: list[str], base_period: str = ""
) -> str:
    """跨周期因子的一句说明（没配就返回空串）。"""
    hit = [(k, cfg.period_of(k, base_period)) for k in keys]
    hit = [(k, p) for k, p in hit if p and p != str(base_period or "")]
    if not hit:
        return ""
    from ttrade import tqdata

    def _name(p: str) -> str:
        return tqdata.PERIOD_LABEL.get(p, p)

    parts = []
    for p in dict.fromkeys(p for _k, p in hit):
        n = sum(1 for _k, q in hit if q == p)
        parts.append(f"{_name(p)} {n} 个")
    base = _name(base_period) if base_period else "策略周期"
    return (f"跨周期因子：{'、'.join(parts)}（基准 {base}）——"
            "只用已收完的 bar，对齐时不会拿到未来数据")


# ---------------------------------------------------------------- 分数构建
def build_score(
    df: pd.DataFrame,
    mode: str,
    keys: list[str],
    cfg: FactorConfig,
    progress=None,
    cancel=None,
    collect_models: list | None = None,
    extra: dict[str, pd.DataFrame] | None = None,
    base_period: str = "",
) -> ScorePack:
    """三种模式共用的分数构建。返回的 `score` 已是 PIT 标准化值。

    `collect_models` 传了就往里塞每一折的模型（见 `walk_forward`）——
    给「离线训练、实盘只预测」用。

    `extra` / `base_period` 是跨周期因子用的：`extra` 是 `{周期: 行情}`，
    `base_period` 是策略本身的周期（见 `period_frames`）。不配跨周期因子时
    两个都不用传，走的就是原来那条单周期路径。
    """
    cfg = cfg.normalized()
    keys = [str(k) for k in keys if F.is_factor(k)]
    if not keys:
        raise ValueError("没有选中任何可用因子")

    fmat, rawf, keys, gone, missing = factor_matrix(
        df, keys, cfg, extra=extra, base_period=base_period)
    if missing:
        # 跨周期的行情根本没读到。**必须报错**：把那个因子悄悄剔掉，用户看到
        # 的是「策略能跑、只是分数变了」，而分数变了这件事在界面上看不出来
        from ttrade import tqdata

        lack = "、".join(f"{tqdata.PERIOD_LABEL.get(p, p)}（"
                        + "、".join(F.spec_label(k) for k in ks)
                        + "）" for p, ks in missing.items())
        raise ValueError(
            f"跨周期因子缺行情：{lack}。这个标的上没有该周期的数据 —— "
            "换成已有周期，或者先去「数据管理」补一次"
        )
    if not keys:
        # 两根原因要分开说：一种是这个标的真的没有这些数据（美股没有持仓量），
        # 另一种是**行情太短**，所有因子都凑不出有效值。合成周期把第二种放大
        # 了 —— 周线「近 1 年」只有五十来根，看着像「因子没数据」，其实是样本
        # 不够，用户照着提示去换因子只会白折腾
        if len(df) < max(60, int(cfg.window) // 2):
            raise ValueError(
                f"这段行情只有 {len(df)} 根，太短了算不出因子（标准化窗口要 "
                f"{int(cfg.window)} 根）—— 把回测区间拉长，或者把窗口调小")
        raise ValueError("选中的因子在该标的上都没有足够数据（如美股无持仓量）")

    fwd = M.forward_return(df, cfg.horizon)
    notes: list[str] = []
    # 标签上的「N 个」是**过滤后**的数量。少掉几个不是配置被改了，而是这几个
    # 因子在当前标的 / 当前区间上没有足够观测（典型：展期、期权那类列只有
    # 本地库才有）。不说明一句，用户看到数字自己变，会以为策略被换了。
    if gone:
        notes.append(
            f"已剔除 {len(gone)} 个在该标的上没有足够数据的因子："
            + "、".join(F.spec_label(k) for k in gone)
        )
    xnote = cross_period_note(cfg, keys, base_period)
    if xnote:
        notes.append(xnote)

    if mode == "single":
        k = keys[0]
        ori = _orientation_frame(fmat[[k]], fwd, cfg)
        score = (fmat[k] * ori[k]).rename("score")
        rev = cfg.is_reversed(k)
        pack = ScorePack(
            score=score, fmat=fmat, raw=rawf, orient=ori,
            label=f"单因子 · {F.spec_label(k)}" + ("（反转）" if rev else ""),
        )
        pack.notes.extend(notes)      # 「剔除了哪几个因子」这类说明
        if cfg.orient == "auto":
            pack.notes.append("方向由扩张窗口 IC 决定（滞后 h 期，无未来信息）")
            onote = _orient_note(ori, cfg)
            if onote:
                pack.notes.append(onote)
        note = _reverse_note(cfg, keys)
        if note:
            pack.notes.append(note)
        note = _warmup_note(score, len(df), cfg)
        if note:
            pack.notes.append(note)
        return pack

    if mode == "multi":
        ori = _orientation_frame(fmat, fwd, cfg)
        signed = fmat * ori
        wmode = str(cfg.weight_mode or "equal")
        wdesc = "等权"
        if wmode == "ic":
            w = rolling_ic_weights(fmat, fwd, cfg)
            score = _weighted_mean(signed, w)
            wdesc = f"滚动 IC 加权 · 每 {int(cfg.ic_rebal)} 根重算"
        elif wmode == "custom":
            w = pd.Series({k: cfg.weight_of(k) for k in signed.columns}, dtype=float)
            w = w.where(w > 0, 0.0)
            if not (float(w.sum()) > 0):
                w = pd.Series(1.0, index=signed.columns)
                notes.append("权重全填了 0，已退回等权")
            score = _weighted_mean(signed, w)
            wdesc = "自定权重"
        else:
            score = signed.mean(axis=1, skipna=True).rename("score")
        n_rev = sum(1 for k in keys if cfg.is_reversed(k))
        pack = ScorePack(
            score=score, fmat=fmat, raw=rawf, orient=ori,
            label=(f"多因子组合 · {len(keys)} 个（{wdesc}）"
                   + (f"（{n_rev} 个反转）" if n_rev else "")),
        )
        pack.notes.extend(notes)      # 「剔除了哪几个因子」这类说明
        if cfg.orient == "auto":
            pack.notes.append(f"各因子先按自身 PIT 方向取号，再{wdesc}合成")
            onote = _orient_note(ori, cfg)
            if onote:
                pack.notes.append(onote)
        if wmode == "custom":
            used = [(k, cfg.weight_of(k)) for k in keys]
            pack.notes.append(
                "权重：" + "、".join(
                    f"{F.spec_label(k)} {v:g}" for k, v in used if v > 0)
                + (f"（另 {sum(1 for _, v in used if v <= 0)} 个权重为 0，不参与合成）"
                   if any(v <= 0 for _, v in used) else "")
            )
        elif wmode == "ic":
            pack.notes.append(
                f"权重按过去 {int(cfg.window)} 根的 |IC| 定，每 "
                f"{int(cfg.ic_rebal)} 根更新一次（滞后 {int(cfg.horizon)} 期，无未来信息）"
            )
        note = _reverse_note(cfg, keys)
        if note:
            pack.notes.append(note)
        note = _warmup_note(score, len(df), cfg)
        if note:
            pack.notes.append(note)
        return pack

    if mode == "ml":
        if len(keys) < 2:
            raise ValueError("机器学习多因子至少需要 2 个因子")
        mask = None
        plan: TriggerPlan | None = None
        wstats: dict = {}
        if str(cfg.sample_mode or "all") == "trigger":
            st, armed, tnotes, rule_sig = trigger_signal(
                df, cfg, extra=extra, base_period=base_period)
            mask, mnotes = trigger_mask(st, armed, cfg)
            notes.extend(tnotes)
            notes.extend(mnotes)
            if not mask.any():
                raise ValueError(
                    "触发条件一根都没命中 —— 把 trigger_z 调低、换一组触发因子，"
                    "或者把 trigger_cool 调小"
                )
            plan = TriggerPlan(
                mask=np.asarray(mask, dtype=bool),
                strength=st.to_numpy(dtype=float),
                armed=armed.to_numpy(dtype=bool),
                rule_sig=rule_sig.to_numpy(dtype=float),
                # 配了触发因子、但它在当前标的上算不出值（美股没持仓量那种），
                # 这条通道就是空的，决策只能靠规则。**不能只看有没有 NaN**：
                # 只配规则时强度是用 1.0 顶上的，那是「占位」不是真信号，
                # 拿它定方向会让每次触发都变成做多
                factor_channel=(bool(cfg.trigger_keys)
                                and bool(np.isfinite(st.to_numpy(dtype=float)).any())),
                rule_channel=bool(cfg.trigger_rule) and cfg.trigger_rule in STRATEGIES,
            )
        pred, imp, n_folds = walk_forward(
            fmat, fwd, cfg, progress=progress, cancel=cancel, sample_mask=mask,
            stats=wstats, collect_models=collect_models,
        )
        if wstats.get("fallback"):
            notes.append(
                f"{int(wstats['fallback'])}/{n_folds} 折的触发样本不足 40 个，"
                "这几折退回全样本训练（触发太稀，把阈值调低或冷却调小）"
            )
        if wstats.get("degenerate_fold"):
            notes.append(
                f"{int(wstats['degenerate_fold'])}/{n_folds} 折的触发样本太少，"
                "训出来的模型只会输出常数（连全样本退回也救不回来），"
                "这几折留空不出信号（触发样本这么少，模型学不出东西）"
            )
        if wstats.get("degenerate"):
            notes.append(
                f"{int(wstats['degenerate'])}/{n_folds} 折的触发样本训出了常数模型，"
                "已退回全样本重训（触发点太稀，把阈值调低或冷却调小）"
            )
        if wstats.get("thin"):
            notes.append(
                f"{int(wstats['thin']):,} 行因子缺失超过一半，已从训练集剔除"
                "（缺得少的按中位数补，不再整行丢弃）"
            )
        score = _rolling_z(pred, cfg.window).rename("score")
        if plan is not None and len(plan.mask) == len(score):
            # 观点只在触发点刷新，触发点之间沿用上一次的值。**标准化要在这之前
            # 做**：`_rolling_z` 的窗口里得有足够观测，稀疏之后一个 252 根的窗口
            # 只剩十几根有值，够不到 min_periods，分数会全变 NaN
            score = score.where(plan.mask).ffill().rename("score")
            notes.append(
                f"观点只在触发点刷新（{int(plan.mask.sum()):,} 个决策点），"
                "触发点之间沿用上一次的分数"
            )
            # 出场按通道分开说：两条通道的出场口径不一样，只报一条会让人以为
            # 规则那条也是看强度
            ent = (f"进场：触发方向 + 模型分数过进场阈值（{cfg.entry_z:g}）"
                   if plan.factor_channel else
                   f"进场：规则刚开仓 / 刚翻向 + 模型分数过进场阈值（{cfg.entry_z:g}）")
            # 「回落到 x 以内」只对非负阈值说得通。负阈值是「反向超过 |x| 才平」，
            # 照原话说会让人以为 -1 比 0 更容易出场，正好相反
            _how = (f"触发强度反向超过 {-cfg.exit_z:g} 才平（离场阈值 {cfg.exit_z:g}）"
                    if cfg.exit_z < 0 else
                    f"触发强度回落到 {cfg.exit_z:g} 以内")
            if plan.factor_channel and plan.rule_channel:
                ext = (f"出场：因子触发的仓位看{_how}，"
                       "规则触发的仓位跟规则自己的 signal")
            elif plan.factor_channel:
                ext = f"出场：{_how}，每根都判"
            else:
                ext = "出场：跟规则自己的 signal（规则变 0 或翻向就平）"
            # 反转了哪条通道要写出来：同一组参数、「反转因子」勾没勾，图上仓位方向
            # 完全相反，而两张图看起来一样正常
            flip = [n for n, on in (("因子", cfg.trigger_flip and plan.factor_channel),
                                    ("规则", cfg.trigger_rule_flip and plan.rule_channel))
                    if on]
            notes.append(f"{ent}；{ext}" + (f"；方向已反转：{'、'.join(flip)}" if flip else ""))
            # 触发点全落在「模型还没开始预测」的前段：一个决策点都用不上，
            # 整条策略一笔不做。不点出来的话，用户只会看到一条水平线
            _sc = score.to_numpy(dtype=float)[np.asarray(plan.mask, dtype=bool)]
            if not np.isfinite(_sc).any():
                notes.append(
                    "触发点全部落在模型还没出预测的前段（首次训练之前），"
                    "这段一笔都不做 —— 拉长回测区间，或把训练窗口 / 首次训练根数调小"
                )
        covered = int(pred.notna().sum())
        _bpd = _bars_per_day(fmat.index)
        notes.append(
            f"{ML.MODELS.get(cfg.model, {}).get('label', cfg.model)} · "
            f"{n_folds} 折滚动训练，训练窗口 {cfg.train_window} 个交易日"
            f"（{_to_bars(cfg.train_window, _bpd):,} 根），"
            f"有效预测 {covered}/{len(df)} 根"
        )
        if n_folds == 0:
            notes.append(
                f"数据不足：首次训练要 {cfg.min_train:,} 根，当前只有 {len(fmat):,} 根"
                f" —— 下载更长的历史，或把训练窗口调小"
            )
        if covered < len(df) * 0.5:
            notes.append("前段样本不足以训练，该区间不持仓（walk-forward 的正常代价）")
        # 反转开关在 ML 模式下不生效：模型自己从数据里学方向和权重，
        # 人工指定方向反而会和它打架。这里只提醒一句，不静默忽略。
        if any(cfg.is_reversed(k) for k in keys):
            notes.append("机器学习自行学习方向，反转开关在 ML 模式下不生效")
        return ScorePack(
            score=score, fmat=fmat, raw=rawf, pred=pred, importance=imp,
            label=f"机器学习多因子 · {ML.MODELS.get(cfg.model, {}).get('label', cfg.model)}",
            folds=n_folds, notes=notes, trigger=plan,
        )

    raise ValueError(f"未知因子模式 {mode}")


def trigger_signal(df: pd.DataFrame, cfg: FactorConfig,
                   extra: dict[str, pd.DataFrame] | None = None,
                   base_period: str = ""):
    """触发强度（带符号）、「规则刚开仓」标记，以及规则自己的持仓。

    强度 = 触发因子的时序 Z-score 均值。**不取绝对值**：符号留着，向上突破
    和向下破位是两种样本，`trigger_side` 靠它筛，进场方向也靠它定。

    规则那条路给的是 `armed`（信号从 0 变成 ±1、或从 +1 翻到 −1 的那一根），
    不带强度 —— 规则本身只说「该动手了」，强度由因子那一侧补。两条路都没有
    就是全 False（调用方会先拦下来，见下面的「至少要有一个触发源」）。

    `rule_sig` 是规则**逐根的持仓**，专门给规则通道出场用：那条通道没有强度，
    「回落到 0.3」对它没意义，只能按规则自己的 signal 平仓。

    **反转**（`trigger_flip` / `trigger_rule_flip`）在这里落地，两条通道各管各的：
    因子那侧把强度取负（向下破位变向上突破），规则那侧把 signal 取负。放在这里
    而不是在消费端取负，是因为强度同时被三处用 —— 分档筛样本、`trigger_side`
    方向过滤、出场判据。任何一处忘了取负，「反转」就只反了一半。

    返回 `(strength, armed, notes, rule_sig)`，都对齐到 `df.index`。
    """
    idx = df.index
    strength = pd.Series(np.nan, index=idx, dtype=float)
    armed = pd.Series(False, index=idx)
    rule_sig = pd.Series(0.0, index=idx, dtype=float)
    notes: list[str] = []

    # 一个触发源都没有：稀疏采样会建出 0 个样本，然后整条策略空仓 ——
    # 「跑通了只是没交易」看起来和正常结果一模一样，必须在源头拦掉
    has_rule = bool(cfg.trigger_rule) and cfg.trigger_rule in STRATEGIES
    if not (cfg.trigger_keys or has_rule):
        raise ValueError(
            "稀疏采样至少要有一个触发源：选一个触发因子，或者选一条触发规则"
            "（两个触发因子都留空也行，那就只用规则）"
        )

    zs = []
    for key in (cfg.trigger_keys or []):
        if not F.is_factor(key):
            continue
        # 触发因子也可能被指到别的周期上（和特征因子是同一份 `factor_periods`）。
        # 不认这件事的话，界面上选了「周线」，强度却还是按基准周期算的
        p = cfg.period_of(key, base_period)
        frame = df
        if p and p != str(base_period or ""):
            frame = (extra or {}).get(p)
            if frame is None or frame.empty:
                continue
        try:
            raw = F.raw(frame, key)
        except Exception:      # noqa: BLE001
            continue
        if raw is None or not len(raw):
            continue
        z = _rolling_z(raw.astype(float), cfg.window)
        if p and p != str(base_period or ""):
            z = align_to_base(z, df.index, base_period, p)
        if z.notna().sum() >= 60:
            zs.append(z.rename(key))
    if zs:
        strength = pd.concat(zs, axis=1).mean(axis=1, skipna=True)
        if cfg.trigger_flip:
            strength = -strength
        notes.append("触发：因子 " + "、".join(F.spec_label(k)
                                              for k in (cfg.trigger_keys or [])
                                              if F.is_factor(k))
                     + ("（反转）" if cfg.trigger_flip else ""))

    if has_rule:
        try:
            out = run_strategy(df, cfg.trigger_rule, dict(cfg.trigger_params or {}))
            sig = pd.Series(out["signal"], index=idx).astype(float).fillna(0.0)
            if cfg.trigger_rule_flip:
                sig = -sig
            rule_sig = sig
            prev = sig.shift(1).fillna(0.0)
            armed = ((sig != 0) & (prev == 0)) | ((sig != 0) & (np.sign(sig) != np.sign(prev)))
            # 规则只给「该动手」的点，没有强度。因子那侧没配就用 1.0 顶上，
            # 让分档退化成「单一档」，至少不会因为 NaN 一个样本都不建
            if not zs:
                strength = pd.Series(np.where(armed, 1.0, np.nan), index=idx)
            # 参数非默认时要写出来：同一句「触发：规则 双均线交叉」，5/20 和
            # 10/60 的样本点差一倍，图上却看不出区别
            prm = {k: v for k, v in dict(cfg.trigger_params or {}).items()
                   if k in STRATEGIES[cfg.trigger_rule].params}
            ptxt = ("（" + "、".join(f"{PARAM_LABELS.get(k, k)} {v:g}"
                                    for k, v in prm.items()) + "）") if prm else ""
            notes.append(f"触发：规则 {STRATEGIES[cfg.trigger_rule].label}{ptxt}"
                         + ("（反转）" if cfg.trigger_rule_flip else ""))
            if not zs:
                # 没有因子那一侧，强度是 1.0 占位，不是真信号：方向和「值不值得做」
                # 全由规则 + 模型分数决定，阈值 / 分档这几个参数在这条路上不生效
                notes.append("触发：只用规则（触发因子都留空），阈值与分档不参与")
        except Exception as exc:      # noqa: BLE001
            notes.append(f"触发规则跑不起来（{exc}），只用因子触发")
    return strength, armed, notes, rule_sig


def trigger_mask(strength: pd.Series, armed: pd.Series,
                 cfg: FactorConfig) -> tuple[np.ndarray, list[str]]:
    """按触发强度分档，挑出要建样本的那几根 bar。

    规则（逐 bar 扫一遍，**只往后看**，天然 PIT）：

    1. 强度不够 `trigger_z` 就跳过；`trigger_side` 决定只做向上、只做向下、还是都做
       （**反转打开后**这里判的是反转后的符号：「只采向上」= 因子原本的向下破位）
    2. 强度分档：`tier = floor((|s| − trigger_z) / trigger_span)`，夹在
       `[0, trigger_tiers−1]`。**分档的意义**是「z=1.2 的突破」和「z=2.6 的
       突破」不该混成一个样本 —— 前者往往是假突破，后者才是趋势启动
    3. 同一档冷却 `trigger_cool` 根：一次突破持续 20 根，不该建 20 个样本

    返回 `(mask, notes)`。
    """
    n = len(strength)
    mask = np.zeros(n, dtype=bool)
    tiers = int(max(1, cfg.trigger_tiers))
    last = np.full(tiers, -(10 ** 9), dtype=int)
    side = str(cfg.trigger_side or "both").lower()
    z0 = float(cfg.trigger_z)

    s_v = strength.to_numpy(dtype=float)
    a_v = armed.to_numpy(dtype=bool) if armed is not None else np.zeros(n, dtype=bool)
    n_trig = n_side = 0
    for i in range(n):
        s = s_v[i]
        ok_arm = bool(a_v[i])
        if not np.isfinite(s) and not ok_arm:
            continue
        mag = abs(s) if np.isfinite(s) else 0.0
        if side == "up" and np.isfinite(s) and s <= 0:
            n_side += 1
            continue
        if side == "down" and np.isfinite(s) and s >= 0:
            n_side += 1
            continue
        if mag < z0 and not ok_arm:
            continue
        n_trig += 1
        if cfg.trigger_tiers >= 2:
            t = int(np.floor((mag - z0) / cfg.trigger_span))
            t = min(max(t, 0), tiers - 1)
        else:
            t = 0
        if i - last[t] < cfg.trigger_cool:
            continue
        last[t] = i
        mask[i] = True

    got = int(mask.sum())
    # 只用规则时强度是 1.0 占位，「越阈值」「按 N 档」这些词在那种情况下不成立 ——
    # 参数确实不参与，说明行再写一遍阈值分档，用户会去找一个根本不起作用的旋钮
    only_rule = not bool(cfg.trigger_keys)
    head = (f"稀疏采样：{n_trig:,} 个规则触发点，冷却 {cfg.trigger_cool} 根后留下 "
            if only_rule else
            f"稀疏采样：{n_trig:,} 根越阈值，按 {max(1, cfg.trigger_tiers)} 档"
            f"冷却 {cfg.trigger_cool} 根后留下 ")
    notes = [head + f"{got:,} 根样本（占全部 {got / max(n, 1) * 100:.1f}%）"]
    if got < 60:
        notes.append("样本太少（< 60），" + ("换一条开仓更频繁的触发规则、把冷却调小"
                                            if only_rule else
                                            "把触发阈值调低、冷却调小，或者放宽分档"))
    if n_side:
        notes.append(f"方向过滤挡掉了 {n_side:,} 根（trigger_side={side}）")
    return mask, notes


# ---------------------------------------------------------------- Walk-forward
def bars_per_day(idx) -> float:
    """平均每个交易日几根 bar。日线 1、1h 约 7~9、5m 约 48~68（含夜盘）。

    截面策略的「调仓间隔」要按**天 / 月**给（选品种不能频繁换仓），落地时得靠
    这个密度换算成根数；机器学习那边的「训练窗口（交易日）」同理。
    """
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return 1.0
    nd = idx.normalize().nunique()
    return len(idx) / float(max(1, nd))


# 老名字，包内和冒烟测试里都在用
_bars_per_day = bars_per_day


def _to_bars(days: int, bpd: float) -> int:
    """交易日 → bar 根数。

    **不能按数据长度夹取。** 夹了的话窗口长度就取决于「总共有多少行」，等于
    用了「未来还有多少数据」这条信息：同一个历史时点，今天回测和昨天回测给出
    的预测会不一样，净值每天都变。数据不够就老老实实不出信号。
    """
    return int(max(60, round(int(days) * bpd)))


def walk_forward(
    fmat: pd.DataFrame,
    fwd: pd.Series,
    cfg: FactorConfig,
    progress=None,
    cancel=None,
    sample_mask=None,
    stats: dict | None = None,
    collect_models: list | None = None,
) -> tuple[pd.Series, pd.Series | None, int]:
    """滚动训练 + 预测。

    时间轴（每折）：

        |<---- 训练窗口 train_window ---->|<-- 预测 step -->|
         （交易日 → 按数据密度换算成根，见 `_to_bars()`）
        a                                b                    c

    训练样本只取到 `b - horizon`：第 i 行的标签 `fwd[i]` 要到 i+h 才成立，
    所以 i > b-h 的行在 b 时刻还是「未知标签」，混进训练集就是未来函数。

    `sample_mask` 是**训练集的稀疏采样**（bool 数组，长度同 fmat）：只在这些
    根上建样本。预测不受影响 —— 稀疏采样的目的是「让模型学得有重点」，不是
    「少预测几根」。None 就是每根都当样本（老行为）。

    `stats` 是可选的计数容器：某些折的稀疏样本不够训练时会退回全样本，退回
    了几折写在 `stats["fallback"]` 里。不传就只做事不记账。

    `collect_models` 是可选的列表：传了就往里塞每一折训出来的模型和它的
    预测区间 `[(model, b, c), ...]`。这是给「离线训练、实盘只预测」用的 ——
    把每一折都存成 .pkl，实盘按 bar 落在哪一折就调用哪一折的模型，
    得到的预测序列和回测**逐根一致**。不传就一个都不留（老行为，不占内存）。

    返回 (预测序列, 因子重要度, 折数)。未覆盖区间为 NaN。
    """
    cfg = cfg.normalized()
    keys = list(fmat.columns)
    X = fmat.to_numpy(dtype=float)
    Y = fwd.to_numpy(dtype=float)
    n = len(fmat)
    pred = np.full(n, np.nan)

    # 配置里是交易日，这里按数据的 bar 密度换成根数
    bpd = _bars_per_day(fmat.index)
    train_bars = _to_bars(cfg.train_window, bpd)
    step_bars = _to_bars(cfg.step, bpd)
    # min_train 是根数，不换算；只保证不超过训练窗口
    min_bars = min(int(cfg.min_train), train_bars)

    if n <= min_bars + cfg.horizon + 10:
        return pd.Series(pred, index=fmat.index), None, 0

    bounds = []
    t = min_bars
    while t < n:
        bounds.append((max(0, t - train_bars), t, min(n, t + step_bars)))
        t += step_bars

    total = len(bounds)
    imp_acc = np.zeros(len(keys))
    imp_n = 0
    done = 0

    for fi, (a, b, c) in enumerate(bounds):
        if cancel and cancel():
            raise Cancelled()
        tr_end = b - cfg.horizon
        if tr_end - a < 60:
            continue
        Xtr, Ytr = X[a:tr_end], Y[a:tr_end]
        fin = np.isfinite(Xtr)
        # 缺值按 0（标准化后的中位数）补，**只把「缺得太多」的行剔掉**。
        # 原来是「任何一个因子缺值就整行不要」，实测 RB 日线选 7 个因子时，
        # 期权那两列（iv_atm / pcr）只有最近 20% 的年份有值 —— 一条规则就把
        # 4275 行砍到 864 行，80% 的样本白扔，模型等于只看最近两三年
        need_cols = max(1, int(np.ceil(len(keys) * 0.5)))
        ok = np.isfinite(Ytr) & (fin.sum(axis=1) >= need_cols)
        if stats is not None:
            thin = int((np.isfinite(Ytr) & (fin.sum(axis=1) < need_cols)).sum())
            if thin:
                stats["thin"] = int(stats.get("thin", 0)) + thin
        Xtr = np.where(fin, Xtr, 0.0)
        need = 60
        if sample_mask is not None:
            m = np.asarray(sample_mask[a:tr_end], dtype=bool)
            if int((ok & m).sum()) >= 40:
                ok = ok & m
                # 稀疏采样之后一折里能剩几十个样本就算不错了，门槛跟着降
                need = 40
            elif stats is not None:
                # 触发太稀：这一折凑不够样本。**退回全样本**而不是跳过整折 ——
                # 日线上 400 根窗口 × 5% 触发只有 20 个点，跳过的话整条策略
                # 一根信号都不出，看起来像「功能坏了」。退回了要记一笔，
                # 否则用户以为稀疏采样一直在生效
                stats["fallback"] = int(stats.get("fallback", 0)) + 1
        if ok.sum() < need:
            continue

        model = ML.build(cfg.model)
        try:
            model.fit(Xtr[ok], Ytr[ok])
        except Exception:
            # 单折失败不该让整条策略崩掉，跳过该折继续
            continue

        # ---- 稀疏样本训出「常数模型」的兜底 ----
        # 几十个样本 + 树的最小叶子样本数 / L1 正则，劈不开 → 只留一棵单叶树
        # → 预测恒定 → 实盘滚动 z 的标准差是 0 → z 分数永远是 NaN。这种模型
        # 在产物文件里看不出异常（重要度还在），但实盘会一直卡在「训练中」。
        # 这里探测到就退回全样本重训一次。
        if _is_degenerate_model(model, Xtr[ok][:64]):
            ok_full = np.isfinite(Ytr) & (fin.sum(axis=1) >= need_cols)
            if ok_full.sum() >= need:
                m2 = ML.build(cfg.model)
                try:
                    m2.fit(Xtr[ok_full], Ytr[ok_full])
                except Exception:  # noqa: BLE001
                    m2 = None
                if m2 is not None and not _is_degenerate_model(m2, Xtr[ok_full][:64]):
                    model = m2
                    if stats is not None:
                        stats["degenerate"] = int(stats.get("degenerate", 0)) + 1
                else:
                    model = None
            else:
                model = None
            if model is None:
                # 稀疏样本和全样本都救不回来：这一折只能出常数预测。
                # 宁可这折整段留 NaN，也不要一个恒定的分数 —— 恒定分数在
                # 滚动 z 里会把标准差抹成 0，后面每一根都跟着算不出来。
                if stats is not None:
                    stats["degenerate_fold"] = int(stats.get("degenerate_fold", 0)) + 1
                continue

        lb = int(getattr(model, "lookback", 0) or 0)
        ctx_start = max(a, b - lb) if lb else b
        ctx = X[ctx_start:c]
        ctx = np.where(np.isfinite(ctx), ctx, 0.0)
        try:
            p = model.predict(ctx)
        except Exception:
            continue
        tail = c - b
        pred[b:c] = p[-tail:] if len(p) >= tail else np.nan

        imp = model.importance()
        if imp is not None and len(imp) == len(keys):
            imp_acc += np.asarray(imp, dtype=float)
            imp_n += 1

        if collect_models is not None:
            # 把这一折的模型留下：`(模型, 预测区间起, 预测区间止)`。
            # 区间是**bar 位置**（相对 `fmat.index`），实盘那边靠它判断
            # 「当前这根该用哪一折的模型」
            collect_models.append((model, int(b), int(c)))

        done += 1
        if progress:
            progress(done, total, f"第 {done}/{total} 折训练完成")

    if cancel and cancel():
        raise Cancelled()

    imp_ser = None
    if imp_n:
        v = imp_acc / imp_n
        if v.sum() > 0:
            v = v / v.sum()
        imp_ser = pd.Series(v, index=keys).sort_values(ascending=False)
    return pd.Series(pred, index=fmat.index, name="pred"), imp_ser, total


# ---------------------------------------------------------------- 逐笔原因
def _entry_reasons(score, sig, entry: float, plan=None) -> np.ndarray:
    """逐根的**进场原因**：信号从空仓 / 反向变成某一方向的那些根。

    只写在**变化的那一根**（决策根）。`run_backtest` 是去决策根取值的
    （`target = roll(sig, 1)`），这里再 shift 一次就多错一根。

    触发模式（稀疏采样）分两条通道，原因也分开写：规则通道开的仓，说「规则
    触发」；因子通道开的仓，把触发强度一起写出来 —— 否则用户看到「因子分
    +2.31 进场」，但那一根的分数其实在触发点之间是阶梯常值，对不上。
    """
    n = len(sig)
    out = np.empty(n, dtype=object)
    out[:] = ""
    sc = None if score is None else np.asarray(score, dtype=float)
    prev = np.r_[0, np.asarray(sig, dtype=int)[:-1]]
    for i in range(n):
        cur = int(sig[i])
        if cur == 0 or cur == int(prev[i]):
            continue
        side = "做多" if cur > 0 else "做空"
        v = float(sc[i]) if sc is not None and i < len(sc) else float("nan")
        if plan is not None:
            rule_v = np.asarray(plan.rule_sig, dtype=float)
            if i < len(rule_v) and np.isfinite(rule_v[i]) and rule_v[i] != 0:
                out[i] = f"规则触发进场（{side}）"
                continue
            st = np.asarray(plan.strength, dtype=float)
            s_v = float(st[i]) if i < len(st) else float("nan")
            if np.isfinite(s_v):
                out[i] = f"触发进场（{side}，强度 {s_v:+.2f} ≥ {entry:g}）"
            else:
                out[i] = f"触发进场（{side}）"
            continue
        if np.isfinite(v):
            out[i] = f"因子分 {v:+.2f} 越过进场阈值 {entry:g}（{side}）"
        else:
            out[i] = f"信号进场（{side}）"
    return out


# ---------------------------------------------------------------- 策略入口
def run_factor_strategy(
    df: pd.DataFrame,
    mode: str,
    keys: list[str],
    cfg: FactorConfig,
    progress=None,
    cancel=None,
    collect_models: list | None = None,
    extra: dict[str, pd.DataFrame] | None = None,
    base_period: str = "",
) -> FactorOutcome:
    """因子策略：分数 → 阈值信号。

    触发模式（稀疏采样）走的是另一套状态机：进场只在触发点、方向由触发定、
    出场看触发强度（见 `trigger_hysteresis`）。单因子 / 多因子没有触发点，
    仍是分数过阈值那一套。

    `collect_models` 传了就收集每一折的模型（见 `walk_forward`）——
    给「离线训练、实盘只预测」用。

    `extra` / `base_period` 给跨周期因子用（见 `period_frames`）。
    """
    cfg = cfg.normalized()
    pack = build_score(df, mode, keys, cfg, progress=progress, cancel=cancel,
                       collect_models=collect_models,
                       extra=extra, base_period=base_period)
    if mode == "ml" and pack.trigger is not None:
        sig = trigger_hysteresis(pack.score.to_numpy(), pack.trigger,
                                 cfg.entry_z, cfg.exit_z)
    else:
        sig = hysteresis(pack.score.to_numpy(), cfg.entry_z, cfg.exit_z)

    notes = list(pack.notes)
    # ATR 口径的出场规则：叠在阈值信号之上（`score` 模式 = 不叠加）。
    # 带上分数和进场阈值，出场后分数重新越过阈值就能再进（见 ExitTracker.step）
    exit_why: list[str] = []
    sig, exit_note = apply_exit_rules(
        sig, df, cfg, score=pack.score.to_numpy(dtype=float), entry=cfg.entry_z,
        reasons_out=exit_why)
    if exit_note:
        notes.append(exit_note)

    out = df.copy()
    out["factor_score"] = pack.score
    for c in pack.raw.columns:
        out[f"f_{c}"] = pack.raw[c]
    out["signal"] = sig
    # 交易记录页的备注列读这两列（`run_backtest` 按决策根取值）。
    # 阈值出场没有 ATR 规则时 `exit_why` 是空的，引擎会兜底写「信号离场」
    out["entry_note"] = _entry_reasons(pack.score, sig, cfg.entry_z,
                                       pack.trigger if mode == "ml" else None)
    out["exit_note"] = exit_why if exit_why else ""

    if cfg.orient == "auto" and mode in ("single", "multi"):
        flips = int((pack.orient.diff().abs() > 0).any(axis=1).sum()) if pack.orient is not None else 0
        if flips:
            notes.append(f"期内方向翻转 {flips} 次（由 IC 转号触发）")
    return FactorOutcome(df=out, pack=pack, label=pack.label, notes=notes)


# ---------------------------------------------------------------- 多品种合并训练
def _pit_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """逐根滚动波动率，只用过去的数据。

    合并训练时各品种的标签必须同尺度：铁矿一天动 2%、玉米动 0.5%，直接拿原始
    前向收益去训，模型会几乎只拟合高波动的那几个品种。除以自己的波动率之后，
    「涨了一个标准差」在所有品种上是同一件事。rolling 只回头看，没有未来信息。
    """
    r = pd.Series(pd.to_numeric(df["close"], errors="coerce")).pct_change()
    return r.rolling(int(max(20, window)), min_periods=20).std()


def pooled_score(
    frames: dict[str, pd.DataFrame],
    keys: list[str],
    cfg: FactorConfig,
    progress=None,
    cancel=None,
    collect_models: list | None = None,
    extras: dict[str, dict[str, pd.DataFrame]] | None = None,
    base_period: str = "",
) -> dict[str, ScorePack]:
    """多品种**合并**训练：N 个品种的样本堆成一张长表，训一个模型。

    和 `build_score` 的区别只在训练集：那边一个品种一个模型（各学各的形态），
    这边全池一个模型（学「这个池子的共同形态」）。预测仍然是各品种在自己的
    bar 上做、各自跑各自的回测 —— 合起来的只是「学到的东西」，不是仓位。

    两处必须对齐，否则合训会变成「高波动品种的模型」：

    - 特征：`factor_frame` 本来就是**逐品种时序标准化**的，同一列在各品种上
      含义一致，可以直接拼
    - 标签：前向收益要除以该品种自己的滚动波动率（`_pit_vol`），换成「几个
      标准差」。不除的话铁矿 2% 的日波动会把玉米 0.5% 的信号完全淹没

    返回 `{品种: ScorePack}`，触发模式下每个 pack 都带自己的 `trigger` 计划
    （决策点 / 强度 / 规则信号都是逐品种的，训练才是合并的）。
    """
    cfg = cfg.normalized()
    keys = [k for k in (keys or []) if F.is_factor(k)]
    if len(keys) < 2:
        raise ValueError("机器学习多因子至少需要 2 个因子")

    trigger = str(cfg.sample_mode or "all") == "trigger"
    parts: list[pd.DataFrame] = []
    plans: dict[str, TriggerPlan] = {}
    info: dict[str, dict] = {}
    dropped: list[str] = []

    for sym, df in frames.items():
        # 跨周期因子：每个品种的「别的周期」行情是各自一份（`extras[sym]`）
        ex = (extras or {}).get(sym) or {}
        fmat, _raw, use, _gone, _miss = factor_matrix(
            df, keys, cfg, extra=ex, base_period=base_period)
        if len(use) < 2:
            dropped.append(sym)
            continue
        if fmat.empty:
            dropped.append(sym)
            continue
        vol = _pit_vol(df, cfg.window) * float(np.sqrt(max(1, cfg.horizon)))
        y = M.forward_return(df, cfg.horizon) / vol.where(vol > 1e-12)
        d = fmat.copy()
        d["__y"] = y.reindex(fmat.index)
        d["__sym"] = sym
        if trigger:
            st, armed, _tn, rule_sig = trigger_signal(
                df, cfg, extra=ex, base_period=base_period)
            m, _mn = trigger_mask(st, armed, cfg)
            if not m.any():
                # 这个品种一根都没触发：合训里没有它的样本，但预测照样要给它，
                # 否则多品种回测会凭空少掉一个品种
                m = np.zeros(len(df), dtype=bool)
            plans[sym] = TriggerPlan(
                mask=np.asarray(m, dtype=bool),
                strength=st.to_numpy(dtype=float),
                armed=armed.to_numpy(dtype=bool),
                rule_sig=rule_sig.to_numpy(dtype=float),
                factor_channel=(bool(cfg.trigger_keys)
                                and bool(np.isfinite(st.to_numpy(dtype=float)).any())),
                rule_channel=bool(cfg.trigger_rule) and cfg.trigger_rule in STRATEGIES,
            )
            d["__m"] = np.asarray(m, dtype=bool)
        else:
            d["__m"] = True
        info[sym] = {"fmat": fmat, "raw": _raw, "mask": plans.get(sym)}
        parts.append(d)

    if not parts:
        raise ValueError("这批品种上都算不出 2 个因子，没法合并训练")
    # 特征列取交集：某个品种缺这一列，整张长表就对不齐。补 0 不行 —— 标准化
    # 之后 0 是「中位数」，等于给这个品种凭空多出一个中性观测
    common = set(parts[0].columns) - {"__y", "__sym", "__m"}
    for p in parts[1:]:
        common &= set(p.columns) - {"__y", "__sym", "__m"}
    use = [k for k in keys if k in common]
    if len(use) < 2:
        raise ValueError("这批品种上没有共同可用的 2 个因子（各自缺的数据不一样）")

    long = pd.concat([p[use + ["__y", "__m", "__sym"]] for p in parts])
    long.index = pd.MultiIndex.from_arrays([long.index, long["__sym"]])
    long = long.sort_index()

    X = long[use].to_numpy(dtype=float)
    fin = np.isfinite(X)
    Xf = np.where(fin, X, 0.0)
    Y = long["__y"].to_numpy(dtype=float)
    MASK = long["__m"].to_numpy(dtype=bool)
    # 因子缺值按 0（标准化后的中位数）补，只把「缺得太多」的行剔掉
    need_cols = max(1, int(np.ceil(len(use) * 0.5)))
    usable = np.isfinite(Y) & (fin.sum(axis=1) >= need_cols)

    ts = long.index.get_level_values(0).unique()
    pos = np.searchsorted(ts, long.index.get_level_values(0).to_numpy())
    n_ts = len(ts)
    bpd = _bars_per_day(pd.DatetimeIndex(ts))
    train_bars = _to_bars(cfg.train_window, bpd)
    step_bars = _to_bars(cfg.step, bpd)
    min_bars = min(int(cfg.min_train), train_bars)
    if n_ts <= min_bars + cfg.horizon + 10:
        raise ValueError(
            f"样本太短：合并训练要 {min_bars:,} 根起训，这批品种只有 {n_ts:,} 根"
        )

    bounds = []
    t = min_bars
    while t < n_ts:
        bounds.append((max(0, t - train_bars), t, min(n_ts, t + step_bars)))
        t += step_bars

    pred = np.full(len(long), np.nan)
    imp_acc = np.zeros(len(use))
    imp_n = 0
    done = 0
    for a, b, c in bounds:
        if cancel and cancel():
            raise Cancelled()
        # 训练集只取到 b - horizon：第 i 行的标签要等到 i+h 才成立
        rows = np.nonzero((pos >= a) & (pos < b - cfg.horizon) & usable & MASK)[0]
        if len(rows) < 60:
            continue
        # 合并之后训练行数是 N 倍，抽稀一下控制耗时（和截面那条路同一个上限）
        stride = max(1, len(rows) // 20000)
        rows = rows[::stride]
        model = ML.build(cfg.model)
        try:
            model.fit(Xf[rows], Y[rows])
        except Exception:      # noqa: BLE001
            continue
        want = np.nonzero((pos >= b) & (pos < c))[0]
        try:
            pred[want] = model.predict(Xf[want])
        except Exception:      # noqa: BLE001
            continue
        if collect_models is not None:
            collect_models.append((model, int(b), int(c)))
        imp = model.importance()
        if imp is not None and len(imp) == len(use):
            imp_acc += np.asarray(imp, dtype=float)
            imp_n += 1
        done += 1
        if progress:
            progress(done, len(bounds), f"合并训练 第 {done}/{len(bounds)} 折")

    if cancel and cancel():
        raise Cancelled()

    imp_ser = None
    if imp_n:
        v = imp_acc / imp_n
        if v.sum() > 0:
            v = v / v.sum()
        imp_ser = pd.Series(v, index=use).sort_values(ascending=False)

    mlabel = ML.MODELS.get(cfg.model, {}).get("label", cfg.model)
    ser = pd.Series(pred, index=long.index, name="pred")
    notes = [
        f"{mlabel} · {len(parts)} 个品种合并训练，{done} 折滚动训练，"
        f"训练窗口 {cfg.train_window} 个交易日"
        f"（{train_bars:,} 根），训练样本 {int((usable & MASK).sum()):,} 行"
    ]
    if dropped:
        notes.append(f"以下品种因子不足 2 个，没进合并训练：{'、'.join(dropped)}")
    if trigger:
        notes.append("稀疏采样：训练样本仍按各品种自己的触发点挑，合并的只是样本池")
        flip = [n for n, on in (("因子", cfg.trigger_flip),
                                ("规则", cfg.trigger_rule_flip)) if on]
        if flip:
            notes.append("触发方向已反转：" + "、".join(flip))

    out: dict[str, ScorePack] = {}
    for sym in info:
        try:
            s = ser.xs(sym, level=1)
        except KeyError:
            continue
        s = s.reindex(frames[sym].index)
        score = _rolling_z(s, cfg.window).rename("score")
        plan = plans.get(sym)
        if plan is not None and len(plan.mask) == len(score):
            score = score.where(plan.mask).ffill().rename("score")
        out[sym] = ScorePack(
            score=score, fmat=info[sym]["fmat"][use], raw=info[sym]["raw"],
            pred=s, importance=imp_ser, folds=done,
            label=f"机器学习多因子 · {mlabel}（{len(parts)} 品种合并训练）",
            notes=list(notes), trigger=plan,
        )
    if not out:
        raise ValueError("合并训练没有产出任何品种的预测")
    return out


def run_pooled_strategy(
    frames: dict[str, pd.DataFrame],
    keys: list[str],
    cfg: FactorConfig,
    progress=None,
    cancel=None,
    collect_models: list | None = None,
    extras: dict[str, dict[str, pd.DataFrame]] | None = None,
    base_period: str = "",
) -> dict[str, FactorOutcome]:
    """多品种合并训练 + 各自交易。返回 `{品种: FactorOutcome}`。

    训练合并、交易分开：每个品种拿到的预测是同一个模型给的，但仓位、成本、
    回测各算各的 —— 否则「多品种等权」就变成了一个组合单。

    `extras` 是逐品种的跨周期行情（`{品种: {周期: 行情}}`），见 `period_frames`。
    """
    cfg = cfg.normalized()
    packs = pooled_score(frames, keys, cfg, progress=progress, cancel=cancel,
                         collect_models=collect_models,
                         extras=extras, base_period=base_period)
    out: dict[str, FactorOutcome] = {}
    for sym, pack in packs.items():
        df = frames[sym]
        if pack.trigger is not None:
            sig = trigger_hysteresis(pack.score.to_numpy(), pack.trigger,
                                     cfg.entry_z, cfg.exit_z)
        else:
            sig = hysteresis(pack.score.to_numpy(), cfg.entry_z, cfg.exit_z)
        # 出场规则逐品种各自判：ATR 是各品种自己的波动尺度，箱体也是各自的高低点
        exit_why: list[str] = []
        sig, exit_note = apply_exit_rules(
            sig, df, cfg, score=pack.score.to_numpy(dtype=float), entry=cfg.entry_z,
            reasons_out=exit_why)
        d = df.copy()
        d["factor_score"] = pack.score
        for c in pack.raw.columns:
            d[f"f_{c}"] = pack.raw[c]
        d["signal"] = sig
        d["entry_note"] = _entry_reasons(pack.score, sig, cfg.entry_z, pack.trigger)
        d["exit_note"] = exit_why if exit_why else ""
        notes = list(pack.notes)
        if exit_note:
            notes.append(exit_note)
        out[sym] = FactorOutcome(df=d, pack=pack, label=pack.label, notes=notes)
    return out


def apply_factor_filter(
    rule_signal: np.ndarray,
    score: np.ndarray,
    cfg: FactorConfig,
) -> np.ndarray:
    """规则信号 + 因子过滤。

    三种语义，差别很大，别混：

    - `mode="entry"`（否决）：只在规则**开仓那一刻**看因子。因子不同意就整笔
      跳过，等规则先平掉再重新开仓才给下一次机会。这是最严格的过滤，
      实测能把唐奇安在沪铜上的仓位从 96% 压到 60% 左右。
    - `mode="delay"`（延后）：因子不同意就先等着，等它转好了再进。只推迟入场，
      最终几乎每笔都会进去，对仓位影响很小 —— 想「筛掉烂交易」不要用这个。
    - `mode="hold"`（转差离场）：等同 delay，外加持仓期间因子一旦转向就立即平仓。
      对回撤的改善最明显，但换手会大幅上升。

    三种模式都只做减法：因子永远不会让规则开出它本来不想开的仓。
    """
    r = np.asarray(rule_signal, dtype=int)
    s = np.asarray(score, dtype=float)
    n = min(len(r), len(s))
    out = np.zeros(n, dtype=int)
    cur = 0
    mode = cfg.filter_mode
    hold = mode == "hold"
    strict = mode == "entry"
    lt, st = float(cfg.long_th), float(cfg.short_th)

    skip_long = False     # 本次规则多头已被因子否决（严格模式）
    skip_short = False

    for i in range(n):
        want = int(r[i])
        v = s[i]
        ok_long = np.isfinite(v) and v > lt
        ok_short = np.isfinite(v) and v < st

        # 规则换方向或平仓 → 清掉否决标记，下次开仓重新判断
        if want != 1:
            skip_long = False
        if want != -1:
            skip_short = False

        if cur == 1 and (want != 1 or (hold and not ok_long)):
            cur = 0
        elif cur == -1 and (want != -1 or (hold and not ok_short)):
            cur = 0

        if cur == 0:
            if want == 1 and not skip_long:
                if ok_long:
                    cur = 1
                elif strict:
                    skip_long = True
            elif want == -1 and not skip_short:
                if ok_short:
                    cur = -1
                elif strict:
                    skip_short = True
        out[i] = cur
    return out


def run_rule_with_filter(
    df: pd.DataFrame,
    rule_signal: np.ndarray,
    mode: str,
    keys: list[str],
    cfg: FactorConfig,
    progress=None,
    cancel=None,
    extra: dict[str, pd.DataFrame] | None = None,
    base_period: str = "",
) -> FactorOutcome:
    """规则策略 + 因子过滤。`rule_signal` 是原规则策略的输出。"""
    cfg = cfg.normalized()
    pack = build_score(df, mode, keys, cfg, progress=progress, cancel=cancel,
                       extra=extra, base_period=base_period)
    filtered = apply_factor_filter(rule_signal, pack.score.to_numpy(), cfg)

    out = df.copy()
    out["factor_score"] = pack.score
    out["rule_signal"] = np.asarray(rule_signal, dtype=int)
    for c in pack.raw.columns:
        out[f"f_{c}"] = pack.raw[c]
    out["signal"] = filtered
    # 备注列：进场原因是「规则 + 因子同意」，出场原因是规则自己走平 / 翻向
    # （过滤只做减法，不会另起一条出场通道）
    _sc = pack.score.to_numpy(dtype=float)
    _prev = np.r_[0, np.asarray(filtered, dtype=int)[:-1]]
    _why = np.empty(len(filtered), dtype=object)
    _why[:] = ""
    for i in range(len(filtered)):
        cur = int(filtered[i])
        if cur == 0 or cur == int(_prev[i]):
            continue
        side = "做多" if cur > 0 else "做空"
        v = float(_sc[i]) if i < len(_sc) else float("nan")
        _why[i] = (f"规则 + 因子过滤（{side}，因子分 {v:+.2f}）"
                   if np.isfinite(v) else f"规则 + 因子过滤（{side}）")
    out["entry_note"] = _why
    out["exit_note"] = ""

    base = int(np.abs(np.diff(np.asarray(rule_signal, dtype=int))).sum())
    kept = int(np.abs(np.diff(filtered)).sum())
    notes = list(pack.notes)
    notes.append(f"规则信号变动 {base} 次 → 过滤后 {kept} 次")
    return FactorOutcome(df=out, pack=pack, label=pack.label, notes=notes)


# ---------------------------------------------------------------- 评估
def outcome_report(df: pd.DataFrame, pack: ScorePack, cfg: FactorConfig) -> pd.DataFrame:
    """对参与合成的因子出 IC/IR 表（与「因子分析」页同一套口径）。

    直接拿 `pack.fmat` 里的列算，不按因子键重算一遍 —— 列名是**实例**
    （`mom_60@1w`），同一个键的两个周期是两列，重算的话两列会变成一模一样，
    还都会算在基准周期上。
    """
    fwd = M.forward_return(df, cfg.horizon)
    rows = []
    for c in pack.fmat.columns:
        base, _p = F.split_spec(c)
        spec = F.FACTORS.get(base)
        label = F.spec_label(c)
        category = spec.category if spec is not None else ""
        try:
            f = pack.fmat[c].astype(float)
            st = M.ic_stats(f, fwd, cfg.window)
            q = M.quantile_returns(f, fwd, 5, cfg.window)
        except Exception as exc:      # noqa: BLE001
            rows.append({"key": c, "label": label, "category": category,
                         "error": str(exc)[:60]})
            continue
        rows.append(
            {
                "key": c, "label": label, "category": category,
                "n": st["n"], "ic": st["ic"], "ic_mean": st["ic_mean"],
                "ic_std": st["ic_std"], "ir": st["ir"], "ic_pos": st["ic_pos"],
                "t": st["t"], "hit": st["hit"], "turnover": st["turnover"],
                "mono": M.monotonicity(q),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty and "ic" in out:
        out = out.reindex(out["ic"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)
