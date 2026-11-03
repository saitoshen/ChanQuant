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
@dataclass
class FactorConfig:
    # 因子构造
    window: int = 252            # 标准化窗口（也是 IC 统计窗口）
    method: str = "z"            # z / rank
    horizon: int = 5             # 预测/评估的前向收益期数

    # 阈值信号
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

    def reversed_keys(self) -> set[str]:
        """真正打开反转的因子。顺手把非真值过滤掉，调用方不用自己判。"""
        return {k for k, v in dict(self.reverse or {}).items() if v}

    def normalized(self) -> "FactorConfig":
        c = FactorConfig(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        # dict 是可变对象，复制一份，免得调用方事后改原配置影响已经算好的结果
        c.reverse = {k: True for k in c.reversed_keys()}
        c.window = int(max(60, c.window))
        c.horizon = int(max(1, c.horizon))
        c.entry_z = float(max(0.0, c.entry_z))
        c.exit_z = float(min(max(0.0, c.exit_z), c.entry_z))
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

    反转在这里生效，而不是在最后对综合分数取负号：多因子等权时，只反一个因子
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
    rev = cfg.reversed_keys() & set(ori.columns)
    if rev:
        ori = ori.copy()
        for c in rev:
            ori[c] = -ori[c]
    return ori


def _reverse_note(cfg: FactorConfig, keys: list[str]) -> str:
    """反转说明。写成「反转使用：A、B」，界面上直接看得出哪几个因子被反着用。"""
    rev = [k for k in keys if k in cfg.reversed_keys()]
    if not rev:
        return ""
    names = "、".join(F.FACTORS[k].label if k in F.FACTORS else k for k in rev)
    return f"反转使用：{names}（等效于把该因子取负号）"


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


# ---------------------------------------------------------------- 阈值状态机
def hysteresis(
    score: np.ndarray, entry: float, exit_: float
) -> np.ndarray:
    """带迟滞的阈值状态机。NaN 分数一律强制空仓（预热期不交易）。"""
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


def _rolling_z(s: pd.Series, window: int) -> pd.Series:
    """对模型预测值再做一次 PIT 标准化，让 entry_z 在各模型间可比。"""
    return F.standardize(s, window=window, method="z")


# ---------------------------------------------------------------- 分数构建
def build_score(
    df: pd.DataFrame,
    mode: str,
    keys: list[str],
    cfg: FactorConfig,
    progress=None,
    cancel=None,
    collect_models: list | None = None,
) -> ScorePack:
    """三种模式共用的分数构建。返回的 `score` 已是 PIT 标准化值。

    `collect_models` 传了就往里塞每一折的模型（见 `walk_forward`）——
    给「离线训练、实盘只预测」用。
    """
    cfg = cfg.normalized()
    keys = [k for k in keys if k in F.FACTORS]
    if not keys:
        raise ValueError("没有选中任何可用因子")

    picked = list(keys)
    keys = F.filter_available(df, keys)
    if not keys:
        raise ValueError("选中的因子在该标的上都没有足够数据（如美股无持仓量）")

    fmat = F.factor_frame(df, keys, window=cfg.window, method=cfg.method)
    rawf = F.raw_frame(df, keys)
    fwd = M.forward_return(df, cfg.horizon)
    notes: list[str] = []
    # 标签上的「N 个」是**过滤后**的数量。少掉几个不是配置被改了，而是这几个
    # 因子在当前标的 / 当前区间上没有足够观测（典型：展期、期权那类列只有
    # 本地库才有）。不说明一句，用户看到数字自己变，会以为策略被换了。
    gone = [k for k in picked if k not in set(keys)]
    if gone:
        notes.append(
            f"已剔除 {len(gone)} 个在该标的上没有足够数据的因子："
            + "、".join(F.FACTORS[k].label for k in gone)
        )

    if mode == "single":
        k = keys[0]
        ori = _orientation_frame(fmat[[k]], fwd, cfg)
        score = (fmat[k] * ori[k]).rename("score")
        rev = k in cfg.reversed_keys()
        pack = ScorePack(
            score=score, fmat=fmat, raw=rawf, orient=ori,
            label=f"单因子 · {F.FACTORS[k].label}" + ("（反转）" if rev else ""),
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
        return pack

    if mode == "multi":
        ori = _orientation_frame(fmat, fwd, cfg)
        signed = fmat * ori
        score = signed.mean(axis=1, skipna=True).rename("score")
        n_rev = len(cfg.reversed_keys() & set(keys))
        pack = ScorePack(
            score=score, fmat=fmat, raw=rawf, orient=ori,
            label=f"多因子等权 · {len(keys)} 个" + (f"（{n_rev} 个反转）" if n_rev else ""),
        )
        pack.notes.extend(notes)      # 「剔除了哪几个因子」这类说明
        if cfg.orient == "auto":
            pack.notes.append("各因子先按自身 PIT 方向取号，再等权平均")
            onote = _orient_note(ori, cfg)
            if onote:
                pack.notes.append(onote)
        note = _reverse_note(cfg, keys)
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
            st, armed, tnotes, rule_sig = trigger_signal(df, cfg)
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
            if plan.factor_channel and plan.rule_channel:
                ext = (f"出场：因子触发的仓位看触发强度回落到 {cfg.exit_z:g} 以内，"
                       "规则触发的仓位跟规则自己的 signal")
            elif plan.factor_channel:
                ext = f"出场：触发强度回落到 {cfg.exit_z:g} 以内，每根都判"
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
        if cfg.reversed_keys() & set(keys):
            notes.append("机器学习自行学习方向，反转开关在 ML 模式下不生效")
        return ScorePack(
            score=score, fmat=fmat, raw=rawf, pred=pred, importance=imp,
            label=f"机器学习多因子 · {ML.MODELS.get(cfg.model, {}).get('label', cfg.model)}",
            folds=n_folds, notes=notes, trigger=plan,
        )

    raise ValueError(f"未知因子模式 {mode}")


def trigger_signal(df: pd.DataFrame, cfg: FactorConfig):
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
        if key not in F.FACTORS:
            continue
        try:
            raw = F.raw(df, key)
        except Exception:      # noqa: BLE001
            continue
        if raw is None or not len(raw):
            continue
        z = _rolling_z(raw.astype(float), cfg.window)
        if z.notna().sum() >= 60:
            zs.append(z.rename(key))
    if zs:
        strength = pd.concat(zs, axis=1).mean(axis=1, skipna=True)
        if cfg.trigger_flip:
            strength = -strength
        notes.append("触发：因子 " + "、".join(F.FACTORS[k].label
                                              for k in (cfg.trigger_keys or [])
                                              if k in F.FACTORS)
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


# ---------------------------------------------------------------- 策略入口
def run_factor_strategy(
    df: pd.DataFrame,
    mode: str,
    keys: list[str],
    cfg: FactorConfig,
    progress=None,
    cancel=None,
    collect_models: list | None = None,
) -> FactorOutcome:
    """因子策略：分数 → 阈值信号。

    触发模式（稀疏采样）走的是另一套状态机：进场只在触发点、方向由触发定、
    出场看触发强度（见 `trigger_hysteresis`）。单因子 / 多因子没有触发点，
    仍是分数过阈值那一套。

    `collect_models` 传了就收集每一折的模型（见 `walk_forward`）——
    给「离线训练、实盘只预测」用。
    """
    cfg = cfg.normalized()
    pack = build_score(df, mode, keys, cfg, progress=progress, cancel=cancel,
                       collect_models=collect_models)
    if mode == "ml" and pack.trigger is not None:
        sig = trigger_hysteresis(pack.score.to_numpy(), pack.trigger,
                                 cfg.entry_z, cfg.exit_z)
    else:
        sig = hysteresis(pack.score.to_numpy(), cfg.entry_z, cfg.exit_z)

    out = df.copy()
    out["factor_score"] = pack.score
    for c in pack.raw.columns:
        out[f"f_{c}"] = pack.raw[c]
    out["signal"] = sig

    notes = list(pack.notes)
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
    keys = [k for k in (keys or []) if k in F.FACTORS]
    if len(keys) < 2:
        raise ValueError("机器学习多因子至少需要 2 个因子")

    trigger = str(cfg.sample_mode or "all") == "trigger"
    parts: list[pd.DataFrame] = []
    plans: dict[str, TriggerPlan] = {}
    info: dict[str, dict] = {}
    dropped: list[str] = []

    for sym, df in frames.items():
        use = F.filter_available(df, keys)
        if len(use) < 2:
            dropped.append(sym)
            continue
        fmat = F.factor_frame(df, use, window=cfg.window, method=cfg.method)
        if fmat.empty:
            dropped.append(sym)
            continue
        vol = _pit_vol(df, cfg.window) * float(np.sqrt(max(1, cfg.horizon)))
        y = M.forward_return(df, cfg.horizon) / vol.where(vol > 1e-12)
        d = fmat.copy()
        d["__y"] = y.reindex(fmat.index)
        d["__sym"] = sym
        if trigger:
            st, armed, _tn, rule_sig = trigger_signal(df, cfg)
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
        info[sym] = {"fmat": fmat, "raw": F.raw_frame(df, use), "mask": plans.get(sym)}
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
) -> dict[str, FactorOutcome]:
    """多品种合并训练 + 各自交易。返回 `{品种: FactorOutcome}`。

    训练合并、交易分开：每个品种拿到的预测是同一个模型给的，但仓位、成本、
    回测各算各的 —— 否则「多品种等权」就变成了一个组合单。
    """
    cfg = cfg.normalized()
    packs = pooled_score(frames, keys, cfg, progress=progress, cancel=cancel,
                         collect_models=collect_models)
    out: dict[str, FactorOutcome] = {}
    for sym, pack in packs.items():
        df = frames[sym]
        if pack.trigger is not None:
            sig = trigger_hysteresis(pack.score.to_numpy(), pack.trigger,
                                     cfg.entry_z, cfg.exit_z)
        else:
            sig = hysteresis(pack.score.to_numpy(), cfg.entry_z, cfg.exit_z)
        d = df.copy()
        d["factor_score"] = pack.score
        for c in pack.raw.columns:
            d[f"f_{c}"] = pack.raw[c]
        d["signal"] = sig
        out[sym] = FactorOutcome(df=d, pack=pack, label=pack.label,
                                 notes=list(pack.notes))
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
) -> FactorOutcome:
    """规则策略 + 因子过滤。`rule_signal` 是原规则策略的输出。"""
    cfg = cfg.normalized()
    pack = build_score(df, mode, keys, cfg, progress=progress, cancel=cancel)
    filtered = apply_factor_filter(rule_signal, pack.score.to_numpy(), cfg)

    out = df.copy()
    out["factor_score"] = pack.score
    out["rule_signal"] = np.asarray(rule_signal, dtype=int)
    for c in pack.raw.columns:
        out[f"f_{c}"] = pack.raw[c]
    out["signal"] = filtered

    base = int(np.abs(np.diff(np.asarray(rule_signal, dtype=int))).sum())
    kept = int(np.abs(np.diff(filtered)).sum())
    notes = list(pack.notes)
    notes.append(f"规则信号变动 {base} 次 → 过滤后 {kept} 次")
    return FactorOutcome(df=out, pack=pack, label=pack.label, notes=notes)


# ---------------------------------------------------------------- 评估
def outcome_report(df: pd.DataFrame, pack: ScorePack, cfg: FactorConfig) -> pd.DataFrame:
    """对参与合成的因子出 IC/IR 表（与「因子分析」页同一套口径）。"""
    keys = list(pack.fmat.columns)
    return M.factor_report(
        df, keys, horizon=cfg.horizon, window=cfg.window, method=cfg.method
    )
