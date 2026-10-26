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


class Cancelled(Exception):
    """训练/预测过程中被用户取消。"""


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
    train_window: int = 1000     # 滚动训练窗口
    step: int = 180              # 每隔多少根重训一次（180≈9 个月）
    min_train: int = 500         # 首次训练所需最少样本

    # 因子过滤
    filter_mode: str = "entry"   # entry（否决整笔）/ delay（延后入场）/ hold（转差离场）
    long_th: float = 0.0
    short_th: float = 0.0

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
        c.min_train = int(max(120, min(c.min_train, c.train_window)))
        return c


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
) -> ScorePack:
    """三种模式共用的分数构建。返回的 `score` 已是 PIT 标准化值。"""
    cfg = cfg.normalized()
    keys = [k for k in keys if k in F.FACTORS]
    if not keys:
        raise ValueError("没有选中任何可用因子")

    keys = F.filter_available(df, keys)
    if not keys:
        raise ValueError("选中的因子在该标的上都没有足够数据（如美股无持仓量）")

    fmat = F.factor_frame(df, keys, window=cfg.window, method=cfg.method)
    rawf = F.raw_frame(df, keys)
    fwd = M.forward_return(df, cfg.horizon)
    notes: list[str] = []

    if mode == "single":
        k = keys[0]
        ori = _orientation_frame(fmat[[k]], fwd, cfg)
        score = (fmat[k] * ori[k]).rename("score")
        rev = k in cfg.reversed_keys()
        pack = ScorePack(
            score=score, fmat=fmat, raw=rawf, orient=ori,
            label=f"单因子 · {F.FACTORS[k].label}" + ("（反转）" if rev else ""),
        )
        if cfg.orient == "auto":
            pack.notes.append("方向由扩张窗口 IC 决定（滞后 h 期，无未来信息）")
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
        if cfg.orient == "auto":
            pack.notes.append("各因子先按自身 PIT 方向取号，再等权平均")
        note = _reverse_note(cfg, keys)
        if note:
            pack.notes.append(note)
        return pack

    if mode == "ml":
        if len(keys) < 2:
            raise ValueError("机器学习多因子至少需要 2 个因子")
        pred, imp, n_folds = walk_forward(
            fmat, fwd, cfg, progress=progress, cancel=cancel
        )
        score = _rolling_z(pred, cfg.window).rename("score")
        covered = int(pred.notna().sum())
        notes.append(
            f"{ML.MODELS.get(cfg.model, {}).get('label', cfg.model)} · "
            f"{n_folds} 折滚动训练，有效预测 {covered}/{len(df)} 根"
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
            folds=n_folds, notes=notes,
        )

    raise ValueError(f"未知因子模式 {mode}")


# ---------------------------------------------------------------- Walk-forward
def walk_forward(
    fmat: pd.DataFrame,
    fwd: pd.Series,
    cfg: FactorConfig,
    progress=None,
    cancel=None,
) -> tuple[pd.Series, pd.Series | None, int]:
    """滚动训练 + 预测。

    时间轴（每折）：

        |<---- 训练窗口 train_window ---->|<-- 预测 step 根 -->|
        a                                b                    c

    训练样本只取到 `b - horizon`：第 i 行的标签 `fwd[i]` 要到 i+h 才成立，
    所以 i > b-h 的行在 b 时刻还是「未知标签」，混进训练集就是未来函数。

    返回 (预测序列, 因子重要度, 折数)。未覆盖区间为 NaN。
    """
    cfg = cfg.normalized()
    keys = list(fmat.columns)
    X = fmat.to_numpy(dtype=float)
    Y = fwd.to_numpy(dtype=float)
    n = len(fmat)
    pred = np.full(n, np.nan)
    if n <= cfg.min_train + cfg.horizon + 10:
        return pd.Series(pred, index=fmat.index), None, 0

    bounds = []
    t = cfg.min_train
    while t < n:
        bounds.append((max(0, t - cfg.train_window), t, min(n, t + cfg.step)))
        t += cfg.step

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
        ok = np.isfinite(Ytr) & np.all(np.isfinite(Xtr), axis=1)
        # 因子缺失允许用 0（标准化后的中位数）补齐，但标签缺失必须丢
        Xtr = np.where(np.isfinite(Xtr), Xtr, 0.0)
        ok &= np.isfinite(Ytr)
        if ok.sum() < 60:
            continue

        model = ML.build(cfg.model)
        try:
            model.fit(Xtr[ok], Ytr[ok])
        except Exception:
            # 单折失败不该让整条策略崩掉，跳过该折继续
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
) -> FactorOutcome:
    """因子策略：分数 → 阈值信号。"""
    cfg = cfg.normalized()
    pack = build_score(df, mode, keys, cfg, progress=progress, cancel=cancel)
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
