"""截面机器学习的**离线训练产物 + 实盘只预测**。

## 为什么不能直接照搬因子的 ML 实盘

`ttrade.ml_artifact` 那套是「每个品种学一个绝对收益模型」：预测值过自己的
滚动 z、过迟滞状态机，再出这个品种自己的多空信号。截面 ML 学的是**相对强弱**
（`cross.ml_score_panel`：每根 bar 上减去当期截面均值），实盘要的是「同一根
bar 上谁更强」的一行分数，不能各品种各自滚 z —— 那是另一条策略。

所以另立一份产物，但沿用同一个方法论：

    离线：cross.ml_score_panel 的 walk-forward（每折模型都收下来）
       ↓ 只留最后一折的模型 + 它用的特征列 / 训练配置
    产物：<用户目录>/ml_models/<sid>_cross.pkl
       ↓
    实盘：factor_frame 重算标准化因子（和训练同一段历史）→ model.predict
       ↓
    一篮子预测值 → 横截面 z-score → cross_live 原来的筛池 / 分钱链路

## 为什么可以用「最后一折」的模型

回测里每一折只负责 `[b, c)` 那一段；实盘越过训练末端后，现有因子 ML 也是
继续用最后一折模型，直到下次离线重训。这里保持一致：模型训练截止到
`b - horizon`（标签成熟前的最后一刻），之后的新 bar 只做预测，不偷偷重训 ——
重训要几十秒到几分钟，放在调仓那一刻等于拿旧价格下单。

新鲜度靠 `trained_at` / `train_until` 暴露给界面；配置或品种池改了则直接拒绝
加载产物，不拿 A 配置的模型去跑 B 配置。
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttrade import cross as CX
from ttrade import factor_strategy as fstrat
from ttrade import factors as F
from ttrade import ml_artifact as MLA
from ttrade import replay

ARTIFACT_VERSION = 1


def artifact_path(sid: str) -> str:
    """截面 ML 产物路径。和因子 ML 同目录，文件名带 `_cross` 防互相顶掉。"""
    return os.path.join(MLA.artifact_dir(),
                        f"{MLA._sid_to_name(sid)}_cross.pkl")


def remove_artifact(sid: str) -> bool:
    """重训前清旧产物。返回是不是真的删掉了一个文件。"""
    p = artifact_path(sid)
    if not os.path.exists(p):
        return False
    try:
        os.remove(p)
        return True
    except OSError:
        return False


def _digest(fcfg, keys: list[str], label_mode: str, symbols: list[str]) -> dict:
    """产物指纹：凡是会影响「这个模型还能不能用来给这一篮子打分」的都算。"""
    d = dict(MLA._cfg_digest(MLA._normalized(fcfg)))
    d.update({
        "keys": list(keys),
        "label_mode": str(label_mode or "volret"),
        "symbols": sorted(str(s) for s in symbols),
        "cross": True,
    })
    return d


@dataclass
class CrossMlArtifact:
    """一份截面 ML 产物。按**父策略 sid** 存，一份管整篮子。"""

    version: int = ARTIFACT_VERSION
    sid: str = ""
    name: str = ""
    model: object = None
    model_key: str = ""
    # 模型实际吃进去的特征列（交集，`ml_score_panel` 里算出来的）
    keys: list[str] = field(default_factory=list)
    # 配置里原本选的全部因子（训练时可能有个别品种算不出被交集剔掉）。
    # 校验「配置改没改」要比这个，不能比 keys —— 后者天然是它的子集。
    # 老产物没有这个字段，mismatch_reason 里按「keys 是配置的子集」兼容。
    requested_keys: list[str] = field(default_factory=list)
    label_mode: str = "volret"
    # 训练时 bind_window 后的配置：实盘重算因子必须用它，不能拿未绑定的配置
    fcfg: object = None
    symbols: list[str] = field(default_factory=list)
    cfg_digest: dict = field(default_factory=dict)
    cfg_hash: str = ""
    trained_at: float = 0.0
    train_until: str = ""          # 训练集用到哪根（时间）
    data_head: str = ""
    data_tail: str = ""
    n_bars: int = 0
    n_folds: int = 0
    train_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def save(self, path: str | None = None) -> str:
        p = path or artifact_path(self.sid)
        tmp = p + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        MLA._replace_retry(tmp, p)
        return p

    @staticmethod
    def load(path: str) -> "CrossMlArtifact":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, CrossMlArtifact):
            raise ValueError("这不是截面 ML 产物（类型不对）")
        if int(getattr(obj, "version", 0)) != ARTIFACT_VERSION:
            raise ValueError(f"截面 ML 产物版本 {obj.version} 不认识"
                             f"（当前 {ARTIFACT_VERSION}），请重新训练")
        return obj

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - float(self.trained_at or 0.0)) / 86400.0)

    def mismatch_reason(self, fcfg, keys, label_mode, symbols) -> str:
        """配置 / 品种池对不上就拒绝加载，别拿错模型下单。

        🔴 `keys` 要比的是「配置原本选了哪些因子」，不是「模型实际吃到几列」：
        训练时 `ml_score_panel` 按各品种的可用列取交集，某个因子（如
        roll_main_oi）在个别品种算不出就被剔掉 —— 产物 `keys` 天然是配置的子集
        （用户 2026-11-25：产物 5 列 / 配置 6 列，训练完立刻被拒绝启动）。

        新产物存了 `requested_keys`（配置全集），指纹也用它算，严格相等即放行。
        老产物（3.9.18 之前）没这个字段、指纹里存的是裁剪后的交集 —— 无法区分
        「训练时正常剔除」和「用户后来改了配置」，所以明确要求重训一次，不能靠
        「子集就放行」蒙混（那会让配置新增因子也通过，等于拿错模型下单）。
        """
        got_keys = set(dict(self.cfg_digest or {}).get("keys") or [])
        if not getattr(self, "requested_keys", None):
            return ("截面 ML 产物是旧格式（没记录配置原始因子），"
                    "请在组合页点「训练模型」重新训练一次即可启动")
        want = _digest(fcfg, keys, label_mode, symbols)   # keys = 当前配置
        got = dict(self.cfg_digest or {})
        if want == got:
            return ""
        diffs = []
        for k in sorted(set(want) | set(got)):
            if want.get(k) != got.get(k):
                diffs.append(f"{k}: 产物={got.get(k)!r} / 当前={want.get(k)!r}")
        return ("截面 ML 产物和当前配置对不上："
                + ("；".join(diffs[:4]) if diffs else "摘要不一致")
                + "。去组合页重新训练模型")

    def stale_note(self) -> str:
        d = self.age_days
        return f"截面 ML 产物 {d:.0f} 天前训练" if d >= 1 else "截面 ML 产物今天训练"


def train_artifact(frames: dict, sid: str, name: str, fcfg, keys: list[str],
                   label_mode: str = "volret", period: str = "",
                   base_period: str = "", progress=None, cancel=None,
                   save: bool = True) -> CrossMlArtifact:
    """离线训练一次，留最后一折模型。**和回测打分走同一个函数**。

    `collect` 是从 `cross.ml_score_panel` 里收出来的：特征列、折边界、绑定后的
    配置、每折模型。这里不重抄训练循环 —— 抄一遍就会出现「回测用 A、实盘用
    B，界面上一个字都不说」那种最难查的分叉。
    """
    collect: dict = {}
    CX.score_panel(frames, list(keys or []), mode="ml", fcfg=fcfg,
                   progress=progress, cancel=cancel, label_mode=label_mode,
                   base_period=base_period, ml_collect=collect)
    models = list(collect.get("models") or [])
    if not models:
        raise ValueError("截面 ML 一折都没训出来（样本不足或模型跑不起来）")
    model, _a, _b, _c, n_train = models[-1]
    if not MLA._picklable(model):
        raise ValueError(f"{model.__class__.__name__} 不能存成 .pkl，"
                         "换 LightGBM / XGBoost / GPR / 线性回归")

    keys_used = list(collect.get("keys") or [])
    # 配置原本选的全部因子（训练时按各品种可用列取交集，`keys_used` 是交集）。
    # 指纹必须用全集：校验时拿的是配置全集，拿交集去存就会出现「训练完立刻
    # 配置和产物对不上」的假红（用户 2026-11-25 油脂截面：roll_main_oi 被交集
    # 剔掉，产物 5 列 / 配置 6 列，启动时被拒）。
    keys_requested = list(keys or [])
    got_fcfg = collect.get("fcfg")
    fcfg_bound = got_fcfg if got_fcfg is not None else fstrat.bind_window(
        fcfg, next(iter(frames.values())).index)
    got_ts = collect.get("ts")
    ts = list(got_ts) if got_ts is not None else []
    b = int(models[-1][2])
    first = next((v for v in frames.values() if v is not None and len(v)), None)
    dig = _digest(fcfg, keys_requested, label_mode, list(frames))
    art = CrossMlArtifact(
        sid=str(sid or ""), name=str(name or ""),
        model=model, model_key=str(getattr(fcfg, "model", "") or ""),
        keys=keys_used, requested_keys=keys_requested,
        label_mode=str(label_mode or "volret"),
        fcfg=fcfg_bound, symbols=list(frames),
        cfg_digest=dig, cfg_hash=MLA._hash_obj(dig),
        trained_at=time.time(),
        train_until=(str(ts[b]) if b < len(ts) and b >= 0 else ""),
        data_head=(str(first.index[0]) if first is not None and len(first) else ""),
        data_tail=(str(first.index[-1]) if first is not None and len(first) else ""),
        n_bars=int(len(first)) if first is not None else 0,
        n_folds=len(models),
        train_seconds=0.0,   # 由调用方填（它才看得到真实起点）
        notes=[f"最后一折训练样本 {n_train:,} 行",
               f"训练截止 {ts[b] if b < len(ts) and b >= 0 else '?'}",
               f"横截面 {len(frames)} 个品种"],
    )
    if save:
        art.save()
    return art


def train_from_strategy(cfg, frames: dict, step2_frames: dict | None = None,
                        progress=None, cancel=None,
                        save: bool = True) -> CrossMlArtifact:
    """从组合页的 `StrategyConfig` 直接训练（它知道怎么还原 cross_config）。"""
    if str(getattr(cfg, "family", "") or "") != "cross":
        raise ValueError("这条不是截面策略")
    bt = replay.make_bt(replay.market_of(cfg),
                        replay.resolve_allow_short(cfg), cfg)
    ccfg = replay.cross_config(cfg, bt, period=cfg.period)
    if str(getattr(ccfg, "factor_mode", "") or "") != "ml":
        raise ValueError("这条截面策略的排名口径不是机器学习")
    fcfg = CX._factor_cfg(ccfg)
    # 排名腿配了独立周期时，模型和实盘 predict 都必须吃那一份行情，
    # 不能拿策略周期偷偷算一份（那又是两条策略）
    score_frames = step2_frames if (ccfg.step2_period and step2_frames) else frames
    t0 = time.time()
    art = train_artifact(
        score_frames, sid=str(cfg.sid), name=str(getattr(cfg, "name", "") or ""),
        fcfg=fcfg, keys=list(ccfg.factor_keys or []),
        label_mode=str(getattr(ccfg, "label_mode", "volret") or "volret"),
        period=str(cfg.period or ""),
        base_period=str(ccfg.step2_period or cfg.period or ""),
        progress=progress, cancel=cancel, save=save)
    art.train_seconds = time.time() - t0
    if save:
        art.save()
    return art


class CrossMlPredictor:
    """实盘侧：只预测，不训练。一篮子一次出一行横截面分数。"""

    def __init__(self, art: CrossMlArtifact) -> None:
        self.art = art
        self.model = art.model
        self.keys = list(art.keys or [])
        self.fcfg = art.fcfg
        if self.model is None or not self.keys or self.fcfg is None:
            raise ValueError("截面 ML 产物不完整（缺模型 / 特征列 / 训练配置）")

    @property
    def age_days(self) -> float:
        return self.art.age_days

    def predict(self, frames: dict) -> pd.DataFrame:
        """给出这一根 bar 上的横截面 z-score（和回测分数同口径）。"""
        parts: dict[str, float] = {}
        for key, df in (frames or {}).items():
            if df is None or df.empty:
                continue
            fm = F.factor_frame(df, self.keys,
                                window=int(getattr(self.fcfg, "window", 252) or 252),
                                method=str(getattr(self.fcfg, "method", "z") or "z"))
            if fm.empty:
                continue
            row = fm.iloc[-1][self.keys].to_numpy(dtype=float).reshape(1, -1)
            X = np.where(np.isfinite(row), row, 0.0)
            try:
                pred = np.asarray(self.model.predict(X), dtype=float).ravel()
            except Exception as exc:                  # noqa: BLE001
                raise ValueError(f"{key} 预测失败：{type(exc).__name__}: {exc}")
            if not len(pred) or not np.isfinite(pred[0]):
                continue
            parts[str(key)] = float(pred[0])
        if len(parts) < CX.MIN_CROSS:
            raise ValueError(f"这一根只有 {len(parts)} 个品种算得出 ML 预测"
                             f"（要 {CX.MIN_CROSS} 个）")
        # 这里不 import cross_live（它反过来会用本模块），自己取时间并集即可。
        # 注意别把上面装预测值的 `parts` 覆盖掉 —— 那会把一列 DatetimeIndex
        # 当成分数喂进 `_cross_z`，报出一个看起来毫不相干的 pandas 加法错。
        idxs = [f.index for f in (frames or {}).values()
                if f is not None and len(f)]
        idx = pd.DatetimeIndex([])
        for p in idxs:
            idx = idx.union(p) if len(idx) else pd.DatetimeIndex(p)
        idx = idx.sort_values()
        ts = idx[-1] if len(idx) else pd.Timestamp.now()
        panel = pd.Series(parts, name="pred").to_frame().T
        panel.index = pd.Index([ts])
        return CX._cross_z(panel).dropna(how="all")


def check_artifact(cfg) -> str:
    """启用前检查：产物在不在、配置对不对得上。返回空串表示可以跑。"""
    if str(getattr(cfg, "factor_mode", "") or "") != "ml":
        return ""
    p = artifact_path(str(getattr(cfg, "sid", "") or ""))
    if not os.path.exists(p):
        return ("这条截面机器学习策略还没有训练产物，"
                f"先在组合页点「训练模型」（{os.path.basename(p)}）")
    try:
        art = CrossMlArtifact.load(p)
    except Exception as exc:                          # noqa: BLE001
        return f"截面 ML 产物读不出来：{type(exc).__name__}: {exc}"
    fcfg = CX._factor_cfg(cfg)
    return art.mismatch_reason(fcfg, list(getattr(cfg, "factor_keys", None) or []),
                               getattr(cfg, "label_mode", "volret"),
                               list(getattr(cfg, "symbols", None) or []))


def load_for_live(cfg) -> tuple[CrossMlPredictor | None, str]:
    """给引擎用：加载产物并校验。失败时返回原因，由调用方决定怎么提示。"""
    why = check_artifact(cfg)
    if why:
        return None, why
    p = artifact_path(str(getattr(cfg, "sid", "") or ""))
    try:
        art = CrossMlArtifact.load(p)
        pred = CrossMlPredictor(art)
    except Exception as exc:                          # noqa: BLE001
        return None, f"截面 ML 产物装载失败：{type(exc).__name__}: {exc}"
    return pred, art.stale_note() if art.age_days >= 14 else ""
