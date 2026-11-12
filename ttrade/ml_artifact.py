"""机器学习训练产物（.pkl）：离线训练 → 落盘 → 实盘只预测。

## 为什么要有这个模块

实盘每根 bar 都要出一次信号。机器学习策略如果每次都重跑一遍 walk-forward
（48 折滚动训练），一个品种实测 **34 秒**（65 因子 × 26 万根 5m），19 个品种
串行一轮就是 **11 分钟** —— 5 分钟的 bar 周期根本追不上，信号永远滞后一根。

实测过并行：8 核机器上 4 个任务并行，串行 54.28s → 并行 25.28s，**加速比只有
2.15x**（LightGBM 的 `fit` 是 C 扩展但没完全释放 GIL）。19 个品种并行最多把
11 分钟压到 5 分钟上下，仍然追不上，而且吃满 CPU。

所以路线是：**训练离线做一次，产物落盘；实盘只加载 + 预测**。

    离线（研究页 / 命令行）                实盘（组合页）
    ─────────────────────                ─────────────────────
    walk_forward 48 折                     加载 .pkl
      ↓ 只留最后一折的模型                   ↓
    打包：模型 + 配置 + 状态                factor_frame（3s，确定性）
      ↓                                   ↓
    artifacts/<sid>.pkl                    predict（毫秒级）
                                          ↓
                                        迟滞状态机续算 → 信号

## 为什么实盘还要重算因子

模型吃的是**标准化后的因子**（`factors.standardize`，滚动 z，窗口 252 根）。
这个标准化是 PIT 的：每一根的值取决于它之前 252 根的分布，所以实盘必须用
**和训练时同一段历史**重算一遍才对得上。

这一遍要 3.02 秒（261441 根 × 65 因子），可以接受 —— 比 34 秒好一个数量级。

**季节因子（`season_avg` / `season_ir` / `season_rank`）的 raw lookback 高达
78111 根（≈3.4 年）**：它们要「往年同月」的 expanding 统计（`MIN_YEARS = 3`）。
所以实盘的历史不能截短，截了季节因子会整列 NaN。产物里记的是「训练用到
哪一天」，实盘拿同一段历史往前接。

## 产物里装了什么

- `model` —— 训练好的模型本体（LightGBM / XGBoost / GPR / Linear 可 pickle；
  Transformer / GNN 内部用闭包反传，**不可 pickle**，见 `_picklable()`）
- `cfg` —— 训练时的 `FactorConfig` 快照 + 因子键列表（加载时校验，防错配）
- `state` —— 续算需要的状态：
  - `pred_tail`：最后一折预测值的尾巴（`_rolling_z` 的滚动窗口要用）
  - `score_tail`：标准化后分数的尾巴
  - `plan_tail`：触发计划的尾巴（mask / strength / armed / rule_sig）
  - `hyst`：迟滞状态机的 `cur` / `src`
- `meta` —— 训练时间戳、训练截止那根 bar 的时间、数据指纹、折数、耗时

## 错配怎么办

产物里记了 `train_keys`（因子键）、`cfg_hash`（配置摘要）、`data_head` /
`data_tail`（首尾 bar 时间）。加载时对不上就**拒绝加载并说明差在哪** —— 拿着
A 配置的模型去跑 B 配置，出来的信号没有任何意义，还很难查。

## 过期怎么办

模型是某时刻训练的。用 `meta["trained_at"]` 和 `meta["train_until"]` 判断新鲜度：
- 超过 `stale_days`（默认 14 天）：实盘仍能跑，但界面上标「模型偏旧，建议重训」
- 数据指纹对不上（历史被重下过 / 品种换了）：直接拒绝加载
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field, replace as _replace

import numpy as np
import pandas as pd

def _replace_retry(tmp: str, dst: str, tries: int = 6) -> None:
    """os.replace 加个小重试：Windows 上杀软会短暂占住刚写完的文件。

    症状是 PermissionError: [WinError 5] 拒绝访问: x.pkl.tmp -> x.pkl，
    而且只发生在**进程里第一次**写这种文件时（后面扫描器已经认得它了），
    所以看起来像「训练产物那一项偶尔红」。不是权限问题，重试几十毫秒就过去了。
    """
    for i in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05 * (i + 1))


from ttrade import factor_metrics as M
from ttrade import factor_strategy as fstrat
from ttrade import factors as F
from ttrade import paths

# 产物格式版本。改产物结构时 +1，旧产物加载时直接拒绝（而不是读出一堆
# 语义错位的数据）。版本号写在文件名旁边，排查时一眼看得出。
ARTIFACT_VERSION = 1

# 产物目录（用户可写目录，打包后也能写）
ARTIFACT_SUBDIR = "ml_models"

# 默认「模型算旧了」的阈值（天）
DEFAULT_STALE_DAYS = 14


# ================================================================== 折产物
@dataclass
class FoldArtifact:
    """一折的模型 + 它负责预测的 bar 区间。

    **为什么要一折一个文件**：实盘要「滚动调用这些 .pkl」——bar 落在哪一折的
    区间里，就用哪一折的模型。把每折拆成独立文件，实盘就能只加载**当前这一折**
    （或者最后几折），不用把 48 个模型全塞进内存。

    区间是**时间戳**（`t_from` / `t_to`），不是 bar 位置 —— bar 位置依赖
    「从哪根开始截历史」，实盘和回测的截法可能差几根；时间戳不会。
    """

    index: int = 0
    model: object = None
    model_key: str = ""
    t_from: str = ""              # 预测区间起点（bar 时间）
    t_to: str = ""                # 预测区间止（bar 时间，右开）
    n_train: int = 0              # 这一折用了多少训练样本
    train_from: str = ""          # 训练集起点（bar 时间）
    train_to: str = ""            # 训练集终点（bar 时间，右开）
    fallback: bool = False        # 触发样本不够，这折退回了全样本

    def to_dict(self) -> dict:
        return {
            "index": int(self.index),
            "model": self.model,
            "model_key": self.model_key,
            "t_from": self.t_from,
            "t_to": self.t_to,
            "n_train": int(self.n_train),
            "train_from": self.train_from,
            "train_to": self.train_to,
            "fallback": bool(self.fallback),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FoldArtifact":
        d = d or {}
        return cls(
            index=int(d.get("index", 0) or 0),
            model=d.get("model"),
            model_key=str(d.get("model_key", "") or ""),
            t_from=str(d.get("t_from", "") or ""),
            t_to=str(d.get("t_to", "") or ""),
            n_train=int(d.get("n_train", 0) or 0),
            train_from=str(d.get("train_from", "") or ""),
            train_to=str(d.get("train_to", "") or ""),
            fallback=bool(d.get("fallback", False)),
        )

    def covers(self, ts: str) -> bool:
        """这一折负责预测 `ts` 这根 bar 吗。

        区间是**右开**的 `[t_from, t_to)` —— 和 `walk_forward` 的 `pred[b:c]`
        一个口径。相邻两折在边界上不重叠。
        """
        if not self.t_from:
            return False
        if ts < self.t_from:
            return False
        if self.t_to and ts >= self.t_to:
            return False
        return True


def fold_path(sid: str, index: int) -> str:
    """某一折的产物路径。"""
    return os.path.join(artifact_dir(), f"{_sid_to_name(sid)}_f{index:03d}.pkl")


# ================================================================== 工具
def _idx_key(ts) -> str:
    """时间戳 → 可比较的字符串键。"""
    return "" if ts is None else str(ts)


def artifact_dir() -> str:
    """产物目录。必须落在用户可写目录 —— 打包后 `bundle_root()` 是只读的。"""
    d = os.path.join(paths.user_dir(), ARTIFACT_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def _sid_to_name(sid: str) -> str:
    """sid → 文件名（**必须可逆**）。

    引擎里多品种策略的子槽位 sid 是 `父sid#品种_交易所`（见
    `strategy_runner.GROUP_SEP`）。产物文件名要能从这个 sid 找回来 ——
    所以不能简单地把非法字符删掉：`AG_SHFE` 和 `AGSHFE` 删完就撞了，
    一个品种的模型会被另一个品种顶掉。

    这里把 `#` 编成 `__H__`（哈希分隔符），其余非字母数字的换成 `_`。
    编码后仍然是人能认出来的（`334ce0afa6f3__H__AG_SHFE`）。
    """
    s = str(sid or "")
    s = s.replace("#", "__H__")
    out = []
    for ch in s:
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)[:96].strip("_") or "unnamed"
    return s


def artifact_path(sid: str) -> str:
    """某个策略 sid 的产物路径。"""
    return os.path.join(artifact_dir(), f"{_sid_to_name(sid)}.pkl")


def list_artifacts() -> list[dict]:
    """列出磁盘上所有**主产物**（给界面选）。

    折文件（`<sid>_f000.pkl`）不在这里 —— 它们由主产物的清单管理，
    单独列出来只会让界面上一堆看不懂的条目。
    """
    out = []
    d = artifact_dir()
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".pkl"):
            continue
        if _is_fold_file(fn):
            continue
        p = os.path.join(d, fn)
        try:
            info = peek(p)
        except Exception as exc:  # noqa: BLE001
            out.append({"path": p, "name": fn, "ok": False,
                        "error": f"{type(exc).__name__}: {exc}"})
            continue
        out.append(info)
    return out


def _is_fold_file(name: str) -> bool:
    """这是不是一个折文件（`<sid>_f000.pkl`）。"""
    stem = name[:-4] if name.endswith(".pkl") else name
    tail = stem.rsplit("_f", 1)[-1] if "_f" in stem else ""
    return bool(tail) and tail.isdigit()


def remove_artifact(sid: str) -> bool:
    p = artifact_path(sid)
    if not os.path.exists(p):
        return False
    os.remove(p)
    return True


def _picklable(model) -> bool:
    """这个模型能不能 pickle。

    `TransformerModel` / `GNNModel` 的参数是 `Var` 对象，而 `Var._bp` 存的是
    **闭包**（局部函数）—— 闭包不能被 pickle。所以这两种模型产不出 .pkl。

    不是所有模型都这样：LightGBM / XGBoost / sklearn GPR 存的是 C 扩展 /
    sklearn 对象，都能 pickle；`LinearModel` 是闭式解，也行。
    """
    try:
        pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001
        return False
    return True


def _hash_obj(obj) -> str:
    """稳定摘要。dict 按 key 排序，list 保序，标量用 repr。"""
    if isinstance(obj, dict):
        s = json.dumps({str(k): _hash_obj(v) for k, v in sorted(
            obj.items(), key=lambda kv: str(kv[0]))}, sort_keys=True, default=str)
    elif isinstance(obj, (list, tuple)):
        s = json.dumps([_hash_obj(v) for v in obj], default=str)
    else:
        s = repr(obj)
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:16]


# ================================================================== 状态
@dataclass
class LiveState:
    """实盘续算需要的全部状态。

    - `pred_tail` / `score_tail`：`_rolling_z` 的滚动窗口要够长的尾巴，否则
      实盘刚接上时 z 分数全是 NaN
    - `plan_*`：触发模式的决策点 / 强度 / 规则信号
    - `cur` / `src`：迟滞状态机（持仓方向 + 是哪条通道开的）
    """

    pred_tail: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    score_tail: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    plan_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    plan_strength: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    plan_armed: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    plan_rule_sig: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    cur: int = 0
    src: str = ""
    # 观点的「持有值」：触发模式下只在决策点刷新，决策点之间沿用上一次的值。
    # 实盘接上之后第一根如果拿到的是「当前这根的原始 z」而不是「上次决策点
    # 的 z」，信号就和回测对不上（实测能差 0.4 个 z，足够翻多空）
    held_score: float = float("nan")
    # 迟滞状态机的「上一根分数」判断依据（触发模式用强度，单因子模式用分数）
    last_ts: str = ""

    def to_dict(self) -> dict:
        return {
            "pred_tail": np.asarray(self.pred_tail, dtype=float),
            "score_tail": np.asarray(self.score_tail, dtype=float),
            "plan_mask": np.asarray(self.plan_mask, dtype=bool),
            "plan_strength": np.asarray(self.plan_strength, dtype=float),
            "plan_armed": np.asarray(self.plan_armed, dtype=bool),
            "plan_rule_sig": np.asarray(self.plan_rule_sig, dtype=float),
            "cur": int(self.cur),
            "src": str(self.src),
            "held_score": float(self.held_score),
            "last_ts": str(self.last_ts),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LiveState":
        d = d or {}
        return cls(
            pred_tail=np.asarray(d.get("pred_tail", []), dtype=float),
            score_tail=np.asarray(d.get("score_tail", []), dtype=float),
            plan_mask=np.asarray(d.get("plan_mask", []), dtype=bool),
            plan_strength=np.asarray(d.get("plan_strength", []), dtype=float),
            plan_armed=np.asarray(d.get("plan_armed", []), dtype=bool),
            plan_rule_sig=np.asarray(d.get("plan_rule_sig", []), dtype=float),
            cur=int(d.get("cur", 0) or 0),
            src=str(d.get("src", "") or ""),
            held_score=float(d.get("held_score", float("nan"))),
            last_ts=str(d.get("last_ts", "") or ""),
        )


# ================================================================== 产物
@dataclass
class MlArtifact:
    """一份训练产物。"""

    version: int = ARTIFACT_VERSION
    sid: str = ""
    name: str = ""
    symbol: str = ""
    exchange: str = ""
    period: str = ""

    model: object = None
    model_key: str = ""
    # **请求的**因子键（用户配置里勾的那些）
    req_keys: list = field(default_factory=list)
    # **实际训练的**因子键（`filter_available` 剔掉算不出值的之后剩下的）。
    # 两者可能不一样 —— 比如美股没有持仓量，展期 / 期权那批因子算不出来。
    # 校验要对 `req_keys`：剔哪些是数据决定的，不是配置决定的
    train_keys: list = field(default_factory=list)
    cfg: dict = field(default_factory=dict)
    cfg_hash: str = ""
    state: LiveState = field(default_factory=LiveState)

    trained_at: float = 0.0
    train_until: str = ""       # 训练用到的最后一根 bar 的时间
    data_head: str = ""         # 训练数据第一根 bar 的时间
    data_tail: str = ""
    n_bars: int = 0
    folds: int = 0
    train_seconds: float = 0.0
    importance: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    # 折清单：`[{index, file, t_from, t_to, n_train, ...}]`。
    # 每折的模型本体在**独立文件**里，这里只记索引和它负责的时间区间 ——
    # 实盘按 bar 时间查清单，命中哪一折就加载哪一个文件
    folds_meta: list = field(default_factory=list)

    # ---- 合并训练（pooled）专用 ----
    # 合训时一个模型管 N 个品种。模型只有一份（上面的 `model` / 折清单），
    # 但每个品种的**续算状态**（迟滞状态机、z 缓冲尾巴、触发计划尾巴）各自
    # 独立 —— 实盘每个品种各建一个 `LivePredictor`，共用模型，各自带状态。
    pooled: bool = False
    # 合训时每个品种的状态：`{"AG_SHFE": LiveState, "AU_SHFE": LiveState, ...}`
    # 逐品种训练时这个字段为空（状态在 `state` 里）
    states: dict = field(default_factory=dict)
    # 合训时每个品种的 symbol / exchange（产物主字段只记一个代表性品种）
    sym_meta: dict = field(default_factory=dict)

    # ---- 存取 ----

    def save(self, path: str | None = None) -> str:
        p = path or artifact_path(self.sid)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        # 先写临时文件再改名：写到一半崩了不会留下半个坏产物
        _replace_retry(tmp, p)
        return p

    @classmethod
    def load(cls, path: str) -> "MlArtifact":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, MlArtifact):
            raise ValueError("这个文件不是机器学习训练产物")
        if int(getattr(obj, "version", 0)) != ARTIFACT_VERSION:
            raise ValueError(
                f"产物格式版本对不上（文件 v{getattr(obj, 'version', 0)}，"
                f"当前程序只认 v{ARTIFACT_VERSION}）—— 重新训练一次"
            )
        return obj

    # ---- 校验 ----

    def mismatch_reason(
        self,
        cfg: "fstrat.FactorConfig",
        keys: list[str],
    ) -> str:
        """和当前配置对不上时，返回原因；完全对得上返回空串。

        对不上的三种情况：
        1. 因子键不一样（换了因子集，模型输入维度都对不上）
        2. 配置摘要不一样（改了阈值 / 训练窗口 / 触发设置）
        3. 模型类型不一样

        **比对的是「请求的因子键」**（`req_keys`），不是「实际训练的」。
        训练时 `filter_available` 会剔掉在这个标的上算不出值的因子（比如美股
        没有持仓量，展期 / 期权那批全是 NaN）—— 那是**数据**决定的，不是配置
        决定的，两次训练剔掉的必然一样（同一份数据）。拿剔过的集合去和用户
        配置比，会永远对不上。

        **配置摘要必须两边都 `normalized()` 之后再比。** 训练那条路走的是
        `cfg.normalized()`（它会把 `train_window` 抬到不低于 `min_train`，
        比如用户填 120、`min_train` 是 200 → 实际按 200 训），而实盘这条路上
        传进来的是用户原始配置 —— 不归一化就比，会出现「配置一个字没改，
        却报配置对不上」的假警报，实盘直接退回重训。
        """
        if self.model_key != getattr(cfg, "model", ""):
            return (f"模型类型对不上：产物是 {self.model_key}，"
                    f"当前配置是 {getattr(cfg, 'model', '')}")
        base = self.req_keys or self.train_keys or []
        want = sorted(k for k in (keys or []) if k)
        have = sorted(k for k in (base or []) if k)
        if want != have:
            gone = [k for k in have if k not in want]
            add = [k for k in want if k not in have]
            bits = []
            if gone:
                bits.append("少 " + "、".join(gone[:4]))
            if add:
                bits.append("多 " + "、".join(add[:4]))
            return "因子集对不上（" + "；".join(bits) + "）"
        # 两边都归一化再比 —— 见上面 docstring
        h = _hash_obj(_cfg_digest(_normalized(cfg)))
        if self.cfg_hash and h != self.cfg_hash:
            return ("配置对不上（阈值 / 训练窗口 / 触发设置改过了）—— "
                    "改了就该重训，旧模型不再对得上")
        return ""

    def age_days(self) -> float:
        if not self.trained_at:
            return float("inf")
        return (time.time() - float(self.trained_at)) / 86400.0

    def stale(self, days: int = DEFAULT_STALE_DAYS) -> bool:
        return self.age_days() > float(days)

    def info(self) -> dict:
        """给界面看的一行摘要。"""
        d = {
            "path": artifact_path(self.sid),
            "ok": True,
            "sid": self.sid,
            "name": self.name,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "period": self.period,
            "model": self.model_key,
            "n_keys": len(self.train_keys or []),
            "trained_at": self.trained_at,
            "age_days": self.age_days(),
            "train_until": self.train_until,
            "n_bars": self.n_bars,
            "folds": self.folds,
            "train_seconds": self.train_seconds,
            "stale": self.stale(),
            "notes": list(self.notes or []),
        }
        return d


def peek(path: str) -> dict:
    """只读元信息（不加载模型本体）。"""
    try:
        a = MlArtifact.load(path)
    except Exception as exc:  # noqa: BLE001
        return {"path": path, "name": os.path.basename(path), "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}
    d = a.info()
    d["path"] = path
    d["name"] = os.path.basename(path)
    return d


# ================================================================== 配置摘要
def _normalized(cfg):
    """配置归一化（带兜底）。

    训练那条路和实盘那条路**必须**用同一份归一化结果去算摘要，否则会出现
    「配置没改却报对不上」。`FactorConfig.normalized()` 存在就用它，不存在
    就原样返回（老版本兼容）。
    """
    try:
        return cfg.normalized()
    except Exception:  # noqa: BLE001
        return cfg


def _cfg_digest(cfg: "fstrat.FactorConfig") -> dict:
    """参与「模型是否还对得上」判断的配置项。

    **不是全部字段**：像 `trigger_params`（触发规则的参数）改了不影响模型本身
    （模型只看因子），但会影响触发计划，所以也该算进去。反过来 `entry_z` /
    `exit_z` 只影响状态机不影响模型 —— 但改了之后实盘和回测的信号就不同了，
    所以也算。保守一点：凡是影响**输出信号**的都算进去。
    """
    return {
        "model": cfg.model,
        "window": int(cfg.window),
        "method": cfg.method,
        "horizon": int(cfg.horizon),
        "train_window": int(cfg.train_window),
        "step": int(cfg.step),
        "min_train": int(cfg.min_train),
        "sample_mode": cfg.sample_mode,
        "trigger_keys": sorted(cfg.trigger_keys or []),
        "trigger_rule": cfg.trigger_rule,
        "trigger_params": dict(cfg.trigger_params or {}),
        "trigger_side": cfg.trigger_side,
        "trigger_z": float(cfg.trigger_z),
        "trigger_tiers": int(cfg.trigger_tiers),
        "trigger_span": float(cfg.trigger_span),
        "trigger_cool": int(cfg.trigger_cool),
        "trigger_flip": bool(cfg.trigger_flip),
        "trigger_rule_flip": bool(cfg.trigger_rule_flip),
        "entry_z": float(cfg.entry_z),
        "exit_z": float(cfg.exit_z),
        "pooled": bool(getattr(cfg, "pooled", False)),
        # 出场规则（`exit_*`）**故意不算**：它只影响平仓时点，不参与训练，
        # 也不影响模型本身。算进去的话，改一个跟踪止损倍数就得重训十分钟的
        # 模型 —— 那是纯粹白等。实盘那边 `LivePredictor` 每根都会按当前配置
        # 重算出场，本来就不依赖产物里存的那份。
    }


# ================================================================== 训练
def _no_cross_periods(cfg, keys=None) -> None:
    """产物这条路还不支持跨周期因子 —— 拦住，而不是算出一份对不上的特征。

    产物里的模型是「按基准周期的行情算出来的特征」训出来的；实盘 `predict`
    只拿得到基准周期的行情。跨周期因子要成立，得把别的周期的行情也塞进产物
    （或者让实盘自己去读），那是另一件事。在训练这一步就报错，好过实盘上
    「模型突然不灵了」—— 后者要查很久。

    两种写法都要认：老配置的 `factor_periods` 映射，和新配置里带 `@周期` 的
    因子实例 id（见 `factors.split_spec`）。
    """
    hit = dict(getattr(cfg, "factor_periods", None) or {})
    for k in (keys or []):
        p = F.split_spec(k)[1]
        if p:
            hit[str(k)] = p
    if hit:
        raise ValueError(
            "跨周期因子的机器学习暂不支持「离线训练产物」："
            "回测和实盘实时重算（后台训练）都能用，"
            "但要把跨周期因子做进产物，得把别的周期的行情一起存进模型文件。"
            "先按「不生成产物」的方式跑，或者把因子改回本周期"
        )


def train_artifact(
    df: pd.DataFrame,
    sid: str,
    cfg: "fstrat.FactorConfig",
    keys: list[str],
    name: str = "",
    symbol: str = "",
    exchange: str = "",
    period: str = "",
    progress=None,
    cancel=None,
    save: bool = True,
) -> MlArtifact:
    """离线训练一次，产出一份可落盘的产物。

    走的是**和回测同一个函数**（`run_factor_strategy`）—— 同源同参是硬要求。
    两边各写一份，「回测赚钱、实盘亏钱」就永远说不清是谁的问题。

    `walk_forward` 会训 48 折，但实盘只需要**最后一折的模型**（最后一折的预测
    区间 `[b, c)` 已经覆盖数据末尾，后面不会再有折了）。这里把每一折训出来的
    模型留下最后一个，其余丢掉 —— 48 个模型塞进产物没有意义，只会撑爆文件。

    产物里的 `state` 从**最后一折跑出来的信号**反推：迟滞状态机的当前持仓
    （`cur` / `src`）和触发计划的尾巴。实盘接上之后从这些状态继续算，不会
    「重启一次就把上次的仓平掉」。
    """
    t0 = time.time()
    cfg = _normalized(cfg)
    keys = [k for k in (keys or []) if F.is_factor(k)]
    if len(keys) < 2:
        raise ValueError("机器学习多因子至少需要 2 个因子")
    if df is None or len(df) < 200:
        raise ValueError(f"数据太少（{0 if df is None else len(df)} 根），训不了模型")
    _no_cross_periods(cfg, keys)

    # ---- 因子矩阵 + 标签（和 build_score 里一样）----
    use = F.filter_available(df, keys)
    if not use:
        raise ValueError("选中的因子在这个标的上都没有足够数据")
    fmat = F.factor_frame(df, use, window=cfg.window, method=cfg.method)
    fwd = M.forward_return(df, cfg.horizon)

    # ---- 触发计划（要跟训练用同一份，否则样本集不一致）----
    mask = None
    plan = None
    if str(cfg.sample_mode or "all") == "trigger":
        st, armed, _tn, rule_sig = fstrat.trigger_signal(df, cfg)
        m, _mn = fstrat.trigger_mask(st, armed, cfg)
        mask = np.asarray(m, dtype=bool)
        plan = fstrat.TriggerPlan(
            mask=mask,
            strength=st.to_numpy(dtype=float),
            armed=armed.to_numpy(dtype=bool),
            rule_sig=rule_sig.to_numpy(dtype=float),
            factor_channel=(bool(cfg.trigger_keys)
                            and bool(np.isfinite(st.to_numpy(dtype=float)).any())),
            rule_channel=bool(cfg.trigger_rule) and cfg.trigger_rule in fstrat.STRATEGIES,
        )

    # ---- 跑一遍完整流程：既拿预测序列，也拿状态机结果 ----
    out = fstrat.run_factor_strategy(df, "ml", use, cfg, progress=progress, cancel=cancel)
    pack = out.pack
    pred = pack.pred if pack.pred is not None else pd.Series(np.nan, index=df.index)
    score = pack.score

    # ---- 只留最后一折的模型 ----
    # `walk_forward` 不返回模型，这里重训一次「最后一折」那个模型。多花 ~0.65s，
    # 换来产物里有一个能直接 predict 的模型。折边界必须跟 walk_forward 一致，
    # 否则训出来的就不是「最后一折」那个模型了。
    model = _fit_last_fold(fmat, fwd, cfg, mask, cancel)

    if model is None:
        raise ValueError(
            "训练失败：最后一折的样本不足或模型跑不起来"
            "（把 train_window / min_train 调小，或者换一组因子）"
        )
    if not _picklable(model):
        raise ValueError(
            f"{model.__class__.__name__} 不能存成 .pkl（内部用闭包做反传，"
            "闭包不能 pickle）。换 LightGBM / XGBoost / GPR / 线性回归"
        )

    # ---- 状态：迟滞状态机 + 触发计划 + z 分数窗口 ----
    state = _extract_state(out, pack, cfg, plan)

    art = MlArtifact(
        sid=sid or "",
        name=name or "",
        symbol=symbol or "",
        exchange=exchange or "",
        period=period or "",
        model=model,
        model_key=cfg.model,
        req_keys=list(keys),
        train_keys=list(use),
        cfg=_cfg_digest(cfg),
        cfg_hash=_hash_obj(_cfg_digest(cfg)),
        state=state,
        trained_at=time.time(),
        train_until=(str(df.index[-1]) if len(df) else ""),
        data_head=(str(df.index[0]) if len(df) else ""),
        data_tail=(str(df.index[-1]) if len(df) else ""),
        n_bars=int(len(df)),
        folds=int(pack.folds or 0),
        train_seconds=time.time() - t0,
        importance=([float(v) for v in pack.importance.to_numpy()]
                    if pack.importance is not None else []),
        notes=list(pack.notes or []),
    )
    if save:
        art.save()
    return art


def _fit_last_fold(fmat: pd.DataFrame, fwd: pd.Series, cfg, sample_mask, cancel=None):
    """重训「最后一折」那个模型。

    折边界必须和 `walk_forward` 一模一样：它从 `min_bars` 起步、每次
    `+= step_bars`，每折是 `(max(0, t - train_bars), t, min(n, t + step_bars))`。
    这里照抄一遍，只训最后一折。
    """
    import ttrade.mlmodels as ML

    keys = list(fmat.columns)
    X = fmat.to_numpy(dtype=float)
    Y = fwd.to_numpy(dtype=float)
    n = len(fmat)
    bpd = fstrat._bars_per_day(fmat.index)
    train_bars = fstrat._to_bars(cfg.train_window, bpd)
    step_bars = fstrat._to_bars(cfg.step, bpd)
    min_bars = min(int(cfg.min_train), train_bars)
    if n <= min_bars + cfg.horizon + 10:
        return None

    a = b = c = None
    t = min_bars
    while t < n:
        a, b, c = max(0, t - train_bars), t, min(n, t + step_bars)
        t += step_bars
    if a is None:
        return None
    if cancel and cancel():
        raise fstrat.Cancelled()

    tr_end = b - cfg.horizon
    if tr_end - a < 60:
        return None
    Xtr, Ytr = X[a:tr_end], Y[a:tr_end]
    fin = np.isfinite(Xtr)
    need_cols = max(1, int(np.ceil(len(keys) * 0.5)))
    ok = np.isfinite(Ytr) & (fin.sum(axis=1) >= need_cols)
    Xtr = np.where(fin, Xtr, 0.0)
    if sample_mask is not None:
        m = np.asarray(sample_mask[a:tr_end], dtype=bool)
        if int((ok & m).sum()) >= 40:
            ok = ok & m
    if ok.sum() < 40:
        return None

    model = ML.build(cfg.model)
    try:
        model.fit(Xtr[ok], Ytr[ok])
    except Exception:  # noqa: BLE001
        return None
    return model


def _extract_state(out, pack, cfg, plan) -> LiveState:
    """从跑完的信号里反推实盘续算需要的状态。

    - `cur` / `src`：迟滞状态机当前持仓。重算一遍状态机，逐根扫，把最后
      非零的那根的方向和通道记下来。
    - 触发计划的尾巴：实盘接着算的时候，触发点的判定要接得上（冷却期
      `trigger_cool` 跨 bar 计数）。
    """
    st = LiveState()
    score = pack.score.to_numpy(dtype=float) if pack.score is not None else np.empty(0)
    sig = out.df["signal"].to_numpy(dtype=float) if "signal" in out.df else np.empty(0)

    # 迟滞状态机：末态就是最后一根的非零持仓；通道来源要单独追
    if len(sig):
        cur = int(sig[-1]) if np.isfinite(sig[-1]) else 0
        st.cur = cur
        if cur != 0 and plan is not None:
            # 通道：看最后一根的触发是谁给的
            if plan.mask is not None and len(plan.mask) and bool(plan.mask[-1]):
                s_v = np.asarray(plan.strength, dtype=float)
                r_v = np.asarray(plan.rule_sig, dtype=float)
                a_v = np.asarray(plan.armed, dtype=bool)
                f_dir = 0
                if plan.factor_channel and len(s_v) and np.isfinite(s_v[-1]) and s_v[-1] != 0:
                    f_dir = 1 if s_v[-1] > 0 else -1
                r_dir = 0
                if plan.rule_channel and len(a_v) and a_v[-1] and len(r_v) \
                        and np.isfinite(r_v[-1]) and r_v[-1] != 0:
                    r_dir = 1 if r_v[-1] > 0 else -1
                if f_dir and r_dir and f_dir != r_dir:
                    st.src = ""
                elif f_dir:
                    st.src = "factor"
                elif r_dir:
                    st.src = "rule"
            if not st.src:
                st.src = "factor"

    # z 分数窗口：`_rolling_z` 的窗口（252 根）里要有足够观测，实盘刚接上
    # 时分数才不会全 NaN
    w = max(20, int(cfg.window))
    if len(score):
        st.score_tail = score[-w:].astype(float)
        st.last_ts = str(pack.score.index[-1]) if len(pack.score.index) else ""
        # 持有值：触发模式下 `score.where(mask).ffill()` 的末值。实盘接上之后
        # 第一根要沿用它 —— 用当前这根的原始 z 就和回测对不上了
        finite = np.flatnonzero(np.isfinite(score))
        if len(finite):
            st.held_score = float(score[finite[-1]])

    # 预测尾巴：同样是为了 `_rolling_z` 能续上
    pred = pack.pred
    if pred is not None and len(pred):
        st.pred_tail = pred.to_numpy(dtype=float)[-w:]

    # 触发计划尾巴：冷却计数跨 bar，得留着
    if plan is not None:
        cw = max(int(cfg.trigger_cool) * 3, 64)
        st.plan_mask = np.asarray(plan.mask, dtype=bool)[-cw:]
        st.plan_strength = np.asarray(plan.strength, dtype=float)[-cw:]
        st.plan_armed = np.asarray(plan.armed, dtype=bool)[-cw:]
        st.plan_rule_sig = np.asarray(plan.rule_sig, dtype=float)[-cw:]
    return st


# ================================================================== 实盘预测
class LivePredictor:
    """实盘侧的「只预测」执行器。

    加载一份产物，然后每根新 bar 只做三件事：

    1. 重算因子矩阵（`factor_frame`，3 秒量级，确定性，和训练时同一段历史）
    2. `model.predict` 出一根预测值（毫秒级）
    3. 把预测值接进 `_rolling_z` 的窗口，接进迟滞状态机 → 信号

    **不训练**。这是这个类和 `MlSignals` 最大的区别。

    状态机是「有记忆」的（`cur` / `src`），所以每算一根要把状态更新回
    `self.state`；进程重启时从产物里恢复，不会「重启一次就把上次的仓平掉」。
    """

    def __init__(self, art: MlArtifact, cfg: "fstrat.FactorConfig",
                 keys: list[str], log=None, fold_set: "FoldSet | None" = None) -> None:
        self.art = art
        self.cfg = _normalized(cfg)
        self.keys = list(art.train_keys or keys or [])
        self.log = log
        self.state = LiveState.from_dict(art.state.to_dict())
        self.model = art.model
        # 折清单（有就用它，实盘可以按 bar 落到哪一折来调哪个模型）
        self.fold_set = fold_set
        self._zbuf = list(np.asarray(self.state.pred_tail, dtype=float))
        self._plan_mask = list(np.asarray(self.state.plan_mask, dtype=bool))
        self._plan_strength = list(np.asarray(self.state.plan_strength, dtype=float))
        self._plan_armed = list(np.asarray(self.state.plan_armed, dtype=bool))
        self._plan_rule = list(np.asarray(self.state.plan_rule_sig, dtype=float))
        self._last_idx = None
        # 上一次的分数。触发模式下「观点只在触发点刷新」，触发点之间沿用
        # 上一次的值 —— 这个值必须挂在实例上，不能是类属性（否则多个
        # 品种的预测器会互相污染）
        # 观点的「持有值」：触发模式下只在决策点刷新，决策点之间沿用上一次。
        # 这个值必须从产物里恢复 —— 恢复错了，实盘第一根就会用「今天这一根的
        # 原始 z」而不是「上次决策点的 z」，和回测对不上
        self._held_score = float(getattr(art.state, "held_score", float("nan")))
        # 上一次已经推进过状态机的 bar。**同一根 bar 不能推进两次** ——
        # 迟滞状态机是有记忆的，重复推进会把同一个信号算两遍（比如分数
        # 从 1.2 掉到 0.5，第一遍平仓，第二遍又判一次）
        self._state_idx = _idx_key(art.state.last_ts)
        # 上一次入过 z 缓冲的 bar（防止同一根重复入缓冲）
        self._pred_idx = _idx_key(art.state.last_ts)
        # 上一次的原始预测值（模型输出，没标准化）
        self._last_pred = float("nan")
        # 缓冲里的预测值是不是**全一样**（模型退化成常数预测器）。
        # 这和「历史预测不够」是两回事，报错文案要分开说。
        self._z_degenerate = False
        # ATR 出场规则的状态机。和回测**同一份实现**（`fstrat.ExitTracker`）——
        # 各写一遍迟早会分叉，而「回测 2.3 年平仓、实盘一直拿着」这种差异
        # 没有任何线索可查。`score` 口径、没开止损、也没勾「不按分数出场」时
        # 不建，省得每根都去算 ATR（`active()` 就是这三个条件的并集）
        _probe = fstrat.ExitTracker(self.cfg)
        self._exit = _probe if _probe.active() else None
        self._exit_synced = False
        # 上一根的 |分数|：出场后要判断「这一根是不是刚越过进场阈值」，
        # 也就是「算不算一笔新交易」。和回测 `apply_exit_rules` 里的 fresh 同一口径
        self._prev_abs_z = float("nan")

    # ---- 状态机 ----

    def _hysteresis_step(self, score: float) -> int:
        """跑一步 `hysteresis`（单因子 / 多因子口径）。"""
        e, x = float(self.cfg.entry_z), float(self.cfg.exit_z)
        cur = int(self.state.cur)
        v = float(score)
        if not np.isfinite(v):
            cur = 0
        else:
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
        self.state.cur = cur
        return cur

    def _trigger_hysteresis_step(self, score: float, strength: float,
                                 is_point: bool, rule_sig: float,
                                 armed: bool = False) -> int:
        """跑一步 `trigger_hysteresis`（触发模式口径）。

        和 `factor_strategy.trigger_hysteresis` 同一套规则，只是**逐根**推进
        而不是整段扫 —— 实盘每次只多一根。

        `is_point` 是决策点（`plan.mask`），**进场只在这里判**；
        `armed` 是「规则刚开仓 / 刚翻向」，只用来确认规则那一侧给的方向。
        这两个不是一回事：冷却期内的 bar `mask` 为假但 `armed` 可能为真。
        """
        e, x = float(self.cfg.entry_z), float(self.cfg.exit_z)
        cur = int(self.state.cur)
        src = str(self.state.src)
        s_v = float(strength)
        r_v = float(rule_sig)

        # ---- 出场：每根都判 ----
        if cur == 1:
            if src == "factor":
                if np.isfinite(s_v) and s_v < x:
                    cur = 0
            elif not (np.isfinite(r_v) and r_v > 0):
                cur = 0
        elif cur == -1:
            if src == "factor":
                if np.isfinite(s_v) and s_v > -x:
                    cur = 0
            elif not (np.isfinite(r_v) and r_v < 0):
                cur = 0

        # ---- 进场：只在决策点 ----
        if cur == 0 and is_point:
            f_dir = 0
            if self.cfg.trigger_keys and np.isfinite(s_v) and s_v != 0:
                f_dir = 1 if s_v > 0 else -1
            r_dir = 0
            if self.cfg.trigger_rule and self.cfg.trigger_rule in fstrat.STRATEGIES \
                    and armed and np.isfinite(r_v) and r_v != 0:
                r_dir = 1 if r_v > 0 else -1
            d, ch = 0, ""
            if f_dir and r_dir and f_dir != r_dir:
                d, ch = 0, ""
            elif f_dir:
                d, ch = f_dir, "factor"
            elif r_dir:
                d, ch = r_dir, "rule"
            v = float(score)
            if d and np.isfinite(v) and (v > e if d == 1 else v < -e):
                cur, src = d, ch

        self.state.cur = cur
        self.state.src = src
        return cur

    def _exit_step(self, df: pd.DataFrame, sig: int, z: float = float("nan")) -> int:
        """把 ATR 出场规则叠到信号上。和回测的 `apply_exit_rules` 同一份状态机。

        口径细节（和回测逐条对齐）：

        - ATR 用 `exit_atr_n` 根的简单均值，和回测同一个函数
        - 箱体边界只取**过去** N 根（不含当根），否则当根的低点天然就是箱体
          下沿，永远触发不了
        - 进程重启后第一次调用时，把仓位对齐到产物里存的 `state.cur`，浮盈从
          当根重新起算 —— 重启那一刻的浮盈已经无从查证，硬编一个反而更假
        """
        if self._exit is None:
            return sig
        try:
            close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
            high = (pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
                    if "high" in df.columns else close)
            low = (pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
                   if "low" in df.columns else close)
            atr = fstrat._atr_array(df, self.cfg.exit_atr_n)
            i = len(df) - 1
            if not self._exit_synced:
                self._exit_synced = True
                cur = int(self.state.cur)
                if cur != 0:
                    self._exit.pos = cur
                    self._exit.entry = (float(close[i]) if np.isfinite(close[i]) else 0.0)
                    self._exit.best = 0.0
            bars = self._exit.bars
            lo = hi = float("nan")
            if i >= bars:
                lo = float(np.nanmin(low[i - bars:i]))
                hi = float(np.nanmax(high[i - bars:i]))
            # 「刚越过进场阈值」= 这笔算出的是新信号，不是上一笔的延续。
            # 出场之后靠它决定能不能再进（否则趋势没走完，仓位就永远空着）
            az = abs(float(z)) if np.isfinite(z) else float("nan")
            e = float(self.cfg.entry_z)
            fresh = bool(np.isfinite(az) and az > e
                         and not (np.isfinite(self._prev_abs_z) and self._prev_abs_z > e))
            self._prev_abs_z = az
            return self._exit.step(int(sig), close[i], atr[i], lo, hi, fresh)
        except Exception as exc:  # noqa: BLE001 - 出场规则炸了不该让整根信号丢
            if self.log:
                self.log(f"出场规则计算失败，这一根按分数信号走：{exc}")
            return sig

    # ---- 主流程 ----

    def predict(self, df: pd.DataFrame) -> dict:
        """对一段历史算最后一根的信号。

        返回 `{signal, score, ok, error}`。`ok=False` 时 `signal` 是 0
        且**不许动仓** —— 「算不出来」和「信号是 0」是两回事。

        `df` 必须是**和训练时同源**的历史（同一段前复权行情、同样长的尾巴）。

        **同一根 bar 重复调用不会重复推进状态机**：迟滞状态机是有记忆的，
        重复推进会把同一个信号算两遍。引擎那边每根新 bar 调一次，这里
        内部按时间戳去重。
        """
        if df is None or len(df) < 30:
            return {"ok": False, "signal": 0, "score": float("nan"),
                    "error": "历史不够"}
        try:
            ts = _idx_key(df.index[-1])
            z, strength, is_point, rule_sig, armed, trig, pred = self._score_of(df)
            if not np.isfinite(z):
                if self._z_degenerate:
                    # 模型退化成常数预测器：预测值全一样 → 滚动 z 的标准差是 0。
                    # 加历史没用，要**重训**（见 `mlmodels.LGBMModel` 的说明）。
                    return {"ok": False, "signal": 0, "score": float("nan"),
                            "error": "模型预测值恒定，算不出 z 分数（要重新训练模型）"}
                return {"ok": False, "signal": 0, "score": float("nan"),
                        "error": "z 分数算不出来（历史预测不够）"}

            # 同一根 bar 已经推进过了：直接返回当前状态，不重复推进
            if ts and ts == self._state_idx:
                return {"ok": True, "signal": int(self.state.cur),
                        "score": float(self._held_score), "error": "", "ts": ts}

            if trig:
                # 观点只在**决策点**刷新，决策点之间沿用上一次的值
                # （回测里是 `score.where(plan.mask).ffill()`）
                if is_point:
                    self._held_score = z
                elif np.isfinite(self._held_score):
                    z = self._held_score
                sig = self._trigger_hysteresis_step(z, strength, is_point, rule_sig,
                                                    armed=armed)
                self._plan_mask.append(is_point)
                self._plan_strength.append(strength)
                self._plan_armed.append(is_point)
                self._plan_rule.append(rule_sig)
                for lst in (self._plan_mask, self._plan_strength,
                            self._plan_armed, self._plan_rule):
                    if len(lst) > 256:
                        del lst[:-256]
            else:
                sig = self._hysteresis_step(z)

            sig = self._exit_step(df, int(sig), z)

            self._held_score = z
            self.state.held_score = z
            self.state.last_ts = ts
            self._state_idx = ts
            self._last_idx = df.index[-1]
            self._last_pred = pred
            return {"ok": True, "signal": int(sig), "score": float(z),
                    "error": "", "ts": ts, "pred": float(pred)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "signal": 0, "score": float("nan"),
                    "error": f"{type(exc).__name__}: {exc}"}

    def _model_for(self, ts: str):
        """bar 时间 `ts` 该用哪个模型。

        **滚动调用折产物**：bar 落在哪一折的预测区间里，就用哪一折的模型 ——
        这和回测里 `walk_forward` 的行为一致（每一段区间由对应那折的模型
        预测），所以实盘算出来的预测序列和回测**逐根相同**。

        超出所有折的区间（新的 bar 来了）→ 用**最后一折**的模型：它是当前
        「最新的模型」，正是实盘该用的那个。

        没有折清单（只有主产物）→ 用主产物里的模型。
        """
        fs = self.fold_set
        if fs is None or not fs.folds:
            return self.model
        fa = fs.fold_for(ts)
        if fa is None:
            fa = fs.last_fold()
        if fa is None:
            return self.model
        if getattr(fa, "model", None) is None:
            if not fs.ensure_loaded(fa):
                return self.model
        return fa.model or self.model

    def _score_of(self, df: pd.DataFrame):
        """算出最后一根的 z 分数（不含状态机）。

        返回 `(z, strength, is_point, rule_sig, armed, is_trigger)`。

        **触发点 = `trigger_mask` 的结果，不是 `armed`。** 这两个不是一回事：
        - `armed` 是「规则刚开仓 / 刚翻向」的那几根
        - `mask` 是**决策点**：在 `armed` 基础上还要过触发强度阈值、方向过滤、
          分档、冷却

        回测里 `score.where(plan.mask).ffill()` 用的是 `mask`。用 `armed` 代替
        的话，冷却期内的 bar 也会刷新分数，和回测对不上（实测差 0.4 个 z）。
        """
        fmat = F.factor_frame(df, self.keys, window=self.cfg.window,
                              method=self.cfg.method)
        if fmat.empty:
            raise ValueError("因子矩阵算不出来")
        X = fmat.to_numpy(dtype=float)
        X = np.where(np.isfinite(X), X, 0.0)

        ts = _idx_key(df.index[-1])
        model = self._model_for(ts)
        lb = int(getattr(model, "lookback", 0) or 0)
        ctx = X[-max(lb, 1):] if lb else X[-1:]
        try:
            p = np.asarray(model.predict(ctx), dtype=float)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"预测失败：{type(exc).__name__}: {exc}") from exc
        if not len(p) or not np.isfinite(p[-1]):
            raise ValueError("预测值是空的")
        pred = float(p[-1])
        # ---- 预测值的滚动 z：把历史预测接在前面 ----
        # 同一根 bar 不重复入缓冲：z 窗口是**滚动**的，多塞一根同样的值
        # 会让均值 / 标准差偏掉，同一根 bar 算两次给出两个分数。
        # （预测值本身是确定的，问题只在缓冲区。）
        if ts != self._pred_idx:
            self._zbuf.append(pred)
            self._pred_idx = ts
            w = max(20, int(self.cfg.window))
            if len(self._zbuf) > w:
                self._zbuf = self._zbuf[-w:]
        z = self._rolling_z_last()

        trig = str(self.cfg.sample_mode or "all") == "trigger"
        if not trig:
            return z, float("nan"), False, 0.0, False, False, pred

        st_s, armed_s, _tn, rule_s = fstrat.trigger_signal(df, self.cfg)
        strength = float(st_s.to_numpy(dtype=float)[-1]) if len(st_s) else float("nan")
        rule_sig = float(rule_s.to_numpy(dtype=float)[-1]) if len(rule_s) else 0.0
        armed = bool(armed_s.to_numpy(dtype=bool)[-1]) if len(armed_s) else False
        # 决策点：和回测用同一个 `trigger_mask`。这里要**整段重算** mask ——
        # 冷却是跨 bar 的，只看最后一根不知道它是不是被冷却掉了
        try:
            m, _mn = fstrat.trigger_mask(st_s, armed_s, self.cfg)
            is_point = bool(np.asarray(m, dtype=bool)[-1]) if len(m) else False
        except Exception:  # noqa: BLE001
            is_point = armed
        return z, strength, is_point, rule_sig, armed, True, pred

    def _rolling_z_last(self) -> float:
        """对预测值缓冲做一次滚动 z（口径与 `factors.standardize` 一致）。

        训练时是 `_rolling_z(pred, cfg.window)`，对整段预测序列做 PIT 标准化。
        实盘这里只有一段尾巴，用同样的公式对尾巴做。窗口不够就返回 NaN
        （宁可不出信号，也不用一个来路不明的分数）。

        缓冲区是**滚动**的：`window` 根之前的值会被丢掉。所以实盘跑得越久，
        z 的口径越接近「只看最近 window 根」—— 和回测在最后一根上的口径一致
        （回测的滚动窗口也只看最近 window 根）。
        """
        s = pd.Series(np.asarray(self._zbuf, dtype=float))
        mp = max(20, int(self.cfg.window) // 4)
        if len(s) < mp:
            return float("nan")
        roll = s.rolling(int(self.cfg.window), min_periods=mp)
        m = roll.mean()
        sd = roll.std(ddof=1)
        if not np.isfinite(m.iloc[-1]) or not np.isfinite(sd.iloc[-1]):
            return float("nan")
        if sd.iloc[-1] == 0:
            # 标准差是 0 = 缓冲里的预测值**全一样**。这不是「历史不够」，
            # 是模型退化成了常数预测器（稀疏样本 + L1 正则把树掐死过，
            # 见 `mlmodels.LGBMModel`）。原因要说清楚，不然用户只会看到
            # 「历史预测不够」，去加历史根本没用 —— 要重训。
            self._z_degenerate = True
            return float("nan")
        self._z_degenerate = False
        z = (s.iloc[-1] - m.iloc[-1]) / sd.iloc[-1]
        return float(np.clip(z, -4.0, 4.0))


def load_for_live(
    sid: str,
    cfg: "fstrat.FactorConfig",
    keys: list[str],
    log=None,
) -> tuple[LivePredictor | None, str]:
    """给实盘用：加载产物并校验。

    返回 `(predictor, 原因)`。`predictor is None` 时原因说明为什么不能用 ——
    调用方（引擎）应该退回「后台重训」的慢路径，而不是静默当 0 信号。

    有折清单就带上折清单（实盘滚动调用各折模型）；没有也能用（退到主产物里
    的最后一个模型）。

    **合并训练**（`cfg.pooled=True`）时，产物按**父 sid** 存（一份模型管 N 个
    品种）。传进来的 `sid` 是 `父sid#品种_交易所`，这里拆出父 sid 加载产物，
    再从 `art.states` 里取这个品种的续算状态。
    """
    # 合训：拆出父 sid，用父 sid 的产物
    pool_sym = ""
    load_sid = sid
    if getattr(cfg, "pooled", False) and "#" in sid:
        load_sid, pool_sym = sid.split("#", 1)

    p = artifact_path(load_sid)
    if not os.path.exists(p):
        return None, f"没有训练产物（{os.path.basename(p)}）"
    try:
        art = MlArtifact.load(p)
    except Exception as exc:  # noqa: BLE001
        return None, f"训练产物读不出来：{type(exc).__name__}: {exc}"
    why = art.mismatch_reason(cfg, keys)
    if why:
        return None, why
    if art.model is None:
        return None, "训练产物里没有模型"

    # 合训：把产物里的 state 换成这个品种的
    if art.pooled and pool_sym:
        sym_state = art.states.get(pool_sym)
        if sym_state is None:
            return None, (f"合并训练产物里没有 {pool_sym} 的状态"
                          f"（产物里有：{', '.join(list(art.states)[:5])}…）")
        # 用品种自己的状态覆盖，LivePredictor 会从 art.state 读
        art = _replace(art, state=sym_state)

    fold_set = None
    try:
        fs = FoldSet.load(load_sid)
        if fs.folds:
            fold_set = fs
    except Exception:  # noqa: BLE001
        fold_set = None
    try:
        pred = LivePredictor(art, cfg, keys, log=log, fold_set=fold_set)
    except Exception as exc:  # noqa: BLE001
        return None, f"训练产物装载失败：{type(exc).__name__}: {exc}"
    return pred, ""


# ================================================================== 折文件集
@dataclass
class FoldSet:
    """一套「一折一个文件」的产物。

    目录布局（都在用户可写目录 `ml_models/` 下）：

        <sid>_f000.pkl      第 0 折的模型
        <sid>_f001.pkl      第 1 折的模型
        ...
        <sid>_f047.pkl      第 47 折的模型
        <sid>.pkl           主产物（配置、状态、折的清单）

    主产物里的 `folds` 是清单：每一折记了模型文件和它负责预测的时间区间。
    实盘按 bar 的时间戳查清单，命中哪一折就加载哪一折的文件。
    """

    sid: str = ""
    folds: list = field(default_factory=list)      # [FoldArtifact]
    master: MlArtifact | None = None

    def save(self, master: MlArtifact) -> str:
        """把每一折写成独立文件，主产物保存清单。"""
        self.sid = master.sid
        self.master = master
        for fa in self.folds:
            p = fold_path(master.sid, fa.index)
            tmp = p + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(fa, fh, protocol=pickle.HIGHEST_PROTOCOL)
            _replace_retry(tmp, p)
        master.folds_meta = [
            {"index": fa.index, "file": os.path.basename(fold_path(master.sid, fa.index)),
             "t_from": fa.t_from, "t_to": fa.t_to, "n_train": fa.n_train,
             "train_from": fa.train_from, "train_to": fa.train_to,
             "fallback": fa.fallback}
            for fa in self.folds
        ]
        return master.save()

    @classmethod
    def load(cls, sid: str) -> "FoldSet":
        """加载主产物 + 清单（**不加载模型本体**）。"""
        p = artifact_path(sid)
        art = MlArtifact.load(p)
        fs = cls(sid=sid, master=art)
        for meta in (art.folds_meta or []):
            fp = os.path.join(artifact_dir(), meta.get("file", ""))
            fa = FoldArtifact(
                index=int(meta.get("index", 0) or 0),
                model=None,
                model_key=art.model_key,
                t_from=str(meta.get("t_from", "") or ""),
                t_to=str(meta.get("t_to", "") or ""),
                n_train=int(meta.get("n_train", 0) or 0),
                train_from=str(meta.get("train_from", "") or ""),
                train_to=str(meta.get("train_to", "") or ""),
                fallback=bool(meta.get("fallback", False)),
            )
            fa._file = fp            # 延迟加载用
            fs.folds.append(fa)
        return fs

    def fold_for(self, ts: str) -> FoldArtifact | None:
        """bar 时间 `ts` 落在哪一折的预测区间里。"""
        for fa in self.folds:
            if fa.covers(ts):
                return fa
        return None

    def last_fold(self) -> FoldArtifact | None:
        """最后一折。**新 bar 超出所有折的区间时用它** —— 它是「当前最新的
        模型」，正是实盘该用的那个。"""
        return self.folds[-1] if self.folds else None

    def ensure_loaded(self, fa: FoldArtifact) -> bool:
        """确保某一折的模型已经加载（延迟加载）。"""
        if fa is None:
            return False
        if getattr(fa, "model", None) is not None:
            return True
        fp = getattr(fa, "_file", "") or fold_path(self.sid, fa.index)
        if not os.path.exists(fp):
            return False
        try:
            with open(fp, "rb") as fh:
                obj = pickle.load(fh)
        except Exception:  # noqa: BLE001
            return False
        if not isinstance(obj, FoldArtifact):
            return False
        fa.model = obj.model
        return True

    def remove(self) -> None:
        """删掉这一套（主产物 + 所有折文件）。"""
        for fa in self.folds:
            fp = getattr(fa, "_file", "") or fold_path(self.sid, fa.index)
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except OSError:
                pass
        try:
            p = artifact_path(self.sid)
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def train_fold_set(
    df: pd.DataFrame,
    sid: str,
    cfg: "fstrat.FactorConfig",
    keys: list[str],
    name: str = "",
    symbol: str = "",
    exchange: str = "",
    period: str = "",
    progress=None,
    cancel=None,
    save: bool = True,
) -> tuple[MlArtifact, FoldSet]:
    """离线训练一次，**每一折存成一个独立的 .pkl**。

    这是「实盘滚动调用这些 .pkl」的实现：训练时把每一折的模型都留下来，
    实盘按 bar 落在哪一折就加载哪一折。

    返回 `(主产物, 折清单)`。主产物里同时装：
    - 最后一折的模型（直接可用）
    - 每一折的清单（索引 + 时间区间 + 文件名）
    - 续算状态

    实盘侧优先走 `FoldSet`（逐折加载），只有折清单缺失时才退回主产物里的
    最后一折模型。
    """
    t0 = time.time()
    cfg = _normalized(cfg)
    keys = [k for k in (keys or []) if F.is_factor(k)]
    if len(keys) < 2:
        raise ValueError("机器学习多因子至少需要 2 个因子")
    if df is None or len(df) < 200:
        raise ValueError(f"数据太少（{0 if df is None else len(df)} 根），训不了模型")
    _no_cross_periods(cfg, keys)

    use = F.filter_available(df, keys)
    if not use:
        raise ValueError("选中的因子在这个标的上都没有足够数据")

    # ---- 触发计划（要和训练用同一份，否则样本集不一致）----
    mask = None
    plan = None
    if str(cfg.sample_mode or "all") == "trigger":
        st, armed, _tn, rule_sig = fstrat.trigger_signal(df, cfg)
        m, _mn = fstrat.trigger_mask(st, armed, cfg)
        mask = np.asarray(m, dtype=bool)
        plan = fstrat.TriggerPlan(
            mask=mask,
            strength=st.to_numpy(dtype=float),
            armed=armed.to_numpy(dtype=bool),
            rule_sig=rule_sig.to_numpy(dtype=float),
            factor_channel=(bool(cfg.trigger_keys)
                            and bool(np.isfinite(st.to_numpy(dtype=float)).any())),
            rule_channel=bool(cfg.trigger_rule) and cfg.trigger_rule in fstrat.STRATEGIES,
        )

    # ---- 跑一遍完整流程，把每一折的模型都收上来 ----
    collected: list = []
    out = fstrat.run_factor_strategy(df, "ml", use, cfg, progress=progress,
                                      cancel=cancel, collect_models=collected)
    pack = out.pack
    if not collected:
        raise ValueError(
            "训练失败：一折都没训出来（样本不足或模型跑不起来）。"
            "把 train_window / min_train 调小，或者换一组因子"
        )

    # 检查能不能 pickle —— 不能的话别写文件（写了也读不回来）
    for model, _b, _c in collected:
        if not _picklable(model):
            raise ValueError(
                f"{model.__class__.__name__} 不能存成 .pkl（内部用闭包做反传，"
                "闭包不能 pickle）。换 LightGBM / XGBoost / GPR / 线性回归"
            )

    idx = df.index
    folds: list[FoldArtifact] = []
    for i, (model, b, c) in enumerate(collected):
        fa = FoldArtifact(
            index=i,
            model=model,
            model_key=cfg.model,
            t_from=str(idx[b]) if b < len(idx) else "",
            t_to=str(idx[c]) if c < len(idx) else "",
        )
        folds.append(fa)

    # 末折的模型 → 主产物（万一折清单丢了，至少还有最后一个能用）
    last_model = folds[-1].model if folds else None
    state = _extract_state(out, pack, cfg, plan)

    art = MlArtifact(
        sid=sid or "",
        name=name or "",
        symbol=symbol or "",
        exchange=exchange or "",
        period=period or "",
        model=last_model,
        model_key=cfg.model,
        req_keys=list(keys),
        train_keys=list(use),
        cfg=_cfg_digest(cfg),
        cfg_hash=_hash_obj(_cfg_digest(cfg)),
        state=state,
        trained_at=time.time(),
        train_until=(str(idx[-1]) if len(idx) else ""),
        data_head=(str(idx[0]) if len(idx) else ""),
        data_tail=(str(idx[-1]) if len(idx) else ""),
        n_bars=int(len(df)),
        folds=len(folds),
        train_seconds=time.time() - t0,
        importance=([float(v) for v in pack.importance.to_numpy()]
                    if pack.importance is not None else []),
        notes=list(pack.notes or []),
    )
    fs = FoldSet(sid=art.sid, folds=folds, master=art)
    if save:
        fs.save(art)
    return art, fs


def train_fold_set_pooled(
    frames: dict,
    sid: str,
    cfg: "fstrat.FactorConfig",
    keys: list[str],
    name: str = "",
    period: str = "",
    progress=None,
    cancel=None,
    save: bool = True,
) -> tuple[MlArtifact, FoldSet]:
    """多品种**合并训练**：N 个品种的样本堆成一张长表，训一个模型。

    和 `train_fold_set` 的区别：
    - 输入是 `{品种键: DataFrame}` 而不是单个 df
    - 产物只有**一份**（按父 sid 存），不是每品种一份
    - 每个品种的续算状态存在 `art.states` 里，实盘各建一个 `LivePredictor`

    标签按各品种自己的滚动波动率折算（见 `pooled_score`），否则高波动品种
    会淹没低波动品种的信号。
    """
    t0 = time.time()
    cfg = _normalized(cfg)
    keys = [k for k in (keys or []) if F.is_factor(k)]
    if len(keys) < 2:
        raise ValueError("机器学习多因子至少需要 2 个因子")
    if not frames or len(frames) < 2:
        raise ValueError("合并训练至少要 2 个品种")
    _no_cross_periods(cfg, keys)

    collected: list = []
    outs = fstrat.run_pooled_strategy(
        frames, keys, cfg, progress=progress, cancel=cancel,
        collect_models=collected,
    )
    if not collected:
        raise ValueError(
            "合并训练失败：一折都没训出来（样本不足或模型跑不起来）。"
            "把 train_window / min_train 调小，或者换一组因子"
        )
    for model, _b, _c in collected:
        if not _picklable(model):
            raise ValueError(
                f"{model.__class__.__name__} 不能存成 .pkl（内部用闭包做反传，"
                "闭包不能 pickle）。换 LightGBM / XGBoost / GPR / 线性回归"
            )

    # 折的时间区间：用第一个品种的索引（所有品种在同一根 bar 时间上是同步的）
    first_sym = next(iter(frames))
    first_df = frames[first_sym]
    idx = first_df.index
    folds: list[FoldArtifact] = []
    for i, (model, b, c) in enumerate(collected):
        fa = FoldArtifact(
            index=i,
            model=model,
            model_key=cfg.model,
            t_from=str(idx[b]) if b < len(idx) else "",
            t_to=str(idx[c]) if c < len(idx) else "",
        )
        folds.append(fa)

    last_model = folds[-1].model if folds else None

    # 每个品种的状态
    states: dict[str, LiveState] = {}
    sym_meta: dict[str, dict] = {}
    for sym, out in outs.items():
        pack = out.pack
        plan = pack.trigger
        # 触发计划：合训里每个品种有自己的 trigger plan
        state = _extract_state(out, pack, cfg, plan)
        states[sym] = state
        df = frames[sym]
        sym_meta[sym] = {
            "symbol": sym.rsplit("_", 1)[0] if "_" in sym else sym,
            "exchange": sym.rsplit("_", 1)[1] if "_" in sym else "",
            "n_bars": len(df),
        }

    # 摘要 notes
    pack0 = outs[first_sym].pack
    notes = list(pack0.notes or [])
    notes.insert(0, f"合并训练：{len(frames)} 个品种，{len(folds)} 折")

    art = MlArtifact(
        sid=sid or "",
        name=name or "",
        symbol="",
        exchange="",
        period=period or "",
        model=last_model,
        model_key=cfg.model,
        req_keys=list(keys),
        # train_keys 必须是合训的**交集**（pooled_score 里取的 `use`），
        # 不是第一个品种的 `filter_available`。否则实盘用第一个品种的可用
        # 因子（可能比交集多）算因子矩阵，喂给模型的列数和训练时不一致 →
        # "X has 42 features, but model is expecting 35"
        train_keys=list(pack0.fmat.columns) if pack0.fmat is not None
                   else list(fstrat.F.filter_available(first_df, keys)),
        cfg=_cfg_digest(cfg),
        cfg_hash=_hash_obj(_cfg_digest(cfg)),
        state=states.get(first_sym, LiveState()),
        trained_at=time.time(),
        train_until=(str(first_df.index[-1]) if len(first_df.index) else ""),
        data_head=(str(first_df.index[0]) if len(first_df.index) else ""),
        data_tail=(str(first_df.index[-1]) if len(first_df.index) else ""),
        n_bars=int(len(first_df)),
        folds=len(folds),
        train_seconds=time.time() - t0,
        importance=([float(v) for v in pack0.importance.to_numpy()]
                    if pack0.importance is not None else []),
        notes=notes,
        pooled=True,
        states=states,
        sym_meta=sym_meta,
    )
    fs = FoldSet(sid=art.sid, folds=folds, master=art)
    if save:
        fs.save(art)
    return art, fs
