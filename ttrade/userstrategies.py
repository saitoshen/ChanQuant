"""用户自定义规则策略：由 Agent 生成、用户确认后注册进策略库。

和 `userfactors` 是一对：因子是「一列数」，策略是「一串仓位」。两者的坑几乎一样
—— 都是把模型写的 Python 在本机执行、都要落盘到用户目录（打包后程序目录只读）、
都要在注册前用真实行情跑一遍。

## 契约：函数返回带 `signal` 列的 DataFrame

`signal` 是**当日收盘后确定的目标仓位**：1 满仓多、0 空仓、-1 满仓空。真正的成交
由回测引擎在下一根开盘执行，所以策略里**不能出现 `shift(-n)`** —— 用了就是未来
函数，回测漂亮、实盘亏钱。

## 两道闸

1. `exec` 的 builtins 白名单，没有 `__import__`；
2. 注册前用真实沪铜日线跑一遍，必须返回等长 DataFrame、有 `signal` 列、取值只在
   {-1, 0, 1} 里。

比因子多查一条「不能全程空仓」：全 0 的信号在界面上看起来是「策略有效但从不持仓」，
用户会以为是数据问题，其实是代码写错了（比如条件永远不成立）。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ttrade.paths import user_dir

FILE_NAME = "user_strategies.json"
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "range": range, "reversed": reversed, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "True": True, "False": False, "None": None,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError, "KeyError": KeyError,
}


@dataclass
class UserStrategy:
    """一条用户策略。字段和 `StrategySpec` 对齐，多出源码与参数元信息。"""

    key: str = ""
    label: str = ""
    desc: str = ""
    params: dict = field(default_factory=dict)
    # name -> [最小, 最大, 步长]。缺省时按默认值推导，见 `_meta_for()`
    param_meta: dict = field(default_factory=dict)
    func: str = ""
    code: str = ""
    bidirectional: bool = True
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserStrategy":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def path() -> str:
    return os.path.join(user_dir(), FILE_NAME)


def load_all() -> list[UserStrategy]:
    """读盘。坏文件当成空表 —— 一条坏策略不该让整个策略库起不来。"""
    p = path()
    if not os.path.exists(p):
        return []
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    items = data.get("strategies") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[UserStrategy] = []
    seen: set[str] = set()
    for d in items:
        if not isinstance(d, dict):
            continue
        s = UserStrategy.from_dict(d)
        if s.key and s.key not in seen:
            seen.add(s.key)
            out.append(s)
    return out


def save_all(items: list[UserStrategy]) -> None:
    """原子写：先写 .tmp 再 replace。"""
    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "strategies": [i.to_dict() for i in items]},
                  fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


# ---------------------------------------------------------------- 编译与试算

def _namespace() -> dict:
    """策略代码能看见的东西：np / pd、内置指标、白名单 builtins。

    指标函数必须给 —— 让模型从零手写 ATR 和持仓状态机，几乎必然写出未来函数或
    漏掉反手逻辑。平台里已经有验证过的实现，直接暴露。
    """
    from ttrade import strategies as S

    return {
        "__builtins__": dict(_SAFE_BUILTINS),
        "np": np,
        "pd": pd,
        "sma": S.sma,
        "ema": S.ema,
        "atr": S.atr,
        "true_range": S.true_range,
        "_position_state": S._position_state,
    }


def compile_one(s: UserStrategy) -> tuple[object | None, str]:
    """编译并取出函数对象。返回 (func, 错误说明)。"""
    if not s.code.strip():
        return None, "没有代码"
    ns = _namespace()
    try:
        exec(compile(s.code, f"<策略 {s.key}>", "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return None, f"代码跑不起来：{type(exc).__name__}: {exc}"
    fn = ns.get(s.func)
    if not callable(fn):
        return None, f"代码里没有名为 {s.func} 的函数"
    return fn, ""


def sample_df() -> pd.DataFrame:
    """试算样本。和因子用同一份真实沪铜日线 —— 别再抄一遍造数逻辑。"""
    from ttrade import userfactors as UF

    return UF.sample_df()


def trial_run(s: UserStrategy) -> tuple[bool, str, str]:
    """试算一遍。返回 (能不能注册, 错误, 警告)。

    错误 = 代码本身有问题；警告 = 能跑但样本上从不持仓（多半是条件写错了）。
    """
    fn, err = compile_one(s)
    if fn is None:
        return False, err, ""
    df = sample_df()
    try:
        out = fn(df, **{k: v for k, v in (s.params or {}).items()})
    except TypeError:
        # 模型有时把参数写死进函数体、不接 **params。退一步按单参数签名再试一次，
        # 能跑就放行 —— 参数不可调不是错误，只是这个策略只能在代码里改。
        try:
            out = fn(df)
        except Exception as exc:  # noqa: BLE001
            return False, f"用真实行情试算失败：{type(exc).__name__}: {exc}", ""
    except Exception as exc:  # noqa: BLE001
        return False, f"用真实行情试算失败：{type(exc).__name__}: {exc}", ""

    if not isinstance(out, pd.DataFrame):
        return False, f"要返回 pd.DataFrame，实际返回 {type(out).__name__}", ""
    if "signal" not in out.columns:
        return False, "返回的 DataFrame 里没有 signal 列", ""
    if len(out) != len(df):
        return False, f"返回长度 {len(out)} 与行情 {len(df)} 不一致", ""

    sig = pd.to_numeric(out["signal"], errors="coerce").fillna(0)
    bad = sorted({float(v) for v in sig.unique()} - {-1.0, 0.0, 1.0})
    if bad:
        return False, f"signal 只能是 -1 / 0 / 1，出现了 {bad[:5]}", ""

    held = int((sig != 0).sum())
    if held == 0:
        return False, "样本上从头到尾都是空仓（signal 全 0），条件多半写错了", ""
    return True, "", f"样本 {len(df)} 根里持仓 {held} 根（{held / len(df):.0%}）"


# ---------------------------------------------------------------- 注册

def _meta_for(name: str, default) -> list:
    """没给参数元信息时，按默认值推一个能拖的滑杆范围。

    直接套 `(1, 100, 1)` 会把 720 这种周期参数夹成 100 —— 用户看到的是「参数
    永远是 100」，而模型写的默认值根本没生效。
    """
    try:
        v = float(default)
    except (TypeError, ValueError):
        return [1, 100, 1]
    if isinstance(default, float):
        step = max(0.01, round(abs(v) / 10, 4))
        return [0.0, max(1.0, abs(v) * 4), step]
    v = int(round(v))
    step = max(1, abs(v) // 20 or 1)
    return [max(1, v // 4 or 1), max(10, v * 4), step]


def register(s: UserStrategy, allow_overwrite: bool = False) -> tuple[bool, str]:
    """校验 → 试算 → 写盘 → 注册进 `STRATEGIES`。返回 (成功, 给人看的一句话)。"""
    from ttrade import strategies as S

    s.key = (s.key or "").strip()
    s.func = (s.func or f"sig_{s.key}").strip()
    s.label = (s.label or "").strip()
    s.desc = (s.desc or "").strip()

    if not KEY_RE.match(s.key):
        return False, "策略代号只能用小写字母开头的字母/数字/下划线（2~41 位）"
    if not s.label:
        return False, "策略要有个显示名"
    if s.key in S.STRATEGIES and not allow_overwrite:
        return False, f"已经有一个叫 {s.key} 的策略了，换个代号"

    if not isinstance(s.params, dict):
        s.params = {}
    # 参数值统一成 int / float，别把 JSON 里的字符串直接喂给 numpy
    clean: dict = {}
    for k, v in s.params.items():
        try:
            clean[str(k)] = float(v) if isinstance(v, float) else int(v)
        except (TypeError, ValueError):
            return False, f"参数 {k} 的值 {v!r} 不是数字"
    s.params = clean

    if not isinstance(s.param_meta, dict):
        s.param_meta = {}
    meta: dict = {}
    for k, v in s.params.items():
        raw = s.param_meta.get(k)
        try:
            lo, hi, step = (float(x) for x in raw)
            meta[k] = [lo, hi, step]
        except (TypeError, ValueError):
            meta[k] = _meta_for(k, v)
    s.param_meta = meta

    ok, err, warn = trial_run(s)
    if not ok:
        return False, err
    # 试算里已经编过一次，这里再取一次函数对象。先编译再写盘：编译失败就不该
    # 留下一条「库里有、跑不起来」的记录
    fn, err = compile_one(s)
    if fn is None:
        return False, err

    items = [i for i in load_all() if i.key != s.key]
    items.append(s)
    save_all(items)

    S.STRATEGIES[s.key] = S.StrategySpec(
        key=s.key, label=s.label, func=fn, params=dict(s.params),
        param_meta={k: tuple(v) for k, v in meta.items()},
        desc=s.desc, bidirectional=s.bidirectional, code=s.code,
    )
    return True, warn or f"已注册「{s.label}」"


def remove(key: str) -> bool:
    """从库里删掉。内置策略不能删（它不在 JSON 里，删了下次启动又回来）。"""
    from ttrade import strategies as S

    items = load_all()
    left = [i for i in items if i.key != key]
    if len(left) == len(items):
        return False
    save_all(left)
    S.STRATEGIES.pop(key, None)
    return True


def register_all_into(strategies: dict) -> int:
    """启动时把库里的策略挂进 `STRATEGIES`。返回成功条数。

    单条坏掉不影响其他条：用户手改 JSON 改坏了，不该让整个程序起不来。
    """
    from ttrade import strategies as S

    n = 0
    for s in load_all():
        fn, _err = compile_one(s)
        if fn is None:
            continue
        meta = {}
        for k, v in (s.params or {}).items():
            try:
                lo, hi, step = (float(x) for x in (s.param_meta or {}).get(k, ()))
                meta[k] = (lo, hi, step)
            except (TypeError, ValueError):
                lo, hi, step = _meta_for(k, v)
                meta[k] = (lo, hi, step)
        strategies[s.key] = S.StrategySpec(
            key=s.key, label=s.label, func=fn, params=dict(s.params or {}),
            param_meta=meta, desc=s.desc, bidirectional=s.bidirectional, code=s.code,
        )
        n += 1
    return n
