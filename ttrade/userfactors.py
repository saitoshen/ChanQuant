"""用户自定义因子：由 Agent 生成、用户确认后注册进因子库。

## 为什么不直接写进 `ttrade/factors.py`

打包后程序目录是只读的（macOS 的 `.app` 还是签名过的），而因子必须在运行时能
新增。所以定义存成 JSON 放在用户目录，启动时 `exec` 出来注册进 `FACTORS`。

## 执行 AI 生成的代码，两道闸

1. **builtins 白名单**。`exec` 的命名空间里只有几十个基础函数，**没有
   `__import__`** —— 所以 `import os` 直接 NameError。这挡不住蓄意攻击（能写
   `().__class__.__bases__` 那一套），但能挡住模型手滑写出的「顺手读个文件」。
2. **注册前用真实行情试算一遍**。必须是 Series、长度对齐、索引一致。跑不通就
   不写盘 —— 否则因子会以「一直返回 NaN」的形态进库，看起来像数据缺失，排查
   时根本想不到是代码没跑通。

试算只报警告不拦的情况也有：因子读的是 `roll_yield` 这类衍生列，而样本上恰好
没有（美股、或没下过衍生数据的品种）。这种注册了也合理，只是当前标的算不出来，
界面会显示「无数据」。所以「全 NaN」给警告，不给错误。
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

FILE_NAME = "user_factors.json"
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

# `exec` 里能看见的全部内置函数。刻意不含 `__import__` / `open` / `eval` /
# `exec` / `compile`：因子只需要算术和容器操作。
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "range": range, "reversed": reversed, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "True": True, "False": False, "None": None,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError,
}


@dataclass
class UserFactor:
    """一条用户因子。字段和 `FactorSpec` 对齐，多出公式与源码。"""

    key: str = ""
    label: str = ""
    category: str = "自定义"
    desc: str = ""
    formula: str = ""
    hint_dir: int = 1
    func: str = ""
    code: str = ""
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserFactor":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def path() -> str:
    return os.path.join(user_dir(), FILE_NAME)


def load_all() -> list[UserFactor]:
    """读盘。坏文件当成空表 —— 一个坏条目不该让整个因子库起不来。"""
    p = path()
    if not os.path.exists(p):
        return []
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    items = data.get("factors") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[UserFactor] = []
    seen: set[str] = set()
    for d in items:
        if not isinstance(d, dict):
            continue
        f = UserFactor.from_dict(d)
        # 手改 JSON 复制粘贴会造出两条同 key。保留第一条：`register()` 写盘时
        # 是「先删同 key 再追加」，所以正常情况下文件里本来就不该有重复。
        if f.key and f.key not in seen:
            seen.add(f.key)
            out.append(f)
    return out


def save_all(items: list[UserFactor]) -> None:
    """原子写：先写 .tmp 再 replace。中途断电就只剩半个 JSON。"""
    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "factors": [i.to_dict() for i in items]},
                  fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


# ---------------------------------------------------------------- 编译与试算

def _namespace() -> dict:
    """因子代码能看见的东西：np / pd、几个内置工具、白名单 builtins。"""
    from ttrade import factors as F

    return {
        "__builtins__": dict(_SAFE_BUILTINS),
        "np": np,
        "pd": pd,
        # 让模型能直接写 `_col(df, "roll_yield")`，和内置因子的写法保持一致
        "_col": F._col,
        "_ret": F._ret,
        "_safe_div": F._safe_div,
        "_nan_series": F._nan_series,
    }


def compile_one(f: UserFactor) -> tuple[object | None, str]:
    """把 code 编译出来并取出函数对象。返回 (func, 错误说明)。"""
    if not f.code.strip():
        return None, "没有代码"
    ns = _namespace()
    try:
        exec(compile(f.code, f"<因子 {f.key}>", "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return None, f"代码跑不起来：{type(exc).__name__}: {exc}"
    fn = ns.get(f.func)
    if not callable(fn):
        return None, f"代码里没有名为 {f.func} 的函数"
    return fn, ""


_SAMPLE: dict = {}


def sample_df() -> pd.DataFrame:
    """试算用的样本。优先用真实的沪铜日线（带期限结构与 IV 列），没有就造一份。

    用真实数据是因为衍生列（`roll_yield`、`iv_term`）只有真数据里才有；合成
    数据造不出有意义的期限结构，试算会把所有展期因子误判成「全 NaN」。
    """
    if "df" in _SAMPLE:
        return _SAMPLE["df"]
    df = None
    try:
        from ttrade import futures as FU
        from ttrade.paths import data_dir

        # 注意是 `data_dir("cn")`：默认参数是 "us"，拿到的目录里没有期货文件，
        # 会静悄悄退化成合成数据，展期/期权类因子就全被判成「无数据」
        df = FU.load_local("CU", "SHFE", data_dir("cn"))
    except Exception:  # noqa: BLE001
        df = None
    if df is None or len(df) < 300:
        df = _fake_df()
    _SAMPLE["df"] = df
    return df


def _fake_df(n: int = 400) -> pd.DataFrame:
    """没有真实数据时的替身：随机游走 OHLCV，够跑动量/波动类因子。"""
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + abs(rng.normal(0, 0.006, n))),
            "low": close * (1 - abs(rng.normal(0, 0.006, n))),
            "close": close,
            "volume": rng.integers(1000, 50000, n).astype(float),
            "open_oi": rng.integers(10000, 90000, n).astype(float),
            "close_oi": rng.integers(10000, 90000, n).astype(float),
        },
        index=pd.DatetimeIndex(idx, name="datetime"),
    )


def trial_run(f: UserFactor) -> tuple[bool, str, str]:
    """试算一遍。返回 (能不能注册, 错误, 警告)。

    错误 = 代码本身有问题（跑不起来 / 返回值不是 Series / 长度不对）；
    警告 = 能跑但样本上没数据（多半是缺衍生列），注册了在当前标的上会显示无数据。
    """
    fn, err = compile_one(f)
    if fn is None:
        return False, err, ""
    df = sample_df()
    try:
        out = fn(df)
    except Exception as exc:  # noqa: BLE001
        return False, f"用真实行情试算失败：{type(exc).__name__}: {exc}", ""
    if not isinstance(out, pd.Series):
        return False, f"要返回 pd.Series，实际返回 {type(out).__name__}", ""
    if len(out) != len(df):
        return False, f"返回长度 {len(out)} 与行情 {len(df)} 不一致", ""
    if not out.index.equals(df.index):
        return False, "返回值的索引要和行情对齐（别 dropna、别 reset_index）", ""
    valid = int(pd.to_numeric(out, errors="coerce").notna().sum())
    if valid == 0:
        return True, "", "样本上这个因子全是 NaN（多半是缺衍生数据列），注册后在该标的上会显示「无数据」"
    return True, "", f"样本 {len(df)} 根里算出 {valid} 个有效值"


# ---------------------------------------------------------------- 注册

def register(f: UserFactor, allow_overwrite: bool = False) -> tuple[bool, str]:
    """校验 → 试算 → 写盘 → 注册进 `FACTORS`。返回 (成功, 给人看的一句话)。"""
    from ttrade import factors as F

    f.key = (f.key or "").strip()
    f.func = (f.func or f"f_{f.key}").strip()
    f.label = (f.label or "").strip()
    f.category = (f.category or "自定义").strip()
    f.desc = (f.desc or "").strip()

    if not KEY_RE.match(f.key):
        return False, "因子代号只能用小写字母开头的字母/数字/下划线（2~41 位）"
    if not f.label:
        return False, "因子要有个显示名"
    if f.key in F.FACTORS and not allow_overwrite:
        return False, f"已经有一个叫 {f.key} 的因子了，换个代号"
    if f.hint_dir not in (-1, 0, 1):
        f.hint_dir = 1

    ok, err, warn = trial_run(f)
    if not ok:
        return False, err

    items = [i for i in load_all() if i.key != f.key]
    items.append(f)
    save_all(items)

    spec = F.FactorSpec(
        key=f.key, label=f.label, func=compile_one(f)[0], category=f.category,
        desc=f.desc, hint_dir=f.hint_dir, formula=f.formula, code=f.code,
    )
    F.FACTORS[f.key] = spec
    # 新类别要补进分组列表。下拉是按 `CATEGORIES` 遍历的，漏了这一步就是
    # 「注册成功但选不到」—— 比注册失败还难排查。
    if spec.category not in F.CATEGORIES:
        F.CATEGORIES.append(spec.category)
    return True, warn or f"已注册「{f.label}」"


def remove(key: str) -> bool:
    """从库里删掉。内置因子不能删（它不在 JSON 里，删了下次启动又回来）。"""
    from ttrade import factors as F

    items = load_all()
    left = [i for i in items if i.key != key]
    if len(left) == len(items):
        return False
    save_all(left)
    F.FACTORS.pop(key, None)
    return True


def register_all_into(factors: dict) -> int:
    """启动时把库里的因子挂进 `FACTORS`。返回成功条数。

    单条坏掉不影响其他条：用户手改 JSON 改坏了，不该让整个程序起不来。
    """
    from ttrade import factors as F

    n = 0
    for f in load_all():
        fn, _err = compile_one(f)
        if fn is None:
            continue
        factors[f.key] = F.FactorSpec(
            key=f.key, label=f.label, func=fn, category=f.category,
            desc=f.desc, hint_dir=f.hint_dir, formula=f.formula, code=f.code,
        )
        n += 1
    return n
