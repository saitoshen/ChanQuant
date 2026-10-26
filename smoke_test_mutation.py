"""变异检验：往因子链路里注入未来函数，确认 smoke_test_factors.py 的探针会变红。

    python smoke_test_mutation.py

为什么需要这个脚本
------------------
写了 100 项断言不等于能抓到 bug。第一版 smoke_test_factors 用了「截断重算」一种手法，
结果两个真问题它完全无感：

1. `pit_orientation` 去掉 `shift(horizon)`（用当期收益定方向）——`sign()` 对单个样本
   扰动极不敏感，截断也改不了符号，测不出来。
2. `walk_forward` 把 `tr_end = b - horizon` 写成 `b`（训练集吃到未来标签）——
   折的训练窗口只取决于 `(a, b)`，与预测终点 `c` 无关，砍数据长度不会改变任何一折。

所以改成三个针对性探针（见 smoke_test_factors.py），再用这个脚本反向验证：
把源码**只改一行**，探针必须变红。红不了，说明探针是摆设。

实现方式：读源码、替换一行、exec 成模块再调用。变异体一定是真代码，只差那一行。
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ttrade import factor_metrics as M  # noqa: E402
from ttrade import factor_strategy as FS  # noqa: E402
from ttrade import factors as F  # noqa: E402

KEYS = ["mom_20", "mom_60", "mom_120", "rev_5", "vol_20", "breakout_pos"]


def make_df(n: int = 1000, seed: int = 3) -> pd.DataFrame:
    """合成日线：趋势 + 噪声，固定种子保证可重复。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    ret = 0.0006 * np.sin(np.linspace(0.0, 9.0, n)) + rng.normal(0.0, 0.013, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    op = close * (1.0 + rng.normal(0.0, 0.004, n))
    high = np.maximum(op, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, n)))
    low = np.minimum(op, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, n)))
    vol = rng.integers(50_000, 400_000, n).astype(float)
    return pd.DataFrame(
        {"open": op, "high": high, "low": low, "close": close,
         "volume": vol, "amount": vol * close},
        index=idx,
    )


def exec_as_module(code: str, path: str, name: str) -> dict:
    """exec 成真正的模块对象并注册到 sys.modules —— dataclass 需要它。"""
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(code, path, "exec"), mod.__dict__)
    return mod.__dict__


def mutate(rel: str, old: str, new: str, name: str) -> dict:
    path = os.path.join(ROOT, "ttrade", rel)
    code = open(path, encoding="utf-8").read()
    if old not in code:
        raise SystemExit(f"变异锚点找不到，源码改过了？\n  文件：{rel}\n  锚点：{old!r}")
    return exec_as_module(code.replace(old, new, 1), path, name)


def same_series(a: pd.Series, b: pd.Series, k: int, tol: float = 1e-9) -> bool:
    x = a.iloc[:k].to_numpy(dtype=float)
    y = b.iloc[:k].to_numpy(dtype=float)
    if len(x) != len(y) or not np.array_equal(np.isfinite(x), np.isfinite(y)):
        return False
    m = np.isfinite(x) & np.isfinite(y)
    return True if m.sum() == 0 else float(np.max(np.abs(x[m] - y[m]))) <= tol


DF = make_df()
CUT = 800
FMAT = F.factor_frame(DF, KEYS, 252)
FWD = M.forward_return(DF, 5)


# ---------------------------------------------------------------- 三个探针
def probe_orientation(pit_fn) -> bool:
    """定向口径：必须等于 sign(扩张 IC 滞后 h 期)。"""
    ori = pit_fn(FMAT["mom_60"], FWD, 5, min_obs=250)
    ic_ref = FMAT["mom_60"].expanding(min_periods=250).corr(FWD).shift(5)
    ref = np.sign(ic_ref.fillna(0.0)).replace(0.0, 1.0).astype(float)
    return same_series(ori, ref, len(DF))


def probe_train_bound(wf_fn) -> bool:
    """训练集边界：改坏 b-h..b 的标签，第一折预测必须一字不变。"""
    cfg = FS.FactorConfig(window=252, horizon=5, train_window=400, step=180,
                          min_train=500, model="lightgbm").normalized()
    fwd = M.forward_return(DF, cfg.horizon)
    base, _, _ = wf_fn(FMAT, fwd, cfg)
    b = cfg.min_train
    bad = fwd.copy()
    bad.iloc[b - cfg.horizon:b] = 5.0
    alt, _, _ = wf_fn(FMAT, bad, cfg)
    sl = slice(b, min(len(DF), b + cfg.step))
    d1, d2 = base.to_numpy()[sl], alt.to_numpy()[sl]
    return bool(np.isfinite(d1).any()
                and np.allclose(d1, d2, atol=1e-9, equal_nan=True))


def probe_std_truncation(frame_fn) -> bool:
    """标准化：截断重算，前段必须逐位相同。"""
    a = frame_fn(DF, KEYS, 252)["mom_60"]
    b = frame_fn(DF.iloc[:CUT], KEYS, 252)["mom_60"]
    return same_series(a, b, CUT)


PASS = 0
FAIL = 0


def check(name: str, cond: bool, want: bool, note: str = "") -> None:
    global PASS, FAIL
    ok = cond == want
    if ok:
        PASS += 1
        print(f"  [ OK ] {name}  = {cond}" + (f"（{note}）" if note else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  = {cond}，应为 {want}" + (f"（{note}）" if note else ""))


print("基线：原始实现，三个探针都应为 True")
check("A 定向口径", probe_orientation(FS.pit_orientation), True)
check("B 训练集边界", probe_train_bound(FS.walk_forward), True)
check("C 标准化截断", probe_std_truncation(F.factor_frame), True)

print("\n变异 1：pit_orientation 去掉 shift(horizon)")
m1 = mutate("factor_strategy.py", "    ic = ic.shift(int(horizon))",
            "    ic = ic  # 变异", "fs_mut1")
check("A 定向口径", probe_orientation(m1["pit_orientation"]), False,
      "探针必须变红，否则抓不到「用当期收益定方向」")

print("\n变异 2：walk_forward 把 tr_end 改成 b")
m2 = mutate("factor_strategy.py", "        tr_end = b - cfg.horizon",
            "        tr_end = b  # 变异", "fs_mut2")
check("B 训练集边界", probe_train_bound(m2["walk_forward"]), False,
      "探针必须变红，否则抓不到「训练集吃未来标签」")

print("\n变异 3：standardize 用全样本均值/标准差")
m3 = mutate("factors.py", "    m = roll.mean()\n    sd = roll.std(ddof=1)",
            "    m = s.mean() + 0 * s\n    sd = s.std(ddof=1) + 0 * s  # 变异", "f_mut3")
check("C 标准化截断", probe_std_truncation(m3["factor_frame"]), False,
      "探针必须变红，否则抓不到「全样本 z-score」")

print("\n" + "=" * 62)
print(f"通过 {PASS} 项，失败 {FAIL} 项")
print("=" * 62)
sys.exit(0 if FAIL == 0 else 1)
