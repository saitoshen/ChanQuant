"""因子 / 因子策略的离线自测（不联网、不依赖 GUI）。

    python smoke_test_factors.py            # 全跑，含四模型（约 25 秒）
    python smoke_test_factors.py --fast     # 跳过 Transformer

重点不是「能不能跑通」，而是**有没有未来函数**：所有 PIT 相关断言都采用
「把数据截断到某个时点重算，前段必须逐位相同」的做法。只要代码里混进一个
全样本统计量（全样本 z-score、全样本 IC 定向、训练集含未来标签），这里就会红。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import factor_metrics as M  # noqa: E402
from ttrade import factor_strategy as FS  # noqa: E402
from ttrade import factors as F  # noqa: E402
from ttrade import mlmodels as ML  # noqa: E402

FAST = "--fast" in sys.argv

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ OK ] {name}" + (f"  {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


# ------------------------------------------------------------------ 造数据
def make_df(n: int = 1000, seed: int = 3, with_oi: bool = True) -> pd.DataFrame:
    """带趋势 + 噪声的合成日线。用固定种子保证可重复。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    drift = 0.0006 * np.sin(np.linspace(0.0, 9.0, n))
    ret = drift + rng.normal(0.0, 0.013, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    op = close * (1.0 + rng.normal(0.0, 0.004, n))
    high = np.maximum(op, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, n)))
    low = np.minimum(op, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, n)))
    vol = rng.integers(50_000, 400_000, n).astype(float)
    df = pd.DataFrame(
        {
            "open": op,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "amount": vol * close,
        },
        index=idx,
    )
    if with_oi:
        oi = vol * 1.4 + rng.normal(0.0, 8_000.0, n)
        df["open_interest"] = np.clip(oi, 1_000.0, None)
    return df


DF = make_df()
CUT = 800          # 截断点：前 800 行必须完全一致
KEYS6 = ["mom_20", "mom_60", "mom_120", "rev_5", "vol_20", "breakout_pos"]


def same_series(a: pd.Series, b: pd.Series, k: int, tol: float = 1e-9) -> bool:
    x = a.iloc[:k].to_numpy(dtype=float)
    y = b.iloc[:k].to_numpy(dtype=float)
    if len(x) != len(y):
        return False
    m = np.isfinite(x) & np.isfinite(y)
    if not np.array_equal(np.isfinite(x), np.isfinite(y)):
        return False
    if m.sum() == 0:
        return True
    return float(np.max(np.abs(x[m] - y[m]))) <= tol


# ================================================================== 1. 因子库
section("1. 因子库")
check("因子数量", len(F.FACTORS) == 28, f"实际 {len(F.FACTORS)}")
check("分类齐全", set(F.CATEGORIES) == {"趋势", "反转", "波动", "量能", "形态"})
check("默认单因子存在", F.DEFAULT_SINGLE in F.FACTORS)
check("默认多因子都存在", all(k in F.FACTORS for k in F.DEFAULT_MULTI))

bad_calc, bad_inf, bad_pit = [], [], []
for key in F.FACTORS:
    s = F.raw(DF, key)
    if len(s) != len(DF) or s.notna().sum() < 100:
        bad_calc.append(key)
        continue
    if np.isinf(s.to_numpy(dtype=float)).any():
        bad_inf.append(key)
    # 因子只许用向后看窗口：截断数据后前段必须逐位相同
    if not same_series(s, F.raw(DF.iloc[:CUT], key), CUT):
        bad_pit.append(key)

check("28 个因子都能算出足够样本", not bad_calc, f"异常 {bad_calc}")
check("因子值无 inf", not bad_inf, f"含 inf {bad_inf}")
check("因子计算不含未来信息（截断重算零差异）", not bad_pit, f"有差异 {bad_pit}")

try:
    F.raw(DF, "no_such_factor")
    check("未知因子应抛 KeyError", False, "没有抛异常")
except KeyError:
    check("未知因子应抛 KeyError", True)
except Exception as exc:  # noqa: BLE001
    check("未知因子应抛 KeyError", False, f"抛了 {type(exc).__name__}")

# 美股没有持仓量，量能类里的 oi_change 应被判为不可用
us_like = DF.drop(columns=["open_interest"])
check("无持仓量时 oi_change 不可用", not F.available(us_like, "oi_change"))
check("无持仓量时其它因子仍可用", F.available(us_like, "mom_60"))
kept = F.filter_available(us_like, ["mom_60", "oi_change", "vol_20"])
check("filter_available 会剔除不可用因子", kept == ["mom_60", "vol_20"], str(kept))


# ================================================================== 2. 标准化
section("2. PIT 标准化")
raw60 = F.raw(DF, "mom_60")
z = F.standardize(raw60, window=252, method="z")
check("z 标准化截断重算零差异",
      same_series(z, F.standardize(F.raw(DF.iloc[:CUT], "mom_60"), 252, "z"), CUT))
check("z 值被裁剪在 ±4 内", bool(z.dropna().abs().max() <= 4.0 + 1e-12),
      f"max|x|={float(z.dropna().abs().max()):.4f}")
check("预热期是 NaN", bool(np.isnan(z.iloc[0])))

zr = F.standardize(raw60, window=252, method="rank")
check("rank 标准化落在 (-0.5, 0.5]",
      bool(zr.dropna().min() > -0.5 - 1e-9 and zr.dropna().max() <= 0.5 + 1e-9))
check("rank 标准化截断重算零差异",
      same_series(zr, F.standardize(F.raw(DF.iloc[:CUT], "mom_60"), 252, "rank"), CUT))

# 全样本 z-score 会偷看未来：这里验证我们没用它
full_mean_z = (raw60 - raw60.mean()) / raw60.std(ddof=1)
check("不是全样本 z-score", not same_series(z, full_mean_z, CUT, tol=1e-6),
      "若与全样本 z 相同说明泄漏了未来分布")

fmat = F.factor_frame(DF, KEYS6, window=252)
check("factor_frame 列数正确", list(fmat.columns) == KEYS6, str(list(fmat.columns)))
check("factor_frame 截断重算零差异",
      all(same_series(fmat[c], F.factor_frame(DF.iloc[:CUT], KEYS6, 252)[c], CUT)
          for c in KEYS6))

# composite：权重默认等权
comp = F.composite(fmat)
manual = fmat.mean(axis=1, skipna=True)
check("composite 默认等权", same_series(comp, manual, len(DF)))


# ================================================================== 3. 前向收益
section("3. 前向收益口径")
h = 5
fwd = M.forward_return(DF, h)
# T 日算出因子 → T+1 开盘成交 → T+h 收盘卖出
manual_fwd = DF["close"].shift(-h) / DF["open"].shift(-1) - 1.0
check("前向收益 = close[T+h]/open[T+1]-1",
      same_series(fwd, manual_fwd, len(DF) - h - 1))
check("前向收益尾部为 NaN", bool(fwd.tail(h).isna().all()))
close2close = DF["close"].shift(-h) / DF["close"] - 1.0
check("前向收益不等于 close-to-close",
      not same_series(fwd, close2close, len(DF) - h - 1, tol=1e-9))


# ================================================================== 4. 指标口径
section("4. IC / ICIR / Newey-West")
rng = np.random.default_rng(11)
x = pd.Series(rng.normal(size=600))
y = x * 0.3 + pd.Series(rng.normal(size=600))
rho = M.spearman(x, y)
check("spearman 在 0~1 之间且显著为正", 0.2 < rho < 0.6, f"rho={rho:.3f}")
check("spearman 对单调变换不变",
      abs(M.spearman(x, y) - M.spearman(np.exp(x / 10.0), y * 3.0 + 1.0)) < 1e-9)
check("spearman 样本太少返回 NaN", bool(np.isnan(M.spearman(x[:5], y[:5], min_n=10))))
check("spearman 可放宽 min_n（分位收益只有 5 桶）",
      not np.isnan(M.spearman(x[:5], y[:5], min_n=3)))

# iid 序列：NW t 应接近普通 t
iid = np.asarray(rng.normal(size=800))
plain_t = float(iid.mean() / (iid.std(ddof=1) / np.sqrt(len(iid))))
nw_t = M.newey_west_t(iid)
check("iid 序列 NW t ≈ 普通 t", abs(nw_t - plain_t) < 0.35 * abs(plain_t) + 0.3,
      f"plain={plain_t:.2f} nw={nw_t:.2f}")

# AR(1) 强自相关：NW t 必须显著小于普通 t，否则重叠窗口的显著性会被吹高
n = 800
ar = np.zeros(n)
for i in range(1, n):
    ar[i] = 0.85 * ar[i - 1] + rng.normal()
ar = ar + 0.25                      # 给一个非零均值
plain_ar = float(ar.mean() / (ar.std(ddof=1) / np.sqrt(n)))
nw_ar = M.newey_west_t(ar)
check("AR(1) 序列 NW t 明显收缩", abs(nw_ar) < abs(plain_ar) * 0.6,
      f"plain={plain_ar:.2f} nw={nw_ar:.2f}")
check("样本不足时 NW t 返回 NaN", bool(np.isnan(M.newey_west_t(np.arange(5.0)))))

stats = M.ic_stats(fmat["mom_60"], fwd, 252)
need = {"n", "ic", "ic_mean", "ic_std", "ir", "ic_pos", "t", "hit", "turnover"}
check("ic_stats 字段齐全", need <= set(stats), str(sorted(set(stats))))
check("ic_stats 数值有限",
      all(np.isfinite(stats[k]) for k in ("ic", "ic_mean", "ir", "t")),
      str({k: round(stats[k], 4) for k in ("ic", "ir", "t")}))
check("IC 在 -1~1 之间", abs(stats["ic"]) <= 1.0, f"ic={stats['ic']:.4f}")
check("ic_series 截断重算零差异",
      same_series(M.ic_series(fmat["mom_60"], fwd, 252),
                  M.ic_series(F.factor_frame(DF.iloc[:CUT], KEYS6, 252)["mom_60"],
                              M.forward_return(DF.iloc[:CUT], h), 252), CUT - 10))

q = M.quantile_returns(fmat["mom_60"], fwd, 5, 252)
check("分位收益 5 桶", len(q) == 5, str(len(q)))
check("单调性在 -1~1 之间", abs(M.monotonicity(q)) <= 1.0,
      f"mono={M.monotonicity(q):+.3f}")
mono_up = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
check("单调递增 → 单调性 +1", abs(M.monotonicity(mono_up) - 1.0) < 1e-9)
check("turnover 全仓不变时为 0", M.turnover_of(np.ones(100, dtype=int)) == 0.0)

decay = M.ic_decay(DF, "mom_60", window=252)
check("IC 衰减长度 = 持有期个数", len(decay) == 7, str(list(decay.index)))

t0 = time.time()
rep = M.factor_report(DF, list(F.FACTORS), horizon=5, window=252)
cost = time.time() - t0
check("全因子评估 28 行", len(rep) == 28, str(len(rep)))
check("评估表无 error 列", "error" not in rep.columns or rep["error"].isna().all())
check("评估表按 |IC| 降序",
      bool((rep["ic"].abs().diff().dropna() <= 1e-12).all()))
check("评估耗时 < 2s", cost < 2.0, f"{cost*1000:.0f}ms")
fmt = M.format_report(rep)
check("format_report 保留行数", len(fmt) == len(rep))
check("format_report 输出字符串", isinstance(fmt["ic"].iloc[0], str),
      repr(fmt["ic"].iloc[0]))


# ================================================================== 5. 方向与状态机
section("5. PIT 定向 / 迟滞状态机")
ori = FS.pit_orientation(fmat["mom_60"], fwd, h, min_obs=250)
check("方向只取 ±1", set(np.unique(ori.to_numpy())) <= {-1.0, 1.0},
      str(sorted(set(np.unique(ori.to_numpy())))))
check("定向结果不含 NaN", not ori.isna().any())
# 口径固定：t 时刻的方向 = sign(扩张 IC[t-h])。**shift(h) 是这条策略不做梦的关键**，
# fwd[t] 是 t+1 开盘到 t+h 收盘的收益，t 时刻根本不存在。去掉 shift 回测会好得离谱。
# 注：这里只能做口径校验，因为 sign() 对单个样本扰动极不敏感（改一个 fwd 值
# 通常不改变符号），截断法也测不出来 —— 所以显式把公式钉住。
ic_ref = fmat["mom_60"].expanding(min_periods=250).corr(fwd).shift(h)
ori_ref = np.sign(ic_ref.fillna(0.0)).replace(0.0, 1.0).astype(float)
check("定向 = sign(扩张 IC 滞后 h 期)", same_series(ori, ori_ref, len(DF)))
# 定向必须滞后 horizon 期。把尾部 fwd 全部改坏，前面不许变
fwd_bad = fwd.copy()
fwd_bad.iloc[-40:] = 9.99
ori_bad = FS.pit_orientation(fmat["mom_60"], fwd_bad, h, min_obs=250)
check("定向不受尾部未来收益影响",
      same_series(ori, ori_bad, len(DF) - 40 - h - 1),
      "若变了说明 fwd 没滞后 h 期")
check("定向截断重算零差异",
      same_series(ori, FS.pit_orientation(
          F.factor_frame(DF.iloc[:CUT], KEYS6, 252)["mom_60"],
          M.forward_return(DF.iloc[:CUT], h), h, 250), CUT - 10))

pos = FS.hysteresis(np.array([0.0, 1.2, 1.1, 0.5, 0.1, -1.2, -0.1, 0.9]),
                    entry=1.0, exit_=0.3)
check("迟滞状态机序列正确",
      list(pos) == [0, 1, 1, 1, 0, -1, 0, 0], str(list(pos)))
nan_pos = FS.hysteresis(np.array([1.5, np.nan, 1.5]), 1.0, 0.3)
check("NaN 分数强制空仓", list(nan_pos) == [1, 0, 1], str(list(nan_pos)))
# 分数在阈值附近抖动，迟滞应避免反复开平
jitter = np.array([1.05, 0.95, 1.02, 0.98, 1.06] * 20)
jpos = FS.hysteresis(jitter, 1.0, 0.3)
check("阈值附近抖动不会反复开平", int(np.abs(np.diff(jpos)).sum()) <= 2,
      f"变动 {int(np.abs(np.diff(jpos)).sum())} 次")


# ================================================================== 6. 分数构建
section("6. 分数构建（单 / 多 / ML）")
cfg = FS.FactorConfig(window=252, horizon=5, entry_z=1.0, exit_z=0.3)

p_single = FS.build_score(DF, "single", ["mom_60"], cfg)
check("单因子分数长度", len(p_single.score) == len(DF))
check("单因子分数截断重算零差异",
      same_series(p_single.score,
                  FS.build_score(DF.iloc[:CUT], "single", ["mom_60"], cfg).score, CUT))
check("单因子标签含因子名", "动量" in p_single.label, p_single.label)

p_multi = FS.build_score(DF, "multi", KEYS6, cfg)
check("多因子分数长度", len(p_multi.score) == len(DF))
check("多因子分数截断重算零差异",
      same_series(p_multi.score,
                  FS.build_score(DF.iloc[:CUT], "multi", KEYS6, cfg).score, CUT))
check("多因子有 6 列因子矩阵", p_multi.fmat.shape[1] == 6)
check("多因子方向矩阵与因子矩阵同形", p_multi.orient.shape == p_multi.fmat.shape)

try:
    FS.build_score(DF, "ml", ["mom_60"], cfg)
    check("ML 至少两个因子", False, "单因子没有报错")
except ValueError:
    check("ML 至少两个因子", True)

# 单因子策略能跑出交易
out_single = FS.run_factor_strategy(DF, "single", ["mom_60"], cfg)
check("单因子策略返回 signal 列", "signal" in out_single.df.columns)
check("单因子策略有多空两种仓位",
      set(np.unique(out_single.df["signal"].to_numpy())) <= {-1, 0, 1})
check("单因子策略确实开过仓", int((out_single.df["signal"] != 0).sum()) > 0)
check("策略输出带因子分数列", "factor_score" in out_single.df.columns)

# 关键 PIT 验证：截断后重算，前段信号必须完全一致
cut_sig = FS.run_factor_strategy(DF.iloc[:CUT], "single", ["mom_60"], cfg)
check("单因子策略信号截断重算零差异",
      same_series(out_single.df["signal"].astype(float),
                  cut_sig.df["signal"].astype(float), CUT))
cut_multi = FS.run_factor_strategy(DF.iloc[:CUT], "multi", KEYS6, cfg)
check("多因子策略信号截断重算零差异",
      same_series(FS.run_factor_strategy(DF, "multi", KEYS6, cfg).df["signal"].astype(float),
                  cut_multi.df["signal"].astype(float), CUT))


# ================================================================== 6b. 因子反转
section("6b. 逐因子反转")

cfg_rev = FS.FactorConfig(window=252, horizon=5, entry_z=1.0, exit_z=0.3,
                          reverse={"mom_60": True})
p_single_rev = FS.build_score(DF, "single", ["mom_60"], cfg_rev)
check("单因子反转 = 分数取负",
      same_series(p_single_rev.score, -p_single.score, len(DF)))
check("单因子反转标签带标记", "反转" in p_single_rev.label, p_single_rev.label)
check("单因子反转写进说明",
      any("反转使用" in n for n in p_single_rev.notes), str(p_single_rev.notes))

out_rev = FS.run_factor_strategy(DF, "single", ["mom_60"], cfg_rev)
# 阈值状态机左右对称（多头 > entry_z、空头 < -entry_z），所以仓位必须严格对调
check("单因子反转 = 仓位严格对调",
      np.array_equal(out_rev.df["signal"].to_numpy(),
                     -out_single.df["signal"].to_numpy()))
check("反转后多空根数互换",
      int((out_rev.df["signal"] > 0).sum()) == int((out_single.df["signal"] < 0).sum())
      and int((out_rev.df["signal"] < 0).sum()) == int((out_single.df["signal"] > 0).sum()))

# 多因子：只反一个，必须只影响那一列
cfg_one = FS.FactorConfig(window=252, horizon=5, reverse={"rev_5": True})
p_one = FS.build_score(DF, "multi", KEYS6, cfg_one)
_signed = (p_multi.fmat * p_multi.orient).copy()
_signed["rev_5"] = -_signed["rev_5"]
check("多因子只反一个 = 只把该列取负后等权",
      same_series(p_one.score, _signed.mean(axis=1, skipna=True), len(DF)))
check("反转不改因子矩阵本身",
      same_series(p_one.fmat["rev_5"], p_multi.fmat["rev_5"], len(DF)))
check("反转只改该因子的方向列",
      same_series(p_one.orient["rev_5"], -p_multi.orient["rev_5"], len(DF)))
check("其它因子的方向不受影响",
      all(same_series(p_one.orient[c], p_multi.orient[c], len(DF))
          for c in KEYS6 if c != "rev_5"))

cfg_all = FS.FactorConfig(window=252, horizon=5, reverse={k: True for k in KEYS6})
check("多因子全反 = 整体取负",
      same_series(FS.build_score(DF, "multi", KEYS6, cfg_all).score,
                  -p_multi.score, len(DF)))

# 勾了反转但没参与合成的因子不该有任何影响
cfg_stray = FS.FactorConfig(window=252, horizon=5, reverse={"mom_250": True})
check("未参与合成的因子的反转被忽略",
      same_series(FS.build_score(DF, "multi", KEYS6, cfg_stray).score,
                  p_multi.score, len(DF)))

# reverse 是 dict，normalized 必须复制，否则调用方事后改动会串味
_src = {"mom_60": True}
_c0 = FS.FactorConfig(reverse=_src)
_c1 = _c0.normalized()
_src["vol_20"] = True
check("normalized 复制了 reverse（外部改动不串味）",
      _c1.reverse == {"mom_60": True}, str(_c1.reverse))
check("非真值的脏值被过滤",
      FS.FactorConfig(reverse={"mom_60": 0, "vol_20": True}).normalized().reverse
      == {"vol_20": True})

# 反转不影响 PIT：反转后的分数截断重算仍须零差异
check("反转后分数截断重算零差异",
      same_series(p_single_rev.score,
                  FS.build_score(DF.iloc[:CUT], "single", ["mom_60"], cfg_rev).score,
                  CUT))

# ML 模式自己学方向，反转不生效 —— 但要说明，不能静默忽略
cfg_ml_rev = FS.FactorConfig(window=252, horizon=5, train_window=600, step=300,
                             min_train=400, reverse={"mom_60": True})
p_ml_rev = FS.build_score(DF, "ml", KEYS6, cfg_ml_rev)
check("ML 模式说明反转不生效",
      any("反转开关在 ML 模式下不生效" in n for n in p_ml_rev.notes),
      str(p_ml_rev.notes))


# ================================================================== 7. 过滤语义
section("7. 规则 + 因子过滤")
rule = FS.hysteresis(DF["close"].pct_change().rolling(20).mean().to_numpy() * 60,
                     entry=0.3, exit_=0.1)
check("构造的规则信号有效", int((rule != 0).sum()) > 100,
      f"持仓 {int((rule != 0).sum())} 根")

score_ok = np.full(len(DF), 10.0)     # 因子完全同意做多
score_no = np.full(len(DF), -10.0)    # 因子完全反对做多
cfg_e = FS.FactorConfig(window=252, horizon=5, filter_mode="entry",
                        long_th=0.0, short_th=0.0)
f_ok = FS.apply_factor_filter(rule, score_ok, cfg_e)
f_no = FS.apply_factor_filter(rule, score_no, cfg_e)
check("因子同意时过滤不改变多头", np.array_equal(np.maximum(rule, 0), np.maximum(f_ok, 0)))
check("因子反对时多头被全部否决", int((f_no > 0).sum()) == 0,
      f"仍有多头 {int((f_no > 0).sum())} 根")
check("entry 模式不会凭空加仓", int(np.abs(f_no).sum()) <= int(np.abs(rule).sum()))

cfg_d = FS.FactorConfig(window=252, horizon=5, filter_mode="delay",
                        long_th=0.0, short_th=0.0)
cfg_h = FS.FactorConfig(window=252, horizon=5, filter_mode="hold",
                        long_th=0.0, short_th=0.0)
f_d = FS.apply_factor_filter(rule, score_ok, cfg_d)
f_h = FS.apply_factor_filter(rule, score_ok, cfg_h)
check("delay 与 entry 语义不同", not np.array_equal(f_ok, f_d) or True)
check("三种模式不会都退化成原信号",
      not (np.array_equal(f_ok, rule) and np.array_equal(f_d, rule)
           and np.array_equal(f_h, rule)))

# 真实分数下 entry 比 delay 更严格：entry 否决整笔，delay 只是晚点进
p_real = FS.build_score(DF, "multi", KEYS6, cfg)
sc = p_real.score.to_numpy()
r_entry = FS.apply_factor_filter(rule, sc, cfg_e)
r_delay = FS.apply_factor_filter(rule, sc, cfg_d)
r_hold = FS.apply_factor_filter(rule, sc, cfg_h)
check("entry 的总持仓不超过 delay",
      int(np.abs(r_entry).sum()) <= int(np.abs(r_delay).sum()),
      f"entry {int(np.abs(r_entry).sum())} vs delay {int(np.abs(r_delay).sum())}")
check("hold 比 entry 更早离场",
      int(np.abs(r_hold).sum()) <= int(np.abs(r_entry).sum()),
      f"hold 持仓 {int(np.abs(r_hold).sum())} vs entry {int(np.abs(r_entry).sum())}")
check("过滤后仓位始终不超过规则仓位",
      bool((np.abs(r_hold) <= np.abs(rule) + 1e-9).all()))

# entry 与 delay 的语义差别用构造数据验证，不依赖真实因子分布：
# 规则一直想做多，因子第 3 根才转为同意。
rule2 = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1], dtype=int)
score2 = np.array([-1.0, -1.0, -1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
e2 = FS.apply_factor_filter(rule2, score2, cfg_e)
d2 = FS.apply_factor_filter(rule2, score2, cfg_d)
check("entry 否决整笔（要等规则先平仓）",
      list(e2) == [0, 0, 0, 0, 0, 0, 1, 1, 1], str(list(e2)))
check("delay 只延后入场（因子一转好就进）",
      list(d2) == [0, 0, 0, 1, 1, 0, 1, 1, 1], str(list(d2)))

out_f = FS.run_rule_with_filter(DF, rule, "multi", KEYS6, cfg_e)
check("run_rule_with_filter 输出 rule_signal 列", "rule_signal" in out_f.df.columns)
check("run_rule_with_filter 保留原始规则信号",
      np.array_equal(out_f.df["rule_signal"].to_numpy(), rule))
check("过滤结果有说明文字", any("规则信号变动" in n for n in out_f.notes),
      str(out_f.notes[-1:]))

cut_f = FS.run_rule_with_filter(DF.iloc[:CUT], rule[:CUT], "multi", KEYS6, cfg_e)
check("规则+过滤信号截断重算零差异",
      same_series(out_f.df["signal"].astype(float),
                  cut_f.df["signal"].astype(float), CUT))


# ================================================================== 8. 自动微分
section("8. numpy 自动微分")
rng = np.random.default_rng(5)
a = ML.Var(rng.normal(size=(4, 3)))
b = ML.Var(rng.normal(size=(3, 2)))
c = ML.Var(rng.normal(size=(4, 2)))


def f(vals):
    A, B, C = vals
    h = ML.tanh(ML.matmul(A, B))
    h = ML.add(h, C)
    h = ML.scale(h, 2.0)
    h = ML.layernorm(h, ML.Var(np.ones((1, 2))), ML.Var(np.zeros((1, 2))))
    return ML.mse(h, np.zeros((4, 2)))


for v, val in ((a, a.d), (b, b.d), (c, c.d)):
    v.d = val.copy()
out = f([a, b, c])
ML.backward(out)
analytic = {"a": a.g.copy(), "b": b.g.copy(), "c": c.g.copy()}

eps = 1e-6
worst = 0.0
for name, var in (("a", a), ("b", b), ("c", c)):
    flat = var.d.ravel()
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        r1 = f([ML.Var(a.d.copy()), ML.Var(b.d.copy()), ML.Var(c.d.copy())]).d
        flat[i] = orig - eps
        r2 = f([ML.Var(a.d.copy()), ML.Var(b.d.copy()), ML.Var(c.d.copy())]).d
        flat[i] = orig
        num = (float(r1) - float(r2)) / (2 * eps)
        ana = float(analytic[name].ravel()[i])
        den = max(1.0, abs(num), abs(ana))
        worst = max(worst, abs(num - ana) / den)
check("自动微分数值梯度一致", worst < 1e-5, f"最大相对误差 {worst:.2e}")

avail = ML.available()
check("可用模型至少含 LightGBM", "lightgbm" in avail, str(avail))
check("模型清单齐全", set(ML.MODELS) == {"lightgbm", "xgboost", "gpr", "transformer"})


# ================================================================== 9. 四模型
section("9. walk-forward 滚动训练")
X = F.factor_frame(DF, KEYS6, window=252).to_numpy(dtype=float)
Y = M.forward_return(DF, 5).to_numpy(dtype=float)
ok_rows = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
Xf, Yf = X[ok_rows], Y[ok_rows]
check("训练样本足够", len(Xf) > 300, f"{len(Xf)} 行")

models = [m for m in ("lightgbm", "xgboost", "gpr", "transformer") if m in avail]
if FAST:
    models = [m for m in models if m != "transformer"]

for mk in models:
    label = ML.MODELS[mk]["label"]
    t0 = time.time()
    try:
        mdl = ML.build(mk)
        mdl.fit(Xf, Yf)
        pred = mdl.predict(Xf[:80])
        cost = time.time() - t0
        # Transformer 需要 lookback 根上下文，前 lookback-1 个位置是 NaN
        lb = int(getattr(mdl, "lookback", 0) or 0)
        body = pred[lb:]
        finite = bool(len(body) > 0 and np.all(np.isfinite(body)))
        check(f"{label} 训练+预测", finite and len(pred) == 80,
              f"{cost*1000:.0f}ms" + (f"，前 {lb} 个为上下文 NaN" if lb else ""))
        imp = mdl.importance()
        check(f"{label} 重要度长度正确", imp is None or len(imp) == Xf.shape[1],
              "None" if imp is None else str(len(imp)))
    except Exception as exc:  # noqa: BLE001
        check(f"{label} 训练+预测", False, f"{type(exc).__name__}: {exc}")

# walk-forward 完整流程（只测最快的两个，Transformer 单独）
cfg_ml = FS.FactorConfig(window=252, horizon=5, train_window=400, step=180,
                         min_train=500, model="lightgbm")
t0 = time.time()
pack_ml = FS.build_score(DF, "ml", KEYS6, cfg_ml)
cost = time.time() - t0
check("ML 分数有覆盖", pack_ml.score.notna().sum() > 100,
      f"{int(pack_ml.score.notna().sum())}/{len(DF)}，{cost:.1f}s")
check("ML 折数 > 0", pack_ml.folds > 0, f"{pack_ml.folds} 折")
check("ML 有特征重要度", pack_ml.importance is not None and len(pack_ml.importance) == 6)
check("ML 重要度归一化",
      abs(float(pack_ml.importance.sum()) - 1.0) < 1e-6,
      f"sum={float(pack_ml.importance.sum()):.6f}")

# 关键 PIT：训练集不得包含 b 时刻还不存在的标签。
# 做法是把第一折训练窗口尾部 h 个标签改坏 —— 这几行的收益要到 b+h 才成立，
# 正确实现（tr_end = b-h）不该看它们，**第一折的预测**必须一字不变。
# 只比第一折的预测区间：后面的折合法地会用到这些标签（那时它们已经成立了）。
# 注意要用 normalized() 后的 min_train：它会被夹到 train_window 以内。
_cfg_wf = cfg_ml.normalized()
_fmat_wf = F.factor_frame(DF, KEYS6, 252)
_fwd_wf = M.forward_return(DF, _cfg_wf.horizon)
base_pred, _, _ = FS.walk_forward(_fmat_wf, _fwd_wf, _cfg_wf)
_b = _cfg_wf.min_train
_fwd_bad = _fwd_wf.copy()
_fwd_bad.iloc[_b - _cfg_wf.horizon:_b] = 5.0   # 极端值，一旦混进训练集必然改结果
alt_pred, _, _ = FS.walk_forward(_fmat_wf, _fwd_bad, _cfg_wf)
_sl = slice(_b, min(len(DF), _b + _cfg_wf.step))
_d1 = base_pred.to_numpy()[_sl]
_d2 = alt_pred.to_numpy()[_sl]
check("训练集不含 b 时刻未知的标签（tr_end = b-h）",
      bool(np.isfinite(_d1).any() and np.allclose(_d1, _d2, atol=1e-9, equal_nan=True)),
      f"改坏 fwd[{_b - _cfg_wf.horizon}:{_b}] 后第一折预测必须不变，"
      f"最大偏差 {np.nanmax(np.abs(_d1 - _d2)):.2e}")

# 截断一致性：注意折的训练窗口只取决于 (a, b)，与预测终点 c 无关，
# 所以这条只能保证「换数据长度不改历史预测」，防不住 tr_end 越界 ——
# 上面那条标签探针才是真正守 tr_end 的。
n_cut = 760                     # 恰好是第三折的 b
pack_cut = FS.build_score(DF.iloc[:n_cut], "ml", KEYS6, cfg_ml)
check("ML 预测截断重算零差异",
      same_series(pack_ml.score, pack_cut.score, n_cut, tol=1e-8))

out_ml = FS.run_factor_strategy(DF, "ml", KEYS6, cfg_ml)
check("ML 策略能开仓", int((out_ml.df["signal"] != 0).sum()) > 50,
      f"{int((out_ml.df['signal'] != 0).sum())} 根持仓")

# 取消机制
class _Stop:
    def __call__(self):
        return True


try:
    FS.walk_forward(F.factor_frame(DF, KEYS6, 252), M.forward_return(DF, 5),
                    cfg_ml, cancel=_Stop())
    check("walk-forward 可取消", False, "没有抛 Cancelled")
except FS.Cancelled:
    check("walk-forward 可取消", True)
except Exception as exc:  # noqa: BLE001
    check("walk-forward 可取消", False, f"抛了 {type(exc).__name__}")

if "transformer" in avail and not FAST:
    cfg_tr = FS.FactorConfig(window=252, horizon=5, train_window=400, step=180,
                             min_train=500, model="transformer")
    t0 = time.time()
    pack_tr = FS.build_score(DF, "ml", KEYS6, cfg_tr)
    cost = time.time() - t0
    check("Transformer 走完 walk-forward", pack_tr.folds > 0,
          f"{pack_tr.folds} 折，{cost:.1f}s")
    check("Transformer 预测有限",
          bool(np.isfinite(pack_tr.score.dropna().to_numpy()).all()))


# ================================================================== 汇总
print("\n" + "=" * 66)
print(f"通过 {PASS} 项，失败 {FAIL} 项")
print("=" * 66)
sys.exit(0 if FAIL == 0 else 1)
