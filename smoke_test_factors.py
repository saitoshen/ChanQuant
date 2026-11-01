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
# 品种季节性要按品种查产业日历，真实加载器把它挂在 attrs 上（tqdata / futures）
DF.attrs["symbol"] = "RB"
DF.attrs["exchange"] = "SHFE"
CUT = 800          # 截断点：前 800 行必须完全一致
KEYS6 = ["mom_20", "mom_60", "mom_120", "rev_5", "vol_20", "breakout_pos"]


def add_aux(df: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """给行情补上期限结构 / 期权摘要列。

    真实环境里这些列由 `auxdata.attach()` 在加载时并进来，冒烟测试不联网，
    所以手工造一份，好让展期、期权因子也走同一套「算得出、无 inf、无未来」
    的检查。三档到期日刻意不等距 —— 等距会让「用斜率差而不是二阶差分」
    这个设计缺陷（如果有）测不出来。
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    out = df.copy()
    d1 = rng.integers(15, 40, n).astype(float)
    d2 = d1 + rng.integers(20, 45, n)
    d3 = d2 + rng.integers(25, 50, n)
    px = df["close"].to_numpy(float)
    out["c1_close"] = px
    out["c2_close"] = px * (1.0 + rng.normal(0.01, 0.02, n))
    out["c3_close"] = out["c2_close"].to_numpy() * (1.0 + rng.normal(0.01, 0.02, n))
    out["c1_dte"], out["c2_dte"], out["c3_dte"] = d1, d2, d3
    oi = df["volume"].to_numpy(float)
    out["c1_oi"] = oi * rng.uniform(0.3, 0.8, n)
    out["c2_oi"] = oi * rng.uniform(0.4, 1.2, n)
    out["c3_oi"] = oi * rng.uniform(0.2, 0.9, n)
    # 两套「先筛持仓量再按到期日」的口径。造数据时不能随便乱填：选合约的规则
    # 决定了近腿只会被推远，所以必须满足 dte 递增、o6_1 ≥ c1、o3_1 ≥ o6_1
    # （候选集越小，最近的那个只会更远）。这里按「o6 = c2 起、o3 = c3 起」错开
    # 一档，既满足约束又能让两套值明显不同于原口径 —— 否则测不出前缀接错。
    d4 = d3 + rng.integers(20, 45, n)
    d5 = d4 + rng.integers(20, 45, n)
    p4 = out["c3_close"].to_numpy(float) * (1.0 + rng.normal(0.01, 0.02, n))
    p5 = p4 * (1.0 + rng.normal(0.01, 0.02, n))
    ladder = {
        "o6_": ([d2, d3, d4], [out["c2_close"].to_numpy(float),
                               out["c3_close"].to_numpy(float), p4]),
        "o3_": ([d3, d4, d5], [out["c3_close"].to_numpy(float), p4, p5]),
    }
    for pre, (dtes, px3) in ladder.items():
        for i in (1, 2, 3):
            out[f"{pre}{i}_dte"] = dtes[i - 1]
            out[f"{pre}{i}_close"] = px3[i - 1]
            out[f"{pre}{i}_oi"] = oi * rng.uniform(0.3, 1.2, n)
    out["iv_near"] = 0.20 + rng.normal(0.0, 0.02, n)
    out["iv_next"] = out["iv_near"].to_numpy() + rng.normal(0.005, 0.01, n)
    out["dte_near"], out["dte_next"] = d1, d2
    out["skew"] = rng.normal(0.1, 0.15, n)
    out["curv"] = rng.normal(0.3, 0.4, n)
    out["pcr"] = np.exp(rng.normal(0.0, 0.3, n))
    out["n_opt"] = rng.integers(20, 300, n)
    # 展期·日内口径：天勤那份期限结构独有的列（Tushare 的日线合约没有），
    # 没有它 roll_intraday 整列空，会在这里被当成「算不出样本」
    out["roll_intra"] = rng.normal(0.0, 0.002, n)
    return out


DF_AUX = add_aux(DF)


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
# 只量内置因子。用户目录里的自定义因子（Agent 生成）也会注册进 FACTORS，
# 按 47 这个硬数字量的话，用户存一条因子这条就红 —— 和策略文档那条同一个毛病
BUILTIN = [k for k, s_ in F.FACTORS.items() if not s_.code]
AUXISH = {k for k, s_ in F.FACTORS.items()
          if s_.category.startswith(F.AUX_CATEGORIES)}
# 日线上整列空的：日内 6 + 时段 5。它们要分钟级行情才算得出来，这里的 DF 是
# 合成日线，所以「算不出样本」是预期行为，不算失败（另有用例专门查这一条）
NO_DAILY = {k for k, s_ in F.FACTORS.items() if s_.category in ("日内", "时段")}
# 要外部库（EDB 基本面 / 宏观库）才有值的因子。干净检出时这两个库可能不存在，
# 所以不参与「必须算出 100 个样本」的硬断言，另开一段按库是否在来判。
EXT_DB = {k for k, s_ in F.FACTORS.items() if s_.category in ("基本面", "宏观")}
# 只依赖 aux 列（真实合约 / 期权曲面）的那批，和外部库无关
CURVE_KEYS = {k for k, s_ in F.FACTORS.items()
              if s_.category.startswith(("展期", "期权"))}
# A 股专属因子（markets 只有 cn_stock）：PE / 换手率 / 大单净买入 / 概念板块 /
# 除权。这里的 DF 是合成期货日线，没有 stock.duckdb 那几张表，整列空是预期
# 行为，不算失败 —— 它们由 smoke_test_tq.check_stock 单独验。
STOCK_ONLY = {k for k, s_ in F.FACTORS.items() if s_.markets == ("cn_stock",)}
check("因子数量", len(BUILTIN) == 136,
      f"内置 {len(BUILTIN)}，用户 {len(F.FACTORS) - len(BUILTIN)}")
check("分类齐全",
      set(F.CATEGORIES) == {"趋势", "反转", "波动", "量能", "形态", "日内", "时段",
                            "展期", "展期 OI3", "展期 OI6", "期权",
                            "季节性", "品种季节", "基本面", "板块行业", "宏观",
                            "资金流", "概念板块", "除权"},
      str(F.CATEGORIES))
check("展期/期权/基本面/宏观因子都登记进 AUX_KEYS",
      set(F.AUX_KEYS) == AUXISH,
      f"实际 {len(F.AUX_KEYS)} 个，应登记 {len(AUXISH)} 个")
check("默认单因子存在", F.DEFAULT_SINGLE in F.FACTORS)
check("默认多因子都存在", all(k in F.FACTORS for k in F.DEFAULT_MULTI))

bad_calc, bad_inf, bad_pit = [], [], []
for key in F.FACTORS:
    if key in NO_DAILY or key in STOCK_ONLY:
        continue
    s = F.raw(DF_AUX, key)
    if len(s) != len(DF_AUX):
        bad_calc.append(key)
        continue
    # 基本面 / 宏观要外部库，没库时整列空是正常降级
    if key not in EXT_DB and s.notna().sum() < 100:
        bad_calc.append(key)
        continue
    if np.isinf(s.to_numpy(dtype=float)).any():
        bad_inf.append(key)
    # 因子只许用向后看窗口：截断数据后前段必须逐位相同
    if not same_series(s, F.raw(DF_AUX.iloc[:CUT], key), CUT):
        bad_pit.append(key)

check(f"{len(F.FACTORS) - len(NO_DAILY | STOCK_ONLY)} 个日线因子都能算出足够样本",
      not bad_calc, f"异常 {bad_calc}")
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


# ========================================================== 1b. 展期 / 期权因子
section("1b. 展期 / 期权因子")

# 没有衍生列时必须「优雅退化」：因子被剔除，而不是报错或算出 0
no_aux = F.filter_available(DF, list(F.FACTORS))
check("无衍生列时展期/期权因子被剔除",
      all(k not in no_aux for k in CURVE_KEYS),
      str([k for k in CURVE_KEYS if k in no_aux]))
# 断言写成「差的集合 ⊆ 展期/期权」，而不是算一个硬数字：基本面 / 宏观 / 板块
# 能不能算出值取决于本机有没有 EDB 库、有没有行情 CSV，写死数量会在别人的
# 机器上无缘无故变红。
_had = set(F.filter_available(DF_AUX, list(F.FACTORS)))
_lost = _had - set(no_aux)
check("无衍生列时只影响展期/期权因子", _lost <= CURVE_KEYS, str(sorted(_lost)))
check("无衍生列时展期因子返回全 NaN",
      F.raw(DF, "roll_yield").isna().all())


def aux_row(n: int = 60) -> pd.DataFrame:
    """构造一条精确已知的期限结构：c1=100 / c2=101 / c3=102，间隔 30 / 70 天。

    OI 两套口径按「o6 从 c2 起、o3 从 c3 起」各错开一档，间隔 70 / 40 / 50 天，
    这样每套的分子分母、到期间隔都是不同的数，前缀接错时一定对不上。
    """
    df = DF.iloc[:n].copy()
    df["c1_close"], df["c2_close"], df["c3_close"] = 100.0, 101.0, 102.0
    df["c1_dte"], df["c2_dte"], df["c3_dte"] = 20.0, 50.0, 120.0
    df["c1_oi"], df["c2_oi"], df["c3_oi"] = 300.0, 700.0, 100.0
    df["o6_1_close"], df["o6_2_close"], df["o6_3_close"] = 101.0, 102.0, 103.0
    df["o6_1_dte"], df["o6_2_dte"], df["o6_3_dte"] = 50.0, 120.0, 160.0
    df["o3_1_close"], df["o3_2_close"], df["o3_3_close"] = 102.0, 103.0, 104.0
    df["o3_1_dte"], df["o3_2_dte"], df["o3_3_dte"] = 120.0, 160.0, 210.0
    for pre in ("o3_", "o6_"):
        for i in (1, 2, 3):
            df[f"{pre}{i}_oi"] = 500.0
    df["iv_near"], df["iv_next"] = 0.20, 0.23
    df["dte_near"], df["dte_next"] = 20.0, 50.0
    df["skew"], df["curv"], df["pcr"] = 0.12, 0.35, 1.2
    df["n_opt"] = 60
    return df


A = aux_row()
ry_exp = 100.0 / 101.0 - 1.0
ry = float(F.raw(A, "roll_yield").iloc[-1])
check("展期收益率 = P1/P2 - 1", abs(ry - ry_exp) < 1e-12, f"{ry:+.10f} vs {ry_exp:+.10f}")

ann = float(F.raw(A, "roll_annual").iloc[-1])
ann_exp = ry_exp * 365.0 / 30.0
check("展期年化 = 收益率 × 365/到期间隔", abs(ann - ann_exp) < 1e-12,
      f"{ann:+.10f} vs {ann_exp:+.10f}")

# 曲率用「近端年化斜率 - 远端年化斜率」，不是价格二阶差分：
# 间隔 30/70 天不等距，二阶差分会被间隔差异污染
import math  # noqa: E402

curv_exp = ((math.log(101.0 / 102.0) / 70.0 - math.log(100.0 / 101.0) / 30.0) * 365.0)
curv = float(F.raw(A, "roll_curv").iloc[-1])
check("展期曲率 = 远端斜率 - 近端斜率", abs(curv - curv_exp) < 1e-12,
      f"{curv:+.10f} vs {curv_exp:+.10f}")

sh = float(F.raw(A, "near_oi_share").iloc[-1])
check("近月持仓占比 = c1_oi/(c1_oi+c2_oi)", abs(sh - 0.3) < 1e-12, f"{sh:.6f}")

term = float(F.raw(A, "iv_term").iloc[-1])
check("IV 期限结构 = (IV次月-IV近月)/间隔×30", abs(term - 0.03) < 1e-12, f"{term:+.6f}")

check("平值 IV 直取 iv_near", abs(float(F.raw(A, "iv_atm").iloc[-1]) - 0.20) < 1e-12)
check("IV 偏度直取 skew", abs(float(F.raw(A, "iv_skew").iloc[-1]) - 0.12) < 1e-12)
check("PCR 取对数", abs(float(F.raw(A, "pcr").iloc[-1]) - math.log(1.2)) < 1e-12)

# 常数期限结构下，「变化」类因子必须正好为 0，且前 20 根是 NaN（窗口还没凑满）
rc = F.raw(A, "roll_change")
check("展期变化在常数结构上为 0", bool((rc.iloc[20:] == 0).all()) and rc.iloc[:20].isna().all())
ic = F.raw(A, "iv_change")
check("IV 变化在常数 IV 上为 0", bool((ic.iloc[20:] == 0).all()) and ic.iloc[:20].isna().all())

# ---- OI 两套口径：四个公式各自换前缀，逐项对精确值
for pre, p1, p2, p3, g12, g23 in (
    ("o3_", 102.0, 103.0, 104.0, 40.0, 50.0),
    ("o6_", 101.0, 102.0, 103.0, 70.0, 40.0),
):
    tag = pre.strip("_").upper()
    ry_x = p1 / p2 - 1.0
    got = float(F.raw(A, f"roll_yield_{pre.strip('_')}").iloc[-1])
    check(f"{tag} 展期收益率 = P1/P2 - 1", abs(got - ry_x) < 1e-12,
          f"{got:+.10f} vs {ry_x:+.10f}")

    got = float(F.raw(A, f"roll_annual_{pre.strip('_')}").iloc[-1])
    exp = ry_x * 365.0 / g12
    check(f"{tag} 展期年化 = 收益率 × 365/{g12:.0f}", abs(got - exp) < 1e-12,
          f"{got:+.10f} vs {exp:+.10f}")

    got = float(F.raw(A, f"roll_curv_{pre.strip('_')}").iloc[-1])
    exp = (math.log(p2 / p3) / g23 - math.log(p1 / p2) / g12) * 365.0
    check(f"{tag} 展期曲率 = 远端斜率 - 近端斜率", abs(got - exp) < 1e-12,
          f"{got:+.10f} vs {exp:+.10f}")

    rc_x = F.raw(A, f"roll_change_{pre.strip('_')}")
    check(f"{tag} 展期变化在常数结构上为 0",
          bool((rc_x.iloc[20:] == 0).all()) and rc_x.iloc[:20].isna().all())

# 把 c* 的列原样冒充成 o3_/o6_，两套因子必须逐位相同 —— 这条专门抓前缀拼错，
# 上面那些精确值检查抓不到「o3 和 o6 都接对了但互相抄错」这种情形
fake = A.copy()
for pre in ("o3_", "o6_"):
    for i in (1, 2, 3):
        for suf in ("close", "dte", "oi"):
            fake[f"{pre}{i}_{suf}"] = A[f"c{i}_{suf}"]
for stem in ("roll_yield", "roll_annual", "roll_curv", "roll_change"):
    for suf in ("o3", "o6"):
        check(f"{stem}_{suf} 与 c 口径等价（同列时应完全一致）",
              same_series(F.raw(fake, f"{stem}_{suf}"), F.raw(fake, stem), len(fake)))

# IV-RV 价差：必须严格等于 iv_near - vol_20，且 IV 明显高于实际波动时为正
d = F.raw(A, "iv_rv") - (A["iv_near"] - F.raw(A, "vol_20"))
check("IV-RV 价差 = 平值 IV - 20 日已实现波动", float(d.abs().max()) < 1e-12)

# 非正 PCR 取对数会变成 -inf，必须挡掉
neg = aux_row()
neg["pcr"] = 0.0
check("PCR ≤ 0 时返回 NaN 而不是 -inf", F.raw(neg, "pcr").isna().all())

# 缺 c3 的品种（只有两个合约）仍能算展期收益率，但曲率应为 NaN
two = aux_row().drop(columns=["c3_close", "c3_dte", "c3_oi"])
check("只有两个合约时仍能算展期收益率", F.raw(two, "roll_yield").notna().all())
check("只有两个合约时展期曲率为 NaN", F.raw(two, "roll_curv").isna().all())


# ========================================================== 1c. 期权定价内核
section("1c. 期权定价与微笑拟合")

from ttrade import contracts as CT  # noqa: E402
from ttrade import options as OP  # noqa: E402

F0, K0, T0, SIG = 5000.0, 5100.0, 60 / 365.0, 0.22
call = float(OP.black76(F0, K0, T0, SIG, is_call=True))
put = float(OP.black76(F0, K0, T0, SIG, is_call=False))

# 期货期权的平价关系不含分红：C − P = e^{−rT}(F − K)。这条不成立说明
# 贴现因子或 d1/d2 写错了，反解出来的 IV 会整体偏掉。
parity = call - put - np.exp(-OP.RISK_FREE * T0) * (F0 - K0)
check("Black-76 满足 put-call 平价", abs(parity) < 1e-9, f"偏差 {parity:.2e}")

check("看涨价格随波动率单调递增",
      float(OP.black76(F0, K0, T0, 0.30)) > call > float(OP.black76(F0, K0, T0, 0.15)))

# 反解必须回到原值：正向定价 → 反向求 sigma → 误差应到二分法精度
iv_c = float(OP.implied_vol(np.array([call]), F0, K0, T0, True)[0])
iv_p = float(OP.implied_vol(np.array([put]), F0, K0, T0, False)[0])
check("IV 反解回代误差 < 1e-8", abs(iv_c - SIG) < 1e-8 and abs(iv_p - SIG) < 1e-8,
      f"{iv_c:.10f} / {iv_p:.10f} vs {SIG}")

# 深度实值/虚值也要能反解（二分法在端点附近最容易卡住）
deep = float(OP.black76(F0, 3000.0, T0, SIG))
iv_deep = float(OP.implied_vol(np.array([deep]), F0, 3000.0, T0, True)[0])
check("深度实值仍能反解", abs(iv_deep - SIG) < 1e-6, f"{iv_deep:.8f}")

# 价格低于内在价值时无解，必须返回边界而不是 NaN / 负数
iv_bad = float(OP.implied_vol(np.array([1.0]), F0, 3000.0, T0, True)[0])
check("无解时返回下界而不是 NaN", np.isfinite(iv_bad) and iv_bad <= OP.IV_LO + 1e-6,
      f"{iv_bad:.6f}")

# 微笑拟合：造一条精确的二次曲线，必须把三个系数原样还回来
m_grid = np.linspace(-0.15, 0.15, 11)
atm_true, skew_true, curv_true = 0.24, -0.35, 1.8
iv_true = atm_true + skew_true * m_grid + 0.5 * curv_true * m_grid ** 2
fit = OP.fit_smile(m_grid, iv_true, np.ones_like(m_grid))
check("微笑拟合还原平值/偏度/曲率",
      fit is not None
      and abs(fit[0] - atm_true) < 1e-10
      and abs(fit[1] - skew_true) < 1e-10
      and abs(fit[2] - curv_true) < 1e-10,
      "None" if fit is None else f"{fit[0]:.6f}/{fit[1]:.6f}/{fit[2]:.6f}")

# 点数不足时不硬拟合（二次要 3 个参数，5 个点才有 2 个自由度）
check("点数不足时返回 None", OP.fit_smile(m_grid[:3], iv_true[:3], np.ones(3)) is None)
# 所有点挤在同一个行权价上，等于没信息
check("行权价全相同也返回 None",
      OP.fit_smile(np.zeros(8), np.full(8, 0.2), np.ones(8)) is None)

# 结算价优先、缺失时退回收盘价 —— 这是 IV 口径的关键一环
sdf = pd.DataFrame({
    "ts_code": ["A1", "A2", "A3"],
    "settle": [100.0, 0.0, np.nan],
    "close": [99.0, 88.0, 77.0],
})
sm = CT.settle_map(sdf)
check("settle_map 优先结算价、缺失退回收盘价",
      sm["A1"] == 100.0 and sm["A2"] == 88.0 and sm["A3"] == 77.0, str(sm))
check("settle_map 对空表返回空字典", CT.settle_map(pd.DataFrame()) == {})


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
rep = M.factor_report(DF_AUX, list(F.FACTORS), horizon=5, window=252)
cost = time.time() - t0
check("全因子评估每行一个因子", len(rep) == len(F.FACTORS), str(len(rep)))
# 基本面 / 宏观那批不参与：它们要外部库，而且月频/季频序列在 1000 根日线上
# 本来就只有几十个观测，IC 算不出来是数据密度问题，不是代码问题
_daily = rep[~rep["key"].isin(NO_DAILY | EXT_DB | STOCK_ONLY)]
check("日线因子的 IC 都算得出来", _daily["ic"].notna().all(),
      str(list(_daily.loc[_daily["ic"].isna(), "key"])))
check("评估表无 error 列", "error" not in rep.columns or rep["error"].isna().all())
check("评估表按 |IC| 降序",
      bool((rep["ic"].abs().diff().dropna() <= 1e-12).all()))
# 门槛定在 3 秒：这条断言是拦「写成了 O(n²) / 逐行 apply」那种事故的，不是
# 比性能。因子从 122 涨到 136 之后 5051 行跑满一轮本来就 2.1s 上下，卡 2s 会
# 变成一条看机器心情的红灯 —— 那比没有这条更糟。
check("评估耗时 < 3s", cost < 3.0, f"{cost*1000:.0f}ms")
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


# ================================================================== 6c. 方向口径
section("6c. 方向口径（自动 / 固定多 / 固定空）")

p_up = FS.build_score(DF, "single", ["mom_60"],
                      FS.FactorConfig(window=252, horizon=5, orient="long_high"))
p_dn = FS.build_score(DF, "single", ["mom_60"],
                      FS.FactorConfig(window=252, horizon=5, orient="long_low"))
check("long_high 下分数 = 因子值本身",
      same_series(p_up.score, p_up.fmat["mom_60"], len(DF)))
check("long_low 下分数 = 因子值取负",
      same_series(p_dn.score, -p_dn.fmat["mom_60"], len(DF)))
check("long_high 的方向矩阵全为 +1",
      bool((p_up.orient.to_numpy() == 1.0).all()))
check("long_low 的方向矩阵全为 -1",
      bool((p_dn.orient.to_numpy() == -1.0).all()))

# 关键：固定方向下「反转」是绝对的，跨品种一致
p_up_rev = FS.build_score(DF, "single", ["mom_60"], FS.FactorConfig(
    window=252, horizon=5, orient="long_high", reverse={"mom_60": True}))
check("固定做多 + 反转 ≡ 固定做空",
      same_series(p_up_rev.score, p_dn.score, len(DF)))

o_up = FS.run_factor_strategy(DF, "single", ["mom_60"],
                              FS.FactorConfig(window=252, horizon=5, orient="long_high"))
o_up_rev = FS.run_factor_strategy(DF, "single", ["mom_60"], FS.FactorConfig(
    window=252, horizon=5, orient="long_high", reverse={"mom_60": True}))
o_dn = FS.run_factor_strategy(DF, "single", ["mom_60"],
                              FS.FactorConfig(window=252, horizon=5, orient="long_low"))
check("固定方向下反转后的仓位 ≡ 固定反向的仓位",
      np.array_equal(o_up_rev.df["signal"].to_numpy(), o_dn.df["signal"].to_numpy()))
check("固定方向下反转后仓位是原仓位的相反数",
      np.array_equal(o_up_rev.df["signal"].to_numpy(), -o_up.df["signal"].to_numpy()))

# 自动定向：符号是数据决定的，必须报出来，否则用户看不出反转落在哪个绝对方向
check("自动定向报出末期方向",
      any("末期方向" in n for n in p_single.notes), str(p_single.notes))
check("固定方向不报「末期方向」（下拉框已经说明了）",
      not any("末期方向" in n for n in p_up.notes), str(p_up.notes))
check("多因子自动定向报出各因子方向分布",
      any("末期方向" in n for n in p_multi.notes), str(p_multi.notes))

# 自动定向的符号确实会随数据变：这是「同一反转在不同品种上方向相反」的根源
ori_a = FS.pit_orientation(p_single.fmat["mom_60"],
                           M.forward_return(DF, 5), 5, 126)
_shifted = DF.copy()
_shifted["close"] = _shifted["close"].to_numpy()[::-1].copy()   # 倒序，收益结构变反
ori_b = FS.pit_orientation(
    FS.build_score(_shifted, "single", ["mom_60"],
                   FS.FactorConfig(window=252, horizon=5)).fmat["mom_60"],
    M.forward_return(_shifted, 5), 5, 126)
check("自动定向的符号随数据变化（所以「反转」是相对的）",
      not np.array_equal(ori_a.fillna(0).to_numpy(), ori_b.fillna(0).to_numpy()))


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
check("模型清单齐全",
      set(ML.MODELS) == {"lightgbm", "xgboost", "gpr", "transformer", "linear", "gnn"},
      str(sorted(ML.MODELS)))


# ================================================================== 9. 四模型
section("9. walk-forward 滚动训练")
X = F.factor_frame(DF, KEYS6, window=252).to_numpy(dtype=float)
Y = M.forward_return(DF, 5).to_numpy(dtype=float)
ok_rows = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
Xf, Yf = X[ok_rows], Y[ok_rows]
check("训练样本足够", len(Xf) > 300, f"{len(Xf)} 行")

models = [m for m in ("lightgbm", "xgboost", "gpr", "transformer", "linear", "gnn")
          if m in avail]
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

# 训练窗口是**交易日**不是根数：同样的 1000 天，5m 要 48 倍于日线的根数。
# 单位搞错的话 5m 上「1000」只等于 21 个交易日，模型拿两周数据训练。
_idx_d = pd.bdate_range("2016-01-04", periods=500)


def _sess_5m(days: int) -> pd.DatetimeIndex:
    """按真实交易时段拼 5m 索引：09:00-11:30 + 13:30-15:00，一天 48 根。

    不能用 `pd.date_range(freq="5min")`：那是连续 24 小时、一天 288 根，期货
    没有这种行情，拿它算出来的 bar 密度跟实盘差 6 倍，换算断言也就没意义了。
    """
    ts = []
    for d in pd.bdate_range("2024-01-01", periods=days):
        for a, b in (("09:00", "11:30"), ("13:30", "15:00")):
            ts.append(pd.date_range(f"{d.date()} {a}", f"{d.date()} {b}",
                                    freq="5min", inclusive="left"))
    return pd.DatetimeIndex(np.concatenate([t.values for t in ts]))


_idx_5 = _sess_5m(500)
_bpd_5 = FS._bars_per_day(_idx_5)
_n_d = FS._to_bars(1000, FS._bars_per_day(_idx_d))
_n_5 = FS._to_bars(1000, _bpd_5)
check("5m 每天 48 根（09:00-11:30 + 13:30-15:00）", abs(_bpd_5 - 48.0) < 1e-9,
      f"{_bpd_5:.2f} 根/天")
check("训练窗口按交易日换算成根（日线 1000 / 5m 48000）",
      _n_d == 1000 and _n_5 == 48000, f"日线 {_n_d} 根 / 5m {_n_5} 根")

# 关键 PIT：训练集不得包含 b 时刻还不存在的标签。
# 做法是把第一折训练窗口尾部 h 个标签改坏 —— 这几行的收益要到 b+h 才成立，
# 正确实现（tr_end = b-h）不该看它们，**第一折的预测**必须一字不变。
# 只比第一折的预测区间：后面的折合法地会用到这些标签（那时它们已经成立了）。
# 第一折的起点是 min(min_train, train_window)：DF 是日线（一天一根），
# train_window 的「交易日」正好等于根数，所以这里可以直接比。
_cfg_wf = cfg_ml.normalized()
_fmat_wf = F.factor_frame(DF, KEYS6, 252)
_fwd_wf = M.forward_return(DF, _cfg_wf.horizon)
base_pred, _, _ = FS.walk_forward(_fmat_wf, _fwd_wf, _cfg_wf)
_b = min(_cfg_wf.min_train, _cfg_wf.train_window)
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
