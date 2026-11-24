"""把 future_quant 的 **HL-Point**（1h / 36 因子 / 十分位多空）注册进 ChanQuant 策略库。

## 先说清楚是哪一套

``future_quant`` 下面有**两套**都叫「HL point」的策略，别混：

| | ``strategy/hl_point/`` ← **本脚本** | ``cross_section/`` |
|---|---|---|
| 周期 | **1h**（``HL_ML_TF=1h``，每日约 6 根） | 日线 |
| 因子 | **36 个**（价格 27 + 持仓量 5 + 展期 4） | 59 个量价 + 15 个 EDB |
| 标签 | 未来 5 天收益（``H = 5×BPD = 30`` 根），回归 | 未来 5 日涨跌 >1%，二分类 |
| 组合 | **十分位**：预测最高 10% 做多、最低 10% 做空，**等权** | 前一半多、后一半空，波动率倒数 |
| 移植件 | ``ttrade/factors_hlpoint.py``（33 新 + 3 映射） | ``ttrade/factors_hl.py``（42 个） |

## 参数逐条映射

| 环节 | 原实现 | ChanQuant 字段 |
|---|---|---|
| 周期 | ``HL_ML_TF=1h`` | ``period="1h"`` |
| 品种池 | DB 内全部 888 商品，排除金融 | ``cross_symbols`` + ``pool_mode="off"`` |
| 持有期 | ``H = H_DAYS×BPD = 5×6``（写死 6） | ``horizon = 5×9.28 ≈ 46``（实测密度，见下） |
| 调仓 | **非重叠**，每 H bar 一次 | ``cross_rebalance=46`` + ``unit="bar"`` |
| 分档 | ``k = len(pr)//10``，top/bot 各 k 个 | ``cross_top_pct=10`` / ``cross_bot_pct=10`` |
| 加权 | ``top.mean() − bot.mean()``（**等权**） | ``cross_weight="equal"`` |
| 资金范围 | 组合日收益 ``Σ(w·r)/2`` ⇒ 净敞口恒 0 | 🔴``cross_weight_scope="legs"`` |
| 杠杆 | 裸 100% 名义额，**无杠杆** | 🔴``sizing="unit"``（见「已知差异 4」） |
| 标准化 | 因子内部已滚动 z（250 根） | ``window=250``、``window_unit="bar"``、``method="z"``（≈ 恒等） |
| 模型 | LightGBM 回归，lr 0.03 / leaves 31 / 300 轮 | ``model="lightgbm"`` |
| 训练 | 逐年滚、**扩张窗口**（2021 起） | ``train_window=1260`` 天（≈7560 根，实际等于扩张）+ ``step=252`` 天（年度重训） |
| 标签 | 未来 H bar **原始收益** | ``cross_label_mode="excess"`` |
| 成本 | ``COST=0.0005`` 单边（``decile_ls`` 里 ``−2×cost``） | 走 ``cost_profile`` 的默认跳数模型 |

## 五处**必须知道**的已知差异

1. **标签 ``excess`` 不是 ``ret``**：ChanQuant 没有「原始收益」这一档。但
   ``excess`` 是「收益 − 当期截面均值」，在**同一个时间戳内**减一个常数不改变
   排序 ⇒ 十分位选哪几个品种完全一样，只是模型拟合的是去均值后的收益
   （等价于加了日期固定效应）。对「选 top/bot 分位」这件事是等价的。
2. **展期 3 个因子是映射、不是新建**：``roll``→``roll_main_oi``、
   ``roll_ann``→``roll_main_annual``、``curv``→``roll_main_curv``。库里这三个
   多一道 5 日均值（原项目用未平滑的日快照），换腿当天有差异，其余时候等价。
   ``roll_chg_20`` 差分的是**价差率**本身、库里 ``roll_main_change`` 差分的是
   **年化值**，不是常倍数关系 ⇒ 那个必须独立注册（已建）。
3. **``ext_hi`` / ``ext_lo`` 默认不勾选**。它们是「250 根内最高/最低价」，是
   **绝对价格水平量**，和 ``tools_ml.py`` 里「只用已归一化特征」的注释矛盾。
   截面 z-score 之后会退化成「本品种在池中的价格排名」，等于塞一个品种哑变量。
   要 36 个全开的忠实版本加 ``--with-price-level``。
4. **绝对收益只有原策略的一半**。ChanQuant 的 ``legs`` 口径让多空**各占 50%**
   （gross 100%），原策略是 ``1.0×top_ret − 1.0×bot_ret``（gross 200%）⇒
   组合收益、成本都减半，**夏普 / IR 不变**。跨平台比绩效请锚夏普。
5. **持有期是 46 根不是 30 根**。原项目 ``BPD = 6`` 是拍出来的「每天约 6 根」，
   本机 1h 面板实测 **9.28 根/天**（夜盘比原项目设想的更长：AU/AG/SC 到
   02:30、CU/AL/ZN 到 01:00）。``H_DAYS=5`` 是设计意图、``BPD=6`` 只是它的
   近似实现，所以默认按实测密度换算成 46 根。要逐字复现源码常量传
   ``--bpd 6``。

用法：
    python _tools/hl_point_register.py                  # 注册 / 覆盖
    python _tools/hl_point_register.py --dry-run        # 只打印，不落库
    python _tools/hl_point_register.py --with-price-level   # 连 ext_hi/ext_lo 一起开
    python _tools/hl_point_register.py --drop-stale     # 顺手删掉上一版日线那条
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttrade import factors as F                     # noqa: E402
from ttrade import strategy_store as SStore         # noqa: E402

NAME = "HL-Point · 商品十分位多空（1h）"

#: 上一版（对象搞错了的那条，日频 59 因子）的名字，``--drop-stale`` 会删掉它
STALE_NAME = "HL point · 商品截面对冲（日线）"

#: 金融期货不进商品截面：股指/国债的驱动是另一套（贴水、资金面），
#: 混进来「谁强谁弱」就变成了「商品 vs 股指」
FINANCIAL = {"IF", "IH", "IC", "IM", "T", "TF", "TS", "TL"}

#: 原项目写死的「每天几根 1h bar」（``BPD = 6 if TF != '1d' else 1``）。
BPD_SRC = 6
#: 本机 1h 面板实测密度（20 个品种 2024-11~2026-11，``cross.run`` 报的
#: ``bars_per_day``）。**比原项目的 6 高得多** —— 因为国内期货夜盘很长
#: （AU/AG/SC 到 02:30、CU/AL/ZN 到 01:00），一个交易日 6~10 小时不等；
#: 原项目那个 6 是拍出来的近似值。
#:
#: 于是「30 根」在原项目里被当成「5 个交易日」，在本机实际只有
#: ``30 / 9.28 ≈ 3.2`` 天。两条路只能选一条：
#:   · ``--bpd 6``   → H=30 根，**逐字复现源码常量**，但持有期变成 3.2 天；
#:   · 默认（9.28） → H=46 根，**复现「持有 5 个交易日」这个意图**。
#: 默认取后者：``H_DAYS=5`` 才是策略设计意图，``BPD=6`` 只是它的实现近似。
BPD_LOCAL = 9.28
H_DAYS = 5

# ------------------------------------------------------------------ 因子映射
# 左 = HL-Point 原名，右 = ChanQuant 因子键，None = 不参与。
# 价格 27 个全部 1:1（都是本轮 ``ttrade/factors_hlpoint.py`` 新注册的）；
# 持仓量 5 个 1:1；展期 4 个里 3 个映射到库里的主力口径。
FEATURE_MAP = [
    # ---- 价格 · 多均线偏离度（5）----
    ("dev_z_20", "dev_z_20"),
    ("dev_z_60", "dev_z_60"),
    ("dev_z_120", "dev_z_120"),
    ("dev_z_agg", "dev_z_agg"),
    ("dev_spread", "dev_spread"),
    # ---- 价格 · 影线（8）----
    ("up_sh", "up_sh"),
    ("dn_sh", "dn_sh"),
    ("up_sh_atr", "up_sh_atr"),
    ("dn_sh_atr", "dn_sh_atr"),
    ("up_sh_int", "up_sh_int"),
    ("dn_sh_int", "dn_sh_int"),
    ("shadow_high_z", "shadow_high_z"),
    ("shadow_low_z", "shadow_low_z"),
    # ---- 价格 · 趋势结构（7）+ 通道（1）----
    ("slope_atr", "slope_atr"),
    ("r2", "r2"),
    ("slope_chg_atr", "slope_chg_atr"),
    ("dist_hi", "dist_hi"),
    ("dist_lo", "dist_lo"),
    # ext_hi / ext_lo：绝对价格水平量，默认不参与（见模块文档「已知差异 3」）
    ("ext_hi", "ext_hi"),
    ("ext_lo", "ext_lo"),
    ("ch_slope_atr", "ch_slope_atr"),
    # ---- 价格 · 关键位（6）----
    ("hit_high", "hit_high"),
    ("hit_low", "hit_low"),
    ("clus_high", "clus_high"),
    ("clus_low", "clus_low"),
    ("level_high", "level_high"),
    ("level_low", "level_low"),
    # ---- 持仓量（5）----
    ("oi_chg_5", "oi_chg_5"),
    ("oi_chg_20", "oi_chg_20"),
    ("oi_z_60", "oi_z_60"),
    ("oi_ratio_60", "oi_ratio_60"),
    ("oi_px_sync_5", "oi_px_sync_5"),
    # ---- 展期（4）：前三个映射到库里的主力口径 ----
    ("roll", "roll_main_oi"),
    ("roll_ann", "roll_main_annual"),
    ("curv", "roll_main_curv"),
    ("roll_chg_20", "roll_chg_20"),
]

#: 默认排除的因子（``--with-price-level`` 才开）
PRICE_LEVEL = {"ext_hi", "ext_lo"}


def feature_keys(with_price_level: bool = False) -> list:
    """36 个原名的因子 → ChanQuant 因子键（去重、剔除不参与的）。"""
    out: list = []
    for src, dst in FEATURE_MAP:
        if not dst:
            continue
        if src in PRICE_LEVEL and not with_price_level:
            continue
        if dst not in F.FACTORS:
            raise KeyError(f"因子键 {dst!r}（来自 {src!r}）不在因子库里 —— 先确认 "
                           f"ttrade/factors_hlpoint.py 注册成功")
        if dst not in out:
            out.append(dst)
    return out


def commodity_symbols() -> list:
    """本机有日线 CSV 的商品品种键（``RB_SHFE`` 这种），排除金融期货。

    直接扫 ``paths.data_dir("cn")/*.csv``：文件名就是键，不走 GUI 的
    ``DataStore``（那个类在 ``gui/app.py`` 里，为了一个清单去 import Qt 不值）。
    """
    import glob

    from ttrade import paths

    keys = []
    for f in glob.glob(os.path.join(paths.data_dir("cn"), "*.csv")):
        key = os.path.basename(f)[:-4]
        sym = key.rpartition("_")[0].upper()
        if sym and sym not in FINANCIAL:
            keys.append(key)
    return sorted(keys)


def build(with_price_level: bool = False, bpd: float = BPD_LOCAL):
    keys = feature_keys(with_price_level)
    syms = commodity_symbols()
    H = max(1, int(round(H_DAYS * float(bpd))))   # 持有期 = 5 个交易日折算成根
    cfg = SStore.StrategyConfig(
        name=NAME,
        family="cross",
        period="1h",
        symbol="RB",
        exchange="SHFE",
        multi_symbols=list(syms),
        cross_symbols=list(syms),

        # ---- 组合构造：十分位多空 ----
        cross_mode="hedge",
        # H = 5 天 × 6 根 = 30 根，非重叠调仓（原项目 `decile_ls` 每 H bar 一次）
        cross_rebalance=H,
        cross_rebalance_unit="bar",
        cross_top_pct=10.0,             # k = len(pr)//10
        cross_bot_pct=10.0,
        cross_weight="equal",           # top.mean() − bot.mean()：腿内等权
        # 🔴 原策略组合收益写的是 `Σ(w·r)/2`，净敞口恒 0。ChanQuant 3.9.12 之后
        # 默认 pooled（多空合并归一、净敞口不钉 0），复现必须显式切 legs。
        cross_weight_scope="legs",
        cross_tiers=0,

        # ---- 第一步：不筛池（原策略用 DB 内全部 888 商品）----
        cross_pool_mode="off",

        # ---- 第二步：36 因子 → LightGBM ----
        factor_mode="ml",
        factor_keys=keys,
        factor_reverse={},
        factor_cfg={
            # 因子本身已在 250 根上做过滚动 z，这一层 z 基本是恒等变换，
            # 但必须显式写成 250 **根**（默认单位是交易日，会变成 1500 根）
            "window": 250,
            "window_unit": "bar",
            "method": "z",
            "horizon": H,               # H 根 = 未来 5 个交易日
            "model": "lightgbm",
            # 原项目：扩张窗口、逐年重训。`train_window` 的单位是交易日，
            # 1260 天 ≈ 7560 根，长于任何一次回测 ⇒ 实际就是扩张窗口
            "train_window": 1260,
            "step": 252,                # 每年重训一次
            "min_train": 1260,          # 前 210 个交易日只累积不出信号
            "sample_mode": "all",
            # 方向：**分高做多**（用户 2026-11-24 明确要求）。
            # ⚠️ 如实说明：在截面的 `mode="ml"` 路径上，`orient` **根本不会被读**
            # —— `ttrade/cross.py` 全文没有 `orient` 这个词，`ml_score_panel`
            # 只用 `label_mode` 定标签、方向和因子交互全交给 LightGBM 自己学
            # （这正是用户要的：「因子有正有反，靠模型组织」）。写 `long_high`
            # 只是把这个意图固化在配置里，`auto` 与它在此模式下逐位相同。
            # 真正读 `orient` 的是 `factor_strategy` 的单因子 / 多因子加权求和，
            # 那两条路才需要显式定向。
            "orient": "long_high",
        },
        cross_label_mode="excess",

        # ---- 执行 ----
        # 🔴 `sizing` 必须是 `unit`（名义敞口 = 权益，lev=1）。
        # `capital` 会走 `cross.leverage_vector` 把每组权重再乘上
        # `资金分配比例 ÷ 保证金率`（80% ÷ 10% = 8 倍），实测 Σ|w| 从 1.0
        # 变成中位 6.81、单腿权重 −5.72 —— 净值两年 19 倍、夏普 9.3 全是假的。
        # 原策略是 `Σ(w·r)/2` 的裸 100% 名义额，没有任何杠杆。
        sizing="unit",
        principal=5_000_000.0,
        capital_pct=80.0,
        stop_atr_n=14,
        stop_mult=5.0,
        volume=1,
        short_only=False,
        note="short",                   # 截面多空两条腿，显式允许做空
        cost_profile="",
        range_years=0.0,
        enabled=False,
    )
    return cfg, syms, keys


def main() -> int:
    dry = "--dry-run" in sys.argv
    with_pl = "--with-price-level" in sys.argv
    drop_stale = "--drop-stale" in sys.argv
    bpd = BPD_LOCAL
    for i, a in enumerate(sys.argv):
        if a == "--bpd":
            bpd = float(sys.argv[i + 1])

    cfg, syms, keys = build(with_pl, bpd)
    H = cfg.cross_rebalance
    print(f"HL-Point 因子映射 {len([s for s in FEATURE_MAP if s[1]])} 条 → "
          f"实开 {len(keys)} 个键{'' if with_pl else '（ext_hi/ext_lo 未开）'}")
    print(f"品种池 {len(syms)} 个：{'、'.join(syms[:12])}"
          f"{' …' if len(syms) > 12 else ''}")
    print(f"周期 {cfg.period}，每 {H} 根调仓 = {H_DAYS} 个交易日 × {bpd:g} 根/天，"
          f"多空各 {cfg.cross_top_pct:.0f}%")
    if abs(bpd - BPD_SRC) > 1e-9:
        print(f"  （原项目写死 BPD={BPD_SRC} ⇒ H={H_DAYS * BPD_SRC} 根；"
              f"传 --bpd {BPD_SRC} 可逐字复现那个常量）")
    print(f"摘要：{cfg.strategy_label()}")
    if dry:
        print("\n[因子清单]")
        for i in range(0, len(keys), 6):
            print("   " + "、".join(keys[i:i + 6]))
        return 0

    out = SStore.add_to_library(cfg, overwrite=True)
    print(f"已写入策略库：{out.title()}（sid={out.sid}）")

    if drop_stale:
        stale = SStore.find_library(STALE_NAME)
        if stale is None:
            print(f"上一版「{STALE_NAME}」已不在库里，无需清理")
        else:
            ok = SStore.remove_from_library(stale.sid)
            print(f"已删除上一版「{STALE_NAME}」（sid={stale.sid}）: {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
