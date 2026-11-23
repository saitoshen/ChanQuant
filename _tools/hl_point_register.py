"""把 future_quant 的 HL point 截面对冲策略注册成一条 ChanQuant 策略库条目。

## 为什么是一个脚本而不是「手点一遍存盘」

复现涉及 60 多个因子键 + 8 个截面参数，手点一次不可复现、也不可审查。
脚本把「原策略的哪个参数」映射到「ChanQuant 的哪个字段」逐条写在下面，
以后改了哪一项、为什么改，`git log` 里查得到。

## 原策略的档位（`D:/CozeTask/future_quant/cross_section/`）

| 环节 | 原实现 | 来源 |
|---|---|---|
| 周期 | 日线 | `strategy_cross.py` |
| 品种池 | 全部商品品种，**排除** IF/IH/IC/IM/T/TF/TS/TL | `data_prep.get_variety_list` |
| 第一步筛池 | 每月选**成交额前 70%** | `backtest_cross.AMOUNT_TOP_PCT=0.70` |
| 特征 | 59 个量价 + 价差结构 + 持仓量 + 15 个 EDB 基本面 | `factor_panel.py` |
| 模型 | LightGBM，滚动训练 12 个月 / 预测 1 个月 / 步长 1 个月 | `train_cross.TRAIN_MONTHS…` |
| 标签 | 未来 5 个交易日收益 > 1% 的二分类 | `LABEL_DAYS=5, THRESHOLD=0.01` |
| 组合 | 分数**前一半做多、后一半做空** | `backtest_cross` |
| 加权 | 波动率倒数，**多空各半仓**（组合日收益 `Σ(w·r)/2`） | `backtest_cross` |
| 调仓 | 每 5 个交易日，14:55 | `strategy_cross.REBALANCE_DAYS=5` |
| 本金 | 500 万，保证金占用 80% | `strategy_cross.initial_capital` |

## 与原策略的**已知差异**（都不影响方向，但解释曲线差异时要用到）

1. **标签口径**：原策略是「收益 > 1% 的二分类概率」，ChanQuant 的截面 ML 是
   「预测收益（回归/排序）」。二分类概率不是期望收益的单调变换（同样的 μ，
   高波动品种 `P(r>1%)` 更高），所以严格说两者选出来的股票不完全一样。
   这里用 `label_mode="excess"`（预测收益减截面均值）：它是「谁涨得多」的
   线性对应，比 `volret` 更接近原意。
2. **标准化**：原策略每天做**截面 z-score**；ChanQuant 是**时序滚动 z**（窗口
   按交易日）之后再横比。这是两套不同的归一化，时序口径对「这只品种相对自己
   历史」更敏感，截面口径对「今天相对别人」更敏感。
3. **持仓排名字段**（`net_pos_ratio` / `long_oi_share` / 短期变化）不参与：
   ChanQuant 没有天勤龙虎榜数据源，注册了也只会是整列 NaN。
4. **`log_price` 不参与**：纯价位，横截面上比的是「谁贵」，没有经济含义。

用法：
    python _tools/hl_point_register.py            # 注册 / 覆盖
    python _tools/hl_point_register.py --dry-run  # 只打印配置，不落库
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttrade import factors as F                     # noqa: E402
from ttrade import strategy_store as SStore         # noqa: E402

NAME = "HL point · 商品截面对冲（日线）"

# 金融期货不进商品截面：股指和国债的驱动是另一套（贴水、资金面），
# 混进商品池会让「谁强谁弱」变成「商品 vs 股指」
FINANCIAL = {"IF", "IH", "IC", "IM", "T", "TF", "TS", "TL"}

# ------------------------------------------------------------------ 因子映射
# 左边是原策略（Rust `compute_single_stock_features` 的语义名），右边是
# ChanQuant 的因子键。None 表示「原策略有、这边不参与」，理由写在注释里。
#
# 只在右边出现的键 = 一个 ChanQuant 因子覆盖了原策略的多个列（原策略里
# 那几列本来就是同一个公式的不同写法，见各自注释）。
FEATURE_MAP = [
    # ---- Group A 基础量价 ----
    ("mom_5d", "mom_5"),
    ("mom_10d", "mom_10"),
    ("mom_20d", "mom_20"),
    ("mom_60d", "mom_60"),
    ("vol_5d", "vol_5"),
    ("vol_20d", "vol_20"),
    ("vol_ratio_5d", "vol_surge"),          # 都是 量/均量
    ("macd", "macd_dif"),
    ("macd_signal", "macd_dea"),
    ("macd_hist", "macd_hist"),
    ("rsi", "rsi_14"),
    ("bb_middle", "ma_gap_20"),             # 中轨=MA20，绝对价位无意义，改偏离度
    ("atr", "atr_pct"),                     # 绝对 ATR 跨品种不可比，用占比
    ("tg_macd_hist", "macd_hist_lag1"),
    # bb_upper / bb_lower / tg_bb_upper / tg_bb_lower：绝对价位，横截面不可比。
    # 布林带的信息已经被 bb_pos_20（相对位置）和 bb_width_20（带宽）覆盖。
    ("bb_upper", "bb_pos_20"),
    ("bb_lower", "bb_pos_20"),
    ("log_price", None),                    # 纯价位，比的是「谁贵」
    ("mom_20d_alt", "mom_20"),              # 原策略里就是 mom_20 的重复列
    # ---- Group C MACD 风洞 ----
    ("funnel_daily", "macd_funnel_d"),
    ("hist_chg", "macd_hist_chg"),
    ("hist_acc", "macd_hist_acc"),
    ("funnel_weekly", "macd_funnel_w"),
    ("funnel_monthly", "macd_funnel_m"),
    ("resonance", "macd_resonance"),
    ("cross_up_d", "macd_cross_up"),
    ("cross_dn_d", "macd_cross_dn"),
    ("at_trough", "macd_at_trough"),
    ("at_peak", "macd_at_peak"),
    # ---- Group D 几何结构 ----
    ("slope_20_norm", "slope_20"),
    ("slope_60_norm", "slope_60"),
    ("dev_20", "bb_pos_20"),                # 只差常数 2，z 之后完全等价
    ("dev_60", "dev_60"),
    ("bb_width_20", "bb_width_20"),
    ("bb_width_60", "bb_width_60"),
    ("ma_align", "ma_align"),
    ("pos_20", "pos_20"),
    ("pos_60", "pos_60"),
    ("new_high_10", "new_high_10"),
    # ---- Group E 缩量短线 ----
    ("vol_ratio_20", "vol_surge"),          # 与 vol_ratio_5d 同一条
    ("vol_shrink", "vol_shrink"),
    ("atr_ratio", "atr_ratio"),
    ("range_ratio", "range_ratio"),
    ("vol_price_diverge", "vol_price_diverge"),
    ("vol_surge_5", "vol_surge_5"),
    ("contraction_score", "contraction_score"),
    # ---- Group F 动量 / 反转 / 成交额 ----
    ("momentum_accel", "mom_accel"),
    ("momentum_divergence", "mom_accel"),   # 与原式恒等，同一列
    ("price_vs_sma20", "ma_gap_20"),
    ("price_vs_sma60", "ma_gap_60"),
    ("high_vs_sma20", "high_ma_gap_20"),
    ("reversal_1d", "mom_1"),               # 只差符号，方向由滚动 IC 定
    ("turnover_est_ma5", "log_amount_20"),  # 成交额，取对数后跨品种可比
    # ---- Group H 日内（日线口径）----
    ("intraday_strength", "intraday_strength"),
    ("close_vs_open_ratio", "mom_1"),       # 日线上与 mom_1 恒等
    ("close_position", "close_pos"),
    ("gap_continuation", "gap"),
    ("gap_fill_ratio", "gap_fill_ratio"),
    # ---- 价差结构 ----
    ("term_spread_near_far", "term_near_far"),
    ("term_spread_main_far", "roll_main_oi"),
    ("roll_yield", "roll_yield"),
    # ---- 持仓量 ----
    ("oi_chg_1d", "oi_change_1"),
    ("oi_chg_5d", "oi_change_5"),
    ("oi_chg_10d", "oi_change_10"),
    ("oi_vol_ratio", "oi_vol_ratio"),
    ("oi_price_corr_5d", "oi_price_corr_5"),
    # ---- EDB 基本面 ----
    ("basis_rate", "spot_basis"),
    ("spot_mom_5", "spot_mom_5"),
    ("spot_mom_20", "spot_mom_20"),
    ("warehouse_rank", "war_pct"),
    ("warehouse_chg_5", "war_chg_5"),
    ("warehouse_chg_20", "warehouse_chg_20"),
    ("inventory_rank", "inv_pct"),
    ("inventory_chg_1w", "inv_chg_5"),
    ("inventory_chg_10w", None),            # 十周变化覆盖品种太少，并入同比
    ("inventory_yoy", "inventory_yoy"),
    ("delivery_rank", "deliv_pct"),
    ("delivery_chg_mom", "deliv_chg_30"),
    ("profit_rank", "profit_pct"),
    ("profit_chg_4w", "profit_chg_5"),
    # ---- 不参与 ----
    # 持仓排名 5 个（net_pos_ratio / long_oi_share / short_oi_share /
    # long_chg_ratio / short_chg_ratio）：要天勤龙虎榜，ChanQuant 没这个数据源
]


def feature_keys() -> list[str]:
    out: list[str] = []
    for _src, dst in FEATURE_MAP:
        if dst and dst in F.FACTORS and dst not in out:
            out.append(dst)
    return out


def commodity_symbols() -> list[str]:
    """本机有日线数据的商品品种键（`RB_SHFE` 这种）。

    直接扫 `data_futures/*.csv` 而不是走 GUI 的 `DataStore`：那个类定义在
    `gui/app.py` 里，为了拿一个品种清单去 import Qt 不值得，脚本也跑不了
    无图形环境。文件名就是键（`paths.data_dir("cn")` 是唯一真源）。
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


def build():
    keys = feature_keys()
    syms = commodity_symbols()
    cfg = SStore.StrategyConfig(
        name=NAME,
        family="cross",
        period="1d",
        symbol="RB",
        exchange="SHFE",
        multi_symbols=list(syms),
        cross_symbols=list(syms),

        # ---- 组合构造 ----
        cross_mode="hedge",
        cross_rebalance=5,
        cross_rebalance_unit="day",
        cross_top_pct=50.0,             # 前一半做多
        cross_bot_pct=50.0,             # 后一半做空
        cross_weight="risk",            # 波动率倒数
        # 🔴 原策略是「多空各半仓」（组合收益 Σ(w·r)/2），净敞口恒 0。
        # ChanQuant 2026-11-23 之后默认是「多空合并分配」，净敞口不钉 0 ——
        # 复现必须显式切成 legs，否则两边波动不一样时组合会自己偏多头。
        cross_weight_scope="legs",
        cross_tiers=0,

        # ---- 第一步：筛池（成交额前 70%，每月重排）----
        cross_pool_mode="single",
        cross_pool_keys=["log_amount_20"],
        cross_pool_pct=70.0,
        cross_pool_reb=1,
        cross_pool_reb_unit="month",

        # ---- 第二步：截面 ML ----
        factor_mode="ml",
        factor_keys=keys,
        factor_reverse={},
        factor_cfg={
            # 原策略 LOOKBACK=90：喂给因子的历史窗口 90 天
            "window": 90,
            "window_unit": "day",
            "method": "z",
            "horizon": 5,               # LABEL_DAYS=5
            "model": "lightgbm",        # 原策略就是 LightGBM
            "train_window": 252,        # TRAIN_MONTHS=12
            "step": 21,                 # STEP_MONTHS=1
            "min_train": 500,
            "sample_mode": "all",
            "orient": "auto",
        },
        cross_label_mode="excess",

        # ---- 执行 ----
        sizing="capital",
        principal=5_000_000.0,          # initial_capital=500 万
        capital_pct=80.0,               # margin_usage_ratio=0.80
        stop_atr_n=14,
        stop_mult=5.0,
        volume=1,
        short_only=False,
        note="short",                   # 显式允许做空（截面多空两条腿）
        cost_profile="",
        range_years=0.0,
        enabled=False,
    )
    return cfg, syms, keys


def main() -> int:
    dry = "--dry-run" in sys.argv
    cfg, syms, keys = build()
    print(f"品种池 {len(syms)} 个：{'、'.join(syms[:10])}"
          f"{' …' if len(syms) > 10 else ''}")
    print(f"因子 {len(keys)} 个：{'、'.join(keys)}")
    print(f"摘要：{cfg.strategy_label()}")
    if dry:
        return 0
    out = SStore.add_to_library(cfg, overwrite=True)
    print(f"已写入策略库：{out.title()}（sid={out.sid}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
