# HL point 截面对冲策略 · 因子盘点与复现说明

> 源项目：`D:/CozeTask/future_quant/cross_section/`（HL point，日频商品截面对冲）
> 复现落点：ChanQuant 3.9.14，`ttrade/factors_hl.py`（42 个新因子）+
> `_tools/hl_point_register.py`（策略库条目「HL point · 商品截面对冲（日线）」）

---

## 一、原策略在做什么

| 环节 | 原实现 | 出处 |
|---|---|---|
| 周期 | 日线 | `strategy_cross.py` |
| 品种池 | 全部商品品种，排除 IF/IH/IC/IM/T/TF/TS/TL | `data_prep.get_variety_list` |
| 第一步筛池 | 每月选**成交额前 70%** | `backtest_cross.AMOUNT_TOP_PCT = 0.70` |
| 特征 | 77 列 Rust 输出（实含信息 59 列）+ 价差结构 3 + 持仓量 5 + 持仓排名 5 + EDB 基本面 15 | `factor_panel.py` |
| 标准化 | 每日**截面 z-score** | Rust `compute_features_cs_rank` |
| 模型 | LightGBM，滚动 12 个月训练 / 1 个月预测 / 步长 1 个月 | `train_cross.TRAIN_MONTHS…` |
| 标签 | 未来 5 个交易日收益 > 1% 的二分类 | `LABEL_DAYS=5, THRESHOLD=0.01` |
| 组合 | 分数**前一半做多、后一半做空** | `backtest_cross` |
| 加权 | 波动率倒数，**多空各半仓**（日收益 `Σ(w·r)/2`） | `backtest_cross` |
| 调仓 | 每 5 个交易日，14:55 前 | `strategy_cross.REBALANCE_DAYS=5` |
| 本金 | 500 万，保证金占用 80% | `strategy_cross.initial_capital` |

---

## 二、因子盘点（原策略 77 列 → ChanQuant）

Rust 的 77 列分成 7 组。**Group B（第 20~37 列，18 个 Fama-French 财务因子）
在期货上恒为 0**（财务列全填 0，`has_financial` 为假，标准化时 std=0 被跳过），
对模型零贡献 —— 直接不移植。剩下 59 列逐条对照：

### Group A · 基础量价（20 列）

| # | HL point | 定义 | ChanQuant | 判定 |
|---|---|---|---|---|
| 0 | `mom_5d` | 5 日涨跌 | **`mom_5`（新增）** | 缺 → 已注册 |
| 1 | `mom_10d` | 10 日涨跌 | `mom_10` | ✅ 有 |
| 2 | `mom_20d` | 20 日涨跌 | `mom_20` | ✅ 有 |
| 3 | `mom_60d` | 60 日涨跌 | `mom_60` | ✅ 有 |
| 4 | `vol_5d` | 5 日收益标准差 | **`vol_5`（新增）** | 缺 → 已注册 |
| 5 | `vol_20d` | 20 日收益标准差 | `vol_20` | ✅ 有 |
| 6 | `vol_ratio_5d` | 当日量/5 日均量 | `vol_surge` | ✅ 等价 |
| 7 | `macd` | DIF | **`macd_dif`（新增）** | 缺 → 已注册 |
| 8 | `macd_signal` | DEA | **`macd_dea`（新增）** | 缺 → 已注册 |
| 9 | `macd_hist` | 柱值 | `macd_hist` | ✅ 有（库里已除收盘价） |
| 10 | `rsi` | RSI(14) | `rsi_14` | ✅ 有 |
| 11 | `bb_upper` | 布林上轨（绝对价） | `bb_pos_20` 代 | ⚠️ 绝对价位跨品种不可比 |
| 12 | `bb_middle` | SMA20（绝对价） | `ma_gap_20` | ✅ 等价（close/MA20−1） |
| 13 | `bb_lower` | 布林下轨（绝对价） | `bb_pos_20` 代 | ⚠️ 同上 |
| 14 | `atr` | ATR(14) 绝对值 | `atr_pct` | ✅ 有（占比版更可比） |
| 15 | `tg_macd_hist` | T−1 的柱值 | **`macd_hist_lag1`（新增）** | 缺 → 已注册 |
| 16 | `tg_bb_upper` | T−1 上轨 | — | ⚠️ 绝对价位，不参与 |
| 17 | `tg_bb_lower` | T−1 下轨 | — | ⚠️ 绝对价位，不参与 |
| 18 | `log_price` | log(收盘) | — | ⚠️ 纯价位，横截面比「谁贵」 |
| 19 | `mom_20d_alt` | 与 #2 完全重复 | `mom_20` | ✅ 重复列 |

### Group C · MACD 风洞共振（10 列）→ 全部**缺**，已整组注册

| # | HL point | ChanQuant | 说明 |
|---|---|---|---|
| 38 | `funnel_daily` | **`macd_funnel_d`** | 柱值在最近 5 根的相对位置，−1~+1 |
| 39 | `hist_chg` | **`macd_hist_chg`** | 柱值一阶差分 |
| 40 | `hist_acc` | **`macd_hist_acc`** | 柱值二阶差分 |
| 41 | `funnel_weekly` | **`macd_funnel_w`** | 21/55/9 参数，回看 8 根 |
| 42 | `funnel_monthly` | **`macd_funnel_m`** | 55/144/9 参数，回看 13 根 |
| 43 | `resonance` | **`macd_resonance`** | 0.4·日 + 0.4·周 + 0.2·月 |
| 44 | `cross_up_d` | **`macd_cross_up`** | 柱值上穿 0 |
| 45 | `cross_dn_d` | **`macd_cross_dn`** | 柱值下穿 0 |
| 46 | `at_trough` | **`macd_at_trough`** | 柱值是 10 根内最低 |
| 47 | `at_peak` | **`macd_at_peak`** | 柱值是 10 根内最高 |

> 「风洞」= `2·(h−min)/(max−min) − 1`，把柱值换成「在最近 n 根里的相对位置」。
> 它和 `macd_hist` 不是一回事：后者看柱子多长，前者看柱子在区间里的位置。

### Group D · 几何结构（10 列）

| # | HL point | ChanQuant | 判定 |
|---|---|---|---|
| 48 | `slope_20_norm` | **`slope_20`（新增）** | 缺 → 已注册 |
| 49 | `slope_60_norm` | **`slope_60`（新增）** | 缺 → 已注册 |
| 50 | `dev_20` = (C−MA20)/σ20 | `bb_pos_20` | ✅ 等价（差常数 2） |
| 51 | `dev_60` | **`dev_60`（新增）** | 缺 → 已注册 |
| 52 | `bb_width_20` | **`bb_width_20`（新增）** | 缺 → 已注册 |
| 53 | `bb_width_60` | **`bb_width_60`（新增）** | 缺 → 已注册 |
| 54 | `ma_align` | **`ma_align`（新增）** | 缺 → 已注册 |
| 55 | `pos_20` | `pos_20` | ✅ 有（定义完全一致） |
| 56 | `pos_60` | **`pos_60`（新增）** | 缺 → 已注册 |
| 57 | `new_high_10` | **`new_high_10`（新增）** | 缺 → 已注册 |

### Group E · 缩量短线（7 列）

| # | HL point | ChanQuant | 判定 |
|---|---|---|---|
| 58 | `vol_ratio_20` | `vol_surge` | ✅ 等价（量/20 日均量） |
| 59 | `vol_shrink` | **`vol_shrink`（新增）** | 缺 → 已注册 |
| 60 | `atr_ratio` | **`atr_ratio`（新增）** | 缺 → 已注册 |
| 61 | `range_ratio` | **`range_ratio`（新增）** | 缺 → 已注册 |
| 62 | `vol_price_diverge` | **`vol_price_diverge`（新增）** | 缺 → 已注册 |
| 63 | `vol_surge_5` | **`vol_surge_5`（新增）** | 缺 → 已注册 |
| 64 | `contraction_score` | **`contraction_score`（新增）** | 缺 → 已注册 |

### Group F · 动量/反转/成交额（7 列）

| # | HL point | ChanQuant | 判定 |
|---|---|---|---|
| 65 | `momentum_accel` | **`mom_accel`（新增）** | 缺 → 已注册 |
| 66 | `momentum_divergence` | 与 #65 **数学恒等** | ⚠️ 原项目重复列，不单独注册 |
| 67 | `price_vs_sma20` | `ma_gap_20` | ✅ 等价 |
| 68 | `price_vs_sma60` | `ma_gap_60` | ✅ 等价 |
| 69 | `high_vs_sma20` | **`high_ma_gap_20`（新增）** | 缺 → 已注册 |
| 70 | `reversal_1d` | `mom_1` | ✅ 只差符号（方向由滚动 IC 定） |
| 71 | `turnover_est_ma5` | `log_amount_20` | ✅ 等价（成交额对数） |

### Group H · 日线级日内（5 列）

| # | HL point | ChanQuant | 判定 |
|---|---|---|---|
| 72 | `intraday_strength` | **`intraday_strength`（新增）** | 缺 → 已注册 |
| 73 | `close_vs_open_ratio` | `mom_1` | ✅ 日线上恒等 |
| 74 | `close_position` | `close_pos` | ✅ 有（库里是 10 日均值） |
| 75 | `gap_continuation` | `gap` | ✅ 有（库里是 10 日均值） |
| 76 | `gap_fill_ratio` | **`gap_fill_ratio`（新增）** | 缺 → 已注册 |

### 非 Rust 部分

| 来源 | HL point | ChanQuant | 判定 |
|---|---|---|---|
| 价差结构 | `term_spread_near_far` | `term_near_far` | ✅ 有 |
| 价差结构 | `term_spread_main_far` | `roll_main_oi` | ✅ 有（同族） |
| 价差结构 | `roll_yield` | `roll_yield` | ✅ 有 |
| 持仓量 | `oi_chg_1d` / `5d` / `10d` | `oi_change_1` / `oi_change_5` / **`oi_change_10`（新增）** | 10 日缺 → 已注册 |
| 持仓量 | `oi_vol_ratio` | **`oi_vol_ratio`（新增）** | 缺 → 已注册 |
| 持仓量 | `oi_price_corr_5d` | **`oi_price_corr_5`（新增）** | 缺 → 已注册 |
| 持仓排名 | `net_pos_ratio` / `long_oi_share` / `short_oi_share` / `long_chg_ratio` / `short_chg_ratio` | — | ❌ **数据源不存在**（要天勤龙虎榜） |
| EDB | `basis_rate` | `spot_basis` | ✅ 有 |
| EDB | `spot_mom_5` / `spot_mom_20` | **`spot_mom_5` / `spot_mom_20`（新增）** | 缺 → 已注册 |
| EDB | `warehouse_rank` / `warehouse_chg_5` / `warehouse_chg_20` | `war_pct` / `war_chg_5` / **`warehouse_chg_20`（新增）** | 20 日缺 → 已注册 |
| EDB | `inventory_rank` / `inventory_chg_1w` / `inventory_chg_10w` / `inventory_yoy` / `inventory_seasonal` | `inv_pct` / `inv_chg_5` / — / **`inventory_yoy`（新增）** / — | `10w` 覆盖太少、`seasonal` 与同比恒等，不注册 |
| EDB | `delivery_rank` / `delivery_chg_mom` | **`deliv_pct`（新增）** / `deliv_chg_30` | 分位缺 → 已注册 |
| EDB | `profit_rank` / `profit_chg_4w` | **`profit_pct`（新增）** / `profit_chg_5` | 分位缺 → 已注册 |

### 汇总

| 结论 | 数量 |
|---|---|
| 原策略**有信息**的列（77 − 18 财务恒 0） | 59 + 3 价差 + 5 持仓 + 5 排名 + 15 EDB = **87** |
| ChanQuant **本来就有**（含等价替代） | 33 |
| **本来缺、本次新注册** | **42** |
| **数据源不存在、不参与** | 持仓排名 5 + `log_price` 1 + 绝对价位 4 = 10 |

---

## 三、复现的配置映射

| 原策略参数 | ChanQuant 字段 | 值 |
|---|---|---|
| 日线 | `period` | `"1d"` |
| 排除金融期货 | `multi_symbols` | 本机 86 个商品品种键 |
| 每月成交额前 70% | `cross_pool_mode="single"` + `cross_pool_keys=["log_amount_20"]` + `cross_pool_pct=70` + `cross_pool_reb=1` + `cross_pool_reb_unit="month"` | |
| 前一半做多 / 后一半做空 | `cross_top_pct=50` / `cross_bot_pct=50` | |
| 波动率倒数加权 | `cross_weight="risk"` | |
| **多空各半仓** | **`cross_weight_scope="legs"`** | 见下文 |
| 每 5 个交易日调仓 | `cross_rebalance=5` + `cross_rebalance_unit="day"` | |
| LightGBM 12 个月滚动 | `factor_mode="ml"` + `model="lightgbm"` + `train_window=252` + `step=21` | |
| 5 日预测 | `factor_cfg.horizon=5` | |
| 回看 90 天 | `factor_cfg.window=90` + `window_unit="day"` | |
| 500 万本金 / 80% 保证金 | `principal=5_000_000` + `capital_pct=80` + `sizing="capital"` | |

### 🔴 `cross_weight_scope` 这一项是复现的关键

原策略的组合日收益写的是 `Σ(w·r)/2`：多头腿内归一、空头腿内归一，**两边各占半仓**，
净敞口恒为 0 —— 这是「对冲」的定义式。

ChanQuant 从 2026-11-23 起默认是**合并分配**（`pooled`）：多 + 空放在一张表里归一化，
Σ|w| = 1 但两边不必等值、净敞口不钉 0。用 `pooled` 跑 HL point，两边波动水平不同时
组合会自己偏多头，就不是同一条策略了。实测（3 只低波动多头 + 3 只高波动空头）：

| 口径 | Σ\|w\| | 净敞口 |
|---|---|---|
| `pooled` | 1.0000 | **+0.8182** |
| `legs` | 1.0000 | **0.0000** |

所以这条注册里显式写了 `cross_weight_scope="legs"`。默认值仍是 `pooled`，
别人的策略不受影响。

---

## 四、与原策略的已知差异（解释曲线差异时用）

1. **标签口径**：原策略是「收益 > 1% 的二分类概率」，ChanQuant 截面 ML 是
   「预测收益（回归/排序）」。二分类概率不是期望收益的单调变换（同样的 μ，
   高波动品种 `P(r>1%)` 更高），严格说选出的品种不完全一样。这里取
   `label_mode="excess"`（预测收益减截面均值），是「谁涨得多」的线性对应，
   比 `volret` 更接近原意。
2. **标准化**：原策略做**截面 z-score**（今天相对别人），ChanQuant 做**时序
   滚动 z**（相对自己历史）再横比。两套不同，第二条对「这只品种自己是不是
   异常」更敏感。
3. **`inventory_chg_10w` / `inventory_seasonal`**：前者十周变化覆盖品种太少，
   后者在原项目里与 `inventory_yoy` 是同一行代码（`iloc[-53]`），不重复注册。
4. **持仓排名 5 个因子不参与**：需要天勤龙虎榜 `symbol_rank_daily`，ChanQuant
   没有这个数据源。

---

## 五、怎么复核

```bash
# 只打印配置，不落库
python _tools/hl_point_register.py --dry-run

# 重新注册 / 覆盖策略库里的那一条
python _tools/hl_point_register.py
```

自检里对应项：**「HL point 复现：新因子 + 资金分配范围（多空各半仓 / 合并）」**。
