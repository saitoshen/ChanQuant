---
name: strategy-authoring
description: 写一套平台没有的规则策略（进出场逻辑）
when_to_use: 用户描述一套具体的进出场规则，现有策略加参数覆盖不了
group: strategy
risk: write
allowed-tools: doc, strategy_list, strategy_info, backtest, strategy
version: 1
---

# 写一套规则策略

## 先做减法

1. **先试参数**。用户说「海龟把 entry 调成 240」—— 先用 `backtest` 动作
   带着 `{"entry": 240}` 试跑，不要为此新增策略。库里多一个几乎重复的策略
   更难维护。
2. **先看 `strategy_list`**。9 套内置规则策略（买入持有、卖出持有两条基准，
   加双均线、唐奇安突破、ATR 通道、MACD、海龟、箱体波幅突破·迷你、
   箱体波幅突破·K线内、IM 波幅突破·时段版）加用户自己注册的，都在里面。
   自研策略的收益**必须和买入持有比** —— 跑不赢它，说明赚的是 beta 不是择时。
3. 逻辑确实不同，再写。

## 动作块

```json
{"action": "strategy", "key": "sbbox_mini", "label": "缩量突破·迷你",
 "desc": "突破 N 日高点入场，波动率过高时不开仓，跌破 M 日低点离场",
 "params": {"entry": 240, "exit": 720, "vol_max": 0.4, "atr_n": 200},
 "param_meta": {"entry": [20, 960, 20], "exit": [60, 1440, 20]},
 "func": "sig_sbbox_mini",
 "code": "def sig_sbbox_mini(df, entry=240, exit=720, vol_max=0.4, atr_n=200):\n    out = df.copy()\n    hi = out['high'].rolling(entry).max().shift(1)\n    lo = out['low'].rolling(exit).min().shift(1)\n    vol = out['close'].pct_change().rolling(20).std()\n    out['signal'] = _position_state(\n        long_in=out['close'] > hi,\n        long_out=out['close'] < lo,\n        short_in=out['close'] < lo,\n        short_out=out['close'] > hi,\n    )\n    out.loc[vol > vol_max, 'signal'] = 0\n    return out"}
```

## 成交时点：`signal` 是收盘定的，下一根开盘成交

用户写的策略只需要给 `signal`，引擎统一「T 日收盘定仓、T+1 开盘成交」。**不要**在
策略里自己 `shift` 去模拟盘中成交 —— 引擎会再 `roll` 一次，变成晚两天。

想要「K 线突破的那个价格进场」（突破类策略的常见要求），内置里已经有
`箱体+波幅突破·K线内`（`sbbox_ib`）：它额外输出 `entry_long` / `entry_short` /
`stop_dist` / `exit_long` / `exit_short` 五列，引擎据此在 K 线走的**过程中**成交。
平仓的那一根不再开新仓（不反手）：持多时看不到空头触发价，多单出场之后下一根起
才可能做空 —— 一根 K 线里「先平后开」和「先开后平」分不出先后，不能按有利的猜。
用户自定义策略目前不支持这条通道 —— 没有 `intrabar` 标记，引擎只会读 `signal`。

## 七条硬约束

1. **签名 `def sig_xxx(df, 参数...)`**。`df` 索引是 datetime，含
   `open/high/low/close/volume`（持仓量用 `_oi(df)`）。
2. **不要写 import**。`np`、`pd`、`sma`、`ema`、`atr`、`true_range`、
   `_position_state(多头进/多头出/空头进/空头出)` 都已注入。
3. **返回带 `signal` 列的等长 DataFrame**。`signal` 是**当日收盘后的目标仓位**：
   1 满仓多、0 空仓、-1 满仓空，只能取这三个值。
4. **必须 PIT**：只用 `rolling` / `ewm` / `shift(正数)`，**绝不 `shift(-n)`**。
5. **只管出仓位**。不要自己算手续费、滑点、手数、资金曲线 —— 回测引擎统一处理。
   自己算一遍会和引擎的假设打架，结果对不上。
6. 参数放进 `params`（带默认值），代码里同名形参接住；`param_meta` 可选，
   格式 `[最小, 最大, 步长]`，不写就按默认值自动推滑杆范围。
7. `key` 用英文小写加下划线，`label` 用中文。

## 注册失败怎么改

- 「信号列有 NaN」→ 预热期的 NaN 要填 0，或者用 `_position_state` 兜住
- 「signal 有 2 之类的值」→ 只能是 -1 / 0 / 1
- 「没有名为 xxx 的函数」→ 用了没注入的原语，去 `doc` 的 `primitives` 查
  「策略里额外能用的」那一节

改好之后重新给一个 `strategy` 动作块。
