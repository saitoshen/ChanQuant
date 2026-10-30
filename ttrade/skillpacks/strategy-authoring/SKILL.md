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
2. **先看 `strategy_list`**。7 套内置规则策略（双均线、唐奇安突破、ATR 通道、
   MACD、海龟、箱体波幅突破·迷你、IM 波幅突破·时段版）加用户自己注册的，
   都在里面。
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
