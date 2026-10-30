---
name: trade-checklist
description: 下单前的检查清单，以及确认框里该写什么
when_to_use: 用户说「买」「卖」「平仓」「上实盘」，或者要给 trade 动作块
group: trade
risk: trade
allowed-tools: trade, strategy_list, query_bars
version: 1
---

# 下单前

## 动作块之前，先把这四件事列清楚

**不要先给动作块再解释。** 用户要在点「确认」之前看到：

1. **买什么** —— 品种代号 + 中文名，合约代码
2. **多少** —— 手数，以及这一手占多少保证金
3. **什么价** —— 市价还是限价，如果是限价给具体数字
4. **为什么** —— 一句话，来自哪条策略或哪个信号

## 两条模式

按策略交易：

```json
{"action": "trade", "mode": "auto", "symbol": "RB", "exchange": "SHFE",
 "period": "1d", "strategy": "ma_cross", "params": {"fast": 50, "slow": 200},
 "volume": 1, "reason": "一句话说明"}
```

主观买卖：

```json
{"action": "trade", "mode": "manual", "symbol": "RB", "exchange": "SHFE",
 "side": "BUY", "offset": "OPEN", "volume": 1, "reason": "一句话说明"}
```

- `side`：`BUY` / `SELL`
- `offset`：`OPEN` 开仓 / `CLOSE` 平仓 / `CLOSETODAY` 平今

## 必须先问的三种情况

- **用户没说手数** → 不要自己定。给一个建议区间，让他选
- **用户没说开仓还是平仓** → 问。平仓方向搞反是最贵的错误
- **账户没连、保证金不够** → 直接说，不要给一个必然失败的动作块

## 红线

- 下单是**唯一一档动真钱**的动作。确认框会弹，但不要指望用户每次都看仔细 ——
  所以你的说明文字本身就是最后一道防线，要写清最坏情况
- 用户开了「不再询问」之后动作会直接执行。这种情况下更要先说清再给动作块
- 不要建议满仓。不要建议加杠杆。不要建议「先试一手看看」这种话术 ——
  一手也是真钱
