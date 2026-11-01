"""下单与回测：动作块解析在这里，执行在界面层。

## 为什么 handler 是空的

这两个动作要动界面的状态 —— 回测要在「策略研究」页套上配置跑一遍、把 K 线 /
绩效 / 因子 / 交易记录 / 阶段分析刷出来（用户接着就能在那儿调参数再跑），下单
要拿到实时会话的真实主力合约。`ttrade` 里做不了，也不该做。

所以这里只负责**协议**：字段长什么样、哪些组合合法、`describe()` 怎么给人看。
真正的执行挂在 `ui="signal"`，由界面层 `trade_requested` 信号接走。

`trade` 的确认框不是可选项 —— 模型说「买 10 手」和真的下单之间必须有一道人手。
"""

from __future__ import annotations

from ttrade.skills.base import (
    RISK_READ,
    RISK_TRADE,
    Tool,
    register,
)

# 只做展示，不参与计算
OFFSET_CN = {"OPEN": "开仓", "CLOSE": "平仓", "CLOSETODAY": "平今"}


def _sym(p: dict) -> str:
    return f"{p.get('symbol', '')}_{p.get('exchange', '')}".strip("_")


def _common(p: dict) -> tuple[dict, str]:
    """trade / backtest 共用的收敛：品种必填、手数至少 1。"""
    if not str(p.get("symbol") or "").strip():
        return p, "缺少品种"
    p["symbol"] = str(p["symbol"]).strip().upper()
    p["exchange"] = str(p.get("exchange") or "").strip().upper()
    p["period"] = str(p.get("period") or "1d").strip() or "1d"
    p["strategy"] = str(p.get("strategy") or "").strip()
    if not isinstance(p.get("params"), dict):
        p["params"] = {}
    try:
        p["volume"] = max(1, int(p.get("volume", 1) or 1))
    except (TypeError, ValueError):
        p["volume"] = 1
    return p, ""


def prepare_backtest(p: dict) -> tuple[dict, str]:
    return _common(dict(p))


def prepare_trade(p: dict) -> tuple[dict, str]:
    p, why = _common(dict(p))
    if why:
        return p, why
    mode = str(p.get("mode") or "auto").lower()
    if mode not in ("auto", "manual"):
        return p, "mode 只能是 auto 或 manual"
    p["mode"] = mode
    if mode == "manual":
        side = str(p.get("side") or "").upper()
        if side not in ("BUY", "SELL"):
            return p, "主观交易要给 side=BUY/SELL"
        p["side"] = side
        off = str(p.get("offset") or "OPEN").upper()
        if off not in ("OPEN", "CLOSE", "CLOSETODAY"):
            return p, "offset 只能是 OPEN/CLOSE/CLOSETODAY"
        p["offset"] = off
    elif not p.get("strategy"):
        return p, "按策略交易要给 strategy"
    p["reason"] = str(p.get("reason") or "").strip()
    return p, ""


def describe_trade(p: dict) -> str:
    sym = _sym(p)
    if p.get("mode") == "manual":
        side = "买入" if p.get("side") == "BUY" else "卖出"
        off = OFFSET_CN.get(p.get("offset", "OPEN"), p.get("offset", ""))
        return f"{side}{off} {sym} {p.get('volume', 1)} 手"
    return (f"按策略交易 {sym} {p.get('period', '')}｜"
            f"{p.get('strategy', '')}｜{p.get('volume', 1)} 手")


def describe_backtest(p: dict) -> str:
    return f"回测 {_sym(p)} {p.get('period', '')}｜{p.get('strategy', '')}"


_BT_SCHEMA = {
    "type": "object",
    "required": ["symbol"],
    "properties": {
        "symbol": {"type": "str"},
        "exchange": {"type": "str", "default": ""},
        "period": {"type": "str", "default": "1d"},
        "strategy": {"type": "str", "default": ""},
        "params": {"type": "dict", "default": {}},
        "volume": {"type": "int", "default": 1},
        "reason": {"type": "str", "default": ""},
    },
}

register(Tool(
    name="backtest",
    group="backtest",
    interrupt="cancel",
    max_result_chars=12000,
    label="跑回测",
    description="按指定品种/周期/策略跑一遍回测，结果显示在策略研究页（可以直接接着调参数）",
    risk=RISK_READ,
    inputSchema=_BT_SCHEMA,
    prepare=prepare_backtest,
    ui="signal",
    describe=describe_backtest,
))

register(Tool(
    name="trade",
    group="trade",
    label="下单",
    description="按策略下单，或者主观买卖开平。**动真钱**",
    risk=RISK_TRADE,
    inputSchema={
        "type": "object",
        "required": ["symbol"],
        "properties": {
            "symbol": {"type": "str"},
            "exchange": {"type": "str", "default": ""},
            "period": {"type": "str", "default": "1d"},
            "strategy": {"type": "str", "default": ""},
            "params": {"type": "dict", "default": {}},
            "volume": {"type": "int", "default": 1},
            "mode": {"type": "str", "default": "auto"},
            "side": {"type": "str", "default": ""},
            "offset": {"type": "str", "default": "OPEN"},
            "reason": {"type": "str", "default": ""},
        },
    },
    prepare=prepare_trade,
    ui="signal",
    describe=describe_trade,
))
