"""派子 Agent：`run_research` / `run_verify`。

两个动作都不干活，只是**把活交出去**，然后拿回一段结论。真正的执行在
`ttrade/subagents.py`，界面层负责放线程里跑（见 `gui/agent_page.py` 的
`SubAgentWorker`）。

## 为什么这两个是 `ui` 动作

子 Agent 要连着调好几次模型，一次可能跑十几秒到一分钟。放在界面线程里会
整个卡住，连「停止」都点不动 —— 所以和 `backtest` 一样交给界面，由它起线程。
代价是 `dispatch()` 调不动它们（MCP server 那边会回「只能由界面执行」），
这个取舍是对的：子 Agent 的输出是给人看的，没有界面就没有意义。

## 为什么 `risk=read`

它确实只读：`subagents.READ_GROUPS` 里全是查数据、查手册、读附件。
不弹确认框是有意的 —— 「帮我查一下螺纹最近的动量分位」不该每次都要点一下。
"""

from __future__ import annotations

from ttrade.skills.base import RISK_READ, Tool, register

register(Tool(
    name="run_research",
    group="delegate",
    label="派研究助手",
    description="让只读的研究助手查一件事，只回结论（中间查了什么不进你的上下文）",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["task"],
        "properties": {
            "task": {"type": "str"},
        },
    },
    ui="research",
    describe=lambda p: "派研究助手：" + str(p.get("task") or "")[:48],
    prompt=(
        "**派研究助手**：`run_research(task=\"...\")`。\n"
        "什么时候用：要查的东西**不止一两个动作**（比如「把黑色系最近一个月的动量"
        "分位都看一遍」），中间过程对你没用、只要结论时。\n"
        "什么时候别用：一句话能查完的（`query_bars` 就够），派出去反而慢。\n"
        "`task` 要写成**自包含**的一句话 —— 研究助手看不到你和用户的对话，"
        "它只知道你写进去的东西。品种、周期、区间、要算什么，都要写清。"
    ),
    max_turns=1,
))

register(Tool(
    name="run_verify",
    group="delegate",
    label="派复核助手",
    description="让一个没看过过程的复核助手挑毛病：未来函数、样本、成本、参数",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["claim"],
        "properties": {
            "claim": {"type": "str"},
        },
    },
    ui="verify",
    describe=lambda p: "派复核助手复核：" + str(p.get("claim") or "")[:40],
    prompt=(
        "**派复核助手**：`run_verify(claim=\"...\")`。\n"
        "什么时候用：**回测跑完、要给「能不能上」的结论之前**。它没看过得出这个"
        "结论的过程，所以不会被你的思路锚定 —— 这是它比你自己回头检查强的地方。\n"
        "`claim` 要写成一段**待复核的结论**，把数字、区间、参数、成本假设都写进去。"
        "写得越具体，它挑得越准；写「这个策略好像不错」等于没写。\n"
        "它回「不可信」时不要急着辩解，先看它指出的那条是不是真的。"
    ),
    max_turns=1,
))
