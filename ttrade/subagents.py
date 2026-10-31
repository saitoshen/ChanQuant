"""子 Agent：把「查一遍」和「复核一遍」交给两个独立的、只读的助手。

## 为什么只要两个

Claude Code 的 coordinator 模式有四个阶段（Research → Synthesis →
Implementation → Verification）加一整支 worker 队伍。那是给通用编码任务
准备的 —— 改代码要并行探索、要分头实现、要独立验收。

量化只有两件事值得外包：

| 子 Agent | 干什么 | 为什么值得单开 |
|---|---|---|
| `research` | 查数据、算因子、读文档，只回结论 | 中间翻了哪些表、试了哪些参数**不进主上下文** —— 主对话只拿到一段结论 |
| `verify` | 独立复核一份回测结论 | 它**没看过**得出这个结论的过程，所以不会被锚定 |

`verify` 是刚需。「回测漂亮但实盘亏」九成是验证没做：用了未来数据、漏了换月、
滑点按 0 算、样本内调参。一个从没见过这个策略的复核者，比主 Agent 自己
回头检查有效得多 —— 后者已经知道自己想看到什么。

## 为什么不给它们写权限

`READ_GROUPS` 里全是只读能力包。子 Agent **不能**下单、不能注册因子、不能写文件、
不能回测（回测要主窗口执行）。这不是保守，是划边界：需要动真钱或改库的操作，
必须回到主对话，由用户看着确认框点一下。

## 为什么是同步函数

`run()` 是纯阻塞的，不碰 Qt。调用方（`gui/agent_page.py`）把它放进 QThread ——
这样命令行、冒烟测试、将来的 MCP server 都能直接调，不用拖一个事件循环。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ttrade import agent_budget as BUD
from ttrade import skills as S

# 子 Agent 能碰的能力包。只读的四个 —— 见文件头「为什么不给它们写权限」
READ_GROUPS: tuple[str, ...] = ("data", "manual", "doc", "monitor")

# 子 Agent 的动作块协议。比主对话的短：它不需要知道怎么注册因子、怎么下单
_PROTOCOL = """## 动作块

需要执行操作时，在回答**最后**单独输出一个 json 代码块，平台会执行并把结果
发回给你：

```json
{"action": "query_bars", "symbol": "RB_SHFE", "period": "1d", "rows": 60}
```

不需要执行任何操作时，**不要输出 json 块**，直接按格式给结论。

动作块写错会被退回来，平台会说明哪个字段、期望什么、收到什么。照着改。"""


@dataclass(frozen=True)
class Spec:
    """一个子 Agent 的定义。"""

    name: str
    label: str
    groups: tuple[str, ...]
    prompt: str
    max_turns: int = 6


@dataclass
class Result:
    ok: bool = False
    summary: str = ""          # 回给主 Agent 的结论正文
    error: str = ""
    steps: int = 0
    tools: list[str] = field(default_factory=list)

    def feed(self) -> str:
        """拼成回灌给主 Agent 的一段。"""
        if not self.ok:
            return f"【{self.error}】"
        used = "、".join(self.tools) if self.tools else "无"
        return (f"【子 Agent 结论】\n{self.summary}\n\n"
                f"（共 {self.steps} 轮，用过：{used}。"
                "这是另一个助手独立查出来的，不是你自己的记忆 —— "
                "引用时按事实用，不要复述这段说明。）")


RESEARCH = Spec(
    name="research",
    label="研究助手",
    groups=READ_GROUPS,
    max_turns=6,
    prompt="""你是 ChanAgent 的**研究助手**。你只做一件事：查清楚一个问题，然后给结论。

你能用的动作：查行情、查因子值、列品种、查平台手册、读附件、扫品种快照。
你**不能**回测、不能下单、不能注册因子 —— 需要那些就在结论里写明「这步要主对话做」。

规矩：

- 只查必要的东西。同一个查询不要重复做，参数从宽到窄一次收敛
- 不要寒暄，不要说「好的我来查一下」这类过程话，直接查
- 查完就停，按下面的格式给结论
- **查不到就说查不到**，不要用常识补一个像样的数字

## 结论格式（严格照这个，不要别的话）

结论：一句话说清
依据：2–4 条，每条带具体数字、区间、样本数
不确定：查不到或互相矛盾的地方（没有就写「无」）
下一步：建议主对话做什么（没有就写「无」）""",
)


VERIFY = Spec(
    name="verify",
    label="复核助手",
    groups=("data", "manual", "doc"),
    max_turns=5,
    prompt="""你是 ChanAgent 的**复核助手**。你的工作不是确认结果好看，是**找它错在哪**。

你没有看过得出这个结论的过程 —— 这是故意的。你只看到下面这段待复核的内容，
和你自己查到的数据。**不要假设它是对的。**

按这个顺序查四件事：

1. **未来函数**：因子/策略代码里有没有 `shift(-n)`、有没有用到当日收盘后才有的
   数据、标准化是不是用了全样本的均值方差
2. **样本**：回测区间多长、交易几次、是不是只覆盖一种市场状态（单边 / 震荡）
3. **成本**：手续费、滑点、换月处理是不是按用户的实际口径。平台默认次日开盘
   成交、双边成本、逐笔保证金 —— 如果结论里的数字比这乐观，问为什么
4. **参数**：窗口/阈值是不是在样本内调出来的。试着想一下「改一点点还成立吗」

## 结论格式（严格照这个，不要别的话）

判定：可信 / 有条件可信 / 不可信
问题：逐条列，每条说清「哪里、为什么是问题、有多严重」（没有就写「无」）
建议：具体怎么改（没有就写「无」）

**不要为了让用户高兴而说「可信」。** 找不到问题是一种结论，但要说清你查了哪些
地方、用什么数据核对的。""",
)

SPECS: dict[str, Spec] = {s.name: s for s in (RESEARCH, VERIFY)}


def get(name: str) -> Spec | None:
    return SPECS.get(str(name or "").strip().lower())


def usable_tools(spec: Spec) -> list[S.Tool]:
    """这个子 Agent 真能调的工具。

    过滤掉三类：

    1. 不在它能力包里的
    2. **非只读的** —— `doc` 包里就有 `write_xlsx` / `write_pdf`，光看能力包
       会把「导出 PDF」也发给一个只读助手
    3. 需要界面执行的（`ui` 非空或没有 `handler`）—— 回测和下单都属于这类
    """
    out: list[S.Tool] = []
    for t in S.REGISTRY.values():
        if t.group not in spec.groups:
            continue
        if not t.is_read_only():
            continue
        if t.handler is None:
            continue
        out.append(t)
    return out


def _system_prompt(spec: Spec) -> str:
    tools = usable_tools(spec)
    catalog = S.catalog(enabled=list(spec.groups))
    names = "、".join(f"`{t.name}`" for t in tools)
    return (spec.prompt
            + "\n\n## 你能用的动作\n\n" + catalog
            + f"\n\n（只有上面这些里带 `handler` 的能直接跑，共 {len(tools)} 个：{names}。"
              "回测、下单、注册因子这类需要主界面执行的动作你调不了。）"
            + "\n\n" + _PROTOCOL)


def run(cfg, spec: Spec, task: str, ctx,
        on_step=None, should_stop=None) -> Result:
    """跑一个子 Agent。阻塞，调用方负责放线程里。

    `on_step(text)` 每轮结束时回调一次（给界面显示进度用）。
    `should_stop()` 返回 True 就中止 —— 用户点了「停止」。
    """
    from ttrade import agent as A

    if cfg is None or not cfg.ready():
        return Result(error=f"{spec.label}跑不起来：模型还没配置好")

    allowed = {t.name for t in usable_tools(spec)}
    if not allowed:
        return Result(error=f"{spec.label}没有可用的动作，先到设置里打开能力包")

    client = A.LLMClient(cfg)
    msgs: list[dict] = [
        {"role": "system", "content": _system_prompt(spec)},
        {"role": "user", "content": task.strip()},
    ]
    used: list[str] = []
    last = ""

    for step in range(1, spec.max_turns + 1):
        if should_stop is not None and should_stop():
            return Result(ok=False, error=f"{spec.label}已停止", steps=step - 1,
                          tools=used)
        try:
            text = client.chat(msgs, should_stop=should_stop)
        except Exception as exc:  # noqa: BLE001 - 子 Agent 挂了不能拖垮主对话
            return Result(error=f"{spec.label}出错：{type(exc).__name__}: {exc}",
                          steps=step - 1, tools=used)

        last = text or ""
        msgs.append({"role": "assistant", "content": last})
        if on_step is not None:
            try:
                on_step(last)
            except Exception:  # noqa: BLE001
                pass

        action = A.parse_action(last)
        if action.error:
            msgs.append({"role": "user", "content": A.action_error_feedback(action)})
            continue
        if not action.ok:
            return Result(ok=True, summary=_strip_block(last), steps=step, tools=used)
        if action.kind not in allowed:
            msgs.append({"role": "user", "content": (
                f"「{action.kind}」这个动作你调不了（需要主界面执行，或者不在你的"
                f"能力范围里）。你能用：{'、'.join(sorted(allowed))}。"
                "按现有信息给结论，把这一步写进「下一步」。"
            )})
            continue

        tool = S.get(action.kind)
        res = S.dispatch(ctx, action.kind, action.payload)
        used.append(action.kind)
        limit = tool.max_result_chars if tool is not None else S.DEFAULT_RESULT_CHARS
        body = BUD.clip(res.feed_text(), limit, f"sub-{spec.name}")
        if not res.ok:
            body = f"【失败】{res.error}\n\n{body}".strip()
        msgs.append({"role": "user", "content": (
            "【平台执行结果】\n" + body
            + "\n\n需要继续就再给一个动作块，不需要就按格式给结论。"
        )})

    return Result(
        ok=True,
        summary=(_strip_block(last).rstrip()
                 + f"\n\n（已用满 {spec.max_turns} 轮，上面是最后一轮的结论，"
                   "可能还没查完）"),
        steps=spec.max_turns, tools=used,
    )


def _strip_block(text: str) -> str:
    """把结论里的动作块去掉 —— 回灌给主 Agent 的应该是人话，不是 JSON。

    顺带清掉模型自带的工具语法（`<|DSML|>invoke …`）：子 Agent 也会这么写，
    不清的话主 Agent 会把一段尖括号当成「研究结论」。
    """
    import re

    from ttrade import agent as A

    body = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", text or "", flags=re.S)
    return A.strip_tool_markup(body).strip()
