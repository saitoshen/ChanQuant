"""查平台手册（`doc` 动作）。

手册里写着行情是怎么来的（三条链路、**每条链路各自的列名**、复权口径）、
平台往命名空间里注入了哪些原语、以及全部因子的完整实现。

这是**写因子之前必查**的东西 —— 凭印象猜列名是最常见的失败原因：日线 CSV 叫
`open_interest`，天勤分钟线叫 `open_oi`，写死一个在另一种数据上就是全 NaN。

只读本地文本，不碰磁盘、不弹框，所以 `risk=read`。
"""

from __future__ import annotations

from ttrade.skills.base import (
    NO_LIMIT,
    RISK_READ,
    SkillResult,
    Tool,
    register,
)

TOPICS = {
    "data": "行情数据：链路、列名、复权、缺数据",
    "primitives": "原语源码（_ret / _col / _oi / standardize 等）",
    "contract": "写因子/策略的契约与常见错误",
    "factors": "全部因子索引（一行一个）",
    "cat:趋势": "整类因子的完整实现",
    "factor:mom_20": "单个因子的完整实现",
}


def _doc(ctx, p: dict) -> SkillResult:
    from ttrade import blocks as B
    from ttrade import factor_manual as M

    topic = str(p["topic"]).strip()
    title, body = M.resolve(topic)
    # 对话里只贴前 6000 字符：整份手册几万字，贴进去 QTextBrowser 会卡，
    # 而模型拿到的仍然是全文（走 feed）
    shown = body if len(body) <= 6000 else body[:6000] + "\n……（完整内容已发给模型）"
    blocks = [
        B.block_metrics(title, [
            ("topic", topic, "flat"),
            ("字数", f"{len(body):,}", "flat"),
        ]),
        B.block_code(shown, "text"),
    ]
    return SkillResult(True, summary=f"已查手册：{title}（{len(body)} 字符）",
                       blocks=blocks, feed=f"平台手册「{title}」的内容：\n\n{body}")


def prepare_doc(p: dict) -> tuple[dict, str]:
    """topic 认不认识交给 `factor_manual.resolve()` —— 那里解析不了会回一份
    索引，比在这里拒掉更有用。"""
    topic = str(p.get("topic") or "").strip()
    if not topic:
        return p, ("缺少 topic（data / primitives / contract / factors / "
                   "cat:趋势 / factor:<key>）")
    p["topic"] = topic
    return p, ""


def _read_output(ctx, p: dict) -> SkillResult:
    """把超长工具结果的分段读回来。

    这条工具是「结果预算」的另一半：`agent_budget.clip()` 把超过上限的结果
    落盘、只回首尾预览，模型看到「完整结果已存为 xxx」时用它接着读。
    没有它，裁剪就等于丢数据 —— 模型只能凭半截预览猜。
    """
    from ttrade import agent_budget as BUD
    from ttrade import blocks as B

    name = str(p["id"]).strip()
    text, total, err = BUD.read_chunk(name, p.get("start", 0), p.get("limit", 6000))
    if err:
        return SkillResult(False, error=err, feed=f"读回落盘结果失败：{err}")

    start = int(p.get("start", 0) or 0)
    blocks = [
        B.block_metrics(f"落盘结果 {name}", [
            ("全文", f"{total:,} 字符", "flat"),
            ("本段", f"{start:,} 起", "flat"),
        ]),
        B.block_code(text[:6000], "text"),
    ]
    return SkillResult(
        True,
        summary=f"读了 {name} 的 {start:,}–{start + len(text):,} 字符（共 {total:,}）",
        blocks=blocks, feed=f"落盘结果 `{name}` 第 {start} 字符起：\n\n{text}",
    )


register(Tool(
    name="read_output",
    group="manual",
    label="读回长结果",
    description="分段读回被截断的长结果（结果里出现「已存为 xxx」时用它）",
    risk=RISK_READ,
    # 它自己按 start / limit 分段，不需要再裁
    max_result_chars=NO_LIMIT,
    inputSchema={
        "type": "object",
        "required": ["id"],
        "properties": {
            "id": {"type": "str"},
            "start": {"type": "int", "default": 0, "minimum": 0},
            "limit": {"type": "int", "default": 6000, "minimum": 200, "maximum": 20000},
        },
    },
    handler=_read_output,
    describe=lambda p: f"读回长结果：{p.get('id', '')}",
    prompt=(
        "**读回长结果**：`read_output`。工具结果被截断时，预览末尾会给一个编号，"
        "用这条按 `start` 分段往后读。一次别超过 8000 字符 —— 拉全文等于没做预算。"
    ),
    max_turns=3,
))


def _read_skill(ctx, p: dict) -> SkillResult:
    """取一份技能的正文。

    这是「渐进披露」的另一半：提示词里只列技能的名字和用途（一行一条），
    真要用时才把正文拉进来。一份技能正文两三千字，六个全量注入就是一万五，
    而一次对话通常只用到其中一个。
    """
    from ttrade import blocks as B
    from ttrade import skillpacks as SP

    name = str(p["name"]).strip()
    body, err = SP.read(name)
    if err:
        return SkillResult(False, error=err, feed=err)

    pack = SP.get(name)
    src = "内置" if (pack is not None and pack.source == "builtin") else "我的"
    blocks = [
        B.block_metrics(f"技能 · {name}", [
            ("来源", src, "flat"),
            ("字数", f"{len(body):,}", "flat"),
        ]),
        B.block_code(body[:8000], "markdown"),
    ]
    return SkillResult(
        True, summary=f"读了技能「{name}」（{len(body)} 字符）",
        blocks=blocks, feed=f"技能「{name}」的正文：\n\n{body}",
    )


register(Tool(
    name="read_skill",
    group="manual",
    label="读技能",
    description="取一份技能的完整做法（提示词里只给了名字，要动手前先读）",
    risk=RISK_READ,
    # 技能正文自己限过长度（`skillpacks.MAX_BODY_CHARS`），不再二次裁剪 ——
    # 裁一半等于让模型照着半份流程干活
    max_result_chars=NO_LIMIT,
    inputSchema={
        "type": "object",
        "required": ["name"],
        "properties": {"name": {"type": "str"}},
    },
    handler=_read_skill,
    describe=lambda p: f"读技能：{p.get('name', '')}",
    prompt=(
        "**读技能**：`read_skill(name=\"factor-authoring\")`。写因子、写策略、"
        "下单、读回测之前，先读对应技能 —— 里面是这套流程的硬约束，"
        "不读就动手是最常见的失败原因。"
    ),
    max_turns=3,
))


register(Tool(
    name="doc",
    group="manual",
    label="查平台手册",
    description="查行情链路/列名/原语源码/因子实现（写因子前必查）",
    risk=RISK_READ,
    # 手册是「要照着写代码」的参考文本，裁一半等于让模型照着半份说明写因子。
    # 它自己按 topic 分段，一段几千到两万字，不会失控。
    max_result_chars=NO_LIMIT,
    inputSchema={
        "type": "object",
        "properties": {"topic": {"type": "str"}},
    },
    prepare=prepare_doc,
    handler=_doc,
    describe=lambda p: f"查平台手册：{p.get('topic', '')}",
    prompt=(
        "**查平台手册**：`doc(topic=\"data\")`。写因子、或者回答「这个数据从哪来」"
        "之前必查 —— **不要凭印象猜列名**。topic 可选：\n"
        "- `data` 行情链路 / 每条链路各自的列名 / 复权 / 缺数据\n"
        "- `primitives` 原语源码（`_ret` / `_col` / `_oi` / `standardize`）\n"
        "- `contract` 写因子和策略的契约、常见错误\n"
        "- `factors` 全部因子索引（一行一个，先看这个）\n"
        "- `cat:趋势` 整类因子的完整实现（趋势/反转/波动/量能/形态/展期/期权）\n"
        "- `factor:mom_20` 单个因子的完整实现\n\n"
        "一次只查一段，别一口气把整份手册拉进对话。"
    ),
    max_turns=3,
))
