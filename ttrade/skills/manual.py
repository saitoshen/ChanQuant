"""查平台手册（`doc` 动作）。

手册里写着行情是怎么来的（三条链路、**每条链路各自的列名**、复权口径）、
平台往命名空间里注入了哪些原语、以及全部因子的完整实现。

这是**写因子之前必查**的东西 —— 凭印象猜列名是最常见的失败原因：日线 CSV 叫
`open_interest`，天勤分钟线叫 `open_oi`，写死一个在另一种数据上就是全 NaN。

只读本地文本，不碰磁盘、不弹框，所以 `risk=read`。
"""

from __future__ import annotations

from ttrade.skills.base import (
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


register(Tool(
    name="doc",
    group="manual",
    label="查平台手册",
    description="查行情链路/列名/原语源码/因子实现（写因子前必查）",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "properties": {"topic": {"type": "str"}},
    },
    prepare=prepare_doc,
    handler=_doc,
    describe=lambda p: f"查平台手册：{p.get('topic', '')}",
    max_turns=3,
))
