"""策略库：查清单、看参数定义。

注册策略和注册因子同理 —— 写盘 + 试算 + 刷新下拉，还要弹确认框给人看代码，
留在界面层的 `strategy` 动作里。这里只做只读查询。
"""

from __future__ import annotations

from ttrade import blocks as B
from ttrade.skills.base import (
    RISK_READ,
    RISK_WRITE,
    SkillResult,
    Tool,
    register,
)


def _strategies():
    from ttrade.strategies import STRATEGIES

    return STRATEGIES


def _param_text(spec) -> str:
    from ttrade.strategies import PARAM_LABELS

    bits = []
    for k, v in (spec.params or {}).items():
        label = PARAM_LABELS.get(k, k)
        bits.append(f"{label}={v}")
    return " ".join(bits) or "—"


def _strategy_list(ctx, p: dict) -> SkillResult:
    strategies = _strategies()
    q = str(p.get("query") or "").strip().lower()

    rows = []
    for k, spec in strategies.items():
        if q and q not in k.lower() and q not in spec.label:
            continue
        rows.append([k, spec.label, "自定义" if spec.code else "内置",
                     _param_text(spec), spec.desc[:36]])
    if not rows:
        raise RuntimeError(f"没有匹配「{q}」的策略")

    note = f"共 {len(rows)} 个"
    blocks = [
        B.block_table(["代号", "名称", "来源", "默认参数", "说明"], rows, note=note),
    ]
    feed = (f"注册的规则策略共 {len(rows)} 个：" +
            "；".join(f"{r[0]}={r[1]}（{r[3]}）" for r in rows))
    return SkillResult(True, summary=f"注册的规则策略共 {len(rows)} 个",
                       blocks=blocks, feed=feed)


def _strategy_info(ctx, p: dict) -> SkillResult:
    strategies = _strategies()
    key = str(p["strategy"] or "").strip().lower()
    spec = strategies.get(key)
    if spec is None:
        near = [k for k in strategies if key in k][:8]
        raise RuntimeError(f"没有名为 {key} 的策略"
                           + (f"（相近的有：{'、'.join(near)}）" if near else ""))

    from ttrade.strategies import PARAM_LABELS

    meta_rows = []
    for k, v in (spec.params or {}).items():
        lo, hi, step = (spec.param_meta.get(k) or ("", "", ""))[:3]
        meta_rows.append([PARAM_LABELS.get(k, k), k, str(v), f"{lo} ~ {hi}", str(step)])

    blocks = [
        B.block_metrics(spec.label, [
            ("代号", spec.key, "flat"),
            ("来源", "自定义" if spec.code else "内置", "flat"),
            ("可做空", "是" if spec.bidirectional else "否", "flat"),
            ("参数", str(len(spec.params or {})), "flat"),
        ]),
        B.block_text(spec.desc or "（没有说明）"),
    ]
    if meta_rows:
        blocks.append(B.block_table(["参数", "代号", "默认", "范围", "步长"], meta_rows))
    if spec.code:
        blocks.append(B.block_code(spec.code, "python"))

    feed = f"{spec.label}（{spec.key}）：{spec.desc}\n默认参数：{_param_text(spec)}"
    if spec.code:
        feed += f"\n实现：\n```python\n{spec.code}\n```"
    return SkillResult(True, summary=f"{spec.label}（{spec.key}）", blocks=blocks, feed=feed)


register(Tool(
    name="strategy_list",
    group="strategy",
    label="查策略库",
    description="列出注册的规则策略及其默认参数",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "properties": {"query": {"type": "str", "default": ""}},
    },
    handler=_strategy_list,
    prompt=(
        "**查策略库**：`strategy_list()`。注意这只列**注册的规则策略代号**；"
        "用户存好的整份模板（带 sid）在每次提问附的那张表里，不在这里。"
        "要按策略回测时，代号从这里抄，不要编。"
    ),
))

register(Tool(
    name="strategy_info",
    group="strategy",
    label="看策略定义",
    description="看一个策略的说明、参数范围和实现代码",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["strategy"],
        "properties": {"strategy": {"type": "str"}},
    },
    handler=_strategy_info,
    prompt=(
        "**看策略定义**：`strategy_info(strategy=\"donchian\")`。"
        "用户问「这个策略怎么做的 / 参数范围多少」，或者你要照着改一个现有策略时用它。"
    ),
))


# ------------------------------------------------------------------ 注册新策略

def prepare_strategy(p: dict) -> tuple[dict, str]:
    """注册策略的动作块。只校验「能不能落地」，不评判策略好坏 ——
    好坏由回测数字说话，不由这里拦。"""
    key = str(p.get("key") or "").strip().lower()
    label = str(p.get("label") or "").strip()
    code = str(p.get("code") or "")
    if not key:
        return p, "缺少策略代号 key"
    if not label:
        return p, "缺少显示名 label"
    if not code.strip():
        return p, "缺少实现代码 code"

    func = str(p.get("func") or f"sig_{key}").strip()
    if f"def {func}" not in code:
        return p, f"code 里要定义 def {func}(df, ...)"

    params = p.get("params") or {}
    if not isinstance(params, dict):
        return p, "params 要是一个对象"
    for k, v in params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return p, f"参数 {k} 的值 {v!r} 不是数字"

    meta = p.get("param_meta") or {}
    if not isinstance(meta, dict):
        meta = {}

    p["key"] = key
    p["label"] = label
    p["func"] = func
    p["code"] = code
    p["desc"] = str(p.get("desc") or "").strip()
    p["params"] = params
    p["param_meta"] = meta
    return p, ""


def describe_strategy(p: dict) -> str:
    return f"注册策略「{p.get('label', '')}」（{len(p.get('params') or {})} 个参数）"


register(Tool(
    name="strategy",
    group="strategy",
    label="注册策略",
    description="把一段 Python 写成新规则策略，注册进策略库。会弹确认框",
    risk=RISK_WRITE,
    inputSchema={
        "type": "object",
        "required": ["key", "label", "code"],
        "properties": {
            "key": {"type": "str"},
            "label": {"type": "str"},
            "desc": {"type": "str", "default": ""},
            "params": {"type": "dict", "default": {}},
            "param_meta": {"type": "dict", "default": {}},
            "func": {"type": "str", "default": ""},
            "code": {"type": "str"},
        },
    },
    prepare=prepare_strategy,
    ui="strategy",
    describe=describe_strategy,
))
