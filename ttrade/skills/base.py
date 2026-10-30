"""Skill / Tool 注册表。

## 两层：能力包（Skill）⊃ 工具（Tool）

- **Tool** 是一次可调用动作，名字就是动作块里的 `"action"`。字段名照 MCP 的
  `Tool` 对象（`name` / `description` / `inputSchema`），这样以后要把这套东西
  吐成 MCP server 只是加一个序列化函数的事。
- **Skill**（这里叫 group）是一组工具 + 一段提示词 + 一个权限开关。提示词只在
  这个包被启用时才注入 —— 现在的 `SYSTEM_PROMPT` 五千多字符是**全量常驻**，
  不管这轮用不用得上网页和下单，那些说明都在发。

## 为什么要有注册表

以前加一个工具要改四处：`parse_action` 的白名单、一个 `_parse_xxx` 做字段校验、
`Action.describe()` 里一个 `if`、`_handle_action()` 里一个 `if`，外加手写一段提示词。
现在加一个工具 = 写一个 `@register(...)` 的函数，其余全自动。

## 风险分三档

| 级别 | 行为 |
|---|---|
| `read` | 自动执行，不弹框（回测、查数据、读文件、查手册） |
| `write` | 弹确认，把要写的内容摊开（注册因子/策略、写文件、导出 PDF、出网） |
| `trade` | 弹确认 + 红字 + 手数/价格/理由 + 日限额预检 |

`trade` 单独一档不是洁癖：它动的是真钱，弹框里必须能一眼看到「买什么、多少、
为什么」，而 `write` 那一档看的是「要写进去什么内容」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

RISK_READ = "read"
RISK_WRITE = "write"
RISK_TRADE = "trade"

# 工具结果的长度上限。超了落盘、只回首尾预览 —— 一次全市场扫描、一张全表
# 因子值，序列化出来就是几 MB，直接塞进对话既看不了也白烧 token。
#
# `NO_LIMIT` 是给「自己已经限过长度」的工具用的（读文件、查手册）：它们再落盘
# 会造成 `读文件 → 存盘 → 再读` 的死循环。
DEFAULT_RESULT_CHARS = 8000
NO_LIMIT = -1

# 能力包。`default_on` 只影响新装用户，用户在设置页改过就以存盘为准。
#
# 除 `web`（会出网）和 `fs`（等于把代码交出去）之外**全部默认开** —— 这是
# 刻意选的向后兼容：这些动作以前就没有开关，突然收掉会让老用户的
# 「按策略下单」变成「能力包未开启」。要收紧就去设置页一个个关。
GROUPS: dict[str, dict] = {
    "data": {
        "label": "查数据",
        "desc": "查行情摘要、查因子数值、列可用品种",
        "default_on": True,
    },
    "manual": {
        "label": "平台手册",
        "desc": "查行情链路、原语源码、每个因子的完整实现",
        "default_on": True,
    },
    "doc": {
        "label": "读写文档",
        "desc": "读 Excel / PDF / Word，把结果导出成 PDF、xlsx",
        "default_on": True,
    },
    "factor": {
        "label": "因子",
        "desc": "注册新因子、查因子库、试算校验",
        "default_on": True,
    },
    "strategy": {
        "label": "策略",
        "desc": "注册新策略、查策略库",
        "default_on": True,
    },
    "backtest": {
        "label": "回测",
        "desc": "跑回测、看结果",
        "default_on": True,
    },
    "web": {
        "label": "联网",
        "desc": "抓网页正文。**会出网**，默认关",
        "default_on": False,
    },
    "trade": {
        "label": "下单",
        "desc": "下单、查持仓。**动真钱**，每次都要确认",
        "default_on": True,
    },
    "monitor": {
        "label": "盯盘",
        "desc": "扫一遍品种快照、找异动",
        "default_on": True,
    },
    "delegate": {
        "label": "派助手",
        "desc": "让只读的研究/复核助手跑一趟，只回结论（中间过程不进上下文）",
        "default_on": True,
    },
    "fs": {
        "label": "平台文件",
        "desc": "读写 ChanQuant 目录下的文件。**等于把代码交给模型**，默认关",
        "default_on": False,
    },
}

# 所有组的顺序（提示词里按这个顺序排，稳定输出便于命中缓存）
GROUP_ORDER = ["data", "manual", "doc", "factor", "strategy", "backtest",
               "web", "monitor", "delegate", "trade", "fs"]


@dataclass
class Tool:
    """一个可调用动作。字段名对齐 MCP 的 Tool 对象。"""

    name: str                      # 动作块里的 "action"，也是 MCP 的 name
    group: str                     # 所属能力包
    label: str                     # 给人看的中文名
    description: str               # 一行说明，进提示词目录（= MCP 的 description）
    inputSchema: dict = field(default_factory=dict)   # 参数校验（JSON Schema 子集）
    risk: str = RISK_READ
    handler: Callable | None = None    # (ctx, params) -> SkillResult；None 表示交给界面
    # schema 表达不了的规则（「mode=manual 时必须给 side」「path 不能有 ..」）
    # 走这里：`(params) -> (params, 错误)`。放在注册表里，`parse_action` 就不用
    # 再为每个动作写一个 `_parse_xxx` 分支。
    prepare: Callable | None = None
    ui: str = ""                   # 界面方法名后缀（`_run_<ui>`）；空表示纯逻辑工具
    self_confirm: bool = False     # 确认框由工具自己弹（`fs` 的读/写风险不同档）
    # 一行说明，给确认框用：`(payload) -> "买入开仓 RB_SHFE 2 手"`。
    # 没有就用 `label` 兜底 —— 确认框里至少得说清要干什么。
    describe: Callable | None = None
    prompt: str = ""               # 详细用法，只在包启用时注入
    max_turns: int = 1             # 同一个工具允许连续调几次（读类可以多）

    # ---- 下面这几项是照 Claude Code 的 `Tool` 契约补的，每项都有明确用处 ----
    #
    # `read_only` 默认从 `risk` 推导（read 就是只读）。显式声明是为了那些
    # 「风险是 read、但其实会写东西」的工具 —— 比如以后的「扫全市场」会落缓存。
    read_only: bool | None = None
    # 能否和别的工具同时跑。只读的能并行，写的不能 —— 两个写操作并发落盘
    # 会互相覆盖。默认从 `is_read_only()` 推导，要覆盖就显式写。
    concurrency_safe: bool | None = None
    # 用户在工具跑着的时候发了新消息，这个工具该被取消还是该跑完。
    # 长跑回测应该 `cancel`：用户改主意了，让它再算五分钟没有意义。
    interrupt: str = "block"
    # 改过名的工具把老名字挂这儿。老会话里的动作块存的是旧名字，改名后重放
    # 会「平台没有这个动作」—— 别名让它们继续能用。
    aliases: tuple[str, ...] = ()
    # 结果超过这么多字符就落盘，只回预览。`NO_LIMIT` 表示不裁。
    max_result_chars: int = DEFAULT_RESULT_CHARS
    # 3–10 个词的能力短语，将来做工具检索时给关键词匹配用。
    # 现在不参与提示词（全量注入反而更省），先把字段留出来。
    search_hint: str = ""

    def is_read_only(self) -> bool:
        """显式声明优先，没声明就看 risk。"""
        return self.risk == RISK_READ if self.read_only is None else bool(self.read_only)

    def is_concurrency_safe(self) -> bool:
        """只读的默认能并行；写的默认不能。"""
        return (self.is_read_only() if self.concurrency_safe is None
                else bool(self.concurrency_safe))

    def clips(self) -> bool:
        return self.max_result_chars != NO_LIMIT

    def spec(self) -> dict:
        """MCP `tools/list` 里的一项。

        `inputSchema` 要翻成**标准 JSON Schema** —— 内部写 `"str"` / `"int"`
        是为了 `validate()` 少打几个字，但这份 schema 是要发给外部 MCP 客户端的，
        类型名不合规对方会直接校验失败。
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": to_json_schema(self.inputSchema),
        }


@dataclass
class SkillResult:
    """工具跑完的结果。

    `blocks` 直接进对话区渲染（见 `ttrade/blocks.py`），`feed` 是回灌给模型的
    文本 —— 两样分开是因为模型要的是能读的字，人要的是能看的表。
    """

    ok: bool = True
    summary: str = ""
    blocks: list = field(default_factory=list)
    error: str = ""
    feed: str = ""

    def feed_text(self) -> str:
        """回灌给模型的文本。没单独给 `feed` 就用 `summary` 兜底。"""
        return (self.feed or self.summary or self.error or "").strip()


REGISTRY: dict[str, Tool] = {}

# 别名 → 正式名。工具改名后老动作块还能用（见 `Tool.aliases`）。
_ALIASES: dict[str, str] = {}


def register(tool: Tool) -> Tool:
    if tool.name in REGISTRY:
        raise RuntimeError(f"工具名重复：{tool.name}")
    if tool.group not in GROUPS:
        raise RuntimeError(f"未知能力包：{tool.group}")
    if tool.interrupt not in ("cancel", "block"):
        raise RuntimeError(f"interrupt 只能是 cancel / block：{tool.name}")
    REGISTRY[tool.name] = tool
    for a in tool.aliases:
        a = str(a).strip().lower()
        if a and a != tool.name:
            _ALIASES.setdefault(a, tool.name)
    return tool


def get(name: str) -> Tool | None:
    """按名字取工具。别名也算 —— 老会话里的动作块用的是改名前的名字。"""
    key = str(name or "").strip().lower()
    tool = REGISTRY.get(key)
    if tool is not None:
        return tool
    real = _ALIASES.get(key)
    return REGISTRY.get(real) if real else None


def names() -> list[str]:
    return list(REGISTRY)


def all_names() -> list[str]:
    """正式名 + 别名。给「模型把动作名写错」时的近似匹配用。"""
    return list(REGISTRY) + list(_ALIASES)


def group_tools(group: str) -> list[Tool]:
    return [t for t in REGISTRY.values() if t.group == group]


# ------------------------------------------------------------------ 参数校验

_TYPE_ALIASES = {
    "str": str, "string": str,
    "int": int, "integer": int,
    "float": float, "number": float,
    "bool": bool, "boolean": bool,
    "list": list, "array": list,
    "dict": dict, "object": dict,
}

# 内部简写 → 标准 JSON Schema 的 type 名
_JSON_TYPES = {
    "str": "string", "int": "integer", "float": "number", "bool": "boolean",
    "list": "array", "dict": "object",
}


def to_json_schema(schema: dict) -> dict:
    """内部简写类型 → 标准 JSON Schema。给 `tools/list` 用。

    只翻 `type`（含 `"str|int"` 这种并集，翻成 `anyOf`），其余字段
    （`default` / `enum` / `minimum` / `maximum` / `required` / `properties`）
    原样带过去 —— 它们本来就是标准写法。
    """
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str) and "|" in value:
            parts = [p.strip() for p in value.split("|") if p.strip()]
            out["anyOf"] = [{"type": _JSON_TYPES.get(p, p)} for p in parts]
        elif key == "type" and isinstance(value, str):
            out["type"] = _JSON_TYPES.get(value, value)
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {k: to_json_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            out["items"] = to_json_schema(value)
        else:
            out[key] = value
    return out


def _coerce(value, type_spec: str):
    """按 schema 里的类型串转一下。`"str|int"` 表示两种都行。"""
    for part in str(type_spec).split("|"):
        part = part.strip()
        py = _TYPE_ALIASES.get(part)
        if py is None:
            continue
        if py is bool:
            if isinstance(value, bool):
                return True, value
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return True, value.lower() == "true"
            if isinstance(value, (int, float)) and value in (0, 1):
                return True, bool(value)
            continue
        if py is int:
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return True, value
            if isinstance(value, float) and float(value).is_integer():
                return True, int(value)
            if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
                return True, int(value.strip())
            continue
        if py is float:
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return True, float(value)
            if isinstance(value, str):
                try:
                    return True, float(value.strip())
                except ValueError:
                    continue
            continue
        if py is str:
            if isinstance(value, str):
                return True, value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True, str(value)
            continue
        if py in (list, dict):
            if isinstance(value, py):
                return True, value
            continue
    return False, value


@dataclass
class FieldError:
    """参数不对。**给模型看**的错误，不是给用户看的。

    只回一句「参数 window 不对」，模型第二轮还是猜。带上 expected / got /
    hint，它一次就能改对。这也是 Claude Code 把 `validateInput`（给模型看）
    和 `checkPermissions`（给用户看）分成两个方法的意义 —— 混在一起会变成
    「参数写错了也弹一次确认框」。
    """

    tool: str = ""
    field: str = ""
    expected: str = ""
    got: str = ""
    hint: str = ""
    reason: str = ""          # 一句话总述：缺参数 / 类型不对 / 超范围 …

    def message(self) -> str:
        """一行，给日志和界面用。"""
        if not self.field:
            return self.reason or "参数不对"
        bits = [f"参数 {self.field}"]
        if self.reason:
            bits.append(self.reason)
        if self.expected and self.got:
            bits.append(f"期望 {self.expected}，收到 {self.got}")
        return f"{bits[0]}：" + "，".join(bits[1:]) if len(bits) > 1 else bits[0]

    def feedback(self) -> str:
        """回灌给模型的那一段。字段名写死成中文标签，模型读起来不用猜。"""
        out = ["【动作块有问题】"]
        if self.tool:
            out.append(f"工具：{self.tool}")
        if self.field:
            out.append(f"字段：{self.field}")
        if self.reason:
            out.append(f"原因：{self.reason}")
        if self.expected:
            out.append(f"期望：{self.expected}")
        if self.got:
            out.append(f"实际：{self.got}")
        if self.hint:
            out.append(f"改法：{self.hint}")
        return "\n".join(out)


def _describe_type(type_spec: str) -> str:
    return {
        "str": "字符串", "string": "字符串",
        "int": "整数", "integer": "整数",
        "float": "数字", "number": "数字",
        "bool": "布尔值", "boolean": "布尔值",
        "list": "数组", "array": "数组",
        "dict": "对象", "object": "对象",
    }.get(str(type_spec).strip(), str(type_spec))


def validate_detail(tool: Tool, params: dict) -> tuple[dict, FieldError | None]:
    """按 `inputSchema` 校验并补默认值。返回 `(干净参数, 结构化错误)`。

    只实现 JSON Schema 里真正用得到的子集：`type` / `required` / `default` /
    `enum` / `minimum` / `maximum`。**不引 jsonschema 库** —— 它 500KB 起步，
    而我们要的就是这五件事，写四十行比装一个库干净。
    """
    schema = tool.inputSchema or {}
    props = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    out: dict[str, Any] = {}

    for key in required:
        if params.get(key) in (None, ""):
            spec = props.get(key) or {}
            return {}, FieldError(
                tool=tool.name, field=key, reason="缺少这个必填参数",
                expected=_describe_type(spec.get("type") or "str"),
                hint=f"至少要给 {key}" + (
                    f"，可选值 {' / '.join(map(str, spec['enum']))}"
                    if spec.get("enum") else ""),
            )

    for key, spec in props.items():
        has = key in params and params[key] is not None
        if not has:
            if "default" in spec:
                out[key] = spec["default"]
            continue
        value = params[key]
        type_spec = spec.get("type") or "str"
        ok, value = _coerce(value, type_spec)
        if not ok:
            return {}, FieldError(
                tool=tool.name, field=key, reason="类型不对",
                expected=_describe_type(type_spec), got=repr(params[key]),
                hint=f"把 {key} 改成{_describe_type(type_spec)}",
            )
        if "enum" in spec and value not in spec["enum"]:
            allowed = " / ".join(map(str, spec["enum"]))
            return {}, FieldError(
                tool=tool.name, field=key, reason="不在可选范围内",
                expected=allowed, got=repr(value),
                hint=f"{key} 只能填 {allowed} 之一",
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if lo is not None and value < lo:
                return {}, FieldError(
                    tool=tool.name, field=key, reason="太小了",
                    expected=f"≥ {lo}", got=str(value),
                    hint=f"{key} 不能小于 {lo}",
                )
            if hi is not None and value > hi:
                return {}, FieldError(
                    tool=tool.name, field=key, reason="太大了",
                    expected=f"≤ {hi}", got=str(value),
                    hint=f"{key} 不能大于 {hi}",
                )
        out[key] = value

    # schema 里没声明的字段直接丢掉：模型偶尔会多塞一个 `"reason"` 之类的，
    # 留着只会让 handler 里的 `params.get` 读到意料外的东西
    return out, None


def validate(tool: Tool, params: dict) -> tuple[dict, str]:
    """`validate_detail` 的字符串版。老调用方（`dispatch`）不用改。"""
    clean, err = validate_detail(tool, params)
    return clean, (err.message() if err is not None else "")


# ------------------------------------------------------------------ 提示词目录

def catalog(enabled: list[str] | None = None, allow_fs: bool = False) -> str:
    """按启用状态拼一份工具目录，给模型看。

    只列 `name` + 一行说明 —— 详细用法（`prompt`）放在 `prompt_block()` 里，
    一起发会把提示词撑回五千字。
    """
    on = set(enabled if enabled is not None else default_groups())
    if allow_fs:
        on.add("fs")
    else:
        on.discard("fs")
    lines = ["## 当前可用的动作（`action` 填下面这些名字，没列出来的就是没有）"]
    for g in GROUP_ORDER:
        tools = [t for t in group_tools(g) if g in on]
        if not tools:
            continue
        meta = GROUPS.get(g, {})
        lines.append(f"\n### {meta.get('label', g)}（{g}）")
        for t in tools:
            risk = {"read": "", "write": "｜写操作，会弹确认",
                    "trade": "｜**动真钱**，会弹确认"}.get(t.risk, "")
            if t.group == "web":
                risk = "｜**会出网**，会弹确认"
            lines.append(f"- `{t.name}` {t.description}{risk}")
    return "\n".join(lines)


def prompt_block(enabled: list[str] | None = None) -> str:
    """启用包里的详细用法。只在有内容时返回非空串。"""
    on = set(enabled if enabled is not None else default_groups())
    parts = []
    for g in GROUP_ORDER:
        if g not in on:
            continue
        chunk = [t.prompt.strip() for t in group_tools(g) if t.prompt.strip()]
        if chunk:
            parts.append("\n\n".join(chunk))
    return "\n\n".join(parts)


def default_groups() -> list[str]:
    return [g for g, meta in GROUPS.items() if meta.get("default_on")]

# ------------------------------------------------------------------ 执行

@dataclass
class SkillContext:
    """handler 能拿到的东西。故意只有三样，别让它变成「什么都能干」。"""

    session: Any = None        # SessionFiles：附件资源缓存
    store: Any = None          # 主窗口的 DataStore，查行情用（`gui/app.py` 定义）
    cfg: Any = None            # AgentConfig，用来读开关（不 import，避免循环依赖）

    def allow(self, group: str) -> bool:
        # `fs` 单独一档：它由 `allow_fs` 管，不在能力包列表里 ——
        # 那个开关的语义是「把整个平台的代码交给模型」，跟「开一个能力包」不是一回事
        if group == "fs":
            return bool(getattr(self.cfg, "allow_fs", False))
        on = getattr(self.cfg, "enabled_groups", None)
        return group in (on if on is not None else default_groups())


def dispatch(ctx: SkillContext, name: str, params: dict) -> SkillResult:
    """跑一个工具。找不到 / 参数不对 / handler 抛异常都变成 `ok=False` 的结果 ——
    让模型看到原因，比让界面崩掉有用得多。"""
    tool = get(name)
    if tool is None:
        return SkillResult(False, error=f"平台没有名为 {name} 的动作")
    clean, why = validate(tool, params or {})
    if why:
        return SkillResult(False, error=why)
    if tool.prepare is not None:
        clean, why = tool.prepare(clean)
        if why:
            return SkillResult(False, error=why)
    if tool.handler is None:
        return SkillResult(False, error=f"{name} 只能由界面执行")
    try:
        res = tool.handler(ctx, clean)
    except Exception as exc:  # noqa: BLE001 - 工具出错不能让整轮对话崩
        return SkillResult(False, error=f"{type(exc).__name__}: {exc}")
    if not isinstance(res, SkillResult):
        return SkillResult(True, summary=str(res))
    return res
