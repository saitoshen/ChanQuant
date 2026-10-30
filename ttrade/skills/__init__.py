"""能力包。import 这个包就等于把所有工具注册进 `REGISTRY`。

## 加一个新工具要改几处

以前是五处：`parse_action` 的白名单、一个 `_parse_xxx` 做字段校验、
`Action.describe()` 里一个 `if`、`_handle_action()` 里一个 `if`，外加手写一段
提示词。现在是**一处**：在对应模块里写一个 `register(Tool(...))`。

- 纯逻辑（查数据、读文件、导出）：给 `handler`
- 要界面参与（下单、回测、注册因子）：给 `ui`，界面层按名字找 `_run_<ui>`
- 参数规则 schema 表达不了的：给 `prepare`

## 顺序有意义

`GROUP_ORDER` 决定提示词里各包的顺序。固定顺序才能命中服务端的上下文缓存 ——
同样的内容换个顺序，缓存就整段失效。
"""

from __future__ import annotations

# 空包会让 `ttrade.skills.REGISTRY` 是空的，所以这里必须真的 import 每个模块。
# 顺序按 GROUP_ORDER 排，方便对照。
from ttrade.skills import base  # noqa: F401
from ttrade.skills import data  # noqa: F401
from ttrade.skills import manual  # noqa: F401
from ttrade.skills import doc  # noqa: F401
from ttrade.skills import factor  # noqa: F401
from ttrade.skills import strategy  # noqa: F401
from ttrade.skills import trade  # noqa: F401
from ttrade.skills import web  # noqa: F401
from ttrade.skills import monitor  # noqa: F401
from ttrade.skills import fs  # noqa: F401

from ttrade.skills.base import (  # noqa: F401
    GROUP_ORDER,
    GROUPS,
    REGISTRY,
    RISK_READ,
    RISK_TRADE,
    RISK_WRITE,
    SkillContext,
    SkillResult,
    Tool,
    catalog,
    default_groups,
    dispatch,
    get,
    group_tools,
    names,
    prompt_block,
    register,
    to_json_schema,
    validate,
)

__all__ = [
    "GROUP_ORDER", "GROUPS", "REGISTRY", "RISK_READ", "RISK_TRADE", "RISK_WRITE",
    "SkillContext", "SkillResult", "Tool", "catalog", "default_groups", "dispatch",
    "get", "group_tools", "names", "prompt_block", "register", "to_json_schema",
    "validate",
]
