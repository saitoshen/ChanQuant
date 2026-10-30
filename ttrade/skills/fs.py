"""读写 ChanQuant 目录下的文件（`fs` 动作）。

## 为什么单独一个包、而且默认关

打开这个开关等于把整个平台的代码交给模型 —— 它能改 `ttrade/factors.py`、
能改启动脚本。所以它**不走**「能力包」那套默认值，由 `allow_fs` 单独管，
而且提示词里不提这件事：关着的时候模型根本不知道有这回事，也就不会写出一个
必然被拒的动作块。

## 确认策略：读不问、写必问

读一个文件是无副作用的，为它弹框只会训练用户无脑点确认；而一次 `write` 是
整文件覆盖，几十行代码说没就没。所以 `self_confirm=True`，由界面层的
`_run_fs` 自己决定弹不弹。
"""

from __future__ import annotations

from ttrade.skills.base import (
    RISK_WRITE,
    Tool,
    register,
)

OPS = ("read", "list", "write")
OP_CN = {"read": "读取", "list": "列出目录", "write": "写入"}


def prepare_fs(p: dict) -> tuple[dict, str]:
    """路径校验。真正的边界检查在 `agent_fs.resolve()` 里 —— 这里只挡住一眼
    就能看出来的错，让模型能立刻改对，不用等一次往返。"""
    op = str(p.get("op") or "").strip().lower()
    if op not in OPS:
        return p, "op 只能是 read / list / write"
    path = str(p.get("path") or "").strip().replace("\\", "/")
    if not path and op != "list":
        return p, "缺少 path"
    if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        return p, "path 只能是相对 ChanQuant 根目录的路径"
    if ".." in path.split("/"):
        return p, "path 里不能出现 .."
    if op == "write" and "content" not in p:
        return p, "写入要带 content（整文件覆盖）"

    p["op"] = op
    p["path"] = path or "."
    p["content"] = str(p.get("content") or "")
    return p, ""


def describe_fs(p: dict) -> str:
    op, path = p.get("op", ""), p.get("path", "")
    if op == "read":
        return f"读取 {path}"
    if op == "list":
        return f"列出目录 {path or '.'}"
    return f"写入 {path}（{len(p.get('content') or '')} 字符）"


register(Tool(
    name="fs",
    group="fs",
    label="平台文件",
    description="读写 ChanQuant 根目录下的文件（op=read/list/write）",
    risk=RISK_WRITE,
    inputSchema={
        "type": "object",
        "required": ["op"],
        "properties": {
            "op": {"type": "str"},
            "path": {"type": "str", "default": ""},
            # **不给默认值**：给了之后 `validate` 会把 `content` 补成空串，
            # `prepare_fs` 就分不出「模型压根没写正文」和「模型想清空文件」了
            "content": {"type": "str"},
        },
    },
    prepare=prepare_fs,
    ui="fs",
    self_confirm=True,
    describe=describe_fs,
))
