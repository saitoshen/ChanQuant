"""把能力包吐成一个 MCP server（stdio + JSON-RPC 2.0）。

## 为什么这么短

注册表那层一开始就是照 MCP 的 `Tool` 对象建的（`name` / `description` /
`inputSchema`），`validate()` 又和 MCP 的参数校验是同构的。所以这里几乎没有
业务代码 —— 只做三件事：读一行 JSON、查注册表、把 `SkillResult` 拍成文本。

## 默认只放只读工具

外部客户端（另一个模型）拿到 `trade` 就等于拿到了下单权，中间没有任何人点确认。
所以：

| 工具 | 默认 | 理由 |
|---|---|---|
| `risk=read` | 暴露 | 查行情、查因子、扫全市场、查手册 |
| `risk=write` | 不暴露（`--allow-write` 才开） | 会落盘、会出网 |
| `risk=trade` | **永不暴露** | 动真钱，没有确认框就没有意义 |
| `handler=None` | 不暴露 | 要界面参与（回测、注册因子），命令行进程里跑不了 |
| `group=fs` | 不暴露 | 等于把平台代码交出去 |

## 用法

```bash
python -m ttrade.mcp_server              # 只读
python -m ttrade.mcp_server --allow-write
python -m ttrade.mcp_server --list       # 打印工具清单，不进协议循环
```

客户端配置（Claude Desktop / 任意 MCP 客户端）：

```json
{"mcpServers": {"trenddesk": {
  "command": "python", "args": ["-m", "ttrade.mcp_server"],
  "cwd": "D:/TrendDesk-Dev"}}}
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from ttrade import skills as S

SERVER_INFO = {"name": "trenddesk", "version": "1.0"}
# 客户端要的版本我们不一定实现过，回自己的；MCP 允许服务端选版本
PROTOCOL_VERSION = "2024-11-05"


# ------------------------------------------------------------------ 工具集
def exposed(allow_write: bool = False) -> list[S.Tool]:
    """这个进程允许外部客户端调用的工具。规则见模块开头那张表。"""
    out: list[S.Tool] = []
    for name in S.names():
        tool = S.get(name)
        if tool is None or tool.handler is None:
            continue
        if tool.group == "fs":
            continue
        if tool.risk == S.RISK_TRADE:
            continue
        if tool.risk == S.RISK_WRITE and not allow_write:
            continue
        out.append(tool)
    return out


_CTX: Any = None


def _ctx() -> S.SkillContext:
    """懒建上下文。

    行情库要读 CSV / DuckDB，第一次用才建 —— `tools/list` 不该等五秒。
    `QT_QPA_PLATFORM=offscreen` 必须在 import `gui.app` 之前设好，否则
    没有显示器的机器上 Qt 会直接退出。
    """
    global _CTX
    if _CTX is not None:
        return _CTX
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ttrade.store import AnyStore

    store = None
    try:
        from gui.app import DataStore

        store = AnyStore({"cn": DataStore("cn"), "us": DataStore("us")})
    except Exception as exc:          # noqa: BLE001
        _log(f"行情库没起来，只有不需要数据的工具能用：{exc}")
    _CTX = S.SkillContext(store=store)
    return _CTX


def _log(msg: str) -> None:
    """日志只能走 stderr —— stdout 是协议通道，混一行就废了。"""
    print(f"[trenddesk-mcp] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ 协议
def _result(rid, payload: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": payload}


def _error(rid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _call_tool(name: str, args: dict, allow_write: bool) -> dict:
    tool = next((t for t in exposed(allow_write) if t.name == name), None)
    if tool is None:
        return {"content": [{"type": "text", "text": f"没有可调用的工具 {name}"}],
                "isError": True}
    try:
        res = S.dispatch(_ctx(), name, args or {})
    except Exception as exc:          # noqa: BLE001
        return {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True}
    if not res.ok:
        return {"content": [{"type": "text", "text": res.error or "执行失败"}],
                "isError": True}
    text = res.feed_text() or res.summary or "（没有输出）"
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle(msg: dict, allow_write: bool = False) -> dict | None:
    """处理一条 JSON-RPC 消息。通知（没有 id）返回 None，不回包。"""
    rid = msg.get("id")
    method = msg.get("method") or ""
    params = msg.get("params") or {}

    if rid is None and method.startswith("notifications/"):
        return None
    if method == "initialize":
        return _result(rid, {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
    if method == "ping":
        return _result(rid, {})
    if method == "tools/list":
        return _result(rid, {"tools": [t.spec() for t in exposed(allow_write)]})
    if method == "tools/call":
        name = str(params.get("name") or "")
        args = params.get("arguments") or {}
        return _result(rid, _call_tool(name, args, allow_write))
    if rid is None:
        return None
    return _error(rid, -32601, f"不支持的方法：{method}")


def serve(fin, fout, allow_write: bool = False) -> None:
    """一行一条消息，读到底就退出。"""
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            fout.write(json.dumps(_error(None, -32700, f"解析失败：{exc}")) + "\n")
            fout.flush()
            continue
        try:
            reply = handle(msg, allow_write)
        except Exception as exc:      # noqa: BLE001
            reply = _error(msg.get("id"), -32603, f"{type(exc).__name__}: {exc}")
        if reply is None:
            continue
        fout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        fout.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ttrade.mcp_server",
                                 description="把 TrendDesk 能力包暴露成 MCP server")
    ap.add_argument("--allow-write", action="store_true",
                    help="连 write 级工具一起暴露（会落盘、会出网）")
    ap.add_argument("--list", action="store_true", help="打印工具清单后退出")
    ns = ap.parse_args(argv)

    if ns.list:
        for tool in exposed(ns.allow_write):
            print(f"{tool.name:16s} {tool.group:10s} {tool.label}　{tool.description}")
        return 0

    _log(f"就绪，暴露 {len(exposed(ns.allow_write))} 个工具"
         f"（{'含' if ns.allow_write else '不含'}写操作）")
    serve(sys.stdin, sys.stdout, ns.allow_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
