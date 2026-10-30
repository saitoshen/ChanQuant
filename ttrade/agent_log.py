"""ChanAgent 的会话日志：append-only 的 JSONL，压缩只做「读时投影」。

## 为什么不能只靠内存里的 history

`_maybe_compress()` 把老对话压成记忆之后就把原始记录删了。结果是：**回测能
复盘，对话不能复盘**。而量化场景最需要复盘的恰恰是对话 —— 「模型当时看到了
什么数字，才决定下这一手」，事后必须查得到。

DeepSeek Harness 的做法是把「模型看到的一切」写进仅追加的事件流，恢复、分叉、
检索、回放全部共享同一份日志。这里照这个思路，但只留对交易有用的字段。

## 记什么

| 事件 | 内容 |
|---|---|
| `user` | 用户说的话（附件只记文件名，不记正文） |
| `assistant` | 模型的回答（超过 4000 字符只记开头，全文另存） |
| `action` | 解析后的动作块：工具名 + 参数 |
| `tool` | 工具结果摘要：原始字符数、回灌字符数、成功与否 |
| `confirm` | 确认框结果：同意 / 取消 |
| `compress` | 压缩事件：压了几条、存成几条记忆 |
| `error` | 出错的环节和原因 |

**永不改写已有行**。压缩只是让 `history` 里少了那几条，日志里一行不少 ——
`replay()` 能把某次会话按原样投影回来。

## 为什么是 JSONL 不是 SQLite

一次会话几千行，JSONL 追加是 O(1)、崩溃只丢最后一行、用记事本能直接看。
SQLite 要建表、要迁移、还要处理文件锁，这个规模用不上。

## 清理

跨会话的旧文件按天轮转：只留最近 `KEEP_FILES` 个。会话**内部**一行不删 ——
删的是「三个月前那次聊天」，不是「那次聊天里的某一句」。
"""

from __future__ import annotations

import json
import os
import threading
import time

from ttrade.paths import user_dir

DIR_NAME = "agent_sessions"
KEEP_FILES = 200
# 单条事件里长文本的截断长度。全文进日志会让一个文件几十 MB，
# 而真正要复盘的是「它看到了什么」，不是「那段代码的每一个字符」
MAX_TEXT = 4000

_lock = threading.Lock()
_sid = ""


def _new_sid() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex()


def session_id() -> str:
    """本次进程的会话编号。第一次调用时生成，之后固定。"""
    global _sid
    if not _sid:
        _sid = _new_sid()
    return _sid


def new_session() -> str:
    """开一段新会话（用户点「清空」时调）。旧日志原样留着。"""
    global _sid
    _sid = _new_sid()
    return _sid


def dir_path() -> str:
    path = os.path.join(user_dir(), DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def path(sid: str = "") -> str:
    return os.path.join(dir_path(), (sid or session_id()) + ".jsonl")


def _trim(text: str) -> str:
    text = "" if text is None else str(text)
    if len(text) <= MAX_TEXT:
        return text
    return text[:MAX_TEXT] + f"\n…（原 {len(text):,} 字符，此处截断）"


def log(ev: str, **fields) -> None:
    """追加一条事件。**任何异常都吞掉** —— 记日志失败不该让一次对话崩掉。"""
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "sid": session_id(), "ev": ev}
    for k, v in fields.items():
        rec[k] = _trim(v) if isinstance(v, str) else v
    try:
        line = json.dumps(rec, ensure_ascii=False) + "\n"
    except (TypeError, ValueError):
        rec = {k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v)
               for k, v in rec.items()}
        try:
            line = json.dumps(rec, ensure_ascii=False) + "\n"
        except (TypeError, ValueError):
            return
    try:
        with _lock:
            with open(path(), "a", encoding="utf-8") as fh:
                fh.write(line)
    except OSError:
        pass


def read(sid: str = "") -> list[dict]:
    """读一段会话的全部事件。坏行跳过，不因为一行坏了整段读不出来。"""
    p = path(sid)
    if not os.path.isfile(p):
        return []
    out: list[dict] = []
    try:
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return out
    return out


def sessions() -> list[dict]:
    """列出还留着的会话，最新的在前。给「历史会话」面板用。"""
    out: list[dict] = []
    try:
        names = [f for f in os.listdir(dir_path()) if f.endswith(".jsonl")]
    except OSError:
        return out
    for fn in names:
        p = os.path.join(dir_path(), fn)
        try:
            st = os.stat(p)
        except OSError:
            continue
        out.append({
            "sid": fn[:-6],
            "path": p,
            "bytes": st.st_size,
            "modified": st.st_mtime,
            "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
        })
    out.sort(key=lambda d: d["modified"], reverse=True)
    return out


def _rotate() -> None:
    """只留最近 KEEP_FILES 个文件。失败就算了。"""
    items = sessions()
    for old in items[KEEP_FILES:]:
        try:
            os.remove(old["path"])
        except OSError:
            continue


def replay(sid: str = "") -> list[dict]:
    """把一段会话投影回 messages 形态 —— 和当时发给模型的顺序一致。

    只投影 `user` / `assistant` 两类，动作和工具结果拼成一段注释附在后面：
    复盘要看的是「说了什么、做了什么」，不是把 JSON 原样摆出来。
    """
    out: list[dict] = []
    pending: list[str] = []
    for rec in read(sid):
        ev = rec.get("ev")
        if ev in ("user", "assistant"):
            if pending:
                out.append({"role": "note", "content": "\n".join(pending)})
                pending = []
            out.append({"role": ev, "content": str(rec.get("text") or "")})
        elif ev == "action":
            args = rec.get("params")
            pending.append(f"[动作] {rec.get('tool')} {json.dumps(args, ensure_ascii=False)}"
                           if isinstance(args, dict) else f"[动作] {rec.get('tool')}")
        elif ev == "tool":
            pending.append(f"[结果] {rec.get('tool')} "
                           f"{'成功' if rec.get('ok') else '失败'} "
                           f"{rec.get('sent', 0)} 字符")
        elif ev == "confirm":
            pending.append(f"[确认] {rec.get('tool')} → "
                           f"{'同意' if rec.get('ok') else '取消'}")
        elif ev == "error":
            pending.append(f"[出错] {rec.get('where')}：{rec.get('why')}")
    if pending:
        out.append({"role": "note", "content": "\n".join(pending)})
    return out


def start() -> None:
    """进程启动时调一次：确保目录在、顺便轮转旧文件。"""
    dir_path()
    _rotate()
