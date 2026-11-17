"""**已废弃（2026-11-17）。** 策略引擎不再读写这个文件。

老设计：每条策略按「自己委托的成交」记一份落盘台账，用它跟目标仓比差额。
被砍掉的原因不是它算不准，而是**它会在没信号的时候发单**：程序一启动就把台账
读回来、立刻跟目标比、对不上就补 / 收窄 —— 用户看到的是「策略加载完成就自动
交易合约，发现仓位异常时单子已经发出去了」。理论与实际不一致，现在统一交给
「一键对齐」（`PortfolioRunner.align()`），由人决定什么时候动钱。

引擎现在用的是 `SlotState.sent`：内存里按合约记「本进程发过多少手」，不落盘、
不认领、不跟账户收窄。换月那一段还用它（老合约上的单是自己发的，自己收尾）。

本模块保留只为两件事：读老版本留下的 `live_positions.json` 做参考，
以及历史上「一键对齐」相关工具还在 import 它。**新代码不要再用。**
"""

from __future__ import annotations

import json
import os
import threading

from ttrade.paths import user_dir

POS_FILE = "live_positions.json"

_LOCK = threading.Lock()


def _path() -> str:
    return os.path.join(user_dir(), POS_FILE)


def exists() -> bool:
    """账本文件在不在。**只作参考**：引擎已经不看它了（见模块头），留这个函数
    是为了不把老的调用方一下打断。"""
    return os.path.exists(_path())


def _read() -> dict:
    try:
        with open(_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _write(data: dict) -> None:
    try:
        if not data:
            # 一条仓位都没有时不留空文件。`exists()` 是「本机已经启用过账本」的
            # 标志，留个空文件会让首次认领（见 `_adopt_if_needed`）永远跳过 ——
            # 那种情况下一启动就会在已有的仓上再开一遍。
            if os.path.exists(_path()):
                os.remove(_path())
            return
        os.makedirs(os.path.dirname(_path()), exist_ok=True)
        tmp = _path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, _path())
    except Exception:  # noqa: BLE001
        pass


def load(account: str) -> dict[str, dict[str, int]]:
    """{策略 id: {合约: 净仓}}。读不出来就当没有 —— 只是账本，不该阻断启动。"""
    with _LOCK:
        row = _read().get(account) or {}
    out: dict[str, dict[str, int]] = {}
    if not isinstance(row, dict):
        return out
    for sid, pos in row.items():
        if not isinstance(pos, dict):
            continue
        clean = {}
        for sym, n in pos.items():
            try:
                v = int(n)
            except (TypeError, ValueError):
                continue
            if v:
                clean[str(sym)] = v
        if clean:
            out[str(sid)] = clean
    return out


def save(account: str, sid: str, held: dict[str, int]) -> None:
    """整条策略的账本覆盖写。仓位归 0 的合约不写。"""
    clean = {str(s): int(n) for s, n in (held or {}).items() if int(n or 0)}
    with _LOCK:
        data = _read()
        row = data.get(account)
        if not isinstance(row, dict):
            row = {}
        if clean:
            row[sid] = clean
        else:
            row.pop(sid, None)
        if row:
            data[account] = row
        else:
            data.pop(account, None)
        _write(data)


def forget(account: str, sid: str) -> None:
    """删掉一条策略的账本。策略被删除时调 —— 留着只会在同 id 新建时误导。"""
    save(account, sid, {})
