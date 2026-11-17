"""每条策略自己记的持仓，落盘。

为什么需要：账户是**一个**，策略是**一群**。同一个品种可能被好几条策略同时交易，
账户里的净仓是它们的和 —— 单看账户根本分不出哪几手是谁的。所以引擎不再拿账户
仓位当自己的仓位，改成按**自己发出去的委托成交了多少**来记：

    st.held["SHFE.rb2701"] = -11      # 这条策略自己空了 11 手

账户里多出来的仓位（手工开的、别的平台开的、别的策略开的）一律不碰。要对齐
账户，只有「一键对齐」一个入口，见 `PortfolioRunner.align()`。

为什么落盘：程序重启后内存里的账本没了，如果从 0 开始记，目标 -11 会再开一遍，
账户直接变 -22。落盘之后重启只是接着记。键是「账户 → 策略 id → 合约」，
策略 id 跟着组合文件走，所以删掉再加回来是同一条。

清仓（仓位归 0）的条目直接删掉，文件不会越积越大。
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
    """账本文件在不在。**第一次**启动（升级上来的老用户）要据此认领账户里已有的
    仓位，见 `PortfolioRunner._adopt_if_needed`。"""
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
