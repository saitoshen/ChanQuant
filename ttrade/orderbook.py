"""本程序下过、还没确认结束的委托号，落盘。

为什么需要：程序被强杀（或者直接关机）时 `close()` 跑不到，限价单会一直挂在
交易所。下次打开账户，用户什么都没点，委托却在成交 —— 那其实是上一次留下的。
所以每发一笔单就记下 order_id，下次连上账户时把这些还活着的单撤掉。

只撤**自己下过的**单，不碰手工单：快期客户端里手点的委托没有这里的 id。
"""

from __future__ import annotations

import json
import os
import threading

from ttrade.daylimit import trading_day
from ttrade.paths import user_dir

ORDER_FILE = "live_orders.json"

_LOCK = threading.Lock()


def _path() -> str:
    return os.path.join(user_dir(), ORDER_FILE)


def _read() -> dict:
    try:
        with open(_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _write(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_path()), exist_ok=True)
        tmp = _path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, _path())
    except Exception:  # noqa: BLE001
        pass


def _ids(data: dict, account: str) -> list[str]:
    """当天还挂着的委托号。跨交易日的直接丢掉 —— 那些单交易所已经作废了。"""
    row = data.get(account) or {}
    if row.get("day") != trading_day():
        return []
    return [str(x) for x in (row.get("ids") or []) if x]


def load(account: str) -> list[str]:
    with _LOCK:
        return _ids(_read(), account)


def remember(account: str, order_id: str) -> None:
    if not order_id:
        return
    with _LOCK:
        data = _read()
        ids = _ids(data, account)
        if order_id not in ids:
            ids.append(order_id)
        data[account] = {"day": trading_day(), "ids": ids[-1000:]}
        _write(data)


def forget(account: str, order_id: str) -> None:
    with _LOCK:
        data = _read()
        ids = _ids(data, account)
        if order_id in ids:
            ids.remove(order_id)
            data[account] = {"day": trading_day(), "ids": ids}
            _write(data)


def clear(account: str) -> None:
    with _LOCK:
        data = _read()
        if account in data:
            data.pop(account, None)
            _write(data)
