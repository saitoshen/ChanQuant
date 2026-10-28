"""日内报单 / 撤单次数限制。

国内期货交易所和期货公司对「日内报单笔数」「日内撤单笔数」都有硬上限：超了
轻则当天限制开仓、重则被交易所列入重点监控。天勤不替我们拦，只能自己数。

两条原则：

- **按账户、按交易日**。夜盘 21:00 起算的是**下一个**交易日，所以计数器不能按
  自然日切 —— 否则 23:59 用完 1000 次、过 0 点又是一条好汉，而交易所眼里还是
  同一个交易日。
- **落盘**。重启程序不清零。交易所记的是账户当天已经发出去多少笔，重启一下就
  绕过限制是自欺欺人。

计数只在真正要发报文时 +1（撤单同理），所以「算出来的手数是 0」「盘口还没来」
这些情况不会白吃额度。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta

from ttrade.paths import user_dir

# 交易所口径的硬上限。写 1000 是保守值：多数期货公司给客户的是这个量级，
# 具体数字各家不同，宁早停不要被限制。
DAY_ORDER_LIMIT = 1000
DAY_CANCEL_LIMIT = 500

LIMIT_FILE = "trade_limits.json"

_LOCK = threading.Lock()


def trading_day(now: datetime | None = None) -> str:
    """现在属于哪个交易日（YYYY-MM-DD）。

    夜盘 20:00 之后算**下一个**工作日：周五夜盘归下周一，周一凌晨 01:00 仍算
    周一。周末顺延。法定节假日不处理 —— 那些日子本来就不交易，计数用不上。
    """
    t = now or datetime.now()
    d = t.date()
    if t.hour >= 20:
        d += timedelta(days=1)
    while d.weekday() >= 5:          # 5 = 周六，6 = 周日
        d += timedelta(days=1)
    return d.isoformat()


def _path() -> str:
    return os.path.join(user_dir(), LIMIT_FILE)


def _read() -> dict:
    try:
        with open(_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - 文件坏了就当没有，不能因此挡住交易
        return {}


def _write(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_path()), exist_ok=True)
        tmp = _path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, _path())      # 原子替换，断电也不会留半个文件
    except Exception:  # noqa: BLE001
        pass


def _row(data: dict, account: str) -> dict:
    """取该账户当天的计数。跨交易日就归零 —— 但不写盘，读的时候顺手算。"""
    day = trading_day()
    row = data.get(account) or {}
    if row.get("day") != day:
        return {"day": day, "orders": 0, "cancels": 0}
    return {
        "day": day,
        "orders": int(row.get("orders", 0) or 0),
        "cancels": int(row.get("cancels", 0) or 0),
    }


def _bump(account: str, field: str, limit: int) -> tuple[bool, int]:
    """给某个计数 +1。额度已满则不加、返回 False。"""
    with _LOCK:
        data = _read()
        row = _row(data, account)
        used = int(row.get(field, 0) or 0)
        if used >= limit:
            data[account] = row
            _write(data)
            return False, used
        row[field] = used + 1
        data[account] = row
        _write(data)
        return True, used + 1


class DayLimit:
    """某个账户当天的报单 / 撤单额度。

    组合页的 `PortfolioRunner` 和账户页的 `LiveHub` 各持有一个实例，但读写的是
    同一份文件 —— 同一个账户在同一个进程里只能有一份计数，不然两边各数各的，
    加起来就超了。
    """

    def __init__(self, account: str) -> None:
        self.account = account or "default"

    def state(self) -> dict:
        with _LOCK:
            return _row(_read(), self.account)

    def order(self) -> bool:
        """记一笔报单。额度用尽返回 False，调用方**不能**再发。"""
        return _bump(self.account, "orders", DAY_ORDER_LIMIT)[0]

    def cancel(self) -> bool:
        """记一笔撤单。额度用尽返回 False。"""
        return _bump(self.account, "cancels", DAY_CANCEL_LIMIT)[0]

    def order_left(self) -> int:
        return max(0, DAY_ORDER_LIMIT - int(self.state()["orders"]))

    def cancel_left(self) -> int:
        return max(0, DAY_CANCEL_LIMIT - int(self.state()["cancels"]))

    def exhausted(self) -> bool:
        """报单或撤单额度已经用尽。界面据此把那一行标红。"""
        return self.order_left() <= 0 or self.cancel_left() <= 0

    def summary(self) -> str:
        s = self.state()
        return (f"今日报单 {s['orders']}/{DAY_ORDER_LIMIT} · "
                f"撤单 {s['cancels']}/{DAY_CANCEL_LIMIT}")
