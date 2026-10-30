"""跨市场行情入口。

`DataStore` 一个市场一个实例（美股 / 国内期货），但 Agent 不跟市场走 ——
用户可能正在研究页看美股，却在 Agent 页问螺纹。所以能力包拿到的永远是这个
合成入口：查一个品种时两个市场都试，先命中的赢。

代号本身不重名（期货带交易所后缀 `RB_SHFE`，美股是纯代码 `AAPL`），所以
「先命中」不会歧义。

放在 `ttrade` 而不是 `gui`：MCP server 是纯命令行进程，不该为了一个字典
包装去 import 整个 Qt 界面。
"""

from __future__ import annotations


class AnyStore:
    """把若干 `DataStore` 合成一个 `store`（能力包只认这个接口）。"""

    def __init__(self, stores) -> None:
        self._stores = [s for s in (stores or {}).values() if s is not None]

    @property
    def items(self) -> list[dict]:
        out: list[dict] = []
        for s in self._stores:
            out.extend(getattr(s, "items", []) or [])
        return out

    @property
    def symbols(self) -> list[str]:
        return [it["key"] for it in self.items]

    def get(self, key: str, period: str = "1d"):
        for s in self._stores:
            if key in getattr(s, "symbols", []):
                return s.get(key, period)
        return None

    def has_period(self, key: str, period: str) -> bool:
        for s in self._stores:
            if key in getattr(s, "symbols", []):
                return s.has_period(key, period)
        return False
