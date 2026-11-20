"""bar 时钟：一根 K 线到几点算走完。

这个模块只有一件事，但它是「实盘信号和回测对不上」这一类问题的共同根：

- 补库（`topup.save`）要把**没走完的最后一根**丢掉，否则半根被永久写进历史；
- 合并（`tqlive.merge_live`）要知道哪根还能被实时值覆盖；
- 算信号（`strategy_runner._refresh_signal`）要知道最后一根是活 bar 还是
  已经收盘 —— 切错了就少算一根，`signal` 和回测差一根，账户就多一手。

三处必须用**同一条口径**，各写各的一定会漂：`topup` 原来按
`index[-1] > now` 判断（天勤的时间戳是 bar 的**开始**时间，几乎永远 <= now，
于是半根从不被丢）；`_refresh_signal` 一律 `iloc[:-1]`（收盘后最后一根已经
走完了，还白切一根，日线策略永远少一天）。

## 判定规则

| 周期 | 一根的结束时刻 |
|---|---|
| 分钟级（1m~1h） | 起点 + 周期长度 |
| 2h / 4h | 起点 + 周期长度（由 1h 合成，边界对齐） |
| 日线 | 当日最后一个**不跨零点**交易时段的收盘（cu 15:00、IF 15:00） |

「跨零点的夜盘段」不能算进日线的结束时刻：它属于**下一个交易日**的行情，
把它算进来会让 11-19 那根日线一直「没走完」到第二天凌晨。
"""

from __future__ import annotations

import pandas as pd

from ttrade import symbols_info
from ttrade.tqfetch import PERIOD_SEC

#: 实时订阅能直接给的周期 → 秒。`PERIOD_SEC` 是下载口径（只有 5m/1h/1d），
#: 天勤的 `get_kline_serial` 还支持 15m/30m/2h/4h。
RT_SEC = {**PERIOD_SEC, "15m": 900, "30m": 1800, "2h": 7200, "4h": 14400}


def rt_sec(period: str) -> int:
    """这个周期一根多少秒（实时口径）。**认不出来返回 0**。

    🔴 表里只有**源周期**（5m / 1h / 1d）。天勤日线的键是 `1d_tq` —— 它和 `1d`
    是同一根 bar，只是数据源不同（见 `tqdata.PERIODS`）。以前这里直接
    `RT_SEC.get(period)`：`1d_tq` 查不到 ⇒ 返回 0 ⇒ `is_unfinished` 一律
    `False` ⇒ **日线策略从来不会切掉「今天这根没走完的日线」**。

    后果不是「慢一天」而是「拿半根算信号」：2026-11-20 14:09 白银那根日线还没
    收盘（当天 AG 从 10805 砸到 10026），分数被这半根从 +0.28 拉到 −0.56，
    空头信号从 0 翻回 −1，于是「图上空单已经出场、实盘还挂着 −2 手并且又去开空」。
    回测是「收盘定信号、下一根开盘成交」，实盘拿盘中价成交 —— 两边根本对不上。

    别名必须折到源周期（`tqdata.PERIODS` 是唯一的那张表，别在这里再抄一份）。
    `2d` / `1w` 是**由日线合成**的，故意不在 `PERIODS` 里：认不出来就返回 0，
    实时门禁自动挡住，不会退化成「按一天一根跑」。
    """
    p = str(period or "").strip()
    if not p:
        return 0
    hit = RT_SEC.get(p)
    if hit:
        return int(hit)
    try:
        from ttrade import tqdata          # 局部导入：barclock 被 topup/tqfetch 早期就拉起来
        src = (tqdata.PERIODS.get(p) or ("", 0))[0]
    except Exception:                      # noqa: BLE001
        src = ""
    return int(RT_SEC.get(src, 0) or 0) if src else 0


def _now(ts=None) -> pd.Timestamp:
    return pd.Timestamp(ts) if ts is not None else pd.Timestamp.now()


def _sessions(symbol: str, exchange: str):
    """交易时段 `[(开始分钟, 结束分钟), …]`，解析不出来返回空。"""
    try:
        return symbols_info.session_ranges(symbol, exchange)
    except Exception:                    # noqa: BLE001
        return []


def expected_last_bar(symbol: str, exchange: str, period: str, ts=None):
    """按交易时段算「此刻本该已经走完的最后一根 bar」的**起点**时间。

    为什么需要它：`hole_days` 只看帧**内部**有没有断口。帧**尾部**整体落后 ——
    本地库几天没更新、实时序列又还没回补完 —— 是查不出来的：合并后的最后一根
    和实时段的最后一根是同一根，索引看着完全连续。2026-11-19 的现场就是这样：
    cu 的 5m 库停在 11-17 14:50，天勤序列刚从 11-17 往回补（3001 根里只有 1 根
    比库新），合并出 220900 根、末根 11-17 14:55，在一份**两天前**的数据上算出
    signal=-1 给账户挂了 -1 手；而回测（库补齐后）末信号是 0。

    返回 `None` 表示「这条规则不适用」：日线及以上（一根就跨一天，按分钟判没
    意义）或交易时段解析不出来。调用方拿到 `None` 就不判 —— 宁可不判，不要误判。

    `ts` 可注入，便于自测；缺省取当前时刻。
    """
    sec = rt_sec(period)
    if sec <= 0 or sec >= 4 * 3600:      # 日线 / 2h 以上：不按分钟判
        return None
    segs = _sessions(symbol, exchange)
    if not segs:
        return None                      # 非期货 / 没配时段 ⇒ 不限制
    now = _now(ts)
    step = pd.Timedelta(seconds=sec)
    midnight = now.normalize()
    best = None
    for a, b in segs:
        # 跨零点的夜盘（21:00→02:30）结束时刻补到第二天
        end_min = b if b > a else b + 1440
        for off in (0, -1):              # 今天这段 / 昨天那段（隔夜延续到今晨）
            base = midnight + pd.Timedelta(days=off)
            start = base + pd.Timedelta(minutes=a)
            end = base + pd.Timedelta(minutes=end_min)
            if now < start:
                continue                 # 这一段还没开
            # 段内**已经走完**的根数：现在落在段中时，正在走的那根不算
            upto = min(now, end)
            n = int((upto - start) / step)
            if n <= 0:
                continue
            last = start + (n - 1) * step
            if best is None or last > best:
                best = last
    return best


def day_close(symbol: str, exchange: str, day) -> pd.Timestamp:
    """某个交易日的**收盘时刻**（日线那根的结束）。

    只取**日盘**时段（开始分钟 < 18:00）：夜盘 21:00→02:30 属于下一个交易日，
    算进今天会让今天的日线一直「没走完」到明天凌晨，日线策略就永远用不上今天。

    🔴 这里原来写的是「不跨零点的时段」（`b > a`），漏掉了**收在 23:00 的夜盘**
    （DCE / CZCE / GFEX：21:00→23:00 不跨零点）—— 于是棕榈油那根日线要等到
    23:00 才算走完：15:00 收盘不切、21:00 夜盘开盘还在用昨天的信号、等切过来
    已经 23:00 收盘了，委托只能压到第二天上午。**等于日线策略永远晚一天进场**，
    而回测是「收盘定信号、下一根开盘成交」。判据统一成「夜盘一律归次日」，
    与 `trading_days()` 的分组口径一致。
    """
    d = pd.Timestamp(day).normalize()
    segs = _sessions(symbol, exchange)
    # 1080 分钟 = 18:00：晚于它的时段一律算夜盘（归下一个交易日）
    ends = [b for a, b in segs if b > a and a < 1080]
    if not ends:
        return d + pd.Timedelta(days=1)
    return d + pd.Timedelta(minutes=max(ends))


def is_unfinished(start, symbol: str, exchange: str, period: str, ts=None) -> bool:
    """`start` 开头的那根 bar，此刻是不是**还在走**（OHLC 还会变）。

    这是「半根 bar」的唯一判据。半根写进历史后不会被任何一次增量更新修好
    （增量只追加更晚的），只能靠全量重下 —— 所以入口必须挡住。

    判不出来时返回 `False`（「不知道」不能当成「没走完」，那会把正常数据一直
    丢掉）。
    """
    sec = rt_sec(period)
    if sec <= 0:
        return False
    now = _now(ts)
    try:
        s = pd.Timestamp(start)
    except (TypeError, ValueError):
        return False
    if sec >= 86400:                     # 日线：收盘时刻由交易时段决定
        return now < day_close(symbol, exchange, s)
    return now < s + pd.Timedelta(seconds=sec)


def drop_unfinished(df: pd.DataFrame, symbol: str, exchange: str, period: str,
                    ts=None) -> pd.DataFrame:
    """丢掉尾部（可能连续多根）还没走完的 bar。

    只从**尾部**丢：中间那些「时间戳有、但内容可疑」的 bar 不动 —— 那是数据
    本身的另一类问题，拿时间判不了，硬丢会造出新的断口。
    """
    if df is None or len(df) == 0:
        return df
    n = len(df)
    cut = n
    while cut > 0 and is_unfinished(df.index[cut - 1], symbol, exchange, period, ts):
        cut -= 1
    return df.iloc[:cut] if cut < n else df


def tail_is_live(last, symbol: str, exchange: str, period: str, ts=None) -> bool:
    """序列最后那根是不是活 bar。`_refresh_signal` 用它决定要不要 `iloc[:-1]`。"""
    return is_unfinished(last, symbol, exchange, period, ts)
