"""把本地行情补到最新。

引擎订阅的 K 线本来就是实时的（`get_kline_serial` 会自己回填最近 3000 根），
策略算信号不缺数据。这个模块解决的是另一件事：**本地库停在几天前**。

组合页打开一条策略，K 线图画的是库里的历史 —— 库里最后一根是上次下载的那天，
图就一直停在那儿，看着像「策略没在跟行情」。所以策略一启动，顺手把引擎手上
这份实时序列写回库：下次打开、下次回测，数据就是最新的。

## 只按源周期落盘

库里的 15m / 30m / 2h / 4h 不是独立存的，是 5m / 1h 合并出来的（见
`tqdata.PERIODS`）。把 15m 的 bar 硬写进 15m 表，读的时候只会去找 5m，
白写一场 —— 界面还是「没有数据」。所以补库一律落到源周期上；而引擎订阅
本来就订的是源周期（见 `PortfolioRunner._subscribe`），这里直接复用那条
序列，不用另外再订一份。

## 正在走的那根不写

`get_kline_serial` 的最后一根是活 bar，OHLC 要到收盘才定。写进库之后再补数据
只会接「严格晚于库里的」，这根半成品就永久留在库尾，回测拿它当收盘价算信号。
所以凡是还没到结束时刻的 bar 一律丢掉。
"""

from __future__ import annotations

import pandas as pd

from ttrade import barclock, symbols_info, tqstore
from ttrade.tqdata import source_period, trading_days
from ttrade.tqlive import bars_from_serial

# 能落库的源周期。1d 不在这里：日线走另一条链路，别混进来
SOURCE_PERIODS = ("5m", "1h")


def source_of(period: str) -> str:
    """这个周期在库里应该落到哪个源周期上。"""
    return source_period(period)


def drop_live_tail(df: pd.DataFrame, symbol: str = "", exchange: str = "",
                   period: str = "5m", ts=None) -> pd.DataFrame:
    """去掉尾部还没走完的 bar（可能不止一根）。

    🔴 老写法是 `if df.index[-1] > now: df.iloc[:-1]` —— **从来没生效过**：
    天勤的 `datetime` 是 bar 的**开始**时间，10:30 那根的开始时间永远 <= 现在，
    判据为假，半根照写不误。2026-11-19 实测到库里 11-03 21:55 / 22:05 / 23:05、
    11-04 10:55 四根 cu 的 close 与天勤差 10~70 元、成交量只有一半 —— 就是那几天
    反复启动程序、每次把正在走的那根写进去留下的。它的危害不是「多一根」：
    这根半成品永久留在库尾，之后增量只追加更晚的，谁也修不好它，回测和实盘
    就永远差这一根的 high/low/close。

    现在按**结束时刻**判（`barclock.is_unfinished`），日线按当日收盘判。
    """
    if df is None or df.empty:
        return df
    if not symbol:
        # 没传品种（老调用方 / 自测）时退回「开始时间 + 周期 > 现在」的粗判，
        # 至少比 `index > now` 正确 —— 那个式子恒为假
        now = pd.Timestamp(ts) if ts is not None else pd.Timestamp.now()
        sec = barclock.rt_sec(period)
        if sec <= 0:
            return df
        step = pd.Timedelta(seconds=sec)
        cut = len(df)
        while cut > 0 and df.index[cut - 1] + step > now:
            cut -= 1
        return df.iloc[:cut] if cut < len(df) else df
    return barclock.drop_unfinished(df, symbol, exchange, period, ts)


def save(serial, symbol: str, exchange: str, period: str) -> dict:
    """把一份实时序列里**库里还没有**的部分写进 DuckDB。

    返回 `{"added": n, "src": "5m", "last": Timestamp|None, "error": "",
    "ok": bool}`。`ok=False` 表示这次没拿到可用数据（比如刚订阅、序列还没
    填充），调用方可以下一轮再试；出错只回 error，不抛 —— 补数据失败不该把
    策略打死。
    """
    src = source_of(period)
    out = {"added": 0, "src": src, "last": None, "error": "", "gap": "", "ok": False}
    if src not in SOURCE_PERIODS:
        out["error"] = f"{period} 不是可落库的源周期"
        out["ok"] = True          # 不重试：这是配置问题，不是「还没数据」
        return out
    try:
        df = bars_from_serial(serial, src)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if df is None or df.empty:
        return out
    df = drop_live_tail(df, symbol, exchange, src)
    if df.empty:
        return out
    # 🔴 交易时段门禁：天勤的模拟账户在非交易时段照样往序列里长 bar，直接落库
    # 就等于把「不存在的 bar」写进历史，回测以后都按它算 —— 这是那种要过很久
    # 才会被发现的污染。判据和实盘门禁是同一条（`symbols_info.in_session`），
    # 不在这儿另写一套时间判断。
    ok_open = [symbols_info.in_session(symbol, exchange, t) for t in df.index]
    if not all(ok_open):
        kept = df[ok_open]
        if kept.empty:
            # 一根合规的都没有 = 这份序列本身可疑（时区错位、时段表配错、或者
            # 里面全是收盘后的快照）。**不能当成功** —— 当成功就会把这个品种
            # 永久钉在旧数据上，而且一句话都不说。报错让调用方过几秒重试。
            out["error"] = f"{len(df)} 根全落在交易时段外，序列可疑（时区或时段表？）"
            return out
        df = kept
    try:
        last = tqstore.last_dt(symbol, exchange, src)
    except Exception:  # noqa: BLE001
        last = None
    if last is not None and df.index[0] > last:
        # 序列的**第一根**就比库里的最后一根晚 —— 中间那几天谁也不在：库里没有，
        # 这份序列也没有。追加下去就是把一段空洞贴成连续（`_signal_frame` 会
        # 据此判定缺交易日、不下单）。这里只负责把话说清楚。
        out["gap"] = (f"库停在 {last:%Y-%m-%d %H:%M}，"
                      f"序列从 {df.index[0]:%Y-%m-%d %H:%M} 才开始")
    if last is not None:
        # **库尾那一根也要写**（`>=`，不是 `>`）。它可能是上一次启动留下的半根：
        # 增量只追加更晚的，那一根就永远修不好（2026-11-19 在库里查到 4 根：
        # close 差 10~70 元、成交量只有一半）。实时这份是完整的，覆盖上去就自愈。
        #
        # 重叠段的 `adj_factor` 必须沿用库里那份：实时那份算出来多半是 1.0，
        # 盖过去会让前复权价在接缝处跳一下。
        df = df[df.index >= last]
    out["ok"] = True
    if df.empty:
        return out
    df = df.copy()
    df["trading_day"] = trading_days(df.index).to_numpy()
    try:
        with tqstore.connect() as con:
            # 因子不能一律写 1.0。切换表里要是已经有「待换月」（换月日进了表、
            # 但 K 线还没走到那一天），这一段就该乘上那次换月的比率 —— 写死 1.0
            # 会让复权价在换月处留一个假跳空（EC 实测 18%），而且要到下一次全量
            # 下载才自愈。按当前切换表算：没有待换月时结果仍然是 1.0，与老行为一致。
            from ttrade.tqfetch import factor_at, switch_series_via

            series = switch_series_via(con, symbol, exchange)
            df["adj_factor"] = (factor_at(series, pd.Series(df["trading_day"]))
                                if len(series) else 1.0)
            if last is not None:
                try:
                    keep = con.execute(
                        "SELECT dt, adj_factor FROM bars WHERE symbol=? AND "
                        "exchange=? AND period=? AND dt >= ?",
                        [symbol, exchange, src, df.index[0]]).fetchall()
                    if keep:
                        old = pd.Series(
                            {pd.Timestamp(dt): float(f or 1.0) for dt, f in keep})
                        hit = df.index.intersection(old.index)
                        if len(hit):
                            df.loc[hit, "adj_factor"] = old.loc[hit].to_numpy()
                except Exception:  # noqa: BLE001
                    pass
            d = df.reset_index().rename(columns={"datetime": "dt"})
            out["added"] = tqstore.upsert_bars(con, symbol, exchange, src, d)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"写库失败：{type(exc).__name__}: {exc}"
        return out
    out["last"] = df.index[-1]
    return out
