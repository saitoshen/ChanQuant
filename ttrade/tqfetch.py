"""天勤行情获取层：连接、切换表、复权、批量下载、真实合约。

从 `download_tq.py` 拆出来的。拆的理由不是「文件太长」，而是**界面要用**：
实时行情接上本地历史时，必须先补齐本地缺口，而补缺口要的就是这里的
「下载 + 按天勤切换表算复权因子」这一整套。界面不能反过来去 import 仓库根
目录下的一个脚本 —— 打包时它根本不在包里（PyInstaller 从 launcher.py 顺着
import 走，根目录脚本不在依赖链上）。

`download_tq.py` 现在只剩命令行入口，逻辑都在这儿。
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import tqstore as S
from .paths import data_dir
from .tqdata import source_period, trading_days
from .tqsdk_symbols import main_symbol, read_account, to_ts


PERIOD_SEC = {"5m": 300, "1h": 3600, "1d": 86400}

MAIN_YEARS = 10        # 主连默认回溯年数（天勤 5m 最早约 2016 年）
CONTRACT_YEARS = 10    # 真实合约默认回溯年数（与主连一致）

BATCH = 30             # query_his_cont_quotes 单次最多查多少个品种


# ----------------------------------------------------------------- 路径


def tq_root() -> str:
    """天勤数据根目录：`data_tq/`，与 `data_futures/` 平级。"""
    return os.path.join(os.path.dirname(data_dir("cn").rstrip("/\\")), "data_tq")


# ----------------------------------------------------------------- 天勤连接


def make_api():
    from tqsdk import TqApi, TqAuth

    user, pwd = read_account()
    return TqApi(auth=TqAuth(user, pwd))


def in_maintenance(now: datetime | None = None) -> bool:
    """天勤每天 19:00-19:30（北京时间）是运维窗口，这段时间连不上。"""
    now = now or datetime.now()
    return now.hour == 19 and now.minute < 31


def wait_maintenance() -> None:
    """撞上运维窗口就等到 19:31，别用重试硬刷 —— 刷再快也连不上。"""
    if not in_maintenance():
        return
    now = datetime.now()
    target = now.replace(hour=19, minute=31, second=30, microsecond=0)
    secs = max(5, (target - now).total_seconds())
    print(f"  [天勤运维窗口 19:00-19:30] 等待 {secs / 60:.0f} 分钟后继续", flush=True)
    time.sleep(secs)


class Session:
    """持有 TqApi，断线能重建。

    长任务（真实合约十年要跑几个小时）必然撞上运维窗口或偶发断连。天勤的 api
    一旦抛异常就不可再用，必须关掉重建；而重跑的成本靠库里的增量控制 ——
    所以这里的策略是「断开就重建，然后从这一条继续」，不是重试同一条请求。
    """

    def __init__(self) -> None:
        self.api = None

    def get(self):
        if self.api is None:
            wait_maintenance()
            self.api = make_api()
        return self.api

    def reset(self) -> None:
        try:
            if self.api is not None:
                self.api.close()
        except Exception:
            pass
        self.api = None

    def close(self) -> None:
        self.reset()


# ----------------------------------------------------------------- 主力切换表


def fetch_switch(api, pairs: list[tuple[str, str]], n: int = 2500) -> dict[str, pd.DataFrame]:
    """下载每个品种的「每日主力合约」，返回 {品种_交易所: DataFrame[date, underlying]}。

    `api.query_his_cont_quotes` 是天勤官方的历史主力映射，一列一个主连，
    值是该日主连对应的真实合约。这是唯一可靠的「天勤口径」切换信息。
    """
    out: dict[str, pd.DataFrame] = {}
    conts = {f"{sym}_{ex}": main_symbol(sym, ex) for sym, ex in pairs}
    keys = list(conts)

    def take(k: str, s: str, df: pd.DataFrame) -> None:
        if s in df.columns:
            sub = df[["date", s]].rename(columns={s: "underlying"}).dropna()
            if len(sub):
                out[k] = sub.sort_values("date").reset_index(drop=True)

    for i in range(0, len(keys), BATCH):
        chunk = keys[i : i + BATCH]
        syms = [conts[k] for k in chunk]
        print(f"  [切换表 {i + 1}-{i + len(chunk)}/{len(keys)}] {syms[0]} …", flush=True)
        try:
            df = api.query_his_cont_quotes(symbol=syms, n=n)
            for k, s in zip(chunk, syms):
                take(k, s, df)
            continue
        except Exception as exc:
            # 天勤会校验「这一批必须全是主连代码」，混进一个非法代码整批失败。
            # 本地品种池里确实有个别对不上的（比如 TL0），所以退化成逐个请求，
            # 让坏的那个单独失败，不要连累同一批的其他 29 个。
            print(f"    整批失败（{type(exc).__name__}），改为逐个重试")
        for k, s in zip(chunk, syms):
            try:
                df = api.query_his_cont_quotes(symbol=s, n=n)
                take(k, s, df)
            except Exception as exc:
                print(f"    {k} 失败：{type(exc).__name__}: {str(exc)[:80]}")
    return out


def switch_records(df: pd.DataFrame, sym: str, ex: str) -> pd.DataFrame:
    """把「每日主力」压成「切换记录」，并把天勤合约名转成 Tushare 命名。

    切换日 = 新主力生效的那一天。`ratio` 这里先留空，由 `fill_ratio()` 补，
    因为算 ratio 需要新旧合约当天的收盘价，得另外去取。
    """
    rows = []
    prev = None
    for _, r in df.iterrows():
        cur = r["underlying"]
        if prev and cur and cur != prev:
            d = pd.Timestamp(r["date"])
            try:
                rows.append(
                    {
                        "switch_date": d,
                        "old_tq": prev,
                        "new_tq": cur,
                        "old_ts": to_ts(prev, d),
                        "new_ts": to_ts(cur, d),
                    }
                )
            except Exception:
                # 解析不出来（比如主力是 888/000 这类伪合约）就跳过，不污染切换表
                pass
        if cur:
            prev = cur
    out = pd.DataFrame(rows, columns=["switch_date", "old_tq", "new_tq", "old_ts", "new_ts"])
    if len(out):
        out.insert(0, "symbol", sym)
        out.insert(1, "exchange", ex)
    return out


def fill_ratio(api, rec: pd.DataFrame, cache: dict[str, pd.DataFrame],
               period: str = "5m") -> pd.DataFrame:
    """给每条切换记录补 ratio = 切换时刻新主力价 / 旧主力价。

    取价方式按精度分两级，优先用第一级：

    1. **切换 bar 同刻价**：在 5m 数据里找到切换交易日的第 1 根 bar，取新旧合约
       在该时刻各自的收盘价。两个价格是同一瞬间的，算出来就是纯价差。
    2. **日线收盘价**（兜底）：切换日 15:00 的日线收盘之比。也是同刻的，但主连
       实际在当晚 21:00 就换了，15:00→21:00 的价差漂移会留下残差（螺纹实测
       最大 1.02%，用第一级能压到 0.60%）。

    这里固定用 5m 做取价基准（不是当前下载的周期）：切换发生在某一瞬间，
    5m 是本地最细的粒度，两个周期共用同一套 ratio 才不会互相打架。

    同一个合约会出现在多条记录里（连续几次换月的两侧），按合约缓存，
    一个品种大约 40 次请求。
    """
    if rec.empty:
        return rec

    def bars(tq_sym: str) -> tuple[pd.Series, pd.Series] | None:
        """返回 (收盘价序列, 所属交易日序列)，index 为北京时间。"""
        if tq_sym in cache:
            return cache[tq_sym]
        try:
            kl = api.get_kline_serial(tq_sym, PERIOD_SEC[period], data_length=10000)
            idx = local_time(kl["datetime"])
            s = pd.Series(kl["close"].astype(float).to_numpy(), index=idx)
            keep = idx > pd.Timestamp("1990-01-01")
            s = s[keep]
            td = pd.Series(trading_days(pd.Series(idx[keep])).to_numpy(), index=idx[keep])
            got = (s.dropna(), td)
        except Exception:
            got = None
        cache[tq_sym] = got
        return got

    def daily(tq_sym: str) -> pd.Series | None:
        key = ("D", tq_sym)
        if key in cache:
            return cache[key]
        try:
            kl = api.get_kline_serial(tq_sym, 86400, data_length=10000)
            # 日线的 datetime 是「交易日 00:00 **北京时间**」，也就是 UTC 的
            # 前一天 16:00。必须先换算时区再 normalize —— 直接对 UTC 值 normalize
            # 会让整条索引早一天，`d[d.index <= day]` 就取到了**下一个交易日**的
            # 收盘，比率整体错后一天（注释曾经写的是「交易日 00:00 UTC」，是错的）。
            idx = local_time(kl["datetime"]).normalize()
            d = pd.Series(kl["close"].astype(float).to_numpy(), index=idx)
            d = d[d.index > pd.Timestamp("1990-01-01")].dropna()
        except Exception:
            d = None
        cache[key] = d
        return d

    def price_at(tq_sym: str, day: pd.Timestamp) -> tuple[float | None, str]:
        got = bars(tq_sym)
        if got is not None:
            s, td = got
            hit = s.index[(td.to_numpy() == day.to_datetime64())]
            if len(hit):
                return float(s.loc[hit[0]]), "bar"
        d = daily(tq_sym)
        if d is not None and len(d):
            hit = d[d.index <= day]
            if len(hit):
                return float(hit.iloc[-1]), "daily"
        return None, ""

    ratios, old_c, new_c, how = [], [], [], []
    for _, r in rec.iterrows():
        day = pd.Timestamp(r["switch_date"]).normalize()
        po, mo = price_at(r["old_tq"], day)
        pn, mn = price_at(r["new_tq"], day)
        ok = po and pn and po > 0
        ratios.append(round(pn / po, 6) if ok else None)
        old_c.append(po)
        new_c.append(pn)
        how.append("bar" if mo == mn == "bar" else "daily")
    out = rec.copy()
    out["ratio"] = ratios
    out["old_close"] = old_c
    out["new_close"] = new_c
    out["price_src"] = how
    # 算不出 ratio 的记录直接丢掉：留着会让复权因子断成两截，
    # 表现为价格序列在某个换月点凭空缩放。
    return out.dropna(subset=["ratio"]).reset_index(drop=True)


def adj_factor_series(rec: pd.DataFrame) -> pd.Series:
    """由切换记录生成后复权因子，index=切换日，值=该日之后的价格段该乘的因子。

    从最新往回累计：`f[k-1] = f[k] * ratio[k]`，最新一段 f=1。

    **序列最前面会多一个哨兵点**（1990-01-01，值 = 全部比率的连乘），它代表
    「第一次换月之前」那一段。少了它，`factor_at` 对早于首条切换记录的 bar 会
    落到默认值 1.0 上，于是整段历史不参与复权 —— 实测 CJ 就是这样：数据从
    2019-01 开始，切换表首条是 2019-11-07，前半段因子停在 1.0、之后跳到 3.78，
    复权价格在 11-07 一根 bar 里凭空涨了 280%。

    哨兵用 1990-01-01 而不是 `Timestamp.min`：天勤最早的行情是 2016 年，1990
    足够早，同时留在 pandas 的可比较范围内，不引入 NaT。
    """
    if rec is None or rec.empty:
        return pd.Series(dtype=float)
    ratios = rec["ratio"].to_numpy(float)
    n = len(ratios)
    f = [1.0] * n
    for k in range(n - 2, -1, -1):
        f[k] = f[k + 1] * ratios[k + 1]
    idx = list(pd.to_datetime(rec["switch_date"]).to_numpy())
    idx.insert(0, np.datetime64("1990-01-01"))
    f.insert(0, f[0] * ratios[0])
    return pd.Series(f, index=pd.DatetimeIndex(idx))


def local_time(ns) -> pd.DatetimeIndex:
    """天勤 K 线的纳秒时间戳 → 北京时间（naive）。

    `get_kline_serial` 返回的是 **UTC** 纳秒，而 `DataDownloader` 写出的 CSV 里
    `datetime` 列是**北京时间**。两者混用会差 8 小时：夜盘 21:00 的 bar 在 UTC 里
    是 13:00，拿北京时间去索引一条都匹配不上，而且不会报错，只是静默返回空。

    天勤的 bar 时间戳是「bar 的结束时刻」，与 DataDownloader 的口径一致，
    所以这里只做时区换算，不加偏移。
    """
    idx = pd.to_datetime(pd.Series(ns).astype("int64"), unit="ns", utc=True)
    return pd.DatetimeIndex(idx.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None))


def factor_at(series: pd.Series, ts) -> "np.ndarray":
    """取每个时间点应乘的因子（<= 该时间的最近一个切换点的因子）。

    返回 ndarray 而不是 Series：调用方都是直接往 DataFrame 列里赋值，
    Series 会按 index 对齐，一旦 ts 的 index 不是 0..n-1 就会静默错位。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(ts)).values.astype("datetime64[ns]")
    out = np.ones(len(idx))
    if series is None or not len(series):
        return out
    sd = pd.DatetimeIndex(series.index).values.astype("datetime64[ns]")
    pos = sd.searchsorted(idx, side="right") - 1
    ok = pos >= 0
    if ok.any():
        out[ok] = series.to_numpy()[pos[ok]]
    return out


def switch_series_via(con, symbol: str, exchange: str) -> pd.Series:
    """在**已有的连接**上读切换表并算出因子序列。

    不调 `S.load_switch()`：它会另开一个只读连接，而调用方这时通常正持着写连接，
    DuckDB 的排他锁会让它直接报 `Cannot open file`。
    """
    df = con.execute(
        "SELECT switch_date, ratio FROM switch WHERE symbol=? AND exchange=?"
        " ORDER BY switch_date",
        [symbol, exchange],
    ).df()
    if df.empty:
        return pd.Series(dtype=float)
    df["switch_date"] = pd.to_datetime(df["switch_date"])
    return adj_factor_series(df)


def refresh_factors(con, symbol: str, exchange: str, period: str,
                    series: pd.Series | None = None) -> int:
    """按**当前**切换表重算该品种该周期全部 bar 的 `adj_factor`。

    `adj_factor` 是派生数据，切换表一变它就过期。写 K 线时算一遍是不够的：

    - 切换表后补 —— 写库那一刻表还是空的，整段因子停在 1.0（CJ、LH 实测如此，
      复权后换月处留 3% 假跳空）；
    - 来了新换月 —— 新 ratio 要乘到**全部历史**上，而增量下载只重写最近 3 天。

    所以这是个自愈动作：写完就刷，纯本地 UPDATE，不联网，5m 十年也就两千多个
    交易日要 join。
    """
    if series is None:
        series = switch_series_via(con, symbol, exchange)
    rows = con.execute(
        "SELECT DISTINCT trading_day FROM bars"
        " WHERE symbol=? AND exchange=? AND period=? AND trading_day IS NOT NULL",
        [symbol, exchange, period],
    ).fetchall()
    if not rows:
        return 0
    td = pd.DatetimeIndex([r[0] for r in rows])
    return S.set_adj_factor(con, symbol, exchange, period, td, factor_at(series, td))


def daily_close_via(con, symbol: str, exchange: str, period: str = "1h") -> pd.Series:
    """从库里聚合出「交易日 → 当日收盘」，用于判断换月点落在哪天。

    优先用 1h：它比 5m 早下完，而且对齐换月只需要日级精度。
    """
    df = con.execute(
        "SELECT trading_day, dt, close FROM bars"
        " WHERE symbol=? AND exchange=? AND period=? AND trading_day IS NOT NULL"
        " ORDER BY dt",
        [symbol, exchange, period],
    ).df()
    if df.empty:
        return pd.Series(dtype=float)
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    g = df.groupby("trading_day")["close"].last()
    return g.sort_index()


def realign_switch(con, symbol: str, exchange: str) -> int:
    """把切换日校正到「主连序列真正跳过去的那一天」，返回改动的记录数。

    **天勤的每日主力映射和主连 K 线偶尔差一个交易日**：`query_his_cont_quotes`
    说 LH 在 2022-04-18 就换成 lh2209 了，但 `KQ.m@LH` 的日线 04-18 还收 13400，
    04-19 才跳到 18030。27 次换月里有 20 次是这样，其余品种没这个问题。

    要复权的是**序列**，所以以序列为准。判据拿 ratio 当指纹：换月那天的原始涨跌
    必然等于新旧合约的同刻价比，错位的那天只是普通波动。

    阈值是「明显错位才挪」：当天误差 > 2% 且次日 < 1% 且次日误差不到当天的三分之一。
    这样**挪完必然收敛** —— 挪过去之后当天误差就等于原来的次日误差（<1%），
    不满足「> 2%」这一条，再跑也不会动。早先用过「次日比当天更贴就挪」，结果
    每跑一遍都往前挪一天，纯随机漂移。
    """
    rec = con.execute(
        "SELECT switch_date, ratio FROM switch WHERE symbol=? AND exchange=?"
        " ORDER BY switch_date",
        [symbol, exchange],
    ).df()
    if rec.empty:
        return 0
    closes = daily_close_via(con, symbol, exchange)
    if len(closes) < 3:
        return 0
    days = pd.DatetimeIndex(closes.index)
    vals = closes.to_numpy(float)
    used = set(pd.to_datetime(rec["switch_date"]))
    moved = 0

    for _i, r in rec.iterrows():
        ratio = r["ratio"]
        if not isinstance(ratio, (int, float)) or not np.isfinite(ratio) or ratio <= 0:
            continue
        d = pd.Timestamp(r["switch_date"])
        if d not in days:
            continue                      # 切换日不在序列里（含「待换月」的未来日）
        k = int(days.get_loc(d))
        if k == 0 or k + 1 >= len(days):
            continue
        here = abs(float(np.log(vals[k] / vals[k - 1] / ratio)))
        nxt = abs(float(np.log(vals[k + 1] / vals[k] / ratio)))
        target = days[k + 1]
        if here > 0.02 and nxt < 0.01 and nxt < here / 3 and target not in used:
            # `.fetchall()` 不能省：DuckDB 的 execute 是惰性的，UPDATE 的结果不消费
            # 就会一直挂在连接上，下一条 execute 直接报
            # `ResourceInUseException: connection has an unfinished result`。
            con.execute(
                "UPDATE switch SET switch_date=? WHERE symbol=? AND exchange=? AND switch_date=?",
                [target.date(), symbol, exchange, d.date()],
            ).fetchall()
            used.discard(d)
            used.add(target)
            moved += 1
    return moved


# ----------------------------------------------------------------- 行情下载


def _dl_path(out_dir: str, period: str, seq: int) -> str:
    """下载用的临时文件路径。**一批任务内每个任务独占一个，不删除。**

    `DataDownloader` 内部是 `open(csv_file_name, write_mode)`，`write_mode` 默认
    `"w"` —— 复用同一路径会自动截断，所以没必要每下完一个就删一次文件。
    这不只是省事：运行环境对「单次批量删除」有阈值保护，删到第 50 个就被拦下来
    把整轮下载打断（踩过两次）。清空是写操作，不受限制。

    **不能再按「8 个槽位循环复用」**：`is_finished()` 只反映协程退出，文件句柄
    的关闭发生在 `_run()` 的 `finally` 里，两者之间有一个窗口。槽位一还回去就
    立刻被下一个任务 `open(..., "w")`，撞上还没关的句柄时 open 直接失败，而
    tqsdk 的 `finally` 又用未定义的 `writer` 去 close，把真正的异常盖成
    `UnboundLocalError: cannot access local variable 'writer'`。实测就是这样
    静默丢掉 54 个品种的合约数据。

    现在按任务序号命名，一个任务一个文件，批内零复用。批数 × 每批合约数最多
    几百个小文件，跑完都是 0 字节，代价可以忽略。
    """
    return os.path.join(out_dir, f"_dl_{period}_{seq}.csv")


def _task_error(task) -> str:
    """取 `DataDownloader` 内部协程的异常。

    `is_finished()` 只是 `task.done()`，协程抛的异常被存在 Task 对象里、没人
    await 就永远不会浮出来。不主动去取，失败和「这个合约本来就没数据」在调用方
    看来完全一样。
    """
    inner = getattr(task, "_task", None)
    if inner is None or not inner.done():
        return ""
    if inner.cancelled():
        return "cancelled"
    exc = inner.exception()
    return "" if exc is None else f"{type(exc).__name__}: {exc}"


def download_many(api, jobs: list[tuple[str, date, date]], period: str,
                  out_dir: str, tag: str = "", max_parallel: int = 8,
                  errors: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """并发下载多个合约，返回 {天勤合约: DataFrame}。

    `DataDownloader` 是「建任务 + 靠 api.wait_update() 推进」的模式，多个任务可以
    同时挂在一个 api 上 —— 这才是真实合约下载的正确姿势。逐个 `while not
    task.is_finished()` 等于把网络往返完全串行化，实测一个品种十年要 11 分钟，
    78 个品种就是十几个小时。

    并发度取 8：再高对服务端不友好，收益也趋平（瓶颈从网络变成解析 CSV）。

    完成的任务**不立刻读**，先放进 `ready` 等下一轮 `wait_update()` 之后再读：
    `is_finished()` 只代表协程退出，文件句柄要到 `_run()` 的 `finally` 才关闭。
    中间那一拍足够 asyncio 把清理跑完，读到的才是完整文件。

    失败的合约写进 `errors`（合约 → 异常文本），不再静默当成「没数据」。
    """
    from tqsdk.tools import DataDownloader

    out: dict[str, pd.DataFrame] = {}
    queue = list(enumerate(jobs))            # (任务序号, (合约, 起, 止))
    running: list[tuple[int, str, object, str]] = []
    ready: list[tuple[int, str, str]] = []   # 已结束、等下一拍再读
    inflight = 0
    limit = max(1, int(max_parallel))
    used: list[str] = []
    try:
        while queue or running or ready:
            # 1) 消化上一拍结束的任务：此刻协程的 finally 已经跑完，文件已关闭
            for seq, sym, path in ready:
                df = _read_downloaded(path, sym)
                if not df.empty:
                    out[sym] = df
                inflight -= 1
            ready = []

            # 2) 补满并发
            while queue and inflight < limit:
                seq, (sym, start, end) = queue.pop(0)
                path = _dl_path(out_dir, period, seq)
                if path not in used:
                    used.append(path)
                task = DataDownloader(
                    api,
                    symbol_list=sym,
                    dur_sec=PERIOD_SEC[period],
                    start_dt=start,
                    end_dt=end,
                    csv_file_name=path,
                )
                running.append((seq, sym, task, path))
                inflight += 1

            if not (running or queue):
                break
            api.wait_update()

            # 3) 挑出这一拍结束的，下一轮再读
            still = []
            for seq, sym, task, path in running:
                if task.is_finished():
                    err = _task_error(task)
                    if err and errors is not None:
                        errors[sym] = err
                    ready.append((seq, sym, path))
                else:
                    still.append((seq, sym, task, path))
            running = still
    finally:
        # 只把内容清空（0 字节），不删文件 —— 下一批还会复用同一批路径
        for p in used:
            _truncate(p)
    return out


def _read_downloaded(path: str, tq_symbol: str) -> pd.DataFrame:
    """读 DataDownloader 写出的 CSV。读完只清空内容，不删文件（见 `_dl_path`）。"""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except Exception:
        _truncate(path)
        return pd.DataFrame()
    _truncate(path)
    if raw.empty:
        return pd.DataFrame()
    ren = {c: c[len(tq_symbol) + 1:] for c in raw.columns if c.startswith(tq_symbol + ".")}
    raw = raw.rename(columns=ren)
    keep = [c for c in ("datetime", "open", "high", "low", "close",
                        "volume", "open_oi", "close_oi") if c in raw.columns]
    if "datetime" not in keep:
        return pd.DataFrame()
    out = raw[keep].copy()
    out["dt"] = pd.to_datetime(out["datetime"])
    out = out.drop(columns=["datetime"]).dropna(subset=["close"])
    return out.sort_values("dt").drop_duplicates("dt", keep="last").reset_index(drop=True)


def _truncate(path: str) -> None:
    """把文件清成 0 字节。

    刻意不用 `os.remove()`：一轮跑下来要清几十次，运行环境对「单次批量删除」
    有阈值保护，删到第 50 个就会被拦下来把整轮下载打断。清空是写操作，
    不受这个限制，而且槽位下次复用本来就会被 `open(..., "w")` 截断。
    """
    try:
        if os.path.exists(path):
            with open(path, "w"):
                pass
    except OSError:
        pass


def _contract_ym(ts_code: str) -> tuple[int, int] | None:
    """从 Tushare 合约名取出 (年, 月)。`CU2601.SHF` → (2026, 1)。

    郑商所的 3 位月份在 `to_ts()` 里已经补成 4 位，所以这里统一按 4 位解析。
    """
    m = re.search(r"(\d{4})\.", str(ts_code))
    if not m:
        return None
    yymm = int(m.group(1))
    return 2000 + yymm // 100, yymm % 100


def _expired(ts_code: str, today: date) -> bool:
    """合约月份早于当前月，就当它已经到期。

    到期日可能是月份中旬（商品）或第三个周五（股指），这里不做精确判断 ——
    只用来决定「要不要跳过重下」，差几天无所谓：没到期的会被重下（便宜），
    真到期了再下一次也只是一次空请求。
    """
    ym = _contract_ym(ts_code)
    return bool(ym and ym < (today.year, today.month))


def write_contracts(api, sym: str, ex: str, period: str, years: int) -> dict:
    """下载一个品种的全部真实合约（不复权，单个合约本身没有换月跳空）。

    十年全品种是几千个合约、上亿行的量级，所以三件事必须做对：

    1. **按合约增量**：已到期的合约有数据就整段跳过，没到期的从最后一根往前推
       3 天续下。没有这层跳过，重跑一次等于重下全部。
    2. **并发下载**：见 `download_many()`。串行下一个品种十年要 11 分钟。
    3. **区间外与空合约也记账**：`query_quotes(expired=True)` 会把品种上市以来
       所有合约都返回，十年前就摘牌的那些整段都不需要；另外有些远月合约从挂牌
       到摘牌一笔没成交，每次跑都会白请求一次。前者按到期月份直接过滤，后者写进
       meta，下次跳过。
    """
    key = f"{sym}_{ex}"
    # 15m/30m 归到 5m、2h/4h 归到 1h：合约明细只有 5m/1h 两种，其余周期在读取时
    # 合并。不归一的话 `PERIOD_SEC[period]` 直接 KeyError
    period = source_period(period)
    end = date.today()
    start = end - timedelta(days=365 * years)

    try:
        conts = api.query_quotes(
            ins_class="FUTURE", exchange_id=ex, product_id=_tq_product_of(sym, ex), expired=True
        )
    except Exception as exc:
        return {"key": key, "period": period, "rows": 0, "status": f"查合约失败 {exc}"}

    have = S.contract_last_dt(sym, ex, period)
    empty = set(S.meta_get(f"empty::{key}::{period}", []) or [])
    todo: list[tuple[str, str, date]] = []   # (天勤合约, Tushare 合约, 起始日)
    skipped = old = blank = 0
    for c in conts:
        try:
            ts_code = to_ts(c, end)
        except Exception:
            ts_code = c  # 解析不了就保留天勤原名，至少不丢数据
        ym = _contract_ym(ts_code)
        if ym is not None and ym < (start.year, start.month):
            old += 1                      # 到期月份早于回溯区间，整段都不需要
            continue
        last = have.get(ts_code)
        if last is not None and (_expired(ts_code, end) or last.date() >= end - timedelta(days=5)):
            skipped += 1
            continue
        if last is None and ts_code in empty and _expired(ts_code, end):
            blank += 1                    # 摘牌时就没数据，不会再有
            continue
        cstart = start if last is None else max(start, (last - timedelta(days=3)).date())
        todo.append((c, ts_code, cstart))

    added = n_con = 0
    new_empty: list[str] = []
    failed: dict[str, str] = {}
    CHUNK = 24  # 每批 24 个合约、8 并发，跑 3 轮；批次太大内存里堆的 DataFrame 会多
    batches = [todo[i : i + CHUNK] for i in range(0, len(todo), CHUNK)]

    # 失败重试一轮：合约级失败基本是下载器偶发（连接抖动、文件句柄），换个批次
    # 再来一次大多能成。**不能静默当成「没数据」** —— 上一轮就是 54 个品种的
    # 合约全被吞掉，日志只写「新下 0 个合约」，看着像正常跳过。
    for attempt in (1, 2):
        retry: list[tuple[str, str, date]] = []
        for batch in batches:
            errs: dict[str, str] = {}
            frames = download_many(
                api, [(c, s, end) for c, _, s in batch], period, tq_root(), key,
                max_parallel=8, errors=errs,
            )
            # 整批共用一个写连接：一个合约一次 connect 的话，十年全品种要开关几千次，
            # 每次都要跑一遍 DDL，纯属浪费。
            with S.connect() as con:
                for c, ts_code, cstart in batch:
                    df = frames.get(c)
                    if df is not None and not df.empty:
                        df["contract"] = ts_code
                        added += S.upsert_contracts(con, sym, ex, period, df)
                        n_con += 1
                        failed.pop(c, None)
                        continue
                    if c in errs:
                        failed[c] = errs[c]
                        retry.append((c, ts_code, cstart))
                    elif _expired(ts_code, end):
                        new_empty.append(ts_code)
        if attempt == 2 or not retry:
            break
        print(f"    {key} {period}: {len(retry)} 个合约失败，重试一轮", flush=True)
        batches = [retry[i : i + CHUNK] for i in range(0, len(retry), CHUNK)]

    if new_empty:
        S.meta_set(f"empty::{key}::{period}", sorted(empty | set(new_empty)))

    if failed:
        status = f"{len(failed)} 个合约失败：{next(iter(failed.values()))[:70]}"
    elif added or skipped or old or blank:
        status = "ok"
    else:
        status = "无数据"
    return {"key": key, "period": period, "rows": added, "contracts": n_con,
            "skipped": skipped, "old": old, "blank": blank + len(new_empty),
            "failed": len(failed), "status": status}


def _tq_product_of(sym: str, ex: str) -> str:
    from ttrade.tqsdk_symbols import tq_product

    return tq_product(sym, ex)


# ----------------------------------------------------------------- 主连


def write_main_many(api, pairs: list[tuple[str, str]], period: str, years: int,
                    on_row=None, max_parallel: int = 8,
                    cancel=None) -> dict[str, int]:
    """批量下载主连并写库（原始价 + adj_factor，复权在读取时做）。

    一个品种十年 5m 串行要几分钟，78 个品种两个周期就是几小时。这里走 8 路并发，
    批次大小正好取一波 —— 断线最多丢一波，已经落库的品种靠 `S.last_dt()` 的
    增量跳过不会重下。

    `on_row(key, rows, first_dt)` 是进度回调：命令行用它打日志，界面用它刷状态。
    返回 `{品种_交易所: 本次写入行数}`。

    **`period` 会先归一成源周期**（`tqdata.source_period`）：天勤只下 5m 和 1h，
    15m/30m 是 5m 合并、2h/4h 是 1h 合并。传 15m 进来也只会去补 5m —— 这才和
    读取端对得上，也顺带避开 `PERIOD_SEC` 里没有 15m 的 KeyError。
    """
    period = source_period(period)
    end = date.today()
    hard_start = end - timedelta(days=365 * years)

    tail: list[tuple[str, date, date, str, str]] = []
    head: list[tuple[str, date, date, str, str]] = []
    factors: dict[tuple[str, str], pd.Series] = {}
    for sym, ex in pairs:
        rec = S.load_switch(sym, ex)
        factors[(sym, ex)] = adj_factor_series(rec) if len(rec) else pd.Series(dtype=float)
        tq = main_symbol(sym, ex)
        last = S.last_dt(sym, ex, period)
        first = S.first_dt(sym, ex, period)
        if last is None:
            tail.append((tq, hard_start, end, sym, ex))
            continue
        # 尾段：往前推 3 天重叠，覆盖「上次下到一半」和节假日边界；主键冲突会覆盖
        tail.append((tq, max(hard_start, (last - timedelta(days=3)).date()), end, sym, ex))
        # 头段：库里最早那根比回溯区间晚，说明前面那段从来没下过。早期版本用两年
        # 窗口跑过一轮，之后增量只看 last_dt，于是那几年永远补不上 —— RB 的 1h
        # 就这样停在 2024-10，每次日志还显示「+14 行」像正常增量。
        if first is not None:
            head_end = (first - timedelta(days=1)).date()
            if head_end > hard_start + timedelta(days=1):
                head.append((tq, hard_start, head_end, sym, ex))

    out: dict[str, int] = {}
    first_seen: dict[str, pd.Timestamp] = {}
    chunk = max(1, int(max_parallel))

    def run(batch: list[tuple[str, date, date, str, str]], label: str, report: bool) -> None:
        frames = download_many(
            api, [(t, s, e) for t, s, e, _, _ in batch], period, tq_root(),
            label, max_parallel=chunk,
        )
        with S.connect() as con:
            for tq_sym, _s, _e, sym, ex in batch:
                key = f"{sym}_{ex}"
                df = frames.get(tq_sym)
                if df is None or df.empty:
                    out.setdefault(key, 0)
                    if report and on_row is not None:
                        on_row(key, out[key], first_seen.get(key))
                    continue
                df["adj_factor"] = factor_at(factors[(sym, ex)], trading_days(df["dt"]))
                n = S.upsert_bars(con, sym, ex, period, df)
                # 上面那行只覆盖本次新下的 bar；切换表一变，历史段的因子也跟着变
                # （新 ratio 要乘到全部历史上）。这里整段重刷一遍，自愈。
                # 先对齐换月点：天勤的映射和序列偶尔差一个交易日，因子要跟序列走。
                realign_switch(con, sym, ex)
                refresh_factors(con, sym, ex, period)
                out[key] = out.get(key, 0) + n
                dt0 = df["dt"].iloc[0]
                if key not in first_seen or dt0 < first_seen[key]:
                    first_seen[key] = dt0
                if report and on_row is not None:
                    on_row(key, out[key], first_seen[key])

    # 头段先跑：补历史缺口。这一段不回调 —— 同一个品种稍后还会跑尾段，
    # 回调两次会让调用方的「第几个品种」计数翻倍。
    for i in range(0, len(head), chunk):
        if cancel is not None and cancel():
            return out
        run(head[i : i + chunk], f"MAIN_{period}_head", report=False)
    for i in range(0, len(tail), chunk):
        # 只在批次之间检查：一批 8 个品种正在飞，掐断它们只会留下一堆半截文件
        if cancel is not None and cancel():
            break
        run(tail[i : i + chunk], f"MAIN_{period}", report=True)
    return out


def switch_stamp(sym: str, ex: str) -> str:
    """切换表「上次算过」的元数据键。"""
    return f"switch_at::{sym}_{ex}"


def ensure_switch(api, pairs: list[tuple[str, str]], on_row=None) -> int:
    """给**还没有切换表**的品种补一张。返回补了几个。

    界面第一次补分钟线时必须先走这一步：`write_main_many` 用切换表算复权因子，
    表是空的，`adj_factor` 就会全是 1.0，价格看起来正常、复权全错。
    已有的不重算 —— 重算一个品种要请求几十次合约日线，78 个品种半小时。
    """
    todo = [(s_, e_) for s_, e_ in pairs if len(S.load_switch(s_, e_)) == 0]
    if not todo:
        return 0
    maps = fetch_switch(api, todo)
    n = 0
    for sym, ex in todo:
        key = f"{sym}_{ex}"
        df = maps.get(key)
        if df is None or len(df) < 2:
            continue
        rec = fill_ratio(api, switch_records(df, sym, ex), {}, "1d")
        with S.connect() as con:
            S.upsert_switch(con, sym, ex, rec)
        S.meta_set(switch_stamp(sym, ex), str(date.today()))
        n += 1
        if on_row is not None:
            on_row(key, len(rec))
    return n


def refresh_all(pairs: list[tuple[str, str]], on_msg=None) -> tuple[int, int]:
    """对齐换月点 + 按当前切换表重刷所有品种、所有已落库周期的复权因子。

    两步都必须做：

    1. **对齐**：天勤的每日主力映射和主连 K 线偶尔差一个交易日（LH 27 次换月里
       20 次如此），切换日要挪到序列真正跳空的那天；
    2. **重刷**：新 ratio 要乘到**全部历史**上，而 K 线表里的 `adj_factor` 只是
       上次写库时的快照。不刷的话切换表补全了历史价格却没动 —— CJ、LH 曾整段
       停在 1.0。

    纯本地计算，78 个品种秒级完成。返回 (校正的换月日数, 刷新的品种周期数)。
    """
    moved = touched = 0
    with S.connect() as con:
        for sym, ex in pairs:
            periods = sorted(S.periods_of(sym, ex, con=con))
            if not periods:
                continue
            k = realign_switch(con, sym, ex)
            if k:
                moved += k
                if on_msg is not None:
                    on_msg(f"{sym}_{ex}: 校正 {k} 个换月日")
            for period in periods:
                if refresh_factors(con, sym, ex, period):
                    touched += 1
    return moved, touched


def update_main(symbol: str, exchange: str, period: str = "5m",
                years: int = MAIN_YEARS, on_row=None) -> int:
    """补齐一个品种的历史，自己开会话、用完就关。

    界面在「选中品种 + 周期」之后调它把本地缺口补上，再交给实时推送。
    断线时抛异常由调用方决定要不要重试 —— 这里不吞。

    `period` 传目标周期即可（15m / 2h 都行），下载时按源周期补：1h 以下（不含
    1h）补 5m，1h 到日线之间补 1h，再由 `tqdata.load_main` 合并成目标周期。
    """
    sess = Session()
    try:
        got = write_main_many(sess.get(), [(symbol, exchange)], period, years, on_row)
    finally:
        sess.close()
    return int(got.get(f"{symbol}_{exchange}", 0))
