"""从天勤（TqSdk）下载国内期货 5 分钟 / 60 分钟行情。

与 `download.py`（Tushare 日线）并列的第三条数据链路。三者的分工：

| 脚本 | 数据源 | 周期 | 用途 |
|---|---|---|---|
| `download.py` | Tushare | 日线 | 策略研发的基准数据 |
| `download_aux.py` | Tushare | 真实合约日线 | 期限结构 / 期权因子 |
| `download_tq.py` | 天勤 | 5m / 1h | 多周期回测、实时行情基底 |

**为什么必须用天勤自己的主力切换表**：Tushare 的 `fut_wsr`（主力切换）按持仓量
规则判定，天勤主连（`KQ.m@`）用的是自己的规则，两者换月日期经常差几天。拿 Tushare
的切换日期去复权天勤的连续合约，切换点会错位，价格序列在换月附近出现假跳空 ——
所以切换表和复权因子都在这里从天勤现取。

**复权口径**（与 `D:\\CozeTask\\future_quant` 里 sbbox / RIA 策略一致）：

    ratio_k    = 切换日新主力收盘 / 旧主力收盘（同日价差，不含跨日涨跌）
    factor(t)  = ∏(ratio_k for 切换日 > t)      ← 后复权，最新段因子恒为 1
    adj_price  = raw_price × factor(t)

后复权而不是前复权，是因为增量更新时前复权的整条历史都会变；后复权只影响切换点
之后新增的部分。展示时再除以最后一根因子即可（`load_local` 的 qfq 就是这么做的）。

用法：

    python download_tq.py --check                    # 只看账户与权限，不下载
    python download_tq.py --switch                   # 只更新主力切换表
    python download_tq.py --main 5m,1h --symbols RB,HC,I
    python download_tq.py --main 5m,1h --all         # 全部品种（慢）
    python download_tq.py --contracts 5m --years 3 --all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import paths  # noqa: E402
from ttrade import tqstore as S  # noqa: E402
from ttrade.futures import EXCHANGES, list_local  # noqa: E402
from ttrade.tqsdk_symbols import (  # noqa: E402
    main_symbol,
    read_account,
    to_ts,
)

# 周期名 → 秒数。天勤的 dur_sec 支持任意秒数，但历史服务只对
# 5m/15m/30m/1h/2h/4h/日 这些整点对齐的周期保证连续，所以只下这两种，
# 其余周期在本地 resample（数据量小一个量级，且不会被服务端规则变化影响）。
PERIOD_SEC = {"5m": 300, "1h": 3600, "1d": 86400}

MAIN_YEARS = 10        # 主连默认回溯年数（天勤 5m 最早约 2016 年）
CONTRACT_YEARS = 10    # 真实合约默认回溯年数（与主连一致）

BATCH = 30             # query_his_cont_quotes 单次最多查多少个品种


# ----------------------------------------------------------------- 路径


def tq_root() -> str:
    """天勤数据根目录：`data_tq/`，与 `data_futures/` 平级。"""
    return os.path.join(os.path.dirname(paths.data_dir("cn").rstrip("/\\")), "data_tq")


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
            # 日线的 datetime 是「交易日 00:00 UTC」，normalize 后正好是交易日日期。
            # 不能套 local_time()：换成北京时间会变成当天 08:00，和「<= 切换日 00:00」
            # 比较时会漏掉切换日自己，整体错后一天。
            d = pd.Series(kl["close"].astype(float).to_numpy(),
                          index=pd.DatetimeIndex(pd.to_datetime(kl["datetime"])).normalize())
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
    """
    if rec is None or rec.empty:
        return pd.Series(dtype=float)
    ratios = rec["ratio"].to_numpy(float)
    n = len(ratios)
    f = [1.0] * n
    for k in range(n - 2, -1, -1):
        f[k] = f[k + 1] * ratios[k + 1]
    return pd.Series(f, index=pd.to_datetime(rec["switch_date"]).to_numpy())


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


def trading_days(dt: pd.Series) -> pd.Series:
    """把 K 线时间戳映射到**交易日**（复权因子的对齐基准）。

    这是复权正确与否的关键。天勤的主连在「新交易日的第一个 bar」切换，而期货的
    交易日从**前一晚 21:00 的夜盘**就开始了：

        2026-09-01 21:00  ← 这是 2026-09-02 交易日的第 1 根，已经属于新主力
        2026-09-01 22:00
        2026-09-02 09:00

    如果按自然日对齐因子（`datetime.date`），前两根夜盘 bar 会拿到旧因子、日盘 bar
    拿到新因子，等于在 22:00 和次日 09:00 之间凭空插了一次缩放 —— 实测这样会把
    换月日的平均跳空从 0.20% 放大到 0.41%，最大跳空从 0.85% 放大到 2.91%。

    规则：20:00 及以后的 bar 属于「下一个有日盘的日期」，其余 bar 属于当天。
    """
    ts = pd.to_datetime(dt)
    day_session = pd.DatetimeIndex(sorted(set(ts[ts.dt.hour < 20].dt.normalize())))
    dates = ts.dt.normalize().to_numpy()
    out = pd.Series(dates, index=ts.index)
    night = (ts.dt.hour >= 20).to_numpy()
    if night.any() and len(day_session):
        pos = day_session.values.searchsorted(dates[night], side="right")
        nxt = np.where(pos < len(day_session), day_session.values[np.clip(pos, 0, len(day_session) - 1)],
                       dates[night])
        out.iloc[np.flatnonzero(night)] = nxt
    return out


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


# ----------------------------------------------------------------- 行情下载


_dl_seq = 0


def download_many(api, jobs: list[tuple[str, date, date]], period: str,
                  out_dir: str, tag: str, max_parallel: int = 8) -> dict[str, pd.DataFrame]:
    """并发下载多个合约，返回 {天勤合约: DataFrame}。

    `DataDownloader` 是「建任务 + 靠 api.wait_update() 推进」的模式，多个任务可以
    同时挂在一个 api 上 —— 这才是真实合约下载的正确姿势。逐个 `while not
    task.is_finished()` 等于把网络往返完全串行化，实测一个品种十年要 11 分钟，
    78 个品种就是十几个小时。

    并发度取 8：再高对服务端不友好，收益也趋平（瓶颈从网络变成解析 CSV）。

    临时文件名用自增序号而不是 `hash(sym)`：`hash()` 在同进程内对同一字符串稳定，
    但不同合约理论上可能撞同一个桶，撞上就是两个任务往同一个 CSV 里写、数据串掉，
    而且这种错误不会报错。序号零成本且绝对唯一。
    """
    from tqsdk.tools import DataDownloader

    global _dl_seq
    out: dict[str, pd.DataFrame] = {}
    queue = list(jobs)
    running: list[tuple[str, object, str]] = []
    paths: list[str] = []
    try:
        while queue or running:
            while queue and len(running) < max_parallel:
                sym, start, end = queue.pop(0)
                _dl_seq += 1
                path = os.path.join(out_dir, f"_dl_{tag}_{_dl_seq}.csv")
                paths.append(path)
                task = DataDownloader(
                    api,
                    symbol_list=sym,
                    dur_sec=PERIOD_SEC[period],
                    start_dt=start,
                    end_dt=end,
                    csv_file_name=path,
                )
                running.append((sym, task, path))
            api.wait_update()
            still = []
            for sym, task, path in running:
                if task.is_finished():
                    df = _read_downloaded(path, sym)
                    if not df.empty:
                        out[sym] = df
                else:
                    still.append((sym, task, path))
            running = still
    finally:
        # 断线时 wait_update 会抛异常，此时还在跑的任务文件留在磁盘上。
        # 不清就会在 data_tq 里堆一堆几百 KB 的 _dl_*.csv，越跑越多。
        for p in paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
    return out


def _read_downloaded(path: str, tq_symbol: str) -> pd.DataFrame:
    """读 DataDownloader 写出的 CSV，去掉合约前缀列名，返回 dt 为北京时间的表。"""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
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


def write_main_all(sess, pairs: list[tuple[str, str]], period: str, years: int,
                   with_retry) -> None:
    """批量下载主连并写库（原始价 + adj_factor，复权在读取时做）。

    一个品种十年 5m 串行要几分钟，78 个品种两个周期就是几小时。这里和真实合约
    一样走 8 路并发，批次大小正好取一波 —— 断线最多丢一波，而且已经落库的品种
    靠 `S.last_dt()` 的增量跳过不会重下。
    """
    end = date.today()
    hard_start = end - timedelta(days=365 * years)

    jobs: list[tuple[str, date, date, str, str]] = []
    factors: dict[tuple[str, str], pd.Series] = {}
    for sym, ex in pairs:
        rec = S.load_switch(sym, ex)
        factors[(sym, ex)] = adj_factor_series(rec) if len(rec) else pd.Series(dtype=float)
        start = hard_start
        have = S.last_dt(sym, ex, period)
        if have is not None:
            # 往前推 3 天重叠，覆盖「上次下到一半」和节假日边界；主键冲突会覆盖
            start = max(start, (have - timedelta(days=3)).date())
        jobs.append((main_symbol(sym, ex), start, end, sym, ex))

    total = len(jobs)
    done = 0
    CHUNK = 8
    for i in range(0, total, CHUNK):
        batch = jobs[i : i + CHUNK]

        def job(api, batch=batch):
            nonlocal done
            frames = download_many(
                api, [(t, s, e) for t, s, e, _, _ in batch], period, tq_root(),
                f"MAIN_{period}", max_parallel=8,
            )
            with S.connect() as con:
                for tq_sym, _s, _e, sym, ex in batch:
                    done += 1
                    df = frames.get(tq_sym)
                    if df is None or df.empty:
                        print(f"  [{period} {done}/{total}] {sym}_{ex}: 无新数据", flush=True)
                        continue
                    df["adj_factor"] = factor_at(factors[(sym, ex)], trading_days(df["dt"]))
                    n = S.upsert_bars(con, sym, ex, period, df)
                    print(f"  [{period} {done}/{total}] {sym}_{ex}: +{n} 行"
                          f"（{df['dt'].iloc[0]:%Y-%m-%d} 起）", flush=True)
            return True

        with_retry(f"主连 {period} 第 {i // CHUNK + 1} 批", job)


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
    CHUNK = 24  # 每批 24 个合约、8 并发，跑 3 轮；批次太大内存里堆的 DataFrame 会多
    for i in range(0, len(todo), CHUNK):
        batch = todo[i : i + CHUNK]
        frames = download_many(
            api, [(c, s, end) for c, _, s in batch], period, tq_root(), key, max_parallel=8
        )
        # 整批共用一个写连接：一个合约一次 connect 的话，十年全品种要开关几千次，
        # 每次都要跑一遍 DDL，纯属浪费。
        with S.connect() as con:
            for c, ts_code, _ in batch:
                df = frames.get(c)
                if df is None or df.empty:
                    if _expired(ts_code, end):
                        new_empty.append(ts_code)
                    continue
                df["contract"] = ts_code
                added += S.upsert_contracts(con, sym, ex, period, df)
                n_con += 1
    if new_empty:
        S.meta_set(f"empty::{key}::{period}", sorted(empty | set(new_empty)))

    return {"key": key, "period": period, "rows": added, "contracts": n_con,
            "skipped": skipped, "old": old, "blank": blank + len(new_empty),
            "status": "ok" if (added or skipped or old or blank) else "无数据"}


def _tq_product_of(sym: str, ex: str) -> str:
    from ttrade.tqsdk_symbols import tq_product

    return tq_product(sym, ex)


# ----------------------------------------------------------------- 入口


def pick_pairs(symbols: str | None, all_: bool) -> list[tuple[str, str]]:
    if all_:
        items = list_local(paths.data_dir("cn"))
        pairs = [(it["symbol"], it["exchange"]) for it in items]
    else:
        want = [s.strip().upper() for s in (symbols or "").split(",") if s.strip()]
        if not want:
            raise SystemExit("请用 --symbols RB,HC 指定品种，或用 --all")
        items = {it["symbol"]: it for it in list_local(paths.data_dir("cn"))}
        pairs = []
        for w in want:
            if w not in items:
                print(f"  [!] {w} 不在本地品种池里，跳过")
                continue
            pairs.append((w, items[w]["exchange"]))
    return pairs


def _switch_stamp(sym: str, ex: str) -> str:
    return f"switch_at::{sym}_{ex}"


def _fresh_switch(sym: str, ex: str, days: int = 1) -> bool:
    """切换表是不是最近算过。

    重算一个品种要请求几十次合约日线（约 20 秒），78 个品种就是半小时，所以
    断线重跑时必须能跳过。这里用「上次计算时间」而不是「最后一次换月日期」：
    低流动性品种可能两个月才换一次月，按后者判断会把它们每次重跑都重算一遍
    —— 实测 78 个品种里有 28 个会中招。

    默认窗口 1 天：同一天内重跑一律跳过，跨天重算一次。想立刻刷新用
    `--force-switch`。
    """
    ts = S.meta_get(_switch_stamp(sym, ex))
    if ts:
        try:
            return (pd.Timestamp.today().normalize() - pd.Timestamp(str(ts))).days < days
        except (ValueError, TypeError):
            pass
    if len(S.load_switch(sym, ex)) == 0:
        return False
    # 老数据没有时间戳（加这一列之前算的），补一个今天的，之后就走上面的日期判断
    S.meta_set(_switch_stamp(sym, ex), str(date.today()))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="从天勤下载 5m/1h 行情到 DuckDB")
    ap.add_argument("--check", action="store_true", help="只检查账户与权限")
    ap.add_argument("--switch", action="store_true", help="更新主力切换表")
    ap.add_argument("--main", default="", help="下载主连，周期逗号分隔：5m,1h")
    ap.add_argument("--contracts", default="", help="下载真实合约，周期逗号分隔")
    ap.add_argument("--years", type=int, default=0, help="回溯年数（默认主连 10、真实合约 10）")
    ap.add_argument("--symbols", default="", help="品种代码，逗号分隔")
    ap.add_argument("--all", action="store_true", help="全部品种")
    ap.add_argument("--precise-ratio", action="store_true",
                    help="用 5m 切换 bar 同刻价算复权比率（更准，但每品种多几十次大请求）")
    ap.add_argument("--force-switch", action="store_true", help="即使切换表较新也重算")
    ap.add_argument("--stats", action="store_true", help="只看仓库统计")
    args = ap.parse_args()

    if args.stats:
        print(json.dumps(S.stats(), ensure_ascii=False, indent=1))
        return 0

    os.makedirs(tq_root(), exist_ok=True)
    t0 = time.time()

    sess = Session()
    if args.check:
        try:
            q = sess.get().query_quotes(ins_class="FUTURE", expired=False)
            print(f"连接正常，未到期期货合约 {len(q)} 个")
        finally:
            sess.close()
        return 0

    pairs = pick_pairs(args.symbols, args.all)
    print(f"品种 {len(pairs)} 个，仓库 {S.db_path()}")

    ratio_cache: dict = {}

    def with_retry(label: str, fn):
        """跑一个品种，断线就重建会话再试一次。失败两次就记下来跳过，不中断全局。"""
        for attempt in (1, 2):
            try:
                return fn(sess.get())
            except Exception as exc:
                msg = str(exc)
                if "连接失败" in msg or "运维" in msg or "Connection" in msg or "closed" in msg:
                    print(f"    {label} 连接中断（{msg[:40]}），重建会话后重试", flush=True)
                    sess.reset()
                    time.sleep(5)
                    continue
                print(f"    {label} 失败：{type(exc).__name__}: {msg[:100]}", flush=True)
                return None
        print(f"    {label} 重试后仍失败，跳过", flush=True)
        return None

    try:
        # ---- 切换表 ----
        if args.switch or args.main:
            print("\n[1/3] 主力切换表")
            todo = [(s_, e_) for s_, e_ in pairs
                    if args.force_switch or not _fresh_switch(s_, e_)]
            print(f"  需要更新 {len(todo)} 个品种（{len(pairs) - len(todo)} 个较新，跳过）", flush=True)
            maps = fetch_switch(sess.get(), todo) if todo else {}
            for i, (sym, ex) in enumerate(todo, 1):
                key = f"{sym}_{ex}"
                df = maps.get(key)
                if df is None or len(df) < 2:
                    print(f"  [{i}/{len(todo)}] {key}: 无历史主力数据")
                    continue

                def job(api, sym=sym, ex=ex, df=df, key=key, i=i):
                    rec = switch_records(df, sym, ex)
                    rec = fill_ratio(api, rec, ratio_cache,
                                     "5m" if args.precise_ratio else "1d")
                    with S.connect() as con:
                        S.upsert_switch(con, sym, ex, rec)
                    S.meta_set(_switch_stamp(sym, ex), str(date.today()))
                    src = "5m 同刻价" if args.precise_ratio else "日线收盘"
                    n_bar = int((rec.get("price_src") == "bar").sum()) if len(rec) else 0
                    print(f"  [{i}/{len(todo)}] {key}: {len(rec)} 次换月"
                          f"（{n_bar} 条取到{src}，{len(rec) - n_bar} 条回退）"
                          f"，{rec['switch_date'].min().date() if len(rec) else '-'} 起",
                          flush=True)
                    return rec

                with_retry(key, job)

        # ---- 主连 ----
        if args.main:
            print("\n[2/3] 主连")
            years = args.years or MAIN_YEARS
            for period in [p.strip() for p in args.main.split(",") if p.strip()]:
                if period not in PERIOD_SEC:
                    print(f"  [!] 不支持的周期 {period}")
                    continue
                write_main_all(sess, pairs, period, years, with_retry)

        # ---- 真实合约 ----
        if args.contracts:
            print("\n[3/3] 真实合约")
            years = args.years or CONTRACT_YEARS
            for period in [p.strip() for p in args.contracts.split(",") if p.strip()]:
                if period not in PERIOD_SEC:
                    print(f"  [!] 不支持的周期 {period}")
                    continue
                for i, (sym, ex) in enumerate(pairs, 1):

                    def job(api, sym=sym, ex=ex, period=period, years=years, i=i):
                        r = write_contracts(api, sym, ex, period, years)
                        print(f"  [{period} {i}/{len(pairs)}] {r['key']}: +{r['rows']} 行"
                              f" / 新下 {r.get('contracts', 0)} 个合约"
                              f" / 跳过 {r.get('skipped', 0)} 个"
                              f" / 区间外 {r.get('old', 0)} 个"
                              f" / 空合约 {r.get('blank', 0)} 个 {r['status']}", flush=True)
                        return r

                    with_retry(f"{sym}_{ex} {period}", job)
    finally:
        sess.close()

    print(f"\n完成，耗时 {time.time() - t0:.0f}s")
    print(json.dumps(S.stats(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
