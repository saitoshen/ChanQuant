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
from .tqdata import trading_days
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


# ----------------------------------------------------------------- 主连


def write_main_many(api, pairs: list[tuple[str, str]], period: str, years: int,
                    on_row=None, max_parallel: int = 8) -> dict[str, int]:
    """批量下载主连并写库（原始价 + adj_factor，复权在读取时做）。

    一个品种十年 5m 串行要几分钟，78 个品种两个周期就是几小时。这里走 8 路并发，
    批次大小正好取一波 —— 断线最多丢一波，已经落库的品种靠 `S.last_dt()` 的
    增量跳过不会重下。

    `on_row(key, rows, first_dt)` 是进度回调：命令行用它打日志，界面用它刷状态。
    返回 `{品种_交易所: 本次写入行数}`。
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

    out: dict[str, int] = {}
    chunk = max(1, int(max_parallel))
    for i in range(0, len(jobs), chunk):
        batch = jobs[i : i + chunk]
        frames = download_many(
            api, [(t, s, e) for t, s, e, _, _ in batch], period, tq_root(),
            f"MAIN_{period}", max_parallel=chunk,
        )
        with S.connect() as con:
            for tq_sym, _s, _e, sym, ex in batch:
                key = f"{sym}_{ex}"
                df = frames.get(tq_sym)
                if df is None or df.empty:
                    out[key] = 0
                    if on_row is not None:
                        on_row(key, 0, None)
                    continue
                df["adj_factor"] = factor_at(factors[(sym, ex)], trading_days(df["dt"]))
                n = S.upsert_bars(con, sym, ex, period, df)
                out[key] = n
                if on_row is not None:
                    on_row(key, n, df["dt"].iloc[0])
    return out


def update_main(symbol: str, exchange: str, period: str = "5m",
                years: int = MAIN_YEARS, on_row=None) -> int:
    """补齐一个品种的历史，自己开会话、用完就关。

    界面在「选中品种 + 周期」之后调它把本地缺口补上，再交给实时推送。
    断线时抛异常由调用方决定要不要重试 —— 这里不吞。
    """
    sess = Session()
    try:
        got = write_main_many(sess.get(), [(symbol, exchange)], period, years, on_row)
    finally:
        sess.close()
    return int(got.get(f"{symbol}_{exchange}", 0))
