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
CONTRACT_YEARS = 3     # 真实合约默认回溯年数（体积大一个量级，见 README）

BATCH = 30             # query_his_cont_quotes 单次最多查多少个品种


# ----------------------------------------------------------------- 路径


def tq_root() -> str:
    """天勤数据根目录：`data_tq/`，与 `data_futures/` 平级。"""
    return os.path.join(os.path.dirname(paths.data_dir("cn").rstrip("/\\")), "data_tq")


def main_dir(period: str) -> str:
    return os.path.join(tq_root(), "main", period)


def switch_dir() -> str:
    return os.path.join(tq_root(), "switch")


def contract_dir(period: str) -> str:
    return os.path.join(tq_root(), "contracts", period)


def ensure_dirs(*ds: str) -> None:
    for d in ds:
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------- 天勤连接


def make_api():
    from tqsdk import TqApi, TqAuth

    user, pwd = read_account()
    return TqApi(auth=TqAuth(user, pwd))


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


def download_bars(api, tq_symbol: str, period: str, start: date, end: date,
                  tmp_csv: str) -> pd.DataFrame:
    """用 DataDownloader 下载单合约单周期，返回 DataFrame（dt = 北京时间）。

    DataDownloader 只认文件路径，所以先落一个临时 CSV 再读进来，读完就删。
    增量由调用方决定 `start`（从库里最后一根往前推几天），这里不做合并。
    """
    from tqsdk.tools import DataDownloader

    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)
    task = DataDownloader(
        api,
        symbol_list=tq_symbol,
        dur_sec=PERIOD_SEC[period],
        start_dt=start,
        end_dt=end,
        csv_file_name=tmp_csv,
    )
    while not task.is_finished():
        api.wait_update()

    if not os.path.exists(tmp_csv):
        return pd.DataFrame()
    raw = pd.read_csv(tmp_csv)
    os.remove(tmp_csv)
    if raw.empty:
        return pd.DataFrame()

    # 列名带合约前缀（`KQ.m@SHFE.cu.open`），统一去掉
    ren = {c: c[len(tq_symbol) + 1:] for c in raw.columns if c.startswith(tq_symbol + ".")}
    raw = raw.rename(columns=ren)
    keep = [c for c in ("datetime", "open", "high", "low", "close",
                        "volume", "open_oi", "close_oi") if c in raw.columns]
    out = raw[keep].copy()
    out["dt"] = pd.to_datetime(out["datetime"])
    out = out.drop(columns=["datetime"]).dropna(subset=["close"])
    return out.sort_values("dt").drop_duplicates("dt", keep="last").reset_index(drop=True)


def write_main(api, sym: str, ex: str, period: str, years: int,
               factors: pd.Series) -> dict:
    """下载一个品种的主连并写入仓库（原始价 + adj_factor，复权在读取时做）。"""
    key = f"{sym}_{ex}"
    end = date.today()
    start = end - timedelta(days=365 * years)
    have = S.last_dt(sym, ex, period)
    if have is not None:
        # 往前推 3 天重叠，覆盖「上次下到一半」和节假日边界；主键冲突会覆盖
        start = max(start, (have - timedelta(days=3)).date())

    tmp = os.path.join(tq_root(), "_tmp_dl.csv")
    df = download_bars(api, main_symbol(sym, ex), period, start, end, tmp)
    if df.empty:
        return {"key": key, "period": period, "rows": S.bar_count(sym, ex, period),
                "status": "无新数据"}

    df["adj_factor"] = factor_at(factors, trading_days(df["dt"]))
    with S.connect() as con:
        S.upsert_bars(con, sym, ex, period, df)
    total = S.bar_count(sym, ex, period)
    return {"key": key, "period": period, "rows": total, "status": "ok",
            "added": len(df), "first": str(df["dt"].iloc[0]), "last": str(df["dt"].iloc[-1])}


def write_contracts(api, sym: str, ex: str, period: str, years: int) -> dict:
    """下载一个品种的全部真实合约（不复权，单个合约本身没有换月跳空）。"""
    key = f"{sym}_{ex}"
    end = date.today()
    start = end - timedelta(days=365 * years)

    try:
        conts = api.query_quotes(
            ins_class="FUTURE", exchange_id=ex, product_id=_tq_product_of(sym, ex), expired=True
        )
    except Exception as exc:
        return {"key": key, "period": period, "rows": 0, "status": f"查合约失败 {exc}"}

    total = 0
    n_con = 0
    tmp = os.path.join(tq_root(), "_tmp_cb.csv")
    for c in conts:
        df = download_bars(api, c, period, start, end, tmp)
        if df.empty:
            continue
        try:
            ts_code = to_ts(c, end)
        except Exception:
            ts_code = c  # 解析不了就保留天勤原名，至少不丢数据
        df["contract"] = ts_code
        with S.connect() as con:
            total += S.upsert_contracts(con, sym, ex, period, df)
        n_con += 1
    return {"key": key, "period": period, "rows": total,
            "status": "ok" if total else "无数据", "contracts": n_con}


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


def main() -> int:
    ap = argparse.ArgumentParser(description="从天勤下载 5m/1h 行情到 DuckDB")
    ap.add_argument("--check", action="store_true", help="只检查账户与权限")
    ap.add_argument("--switch", action="store_true", help="更新主力切换表")
    ap.add_argument("--main", default="", help="下载主连，周期逗号分隔：5m,1h")
    ap.add_argument("--contracts", default="", help="下载真实合约，周期逗号分隔")
    ap.add_argument("--years", type=int, default=0, help="回溯年数（默认主连 10、真实合约 3）")
    ap.add_argument("--symbols", default="", help="品种代码，逗号分隔")
    ap.add_argument("--all", action="store_true", help="全部品种")
    ap.add_argument("--precise-ratio", action="store_true",
                    help="用 5m 切换 bar 同刻价算复权比率（更准，但每品种多几十次大请求）")
    ap.add_argument("--stats", action="store_true", help="只看仓库统计")
    args = ap.parse_args()

    if args.stats:
        st = S.stats()
        print(json.dumps(st, ensure_ascii=False, indent=1))
        return 0

    ensure_dirs(tq_root())
    t0 = time.time()

    if args.check:
        api = make_api()
        try:
            q = api.query_quotes(ins_class="FUTURE", expired=False)
            print(f"连接正常，未到期期货合约 {len(q)} 个")
        finally:
            api.close()
        return 0

    pairs = pick_pairs(args.symbols, args.all)
    print(f"品种 {len(pairs)} 个，仓库 {S.db_path()}")

    api = make_api()
    ratio_cache: dict = {}
    try:
        # ---- 切换表 ----
        if args.switch or args.main:
            print("\n[1/2] 主力切换表")
            maps = fetch_switch(api, pairs)
            for i, (sym, ex) in enumerate(pairs, 1):
                key = f"{sym}_{ex}"
                df = maps.get(key)
                if df is None or len(df) < 2:
                    print(f"  [{i}/{len(pairs)}] {key}: 无历史主力数据")
                    continue
                rec = switch_records(df, sym, ex)
                # 默认走日线：一次请求几 KB，全品种十几分钟跑完。
                # --precise-ratio 改用 5m 切换 bar 同刻价，换月残差能从 1.02% 压到 0.85%，
                # 但每个品种要多几十次 10000 根的大请求，全量要几个小时。
                rec = fill_ratio(api, rec, ratio_cache,
                                 "5m" if args.precise_ratio else "1d")
                with S.connect() as con:
                    S.upsert_switch(con, sym, ex, rec)
                n_bar = int((rec.get("price_src") == "bar").sum()) if len(rec) else 0
                src = "5m 同刻价" if args.precise_ratio else "日线收盘"
                print(f"  [{i}/{len(pairs)}] {key}: {len(rec)} 次换月"
                      f"（{n_bar} 条取到 {src}，{len(rec) - n_bar} 条回退）"
                      f"，{rec['switch_date'].min().date() if len(rec) else '-'} 起",
                      flush=True)

        # ---- 主连 ----
        if args.main:
            print("\n[2/2] 主连")
            years = args.years or MAIN_YEARS
            for period in [p.strip() for p in args.main.split(",") if p.strip()]:
                if period not in PERIOD_SEC:
                    print(f"  [!] 不支持的周期 {period}")
                    continue
                for i, (sym, ex) in enumerate(pairs, 1):
                    rec = S.load_switch(sym, ex)
                    factors = adj_factor_series(rec) if len(rec) else pd.Series(dtype=float)
                    r = write_main(api, sym, ex, period, years, factors)
                    print(f"  [{period} {i}/{len(pairs)}] {r['key']}: {r['rows']} 行"
                          f"（+{r.get('added', 0)}）{r['status']}", flush=True)

        # ---- 真实合约 ----
        if args.contracts:
            print("\n[3/3] 真实合约")
            years = args.years or CONTRACT_YEARS
            for period in [p.strip() for p in args.contracts.split(",") if p.strip()]:
                for i, (sym, ex) in enumerate(pairs, 1):
                    r = write_contracts(api, sym, ex, period, years)
                    print(f"  [{period} {i}/{len(pairs)}] {r['key']}: {r['rows']} 行"
                          f" / {r.get('contracts', 0)} 个合约 {r['status']}", flush=True)
    finally:
        api.close()

    st = S.stats()
    print(f"\n完成，耗时 {time.time() - t0:.0f}s")
    print(json.dumps(st, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
