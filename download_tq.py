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
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import tqstore as S  # noqa: E402
from ttrade.futures import list_local  # noqa: E402
from ttrade.paths import data_dir  # noqa: E402
from ttrade.tqfetch import (  # noqa: E402
    CONTRACT_YEARS,
    MAIN_YEARS,
    PERIOD_SEC,
    Session,
    adj_factor_series,
    fetch_switch,
    fill_ratio,
    refresh_all,
    switch_records,
    tq_root,
    write_contracts,
    write_main_many,
)


def pick_pairs(symbols: str | None, all_: bool) -> list[tuple[str, str]]:
    if all_:
        items = list_local(data_dir("cn"))
        pairs = [(it["symbol"], it["exchange"]) for it in items]
    else:
        want = [s.strip().upper() for s in (symbols or "").split(",") if s.strip()]
        if not want:
            raise SystemExit("请用 --symbols RB,HC 指定品种，或用 --all")
        items = {it["symbol"]: it for it in list_local(data_dir("cn"))}
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
    ap.add_argument("--refresh-factors", action="store_true",
                    help="只按当前切换表重算复权因子，不联网、不下载")
    ap.add_argument("--stats", action="store_true", help="只看仓库统计")
    args = ap.parse_args()

    if args.stats:
        print(json.dumps(S.stats(), ensure_ascii=False, indent=1))
        return 0

    if args.refresh_factors:
        pairs = pick_pairs(args.symbols, args.all)
        print(f"品种 {len(pairs)} 个，重算复权因子")
        refresh_all(pairs)
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

            if todo:
                # 切换表刚变过：新 ratio 要乘到全部历史上，已落库的 adj_factor 过期了
                print("  切换表有变化，重算历史复权因子", flush=True)
                moved, touched = refresh_all(pairs, on_msg=lambda m: print(f"  {m}", flush=True))
                print(f"  换月日校正 {moved} 处，复权因子刷新 {touched} 个品种周期", flush=True)

        # ---- 主连 ----
        if args.main:
            print("\n[2/3] 主连")
            years = args.years or MAIN_YEARS
            for period in [p.strip() for p in args.main.split(",") if p.strip()]:
                if period not in PERIOD_SEC:
                    print(f"  [!] 不支持的周期 {period}")
                    continue
                def job(api, period=period, years=years):
                    done = 0
                    total = len(pairs)

                    def on_row(key, rows, first):
                        nonlocal done
                        done += 1
                        if rows:
                            print(f"  [{period} {done}/{total}] {key}: +{rows} 行"
                                  f"（{first:%Y-%m-%d} 起）", flush=True)
                        else:
                            print(f"  [{period} {done}/{total}] {key}: 无新数据", flush=True)

                    write_main_many(api, pairs, period, years, on_row=on_row)

                with_retry(f"主连 {period}", job)

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
