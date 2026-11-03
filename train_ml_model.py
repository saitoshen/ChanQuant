"""离线训练机器学习策略，把模型落成 .pkl 供实盘直接调用。

## 为什么要有这个脚本

实盘每根 bar 都要出一次信号。机器学习策略如果每次都重跑一遍 walk-forward
（48 折滚动训练），一个品种实测 **34 秒**（65 因子 × 26 万根 5m），19 个品种
串行一轮就是 **11 分钟** —— 5 分钟的 bar 周期根本追不上。

实测过并行：8 核机器上 4 个任务并行，串行 54.28s → 并行 25.28s，**加速比只有
2.15x**（LightGBM 的 `fit` 是 C 扩展但没完全释放 GIL）。19 个品种并行最多把
11 分钟压到 5 分钟上下，仍然追不上。

所以路线是：**训练离线做一次，产物落盘；实盘只加载 + 预测**。本脚本就是
「离线做一次」的那一半。

## 产物长什么样

每折一个独立文件（`<sid>_f000.pkl` …），外加一个主产物（`<sid>.pkl`）记
配置、因子键和续算状态。实盘按 bar 落在哪一折就加载哪一折的模型 ——
和回测里 `walk_forward` 的行为一致，得到的预测序列和回测**逐根相同**。

```
    python train_ml_model.py --sid 334ce0afa6f3 --symbol AG --exchange SHFE
    python train_ml_model.py --sid 334ce0afa6f3 --all      # 19 个品种全训
    python train_ml_model.py --list                        # 看有哪些产物
```

## sid 要对上

**组合实例和策略库条目的 sid 不一样** —— `add_to_portfolio()` 总是复制一份新 sid
（「组合是实例，和库里的模板解耦」）。实盘引擎跑的是**组合实例**，所以产物必须按
**组合实例**的 sid 存，引擎才找得到。

`--sid` 会在**组合和策略库**两处都找（先组合后策略库），所以传哪个都行。
用组合页的「训练模型」按钮训练就一定对得上 —— 它拿的就是组合实例的 sid。

## 训练用哪段历史

和回测同一段：按 `range_years` 切（`0` = 全部历史），前面多留 250 根预热。
**历史不能随便截短** —— 季节因子（`season_avg` / `season_ir` / `season_rank`）
的 raw lookback 高达 78111 根（≈3.4 年，因为要「往年同月」且 `MIN_YEARS = 3`），
截短了它们整列 NaN。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from ttrade import ml_artifact
from ttrade import replay
from ttrade import strategy_store as SStore
from ttrade import tqdata


def _cfg_of(scfg: SStore.StrategyConfig):
    """策略配置 → 因子配置。走 `replay.factor_config`，和回测、实盘同一套。"""
    return replay.factor_config(scfg.factor_cfg or {})


def _load_frame(scfg: SStore.StrategyConfig, symbol: str, exchange: str):
    """读这个品种的行情（和回测同一份前复权口径）。"""
    df = tqdata.load_main(symbol, exchange, scfg.period, adjust="qfq")
    if df is None or df.empty:
        return None
    df = replay.window(df, getattr(scfg, "range_years", 0) or 0)
    return df


def _split_key(key: str, market: str):
    sym, ex = replay.split_key(key, market)
    return sym, ex


def train_one(scfg: SStore.StrategyConfig, symbol: str, exchange: str,
              save: bool = True, quiet: bool = False) -> dict:
    """训练一个品种，产出 .pkl。返回结果摘要。"""
    t0 = time.time()
    df = _load_frame(scfg, symbol, exchange)
    if df is None or len(df) < 200:
        return {"ok": False, "symbol": symbol,
                "error": f"没有 {symbol}_{exchange} 的 {scfg.period} 数据"}

    cfg = _cfg_of(scfg)
    keys = list(scfg.factor_keys or [])
    if len(keys) < 2:
        return {"ok": False, "symbol": symbol, "error": "机器学习至少要 2 个因子"}

    # 产物按「父 sid#品种键」命名，和引擎里子槽位的 sid 完全一致
    # （引擎的 `children_of` 就是这么拼的，见 `strategy_runner.GROUP_SEP`）
    sid = scfg.sid
    if len(scfg.multi_symbols or []) >= 2:
        sid = f"{scfg.sid}#{symbol}_{exchange}"

    def on_progress(done, total, msg):
        if not quiet:
            print(f"    {symbol}: {msg}", flush=True)

    try:
        art, fs = ml_artifact.train_fold_set(
            df, sid=sid, cfg=cfg, keys=keys, name=scfg.name,
            symbol=symbol, exchange=exchange, period=scfg.period,
            progress=on_progress, save=save,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "symbol": symbol,
                "error": f"{type(exc).__name__}: {exc}"}

    return {"ok": True, "symbol": symbol, "exchange": exchange, "sid": sid,
            "folds": len(fs.folds), "bars": len(df), "keys": len(keys),
            "seconds": time.time() - t0,
            "path": ml_artifact.artifact_path(sid),
            "train_until": art.train_until}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="离线训练机器学习策略，产出 .pkl 供实盘直接调用")
    ap.add_argument("--sid", default="", help="策略库里的策略 id")
    ap.add_argument("--symbol", default="", help="品种，如 AG")
    ap.add_argument("--exchange", default="", help="交易所，如 SHFE")
    ap.add_argument("--all", action="store_true", help="训这条策略的所有品种")
    ap.add_argument("--list", action="store_true", help="列出磁盘上的产物")
    ap.add_argument("--remove", default="", help="删掉某个策略的产物")
    ap.add_argument("--no-save", action="store_true", help="只训练，不落盘")
    ap.add_argument("--quiet", action="store_true", help="不打印逐折进度")
    args = ap.parse_args()

    if args.list:
        items = ml_artifact.list_artifacts()
        if not items:
            print("还没有任何训练产物。")
            print("用：python train_ml_model.py --sid <策略id> --all")
            return 0
        for it in items:
            if not it.get("ok"):
                print(f"  [坏] {it.get('name')}: {it.get('error')}")
                continue
            age = it.get("age_days", 0)
            flag = "  (偏旧)" if it.get("stale") else ""
            print(f"  {it.get('name'):<28} {it.get('model',''):<12} "
                  f"{it.get('n_keys',0):>3} 因子  {it.get('folds',0):>3} 折  "
                  f"{age:.1f} 天前{flag}")
        return 0

    if args.remove:
        ok = False
        try:
            fs = ml_artifact.FoldSet.load(args.remove)
            fs.remove()
            ok = True
        except Exception:  # noqa: BLE001
            ok = ml_artifact.remove_artifact(args.remove)
        print("已删除" if ok else "没找到产物")
        return 0 if ok else 1

    if not args.sid:
        ap.error("要 --sid（策略 id）或者 --list")

    # **组合实例和策略库条目的 sid 不一样** —— `add_to_portfolio()` 总是复制一份
    # 新 sid（「组合是实例，和库里的模板解耦」）。实盘引擎跑的是**组合实例**，
    # 所以产物必须按组合实例的 sid 存，否则引擎永远找不到它。
    #
    # 所以这里两处都找：先在组合里找（实盘真正用的那份），再退回策略库。
    pool = []
    try:
        pool.extend(SStore.load_portfolio())
    except Exception:  # noqa: BLE001
        pass
    try:
        pool.extend(SStore.load_library())
    except Exception:  # noqa: BLE001
        pass
    hit = None
    for c in pool:
        if c.sid == args.sid or c.name == args.sid:
            hit = c
            break
    if hit is None:
        print(f"组合和策略库里都没有 id = {args.sid}")
        for c in pool[:20]:
            print(f"    {c.sid}  {c.name}")
        return 1

    if (hit.family or "") != "factor" or (hit.factor_mode or "") != "ml":
        print(f"这不是机器学习策略（族={hit.family}，因子模式={hit.factor_mode}）")
        return 1

    jobs = []
    if args.all:
        for k in (hit.multi_symbols or hit.cross_symbols or []):
            sym, ex = _split_key(k, hit.market)
            jobs.append((sym, ex))
    elif args.symbol:
        jobs.append((args.symbol, args.exchange or "SHFE"))
    else:
        sym, ex = _split_key(hit.key(), hit.market)
        jobs.append((sym, ex))

    if not jobs:
        print("这条策略没有品种清单")
        return 1

    cfg = _cfg_of(hit)

    # 合并训练：所有品种一个模型
    if getattr(cfg, "pooled", False) and len(jobs) >= 2:
        print(f"策略：{hit.name or hit.sid}")
        print(f"品种：{len(jobs)} 个    因子：{len(hit.factor_keys or [])} 个"
              f"    周期：{hit.period}    [合并训练]")
        print()
        t0 = time.time()
        try:
            frames = {}
            for sym, ex in jobs:
                df = _load_frame(hit, sym, ex)
                if df is not None and len(df) >= 200:
                    frames[f"{sym}_{ex}"] = df
            if len(frames) < 2:
                print(f"只有 {len(frames)} 个品种有数据，合训至少要 2 个")
                return 1
            art, fs = ml_artifact.train_fold_set_pooled(
                frames, sid=hit.sid, cfg=cfg, keys=list(hit.factor_keys or []),
                name=hit.name, period=hit.period, save=not args.no_save,
            )
            print(f"✓ 合并训练完成：{len(fs.folds)} 折  "
                  f"{len(frames)} 品种一个模型  {time.time() - t0:.1f}s")
            if not args.no_save:
                print("实盘启动时会自动加载这份产物（只预测，不重训）。")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"✗ 合并训练失败：{type(exc).__name__}: {exc}")
            return 1

    print(f"策略：{hit.name or hit.sid}")
    print(f"品种：{len(jobs)} 个    因子：{len(hit.factor_keys or [])} 个"
          f"    周期：{hit.period}")
    print()

    ok_n = 0
    t0 = time.time()
    for sym, ex in jobs:
        r = train_one(hit, sym, ex, save=not args.no_save, quiet=args.quiet)
        if r.get("ok"):
            ok_n += 1
            print(f"  ✓ {sym}_{ex}  {r['folds']} 折  {r['bars']:,} 根  "
                  f"{r['seconds']:.1f}s  → {os.path.basename(r['path'])}")
        else:
            print(f"  ✗ {sym}_{ex}  {r.get('error')}")

    print()
    print(f"完成 {ok_n}/{len(jobs)}，总耗时 {time.time() - t0:.1f}s")
    if ok_n and not args.no_save:
        print("实盘启动时会自动加载这些产物（只预测，不重训）。")
    return 0 if ok_n == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
