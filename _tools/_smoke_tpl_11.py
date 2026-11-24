"""新收编的 11 条模板里挑几条**真跑一遍**回测（不只是能解析）。

语法 / 字段等价性已经由 `_gen_tpl_lib_11.py` 的回放校验管住了；这一条管的是
「真接到行情上会不会当场报错」—— 因子键拼错、`sizing='unit'` 没实现、
`window_unit='bar'` 走岔路这类问题只有跑起来才看得见。

用法：
    python _tools/_smoke_tpl_11.py             # 默认 2 年、小样本
    python _tools/_smoke_tpl_11.py --years 5
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from ttrade import auxdata, replay, strategy_templates as T, tqdata  # noqa: E402


def load(sym: str, ex: str, period: str, years: float):
    # 品种池里有退市 / 没下过数据的合约（BB、FB、RO 那些）。界面走
    # `_frames_for()` 时这类品种是**跳过并计数**的（返回 `(frames, missing)`），
    # 所以这里也跳过 —— 让冒烟因为一个死合约整体失败，会把「配置能不能跑」
    # 和「数据缺不缺」两件事混在一起。
    try:
        df = tqdata.load_main(sym, ex, period, adjust="qfq")
    except Exception:  # noqa: BLE001 - 缺数据/读失败都按「这品种没有」处理
        return None
    if df is None or len(df) < 300:
        return None
    df = auxdata.attach_for_period(df, sym, ex, period)
    if years:
        df = df[df.index >= df.index[-1] - pd.Timedelta(days=int(365 * years))]
    return df


def need_periods(cfg) -> set:
    """这条配置除了主周期之外还要哪几档行情（`factor_periods` / 筛池周期）。"""
    out = set()
    fc = cfg.factor_cfg or {}
    for v in (fc.get("factor_periods") or {}).values():
        if v:
            out.add(str(v))
    for v in ((cfg.timing_factor_cfg or {}).get("factor_periods") or {}).values():
        if v:
            out.add(str(v))
    if cfg.cross_mode == "select" and cfg.cross_step1_period:
        out.add(cfg.cross_step1_period)
    out.discard(cfg.period)
    return out


def run_one(key: str, years: float, limit: int = 0) -> bool:
    cfg = T.to_config(key)
    if cfg is None:
        print(f"[BAD] {key} 还原不出来")
        return False
    syms = cfg.symbols
    if limit:
        syms = syms[:limit]
    t0 = time.time()
    extra_periods = need_periods(cfg)
    frames, extras = {}, {}
    for sk in syms:
        sym, ex = replay.split_key(sk)
        df = load(sym, ex, cfg.period, years)
        if df is None:
            continue
        e = {}
        for p in extra_periods:
            d2 = load(sym, ex, p, years)
            if d2 is not None:
                e[p] = d2
        frames[sk] = df
        if e:
            extras[sk] = e
    if not frames:
        print(f"[BAD] {key} 一个品种的行情都没读到")
        return False

    if cfg.family == "cross":
        rp = replay.run_cross(cfg, frames, timing_extra=extras or None,
                              factor_extra=extras or None)
    elif len(cfg.symbols) > 1 and not limit:
        rp = replay.run_multi(cfg, frames, extras=extras or None)
    else:
        one = cfg.clone(symbol=replay.split_key(list(frames)[0])[0],
                        exchange=replay.split_key(list(frames)[0])[1],
                        multi_symbols=[], cross_symbols=[])
        rp = replay.run(one, frames[list(frames)[0]],
                        extra=extras.get(list(frames)[0]))
    dt = time.time() - t0
    if not rp.ok:
        print(f"[BAD] {key:18s} {len(frames)} 品种 {dt:5.1f}s  {rp.error}")
        return False
    m = rp.res.metrics
    print(f"[OK ] {key:18s} {len(frames):3d} 品种 {dt:5.1f}s  "
          f"笔数 {m.get('n_trades', 0):>5}  净值 {m.get('total_return', 0):+.3f}  "
          f"卡玛 {m.get('calmar', 0):+.2f}  {rp.res.label[:40]}")
    return True


def main() -> int:
    years, limit = 2.0, 0
    for i, a in enumerate(sys.argv):
        if a == "--years":
            years = float(sys.argv[i + 1])
        if a == "--limit":
            limit = int(sys.argv[i + 1])

    # (键, 品种上限)。截面那两条太重，冒烟只取前 8 / 12 个品种；
    # 多品种规则那条取全部 7 个（它本来就是小组合）
    cases = [
        ("tpl_lib_factor_70", 0),   # NR 1h，带 @1d_tq 跨周期因子
        ("tpl_lib_factor_71", 0),   # L 日线
        ("tpl_lib_factor_72", 0),   # 实跑 EG 的 5m
        ("tpl_lib_factor_73", 0),
        ("tpl_lib_factor_74", 0),
        ("tpl_lib_factor_75", 0),
        ("tpl_lib_factor_76", 0),   # CU 30m，horizon 56
        ("tpl_lib_rule_25", limit or 6),   # 34 品种 1h，只跑前几个
        ("tpl_lib_rule_26", 0),     # 7 品种 1h，sizing=risk + 只做空
        ("tpl_lib_cross_02", limit or 8),
        ("tpl_lib_cross_03", limit or 12),
    ]
    ok = 0
    for key, lim in cases:
        ok += 1 if run_one(key, years, lim) else 0
    print(f"\n{ok}/{len(cases)} 通过")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
