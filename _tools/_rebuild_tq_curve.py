# -*- coding: utf-8 -*-
"""从天勤真实合约库重建全品种展期曲线，并输出重建前后对比。"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ttrade import tqcurve, tqstore  # noqa: E402

OUT = os.path.join(ROOT, "reports", "tq_curve_rebuild_20261125.json")
FOCUS = [("RB", "SHFE"), ("MA", "CZCE"), ("HC", "SHFE"),
         ("TA", "CZCE"), ("I", "DCE")]


def snap(pair):
    sym, ex = pair
    path = tqcurve.curve_path(sym, ex)
    if not os.path.exists(path):
        return {"rows": 0, "first": "", "last": "", "roll_intra": 0}
    d = pd.read_csv(path)
    date = pd.to_datetime(d["date"], errors="coerce")
    return {
        "rows": int(len(d)),
        "first": "" if date.dropna().empty else str(date.min().date()),
        "last": "" if date.dropna().empty else str(date.max().date()),
        "roll_intra": int(d["roll_intra"].notna().sum()) if "roll_intra" in d.columns else 0,
    }


def main():
    with tqstore.connect(read_only=True) as con:
        pairs = [tuple(r) for r in con.execute(
            "SELECT DISTINCT symbol, exchange FROM contract_bars"
            " ORDER BY symbol, exchange").fetchall()]
        periods = {
            f"{s}_{e}": [r[0] for r in con.execute(
                "SELECT DISTINCT period FROM contract_bars"
                " WHERE symbol=? AND exchange=? ORDER BY period", [s, e]).fetchall()]
            for s, e in pairs
        }
    print(f"待重建品种：{len(pairs)} 个合约对", flush=True)

    before = {f"{s}_{e}": snap((s, e)) for s, e in pairs}
    t0 = time.time()
    notes = {}

    def on_row(key, n, note):
        notes[key] = note
        i = len(notes)
        if i % 10 == 0 or i == len(pairs):
            print(f"  进度 {i}/{len(pairs)}：{key} -> {n} 行（{note}）", flush=True)

    res = tqcurve.build_all(pairs, period="1d", on_row=on_row, replace=True)
    print(f"重建耗时 {time.time() - t0:.1f} 秒", flush=True)

    after = {f"{s}_{e}": snap((s, e)) for s, e in pairs}
    payload = {
        "pairs": len(pairs),
        "periods": periods,
        "before": before,
        "after": after,
        "built": res,
        "notes": notes,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    bad = [k for k, v in res.items() if not v]
    print(f"失败品种 {len(bad)} 个：{bad[:20]}", flush=True)
    grew = [(k, before[k]["rows"], after[k]["rows"]) for k in after
            if after[k]["rows"] > before[k]["rows"]]
    shrank = [(k, before[k]["rows"], after[k]["rows"]) for k in after
              if after[k]["rows"] < before[k]["rows"]]
    print(f"行数增加 {len(grew)} 个，减少 {len(shrank)} 个", flush=True)
    for k, a, b in grew[:10]:
        print(f"  + {k}: {a} -> {b}", flush=True)
    for k, a, b in shrank[:10]:
        print(f"  - {k}: {a} -> {b}", flush=True)

    for s, e in FOCUS:
        k = f"{s}_{e}"
        d = pd.read_csv(tqcurve.curve_path(s, e))
        d["date"] = pd.to_datetime(d["date"])
        y19 = d[d["date"] < "2020-01-01"]
        print(f"\n【{k}】{before[k]} -> {after[k]} 说明={notes.get(k, '')}", flush=True)
        print(f"  2019 年前行数 {len(y19)}；c1_code 样例："
              f"{list(y19['c1_code'].head(3))} -> {list(y19['c1_code'].tail(3))}", flush=True)
        if y19.empty:
            continue
        print(f"  2019 年前 c2_code 样例："
              f"{list(y19['c2_code'].head(3))} -> "
              f"{list(y19['c2_code'].tail(3))}", flush=True)
        print(f"  roll_intra 非空 {int(d['roll_intra'].notna().sum())}"
              f" / {len(d)}", flush=True)
    print(f"\n明细已写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
