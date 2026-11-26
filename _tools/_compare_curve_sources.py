# -*- coding: utf-8 -*-
"""重建后的天勤展期 vs Tushare 展期：只在共同日期上比较。

输出三张表：
- by_product：逐品种覆盖率、合约腿一致率、数值差、因子相关
- factor_summary：每个因子的总体表现（按品种中位 + 横截面秩相关）
- cross_by_year：横截面秩相关按年分布
"""
import os
import sys
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ttrade import auxdata as AD, factors as F  # noqa: E402

TS_DIR = os.path.join(ROOT, "data_futures_aux")
TQ_DIR = os.path.join(ROOT, "data_tq", "curve")
OUT_DIR = os.path.join(ROOT, "reports")

CODE_COLS = ["c1_code", "c2_code", "c3_code"]
NUM_COLS = [
    "c1_close", "c1_dte", "c1_oi",
    "c2_close", "c2_dte", "c2_oi",
    "c3_close", "c3_dte", "c3_oi",
    "o3_1_close", "o3_1_dte", "o3_1_oi",
    "o6_1_close", "o6_1_dte", "o6_1_oi",
]
FACTORS = [
    "roll_yield", "roll_annual", "roll_curv", "roll_change",
    "near_oi_share", "roll_yield_o3", "roll_yield_o6",
    "roll_main_oi", "roll_main_annual", "roll_main_curv",
]


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).set_index("date").sort_index()
    return d[~d.index.duplicated(keep="last")]


def safe_corr(a: pd.Series, b: pd.Series, method="pearson"):
    m = pd.concat([a, b], axis=1).dropna()
    if len(m) < 10 or m.iloc[:, 0].nunique() < 3 or m.iloc[:, 1].nunique() < 3:
        return np.nan
    return float(m.iloc[:, 0].corr(m.iloc[:, 1], method=method))


def rel_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    den = pd.concat([a.abs(), b.abs()], axis=1).max(axis=1)
    out = (a - b).abs() / den.where(den > 0)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def main() -> None:
    names = sorted(set(f[:-4] for f in os.listdir(TS_DIR)
                       if f.endswith(".csv") and not f.startswith("_"))
                   & set(f[:-4] for f in os.listdir(TQ_DIR)
                         if f.endswith(".csv")))
    print(f"共同品种 {len(names)} 个", flush=True)

    rows = []
    cross: dict[str, dict[str, pd.Series]] = {k: {"ts": {}, "tq": {}} for k in FACTORS}
    for i, name in enumerate(names, 1):
        sym, ex = name.rsplit("_", 1)
        ts, tq = load(os.path.join(TS_DIR, name + ".csv")), load(os.path.join(TQ_DIR, name + ".csv"))
        common = ts.index.intersection(tq.index)
        rec = {
            "key": name, "symbol": sym, "exchange": ex,
            "ts_rows": len(ts), "tq_rows": len(tq), "common_rows": len(common),
            "ts_first": str(ts.index.min().date()), "ts_last": str(ts.index.max().date()),
            "tq_first": str(tq.index.min().date()), "tq_last": str(tq.index.max().date()),
            "common_first": str(common.min().date()) if len(common) else "",
            "common_last": str(common.max().date()) if len(common) else "",
        }
        if len(common) < 30:
            rows.append(rec)
            continue
        a, b = ts.loc[common], tq.loc[common]
        for c in CODE_COLS:
            if c in a.columns and c in b.columns:
                same = a[c].astype(str).eq(b[c].astype(str))
                rec[c + "_match"] = float(same.mean())
        for c in NUM_COLS:
            if c in a.columns and c in b.columns:
                d = rel_diff(a[c], b[c])
                rec[c + "_med_rel"] = float(d.median()) if len(d) else np.nan
                rec[c + "_p90_rel"] = float(d.quantile(0.9)) if len(d) else np.nan

        # attach() 用 join 并列；空列NumPy 的 DataFrame 会把右侧列全丢掉，
        # 必须留一列占位（roll 因子不读它，只为保住索引和列）。
        base = pd.DataFrame(index=common)
        base["_anchor"] = 0.0
        for src, tag in (("tushare", "ts"), ("tq", "tq")):
            f = AD.attach(base.copy(), sym, ex, AD._default_dir(), curve_source=src)
            for k in FACTORS:
                s = F.raw(f, k).reindex(common)
                cross[k][tag][name] = s
                if src == "tushare":
                    rec[k + "_ts_med"] = float(s.median()) if s.notna().any() else np.nan
                else:
                    rec[k + "_tq_med"] = float(s.median()) if s.notna().any() else np.nan
        for k in FACTORS:
            sa, sb = cross[k]["ts"][name], cross[k]["tq"][name]
            rec[k + "_pearson"] = safe_corr(sa, sb)
            rec[k + "_spearman"] = safe_corr(sa, sb, "spearman")
            m = pd.concat([sa, sb], axis=1).dropna()
            rec[k + "_mad"] = float((m.iloc[:, 0] - m.iloc[:, 1]).abs().median()) if len(m) else np.nan
            rec[k + "_sign"] = float((np.sign(m.iloc[:, 0]) == np.sign(m.iloc[:, 1])).mean()) if len(m) else np.nan
        rows.append(rec)
        if i % 20 == 0:
            print(f"  已处理 {i}/{len(names)}", flush=True)

    by = pd.DataFrame(rows).sort_values("key").reset_index(drop=True)
    by.to_csv(os.path.join(OUT_DIR, "curve_source_compare_by_product.csv"),
              index=False, encoding="utf-8-sig")

    # 总体摘要
    common_days = int(by["common_rows"].sum())
    dates = []
    for name in names:
        ts, tq = load(os.path.join(TS_DIR, name + ".csv")), load(os.path.join(TQ_DIR, name + ".csv"))
        dates.extend(ts.index.intersection(tq.index))
    summary = {
        "pairs": len(names),
        "common_product_days": common_days,
        "common_first": str(min(dates).date()) if dates else "",
        "common_last": str(max(dates).date()) if dates else "",
        "code_match_mean": {
            c: float(by[c + "_match"].mean(skipna=True)) for c in CODE_COLS
        },
        "numeric_med_rel": {
            c: float(by[c + "_med_rel"].median(skipna=True)) for c in NUM_COLS
        },
    }
    with open(os.path.join(OUT_DIR, "curve_source_compare_summary.json"), "w",
              encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    # 每个因子：品种内相关 + 横截面（每个交易日跨品种）秩相关
    fac_rows = []
    year_rows = []
    for k in FACTORS:
        A = pd.DataFrame({n: cross[k]["ts"].get(n) for n in names}).sort_index()
        B = pd.DataFrame({n: cross[k]["tq"].get(n) for n in names}).sort_index()
        cs = []
        for dt in A.index.intersection(B.index):
            m = pd.concat([A.loc[dt], B.loc[dt]], axis=1).dropna()
            if len(m) >= 3:
                cs.append((dt, float(m.iloc[:, 0].corr(m.iloc[:, 1], method="spearman"))))
        cs = pd.Series(dict(cs)).dropna()
        fac_rows.append({
            "factor": k,
            "n_products": int(by[k + "_pearson"].notna().sum()),
            "pearson_median": float(by[k + "_pearson"].median(skipna=True)),
            "spearman_median": float(by[k + "_spearman"].median(skipna=True)),
            "mad_median": float(by[k + "_mad"].median(skipna=True)),
            "sign_agree_median": float(by[k + "_sign"].median(skipna=True)),
            "cross_rank_median": float(cs.median()) if len(cs) else np.nan,
            "cross_days": len(cs),
        })
        yr = pd.DataFrame({"v": cs})
        if len(yr):
            yr["year"] = yr.index.year
            for y, g in yr.groupby("year"):
                year_rows.append({"factor": k, "year": int(y),
                                  "cross_rank_median": float(g["v"].median()),
                                  "days": len(g)})
    fs = pd.DataFrame(fac_rows)
    fs.to_csv(os.path.join(OUT_DIR, "curve_source_compare_factor_summary.csv"),
              index=False, encoding="utf-8-sig")
    pd.DataFrame(year_rows).to_csv(
        os.path.join(OUT_DIR, "curve_source_compare_cross_by_year.csv"),
        index=False, encoding="utf-8-sig")

    print("\n=== 覆盖 ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\n=== 因子 ===", flush=True)
    print(fs.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    for c in CODE_COLS:
        print(f"合约腿 {c} 一致率：{summary['code_match_mean'][c]:.3%}", flush=True)
    for c in ["c1_close", "c1_oi", "c2_close", "c2_oi", "o3_1_close", "o6_1_close"]:
        print(f"数值 {c} 中位相对差：{summary['numeric_med_rel'][c]:.6f}", flush=True)


if __name__ == "__main__":
    main()
