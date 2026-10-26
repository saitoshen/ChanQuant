"""数据增量更新：美股（东方财富）与国内期货（Tushare）。

GUI 的「数据更新」按钮和命令行都走这里。

设计要点
--------
1. **只补不重下**：非「全部」区间只请求最近一段，把结果合并进已有 CSV，
   而不是重下 20 年。
2. **起点取「区间起点」与「本地最后一天往前 10 天」的较早者**，
   这样既能补齐本地缺失的尾巴，也能顺便刷新最后几根可能未收盘的 K 线。
3. **期货复权因子是绝对累计因子**（锚定上市日，与查询窗口无关），
   因此只取最近 30 天也能得到与历史同量级的因子，可以直接拼接。
4. **「全部 20 年」走全量重下**：美股 34 次请求；期货 94 个品种 ×2 次请求，
   耗时数分钟，通过 progress 回调报告进度，可中途取消。
5. 写文件先写 `.tmp` 再 `os.replace`，避免中途崩溃留下半个 CSV。

命令行用法（便于不开 GUI 时补数据）::

    python -m ttrade.updater --market us --range 1m
    python -m ttrade.updater --market cn --range all
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from . import eastmoney as em
from . import futures as fut
from . import us_tushare as us_t
from .paths import data_dir as resolve_data_dir

# (键, 显示名, 天数)。0 表示「全部」，走全量重下。
RANGES: list[tuple[str, str, int]] = [
    ("1m", "近 1 个月", 31),
    ("3m", "近 3 个月", 92),
    ("6m", "近半年", 183),
    ("1y", "近 1 年", 366),
    ("all", "全部 20 年", 0),
]
RANGE_LABELS = {k: label for k, label, _ in RANGES}

FULL_YEARS = 20
OVERLAP_DAYS = 10          # 区间起点再往前多取几天，覆盖未收盘的尾部
SLEEP_US = 0.20            # 东财限流较紧，逐标的之间留间隔
MIN_ROWS_NEW = 200         # 新发现的期货品种少于这么多行视为无效
RETRIES = 2                # 单标的失败重试次数（默认 5 次会让断网时白等几分钟）
ABORT_AFTER = 3            # 连续这么多个标的失败且一个没成功，判定为网络问题直接中止


@dataclass
class UpdateResult:
    market: str
    range_key: str
    source: str = "auto"   # auto / tushare / eastmoney
    ok: int = 0            # 成功处理的标的数
    failed: int = 0
    added: int = 0         # 新增（或刷新）的 K 线根数
    new_symbols: int = 0   # 本次新发现的标的数
    last_date: str = ""    # 全部标的里最新的日期
    cancelled: bool = False
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)      # 非致命提示（如回退）
    sources: dict[str, int] = field(default_factory=dict)  # 各数据源用了多少个标的
    # 衍生数据（真实合约 / 期权）的结果。只有界面上勾选了才会跑，所以是
    # 可选的：`aux=None` 表示「这次没跑」，不是「跑了但没数据」。
    aux: object | None = None
    aux_error: str = ""

    @property
    def label(self) -> str:
        return RANGE_LABELS.get(self.range_key, self.range_key)

    def summary(self) -> str:
        if self.cancelled:
            head = "已取消"
        elif self.failed and not self.ok:
            head = "更新失败"
        else:
            head = "更新完成"
        parts = [f"{head}（{self.label}）", f"{self.ok} 个标的"]
        if self.new_symbols:
            parts.append(f"新发现 {self.new_symbols} 个")
        if self.added:
            parts.append(f"新增 {self.added} 根 K 线")
        else:
            parts.append("无新增数据")
        if self.market == "cn":
            # 期货的 CSV 存的是「原始价 + 复权因子」，因子和日线是一起下的，
            # 这里明写一句，免得用户以为只更新了裸价格。
            parts.append("含复权因子")
        if self.failed:
            parts.append(f"{self.failed} 个失败")
        if self.last_date:
            parts.append(f"最新 {self.last_date}")
        if self.sources:
            tag = " / ".join(
                f"{'Tushare' if k == 'tushare' else '东财'} {v}"
                for k, v in sorted(self.sources.items())
            )
            parts.append(f"来源：{tag}")
        parts.append(f"耗时 {self.elapsed:.1f}s")
        return " · ".join(parts)


# ------------------------------------------------------------------ 工具
def _days_of(range_key: str) -> int:
    for key, _label, days in RANGES:
        if key == range_key:
            return days
    raise KeyError(f"未知区间 {range_key}，可选：{[k for k, _, _ in RANGES]}")


def _cancelled(cancel) -> bool:
    return bool(cancel and cancel())


def _report(progress, done: int, total: int, message: str) -> None:
    if progress:
        progress(done, total, message)


def _window_start(days: int, last: date | None) -> date:
    """区间起点：取「今天往前 days 天」与「本地最后一天往前 OVERLAP_DAYS 天」的较早者。

    days=0 表示全部，直接回溯 20 年。
    """
    today = date.today()
    if days <= 0:
        return today - timedelta(days=int(FULL_YEARS * 365.25) + 30)
    start = today - timedelta(days=days)
    if last is not None:
        start = min(start, last - timedelta(days=OVERLAP_DAYS))
    return start


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """把新数据合并进旧表：同一天以新数据为准，列顺序沿用旧表。"""
    cols = list(old.columns)
    a, b = old.copy(), new.copy()
    a["date"] = pd.to_datetime(a["date"])
    b["date"] = pd.to_datetime(b["date"])
    out = pd.concat([a, b], ignore_index=True, sort=False)
    out = out.drop_duplicates("date", keep="last").sort_values("date")
    rest = [c for c in out.columns if c not in cols]
    out = out[cols + rest].reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def _write_csv(path: str, df: pd.DataFrame) -> None:
    """先写临时文件再原子替换，避免写到一半崩溃留下半个 CSV。"""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def _load_manifest(data_dir: str) -> list[dict]:
    path = os.path.join(data_dir, "_manifest.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def _save_manifest(data_dir: str, items: list[dict]) -> None:
    if not items:
        return
    path = os.path.join(data_dir, "_manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _csv_files(data_dir: str) -> list[str]:
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        f for f in os.listdir(data_dir) if f.endswith(".csv") and not f.startswith("_")
    )


def _date_range_of(df: pd.DataFrame) -> tuple[str, str]:
    d = pd.to_datetime(df["date"])
    return d.iloc[0].strftime("%Y-%m-%d"), d.iloc[-1].strftime("%Y-%m-%d")


class NetworkAbort(RuntimeError):
    """连续多个标的失败，判定为网络/接口问题，不再逐个空跑。"""


def _check_abort(consec: int, ok: int, last_err: str) -> None:
    if ok == 0 and consec >= ABORT_AFTER:
        raise NetworkAbort(
            f"连续 {consec} 个标的都失败，已中止（避免逐个空等）：{last_err}"
        )


# ------------------------------------------------------------------ 美股
SOURCE_TUSHARE = "tushare"
SOURCE_EASTMONEY = "eastmoney"

SOURCE_LABELS = {
    "auto": "自动（东财优先，失败回退 Tushare）",
    SOURCE_TUSHARE: "仅 Tushare（未复权）",
    SOURCE_EASTMONEY: "仅东方财富（后复权）",
}


def _us_full_start() -> date:
    return date.today() - timedelta(days=int(FULL_YEARS * 365.25) + 30)


def _source_order(requested: str) -> list[str]:
    """本次更新尝试数据源的顺序。

    东财是美股的主源：它给的是**后复权**价格，与项目里已有的历史数据同口径。
    Tushare 的 `us_daily` 是未复权，且额度极紧（实测 5 次/天），只作兜底：
    东财整体不可用时才会用到，且一旦换源会对该标的整段重下，
    避免两种口径拼在同一个文件里。
    """
    if requested == SOURCE_EASTMONEY:
        return [SOURCE_EASTMONEY]
    if requested == SOURCE_TUSHARE:
        return [SOURCE_TUSHARE]
    return [SOURCE_EASTMONEY, SOURCE_TUSHARE]


def update_us(
    data_dir: str | None = None,
    range_key: str = "1m",
    progress=None,
    cancel=None,
    source: str = "auto",
    min_interval: float = us_t.DEFAULT_INTERVAL,
) -> UpdateResult:
    """更新美股 CSV。只处理本地已有的文件，不新增标的。

    `source`：
      - "auto"（默认）：**东财优先**，东财不可用时回退 Tushare；
      - "eastmoney"：只用东方财富（后复权）；
      - "tushare"：只用 Tushare（未复权，额度极紧，实测 5 次/天）。

    `min_interval`：Tushare 两次调用之间的最小间隔，默认 60 秒。
    东财有自己的节流（`SLEEP_US`），不受这个参数影响。

    数据一致性：东财是后复权、Tushare 是未复权，两者价格基准能差几十倍，
    **同一个文件只允许一种来源**。若实际用的源与文件记录的原来源不一致，
    会对该标的整段重下（20 年），而不是把两种口径拼在一起。
    """
    data_dir = data_dir or resolve_data_dir("us")
    days = _days_of(range_key)
    t0 = time.time()
    res = UpdateResult(market="us", range_key=range_key, source=source)

    files = _csv_files(data_dir)
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有 CSV，请先运行 python download.py")

    manifest = _load_manifest(data_dir)
    index = {x.get("symbol"): x for x in manifest}
    session = em._session()

    us_t.set_rate_limit(min_interval)
    us_t.reset_call_clock()
    us_t.reset_availability()

    total = len(files)
    ends: list[str] = []
    consec = 0

    for i, fn in enumerate(files, 1):
        if _cancelled(cancel):
            res.cancelled = True
            break
        sym = fn[:-4].replace("_", ".")
        _report(progress, i - 1, total, f"{sym}　{i}/{total}")
        try:
            path = os.path.join(data_dir, fn)
            old = pd.read_csv(path)
            if old.empty:
                raise RuntimeError("本地文件为空")

            rec = index.get(sym) or {}
            file_src = rec.get("source") or SOURCE_EASTMONEY
            last = pd.to_datetime(old["date"]).max().date()

            def window(for_src: str) -> tuple[date, bool]:
                """返回 (起点, 是否需要整段重下)。

                换源必须整段重下：东财是后复权、Tushare 是未复权，
                两种口径拼在一起会出现几十倍的假跳空。
                """
                replace = for_src != file_src
                return (_us_full_start() if replace else _window_start(days, last), replace)

            new = None
            used = None
            start = date.today()
            replace = False
            last_err: Exception | None = None

            for cand in _source_order(source):
                if cand == SOURCE_TUSHARE and not us_t.available():
                    continue
                start, replace = window(cand)
                try:
                    if cand == SOURCE_TUSHARE:
                        new = us_t.fetch_daily(
                            sym, start.strftime("%Y%m%d"),
                            date.today().strftime("%Y%m%d"), cancel=cancel,
                        )
                    else:
                        secid = rec.get("secid")
                        if not secid:
                            sec = em.resolve_secid(sym, session)
                            if sec is None:
                                raise RuntimeError("东财未找到该代码")
                            secid = sec.secid
                            rec["secid"] = secid
                        new = em.fetch_daily(
                            secid, start=start.strftime("%Y%m%d"), adjust=2,
                            session=session, retries=RETRIES,
                        )
                    if new is None or new.empty:
                        raise RuntimeError(f"{cand} 返回空数据")
                    used = cand
                    break
                except us_t.Cancelled:
                    raise
                except us_t.QuotaExceeded as exc:
                    if getattr(exc, "permanent", False):
                        # 当天没额度，熔断，后续标的连试都不试
                        us_t.mark_unavailable(str(exc))
                    last_err = exc
                    if source == SOURCE_TUSHARE:
                        raise
                    res.notes.append(f"{sym}: Tushare 额度不足，本次不再尝试")
                except Exception as exc:
                    last_err = exc
                    if source != "auto":
                        raise
                    # auto：东财挂了就用 Tushare 兜底，并记一笔
                    if cand == SOURCE_EASTMONEY:
                        res.notes.append(f"{sym}: 东财失败，改用 Tushare 兜底")

            if new is None or used is None:
                raise last_err or RuntimeError("所有数据源都不可用")

            if "symbol" not in new.columns:
                new.insert(0, "symbol", sym)

            if replace:
                merged = new.copy()
                merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
                merged = merged.drop_duplicates("date").sort_values("date").reset_index(drop=True)
                res.added += len(merged)
            else:
                merged = _merge(old, new)
                res.added += max(len(merged) - len(old), 0)

            _write_csv(path, merged)

            s0, s1 = _date_range_of(merged)
            ends.append(s1)
            if not rec:
                rec = {"symbol": sym, "name": "", "exchange": ""}
                manifest.append(rec)
                index[sym] = rec
            rec.update(
                rows=len(merged), start=s0, end=s1, file=fn,
                source=used, adjusted=(used == SOURCE_EASTMONEY),
            )
            res.sources[used] = res.sources.get(used, 0) + 1
            res.ok += 1
            consec = 0
        except us_t.Cancelled:
            res.cancelled = True
            break
        except Exception as exc:
            res.failed += 1
            consec += 1
            res.errors.append(f"{sym}: {str(exc)[:120]}")
            _check_abort(consec, res.ok, str(exc)[:120])
        time.sleep(SLEEP_US)

    _save_manifest(data_dir, manifest)
    res.last_date = max(ends) if ends else ""
    res.elapsed = time.time() - t0
    _report(progress, total, total, "完成")
    return res


# ------------------------------------------------------------------ 期货
def _fut_targets(
    data_dir: str,
    days: int,
    include_new: bool,
    manifest: list[dict] | None = None,
) -> list[dict]:
    """列出要处理的品种：本地已有的 + （全量模式下）清单里新出现的。

    ts_code 优先取清单里记录的（最可靠），没有才按交易所映射拼——
    注意 ts_code 后缀与交易所代码**不一样**（SHFE→SHF、CZCE→ZCE、
    CFFEX→CFX、GFEX→GFE），拼错会静默返回空表。
    """
    by_file = {x["file"]: x for x in (manifest or []) if x.get("file")}

    targets: list[dict] = []
    for fn in _csv_files(data_dir):
        stem = fn[:-4]
        if "_" not in stem:
            continue
        symbol, exchange = stem.rsplit("_", 1)
        rec = by_file.get(fn) or {}
        targets.append(
            {
                "ts_code": rec.get("ts_code") or fut.ts_code_of(symbol, exchange),
                "symbol": symbol,
                "exchange": exchange,
                "name": rec.get("name") or "",
                "file": fn,
                "new": False,
            }
        )

    if days == 0 and include_new:
        try:
            universe = fut.get_universe()
        except Exception:
            universe = []
        have = {t["ts_code"] for t in targets}
        for c in universe:
            if c.ts_code in have:
                continue
            targets.append(
                {
                    "ts_code": c.ts_code,
                    "symbol": c.symbol,
                    "exchange": c.exchange,
                    "name": c.name,
                    "exchange_cn": c.exchange_cn,
                    "file": f"{c.symbol}_{c.exchange}.csv",
                    "new": True,
                }
            )
    return targets


def update_futures(
    data_dir: str | None = None,
    range_key: str = "1m",
    progress=None,
    cancel=None,
    include_new: bool = True,
) -> UpdateResult:
    """更新期货 CSV。

    期货走 Tushare，接口额度够用，**不做额外限速**（每个标的之间只留 0.15 秒
    的自然间隔）。限速只针对美股 `us_daily`——那个接口额度极紧。

    缺少 Tushare token 时 `read_token()` 会抛 RuntimeError，
    错误信息里已写明三种配置方式，这里直接向上抛，由界面显示。
    """
    data_dir = data_dir or resolve_data_dir("cn")
    days = _days_of(range_key)
    t0 = time.time()
    res = UpdateResult(market="cn", range_key=range_key, source="tushare")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"目录不存在：{data_dir}")

    pro = fut.get_pro()  # 无 token 时在这里抛出可读错误

    manifest = _load_manifest(data_dir)
    index = {x.get("ts_code"): x for x in manifest}

    targets = _fut_targets(data_dir, days, include_new, manifest)
    if not targets:
        raise FileNotFoundError(f"{data_dir} 下没有 CSV，请先运行 python download_futures.py")

    total = len(targets)
    ends: list[str] = []
    consec = 0

    for i, t in enumerate(targets, 1):
        if _cancelled(cancel):
            res.cancelled = True
            break
        ts_code = t["ts_code"]
        _report(progress, i - 1, total, f"{ts_code}　{t['name'] or ''}　{i}/{total}")
        try:
            path = os.path.join(data_dir, t["file"])
            old = None
            if not t["new"] and os.path.exists(path):
                old = pd.read_csv(path)
            last = None
            if old is not None and not old.empty:
                last = pd.to_datetime(old["date"]).max().date()
            start = _window_start(days, last)
            s, e = start.strftime("%Y%m%d"), date.today().strftime("%Y%m%d")

            new = fut.fetch_daily(ts_code, s, e, pro=pro, retries=RETRIES)
            if new.empty:
                # 没有新数据（可能停牌/已摘牌），不算失败
                if old is not None:
                    res.ok += 1
                    consec = 0
                    ends.append(_date_range_of(old)[1])
                continue

            if t["new"] and len(new) < MIN_ROWS_NEW:
                continue  # 新品种历史太短，视为无效合约

            # 复权因子跟着日线一起下。单独兜底：因子接口抖一下不该让这个标的
            # 连 K 线都写不进去，否则一次瞬时错误就白跑一个品种。
            try:
                adj = fut.fetch_adj_close(ts_code, s, e, pro=pro, retries=RETRIES)
            except fut.Cancelled:
                raise
            except Exception as exc:
                adj = pd.DataFrame()
                res.notes.append(
                    f"{ts_code}: 复权因子未取到，沿用本地因子（{str(exc)[:60]}）"
                )
            new = fut.attach_adj_factor(new, adj)
            # attach_adj_factor 在复权数据为空时会把因子置 1.0。增量场景下这是错的：
            # 老数据的因子可能已经是 8.0，新行填 1.0 会在拼接处造出一个假跳空。
            # 因此复权数据为空但本地已有因子时，沿用最后一根已知因子。
            if adj.empty and old is not None and "adj_factor" in old.columns:
                prev = pd.to_numeric(old["adj_factor"], errors="coerce").dropna()
                if len(prev) and float(prev.iloc[-1]) != 1.0:
                    new["adj_factor"] = float(prev.iloc[-1])
            name = t["name"] or (old["name"].iloc[0] if old is not None and "name" in old else "")
            new.insert(0, "ts_code", ts_code)
            new.insert(1, "name", name)

            if old is not None and not old.empty:
                merged = _merge(old, new)
                res.added += max(len(merged) - len(old), 0)
            else:
                merged = new.reset_index(drop=True)
                merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
                res.added += len(merged)
                res.new_symbols += 1

            _write_csv(path, merged)

            s0, s1 = _date_range_of(merged)
            ends.append(s1)
            fac = pd.to_numeric(merged.get("adj_factor"), errors="coerce").fillna(1.0)
            rec = index.get(ts_code)
            if rec is None:
                rec = {
                    "ts_code": ts_code,
                    "symbol": t["symbol"],
                    "name": name,
                    "exchange": t["exchange"],
                    "exchange_cn": t.get("exchange_cn") or fut.EXCHANGES.get(t["exchange"], ""),
                }
                manifest.append(rec)
                index[ts_code] = rec
            rec.update(
                rows=len(merged),
                start=s0,
                end=s1,
                med_amount=round(fut._liquidity(merged), 1),
                adj_factor_end=round(float(fac.iloc[-1]), 6) if len(fac) else 1.0,
                adj_factor_span=(
                    round(float(fac.max() / fac.min()), 4)
                    if len(fac) and fac.min() > 0
                    else None
                ),
                # 判断依据是「合并后的因子列」，不是「这次有没有拿到复权数据」。
                # 增量更新时 adj 偶尔为空，新行会沿用旧因子，文件其实仍然是复权的；
                # 若按 adj.empty 标记，这个品种会从「已复权」被误降级成「未复权」。
                adjusted=bool(len(fac) and not (fac.round(6) == 1.0).all()),
                file=t["file"],
            )
            res.ok += 1
            consec = 0
        except fut.Cancelled:
            res.cancelled = True
            break
        except Exception as exc:
            res.failed += 1
            consec += 1
            res.errors.append(f"{ts_code}: {str(exc)[:120]}")
            _check_abort(consec, res.ok, str(exc)[:120])
        time.sleep(0.15)

    _save_manifest(data_dir, manifest)
    res.last_date = max(ends) if ends else ""
    res.elapsed = time.time() - t0
    _report(progress, total, total, "完成")
    return res


# ------------------------------------------------------------------ 衍生数据
def update_aux(
    data_root: str | None = None,
    range_key: str = "1m",
    want_contracts: bool = True,
    want_options: bool = True,
    progress=None,
    cancel=None,
    keep_raw_options: bool = False,
):
    """更新真实合约 / 期权衍生数据（期限结构、隐含波动率）。

    比主力连续的增量更新慢一个量级：每个交易日都要请求一次全市场横截面，
    20 年的全量约 40~90 分钟。所以界面上做成可选项，默认不跟着更新。

    只补「行情表里有、衍生数据还没算过」的交易日，所以日常更新只会请求
    最近几个交易日。返回 `auxbuild.BuildResult`。
    """
    from . import auxbuild as AB
    from . import contracts as C

    root = data_root or os.path.dirname(resolve_data_dir("cn"))
    days = _days_of(range_key)
    start = _window_start(days, None).strftime("%Y%m%d")

    all_days = C.trading_days(os.path.join(root, "data_futures"), start=start)
    if not all_days:
        raise FileNotFoundError(f"{root}/data_futures 下没有行情数据")

    man = C.load_manifest(os.path.join(root, "data_futures_aux"))
    done_c = set(man.get("dates") or [])
    done_o = set(man.get("opt_dates") or [])
    todo = [
        d for d in all_days
        if (want_contracts and d not in done_c) or (want_options and d not in done_o)
    ]
    if not todo:
        res = AB.BuildResult(want_contracts=want_contracts, want_options=want_options)
        if progress:
            progress(1, 1, "合约 / 期权数据已是最新")
        return res

    return AB.build(
        todo, data_root=root,
        want_contracts=want_contracts, want_options=want_options,
        keep_raw_options=keep_raw_options,
        progress=progress, cancel=cancel,
    )


# ------------------------------------------------------------------ 命令行
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="增量更新行情数据")
    ap.add_argument("--market", choices=["us", "cn"], default="us")
    ap.add_argument("--range", dest="range_key", choices=[k for k, _, _ in RANGES], default="1m")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument(
        "--source", choices=["auto", "tushare", "eastmoney"], default="auto",
        help="美股数据源；期货固定走 Tushare",
    )
    ap.add_argument(
        "--interval", type=float, default=us_t.DEFAULT_INTERVAL,
        help=f"Tushare 美股两次调用的最小间隔（秒），默认 {us_t.DEFAULT_INTERVAL:g}",
    )
    ap.add_argument(
        "--aux", action="store_true",
        help="期货更新后接着更新衍生数据（真实合约期限结构 + 期权 IV），很慢",
    )
    ap.add_argument(
        "--aux-what", choices=["both", "contracts", "options"], default="both",
        help="配合 --aux 用，指定只更新其中一项",
    )
    args = ap.parse_args(argv)

    def on_progress(done: int, total: int, msg: str) -> None:
        print(f"\r  [{done}/{total}] {msg:<40}", end="", flush=True)

    try:
        if args.market == "us":
            res = update_us(
                args.data_dir, args.range_key, progress=on_progress,
                source=args.source, min_interval=args.interval,
            )
        else:
            res = update_futures(args.data_dir, args.range_key, progress=on_progress)
            if args.aux:
                # 衍生数据排在主行情之后：它失败也不影响已经落盘的主力连续。
                # data_root 是「上一级」，因为 auxbuild 要自己拼 data_futures /
                # data_contracts / data_futures_aux 三个子目录。
                aux_root = (os.path.dirname(args.data_dir.rstrip("/\\"))
                            if args.data_dir else None)
                aux = update_aux(
                    aux_root, args.range_key,
                    want_contracts=args.aux_what in ("both", "contracts"),
                    want_options=args.aux_what in ("both", "options"),
                    progress=on_progress,
                )
                res.notes.append("衍生数据：" + aux.summary())
                res.notes.extend(aux.errors[:5])
    except Exception as exc:
        print(f"\n失败：{exc}")
        return 1

    print()
    print(res.summary())
    for n in res.notes[:5]:
        print("  ·", n)
    for e in res.errors[:10]:
        print("  ", e)
    return 0 if not res.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
