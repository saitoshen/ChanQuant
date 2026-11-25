"""实盘数据就绪检查 / 自动补全 / 日内衍生列现算。

## 为什么要有这一层

实盘引擎以前只做两件事：订主连 K 线、把本地衍生 CSV 挂到 df 上。于是三种
「看着在跑、其实是死的」：

1. 本地日线停在几天前，没人补；
2. 期限结构 / 期权 IV 的 CSV 停在几天前，`_check_aux_fresh` 只写一行日志，
   展期 / 期权因子照用旧快照；
3. 分钟 / 小时级策略要的展期、IV 只有**日频**快照，日内那一段根本没有值
   （`roll_intra` 靠分钟级真实合约，IV 靠期权实时报价）。

这里把「用到什么 → 够不够新 → 不够怎么补 → 日内怎么现算」集中在一处：

- `needed_aux()`：按因子名判定要不要期限结构 / 期权；
- `freshness()`：行情、展期、IV 三末日对照上一交易日；
- `supplement()` / `SupplementThread`：后台补日线、真实合约、重算展期 / IV；
- `watch_symbols()`：日内需要的真实合约 / 期权合约清单（交给行情源订阅）；
- `LiveAux`：用订阅到的实时报价现算 `roll_intra` 和 IV 摘要，覆盖到当前 bar。

所有联网 / 写库动作都在后台线程且吞掉异常 —— 补数据失败不该把策略打死，
但会在日志里说清楚缺什么。
"""

from __future__ import annotations

import os
import threading
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

#: 展期族：读 `data_futures_aux/<SYM>.csv` 的 c/o3/o6 列，读不到就没值
ROLL_PREFIXES = ("roll_", "near_oi_share")
#: 期权族：读 `iv_*` / `pcr` / `skew` / `curv`，`roll_intra` 例外（它走真实合约）
OPT_PREFIXES = ("iv_", "pcr", "skew", "curv", "opt_")


def needed_aux(period: str, factor_keys, trigger_keys=()) -> dict[str, bool]:
    """这批因子在 `period` 上要哪些衍生数据源。

    因子 id 可能带周期后缀（`roll_main_oi@1d_tq`），比较前先剥掉。
    """
    keys = [str(k) for k in list(factor_keys or []) + list(trigger_keys or [])]
    keys = [k.split("@", 1)[0] for k in keys if k]
    return {
        "curve": any(k.startswith(ROLL_PREFIXES) or k == "near_oi_share"
                     for k in keys),
        "opt": any(k.startswith(OPT_PREFIXES) for k in keys),
    }


def is_intraday(period: str) -> bool:
    """比日线更细就要日内更新（含 `1h` / `5m` / `15m`…）。"""
    return str(period or "1d").split("@")[0] not in ("1d", "2d", "1w", "1mo")


def prev_trading_day(today=None) -> date:
    """上一个交易日。优先用库日历，取不到就退到「昨天，跳过周末」。"""
    try:
        from ttrade import tqstore

        cal = tqstore.calendar()
        if cal:
            ref = pd.Timestamp(today or date.today()).normalize().date()
            past = [d for d in cal if pd.Timestamp(d).date() < ref]
            if past:
                return max(past)
    except Exception:                                   # noqa: BLE001
        pass
    d = pd.Timestamp(today or date.today()).normalize().date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _last_date(x) -> date | None:
    if x is None:
        return None
    try:
        if isinstance(x, pd.DataFrame):
            if x.empty:
                return None
            x = x.index[-1]
        return pd.Timestamp(x).normalize().date()
    except Exception:                                   # noqa: BLE001
        return None


def _daily_csv_last(sym: str, ex: str) -> date | None:
    """Tushare 日线 CSV 的末日（主连日线的另一份权威来源）。"""
    try:
        from .paths import data_dir

        path = os.path.join(data_dir("cn"), f"{sym}_{ex}.csv")
        if not os.path.exists(path):
            return None
        tail = pd.read_csv(path, usecols=["date"]).tail(1)
        return _last_date(tail["date"].iloc[0]) if len(tail) else None
    except Exception:                                   # noqa: BLE001
        return None


def freshness(sym: str, ex: str, need: dict[str, bool]) -> dict:
    """行情 / 展期 / IV 三末日 + 缺哪几样。只读，不发网络请求。"""
    want = prev_trading_day()
    out: dict = {"want": want, "price_1d": None, "curve": None, "opt": None,
                 "stale": []}
    try:
        from ttrade import tqstore

        out["price_1d"] = _last_date(tqstore.last_dt(sym, ex, "1d"))
    except Exception:                                   # noqa: BLE001
        pass
    csv_last = _daily_csv_last(sym, ex)
    if csv_last is not None:
        out["price_1d"] = (csv_last if out["price_1d"] is None
                           else max(out["price_1d"], csv_last))
    try:
        from ttrade import auxdata

        aux = auxdata._default_dir()
        if need.get("curve"):
            out["curve"] = _last_date(auxdata.load_curve(aux, sym, ex))
        if need.get("opt"):
            out["opt"] = _last_date(auxdata.load_opt(aux, sym, ex, "1d"))
    except Exception:                                   # noqa: BLE001
        pass
    for k in ("price_1d", "curve", "opt"):
        if k == "curve" and not need.get("curve"):
            continue
        if k == "opt" and not need.get("opt"):
            continue
        if out[k] is None or out[k] < want:
            out["stale"].append(k)
    return out


def supplement(sym: str, ex: str, need: dict[str, bool], progress=None) -> dict:
    """联网补齐日线与衍生数据。慢，必须放后台线程调用。

    `updater.update_futures` 只补这一条品种（新增 `symbols` 过滤），随后用天勤
    真实合约重算展期、用期权库重算 IV。某一步失败不阻断后面的步骤，结果里逐项
    写明原因，由调用方打日志。
    """
    out: dict = {"symbol": f"{sym}_{ex}", "price": "", "curve": "", "opt": ""}
    if need.get("price") is not False:                  # 默认补日线
        try:
            from ttrade import updater

            r = updater.update_futures(symbols=[f"{sym}_{ex}"], range_key="1m",
                                       progress=progress)
            out["price"] = f"+{r.added} 根，错误 {r.failed}"
        except Exception as exc:                        # noqa: BLE001
            out["price"] = f"失败：{type(exc).__name__}: {str(exc)[:100]}"
    try:
        if need.get("curve") or need.get("opt"):
            from ttrade import updater

            res = updater.update_aux(
                None, range_key="1y",
                want_contracts=bool(need.get("curve")),
                want_options=bool(need.get("opt")),
                progress=progress)
            out["aux"] = f"补了 {getattr(res, 'days', 0)} 个交易日"
    except Exception as exc:                            # noqa: BLE001
        out["aux"] = f"失败：{type(exc).__name__}: {str(exc)[:100]}"
    sess = None
    try:
        if need.get("curve"):
            from ttrade import tqcurve, tqfetch

            sess = sess or tqfetch.Session()
            api = sess.get()
            tqfetch.write_contracts(api, sym, ex, "1d", years=1, days=7)
            tqcurve.build_all([(sym, ex)], "1d")
            out["curve"] = "已重算"
    except Exception as exc:                            # noqa: BLE001
        out["curve"] = f"失败：{type(exc).__name__}: {str(exc)[:100]}"
    try:
        if need.get("opt"):
            from ttrade import optfetch, tqfetch

            # 🔴 必须按品种过滤：不传 varieties 会对全市场 2974 条期权链逐条检查
            # （实测品种过滤能砍掉 98.3%），实盘后台补数会被它拖死。
            # 🔴 大小写两种都要传：variety_of() 对商品期权返回小写（rb/au/m），
            # 对中金所返回大写（IO/MO/HO）—— 只传小写会把股指期权全过滤掉。
            sess = sess or tqfetch.Session()
            varieties = list({sym.lower(), sym.upper()})
            optfetch.update(sess, periods=("1d", "1h"), varieties=varieties)
            out["opt"] = "已重算"
    except Exception as exc:                            # noqa: BLE001
        out["opt"] = f"失败：{type(exc).__name__}: {str(exc)[:100]}"
    finally:
        if sess is not None:
            try:
                sess.close()
            except Exception:                           # noqa: BLE001
                pass
    try:
        from ttrade import auxdata

        auxdata.clear_cache()
    except Exception:                                   # noqa: BLE001
        pass
    return out


class SupplementThread:
    """单例后台补数线程。同时只跑一个任务，重复请求合并，失败过 10 分钟再试。"""

    _inst: "SupplementThread | None" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "SupplementThread":
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    def __init__(self) -> None:
        self._q: list[tuple[str, str, dict]] = []
        self._busy = False
        self._last: dict[str, float] = {}
        self._log = None

    def submit(self, sym: str, ex: str, need: dict[str, bool], log=None) -> bool:
        key = f"{sym}_{ex}"
        now = time.monotonic()
        if now - self._last.get(key, 0.0) < 600:
            return False
        self._last[key] = now
        if log is not None:
            self._log = log
        self._q.append((sym, ex, dict(need)))
        if not self._busy:
            self._busy = True
            threading.Thread(target=self._run, daemon=True,
                             name="live-data-supplement").start()
        return True

    def _run(self) -> None:
        while True:
            if not self._q:
                self._busy = False
                return
            sym, ex, need = self._q.pop(0)
            log = self._log
            try:
                r = supplement(sym, ex, need)
                if log is not None:
                    log(f"{sym}_{ex}：数据补齐 {r}")
            except Exception as exc:                    # noqa: BLE001
                if log is not None:
                    log(f"{sym}_{ex}：数据补齐失败 "
                        f"{type(exc).__name__}: {exc}")


def watch_symbols(sym: str, ex: str, need: dict[str, bool],
                  period: str = "1d", max_options: int = 12) -> dict[str, list[str]]:
    """日内因子要额外订阅哪些天勤合约。

    - 展期：期限结构表最后一天的 `c1_code` / `c2_code`（Tushare 格式）转成
      天勤格式；日内价差漂移必须有这两个真实合约的报价才算得出来；
    - 期权：最近两个到期月里离平值最近的若干档（再远没有流动性，订阅了也是
      白烧服务端订阅名额）。

    返回 `{"curve": [...], "opt": [...]}`。读库失败返回空表，不抛。
    """
    out: dict[str, list[str]] = {"curve": [], "opt": []}
    try:
        from ttrade import auxdata, tqsdk_symbols as TS

        curve = auxdata.load_curve(auxdata._default_dir(), sym, ex)
        if need.get("curve") and not curve.empty and "c1_code" in curve.columns:
            row = curve.iloc[-1]
            for col in ("c1_code", "c2_code"):
                code = str(row.get(col) or "")
                if not code or code == "nan":
                    continue
                try:
                    out["curve"].append(TS.to_tq(code))
                except Exception:                       # noqa: BLE001
                    continue
    except Exception:                                   # noqa: BLE001
        pass
    try:
        from ttrade import optstore as OS

        if need.get("opt") and OS.exists():
            months = OS.list_months()
            if months is not None and not months.empty:
                m = months[months["variety"].astype(str).str.upper() == sym.upper()]
                m = m[m["expiry"] >= pd.Timestamp(date.today())]
                for _, r in m.sort_values("expiry").head(2).iterrows():
                    strikes = OS.list_strikes(str(r["variety"]),
                                              str(r["underlying"]), r["expiry"])
                    if strikes.empty:
                        continue
                    mid = float(pd.to_numeric(strikes["strike"],
                                              errors="coerce").median())
                    s = strikes.assign(
                        _d=(pd.to_numeric(strikes["strike"], errors="coerce") - mid).abs()
                    ).nsmallest(max_options, "_d")
                    out["opt"].extend(str(c) for c in s["contract"])
    except Exception:                                   # noqa: BLE001
        pass
    return out


class LiveAux:
    """用订阅到的实时报价现算日内衍生列，覆盖到 df 的最后一根。

    保持 PIT：日线 / 分钟历史那一段仍是 `attach_for_period` 给的上一交易日快照，
    **只改当前正在走的那根**。合约开盘价按自然日在缓存里记一份，收工后不会串到
    下一天；换日时自动清掉。
    """

    def __init__(self) -> None:
        self.open_px: dict[str, tuple[date, float]] = {}
        self.quote_syms: list[str] = []
        # 展期只能拿「真实合约」的报价算：quote_syms 是 curve+opt 的扁平合并，
        # 期权合约混进前两位时，一旦真实合约还没报价，就会把期权价当远近月价，
        # 算出一个完全是噪声的 roll_intra。分开记一份，取价只认这一组。
        self.curve_syms: list[str] = []
        self.last: dict = {}

    def set_watch(self, syms: list[str], curve_syms: list[str] | None = None) -> None:
        self.quote_syms = list(dict.fromkeys(list(syms)))
        self.curve_syms = (list(dict.fromkeys(list(curve_syms)))
                           if curve_syms is not None else [])

    def _px(self, q) -> tuple[float, float]:
        try:
            px = float(getattr(q, "last_price", 0) or 0)
        except Exception:                               # noqa: BLE001
            px = 0.0
        try:
            oi = float(getattr(q, "open_interest", 0) or 0)
        except Exception:                               # noqa: BLE001
            oi = 0.0
        return px, oi

    def snapshot(self, quotes: dict, sym: str, ex: str, need: dict[str, bool],
                 spot: float = 0.0) -> dict:
        """算一份当前 tick 的衍生值。`quotes` 是 `{天勤合约: quote}`。

        `spot` 是标的现价（主连最新价）—— 期权 IV 反解必须用标的价当 F，
        绝不能用某个期权合约的报价（那是权利金，不是标的）。
        """
        today = date.today()
        out: dict = {}
        if need.get("curve"):
            # 🔴 只认 curve_syms（真实合约），别用扁平的 quote_syms —— 后者在
            # 期权因子同时启用时前两位可能是期权，拿期权价当远近月价会算出噪声。
            # 没显式给 curve_syms 的老调用（例如单测只订了真实合约）回退用全表。
            csyms = [s for s in (self.curve_syms or self.quote_syms) if s in quotes][:2]
            c = [quotes[s] for s in csyms]
            px = [self._px(q)[0] for q in c]
            if len(px) == 2 and min(px) > 0:
                now = px[0] / px[1] - 1.0
                opens = []
                for s, q in zip(csyms, c):
                    p = self._px(q)[0]
                    d0, p0 = self.open_px.get(s, (today, p))
                    if d0 != today or not (p0 > 0):
                        p0 = p
                    self.open_px[s] = (today, p0)
                    opens.append(p0)
                if min(opens) > 0:
                    out["roll_intra"] = now - (opens[0] / opens[1] - 1.0)
        if need.get("opt"):
            out.update(self._iv_snapshot(quotes, sym, ex, spot))
        self.last = out
        return out

    def _iv_snapshot(self, quotes: dict, sym: str, ex: str,
                     spot: float = 0.0) -> dict:
        """用期权实时报价反解近 / 次近两个到期月的 IV 摘要。"""
        try:
            from ttrade import optstore as OS, options as OP
        except Exception:                               # noqa: BLE001
            return {}
        if not OS.exists():
            return {}
        want: dict = {}
        try:
            meta = OS.load_meta(varieties=[sym])
            if meta.empty:
                return {}
            today = pd.Timestamp(date.today())
            meta = meta[pd.to_datetime(meta["expiry"]) > today]
            if meta.empty:
                return {}
            rows = []
            for exp, grp in meta.groupby("expiry", sort=True):
                sel = grp[grp["contract"].astype(str).isin(self.quote_syms)]
                if len(sel) < 4:
                    continue
                dte = int((pd.Timestamp(exp) - today).days)
                if not (OP.MIN_DTE <= dte <= OP.MAX_DTE):
                    continue
                # 🔴 F 必须是标的价（引擎传主连最新价）。原来这里从 quotes 里
                # 抓第一个报价当 spot —— 那是期权权利金（几十块），不是标的
                # （几千点），money=ln(K/F) 全部超出区间 ⇒ IV 恒算不出来。
                if not (spot > 0):
                    continue
                recs = []
                call_oi = put_oi = 0.0
                for _, r in sel.iterrows():
                    c = str(r["contract"])
                    if c not in quotes:
                        continue
                    px, oi = self._px(quotes[c])
                    k = float(r["strike"])
                    if not (px > OP.MIN_PRICE and k > 0):
                        continue
                    is_call = str(r["opt_class"]).upper().startswith("C")
                    if oi > 0:
                        if is_call:
                            call_oi += oi
                        else:
                            put_oi += oi
                    # 只用虚值：实值的时间价值太少，一个跳动就能让 IV 翻倍
                    if not ((k >= spot and is_call) or (k < spot and not is_call)):
                        continue
                    money = float(np.log(k / spot))
                    if abs(money) > OP.MONEY_BAND:
                        continue
                    iv = float(OP.implied_vol(np.array([px]), spot, np.array([k]),
                                              dte / 365.0,
                                              np.array([is_call]))[0])
                    if not (OP.IV_LO * 1.2 < iv < OP.IV_HI * 0.95):
                        continue
                    recs.append((money, iv, np.sqrt(max(oi, 1.0))))
                if len(recs) < 4:
                    continue
                moneys, ivs, ws = (np.array([x[i] for x in recs])
                                   for i in range(3))
                fit = OP.fit_smile(moneys, ivs, ws)
                if fit is None:
                    continue
                iv, skew, curv = (float(x) for x in fit)
                rows.append((dte, iv, skew, curv,
                             put_oi / call_oi if call_oi > 0 else np.nan,
                             len(recs)))
            rows.sort(key=lambda x: x[0])
            if not rows:
                return {}
            n, iv, skew, curv, pcr, cnt = rows[0]
            want = {"iv_near": round(iv, 4), "dte_near": n,
                    "skew": round(skew, 4), "curv": round(curv, 4),
                    "pcr": float(pcr), "n_opt": int(cnt)}
            if len(rows) > 1:
                want["iv_next"] = round(rows[1][1], 4)
                want["dte_next"] = rows[1][0]
        except Exception:                               # noqa: BLE001
            return {}
        return want

    def overlay(self, df: pd.DataFrame, period: str,
                need: dict[str, bool]) -> pd.DataFrame:
        """把当前 tick 的值写到最后一根（只写活 bar，历史一律不碰）。"""
        if df is None or df.empty or not self.last or not is_intraday(period):
            return df
        out = df
        for col, val in self.last.items():
            if col not in out.columns:
                out[col] = np.nan
            try:
                out.iloc[-1, out.columns.get_loc(col)] = val
            except Exception:                           # noqa: BLE001
                pass
        return out
