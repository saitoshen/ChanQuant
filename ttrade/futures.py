"""Tushare 国内期货数据下载器。

覆盖上期所(SHFE)、大商所(DCE)、郑商所(CZCE)、中金所(CFFEX)、能源中心(INE)、广期所(GFEX)
全部品种的主力合约，回溯约 20 年日线。

说明：
- 使用「主力合约」（如 CU.SHF），换月时价格直接拼接会跳空，因此用 Tushare 的
  `fut_daily_adj`（期货复权行情）反推复权因子 `adj_factor = 复权收盘 / 原始收盘`，
  再统一乘到 OHLC 上，消除换月跳空；
- 复权因子是「后复权」累计因子（首日附近为 1，之后按每次移仓的价差连乘），
  与查询窗口无关，可以逐日缓存；
- fut_daily / fut_daily_adj 单次最多返回 2000 条，因此按 offset 分页抓取；
- 字段统一映射为 date/open/high/low/close/volume/amount/open_interest，
  与美股数据保持同一套回测接口；成交额、持仓量不做复权。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from .paths import bundle_root, user_dir

# Tushare token 不写进代码库，避免推到公开仓库后被人白嫖积分。
# 取值优先级见 read_token()。
TOKEN_ENV = "TUSHARE_TOKEN"

EXCHANGES = {
    "SHFE": "上期所",
    "DCE": "大商所",
    "CZCE": "郑商所",
    "CFFEX": "中金所",
    "INE": "能源中心",
    "GFEX": "广期所",
}

PAGE = 2000  # fut_daily 单次上限


class Cancelled(Exception):
    """下载过程中被用户取消。

    必须在这里定义：`fetch_daily` / `fetch_adj_close` 里有 `except Cancelled: raise`，
    如果这个名字不存在，真正出错时会先抛 `NameError: name 'Cancelled' is not defined`，
    把原始错误整个盖掉，排查时看到的全是假线索。
    """

# 交易所代码 → Tushare ts_code 后缀。**两者并不相同**，6 个里有 4 个不一样：
#   SHFE→SHF、CZCE→ZCE、CFFEX→CFX、GFEX→GFE（DCE、INE 同名）
# 直接用 f"{symbol}.{exchange}" 拼会静默返回空表，看起来像「这个品种没数据」。
TS_SUFFIX = {
    "SHFE": "SHF",
    "DCE": "DCE",
    "CZCE": "ZCE",
    "CFFEX": "CFX",
    "INE": "INE",
    "GFEX": "GFE",
}


def ts_code_of(symbol: str, exchange: str) -> str:
    """把「品种代码 + 交易所」拼成 Tushare 的 ts_code（如 CU_SHFE → CU.SHF）。"""
    return f"{symbol}.{TS_SUFFIX.get(exchange, exchange)}"

_pro = None


def read_token() -> str:
    """按优先级取 Tushare token：

    1. 环境变量 `TUSHARE_TOKEN`（CI / 临时会话用）
    2. 项目根目录 `token.txt`（本机开发用，已在 .gitignore 里）
    3. 用户配置目录 `TrendDesk/token.txt`（打包后或不想放项目里时用）

    都找不到就抛错，并把怎么配置写清楚，而不是留一个空字符串让接口
    返回一句「请指定正确的接口名」这种无关的报错。
    """
    env = os.environ.get(TOKEN_ENV, "").strip()
    if env:
        return env

    for path in (
        os.path.join(bundle_root(), "token.txt"),
        os.path.join(user_dir(), "token.txt"),
    ):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                tok = fh.read().strip()
        except OSError:
            continue
        if tok:
            return tok

    raise RuntimeError(
        "未找到 Tushare token。三种配置方式任选其一：\n"
        f"  1) 设置环境变量 {TOKEN_ENV}\n"
        "  2) 在项目根目录新建 token.txt，内容只有一行 token\n"
        "  3) 在用户配置目录下新建 token.txt（Windows: "
        "%LOCALAPPDATA%\\TrendDesk；macOS: ~/Library/Application Support/TrendDesk）\n"
        "token 获取地址：https://tushare.pro/user/token"
    )


def get_pro():
    """惰性创建 tushare pro 客户端。"""
    global _pro
    if _pro is None:
        import tushare as ts

        ts.set_token(read_token())
        _pro = ts.pro_api()
    return _pro


@dataclass
class FutContract:
    ts_code: str      # CU.SHF
    symbol: str       # CU
    name: str         # 沪铜
    exchange: str     # SHFE
    exchange_cn: str  # 上期所


def _clean_name(name: str, exchange: str = "") -> str:
    for suffix in ("主力", "连续"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if exchange == "CFFEX":
        name = name.replace("CFFEX", "")
        if name.endswith("期货"):
            name = name[:-2]
    return name.strip()


def get_universe(include_cffex: bool = True) -> list[FutContract]:
    """拉取所有交易所的主力合约清单（不含「连续」合约）。

    各交易所命名不一致：
      - 商品所（SHFE/DCE/CZCE/INE/GFEX）：name 以「主力」结尾，连续合约为 XXL；
      - 中金所（CFFEX）：主力为 IF/IC/IH/IM/TS/T/TF/TL0，连续合约为 IFL/IFL1/…，
        name 里带「连续」或「当月/次月/当季/下季」。
    """
    import re

    pro = get_pro()
    out: list[FutContract] = []
    for ex, cn in EXCHANGES.items():
        if ex == "CFFEX" and not include_cffex:
            continue
        try:
            df = pro.fut_basic(exchange=ex, fut_type="2", limit=500)
        except Cancelled:
            raise
        except Exception:
            continue
        if df is None or df.empty:
            continue

        for _, r in df.iterrows():
            name = str(r.get("name") or "")
            ts_code = str(r["ts_code"])
            sym = ts_code.split(".")[0]

            # 排除「月均价」与「TAS 指令」等非独立品种
            if "月均价" in name or "TAS" in name.upper() or "_F" in sym:
                continue

            if ex == "CFFEX":
                # 连续合约在名称里带「当月/次月/当季/下季/隔季/连续」，主力不带
                if any(k in name for k in ("连续", "当月", "次月", "当季", "下季", "隔季")):
                    continue
            else:
                # 只认「XX主力」；连续合约为「XX连续」或 XXL（如 CUL/RBL/ALL）。
                # 注意：不能用 sym.endswith("L") 判连续，否则会误杀沪铝(AL)、塑料(L)。
                if not name.endswith("主力"):
                    continue

            out.append(
                FutContract(
                    ts_code=ts_code,
                    symbol=sym,
                    name=_clean_name(name, ex),
                    exchange=ex,
                    exchange_cn=cn,
                )
            )
    out.sort(key=lambda c: (c.exchange, c.symbol))
    return out


def fetch_daily(
    ts_code: str,
    start: str,
    end: str,
    pro=None,
    pause: float = 0.12,
    retries: int = 3,
) -> pd.DataFrame:
    """分页拉取单个合约的日线。"""
    pro = pro or get_pro()
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        last_err = None
        df = None
        for attempt in range(retries):
            try:
                df = pro.fut_daily(
                    ts_code=ts_code,
                    start_date=start,
                    end_date=end,
                    offset=offset,
                    limit=PAGE,
                )
                break
            except Cancelled:
                raise
            except Exception as exc:  # 限频/网络抖动
                last_err = exc
                time.sleep(3 * (attempt + 1))
        if df is None:
            raise RuntimeError(f"{ts_code} offset={offset} 失败: {last_err}")

        if df.empty:
            break
        frames.append(df)
        if len(df) < PAGE:
            break
        offset += PAGE
        time.sleep(pause)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates("trade_date").sort_values("trade_date")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d"),
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw.get("vol"), errors="coerce"),
            "amount": pd.to_numeric(raw.get("amount"), errors="coerce"),
            "open_interest": pd.to_numeric(raw.get("oi"), errors="coerce"),
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.reset_index(drop=True)


def fetch_adj_close(
    ts_code: str,
    start: str,
    end: str,
    pro=None,
    pause: float = 0.12,
    retries: int = 3,
) -> pd.DataFrame:
    """拉取 Tushare 期货复权行情的收盘价（`fut_daily_adj`）。

    该接口直接给复权后的 OHLC，但没有单独的因子列，因此复权因子由
    「复权收盘 / 原始收盘」反推——两者比值就是逐日累计因子，
    在非换月日恒定、在换月日跳变，正是标准复权因子的形态。

    注意：广期所（GFEX，碳酸锂/工业硅/多晶硅）目前没有复权数据，返回空表，
    调用方应退化为不复权（因子 = 1）。
    """
    pro = pro or get_pro()
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        last_err = None
        df = None
        for attempt in range(retries):
            try:
                df = pro.query(
                    "fut_daily_adj",
                    ts_code=ts_code,
                    start_date=start,
                    end_date=end,
                    offset=offset,
                    limit=PAGE,
                )
                break
            except Cancelled:
                raise
            except Exception as exc:  # 限频/网络抖动
                last_err = exc
                time.sleep(3 * (attempt + 1))
        if df is None:
            raise RuntimeError(f"{ts_code} adj offset={offset} 失败: {last_err}")

        if df is None or df.empty:
            break
        frames.append(df[["trade_date", "close"]])
        if len(df) < PAGE:
            break
        offset += PAGE
        time.sleep(pause)

    if not frames:
        return pd.DataFrame(columns=["date", "adj_close"])

    raw = pd.concat(frames, ignore_index=True).drop_duplicates("trade_date")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d"),
            "adj_close": pd.to_numeric(raw["close"], errors="coerce"),
        }
    )
    return out.sort_values("date").reset_index(drop=True)


def attach_adj_factor(df: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """把复权因子并到原始日线上。

    `adj_factor = 复权收盘 / 原始收盘`，只在换月日变化。若某品种没有复权数据
    （广期所）或个别行原始收盘为 0（郑商所僵尸合约的占位行），
    用前值填充，最终兜底为 1.0（等价于不复权）。
    """
    import numpy as np

    out = df.copy()
    if adj is None or adj.empty:
        out["adj_factor"] = 1.0
        return out

    out = out.merge(adj, on="date", how="left")
    ratio = out["adj_close"] / out["close"].replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    # 只在换月日跳变，前向填充即可；序列开头若缺值则后向填充
    ratio = ratio.ffill().bfill().fillna(1.0)
    out["adj_factor"] = ratio.astype(float)
    return out.drop(columns=["adj_close"])


def download_all(
    out_dir: str,
    years: int = 20,
    min_rows: int = 200,
    verbose: bool = True,
    skip_existing: bool = False,
) -> list[dict]:
    """下载全部品种并落地 CSV。

    skip_existing=True 时只补缺失品种（已有 CSV 且数据较新则跳过）；
    若已有 CSV 缺少 `adj_factor` 列（早期版本下载的未复权数据），强制重下。
    """
    os.makedirs(out_dir, exist_ok=True)
    start = (date.today() - timedelta(days=int(years * 365.25) + 30)).strftime("%Y%m%d")
    end = date.today().strftime("%Y%m%d")

    contracts = get_universe()
    if verbose:
        print(f"共 {len(contracts)} 个期货主力合约，区间 {start} ~ {end}")

    pro = get_pro()
    manifest: list[dict] = []

    old: dict[str, dict] = {}
    mpath = os.path.join(out_dir, "_manifest.json")
    if skip_existing and os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            old = {x["ts_code"]: x for x in json.load(fh)}
    fresh_cut = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")

    skipped = 0
    no_factor = 0
    for i, c in enumerate(contracts, 1):
        path = os.path.join(out_dir, f"{c.symbol}_{c.exchange}.csv")
        prev = old.get(c.ts_code)
        if (
            skip_existing
            and prev
            and os.path.exists(path)
            and prev.get("end", "") >= fresh_cut
            and _has_adj_column(path)
        ):
            manifest.append(prev)
            skipped += 1
            continue
        try:
            df = fetch_daily(c.ts_code, start, end, pro=pro)
            if len(df) < min_rows:
                if verbose:
                    print(f"  [{i}/{len(contracts)}] 跳过 {c.ts_code} {c.name}: 仅 {len(df)} 行")
                continue

            adj = fetch_adj_close(c.ts_code, start, end, pro=pro)
            if adj.empty:
                no_factor += 1
            df = attach_adj_factor(df, adj)

            out = df.copy()
            out.insert(0, "ts_code", c.ts_code)
            out.insert(1, "name", c.name)
            out.to_csv(path, index=False, encoding="utf-8-sig")
            fac = out["adj_factor"]
            manifest.append(
                {
                    "ts_code": c.ts_code,
                    "symbol": c.symbol,
                    "name": c.name,
                    "exchange": c.exchange,
                    "exchange_cn": c.exchange_cn,
                    "rows": len(df),
                    "start": df["date"].iloc[0].strftime("%Y-%m-%d"),
                    "end": df["date"].iloc[-1].strftime("%Y-%m-%d"),
                    "med_amount": round(_liquidity(df), 1),
                    "adj_factor_end": round(float(fac.iloc[-1]), 6),
                    "adj_factor_span": round(float(fac.max() / fac.min()), 4) if fac.min() > 0 else None,
                    "adjusted": bool(not adj.empty),
                    "file": os.path.basename(path),
                }
            )
            if verbose:
                note = "" if not adj.empty else "  [无复权数据]"
                print(
                    f"  [{i}/{len(contracts)}] {c.ts_code:<10} {c.name:<8} "
                    f"{len(df):>5}行 {df['date'].iloc[0]:%Y-%m-%d} ~ {df['date'].iloc[-1]:%Y-%m-%d}"
                    f"  因子 {fac.iloc[-1]:.4f}{note}"
                )
        except Exception as exc:
            if verbose:
                print(f"  [{i}/{len(contracts)}] 失败 {c.ts_code}: {str(exc)[:100]}")
        time.sleep(0.15)

    if skipped and verbose:
        print(f"  已跳过 {skipped} 个本地已是最新的品种")
    if no_factor and verbose:
        print(f"  {no_factor} 个品种无复权数据（广期所），按不复权处理")
    if manifest:
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


def _has_adj_column(path: str) -> bool:
    """检查已有 CSV 是否带复权因子列（只读表头，不加载全表）。"""
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    return "adj_factor" in head.columns


def load_local(
    symbol: str,
    exchange: str,
    data_dir: str,
    clean: bool = True,
    adjust: str = "qfq",
    aux_dir: str | None = None,
) -> pd.DataFrame:
    """读取本地期货 CSV，返回与美股一致的 DataFrame。

    clean=True 时剔除价格非正的占位行：郑商所一批僵尸合约（粳稻/普麦/早籼稻/
    强麦/动力煤/胶合板）在无成交日会返回 0 价，直接参与回测会导致除零、净值变 inf。

    adjust 复权方式：
      - "qfq"（默认）：前复权。OHLC 乘以 `adj_factor / adj_factor[-1]`，
        最后一根 K 线等于真实市场价，历史价格按换月价差回溯调整；
      - "hfq"：后复权。OHLC 乘以 `adj_factor`，最新价被放大（铁矿约 8 倍），
        历史固定不变；
      - "none"：不复权，保留主力连续合约的换月跳空。

    三种模式的**收益率完全相同**（只是整体乘一个常数），差异仅在价格绝对水平；
    默认用 qfq 是为了让 GUI 显示的价格与真实盘面一致。
    成交量、成交额、持仓量不参与复权。

    最后会从 `aux_dir`（默认是 data_futures 旁边的 data_futures_aux）并入
    期限结构与期权波动率列，供展期/期权因子使用。缺文件时静默跳过。
    """
    path = os.path.join(data_dir, f"{symbol}_{exchange}.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if clean:
        px = df[["open", "high", "low", "close"]]
        df = df[px.gt(0).all(axis=1)].reset_index(drop=True)

    mode = (adjust or "none").lower()
    if mode in ("qfq", "hfq") and "adj_factor" in df.columns:
        f = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if mode == "qfq":
            base = f[-1] if len(f) and f[-1] > 0 else 1.0
            f = f / base
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].to_numpy(dtype=float) * f

    df = df.set_index("date")
    if aux_dir is None:
        aux_dir = os.path.join(os.path.dirname(data_dir.rstrip("/\\")), "data_futures_aux")
    try:
        from .auxdata import attach as attach_aux

        df = attach_aux(df, symbol, exchange, aux_dir)
    except Exception:
        pass
    # 因子层靠它认品种（品种季节性日历要按品种查表）
    df.attrs["symbol"] = symbol
    return df


def _liquidity(df: pd.DataFrame, tail: int = 250) -> float:
    """近一年日均成交额（万元），用于剔除几乎无成交的僵尸品种。

    注意不能先剔除 0 价行再统计：动力煤 ZC 自 2026-04 起被冻结，
    open/volume/amount 全为 0、close 固定不变，若剔除 0 价行后取尾部 250 行，
    统计到的其实是冻结前一年的成交额，会把僵尸品种误判为活跃品种。
    """
    if df.empty or "amount" not in df:
        return 0.0
    amt = pd.to_numeric(df["amount"], errors="coerce").tail(tail)
    med = float(amt.median())
    return 0.0 if med != med else med  # NaN → 0


def refresh_manifest(data_dir: str) -> list[dict]:
    """为已有清单补充流动性字段 med_amount（万元），不重新下载数据。"""
    path = os.path.join(data_dir, "_manifest.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh)
    for x in items:
        f = os.path.join(data_dir, x["file"])
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        x["med_amount"] = round(_liquidity(df), 1)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
    return items


# ---------------------------------------------------------------- 板块分类
# 交易所分类对策略研究意义不大，品种板块（黑色/有色/能化/农产品…）才是
# 决定趋势性的关键维度，因此按品种代码手工归类。
SECTOR_MAP = {
    "贵金属": "AU AG PT PD",
    "有色": "CU AL ZN PB NI SN AO BC AD SS",
    "黑色": "RB HC I J JM SF SM WR",
    "能源": "SC FU LU BU PG ZC TC",
    "化工": "TA MA ME PP L PL V EG EB PF PR PX SH SA UR RU NR BR SP OP FG BZ",
    "农产品": "A B M Y P C CS JD LH CF CY SR AP CJ RM OI RO RS WH WS WT PM RI ER JR LR RR PK LG",
    "金融": "IF IH IC IM T TF TS TL0",
    "新能源": "LC SI PS",
    "航运": "EC",
}
_SECTOR_OF: dict[str, str] = {s: k for k, v in SECTOR_MAP.items() for s in v.split()}


def sector_of(symbol: str) -> str:
    return _SECTOR_OF.get(symbol.upper(), "其他")


def list_local(
    data_dir: str,
    only_active: bool = True,
    stale_days: int = 365,
    min_amount: float = 5000.0,
) -> list[dict]:
    """读取本地清单，剔除已摘牌/长期无数据的僵尸合约与流动性不足的品种。

    郑商所有一批遗留代码（ME 甲醇、RO 菜油、WS 强麦、ER 早籼稻、TC 动力煤、
    WT 硬麦）已被 MA/OI/WH/RI/ZC 取代，另有 LR 晚籼稻 2022 年停止交易。
    它们的数据只到 2013~2022 年，留在池子里会让组合净值在近期失真。

    min_amount 为近一年日均成交额下限（万元），默认 5000 万元。
    粳稻/普麦/早籼稻/强麦/胶合板/线材/菜籽/纤维板 日均成交不足 500 万元，
    属于「挂着牌但不交易」的品种，无法真实成交，必须剔除。
    """
    path = os.path.join(data_dir, "_manifest.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh)
    if only_active:
        cutoff = (date.today() - timedelta(days=stale_days)).strftime("%Y-%m-%d")
        items = [x for x in items if str(x.get("end", "")) >= cutoff]
    if min_amount and any("med_amount" in x for x in items):
        items = [x for x in items if float(x.get("med_amount") or 0) >= min_amount]
    for x in items:
        x["key"] = f"{x['symbol']}_{x['exchange']}"
        x["sector"] = sector_of(x["symbol"])
    items.sort(key=lambda r: (r["exchange"], r["symbol"]))
    return items
