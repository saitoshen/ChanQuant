"""Tushare 美股日线（`us_daily`）—— 主数据源，东财作为备用。

为什么单独一个模块
------------------
1. **口径不同**：`us_daily` 返回**未复权**价格，东财用的是后复权（fqt=2）。
   两者的价格基准能差几十倍（分红多的票尤甚），**绝对不能混进同一个 CSV**，
   否则拼接处会出现一个巨大的假跳空。因此这里只负责取数与规范化，
   由 updater 负责「换源就整段重下」的一致性约束。
2. **额度极紧**：本机账号实测 `us_daily` 是 **5 次/天**（不是每分钟）。
   额度用尽后接口会一直报「频率超限」，所以这里带一个「本次会话已不可用」
   的熔断开关，让调用方立刻回退到东财，而不是逐个标的空跑。
3. **限速**：默认两次调用间隔 60 秒，可按账号额度调整。

额度报错分两种：
- 「...次/天」→ 当天没额度了，熔断，不再尝试；
- 「...每分钟」→ 只是瞬时超频，等一会儿还能用。
"""

from __future__ import annotations

import time

import pandas as pd

from . import futures as fut

# 两次 us_daily 调用之间的最小间隔（秒）。用户要求：一分钟一次。
DEFAULT_INTERVAL = 60.0

_MIN_INTERVAL = DEFAULT_INTERVAL
_last_call_at = 0.0

# 熔断开关：一旦确认当天额度耗尽，本次运行不再尝试
_unavailable_reason: str | None = None


class Cancelled(Exception):
    """限速等待期间被用户取消。"""


class QuotaExceeded(RuntimeError):
    """Tushare 额度/频次不足。permanent=True 表示当天已无额度。"""

    def __init__(self, message: str, permanent: bool):
        super().__init__(message)
        self.permanent = permanent


# ------------------------------------------------------------------ 限速
def set_rate_limit(seconds: float) -> None:
    global _MIN_INTERVAL
    _MIN_INTERVAL = max(0.0, float(seconds))


def get_rate_limit() -> float:
    return _MIN_INTERVAL


def reset_call_clock() -> None:
    """把上次调用时间清零，下一次调用不再等待（新一轮更新开始时用）。"""
    global _last_call_at
    _last_call_at = 0.0


def _throttle(cancel=None) -> None:
    """等到满足最小间隔。分片睡眠，期间响应取消。"""
    global _last_call_at
    if _MIN_INTERVAL <= 0:
        _last_call_at = time.monotonic()
        return
    if _last_call_at <= 0:
        _last_call_at = time.monotonic()
        return
    wait = _MIN_INTERVAL - (time.monotonic() - _last_call_at)
    while wait > 0:
        if cancel and cancel():
            raise Cancelled()
        time.sleep(min(0.2, wait))
        wait -= 0.2
    _last_call_at = time.monotonic()


# ------------------------------------------------------------------ 熔断
def available() -> bool:
    return _unavailable_reason is None


def unavailable_reason() -> str:
    return _unavailable_reason or ""


def mark_unavailable(reason: str) -> None:
    global _unavailable_reason
    _unavailable_reason = reason


def reset_availability() -> None:
    global _unavailable_reason
    _unavailable_reason = None


# ------------------------------------------------------------------ 取数
def _is_quota_error(msg: str) -> tuple[bool, bool]:
    """判断异常是否为额度问题，返回 (是额度问题, 是否当天耗尽)。

    Tushare 的措辞不统一，实测见过：
      - 「您访问接口(us_daily)频率超限(5次/天)」        → 当天额度耗尽
      - 「您每分钟最多访问该接口1次」                    → 瞬时超频，等一会儿就好
    所以按关键词集合判断，再单独区分「按天」和「按分钟」。
    """
    markers = ("频率超限", "频次", "超限", "最多访问", "每分钟", "次/天", "次/分")
    if not any(m in msg for m in markers):
        return False, False
    permanent = ("天" in msg) and ("分钟" not in msg) and ("次/分" not in msg)
    return True, permanent


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """把 us_daily 的返回整理成与东财一致的列与顺序。"""
    df = raw.copy()
    rename = {"trade_date": "date", "vol": "volume"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    need = ["date", "open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"us_daily 返回缺列 {missing}，实际列：{list(raw.columns)}")

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA

    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # 列顺序刻意与东财 CSV 保持一致（date, open, close, high, low, ...），
    # 这样两个来源写出来的文件格式相同，读取端不用区分。
    return df[["date", "open", "close", "high", "low", "volume", "amount"]]


def fetch_daily(
    ts_code: str,
    start: str,
    end: str,
    cancel=None,
    retries: int = 2,
) -> pd.DataFrame:
    """取单只美股日线（未复权）。

    start/end 为 `YYYYMMDD` 字符串。额度耗尽时抛 QuotaExceeded。
    """
    if not available():
        raise QuotaExceeded(unavailable_reason(), True)

    pro = fut.get_pro()
    last_err: Exception | None = None

    for attempt in range(retries):
        _throttle(cancel)
        try:
            raw = pro.us_daily(ts_code=ts_code, start_date=start, end_date=end)
        except Exception as exc:  # 额度 / 网络
            msg = str(exc)
            is_quota, permanent = _is_quota_error(msg)
            if is_quota:
                if permanent:
                    mark_unavailable(msg)
                raise QuotaExceeded(msg, permanent) from exc
            last_err = exc
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            continue

        if raw is None or raw.empty:
            return pd.DataFrame()
        return _normalize(raw)

    raise RuntimeError(f"us_daily {ts_code} 失败：{last_err}")
