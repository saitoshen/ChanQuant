# -*- coding: utf-8 -*-
"""全局行情源：一条行情连接，所有账户 / 策略共用。

## 为什么要这个

以前是**每个账户一条连接，行情和交易绑在一起**（`PortfolioRunner.start()` 里
`TqApi(TqAccount(...))` 或 `BrokerApi(...)`）。后果有三个，2026-11-20 全部撞上：

1. **柜台账户没有行情**。`快期模拟`（kind=`kq`，`simnow_7x24`）只能订到商品，
   中金所（T / TL / IF / IC / IM / IH）全线 `quote` 空、`kline` 空 ⇒ 那些策略
   永远算不出信号，界面显示「理论 0」，和「策略真的看空」长得一模一样。
2. **策略离不开账户**。账户没登录就不建 runner，策略跟着停摆；而实盘账户是要
   日终登出 / 次日登录的，策略被迫跟着断，谈不上 7×24。
3. **同一品种被重复订阅**。44 条策略只占 27 个「品种@周期」，一账户一连接时
   同一份 K 线在多个连接里各订一遍，切换账户还会订出两份不同步的。

## 三个源，可选，互不影响

| key | 名字 | 连什么 | 默认 |
|-----|------|--------|------|
| `tq` | 天勤 | **纯行情 `TqApi`**，只要 `TqAuth`，不要资金账户 | ✅ 全局默认 |
| `ctp` | CTP · SimNow 第一套 | `BrokerApi(ctp 账户)` = `CtpHub` | 备用 |
| `rohon` | 融航 | `BrokerApi(rohon 账户)` = `RohonHub` | 备用 |

**账号用的是账户表里自己的那一个，和数据下载不共用**：

- `tq` 走**天勤模拟账户**（`kind=kq` 的「快期模拟」，也就是 `TqKq`）的
  user / password。以前这里走 `read_account()`（环境变量 / `tq_account.txt`），
  那份是**数据下载**的账号，两者混用会出现「下载在跑、行情被挤掉」这种事，
  而且用户在界面上看不到自己在用谁的号。
- `ctp` / `rohon` 走账户表里对应的柜台账户 —— 柜台账户本来就有 `md_front`，
  融航的行情前置就是标准 CTP md 服务。

`fast` 说明：CTP 用 **SimNow 第一套**（`ctp_preset=simnow_day`，
`182.254.243.31:30011`），不是 7×24 那套。7×24 那套只有商品、没有中金所，
拿它当行情源等于把问题 1 又搬回来一次。

## 怎么做

- 天勤的**行情**只要 `TqAuth`，不需要任何资金账户（`LiveHub(specs=[])` 早就
  这么干了）。所以 `tq` 源是一个**纯行情 TqApi**。
- 三个源**同时只有一个在跑**（`select()` 切）。同时跑三条会互相抢柜台通道，
  而且「同一根 bar 两条源各推一次」会让信号抖。
- 🔴 **TqApi 不是线程安全的**：`wait_update()` 只能在创建它的那条线程里调，
  其它线程连 `get_kline_serial()` 都不该直接调。所以每个源都是：
  - 只有 `_loop` 线程碰 `self._api`；
  - 策略线程要订阅，把请求投进 `_pending`，由 `_loop` 在下一轮 `wait_update`
    之后统一建好，再通过条件变量通知回来；
  - 策略线程读 serial 是本进程内的活引用，读 DataFrame 时持 `lock()`（见
    `read_kline`），避免和 `wait_update` 写入撕成半张表。
- 没人订阅时 `wait_update()` 会立刻返回（空转烧 CPU），所以下限用
  `deadline` + 一个极小的 sleep。

## 用法

    feed = MarketFeed.get()          # 全局唯一，首次 get 就按默认源 start
    ser  = feed.kline("KQ.m@CFFEX.T", 300, 3000)     # 活引用
    q    = feed.quote("KQ.m@CFFEX.T")
    feed.wait(0.8)                   # 驱动一次，等价于原来的 api.wait_update()
    df   = feed.read_kline("KQ.m@CFFEX.T")           # 加锁拷一份（要改/要算用它）
    feed.select("ctp")               # 换成 SimNow 第一套（界面上的独立选项框）

`close()` 只在退出程序时调；`get()` 之后 `start()` 失败不抛，`error` 里留原因，
调用方据此决定「显示未连接」还是「继续用本地历史算信号」—— 后者才是 7×24 的
关键：行情断了，策略仍可按最后一根已完成的 bar 出信号，只是不进新仓。
"""

from __future__ import annotations

import json
import os
import threading
import time

import pandas as pd

__all__ = ["MarketFeed", "SOURCE_LABELS", "SOURCE_KEYS"]


#: 等一次 `wait_update` 的上限（秒）。没订阅时它会立刻返回，靠这个兜住 CPU。
_WAIT_SLICE = 0.5

#: 两轮行情推进之间的**最小间隔**（秒）。盘中每个品种每个 tick 都会让
#: `wait_update()` 立刻返回，不设下限时 `_loop` 会以几千次/秒空转：每次都抢一次
#: 锁 + `notify_all` 唤醒全部等待者，GIL 被它吃满，Qt 主线程画界面就卡住了
#: （2026-11-20 用户报「界面经常卡住」）。策略本来只按 0.4 秒的节拍看盘，
#: 120ms 一轮完全够。
_MIN_INTERVAL = 0.12

#: 行情源清单。顺序 = 界面选项框的顺序，第一个是默认。
SOURCE_KEYS = ("tq", "ctp", "rohon")
SOURCE_LABELS = {
    "tq": "天勤（默认）",
    "ctp": "CTP · SimNow 第一套",
    "rohon": "融航（备用）",
}

#: CTP 源只认 SimNow **第一套**。7×24 那套没有中金所，拿它当行情源等于白连。
CTP_FEED_PRESET = "simnow_day"

#: 程序打开时要不要自动连默认源。**自检 / 离线脚本把它置 False** ——
#: 自检要能离线重复跑，不该拿用户的账号去登录一次真实行情（登录失败还会让
#: 「构建主窗口」那一项随网络状况时红时绿）。开关只有这一处，别在各处判环境。
AUTO_CONNECT = True


# ------------------------------------------------------------------ 选源存档


def _settings_path() -> str:
    from ttrade.paths import user_dir

    return os.path.join(user_dir(), "feed.json")


def load_settings() -> dict:
    """读选源存档。文件不存在 / 坏了都返回空 dict，不抛。"""
    try:
        with open(_settings_path(), encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_settings(data: dict) -> None:
    """写选源存档（原子替换，写到一半断电不会留下半个 JSON）。"""
    path = _settings_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass


def pick_account(source: str, name: str = ""):
    """按源挑一个账户：`name` 优先，否则按「最像行情源」的规则自动挑。

    自动挑的规则写在这里而不是散在界面里：`tq` 要**模拟账户**（`kq` 优先，
    其次是本地 `sim`，最后才是实盘 `account` —— 行情不需要下单权限，用实盘
    账户纯属多登一次）；`ctp` 要 SimNow 第一套；`rohon` 只要有一个就行。
    """
    from ttrade.tqlive import load_accounts

    specs = load_accounts()
    typed = [s for s in specs if getattr(s, "backend", "tq") == source]
    if name:
        for s in typed:
            if s.name == name:
                return s
    if not typed:
        return None
    if source == "tq":
        for kind in ("kq", "sim", "account"):
            for s in typed:
                if s.kind == kind:
                    return s
        return typed[0]
    if source == "ctp":
        for s in typed:
            if getattr(s, "ctp_preset", "") == CTP_FEED_PRESET:
                return s
        return typed[0]
    return typed[0]


# ------------------------------------------------------------------ 单源通道


class _Chan:
    """一个行情源。子类只需实现 `_connect` / `_subscribe` / `_drive`。

    线程模型见模块 docstring：`_loop` 是唯一碰底层 api 的线程，订阅走 `_pending`。
    """

    label = ""

    def __init__(self, source: str, spec) -> None:
        self.source = source
        self.spec = spec
        self._api = None
        self._cond = threading.Condition(threading.RLock())
        # 🔴 K 线缓存按 **(合约, 周期秒数)** 建键。只按合约建键时，同一个品种被
        # 两个周期的槽位订阅会**互相串号** —— 谁先订成功谁占坑，另一条直接拿去
        # 用，而且不报错。2026-11-23 实测：品种池里那条 1h 的「sbbox多品种波段」
        # 含 `P_DCE`，它先订成功占住了坑，于是棕榈油的 `1d_tq` 策略拿到的是
        # **1h 序列** —— 合并后 497 根（日线本该 492 根）、末根 `14:00`，因子在
        # 小时线上重算，signal 从 0 翻成 −1，用户看到「回测空仓、实盘 −2 手」。
        # 盘口没有周期概念，仍然按合约建键。
        self._serials: dict[tuple[str, int], object] = {}
        self._quotes: dict[str, object] = {}
        # 待订阅：键 = (tq, 周期秒, kind)，值 = (tq, 周期秒, 根数, kind)。
        # **键里必须带周期**：只用 (tq, kind) 的话，同一合约的日线和 1h 是同一个
        # 键，后投的请求会被当成「已经在队列里了」而丢掉
        self._pending: dict[tuple[str, int, str], tuple] = {}
        self._tick = 0
        self._last_ts = 0.0
        self._last_drive = 0.0
        self._err = ""
        self._stop = False
        self._thread: threading.Thread | None = None
        self._started = False
        # 建连接只许一次：`get()` 现在在锁外 start，两个线程同时进来会各建一条
        # TqApi（同账号双连接，订阅翻倍、还容易被判多端登录）
        self._start_lock = threading.Lock()

    # ---- 子类实现 ----

    def _connect(self):
        raise NotImplementedError

    def _subscribe(self, api, tq: str, dur: int, length: int, kind: str):
        """在 `_loop` 线程里建订阅，返回活引用。"""
        raise NotImplementedError

    def _drive(self, api) -> bool:
        """推进一次。返回这期间有没有更新。"""
        raise NotImplementedError

    def _close_api(self, api) -> None:
        try:
            api.close()
        except Exception:                        # noqa: BLE001
            pass

    # ---- 生命周期 ----

    def start(self) -> bool:
        """建连接并起后台线程。**不抛异常**：行情起不来不该把界面带崩。"""
        with self._start_lock:
            return self._start_locked()

    def _start_locked(self) -> bool:
        with self._cond:
            if self._started and (self._api is not None or not self._err):
                return bool(self._api)
            self._started = True
            self._stop = False
            old, self._api = self._api, None
            self._err = ""
            self._serials.clear()
            self._quotes.clear()
            self._pending.clear()
        if old is not None:
            self._close_api(old)
        try:
            api = self._connect()
        except Exception as exc:                 # noqa: BLE001
            with self._cond:
                self._err = f"{type(exc).__name__}: {exc}"
                self._cond.notify_all()
            return False
        with self._cond:
            self._api = api
            self._err = ""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name=f"feed-{self.source}")
            self._thread.start()
        return True

    def retry(self) -> bool:
        with self._cond:
            if self._api is not None and not self._err:
                return True
        return self.start()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._started = False
            self._cond.notify_all()
            api, self._api = self._api, None
        if api is not None:
            self._close_api(api)

    # ---- 后台线程 ----

    def _loop(self) -> None:
        while True:
            with self._cond:
                if self._stop:
                    return
                api = self._api
            if api is None:
                time.sleep(0.2)
                continue
            # 🔴 限速：盘中 `wait_update()` 会因为任意一个品种的 tick 立刻返回，
            # 不限的话这里是几千次/秒的空转（每次抢锁 + notify_all 叫醒所有策略
            # 线程），GIL 被它吃满，Qt 主线程画界面就卡住了
            gap = _MIN_INTERVAL - (time.time() - self._last_drive)
            if gap > 0:
                time.sleep(gap)
            self._last_drive = time.time()
            try:
                self._drive(api)
            except Exception as exc:             # noqa: BLE001
                with self._cond:
                    self._err = f"{type(exc).__name__}: {exc}"
                    self._api = None
                    self._cond.notify_all()
                continue
            # 订阅请求只能在这里建：`get_kline_serial` / `get_quote` 要碰 api 内部
            # 状态，天勤不保证跨线程安全
            with self._cond:
                jobs = list(self._pending.items())
                self._pending.clear()
            for _pkey, (tq, dur, length, kind) in jobs:
                try:
                    got = self._subscribe(api, tq, dur, length, kind)
                    with self._cond:
                        if kind == "k":
                            self._serials[self._kk(tq, dur)] = got
                        else:
                            self._quotes[tq] = got
                except Exception as exc:         # noqa: BLE001
                    with self._cond:
                        self._err = f"订阅 {tq} 失败：{type(exc).__name__}: {exc}"
            with self._cond:
                self._tick += 1
                if not jobs:
                    self._last_ts = time.time()
                self._cond.notify_all()

    # ---- 订阅 / 读 ----

    @staticmethod
    def _kk(tq: str, dur: int) -> tuple[str, int]:
        """K 线缓存的键 = 合约 + 周期秒数。**别退化成只用合约**（见 `__init__`）。"""
        return (str(tq), int(dur or 0))

    def _ask(self, tq: str, dur: int, length: int, kind: str, timeout: float = 8.0):
        d = int(dur or 0) if kind == "k" else 0
        kk = self._kk(tq, d)
        with self._cond:
            if kind == "k" and kk in self._serials:
                return self._serials[kk]
            if kind == "q" and tq in self._quotes:
                return self._quotes[tq]
            if self._api is None:
                return None                      # 没连上：调用方用本地历史兜底
            self._pending[(tq, d, kind)] = (tq, d, int(length), kind)
            ok = self._cond.wait_for(
                lambda: (kk in self._serials if kind == "k" else tq in self._quotes)
                        or self._api is None, timeout)
            if not ok:
                return None
            return self._serials.get(kk) if kind == "k" else self._quotes.get(tq)

    def kline(self, tq: str, duration_sec: int, length: int = 3000):
        return self._ask(tq, int(duration_sec), int(length), "k")

    def quote(self, tq: str):
        return self._ask(tq, 0, 0, "q")

    # ---- 非阻塞版：投递请求，本轮不等回填 ----

    def request(self, tq: str, duration_sec: int, length: int, kind: str = "k") -> None:
        """**只投递订阅请求，不等结果**。

        为什么要这个：天勤首次订阅一个品种要把 3000~4500 根历史下回来，实测
        **0.5~2.3 秒/个**。串行等 27 个「品种@周期」就是 30~60 秒 —— 那段时间
        引擎卡在第一次 tick 里，界面不刷新，用户看到的就是「启动卡死」
        （2026-11-20）。改成投递后立刻返回，回填由 `_loop` 自己完成，下一 tick
        再取；期间策略显示「行情未就绪」，不影响界面。
        """
        if kind == "k" and self._kk(tq, duration_sec) in self._serials:
            return
        if kind == "q" and tq in self._quotes:
            return
        d = int(duration_sec or 0) if kind == "k" else 0
        with self._cond:
            if self._api is None:
                return                            # 没连上，投也没用
            if (tq, d, kind) in self._pending:
                return
            self._pending[(tq, d, kind)] = (tq, d, int(length), kind)

    def peek(self, tq: str, dur: int = 0, kind: str = "k"):
        """取已经回填好的活引用（没回填就 None）。**不阻塞**。

        `dur` 是**周期秒数**，K 线必传 —— 缓存按 (合约, 周期) 建键，不传等于
        取「0 秒」那一档，永远拿不到东西。
        """
        with self._cond:
            if kind != "k":
                return self._quotes.get(tq)
            return self._serials.get(self._kk(tq, dur))

    def counts(self) -> tuple[int, int]:
        with self._cond:
            return len(self._serials), len(self._quotes)

    def lock(self):
        return self._cond

    def read_kline(self, tq: str, copy: bool = True, duration_sec: int = 0):
        with self._cond:
            ser = self._serials.get(self._kk(tq, duration_sec))
            if ser is None:
                return None
            return ser.copy() if copy else ser

    def read_quote(self, tq: str):
        with self._cond:
            return self._quotes.get(tq)

    def wait(self, timeout: float = 0.8) -> bool:
        with self._cond:
            t0 = self._tick
            self._cond.wait_for(lambda: self._tick != t0 or self._api is None, timeout)
            return self._tick != t0

    # ---- 状态 ----

    @property
    def ready(self) -> bool:
        return self._api is not None

    @property
    def error(self) -> str:
        with self._cond:
            return self._err

    def stats(self) -> dict:
        with self._cond:
            return {
                "source": self.source,
                "label": self.label,
                "account": getattr(self.spec, "name", "") if self.spec else "",
                "ready": self._api is not None,
                "error": self._err,
                "klines": len(self._serials),
                "quotes": len(self._quotes),
                "tick": self._tick,
                "idle": (time.time() - self._last_ts) if self._last_ts else None,
            }


class TqChan(_Chan):
    """天勤纯行情通道：只要 `TqAuth`，不要资金账户。

    账号用账户表里那个**天勤模拟账户**（`kind=kq`），不从数据下载那份
    `read_account()` 取 —— 两者混用会出现「下载占着连接、行情挤掉」而且界面上
    看不出用的是谁的号。
    """

    label = "天勤"

    def _connect(self):
        from tqsdk import TqApi, TqAuth

        spec = self.spec
        user = getattr(spec, "user", "") if spec is not None else ""
        pwd = getattr(spec, "password", "") if spec is not None else ""
        if not user or not pwd:
            raise ValueError(
                "没有可用的天勤账户。请在「账户透视和交易」页添加一个天勤模拟账户"
                "（快期模拟），行情源要用它的账号登录 —— 数据下载那份账号不共用")
        return TqApi(auth=TqAuth(user, pwd), web_gui=False)

    def _subscribe(self, api, tq: str, dur: int, length: int, kind: str):
        if kind == "k":
            return api.get_kline_serial(tq, dur, data_length=length)
        return api.get_quote(tq)

    def _drive(self, api) -> bool:
        return bool(api.wait_update(deadline=time.time() + _WAIT_SLICE))


class BrokerChan(_Chan):
    """柜台行情通道：`BrokerApi` 装成 `TqApi` 的样子，一个 hub 订阅多个合约。

    `CtpHub` / `RohonHub` 自己只订一个合约（`subscribe` 会重置 `_builder`），
    但 `BrokerApi` 在上面按合约分桶，`md.subscribe([ins])` 是一次传一个列表，
    所以一个 hub 能同时喂多个品种 —— 这正是行情源要的形状。
    """

    def _connect(self):
        from ttrade.broker_api import BrokerApi

        if self.spec is None:
            raise ValueError(
                f"没有可用的{SOURCE_LABELS.get(self.source, self.source)}账户"
                "（柜台账户要有前置地址，见「账户透视和交易」页）")
        api = BrokerApi(self.spec, feed=self.source, log=self._log)
        api.start()
        if getattr(api, "last_error", ""):
            raise RuntimeError(api.last_error)
        return api

    @staticmethod
    def _log(msg: str) -> None:
        # 行情源的日志走 stderr 不合适（GUI 下没人看），这里只在异常里带出去
        pass

    @property
    def label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)

    def _subscribe(self, api, tq: str, dur: int, length: int, kind: str):
        if kind == "k":
            return api.get_kline_serial(tq, dur, data_length=length)
        return api.get_quote(tq)

    def _drive(self, api) -> bool:
        return bool(api.wait_update(deadline=time.time() + _WAIT_SLICE))


# ------------------------------------------------------------------ 门面


class MarketFeed:
    """全局行情源门面。内部按源建通道，`select()` 切换，其余方法都转发给当前源。"""

    _inst: "MarketFeed | None" = None
    _inst_lock = threading.Lock()

    # ------------------------------------------------------------------ 单例

    @classmethod
    def get(cls) -> "MarketFeed":
        """取全局实例。第一次调用就把**默认源**连起来（程序打开即连）。

        🔴 `start()` 放在锁**外面**：`TqApi` 登录要 3~4 秒，攥着锁去连会把其他
        线程（另一个 runner、界面上的连接按钮）一起堵在 `get()` 上。
        """
        made = None
        with cls._inst_lock:
            if cls._inst is None:
                cls._inst = cls()
                made = cls._inst
        if made is not None:
            made.start()
        return cls._inst

    @classmethod
    def peek(cls) -> "MarketFeed | None":
        """已经建过的实例（没建返回 None）。给「状态栏 / 自检」用，不会触发连接。"""
        return cls._inst

    @classmethod
    def warm_up(cls) -> None:
        """**程序打开就在后台连上默认源**（不阻塞界面）。

        ``TqApi`` 登录要几秒（DNS + 鉴权 + 订阅回填），放主线程会把首帧一起卡住。
        失败不抛：`error` 里留着原因，界面上那个行情源框会显示出来。
        """
        if not AUTO_CONNECT:
            return

        def _run() -> None:
            try:
                cls.get()
            except Exception:                    # noqa: BLE001
                pass

        t = threading.Thread(target=_run, daemon=True, name="feed-warm-up")
        t.start()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._chans: dict[str, _Chan] = {}
        self._source = ""
        self._gen = 0                  # 每切一次源 +1，runner 据此清缓存
        cfg = load_settings()
        self._want = str(cfg.get("source") or SOURCE_KEYS[0])
        if self._want not in SOURCE_KEYS:
            self._want = SOURCE_KEYS[0]
        self._accounts = dict(cfg.get("accounts") or {})

    # ------------------------------------------------------------------ 选源

    @property
    def source(self) -> str:
        return self._source or self._want

    @property
    def generation(self) -> int:
        """切源计数。runner 缓存的 serial 在切源后失效，靠这个发现。"""
        return self._gen

    def chan(self, source: str = "") -> _Chan:
        """取某个源的通道（不存在就建，已建就直接用）。不触发连接。"""
        key = source or self.source
        with self._lock:
            ch = self._chans.get(key)
            if ch is None:
                spec = pick_account(key, self._accounts.get(key, ""))
                cls = TqChan if key == "tq" else BrokerChan
                ch = cls(key, spec)
                self._chans[key] = ch
            return ch

    def start(self) -> bool:
        """连默认源。返回是否连上（失败不抛）。"""
        return self.select(self._want)

    def select(self, source: str) -> bool:
        """切换行情源并连上它。先停掉别的源 —— 三条同时跑会互相抢柜台通道。

        🔴 `close()` **必须在锁外**：柜台通道关闭要等 DLL 回调（几秒），攥着
        `self._lock` 去关，界面线程的 `stats()` 会跟着堵在那儿 —— 表现就是
        「切个行情源，整个界面卡住」。
        """
        if source not in SOURCE_KEYS:
            return False
        stale: list[_Chan] = []
        with self._lock:
            if self._source == source:
                return self.chan(source).ready or self.chan(source).start()
            for key, ch in self._chans.items():
                if key != source:
                    stale.append(ch)
            self._source = source
            self._want = source
            self._gen += 1
            cfg = load_settings()
            cfg["source"] = source
            cfg["accounts"] = dict(self._accounts)
            save_settings(cfg)
        for ch in stale:
            ch.close()
        return self.chan(source).start()

    def set_account(self, source: str, name: str) -> bool:
        """指定某个源用账户表里的哪个账户，顺带重连。"""
        with self._lock:
            self._accounts[source] = str(name or "")
            ch = self._chans.pop(source, None)
            if ch is not None:
                ch.close()
            cfg = load_settings()
            cfg["accounts"] = dict(self._accounts)
            save_settings(cfg)
        return self.select(source) if source == self.source else True

    def account_name(self, source: str = "") -> str:
        key = source or self.source
        spec = self.chan(key).spec
        return getattr(spec, "name", "") if spec is not None else ""

    def sources(self) -> list[dict]:
        """界面用的源清单：每个源一条，带「有没有账户 / 连没连 / 错在哪」。"""
        out = []
        for key in SOURCE_KEYS:
            spec = pick_account(key, self._accounts.get(key, ""))
            ch = self._chans.get(key)
            st = ch.stats() if ch is not None else {}
            out.append({
                "key": key,
                "label": SOURCE_LABELS[key],
                "account": getattr(spec, "name", "") if spec is not None else "",
                "accounts": [s.name for s in _accounts_of(key)],
                "ready": bool(st.get("ready")) if st else False,
                "active": key == self.source,
                "error": st.get("error", "") if st else "",
            })
        return out

    def retry(self) -> bool:
        return self.chan().retry()

    def close(self) -> None:
        with self._lock:
            chans = list(self._chans.values())
        for ch in chans:
            ch.close()

    # ------------------------------------------------------------------ 转发

    def kline(self, tq: str, duration_sec: int, length: int = 3000):
        return self.chan().kline(tq, duration_sec, length)

    def quote(self, tq: str):
        return self.chan().quote(tq)

    # 名字和 `TqApi` / `BrokerApi` 对齐：引擎里本来就写着 `api.get_kline_serial()`
    # / `api.get_quote()` / `api.wait_update()`，把 `api` 换成 `feed` 就能原样用。
    # 少一层「记得它叫 kline 还是 get_kline_serial」的心智负担。
    def get_kline_serial(self, tq: str, duration_sec: int = 300,
                         data_length: int = 3000):
        return self.kline(tq, duration_sec, data_length)

    def get_quote(self, symbol: str):
        return self.quote(symbol)

    def wait_update(self, deadline: float | None = None) -> bool:
        timeout = 0.8 if deadline is None else max(0.0, deadline - time.time())
        return self.wait(timeout)

    # ---- 非阻塞订阅（引擎启动时用它，别让界面对着「卡住」） ----

    def request_kline(self, tq: str, duration_sec: int = 300,
                      length: int = 3000) -> None:
        self.chan().request(tq, duration_sec, length, "k")

    def request_quote(self, tq: str) -> None:
        self.chan().request(tq, 0, 0, "q")

    def peek_kline(self, tq: str, duration_sec: int = 0):
        return self.chan().peek(tq, duration_sec, "k")

    def peek_quote(self, tq: str):
        return self.chan().peek(tq, 0, "q")

    def lock(self):
        return self.chan().lock()

    def read_kline(self, tq: str, copy: bool = True, duration_sec: int = 0):
        return self.chan().read_kline(tq, copy, duration_sec)

    def read_quote(self, tq: str):
        return self.chan().read_quote(tq)

    def wait(self, timeout: float = 0.8) -> bool:
        return self.chan().wait(timeout)

    @property
    def ready(self) -> bool:
        return self.chan().ready

    @property
    def error(self) -> str:
        return self.chan().error

    def stats(self) -> dict:
        st = self.chan().stats()
        st["generation"] = self._gen
        return st

    # ------------------------------------------------------------------ 工具

    @staticmethod
    def is_stale(serial, symbol: str, exchange: str, period: str,
                 tolerance_sec: float = 600.0) -> bool:
        """这条实时序列是不是停在「该到的那根」之前太久。

        柜台 / 断线的序列会一直停在上次填充的位置 —— 此时算出的信号是历史值，
        不该拿去开新仓。判据用 `barclock` 的同一套周期表，不另外写一份。
        """
        try:
            from ttrade import barclock
            from ttrade.tqfetch import local_time

            if serial is None or not len(serial):
                return True
            last = int(serial["datetime"].iloc[-1])
            if last <= 0:
                return True
            exp = barclock.expected_last_bar(symbol, exchange, period)
            if exp is None:
                return False
            # 时区换算复用 `tqfetch.local_time`（UTC 纳秒 → 北京 naive），
            # 自己再写一份 `+8h` 就是第二套判据了
            start = local_time([last])[0]
            return (exp - start).total_seconds() > float(tolerance_sec)
        except Exception:                          # noqa: BLE001
            return False


def _accounts_of(source: str) -> list:
    """某个源能用的账户清单（界面选项框用）。"""
    from ttrade.tqlive import load_accounts

    return [s for s in load_accounts() if getattr(s, "backend", "tq") == source]
