"""大模型接入层。

「交易Agent」页用它做三件事：自然语言对话、把思路写成因子/策略、把一句话变成
一笔待确认的交易指令。这里只管**跟模型说话**，不碰界面、不碰天勤。

## 为什么不用 openai SDK

只需要 `POST /chat/completions` 一个接口，而 openai SDK 会带进 httpx、pydantic
一整套依赖，版本还跟 tqsdk 打架。用 `requests` 手写 SSE 解析，几十行，可控。

## 各家都是 OpenAI 兼容协议

DeepSeek、Kimi、GLM、混元、MiniMax 都提供 `/chat/completions` 的兼容端点，差别只在
`base_url` 和模型名。所以这里只有一张表 + 一个客户端，不写六个适配器。
自定义供应商就是「自己填 base_url 和模型名」，走同一条路。

## token 明文存在本机

和 `accounts.json` 一个口径：`%LOCALAPPDATA%/TrendDesk/agent.json`，不进版本库。
要加密就得管一个主密码，对本机自用工具来说更麻烦。**别把这个文件提交上去。**

## 动作协议

模型要执行操作时，在回答末尾输出一个 ```json 代码块：

    {"action": "trade", "mode": "auto", "symbol": "RB", ...}

界面解析出来**先弹确认框**再执行 —— 大模型说「买 10 手」和真的下单之间必须有一道
人手。`parse_action()` 只负责解析，执行永远在界面层。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

import requests

from ttrade.paths import user_dir

CONF_FILE = "agent.json"


# ------------------------------------------------------------------ 供应商


@dataclass
class Provider:
    key: str
    label: str
    base_url: str
    models: list = field(default_factory=list)
    note: str = ""


# 地址和模型名都**可以改** —— 各家改端点、上新型号很频繁，写死在代码里就会过期。
# 这里给的是「大概率还能用」的默认值，界面上 base_url 与模型名都是可编辑的输入框。
PROVIDERS: list[Provider] = [
    Provider(
        "deepseek", "DeepSeek", "https://api.deepseek.com/v1",
        ["deepseek-chat", "deepseek-reasoner"],
        "OpenAI 兼容。推理型模型（reasoner）不支持温度参数。",
    ),
    Provider(
        "openai", "OpenAI GPT", "https://api.openai.com/v1",
        ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o4-mini"],
        "需要能访问 api.openai.com 的网络。",
    ),
    Provider(
        "moonshot", "Kimi（月之暗面）", "https://api.moonshot.cn/v1",
        ["moonshot-v1-32k", "moonshot-v1-128k", "moonshot-v1-8k"],
        "长上下文，适合整段策略说明。",
    ),
    Provider(
        "zhipu", "GLM（智谱）", "https://open.bigmodel.cn/api/paas/v4",
        ["glm-4-plus", "glm-4-air", "glm-4-flash"],
        "glm-4-flash 免费额度大，适合先试通。",
    ),
    Provider(
        "hunyuan", "混元 HY（腾讯）", "https://api.hunyuan.cloud.tencent.com/v1",
        ["hunyuan-turbos-latest", "hunyuan-large"],
        "腾讯云控制台拿 API Key。",
    ),
    Provider(
        "minimax", "MiniMax", "https://api.minimax.chat/v1",
        ["abab6.5s-chat", "MiniMax-Text-01"],
        "部分账号需要先开通 OpenAI 兼容接口。",
    ),
    Provider(
        "custom", "自定义（任意 OpenAI 兼容）", "", [],
        "填 base_url（含 /v1）和模型名，比如本地 vLLM、Ollama 的 /v1。",
    ),
]


def provider(key: str) -> Provider:
    for p in PROVIDERS:
        if p.key == key:
            return p
    return PROVIDERS[0]


# ------------------------------------------------------------------ 配置


@dataclass
class AgentConfig:
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 120
    # 下单前弹确认框。用户可以勾「下次不提醒」把它关掉 —— 关掉之后模型给的动作
    # 会直接执行，所以设置页里必须能重新打开。
    confirm_trade: bool = True
    # 升级权限：允许 Agent 读写 TrendDesk 目录下的文件（`fs` 动作）。
    # 默认关。打开等于把整个平台的代码交给模型，设置页里必须写清楚这一点。
    allow_fs: bool = False

    def endpoint(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def ready(self) -> tuple[bool, str]:
        """能不能发请求。返回 (能否, 缺什么)。"""
        if not self.base_url.strip():
            return False, "还没填接口地址"
        if not self.model.strip():
            return False, "还没选模型"
        if not self.api_key.strip():
            return False, "还没填 API Key"
        return True, ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentConfig":
        keep = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        cfg = cls(**keep)
        try:
            cfg.temperature = min(2.0, max(0.0, float(cfg.temperature)))
            cfg.max_tokens = max(64, int(cfg.max_tokens))
            cfg.timeout = max(10, int(cfg.timeout))
        except (TypeError, ValueError):
            cfg.temperature, cfg.max_tokens, cfg.timeout = 0.3, 2000, 120
        return cfg


def conf_path() -> str:
    return os.path.join(user_dir(), CONF_FILE)


def load_config() -> AgentConfig:
    path = conf_path()
    if not os.path.exists(path):
        return AgentConfig()
    try:
        with open(path, encoding="utf-8") as fh:
            return AgentConfig.from_dict(json.load(fh))
    except (OSError, ValueError):
        return AgentConfig()


def save_config(cfg: AgentConfig) -> None:
    """原子写。这个文件里有 token，但更重要的是别写坏 —— 写坏了用户要重填一遍。"""
    path = conf_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ------------------------------------------------------------------ 客户端


class AgentError(Exception):
    """网络/接口错误。消息直接给人看，不要包成英文堆栈。"""


class LLMClient:
    """一个 OpenAI 兼容的对话客户端。

    线程无关：界面层把它放进 QThread 里跑，`on_delta` 里 emit 信号即可。
    """

    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg
        self.last_error = ""
        self._resp = None

    def abort(self) -> None:
        """从**另一个线程**掐断正在进行的流式读。

        只靠 `should_stop` 不够：那个标志在 `iter_lines()` 每两行之间才检查一次，
        而模型思考时可能几十秒不推任何字节，此时线程阻塞在 socket 读上，标志形同
        虚设。关掉 response 会让读立刻抛异常，线程才能退出。

        `close()` 是幂等的，和 `with r:` 的退出顺序无关；异常一律吞掉 —— 这里是
        收尾路径，抛出去只会让「停止」变成一个错误弹窗。
        """
        r, self._resp = self._resp, None
        if r is not None:
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass

    # ---- 请求头 ----

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.cfg.api_key.strip()}",
            "Content-Type": "application/json",
        }

    def _body(self, messages: list[dict], stream: bool) -> dict:
        body = {
            "model": self.cfg.model.strip(),
            "messages": messages,
            "stream": stream,
        }
        # 推理型模型（deepseek-reasoner / o 系列）会拒绝 temperature
        name = self.cfg.model.lower()
        if not any(k in name for k in ("reasoner", "o1", "o3", "o4")):
            body["temperature"] = self.cfg.temperature
        if self.cfg.max_tokens > 0:
            body["max_tokens"] = self.cfg.max_tokens
        return body

    @staticmethod
    def _http_error(r: requests.Response) -> str:
        try:
            text = r.text
        except Exception:  # noqa: BLE001
            text = ""
        # 各家错误体格式不一样，但都有一段可读的 message。截前 300 字够定位了。
        snippet = re.sub(r"\s+", " ", text)[:300]
        hint = {
            401: "Key 不对或没生效",
            403: "没有权限（可能没开通这个模型）",
            404: "接口地址或模型名不对",
            429: "额度用完或请求太频繁",
        }.get(r.status_code, "")
        return f"HTTP {r.status_code} {hint}：{snippet}"

    # ---- 对话 ----

    def chat(self, messages: list[dict], on_delta=None, should_stop=None) -> str:
        """发一轮对话，返回完整回答。

        `on_delta(str)` 每收到一段就回调一次，界面拿它做打字机效果；
        `should_stop()` 返回 True 时提前收尾（用户点了「停止」）。
        """
        ok, why = self.cfg.ready()
        if not ok:
            raise AgentError(why)

        url = self.cfg.endpoint("/chat/completions")
        try:
            r = requests.post(
                url,
                headers=self._headers(),
                json=self._body(messages, stream=True),
                stream=True,
                timeout=(10, self.cfg.timeout),
            )
        except requests.RequestException as exc:
            raise AgentError(f"连不上 {url}：{exc}") from exc

        # 记下来，好让 `abort()` 能从界面线程掐断阻塞中的读
        self._resp = r
        try:
            with r:
                if r.status_code >= 400:
                    raise AgentError(self._http_error(r))
                ctype = (r.headers.get("Content-Type") or "").lower()
                if "event-stream" not in ctype:
                    # 有的网关会忽略 stream=true，直接回一个完整 JSON
                    return self._read_plain(r)

                parts: list[str] = []
                for raw in r.iter_lines(decode_unicode=False):
                    if should_stop is not None and should_stop():
                        break
                    if not raw:
                        continue
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue
                    if "error" in obj:
                        raise AgentError(str(obj["error"])[:300])
                    for ch in obj.get("choices") or []:
                        delta = ch.get("delta") or {}
                        piece = delta.get("content") or ""
                        if piece:
                            parts.append(piece)
                            if on_delta is not None:
                                on_delta(piece)
                return "".join(parts)
        finally:
            # 正常结束、出错、被 abort 掐断，三种情况都要清掉，否则下一次 chat()
            # 的 abort() 会去关一个已经死掉的 response
            self._resp = None

    def _read_plain(self, r: requests.Response) -> str:
        try:
            obj = r.json()
        except ValueError as exc:
            raise AgentError("接口返回的不是 JSON，检查 base_url 是否指到了 /v1") from exc
        if "error" in obj:
            raise AgentError(str(obj["error"])[:300])
        try:
            return obj["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentError(f"返回结构不认识：{str(obj)[:200]}") from exc

    # ---- 自检 ----

    def ping(self) -> str:
        """发一条最短的请求，用来验证 Key / 地址 / 模型名。返回耗时描述。"""
        t0 = time.time()
        out = self.chat([{"role": "user", "content": "回复两个字：就绪"}])
        return f"{out.strip()[:20]}（{time.time() - t0:.1f}s）"

    def models(self) -> list[str]:
        """列出可用模型。不是所有供应商都实现，失败就返回空。"""
        try:
            r = requests.get(
                self.cfg.endpoint("/models"), headers=self._headers(), timeout=15
            )
            if r.status_code >= 400:
                return []
            data = r.json()
        except (requests.RequestException, ValueError):
            return []
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        return [str(it.get("id", "")) for it in items if isinstance(it, dict) and it.get("id")]


# ------------------------------------------------------------------ 提示词


SYSTEM_PROMPT = """你是「趋势跟踪研究台」里的交易助手，服务对象是一位做国内商品期货和美股的个人交易者。
你的回答必须中文、简洁、给具体数字，不要客套话。

## 平台里有什么

- 数据：国内商品期货 78 个品种（日线 20 年，5 分钟/1 小时 10 年，天勤主连 + 真实合约），美股 34 只（东财后复权）
- 47 个内置因子：趋势/反转/波动/量能/形态/展期/期权七类，每个都有数学公式与实现代码；
  用户注册的自定义因子会追加进同一个库
- 5 套内置规则策略：双均线、唐奇安突破、ATR 通道、MACD、海龟（突破 + ATR 止损），
  用户注册的自定义策略会追加进同一个库
- 因子策略三种模式：单因子、多因子等权、机器学习多因子（含线性模型与图神经网络）
- 回测：次日开盘成交、双边成本、逐笔保证金、不强平（亏穿也跑到底）；方向由滚动 IC 决定，不用未来信息
- 实盘：天勤 TqSdk，支持快期模拟与实盘账户，多策略组合、追单、持仓对齐

## 你要做的

1. 用户问数据/策略/因子时，先给结论，再给依据（数字、区间、样本数）
2. 用户描述一个想法时，先看**平台已有的因子或策略**能不能表达；确实没有的，
   按下面「注册新因子」写成 factor 动作块，不要让用户自己去填表
3. 用户要下单时，先把「买什么、多少手、什么价、为什么」列清楚，再给动作块
4. 不确定的地方直接说不确定，不要编造回测数字 —— 你没跑过回测

## 动作块

需要让平台执行操作时，在回答**最后**单独输出一个 json 代码块，平台会弹确认框：

```json
{"action": "trade", "mode": "auto", "symbol": "RB", "exchange": "SHFE",
 "period": "1d", "strategy": "ma_cross", "params": {"fast": 50, "slow": 200},
 "volume": 1, "reason": "一句话说明"}
```

- 按策略交易：`"mode": "auto"`，带 `strategy` 与 `params`
- 主观买卖：`"mode": "manual"`，带 `"side": "BUY"|"SELL"`、`"offset": "OPEN"|"CLOSE"|"CLOSETODAY"`
- 只想回测不交易：把 `action` 换成 `backtest`，其余字段同上但不带 `mode`：

```json
{"action": "backtest", "symbol": "RB", "exchange": "SHFE", "period": "1d",
 "strategy": "ma_cross", "params": {"fast": 50, "slow": 200}}
```

- 要往库里加**新策略**：`{"action": "strategy", ...}`，见下
- 需要查平台手册时：`{"action": "doc", ...}`，见下
- 不需要执行任何操作时，**不要输出 json 块**

## 平台手册（写因子之前必查）

平台有一份手册，写清了行情是怎么来的（三条数据链路、**每条链路各自的列名**、
复权口径、缺数据长什么样）、平台往命名空间里注入了哪些原语（`_ret` / `_col` /
`_oi` 的真实源码）、以及全部因子的完整实现。

**不要凭印象猜列名**。写因子、或者回答「这个数据从哪来」之前，先查手册：

```json
{"action": "doc", "topic": "factors"}
```

| topic | 内容 |
|---|---|
| `data` | 行情数据：链路、列名、复权、缺数据 |
| `primitives` | 原语源码（`_ret`、`_col`、`_oi`、`standardize` 等） |
| `contract` | 写因子/策略的契约与常见错误 |
| `factors` | 全部因子索引（一行一个，先看这个） |
| `cat:趋势` | 整类因子的完整实现（趋势/反转/波动/量能/形态/展期/期权） |
| `factor:mom_20` | 单个因子的完整实现 |

一次只查一段，别一口气把整份手册拉进对话。

**注册失败、或者试算报错时，先查手册再改**。平台会把报错原样发给你，那是最准的
线索：说「长度不一致」就去 `contract` 看返回值的约定，说「没有名为 xxx 的函数」
就是 `func` 和代码里的函数名对不上。不要凭猜测连着改三次同一个地方。

## 附件

用户可能直接丢图片或文档进来：

- **图片**（截图、行情图、别人的策略截图）：图像本身会随消息传给你，直接看。
  看不清就说看不清，不要猜数字
- **文档**（md/txt/csv/json/docx）：正文会以「【附件 N：文件名】」开头拼在消息里，
  那是文件内容，不是用户说的话

拿到附件先判断它要你干什么：是让你照着重写一个因子，还是让你核对一份数据。
**不要问「附件里是什么」** —— 你能看见。

## 注册新因子

平台**支持运行时注册因子**，不用让用户手动去因子管理页填表。用户描述一个平台
没有的想法时，把它写成动作块，用户确认后就会进因子库：

```json
{"action": "factor", "key": "vol_price_corr", "label": "量价相关·20 日",
 "category": "量能", "hint_dir": 0,
 "formula": "corr(Δlog(close), Δlog(volume), 20)",
 "desc": "成交量和价格同向变化，说明这段趋势有资金跟随",
 "func": "f_vol_price_corr",
 "code": "def f_vol_price_corr(df):\\n    c = _col(df, 'close').pct_change()\\n    v = _col(df, 'volume').pct_change()\\n    return c.rolling(20).corr(v)"}
```

写代码的硬约束，违反任何一条都会注册失败：

1. **签名固定** `def f_xxx(df) -> pd.Series`，`df` 的索引是 datetime，一定有
   `open/high/low/close/volume`。持仓量**不要直接写列名**：日线 CSV 叫
   `open_interest`、天勤分钟线叫 `open_oi`，写死一个在另一种数据上就是全 NaN。
   用 `_oi(df)`，两个名字都认
2. **不要写 import**。`np`、`pd` 已经注入；`_col(df, 列名)` 取衍生列（没有该列
   时返回全 NaN，不要自己判空）、`_ret(c, n)`、`_safe_div(a, b)` 也可用。
   完整清单和源码查 `doc` 动作的 `primitives`
3. **返回等长 Series，索引和 df 完全一致**。不要 `dropna()`、不要 `reset_index()`
   —— 预热期的 NaN 要留着，下游统一处理
4. **必须 PIT**：只用 `rolling` / `ewm` / `shift(正数)`，**绝不出现 `shift(-n)`**。
   用未来数据的因子在回测里会好得离谱、实盘一分钱赚不到
5. 因子只算**原始值**，标准化（滚动 z-score）平台统一做，不要自己减均值除标准差
6. `key` 用英文小写加下划线，`label` 用中文；`hint_dir` 是历史倾向（+1 高分偏涨、
   -1 偏跌、0 无明显方向），**只是展示提示**，回测里方向仍由滚动 IC 决定

如果这个想法用现有因子组合就能表达（比如「动量除以波动」），优先建议用多因子模式，
不要为此新增因子 —— 库里多一个几乎重复的因子，比少一个更难维护。

## 注册新策略

平台**也支持运行时注册规则策略**。用户描述一套平台没有的进出场逻辑时，写成动作块：

```json
{"action": "strategy", "key": "sbbox_mini", "label": "缩量突破·迷你",
 "desc": "突破 N 日高点入场，波动率过高时不开仓，跌破 M 日低点离场",
 "params": {"entry": 240, "exit": 720, "vol_max": 0.4, "atr_n": 200},
 "param_meta": {"entry": [20, 960, 20], "exit": [60, 1440, 20]},
 "func": "sig_sbbox_mini",
 "code": "def sig_sbbox_mini(df, entry=240, exit=720, vol_max=0.4, atr_n=200):\\n    out = df.copy()\\n    hi = out['high'].rolling(entry).max().shift(1)\\n    lo = out['low'].rolling(exit).min().shift(1)\\n    vol = out['close'].pct_change().rolling(20).std()\\n    out['signal'] = _position_state(\\n        long_in=out['close'] > hi,\\n        long_out=out['close'] < lo,\\n        short_in=out['close'] < lo,\\n        short_out=out['close'] > hi,\\n    )\\n    out.loc[vol > vol_max, 'signal'] = 0\\n    return out"}
```

硬约束，违反任何一条都会注册失败：

1. **签名 `def sig_xxx(df, 参数...)`**，`df` 索引是 datetime，含
   `open/high/low/close/volume`（持仓量用 `_oi(df)`）
2. **不要写 import**。`np`、`pd`、`sma`、`ema`、`atr`、`true_range`、
   `_position_state(多头进/多头出/空头进/空头出)` 都已注入，直接用
3. **返回带 `signal` 列的等长 DataFrame**。`signal` 是当日收盘后的目标仓位：
   1 满仓多、0 空仓、-1 满仓空，只能取这三个值
4. **必须 PIT**：只用 `rolling` / `ewm` / `shift(正数)`，**绝不出现 `shift(-n)`**
5. **只管出仓位**：不要自己算手续费、滑点、手数、资金曲线，回测引擎统一处理
6. 参数放进 `params`（带默认值），代码里同名形参接住；`param_meta` 可选，不写就按
   默认值自动推滑杆范围
7. `key` 用英文小写加下划线，`label` 用中文

如果现有 5 套策略加参数就能覆盖（比如「海龟把 entry 调成 240」），**先用 backtest
动作试跑**，不要为此新增策略 —— 库里多一个几乎重复的策略更难维护。
"""


# 只在「升级权限」打开时追加。默认权限下模型根本不知道有这回事，
# 也就不会写出一个必然被拒的动作块。
FS_PROMPT = """

## 升级权限：读写平台文件

用户已打开「升级权限」，你可以读写 TrendDesk 根目录（`{root}`）下的文件。
改代码之前先 `read` 看清楚现状，改完告诉用户改了什么、要重启哪一部分才生效。

```json
{"action": "fs", "op": "read", "path": "ttrade/factors.py"}
```

```json
{"action": "fs", "op": "list", "path": "ttrade"}
```

```json
{"action": "fs", "op": "write", "path": "notes/螺纹.txt", "content": "第一行\\n第二行"}
```

- `path` 只能是**相对根目录**的路径，不能出现 `..`，不能是绝对路径
- `write` 是**整文件覆盖**，不是打补丁：先 `read` 拿全文，改完把全文写回去。
  只写你真正要改的那几行，别顺手重排格式 —— 用户看不懂 diff 就没法复核
- 写之前平台会弹确认框给人看内容，所以一次只写一个文件
- 改 `gui/` 或 `ttrade/` 下的代码都要重启程序才生效；配置文件一般也是重启读一次
- 不确定要改哪个文件时，先 `list` 再看，别猜路径
"""


def _system_prompt(allow_fs: bool) -> str:
    """基础提示词 + 升级权限那段。根目录在每次拼的时候现取 ——
    源码运行和打包运行不是同一个目录，写死进提示词会让模型照着错的路径去改。"""
    if not allow_fs:
        return SYSTEM_PROMPT
    try:
        from ttrade import agent_fs as FS

        root = FS.root().replace("\\", "/")
    except Exception:  # noqa: BLE001 - 取不到路径也不该让整轮对话发不出去
        root = "（取不到，用相对路径即可）"
    return SYSTEM_PROMPT + FS_PROMPT.replace("{root}", root)


def text_of(content) -> str:
    """把消息内容摊平成纯文本。

    带附件的消息 `content` 是 parts 列表（`[{"type": "text", ...},
    {"type": "image_url", ...}]`）。记忆压缩、长度统计这些地方只想要文字，
    直接 `len(content)` 数出来的是「2 个部件」，不是字数。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                out.append(str(p.get("text") or ""))
            elif p.get("type") == "image_url":
                out.append("（一张图片）")
        return "\n".join(x for x in out if x)
    return str(content or "")


def _prune_images(history: list[dict], keep: int = 2) -> list[dict]:
    """只保留最近 `keep` 条带图消息的图像数据。

    一张截图 base64 后两三百 KB，四十条历史全留着就是十几兆的请求体 —— 又慢又
    容易被网关按体积拒掉。图早就被模型看过了，隔了几轮再传一遍没有意义。
    """
    out = list(history)
    seen = 0
    for i in range(len(out) - 1, -1, -1):
        c = out[i].get("content")
        if not isinstance(c, list):
            continue
        if not any(isinstance(p, dict) and p.get("type") == "image_url" for p in c):
            continue
        seen += 1
        if seen <= keep:
            continue
        out[i] = {
            "role": out[i].get("role", "user"),
            "content": [
                {"type": "text", "text": "（早前发过的图片，为省体积这里不再重复传）"}
                if isinstance(p, dict) and p.get("type") == "image_url" else p
                for p in c
            ],
        }
    return out


def build_messages(history: list[dict], user_text: str | list, memory: str = "",
                   allow_fs: bool = False) -> list[dict]:
    """拼一份完整消息列表：系统提示 + 长期记忆 + 历史 + 本轮。

    记忆单独占一条 system 消息，不拼进 `SYSTEM_PROMPT`：一是基础提示词不变，
    各家服务端的上下文缓存还能命中；二是调试时一眼能看出「这句是记忆喂进去的」。

    `allow_fs` 为真时追加「读写平台文件」那段。**这是能力开关，不是提示语**：
    关着的时候提示词里不提这件事，模型就不会写出必然被拒的动作块。

    `user_text` 可以是字符串，也可以是带图的多模态 parts 列表（见
    `attachments.content()`）—— 这里只负责原样放进消息，不关心它是什么形态。
    """
    msgs = [{"role": "system", "content": _system_prompt(allow_fs)}]
    if memory.strip():
        msgs.append({
            "role": "system",
            "content": "## 长期记忆\n\n以下是你在以前的对话里自己压缩出来的要点，"
                       "不是用户这一轮说的。用到时当作已知事实，不要向用户复述这份清单。\n\n"
                       + memory.strip(),
        })
    for m in _prune_images(history):
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs


MEMORY_PROMPT = """把下面这段「交易助手 ↔ 用户」的对话压缩成长期记忆，供以后接着聊时参考。

只写以后还用得上的：
- 用户关注的标的、周期、账户、手数、风险口径
- 已经跑出来的数字和结论（收益、夏普、参数、阈值）
- 已经做过的决定（注册了哪个因子/策略、跑过什么、否决了什么）
- 用户明确表达过的偏好和要求

不要写：客套话、你的推理过程、被否决的中间方案、代码。
不要写「用户问了…」这种过程描述，直接写事实。
一条一行，最多 6 行，每行不超过 40 字。没有值得记的就只回两个字：无。

对话：
"""

_CODE_BLOCK = re.compile(r"```.*?```", re.S)


def memory_messages(convo: list[dict]) -> list[dict]:
    """把一段对话拼成「压缩成记忆」的请求。

    代码块先剔掉：模型给的因子/策略源码能占上千字，压进记忆既没用又会把
    真正该记的数字挤掉。
    """
    lines = []
    for m in convo:
        role = "用户" if m.get("role") == "user" else "助手"
        text = _CODE_BLOCK.sub("（此处有一段代码或数据块，已省略）", text_of(m.get("content")))
        lines.append(f"{role}：{text.strip()}")
    return [{"role": "user", "content": MEMORY_PROMPT + "\n".join(lines)}]


# ------------------------------------------------------------------ 动作解析

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


@dataclass
class Action:
    kind: str                 # trade / backtest / factor / strategy / fs / 空串表示没有动作
    payload: dict = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.kind) and not self.error

    def describe(self) -> str:
        """一行说明，给确认框用。"""
        p = self.payload
        sym = f"{p.get('symbol', '')}_{p.get('exchange', '')}".strip("_")
        if self.kind == "trade" and p.get("mode") == "manual":
            side = "买入" if p.get("side") == "BUY" else "卖出"
            off = {"OPEN": "开仓", "CLOSE": "平仓", "CLOSETODAY": "平今"}.get(
                p.get("offset", "OPEN"), p.get("offset", "")
            )
            return f"{side}{off} {sym} {p.get('volume', 1)} 手"
        if self.kind == "trade":
            return (f"按策略交易 {sym} {p.get('period', '')}｜"
                    f"{p.get('strategy', '')}｜{p.get('volume', 1)} 手")
        if self.kind == "backtest":
            return f"回测 {sym} {p.get('period', '')}｜{p.get('strategy', '')}"
        if self.kind == "factor":
            return f"注册因子「{p.get('label', '')}」（{p.get('category', '')}）"
        if self.kind == "strategy":
            n = len(p.get("params") or {})
            return f"注册策略「{p.get('label', '')}」（{n} 个参数）"
        if self.kind == "fs":
            op = p.get("op", "")
            path = p.get("path", "")
            if op == "read":
                return f"读取 {path}"
            if op == "list":
                return f"列出目录 {path or '.'}"
            n = len(p.get("content") or "")
            return f"写入 {path}（{n} 字符）"
        if self.kind == "doc":
            return f"查平台手册：{p.get('topic', '')}"
        return self.kind


def _parse_factor(obj: dict) -> Action:
    """注册因子的动作块。字段校验只做「能不能落地」，不评判因子好坏。"""
    key = str(obj.get("key") or "").strip().lower()
    label = str(obj.get("label") or "").strip()
    code = str(obj.get("code") or "")
    if not key:
        return Action("factor", obj, "缺少因子代号 key")
    if not label:
        return Action("factor", obj, "缺少显示名 label")
    if not code.strip():
        return Action("factor", obj, "缺少实现代码 code")

    func = str(obj.get("func") or f"f_{key}").strip()
    if f"def {func}" not in code:
        return Action("factor", obj, f"code 里要定义 def {func}(df)")

    obj["key"] = key
    obj["label"] = label
    obj["func"] = func
    obj["code"] = code
    obj["category"] = str(obj.get("category") or "自定义").strip()
    obj["desc"] = str(obj.get("desc") or "").strip()
    obj["formula"] = str(obj.get("formula") or "").strip()
    try:
        hint = int(obj.get("hint_dir", 1))
    except (TypeError, ValueError):
        hint = 1
    obj["hint_dir"] = hint if hint in (-1, 0, 1) else 1
    return Action("factor", obj)


def _parse_strategy(obj: dict) -> Action:
    """注册策略的动作块。只校验「能不能落地」，不评判策略好坏 ——
    好坏由回测数字说话，不由这里拦。"""
    key = str(obj.get("key") or "").strip().lower()
    label = str(obj.get("label") or "").strip()
    code = str(obj.get("code") or "")
    if not key:
        return Action("strategy", obj, "缺少策略代号 key")
    if not label:
        return Action("strategy", obj, "缺少显示名 label")
    if not code.strip():
        return Action("strategy", obj, "缺少实现代码 code")

    func = str(obj.get("func") or f"sig_{key}").strip()
    if f"def {func}" not in code:
        return Action("strategy", obj, f"code 里要定义 def {func}(df, ...)")

    params = obj.get("params") or {}
    if not isinstance(params, dict):
        return Action("strategy", obj, "params 要是一个对象")
    for k, v in params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return Action("strategy", obj, f"参数 {k} 的值 {v!r} 不是数字")

    meta = obj.get("param_meta") or {}
    if not isinstance(meta, dict):
        meta = {}

    obj["key"] = key
    obj["label"] = label
    obj["func"] = func
    obj["code"] = code
    obj["desc"] = str(obj.get("desc") or "").strip()
    obj["params"] = params
    obj["param_meta"] = meta
    return Action("strategy", obj)


def _parse_fs(obj: dict) -> Action:
    """读写平台文件的动作块。真正的路径校验在 `agent_fs.resolve()` 里 ——
    这里只挡住一眼就能看出来的错，让模型能立刻改对。"""
    op = str(obj.get("op") or "").strip().lower()
    if op not in ("read", "list", "write"):
        return Action("fs", obj, "op 只能是 read / list / write")
    path = str(obj.get("path") or "").strip().replace("\\", "/")
    if not path and op != "list":
        return Action("fs", obj, "缺少 path")
    if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        return Action("fs", obj, "path 只能是相对 TrendDesk 根目录的路径")
    if ".." in path.split("/"):
        return Action("fs", obj, "path 里不能出现 ..")
    if op == "write" and "content" not in obj:
        return Action("fs", obj, "写入要带 content（整文件覆盖）")

    obj["op"] = op
    obj["path"] = path or "."
    obj["content"] = str(obj.get("content") or "")
    return Action("fs", obj)


def _parse_doc(obj: dict) -> Action:
    """查手册的动作块。topic 认不认识交给 `factor_manual.resolve()` ——
    那里解析不了会回一份索引，比在这里拒掉更有用。"""
    topic = str(obj.get("topic") or "").strip()
    if not topic:
        return Action("doc", obj, "缺少 topic（data / primitives / contract / factors / "
                                  "cat:趋势 / factor:<key>）")
    obj["topic"] = topic
    return Action("doc", obj)


def parse_action(text: str) -> Action:
    """从回答里挖出动作块。没有就返回 kind 为空串的 Action。

    只认最后一个代码块：模型有时会在解释里贴示例，最后一个才是它真要执行的。
    """
    blocks = _JSON_BLOCK.findall(text or "")
    if not blocks:
        return Action("")
    raw = blocks[-1]
    try:
        obj = json.loads(raw)
    except ValueError:
        return Action("", {}, "动作块不是合法 JSON")
    if not isinstance(obj, dict):
        return Action("", {}, "动作块不是对象")

    kind = str(obj.get("action") or "").strip().lower()
    if kind not in ("trade", "backtest", "factor", "strategy", "fs", "doc"):
        return Action("")
    if kind == "factor":
        # 注册因子不需要品种 —— 因子是横截面的，对哪个标的算由用户决定
        return _parse_factor(obj)
    if kind == "strategy":
        # 注册的是策略**定义**，也不带品种：同一套逻辑要跑哪个品种是另一件事
        return _parse_strategy(obj)
    if kind == "fs":
        # 也不带品种：改哪个文件是 path 说了算
        return _parse_fs(obj)
    if kind == "doc":
        # 查手册更不需要品种
        return _parse_doc(obj)
    if not obj.get("symbol"):
        return Action(kind, obj, "缺少品种")

    try:
        obj["volume"] = max(1, int(obj.get("volume", 1)))
    except (TypeError, ValueError):
        obj["volume"] = 1
    if kind == "trade":
        mode = str(obj.get("mode") or "auto").lower()
        if mode not in ("auto", "manual"):
            return Action(kind, obj, "mode 只能是 auto 或 manual")
        obj["mode"] = mode
        if mode == "manual":
            side = str(obj.get("side") or "").upper()
            if side not in ("BUY", "SELL"):
                return Action(kind, obj, "主观交易要给 side=BUY/SELL")
            obj["side"] = side
            off = str(obj.get("offset") or "OPEN").upper()
            if off not in ("OPEN", "CLOSE", "CLOSETODAY"):
                return Action(kind, obj, "offset 只能是 OPEN/CLOSE/CLOSETODAY")
            obj["offset"] = off
        elif not obj.get("strategy"):
            return Action(kind, obj, "按策略交易要给 strategy")
    return Action(kind, obj)


# ------------------------------------------------------------------ 历史

MAX_HISTORY = 40


def trim_history(history: list[dict], limit: int = MAX_HISTORY) -> list[dict]:
    """只留最近若干轮。整段历史塞进去，长对话很快就爆上下文。"""
    if len(history) <= limit:
        return list(history)
    return list(history[-limit:])
