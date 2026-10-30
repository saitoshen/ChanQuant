# 交易 Agent 改造方案：Skill 体系 + 文档能力 + 消息渲染

> 状态：**设计稿，未动代码**。写完等拍板再实施。
> 依据：`ttrade/agent.py`（988 行）、`gui/agent_page.py`（2223 行）、
> `ttrade/attachments.py`、`ttrade/agent_fs.py`、`ttrade/factor_pdf.py` 的现状通读，
> 加上 Qt 富文本能力的实机探测（见 §5.3）。

---

## 0. 结论摘要

四个判断，先说清楚：

1. **没有 tool registry。** 现在是「系统提示词里手写 6 种动作说明 + `parse_action` 里一个
   白名单 + 每个动作一个 `_parse_*` + `_handle_action` 里一个 `if` 分支」。加一个工具要
   改 4 处代码加 1 段提示词，而且没有任何参数 schema 校验 —— 全靠手写 `if`。
   `trade` / `backtest` 不是"注册"进去的，是**枚举值写死**在 `agent.py:939`。

2. **聊天不友好的根因不是配色，是模型回复从来没做过渲染。**
   `_on_delta` 用 `insertPlainText` 逐字追加（`agent_page.py:1361`），`_on_done` 收尾也
   不回头重排。所以 `**加粗**`、`| 表格 |`、```代码块``` 全是原文照贴。先修这个，
   收益比做任何卡片都大。

3. **你的 skill 拆分方向对，但有 3 个结构性问题**：`skill_doc` 把"本地只读"和"出网"
   混在一个开关里；`skill_backtest` 把"算"和"输出"混在一起；`skill_monitor` 目前
   **做不了**（平台没有后台常驻和托盘通知，已确认无 `QSystemTrayIcon`）。另外缺了
   最常用的 `skill_data`（查因子值 / 查行情）。

4. **图片粘贴不用装任何包。** Qt 自带剪贴板、拖放、`data:image/png;base64` 内联渲染
   （§5.3 实测支持）。真正要装的只有 `pypdf` 和 `openpyxl` 两个，加起来约 8MB 纯
   Python，不动安装包体积的量级。

---

## 1. 现状盘点

### 1.1 代码骨架

| 模块 | 行数 | 职责 |
|---|---|---|
| `ttrade/agent.py` | 988 | Provider 表、`AgentConfig`、`LLMClient`（OpenAI 兼容 + SSE 流式）、`SYSTEM_PROMPT`（5495 字符）、`parse_action`、历史裁剪 |
| `gui/agent_page.py` | 2223 | 左侧因子/策略/报告三页、中间对话、右侧详情、`ConfirmDialog`、附件条、记忆压缩 |
| `ttrade/attachments.py` | ~280 | 图片→base64 data URI（长边缩到 1600）、文本类→纯文本、docx 自己解 zip |
| `ttrade/agent_fs.py` | ~200 | 「升级权限」下的文件读写沙箱（四道闸：根目录内 / 写必确认 / 大小上限 / 原子替换） |
| `ttrade/factor_manual.py` | — | 手册 topic 解析（`data` / `primitives` / `contract` / `factors` / `cat:趋势` / `factor:xxx`） |
| `ttrade/agent_memory.py` | — | 长期记忆压缩（每 16 条压一次，保留最近 8 条） |

### 1.2 现有的「工具」是怎么定义的

6 种动作，全部硬编码：

| action | 风险 | 确认框 | 落点 |
|---|---|---|---|
| `trade` | **高** | 弹（受 `confirm_trade` 开关控制） | `trade_requested` 信号 → 交易页 |
| `backtest` | 只读 | 不弹 | `_after_tool` 回灌 |
| `factor` | 写（执行模型写的代码） | 必弹，摊开代码 | 因子库 |
| `strategy` | 写（同上） | 必弹，摊开代码 | 策略库 |
| `fs` | 读写本机 | 写必弹 / 读不弹 | `agent_fs` 沙箱 |
| `doc` | 只读 | 不弹 | `factor_manual.resolve` |

加第 7 个工具要动的 4 处：

1. `agent.py:939` 的 `if kind not in ("trade", "backtest", ...)` 白名单
2. 写一个 `_parse_xxx()` 做字段校验
3. `Action.describe()` 里加一个 `if`（`agent.py:786`）
4. `_handle_action()` 里加一个 `if`（`agent_page.py:1512`）
5. 外加 `SYSTEM_PROMPT` 里手写一段用法说明

### 1.3 提示词成本

- `SYSTEM_PROMPT` 5495 字符，**每轮都发**
- 外加 `strategy_catalog()` 现拼的策略库清单（`strategy_lib.json` 66KB，模板多的时候
  这一段的体积会超过基础提示词）
- 外加 `Mem.brief()` 记忆、手册内容、工具回灌结果

**问题**：不管这轮用不用得上 PDF、网页、下单，那 5495 字符里关于它们的说明都在发。
改成「按启用的能力包注入」能直接砍掉三到四成。

### 1.4 附件现在的处理方式（doc 能力的最大障碍）

`attachments.load()` 一次读完就摊平：

- 图片 → base64（进多模态 parts）
- 文本/md/csv → 整份纯文本，最多 6 万字，拼进 user 消息
- docx → 解 `word/document.xml`，**表格结构丢失**
- **pdf → 没装 pypdf，现在直接报"读不了"**

后果：一份 200 行的因子 Excel，会被压成 6 万字纯文本塞进上下文，模型还看不见列对齐；
而且**读完之后没有第二次机会** —— 模型想问"第 2 个 sheet 呢"，只能让用户重发。

---

## 2. Skill 清单评审

### 2.1 先厘清两层概念

你的清单里 `skill_doc` 是"包"、`read_excel` 是"工具"，这个层级是对的，但要在注册表里
写实，不能混着叫：

| 层 | 是什么 | 作用 |
|---|---|---|
| **Skill（能力包）** | 一组工具 + 一段 prompt 片段 + 默认权限 + 开关 | 权限单元、token 单元 |
| **Tool（工具）** | 一次可调用动作（= 现在的 `Action.kind`） | 执行单元 |

**执行仍然是扁平的**：`Action.kind` 直接等于 tool 名，老 JSON 不用改。Skill 只是给工具
分组，让权限开关和提示词注入有地方挂。

为什么值得分两层：现在 5495 字符的提示词是"全量常驻"，分层之后只有启用的包才注入自己
那段说明。关掉 `web` 包，网页那段 600 字符就不发了。

### 2.2 逐条评审

| 你的分法 | 判断 | 问题 | 建议 |
|---|---|---|---|
| `skill_doc`：read_excel / parse_pdf / web_fetch | ⚠ 拆得过宽 | 三件事的风险和依赖完全不同：前两个是**本地只读**，`web_fetch` 是**出网**（SSRF、超大响应、内网探测），不该共用一个开关 | 拆成 `doc`（read_excel / parse_pdf / read_docx / write_pdf / write_xlsx）和 `web`（fetch / search） |
| `skill_factor`：factor_register / 查询 / 校验 | ✅ 合理，缺一个 | register 是**写**（要确认），查询是**读**（不该弹框）；"校验"其实就是试算，`factor_profile` 已有实现 | 保留，补 `factor_value`（查某因子在某品种某段的实际数值序列） |
| `skill_strategy`：strategy_register / 查询 | ✅ 合理 | 同 factor，register 是写 | 保留，和 factor 共用"摊开代码"确认框 |
| `skill_backtest`：run / 回测报告生成 | ⚠ 职责混了 | "报告生成"不是回测，是**输出格式**，和因子 PDF 手册是同一套 `QPdfWriter` | backtest 只留 `run` / `compare`；报告生成挪到 `doc` 的 `write_pdf` |
| `skill_trade`：下单 / 仓位查询 | ✅ 合理，风控设计不够 | "可单独开关"不够细。要回答：开关管到哪一层、下单前拦不拦、日限额怎么算 | 见 §3.3，复用已有的 `ttrade/daylimit.py` |
| `skill_monitor`：品种监控 / 告警生成 | ❌ 暂时做不了 | 平台没有后台常驻、没有托盘图标、没有通知通道（已确认）。做"订阅式"要先建一套常驻基础设施 | 降到 P3。先做**快照式**（用户问就现算一次），不做**订阅式**（后台轮询 + 推送） |
| （缺）`skill_data` | 🔴 最常用却没列 | "螺纹最近 20 天动量多少""这个因子今天多少分"—— 现在模型只能靠跑回测间接算，或者答"我没跑过回测" | 加，只读、零风险，**优先级最高** |
| （缺）`skill_memory` | 🟡 可选 | 有 `agent_memory` 但模型不能主动写 | 加 `remember(key, value)`，低风险，P2 |

### 2.3 建议的最终清单：7 个包 / 17 个工具

| 包 | 工具 | 风险 | 优先级 |
|---|---|---|---|
| **data** | `query_bars`（某品种某段 K 线摘要）<br>`query_factor`（某因子在某段的数值 / 分位）<br>`list_symbols`（可用品种与周期） | read | **P1** |
| **doc** | `read_excel`（按 sheet / 列 / 行段取）<br>`parse_pdf`（抽文本 + 表格）<br>`read_docx`（含表格）<br>`write_pdf`（HTML → PDF，复用 QPdfWriter）<br>`write_xlsx`（DataFrame → xlsx） | read / write | **P1** |
| **web** | `web_fetch`（URL → 正文）<br>`web_search`（需 API key，可后置） | read（出网） | **P2** |
| **factor** | `factor_register`<br>`factor_list`（按类别 / 关键词查）<br>`factor_value`（= data.query_factor 的别名，见下） | write / read | P1 |
| **strategy** | `strategy_register`<br>`strategy_list`<br>`strategy_run`（跑一次给定配置） | write / read | P1 |
| **backtest** | `backtest_run`<br>`backtest_compare`（多参数/多策略对比） | read | P1 |
| **trade** | `trade_order`（下单）<br>`query_position`（持仓 / 可用资金 / 挂单） | **trade** | P2 |
| **monitor** | `snapshot_watch`（一次性快照：涨跌幅 / 突破 / 波动异常） | read | **P3** |

> `factor_value` 和 `data.query_factor` 重复了 —— 只保留一个，挂在 `data` 下。
> `factor` 包只管"定义"（注册 / 查定义 / 校验），`data` 包只管"取值"。这条边界要守住，
> 否则以后会到处出现"这个查询到底归谁"。

---

## 3. 注册表设计

### 3.1 数据结构

新增 `ttrade/skills/` 包：

```
ttrade/skills/
    __init__.py       # REGISTRY、catalog()、dispatch()
    base.py           # Tool / Skill / Block / SkillResult / 校验
    data.py           # query_bars / query_factor / list_symbols
    doc.py            # read_excel / parse_pdf / read_docx / write_pdf / write_xlsx
    web.py            # web_fetch / web_search
    factor.py         # factor_register / factor_list
    strategy.py       # strategy_register / strategy_list / strategy_run
    backtest.py       # backtest_run / backtest_compare
    trade.py          # trade_order / query_position
```

```python
@dataclass(frozen=True)
class Tool:
    name: str                    # "read_excel"，= Action.kind
    group: str                   # "doc"，所属能力包
    label: str                   # "读 Excel"
    summary: str                 # 一行，进 prompt 目录
    schema: dict                 # 参数校验（见 3.2）
    risk: str                    # "read" / "write" / "trade"
    handler: Callable            # (ctx, params) -> SkillResult
    prompt: str = ""             # 详细用法，只在启用该包时注入
    max_turns: int = 1           # 允许模型连调几次（read 类可以多）

@dataclass
class SkillResult:
    ok: bool
    summary: str                 # 一行回执，进对话
    blocks: list[dict]           # 结构化块，见 §5.2
    error: str = ""
```

注册用装饰器，模块导入即完成：

```python
@register("doc")
def read_excel(ctx, params): ...
```

### 3.2 参数校验：轻量 schema，不引 jsonschema

```python
schema = {
    "file_id": {"type": "str", "required": True},
    "sheet":   {"type": "str|int", "default": 0},
    "rows":    {"type": "int", "default": 20, "min": 1, "max": 200},
    "cols":    {"type": "list[str]", "default": None},
}
```

一个 40 行的 `validate(params, schema) -> (clean, error)` 就够。理由：`jsonschema`
这个包 500KB 起步，而我们只需要"类型 + 必填 + 范围 + 枚举"四件事，现在的 `_parse_*`
里手写的那些 `if` 本来就是在做这个。

**保留专属校验**：`factor_register` 里"`code` 必须定义 `def f_xxx`"这种语义检查留在
handler 里，schema 表达不了。

### 3.3 风险分级与权限

四级，比现在的两个开关（`confirm_trade` / `allow_fs`）细一档：

| 级别 | 行为 | 例 |
|---|---|---|
| `read` | 自动执行，不弹框 | 回测、查数据、读 Excel、查手册、读文件 |
| `write` | 弹确认，**摊开要写的内容** | 注册因子/策略、写文件、导出 PDF |
| `net` | 弹确认（首次），之后记住 | `web_fetch` |
| `trade` | 弹确认 + **红框 + 手数/价格/理由三行大字** + 日限额预检 + 可全局关 | `trade_order` |

配置项（`AgentConfig` 扩展）：

```python
enabled_groups: list[str]   # ["data", "doc", "factor", "strategy", "backtest"]
allow_write: bool = False   # 写文件 / 导出
allow_net: bool = False     # 出网
allow_trade: bool = False   # 下单（默认关，和现在 confirm_trade 语义合并）
```

**trade 的三道闸**（不是一道）：

1. 开关必须打开（默认关）
2. 弹框里手数、价格、理由、保证金占用、日剩余额度必须都写出来
3. 下单前调 `daylimit` 预检 —— 现在 `daylimit` 只在真正发报文时 +1，**没有"预检"接口**，
   要补一个 `peek(account, n) -> (ok, remain)`，让弹框能显示"今天还剩 837 笔"

第 3 条是这次要新写的代码，但复用的是现成的计数文件，不引入新状态。

### 3.4 和现有代码怎么接（改动清单）

| 现在 | 改成 |
|---|---|
| `parse_action` 的白名单（`agent.py:939`） | 查 `REGISTRY`，未注册 → 返回"平台没有这个工具" |
| `_parse_factor` / `_parse_strategy` / `_parse_fs` / `_parse_doc` | 通用 `validate()` + 各自 handler 内的语义检查 |
| `Action.describe()` 的一串 `if` | 从 schema 生成一行摘要（`describe(tool, params)`） |
| `_handle_action` 的一串 `if` | `REGISTRY[name].handler(ctx, payload)`，确认策略按 `risk` 决定 |
| `SYSTEM_PROMPT` 里 6 段动作说明（约 2500 字符） | `catalog(enabled)` 现拼，只注入启用的包 |
| `ConfirmDialog` 的 `if kind == "factor"` / `"strategy"` 两段 | 按 `risk` + `blocks` 渲染通用确认框 |

**兼容性**：`action` 字符串值全部保持不变（`trade` / `backtest` / `factor` / `strategy` /
`fs` / `doc`），所以历史对话、已存记忆、现有提示词示例都不用改。新工具只是新增名字。

### 3.5 公共契约：借哪两套标准，借到什么程度

你问的「有没有公共的 skill 契约，方便下载公开 skill」—— 有，而且现成的是**两套**，
但它们解决的是不同问题，别混着抄：

| 标准 | 是什么 | 解决的问题 | 我们能借多少 |
|---|---|---|---|
| **Agent Skills**（Anthropic，2025-10） | 一个目录 + `SKILL.md`（YAML frontmatter：`name` / `description`，可选 `license` / `allowed-tools`）+ 脚本和资源文件 | 「怎么做事」的知识怎么组织、怎么按需加载 | **能力包的描述格式照抄** |
| **MCP**（Model Context Protocol） | 工具对象的契约：`name` / `description` / `inputSchema`（标准 JSON Schema），JSON-RPC 传输 | 「能调什么」怎么被机器发现和调用 | **Tool 的结构对齐** |

这两套的分工正好对应我们的两层：

```
Skill（能力包）  ← 格式照 SKILL.md
   └─ Tool（工具） ← 结构照 MCP 的 Tool 对象
```

#### 具体怎么对齐

**Tool 用 MCP 的字段名，不用自己发明**：

```python
@dataclass(frozen=True)
class Tool:
    name: str            # MCP: name
    description: str     # MCP: description（一行，进 prompt 目录）
    inputSchema: dict    # MCP: inputSchema（标准 JSON Schema）
    # 以下是我们的扩展，MCP 里没有对应概念
    group: str           # 能力包
    risk: str            # read / write / net / trade
    handler: Callable
    prompt: str = ""     # 详细用法，按需注入
```

这么做的三个好处：

1. **本地注册表能直接吐成 MCP server**。`tools/list` 就是遍历 `REGISTRY` 拼
   `{"name", "description", "inputSchema"}`，`tools/call` 就是 `dispatch`。以后想让
   Claude Desktop / 别的 agent 用我们的回测和因子，不用重写一遍。
2. **将来接 MCP 客户端时，远端工具和本地工具在注册表里长得一样**。文件系统、fetch、
   sqlite、playwright 这些现成 MCP server 接进来就是多几个 `Tool` 条目，`parse_action`
   和执行层都不用改。
3. **`inputSchema` 用标准 JSON Schema**，我前面说的「轻量 schema」只是它的子集
   （`type` / `required` / `minimum` / `maximum` / `enum`）。写的时候按标准写，
   校验器先只实现子集，以后要换 `jsonschema` 库零成本。

**Skill 用 `SKILL.md` 兼容格式**：

```markdown
---
name: doc
description: 读写本地文档 —— Excel / PDF / Word，以及把结果导成 PDF、xlsx
group: doc
risk: read+write
tools: [read_excel, parse_pdf, read_docx, write_pdf, write_xlsx]
---

（正文：什么时候用、注意什么、常见坑。**只在启用这个包时注入提示词**）
```

`name` / `description` 是公开 skill 的标准字段，照写；`group` / `risk` / `tools` 是我们的
扩展。这样从公开仓库拿一个 skill 下来，frontmatter 能直接读，正文按需加载 ——
**渐进披露（progressive disclosure）** 这个思路正好和 §3.3 的「按包注入」是一回事：
只有 `name` + `description` 常驻，正文等真要用时才进上下文。

#### 一个必须说清楚的限制

**公开 skill 大多「下载了不能直接用」**。因为现成的 skill 几乎都是给「有 bash、有
文件系统、能跑脚本」的 agent 写的 —— 正文里写的是 `python scripts/parse.py input.xlsx`
这种。我们的 Agent 没有通用 shell，只能调注册表里的工具。

所以「下载即用」只对**纯提示词型 skill** 成立（就是一份「怎么分析财报」「怎么读
仓单报告」的说明书，不需要额外代码）。带脚本的必须人工适配：把脚本能力包成一个
`Tool`，正文里把 `python scripts/x.py` 改成 `调用 read_excel(...)`。

这不是设计缺陷，是安全边界带来的必然代价 —— 一个能跑任意脚本的 Agent 和「只能调
白名单工具」的 Agent，风险不是一个量级。§3.3 的 `risk` 分级正是建立在这个前提上。

#### 落地顺序

| 步骤 | 做什么 | 什么时候 |
|---|---|---|
| 1 | `Tool` 字段直接按 MCP 命名写（`name` / `description` / `inputSchema`） | **P1 就做**，改名字是零成本的，后补是重构 |
| 2 | `catalog()` 输出 `SKILL.md` 兼容的 frontmatter 结构 | P1 |
| 3 | 导出 MCP server（`tools/list` + `tools/call`） | P3，等工具稳定了再暴露 |
| 4 | 接 MCP 客户端，支持从外部装工具 | P4，看实际需求 |

---

## 4. 文档能力

### 4.1 缺什么（本机实测）

| 包 | 状态 | 影响 |
|---|---|---|
| `pypdf` | ❌ 未装 | **PDF 附件现在读不了**（`attachments.py` 注释里写"装了 pypdf 就走 pypdf"，实际没装） |
| `openpyxl` | ❌ 未装 | **xlsx 读不了**（`pandas.read_excel` 硬依赖它） |
| `python-docx` | ❌ 未装 | docx 能读但**表格丢失**（现在自己解 zip 抠 `<w:t>`） |
| `beautifulsoup4` 4.15 | ✅ | 网页解析可用 |
| `lxml` 6.1 | ✅ | 同上 |
| `Pillow` 13 | ✅ | 图片缩放可用 |
| `requests` 2.34 | ✅ | HTTP 可用 |

### 4.2 装什么

| 包 | 体积 | 为什么 | 替代方案与取舍 |
|---|---|---|---|
| `pypdf` | ~3MB 纯 py，零依赖 | PDF 文字层抽取、页数、简单表格 | `pymupdf`（fitz）抽取质量更好但 20MB+ 二进制轮子，安装包从 ~99MB 涨到 ~120MB —— 不值 |
| `openpyxl` | ~5MB 纯 py | xlsx 读（pandas 必需）+ 写（导出因子表） | `xlrd` 只支持老 xls；`calamine` 要 Rust |
| `python-docx` | ~2MB | Word 表格结构 | 不装的话只能给"这段文档里有表格，请截图" |

**不装**：

- `markdown-it-py` / `mistune` —— 见 §5.4，目标是 Qt 富文本不是浏览器 HTML，库产出的
  HTML 大部分要重写
- `trafilatura` / `readability-lxml` —— 依赖链重（courlan / justext / 一堆），先用
  bs4 + lxml 手写正文抽取（去 `script/style/nav/header/footer`，取文本密度最高的块）
- `pygments` —— 代码高亮先不做
- `tiktoken` —— 现在用字数估，够用
- `re` —— **标准库，不用装**（你问的这个）

### 4.3 附件资源化（这是 doc 能力的前提）

现在的流程是"一次性摊平"，必须改成"会话资源 + 按需取"：

```
用户拖 xlsx 进来
   ↓
attachments.load() 识别类型
   ↓
Excel → openpyxl 读成 DataFrame → 存进 SessionFiles，分配 file_id = "f1"
   ↓
消息里只放**摘要**：sheet 名、行列数、列名、前 5 行（约 300 字符，不是 6 万字）
   ↓
模型看到摘要 → 决定调 read_excel(file_id="f1", sheet="因子", rows=100)
   ↓
拿到真数据 → 推断公式 → factor_register（现有链路，不改）
```

**收益**：

- 一份 200 行 × 15 列的 Excel，现在塞 6 万字，改后塞 300 字符
- 模型可以二次、三次取数（"看看第 3 个 sheet""只要 IC 那两列"）
- 附件变成有 id 的东西，`read_excel` 才能被当成工具调用

**代价**：要新增 `SessionFiles`（内存缓存 + 生命周期跟对话走），以及 `attachments`
里加 `kind="table"` 的分支。这是这次改造里**唯一需要动数据流的地方**。

### 4.4 PDF

**读**：`pypdf` 抽文本层。扫描件（没有文本层）→ 明确回"这是图片型 PDF，请截图发我"，
**不硬凑**（凑出来的乱码比直说更浪费时间）。表格用 `extract_text()` 的布局模式，
效果一般就退回"按行给文本 + 提示用户截图"。

**写**：复用 `factor_pdf.py` 的路子 —— `QTextDocument` + `QPdfWriter`，零新依赖，
中文靠系统字体子集嵌入。抽成 `ttrade/docgen.py`：

```python
def html_to_pdf(html: str, path: str, title: str = "") -> str
```

`factor_pdf.py` 里那段"只用 Qt 富文本支持的 HTML/CSS 子集"的注释要搬进 `docgen`，
让以后所有 PDF 都守同一个约束。

### 4.5 网页

`web_fetch(url)`：`requests` 取 → 判断 `Content-Type` 是 text/html → bs4 去噪 →
取正文 → 截断 8000 字 → 返回 `blocks=[{"type": "text", ...}]` + 原文链接。

**安全清单**（缺一不可）：

| 风险 | 处理 |
|---|---|
| SSRF 打内网 | 解析后的 IP 落在 `127./10./172.16-31./192.168./169.254./::1` → 拒 |
| DNS 重绑定 | 解析一次、连接用同一个 IP（`requests` 的 `allow_redirects` 要逐跳检查） |
| 超大响应 | `stream=True` + 读满 5MB 就断 |
| 非文本 | `Content-Type` 不在白名单（html/text/json）→ 拒 |
| 无限重定向 | 最多 3 跳，逐跳做 IP 检查 |
| 卡死 | 连接 5s / 读 10s 超时 |

`web_search` 需要 API key（Tavily / Serper / Bing）。建议**先只做 `web_fetch`** —— 用户
把链接贴过来就够了，搜索等有 key 再说。

---

## 5. 消息渲染

### 5.1 根因（一行代码的事，但影响全部体验）

```python
def _on_delta(self, piece: str) -> None:
    self._stream_buf += piece
    self.log.insertPlainText(piece)      # ← 纯文本，不解释 markdown
```

```python
def _on_done(self, text: str) -> None:
    final = text or self._stream_buf
    self._append("<p></p>")              # ← 只补一个空段落，不重排
```

所以模型回的 `**结论**`、`| 因子 | IC |`、三反引号代码块，用户看到的就是原文符号。

### 5.2 Block 协议（工具返回和消息渲染共用一套）

这是整个改造的枢纽 —— **工具返回结构化块，聊天区按块渲染**。以后换渲染实现，协议不变。

```python
Block = {
    "type": "text" | "table" | "code" | "metrics" | "warning" | "image" | "link",
    ... 各自字段
}
```

| type | 字段 | 用于 |
|---|---|---|
| `text` | `text` | 普通段落、说明 |
| `table` | `columns`, `rows`, `note` | 参数预览、因子定义、Excel 前 N 行 |
| `code` | `lang`, `text` | 因子/策略实现代码 |
| `metrics` | `items: [(名, 值, 好/坏)]`, `link` | 回测结果卡片 |
| `warning` | `title`, `lines` | 风控提示、错误、注册失败 |
| `image` | `b64` 或 `path`, `width` | 缩略图、图表 |
| `link` | `text`, `url`, `internal` | "打开结果页"（内部跳转） |

两条产出来源：

1. **模型回复**：markdown → 解析器 → Block（`| a | b |` → `table`，``` → `code`，
   `> ` → `warning` 可选）
2. **工具返回**：handler 直接吐 Block（`backtest_run` 吐 `metrics` + `link`，
   `read_excel` 吐 `table` + `note="还有 180 行"`）

### 5.3 Qt 富文本能力实测（这决定卡片能做到什么程度）

拿 `QTextDocument` 渲染到 `QImage` 逐像素验证的结果：

| 样式 | 支持 | 说明 |
|---|---|---|
| `background`（div / td） | ✅ | 卡片底色没问题 |
| `<td width="30%">` | ✅ | **列宽必须写在每个 td/th 上**，只写 th 不生效（已踩） |
| `table border="1"` | ✅ | 默认 `#c0c0c0` 灰线 |
| `pre` 的 `background-color` | ✅ | 代码块底色没问题 |
| `border-left: 5px solid #b91c1c` | ⚠️ **只在 td 上生效** | 写在 `div` / `p` 上会被直接忽略（`toHtml()` 里都不出现）。色条要拼「窄 td + 内容 td」的 1×2 表格 |
| `border-radius` | ❌ | **不支持**，方角。0px 和 12px 渲染完全一致 |
| `<img src="data:image/png;base64,...">` | ✅ | **支持**！截图可以直接内联，不用落临时文件 |
| `<img src="file:///...">` | ✅ | 本地路径也行 |
| `document().addResource()` | ✅ | 第三种方式，适合大图复用 |

**结论**：五种卡片都能做，唯一妥协是**方角**。要圆角就得换成
`QScrollArea + 控件列表`（每条消息一个 QWidget），工作量差一档，而且流式渲染和
跨消息选中复制都要重做。

### 5.4 五种消息类型的落地样式

| 类型 | 视觉 | Qt 实现 |
|---|---|---|
| **普通文本** | 段落，行高 160%，正文色 | `<p style="line-height:160%">` |
| **参数预览卡片** | 表头 `#eef2f7` 灰底，边框 `#d1d5db`，左侧 4px `#2563eb` 色条 | `<table border="1">` + 首列 `border-left` |
| **文档提取预览** | 同上 + 底部灰字"共 200 行，已显示前 20 行" | 同上 + 尾部 `<p style="color:#6b7280;font-size:11px">` |
| **风险告警块** | 左侧 5px `#b91c1c` 红条 + `#fdeaea` 底 + 粗体标题 | `<div style="border-left:5px solid #b91c1c;background:#fdeaea;padding:8px">` |
| **执行结果卡片** | 指标 2 列网格，正数绿负数红，底部"打开结果页"链接 | `<table>` + `color:#166534/#b91c1c` + `<a href="app://result">` |
| **代码块** | 等宽 Consolas，`#f3f4f6` 底 | `<pre style="background-color:#f3f4f6;font-family:Consolas">` |

内部跳转用 `<a href="app://...">` + `QTextBrowser.anchorClicked` 信号（现在只开了
`setOpenExternalLinks`，内部协议要自己接）。

### 5.5 markdown → Block 解析器

新增 `gui/chat_render.py`，**手写，不引库**：

```python
def parse_markdown(text: str) -> list[Block]
def render_blocks(blocks: list[Block]) -> str        # → Qt 富文本 HTML
```

要支持的最小集合（覆盖模型实际会用的 95%）：

- 标题 `#`~`###`
- 粗体 `**`、斜体 `*`、行内码 `` ` ``
- 无序列表 `-` / `*`、有序列表 `1.`
- 表格 `| a | b |` + 分隔行
- 代码块三反引号（带语言标记，不染色）
- 引用 `>` → 灰底段落（模型爱用它写"注意"）
- 链接 `[text](url)`

**为什么不用 `markdown-it-py`**：它产出标准 HTML（`<h1>`、`<p>` 嵌套、`class` 属性），
而 `QTextDocument` 只吃一小撮 CSS 且会忽略 `class`。拿到之后还是要遍历 DOM 重写成
行内样式，等于多一层依赖做同样的活。手写 200 行，用 `re` 就够。

### 5.6 流式怎么处理

现在逐字 `insertPlainText`，收尾不重排。改成：

```python
def _on_delta(self, piece):
    if self._msg_start is None:
        self._msg_start = self.log.textCursor().position()
    self._stream_buf += piece
    self.log.insertPlainText(piece)          # 流式仍走纯文本（快）

def _on_done(self, text):
    cur = self.log.textCursor()
    cur.setPosition(self._msg_start)                      # 回到本条起点
    cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
    cur.removeSelectedText()                              # 抹掉流式纯文本
    cur.insertHtml(render_blocks(parse_markdown(final)))  # 换成渲染结果
    self._msg_start = None
```

**代价**：收尾瞬间有一次重排，长回答可能看到轻微跳动。可接受（现在收尾本来也要补
一个空段落）。

**进阶（可选，后做）**：流式期间用 200ms 节流，每次重渲染整条消息。肉眼几乎无感，
但长回答的 CPU 占用会上去，先不做。

### 5.6.1 一个必须记住的坑：`insertHtml` 会并块

光标停在「最后一段的末尾」时，`insertHtml` 会把片段里的**第一个块并进当前段落**。
实测表现：欢迎语末尾粘出一个「你」、「助手」粘在用户提问后面、`## 结论` 直接缩成
行内跟上一句连在一起 —— 整段对话糊成一坨。

修法是插入前先看当前块有没有字，有就 `insertBlock()` 断开：

```python
cur = self.log.textCursor()
cur.movePosition(QTextCursor.End)
if cur.block().text().strip():
    cur.insertBlock()
self.log.insertHtml(html)
```

`_render_reply` 里同理（那里光标停在「助手」那一行末尾）。这个问题冒烟测试里
单独加了一条断言盯着（「追加：两段各占一行」）。

### 5.7 输入框：图片粘贴 / 拖放

**不需要任何新包**，全在 PySide6 里：

```python
class ChatInput(QPlainTextEdit):
    files_dropped = Signal(list)          # 交给页面加进附件条

    def canInsertFromMimeData(self, src):
        return src.hasImage() or src.hasUrls() or super().canInsertFromMimeData(src)

    def insertFromMimeData(self, src):
        if src.hasImage():                # Ctrl+V 一张截图
            self.files_dropped.emit([src.imageData()])
            return
        if src.hasUrls():                 # 从资源管理器复制文件
            self.files_dropped.emit([u.toLocalFile() for u in src.urls()])
            return
        if src.hasHtml():                 # 从网页复制的表格 —— 这个很实用
            ...转成 markdown 表格插进文本...
            return
        super().insertFromMimeData(src)

    def dropEvent(self, e): ...           # 拖放，同上
```

配套改动：

- `setAcceptDrops(True)` + `dragEnterEvent`（只接受图片/文档后缀，超过 8 个忽略）
- `attachments.load()` 支持直接吃 `QImage` 对象（现在是吃路径），存临时文件或直接
  编码成 base64 —— 因为 §5.3 验证了 data URI 能渲染，**可以全程不落盘**
- 附件条改成 40×40 缩略图 + 文件名 + 尺寸（现在只有文字按钮 `名字 ✕`）

**从网页复制表格**这条值得单独说：用户看研报经常直接框选表格复制，现在粘进去是一堆
用 Tab 隔开的乱文本。`mime.hasHtml()` 能拿到 `<table>`，转成 markdown 表格插进输入框，
模型读起来准确得多。这是零成本的高频收益。

---

## 6. 分阶段实施

### P0 —— ✅ 已完成（2026-10-30）

| 做完了什么 | 落在哪 |
|---|---|
| markdown → Block → Qt 富文本（标题/粗斜体/行内码/列表/表格/代码块/引用/分隔线/链接） | 新增 `gui/chat_render.py`（500 行） |
| 流式收尾重排：`_on_delta` 记起点，`_on_done` 抹掉纯文本换 HTML | `gui/agent_page.py` 的 `_on_delta` / `_render_reply` / `_on_done` |
| 出错也保留半截 + 红块 | `_on_failed` |
| 粘贴截图 / 文件 / 网页表格，拖放文件 | 新增 `ChatInput` 子类 + `html_table_to_md()` |
| 附件条改成 34×34 缩略图 + 名字 + 尺寸 | `_att_chip()` / `_thumb()` |
| QImage 直接变附件，全程不落盘 | `attachments.from_qimage()` |
| 卡片样式：参数表 / 指标卡 / 风控块 / 代码块 / 图片 / 链接 | `chat_render` 的 `block_*` 构造器（P1 的工具直接吐这些） |

验收：冒烟 2804 项（新增 27 项）/ 自检全 OK。方角卡片，无新依赖。

**P0 单独成阶段**的理由：它不改架构、不装依赖、风险最低，但用户感知的提升最大 ——
先拿到反馈，再决定后面投入多少。事实证明这个判断是对的：P0 做完之后，
`_on_delta` 那行 `insertPlainText` 一改，表格和粗体立刻正常了。

### P1 —— ✅ 已完成（2026-10-31）

| 做完了什么 | 落在哪 |
|---|---|
| 能力包注册表：`Tool` / `SkillResult` / `REGISTRY` / `validate` / `catalog` / `dispatch` | 新增 `ttrade/skills/base.py` |
| 6 个老动作（trade / backtest / factor / strategy / fs / doc）迁进注册表，`parse_action` 的白名单改成查 `REGISTRY`，`describe()` 由工具自己提供 | `ttrade/agent.py` 的动作解析段（净删 200 行） |
| 参数校验只实现 JSON Schema 五件事（`type`/`required`/`default`/`enum`/`min`/`max`），不引 jsonschema | `base.validate()` |
| schema 表达不了的规则走 `Tool.prepare`（`mode=manual` 必须给 `side`、`path` 不能有 `..`） | `skills/trade.py` / `skills/fs.py` |
| `data` 包：`query_bars` / `query_factor` / `list_symbols` | 新增 `ttrade/skills/data.py` |
| `doc` 包：`read_excel` / `parse_pdf` / `read_docx` / `write_xlsx` / `write_pdf` | 新增 `ttrade/skills/doc.py` |
| `factor` / `strategy` 包：查询 + 试算（注册仍走界面，因为要弹框看代码） | `skills/factor.py` / `skills/strategy.py` |
| 附件资源化：`SessionFiles` 分配 `file_id`，消息里只放 300 字符摘要 | 新增 `ttrade/session_files.py` |
| PDF / xlsx 导出，和聊天共用同一个渲染器 | 新增 `ttrade/docgen.py`；`gui/chat_render.py` 移到 `ttrade/render.py` |
| 提示词按包注入：关掉的包不列工具，模型不会写出必然被拒的动作块 | `agent.build_messages(..., enabled=)` + `tool_catalog()` |
| 设置页加「能力包」开关组，状态栏显示已关的包 | `SettingsDialog` / `_refresh_status` |

验收：冒烟 2843 项（新增 39 项，0 失败）/ 自检新增「能力包 20 个动作，附件读表 通」。

### P2 —— ✅ 已完成（2026-10-31）

| 做完了什么 | 落在哪 |
|---|---|
| `web` 包只做 `web_fetch`：四道闸（只 http/https、解析后查 IP、逐跳查重定向、5MB 截断） | 新增 `ttrade/skills/web.py` |
| 网页 → 纯文本（丢 script/style/nav，块级标签转换行，实体反转义） | `web.html_to_text()` |
| 风险分级落地：`read` 自动 / `write` 弹框 / `trade` 弹框 + `confirm_trade`；`fs` 自己管（读不问、写必问） | `_handle_action()` |
| `daylimit.peek()` **不需要新增** —— `DayLimit.order_left()` / `state()` 已经够用 | 未改 `daylimit.py` |

`web_fetch` 的白名单实测：`127.0.0.1` / `localhost` / `10.x` / `192.168.x` /
`169.254.169.254`（云元数据）/ `[::1]` / `file://` / `ftp://` / `data:` 全部拦住。

### P3 —— 待做

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **P3**<br>1 周+ | ① `monitor` 包：先快照式，再做订阅式<br>② 系统托盘 + 通知通道 + 开机自启<br>③ 导出 MCP server（`tools/list` / `tools/call`） | 先做"现在帮我扫一遍哪些品种突破 20 日高点"，再挂托盘 |

### 顺带做完的两件事

| 事 | 说明 |
|---|---|
| **Agent 页字号整体放大** | 全局默认 12px 对这一页太小（一屏要读很多字）。新增 `FS_BODY=15 / FS_SMALL=12 / FS_CODE=13 / FS_HEAD=17 / FS_UI=14` 六个常量 + 页面级 `AGENT_QSS`，只管这一页，别的页密度不动。渲染器里的字号也提到 `ttrade/render.py` 的 `BODY_PX/SMALL_PX/CODE_PX/HEAD_PX`，聊天和导出 PDF 一起变 |
| **应用图标改成 QuantBuddy 风格** | `packaging/make_icon.py` 重画：靛蓝→紫渐变圆角底 + 白色对话气泡 + 三根抬高的柱子 + 琥珀顶盖。运行时用 `app_icon()` 显式 `setWindowIcon`，`icon.png` 进 `DATAS`，源码运行和打包版看起来一样 |

---

## 7. 已拍板的 5 件事

| # | 问题 | 结论 |
|---|---|---|
| 1 | web 做到哪一步 | **只做 `web_fetch`**。搜索要 API key，等真需要再说 |
| 2 | 附件资源化 | **接受**。图片走粘贴/拖放直接进输入框；Excel / PDF / Word 走附件 |
| 3 | monitor 要不要常驻 + 托盘 | **要，但放到 P3**。先按"常驻进程 + 托盘图标 + 通知"设计，别做半套 |
| 4 | 渲染接不接受方角 | **接受**，只要阅读友好。不为了圆角换控件列表 |
| 5 | `write` 级确认边界 | **维持现状**（`allow_fs` 那种：读不弹、写弹）。体验可以，不动 |

由此确定的实施约束：

- **P0 保持 `QTextBrowser`**，不引 `QScrollArea + 控件列表`
- **P2 的 web 包只实现 `fetch`**，`search` 留接口不实现
- **P3 才动托盘**，且那是一次独立的基础设施改造（常驻、通知、自启），不塞进 skill 包里
- **风险分级只做三档**：`read`（自动）/ `write`（弹框）/ `trade`（弹框 + 红字），
  不额外加 `net` 一档 —— 出网归到 `write` 同级弹一次就行，少一个开关少一份心智负担

---

## 附：改动文件清单（实施时对照）

| 文件 | 动作 |
|---|---|
| `gui/chat_render.py` | **新增** markdown → Block → Qt HTML |
| `ttrade/skills/` | **新增** 7 个模块 + 注册表 |
| `ttrade/docgen.py` | **新增** HTML → PDF（从 `factor_pdf.py` 抽） |
| `ttrade/session_files.py` | **新增** 附件资源缓存 |
| `ttrade/agent.py` | 改：`parse_action` 走注册表、`describe` 由 schema 生成、提示词按包注入 |
| `gui/agent_page.py` | 改：流式覆盖渲染、输入框子类化、确认框通用化、设置页开关组 |
| `ttrade/attachments.py` | 改：支持 `QImage` 入参、`kind="table"` 分支 |
| `ttrade/daylimit.py` | 改：加 `peek()` 预检 |
| `requirements.txt` | 改：加 `pypdf` / `openpyxl` / `python-docx` |
| `packaging/TrendDesk.spec` | 改：`hiddenimports` 加新模块（`openpyxl` 有动态导入，必须显式列） |
