# 四个 Agent 框架的拆解，与 ChanAgent 的取舍

> 研究对象：Claude Code（`D:\claude-code-main`）、Hermes Agent（Nous Research）、  
> DeepSeek Harness / dsh（DeepSeek AI）、WorkBuddy（本对话所在的助手）。  
> 落点：ChanAgent —— 专项量化研究与实盘交易，纯 Python，稳健可控优先。
>
> 这份文档只做判断，不改代码。



---

## 0. 先说结论

四家解决的是**同一个问题的四个不同切面**：

| 框架               | 它真正在解决的问题         | 一句话架构                                     |
| ---------------- | ----------------- | ----------------------------------------- |
| Claude Code      | 通用编码任务**跑得久、不出事** | 40+ 工具的强契约 + 四层权限 + 四段式上下文压缩 + 多 Agent 编排 |
| Hermes           | Agent **自己长大**    | 技能自写自更 + 文件即记忆 + Engine 循环                |
| DeepSeek Harness | Agent **可拆可换可复盘** | Cordis 插件内核 + 一切皆插件 + append-only 事件流     |
| WorkBuddy        | 通用助手**在桌面端好用**    | 多轮工具循环 + 三层记忆 + 技能分层 + 产物呈现               |

**ChanAgent 的问题不是功能少，是「边界不清」。**

现在 `ttrade/agent.py` 里那 5000 多字符的 `SYSTEM_PROMPT` 同时承担了四件事：  
系统规则、工具说明、因子目录、策略目录。这正是 HARNESS.md 列的头号反模式  
（把工具实现逻辑写进 System Rules）—— 改一个因子说明要动系统提示词，  
加一个工具要同时改提示词和注册表，两边一旦不同步，模型就按过时的说明书干活。

所以这份文档的结论不是「加功能」，而是**先把 Harness 拆正交，再加三样东西**：  
结果预算、结构化错误、可复盘日志。

---

## 1. 八个维度逐项拆

### 1.1 主循环

**Claude Code**：`query.ts` 的 `queryLoop()` 是一个 `async function*` 生成器，  
`while(true)` 里每轮做九件事，顺序固定：

```
1. applyToolResultBudget   超长工具结果换成磁盘引用（按 tool_use_id 替换，内容不可见）
2. snipCompact             历史剪枝（HISTORY_SNIP）
3. microcompact            按 tool_use_id 折叠旧结果（缓存友好，不读内容）
4. contextCollapse         把老对话投影成摘要视图（commit log 可回放）
5. autocompact             还不够就整段摘要
6. 发请求
7. runTools / streamingToolExecutor   工具执行（流式：边生成边执行）
8. handleStopHooks         收尾钩子，可 preventContinuation / 塞 blockingErrors
9. maxTurns 检查 → 继续或终止
```

关键设计：**压缩分四级，从便宜到贵**。前三级都不调模型，只有 `autocompact`  
要花一次请求。而且 `microcompact` 故意只按 id 替换、不看内容，这样它能和  
prompt cache 共存 —— 内容变了缓存就废了。

**Hermes**：Engine 模式，执行循环 + 优先闭环 + 学习系统。循环本身不特殊，  
特殊的是**循环结束时多一步**：判断这次任务值不值得沉淀成技能。

**dsh**：循环是插件。四种运行模式（standard / PTC / minimal / creative）  
本质是四套默认插件集，连「Agent 循环」本身都能被替换。PTC 模式更激进 ——  
模型先生成一段代码，由代码去编排多轮工具调用，把 N 次模型往返压成 1 次。

**WorkBuddy**：单轮工具循环 + 明确的三种工作模式（Agent 直接做 / Plan 先出方案  
再执行 / Ask 只读不动手）。模式是**用户显式选的**，不是模型猜的。

**ChanAgent 现状**：`gui/agent_page.py` 的 `_after_tool()` 是一个「回灌-重发」  
循环，上限 `MAX_TOOL_TURNS = 4`，全局固定。

> ⚠️ 体检发现：`ttrade/skills/base.py` 的 `Tool.max_turns` 字段，  
> `data.py` / `doc.py` / `manual.py` / `monitor.py` / `web.py` 八处都设了值  
> （2 或 3），但**全项目没有任何一行代码读它**。是个死字段 —— 声明了意图，  
> 没接上执行。



---

### 1.2 工具契约

**Claude Code 的 `Tool` 类型是所有样本里最完整的**（`Tool.ts:362`），值得逐条看：

```ts
export type Tool<Input, Output, P> = {
  aliases?: string[]              // 改名后老名字还能用
  searchHint?: string             // 3–10 个词，给关键词检索用
  call(args, context, canUseTool, parentMessage, onProgress?)
  description(input, {isNonInteractiveSession, toolPermissionContext, tools})
  inputSchema: Input              // Zod
  inputJSONSchema?: ...           // MCP 工具直接给 JSON Schema
  outputSchema?: z.ZodType
  inputsEquivalent?(a, b)         // 判断两次调用是否等价（去重用）
  isConcurrencySafe(input): boolean   // 能否并行
  isEnabled(): boolean
  isReadOnly(input): boolean
  isDestructive?(input): boolean       // 只有删/覆盖/发送才 true
  interruptBehavior?(): 'cancel' | 'block'   // 用户中途插话时怎么办
  isSearchOrReadCommand?(input)   // UI 折叠用
  isOpenWorld?(input): boolean
  shouldDefer?: boolean           // 延迟加载，要先 ToolSearch
  alwaysLoad?: boolean            // 反向：永远出现在首轮
  maxResultSizeChars: number      // 超了落盘，只回预览
  validateInput?(input, context): Promise<ValidationResult>
  checkPermissions(input, context): Promise<PermissionResult>
  getPath?(input): string
  preparePermissionMatcher?(input): Promise<(pattern) => boolean>
  prompt(options): Promise<string>
  userFacingName(input): string
  getToolUseSummary?(input): string | null
  ...
}
```

几个值得抄的点：

1. **`validateInput` 和 `checkPermissions` 是两件事，顺序固定**。  
   前者回答「这个参数在这个上下文里合不合法」（模型自己该看到错误并改正），  
   后者回答「要不要问用户」。混在一起就会变成「参数写错了也弹一次确认框」。
2. **`maxResultSizeChars` 是每个工具自己声明的**。`Read` 工具设 `Infinity`  
   —— 因为它已经自己限了行数，再落盘会造成 `Read → 文件 → Read` 的死循环。  
   这种「工具自己最清楚自己的输出该多大」的思路很对。
3. **`isConcurrencySafe`**：只读的能并行，写的不能。Claude Code 靠这个决定  
   多个 tool_use 是并发跑还是排队。
4. **`interruptBehavior`**：用户在工具跑着的时候发新消息，这个工具该被取消  
   还是该跑完。默认 `block`（跑完），但长跑的命令应该 `cancel`。
5. **`aliases`**：工具改名不破坏历史会话。这个对 ChanAgent 特别有用 ——  
   老会话里存着旧动作块，改名后重放会失败。

**dsh 的工具约定**更朴素但很硬：工具描述**必须**写清四件事 ——  
「何时调用、必要前置条件、失败语义、副作用」。它把这当成 schema 的一部分，  
不是可选的注释。

**WorkBuddy** 的做法是**延迟加载**：工具的完整 schema 不一开始全给模型，  
先用 `ToolSearch` 按关键词或 `select:A,B,C` 精确取，取到之后才可调用。  
省的是首轮 token，代价是多一次往返。

**ChanAgent 现状**：`ttrade/skills/base.py` 的 `Tool` dataclass 有 13 个字段，  
已经很干净了：

```python
name / group / label / description / inputSchema / risk / handler
prepare / ui / self_confirm / describe / prompt / max_turns
```

缺的是：`read_only`（现在从 `risk` 反推，间接）、`concurrency_safe`、  
`max_result_chars`、`interrupt`、`aliases`、`search_hint`。

`validate()` 手写了一个 JSON Schema 子集（type / required / default / enum /  
minimum / maximum），这个决定是对的 —— 为这五件事装 `jsonschema` 不值。  
`spec()` 能把内部简写翻成标准 JSON Schema 给 MCP 客户端，这点比很多框架干净。

---

### 1.3 权限与风险

**Claude Code** 是我见过分得最细的：

- **模式**（`types/permissions.ts`）：`default` / `plan` / `acceptEdits` /  
  `dontAsk` / `bypassPermissions`，内部还有 `auto`（转录分类器自动判）  
  和 `bubble`。
- **规则来源**（`PermissionRuleSource`）：`userSettings` / `projectSettings` /  
  `localSettings` / `flagSettings` / `policySettings` / `cliArg` / `command` /  
  `session` —— 八层，后面的可以覆盖前面的。
- **行为三态**：`allow` / `deny` / `ask`。
- **上下文**（`ToolPermissionContext`）：`alwaysAllowRules` /  
  `alwaysDenyRules` / `alwaysAskRules` / `shouldAvoidPermissionPrompts`  
  （后台 Agent 没法弹框就自动拒）/ `awaitAutomatedChecksBeforeDialog`。
- **工具自己也能判**：`preparePermissionMatcher` 让工具把 `Bash(git *)`  
  这种模式预编译成闭包，避免每次现解析。

**dsh** 的权限哲学更简单：**没选工作区，输入框直接禁用**。这是「默认拒绝」  
的物理实现 —— 不是弹个框问你，是让你连输入都输不了。凭据同理：  
只写不读，保存后页面只显示脱敏描述符，永不回显。

**WorkBuddy** 在文件操作上有一条很硬的规矩：扫描一律只读、破坏性操作必须先  
警告 + 列清单 + 要显式确认、优先走回收站、每批不超过 10 个。

**ChanAgent 现状**：`RISK_READ` / `RISK_WRITE` / `RISK_TRADE` 三档，  
`_handle_action()` 里三层判断（能力包开没开 → 要不要确认 → 谁来执行）。  
这个设计**是对的，而且比 Claude Code 更适合交易**：

- `trade` 单独一档不是洁癖 —— 它动真钱，确认框里必须一眼看到  
  「买什么、多少、为什么」；而 `write` 那档看的是「要写进去什么内容」。
- `_deny()` 会指名道姓告诉用户「到左下角设置 → 能力包里打开它」。  
  这一点比 Claude Code 做得好 —— 只说「没有权限」等于没说。

缺的是：**没有「这次会话临时允许」这个中间态**。现在要么全局开能力包，  
要么每次弹框。Claude Code 的 `session` 规则来源正好补这个缝。

---

### 1.4 上下文与预算

这是四个样本里**最值得 ChanAgent 抄**的一块。

**Claude Code** 的预算体系有四层：

| 层    | 机制                             | 触发           | 是否调模型     |
| ---- | ------------------------------ | ------------ | --------- |
| 工具结果 | `maxResultSizeChars` → 落盘 + 预览 | 单条结果超限       | 否         |
| 批量   | `applyToolResultBudget`        | 一轮总结果超限      | 否         |
| 历史   | `snipCompact` / `microcompact` | 条数或 token 超阈 | 否         |
| 全局   | `autocompact`                  | 还不够          | 是（一次摘要请求） |

**Skill 列表也有预算**：`SKILL_BUDGET_CONTEXT_PERCENT = 0.01`  
（上下文窗口的 1%），单条描述硬上限 `MAX_LISTING_DESC_CHARS = 250`。  
超预算时先砍非内置技能的描述，内置的永不截断 —— 因为内置的才是主力。

**WorkBuddy** 的做法不同：不压缩历史，而是**把检索结果交给子 Agent**。  
Explore 子 Agent 只读、只回结论，中间翻了哪些文件不进主上下文。  
这是「用隔离换预算」，比压缩更干净，但只适合可分离的检索任务。

**ChanAgent 现状**：

- `_after_tool(result)` 把工具结果**整段**塞进 `history` 的 user 消息，  
  **没有任何长度上限**。一次全市场扫描、一次全表因子值、一份长研报，  
  都能直接把这轮上下文吃光。
- `_maybe_compress()` 按**条数**（`MEMO_KEEP`）触发，不是按 token。  
  40 条短消息和 40 条长消息待遇一样。
- `trim_history(limit=40)` 也是按条数裁。

> 这是当前最实际的风险点。量化场景天然产出大结果 —— 一个 68 因子 × 5000 根  
> bar 的矩阵，序列化出来就是几 MB。模型看不了，钱也白花。

---

### 1.5 记忆

**Claude Code** 的记忆系统有一条很硬的原则（`memdir/memoryTypes.ts`）：

> Memories are constrained to four types capturing context **NOT derivable**  
> from the current project state. Code patterns, architecture, git history,  
> and file structure are derivable and should NOT be saved as memories.

四类：

| 类型          | 记什么                                     |
| ----------- | --------------------------------------- |
| `user`      | 用户的角色、目标、知识背景、偏好                        |
| `feedback`  | 用户对工作方式的指导 —— **失败和成功都记**（只记纠错会让人越来越保守） |
| `project`   | 项目当前在做什么、为什么                            |
| `reference` | 外部系统的指针（文档在哪、怎么查）                       |

`MEMORY.md` 有硬上限：**200 行 / 25KB**。超了截断并写明是哪条线触发的。  
理由很实在 —— 观察到的 p100 是「197KB 挤在 200 行里」，所以光限行数不够。

**autoDream** 是后台整合：三道门（24 小时 / 5 个会话 / 无锁冲突），  
四阶段（Orient → Gather → Consolidate → Prune）。**Prune 是必须的一步** ——  
不做减法的记忆系统三个月就废了。

**Hermes** 反过来：「文件即记忆」，Markdown 笔记本模式，容量无限。  
好处是轻、可读、可 diff；坏处是没有上限，早晚会撑爆上下文。

**WorkBuddy** 分三层，各管一段：

| 层         | 位置                                                 | 谁写 | 管什么       |
| --------- | -------------------------------------------------- | -- | --------- |
| 云 profile | 服务端                                                | 自动 | 长期画像，只读   |
| 用户级       | `~/.workbuddy-ai/MEMORY.md`                        | 显式 | 跨项目的硬规矩   |
| 工作区       | `.workbuddy-ai/memory/YYYY-MM-DD.md` + `MEMORY.md` | 显式 | 本项目的日志与沉淀 |

关键设计：**日志 append-only，永不覆盖**；**只记有跨会话价值的东西**，  
不记搜索结果、临时路径、工具报错。

**ChanAgent 现状**：`MEMORY_PROMPT` 只做一件事 —— 把一段对话压成长期记忆。  
没有分类、没有上限、没有「什么不该记」的负面清单、没有 Prune。

---

### 1.6 技能与渐进披露

> 这一节顺带回答上次那个问题：**现在的 Skills 系统不是 Agent Skills 协议。**

**Agent Skills 协议的核心是「目录 + frontmatter + 渐进披露」**：

```
skills/
└── factor-mining/
    ├── SKILL.md          # frontmatter: name / description / when_to_use /
    │                     #   allowed-tools / version / context
    ├── references/
    │   └── ic-guide.md   # 按需读，不默认注入
    └── scripts/
        └── scan.py
```

三件事缺一不可：

1. **SKILL.md 带 YAML frontmatter** —— 元数据可被程序读，不靠解析自然语言。
2. **目录索引进上下文，正文不进** —— 模型先看到「有这么个技能，什么时候用」，  
   真要用时才 `read` 正文。
3. **技能目录可被 `Read` / `Grep` 访问** —— 正文和附件是文件，不是字符串常量。

**Claude Code 的实现**（`skills/loadSkillsDir.ts` + `bundledSkills.ts`）：

- 磁盘技能从 `userSettings` / `projectSettings` / `policySettings` 三个位置加载，  
  用 `realpath` 去重（同一个文件走符号链接进来只算一次）。
- frontmatter 解析出 16 个字段：`description` / `allowed-tools` /  
  `argument-hint` / `arguments` / `when_to_use` / `version` / `model` /  
  `disable-model-invocation` / `user-invocable` / `hooks` / `context` /  
  `agent` / `effort` / `shell` / `paths` / `name`。
- **内置技能也能带文件**：`registerBundledSkill({files: {...}})` 在首次调用时  
  把文件抽到磁盘，然后在提示词前加一行  
  `Base directory for this skill: <dir>` —— 这样模型能用 Read/Grep 去翻，  
  和磁盘技能同一套契约。抽取路径做了三层防护：进程级 nonce 目录、  
  `0o700/0o600` 权限、`O_NOFOLLOW|O_EXCL`（Windows 用 `'wx'`，  
  因为数字 flag 经 libuv 会 EINVAL）。
- **列表预算 1%**，单条 250 字符上限，超预算先砍非内置的。

**dsh** 把技能当插件的一种，和工具、模型、沙箱平级。

**Hermes** 最激进：**技能由 Agent 自己写**。完成 5 次以上工具调用的复杂任务后  
自动生成 Markdown 技能文档，下次同类任务直接复用；执行中发现更优方案会自动  
更新。官方说法是效率提升 40%。

**ChanAgent 现状**：`ttrade/skills/` 下是**Python 模块**（`data.py` / `doc.py` /  
`factor.py` / `strategy.py` / `trade.py` / `web.py` / `monitor.py` / `fs.py` /  
`manual.py`），工具靠 `@register` 装饰器进 `REGISTRY`。  
`GROUPS` 是十个能力包的开关表，`Tool.prompt` 字段存详细用法。

**这不是 Agent Skills 协议**，是「工具注册表 + 分组开关」。差别在：

|       | 现在                    | Agent Skills 协议                   |
| ----- | --------------------- | --------------------------------- |
| 技能载体  | Python 模块里的字符串        | 磁盘上的 `SKILL.md` 目录                |
| 元数据   | `Tool` 的 dataclass 字段 | YAML frontmatter（可被任何工具读）         |
| 正文注入  | 包启用时全量注入              | 只注入索引，按需读正文                       |
| 附件    | 无                     | `references/` `scripts/`，模型可 Grep |
| 用户能否写 | 不能（要改 Python）         | 能（扔个目录进去）                         |

`docs/agent-skill-plan.md` §3.5 里其实已经设计了  
（`name` / `description` / `group` / `risk` / `tools`），但 `catalog()`  
只输出 markdown 标题，**没有真正实现**。



---

### 1.7 子 Agent 与编排

**Claude Code 的 coordinator 模式**（`coordinator/coordinatorMode.ts`）值得整段读，
因为它把「怎么当包工头」写成了提示词：

四阶段，**谁做哪一段是写死的**：

| 阶段 | 谁 | 目的 |
|---|---|---|
| Research | Workers（并行） | 查代码、找文件、理解问题 |
| Synthesis | **Coordinator 自己做** | 读结论、想清楚、写出实施规格 |
| Implementation | Workers | 按规格改，提交 |
| Verification | Workers | 验证能用 |

三条硬规矩：

1. **「Never write 'based on your findings'」** —— 不能把理解这件事转包出去。
   规格里必须出现具体文件路径、行号、改什么。
2. **Verification 要换新人**：`Spawn fresh` 而不是 `Continue`，
   因为验证者不该背着实现者的假设看代码。
3. **Worker 结果用 XML 回来**：
   ```xml
   <task-notification>
   <task-id>agent-a1b</task-id>
   <status>completed|failed|killed</status>
   <summary>...</summary>
   <result>...</result>
   <usage><total_tokens>N</total_tokens>...</usage>
   </task-notification>
   ```
   它伪装成 user 消息进来，但带 `<task-notification>` 标签 —— 模型能区分
   「用户说的」和「worker 报的」。

**Agent 定义也是 frontmatter**（`tools/AgentTool/loadAgentsDir.ts`）：
`tools` / `disallowedTools` / `skills`（预加载哪几个技能） / `mcpServers` /
`hooks` / `model` / `effort` / `permissionMode` / `maxTurns` / `memory` /
`background` / `isolation`（worktree 隔离） / `omitClaudeMd`。

最后那个 `omitClaudeMd` 是个很好的成本意识：只读的 Explore / Plan 子 Agent
不需要 commit/PR/lint 规范，省下的是每周 5–15 G token。

**WorkBuddy** 只留了两个子 Agent：Explore（只读搜索，省上下文）和
general-purpose。少而精 —— 子 Agent 一多，调度成本就超过收益。

**ChanAgent 现状**：没有子 Agent。

---

### 1.8 可观测性与会话

**dsh 的 append-only 事件流**是这一维度最好的答案：

> 模型看到的**一切**都写进仅追加的会话日志 —— 系统提示词、思维链、工具调用
> 与结果、子 Agent 调度、每一次上下文注入。会话的恢复、分叉、检索、回放
> **全部共享同一套事件流**。

价值在事后：Agent 在第几十步做错决定，可以回到「模型当时真正看到的内容」，
判断问题出在模型判断、工具返回、提示词，还是上下文注入。

**Claude Code** 有 `queryCheckpoint()` 埋点（`query_fn_entry` /
`query_autocompact_start` / `query_tool_execution_end` …），
但那是给内部性能分析用的，不是给用户复盘的。

**ChanAgent 现状**：`history` 是内存里的 messages 数组，
`_maybe_compress()` 压完就把原始记录扔了。**回测能复盘，对话不能复盘。**

对交易 Agent 来说这是真缺口 —— 「模型看了什么数字才决定下这一单」，
事后必须能查。

---

## 2. ChanAgent 现状体检

| 项 | 现状 | 位置 |
|---|---|---|
| 系统提示词 | 5000+ 字符全量常驻，混了规则/工具/因子/策略 | `ttrade/agent.py:389` |
| 工具注册 | `@register` → 扁平 `REGISTRY`，10 个能力包 | `ttrade/skills/base.py:158` |
| 工具契约 | 13 字段，缺 `read_only` / `concurrency_safe` / `max_result_chars` / `interrupt` / `aliases` | `base.py:102` |
| `Tool.max_turns` | **声明了、8 处设了值、无人读取** | `base.py:123` |
| 主循环上限 | 固定 `MAX_TOOL_TURNS = 4` | `agent_page.py:2196` |
| 工具结果 | 整段塞进 history，无长度上限 | `agent_page.py:2288` |
| 压缩触发 | 按条数（`MEMO_KEEP`），非 token | `agent_page.py:2034` |
| 历史裁剪 | 按条数（40） | `agent.py:897` |
| 动作块解析 | 失败只回一句「动作块有问题：xxx」 | `agent.py:855` |
| 记忆 | 只有「对话 → 摘要」，无分类/上限/Prune | `agent.py:784` |
| 技能 | Python 模块 + 分组开关，非 Skills 协议 | `ttrade/skills/` |
| 子 Agent | 无 | — |
| 会话日志 | 内存数组，压缩即丢 | `agent_page.py` |
| MCP | 已有 `mcp_server.py`，`spec()` 吐标准 JSON Schema | `ttrade/mcp_server.py` |
| 权限拒绝文案 | **指名去哪儿打开**（做得好） | `agent_page.py:_deny` |
| 风险分档 | `read` / `write` / `trade`（做得好） | `base.py:35` |

---

## 3. 该抄什么

### P0 —— 便宜、直接止血

#### ① 工具结果预算（最高优先）

借 Claude Code 的 `maxResultSizeChars` + `applyToolResultBudget`。

```python
@dataclass
class Tool:
    ...
    max_result_chars: int = 8000   # 超了落盘，只回摘要 + 文件路径
```

```python
def clip_result(text: str, tool: Tool, tag: str) -> str:
    """超长结果落盘，只回预览。落盘路径回灌给模型，它需要细节时可以再读。"""
    if len(text) <= tool.max_result_chars:
        return text
    p = Path(out_dir) / f"{tag}-{ts}.txt"
    p.write_text(text, encoding="utf-8")
    head = text[: tool.max_result_chars // 2]
    tail = text[-tool.max_result_chars // 4 :]
    return (f"{head}\n\n…（中间省略 {len(text) - len(head) - len(tail)} 字符）…\n\n"
            f"{tail}\n\n完整结果已存到 {p}，需要细节可以用 read_file 打开。")
```

`Read` 类工具（读文件、读手册）应该像 Claude Code 一样设 `Infinity` ——
它们自己已经限了长度，再落盘会造成读→存→读的死循环。

**这一条能立刻解决「一次全市场扫描把上下文吃光」的问题。**

#### ② 结构化错误回灌

现在 `parse_action()` 失败只给一句人话，模型不知道该改哪儿。改成：

```python
@dataclass
class ActionError:
    tool: str          # 模型想调哪个
    field: str         # 哪个字段
    expected: str      # 期望什么
    got: str           # 实际给了什么
    hint: str          # 怎么改（可用值列表、范围）
```

回灌文本变成：

```
【动作块有问题】
工具 query_factor 的参数 window 不对：
  期望：整数，1–2000
  实际："250根"
  改法：window: 250
```

模型第二轮就能改对，而不是瞎猜重试。

#### ③ 接上 `max_turns`，改成预算池

现在 `MAX_TOOL_TURNS = 4` 是全局固定。改成「按工具声明的 `max_turns` 累加」：

```python
def consume_budget(self, tool: Tool) -> bool:
    """读类工具可以连调，交易类单次即止。"""
    if self._tool_turns >= self.MAX_TOOL_TURNS:
        return False
    if tool.risk == S.RISK_TRADE:
        self._tool_turns = self.MAX_TOOL_TURNS   # 交易后不再自动续跑
    else:
        self._tool_turns += max(1, tool.max_turns) // 2
    return True
```

顺带把 `trade` 的语义写死：**下完单不自动续跑**，让用户先看一眼。

#### ④ 工具契约补四个字段

```python
read_only: bool = None        # None = 从 risk 推导
concurrency_safe: bool = False
interrupt: str = "block"      # "cancel" 用于长跑回测
aliases: list[str] = field(default_factory=list)
```

`aliases` 对 ChanAgent 特别有用 —— 老会话里的动作块存的是旧工具名，
改名后重放会失败。

---

### P1 —— 值得做，中等成本

#### ⑤ Skill 目录化 + 渐进披露

把 `Tool.prompt` 里那些长说明搬出去，变成磁盘上的技能目录：

```
skills/
├── factor-mining/
│   └── SKILL.md
├── backtest-reading/
│   └── SKILL.md
└── trade-checklist/
    ├── SKILL.md
    └── references/lot-sizing.md
```

`SKILL.md`：

```markdown
---
name: factor-mining
description: 从主观判断出发找因子，写代码、试算、看 IC
when_to_use: 用户说「我想抓 XX 现象」「这个因子有没有用」
allowed-tools: register_factor, query_factor, run_backtest
risk: write
group: factor
version: 1
---

# 因子挖掘

## 步骤
1. 先把用户的主观描述翻译成可计算的定义 …
2. 用 `query_factor` 试算，看覆盖率和分布 …
...
```

加载器只把索引注入上下文（预算 = 上下文的 1%）：

```
可用技能（要用时先 read_skill 取正文）：
- factor-mining: 从主观判断出发找因子 — 用户说「我想抓 XX 现象」时用
- backtest-reading: 读回测报告，指出过拟合风险 — 回测跑完时用
```

模型真要用才读正文。这样 5000 字符的 `SYSTEM_PROMPT` 能砍到 1500 左右，
而且**用户自己就能加技能** —— 扔个目录进去就行，不用改 Python。

#### ⑥ 记忆分四类 + 硬上限 + Prune

借 Claude Code 的分类，换成量化场景：

| 类型 | 记什么 | 例子 |
|---|---|---|
| `user` | 交易风格、风险偏好、常用品种、仓位习惯 | 「偏好日线、单笔风险不超 1%、不做夜盘」 |
| `feedback` | 纠正过我的、验证过有效的 | 「别拿 z=1.0 直接上实盘，先看 IC 衰减」 |
| `project` | 某个策略/因子实验的结论 | 「AG 30m 动量 2024 后失效，换月前后两周噪声大」 |
| `reference` | 数据口径 | 「手续费按双边万三、滑点按一跳、主力换月用持仓量」 |

加三条硬规矩：

1. `MEMORY.md` **不超过 200 行 / 25KB**，超了截断并写明是哪条线。
2. **只记不能从数据推导出来的**。因子数值、回测结果、K 线形态都是可推导的，
   不该进记忆 —— 需要时重算。
3. **定期 Prune**。每次压缩时问一句「哪条已经过时了」，删掉。

#### ⑦ append-only 会话日志

借 dsh。现在的 `history` 是内存数组，压缩即丢。改成：

```
~/.chanquant/sessions/<sid>.jsonl
```

每行一条事件：`user_msg` / `assistant_msg` / `action` / `tool_result` /
`confirm` / `compress`。压缩只是「读时投影」—— 原始行永不删。

对交易 Agent 这是刚需：**「模型看了什么数字才决定下这一单」，事后必须能查。**

成本很低 —— `_append()` 里多写一行 JSON 就行。

#### ⑧ 只开两个子 Agent

不要抄 coordinator 的完整四阶段，只留两个：

| 子 Agent | 权限 | 干什么 |
|---|---|---|
| `research` | 只读 | 跑回测、查数据、读文档，只回结构化结论 |
| `verify` | 只读 | 独立复核：换区间重跑、查未来函数、查滑点假设 |

`verify` 对量化是刚需。「回测漂亮但实盘亏」九成是验证没做 ——
用未来数据、漏了换月、滑点按 0 算、样本内调参。

关键：**verify 必须换新上下文**（Claude Code 的 `Spawn fresh`），
不能让它看见 research 的推理过程，否则会被锚定。

---

### P2 —— 想做但可以等

- **会话临时权限**（借 `session` 规则来源）：允许「本次会话内不再问这个工具」，
  而不是只能全局开能力包。
- **工具检索**（借 ToolSearch）：工具数超过 25 个之后再考虑，现在 20 个不到，
  全量注入反而更省。
- **PTC 代码模式**（借 dsh）：让模型生成一段 Python 编排多轮回测，
  把 N 次往返压成 1 次。对「扫 50 个品种 × 10 组参数」这种任务收益很大，
  但沙箱、超时、配额要配套，先不动。

---

## 4. 明确不抄的，以及为什么

这一节和上一节一样重要。

| 不抄 | 来自 | 为什么 |
|---|---|---|
| BUDDY 电子宠物 | Claude Code | 交易界面里放个宠物，用户会怀疑这软件靠不靠谱 |
| KAIROS 常驻 `<tick>` | Claude Code | 15 秒阻塞预算 + 后台轮询，对实盘是干扰源不是帮手 |
| ULTRAPLAN 远程 30 分钟规划 | Claude Code | 量化决策要可解释、可回滚，不接受「云端想了半小时」 |
| Undercover / voice / vim / ink TUI | Claude Code | 与量化无关 |
| team / swarm 多 Agent 市场 | Claude Code | 子 Agent 一多，调度成本超过收益；交易决策链越短越好 |
| **插件热加载**（Cordis 时空可组合） | dsh | 交易系统要的是**版本固定、可审计**。运行时热插拔意味着「昨天跑的策略今天行为不同」 |
| **`bypassPermissions` / `yolo`** | Claude Code | 真金白银，永远 `default` |
| **自进化技能**（模型自己写技能并自动生效） | Hermes | **对交易是灾难**。模型改了自己的下单规则，没人审计、没人回滚，出了事查不到是哪次自动更新引入的 |
| 内存无限增长 | Hermes | 上下文会撑爆；量化场景尤其 |
| 模型热切换 | dsh | dsh 自己也做了「会话内模型冻结」，这条是对的；但 ChanAgent 更该**策略版本冻结** —— 回测用的因子代码和实盘跑的不是同一份，是最大的坑 |

关于 Hermes 的自进化，值得多说一句：它的思路（任务做完沉淀成技能）
**在离线研究阶段是有价值的** —— 但必须是「建议 → 人审 → 入库」，
不能是「自动生效」。这条可以做成 P2：回测跑完后提示「这次的做法值得存成模板吗」，
用户点确认才写进技能库。

---

## 5. WorkBuddy 自身值得借鉴的三点

研究自己有点奇怪，但这三条确实对 ChanAgent 有直接价值：

**① 三层记忆的写入门槛**

WorkBuddy 明确规定了「什么算值得记」：建了网站、修了 bug、写了报告、
做了架构决策、用户分享了项目约定。**问候、简单查询、短问答不记。**

ChanAgent 的 `MEMORY_PROMPT` 现在是「把对话压成记忆」，没有门槛 ——
什么都会往里塞。加一条「这一轮有没有产生跨会话有价值的东西？没有就返回空」
能省掉大量噪声。

**② 权限拒绝要给路径，不是给结论**

WorkBuddy 的文件安全规则里，拒绝时必须说清「去哪儿改」。
ChanAgent 的 `_deny()` 已经做到了（「到左下角设置 → 能力包里打开它」），
这点保持。

**③ 产物统一呈现入口**

WorkBuddy 所有交付物都走 `present_files` —— 不管是 HTML、图片、报告还是
pptx，用户看到的是同一套卡片。ChanAgent 现在 `SkillResult.blocks` 已经
统一了渲染，思路一致，可以再往前一步：**研报、回测图、因子曲线走同一个
「产物」概念**，用户能像翻文件一样翻历史产物。

---

## 6. 落地顺序

如果只做一件事：**做 ①（结果预算）**。它最便宜，且直接堵住最大的漏。

如果做三件事：**① 结果预算 + ② 结构化错误 + ⑦ append-only 日志**。
三件加起来大约两天，覆盖「上下文爆炸」「模型瞎试」「事后查不了」三个真痛点。

如果做五件事：再加 **⑤ Skill 目录化** 和 **⑥ 记忆四分类**。
这两件是结构性改造，做完 `SYSTEM_PROMPT` 能从 5000 字符降到 1500 左右，
而且用户能自己扩技能 —— 这是从「工具集」变成「平台」的分界线。

**子 Agent（⑧）放最后**。它是收益最高但也最容易做错的一件：
两个就够，且 verify 必须换新上下文。

---

## 附：四家的架构速查

```
Claude Code
  queryLoop (async generator)
    ├─ 四级压缩：budget → snip → microcompact → autocompact
    ├─ runTools / streamingToolExecutor（流式边生成边执行）
    ├─ handleStopHooks（可阻止继续）
    └─ maxTurns 检查
  40+ 工具，每个都有完整 Tool 契约
  权限：5 模式 × 8 规则来源 × 3 行为
  记忆：4 类型，MEMORY.md ≤200 行/25KB，autoDream 四阶段
  编排：coordinator（Research→Synthesis→Implementation→Verification）

Hermes
  Engine 循环 + 优先闭环 + 学习系统
    └─ 循环末尾判断「值不值得沉淀成技能」
  技能：AI 自己写 Markdown，自己更新
  记忆：文件即 Markdown，无限容量

DeepSeek Harness (dsh)
  Cordis 内核（只做加载/卸载/依赖管理，时空可组合）
    └─ 一切皆插件：模型/工具/技能/会话/沙箱/存储/循环/调度/UI
  四种模式 = 四套默认插件集（standard/PTC/minimal/creative）
  append-only 事件流 = 唯一事实源（恢复/分叉/检索/回放共享）
  权限：未选工作区则输入框禁用

WorkBuddy
  单轮工具循环 + 三种模式（Agent/Plan/Ask）
  工具延迟加载（ToolSearch → DeferExecuteTool）
  三层记忆（云 profile / 用户级 / 工作区）
  技能分层（用户级 / 项目级）
  子 Agent：Explore（只读检索）/ general-purpose
  产物统一走 present_files
```

---

## 7. 落地记录（v1.8.1）

上面八项全部做完了。逐项对应到文件：

| # | 项 | 落在哪 | 关键点 |
|---|---|---|---|
| ① | 结果预算 | `ttrade/agent_budget.py`、`skills/manual.py:read_output` | 超限落盘、首尾预览、分段读回；`doc`/`factor_info`/`read_skill` 设 `NO_LIMIT` |
| ② | 结构化错误 | `skills/base.py:FieldError`、`agent.py:action_error_feedback` | 四类错误都带 field/expected/got/hint；动作名写错列近似名；**回灌给模型** |
| ③ | 预算池 | `gui/agent_page.py:_spend_turn` | 总预算 6 次 + 单工具 `max_turns`；交易后不续跑 |
| ④ | 契约补字段 | `skills/base.py:Tool` | `read_only`/`concurrency_safe`/`interrupt`/`aliases`/`max_result_chars`/`search_hint` |
| ⑤ | 技能目录化 | `ttrade/skillpacks.py`、`ttrade/skillpacks/*/SKILL.md`、`skills/manual.py:read_skill` | 索引 537 字符，正文按需读；系统提示词 5100 → 3070 |
| ⑥ | 记忆四分类 | `ttrade/agent_memory.py` | `user`/`feedback`/`project`/`reference`，分组渲染，编号删除（Prune），覆盖前留 `.bak` |
| ⑦ | append-only 日志 | `ttrade/agent_log.py` | `agent_sessions/<sid>.jsonl`，7 类事件，`replay()` 投影 |
| ⑧ | 两个子 Agent | `ttrade/subagents.py`、`skills/delegate.py`、`gui/agent_page.py:SubAgentWorker` | 只读过滤按 `is_read_only()` 而不是按能力包 |

### 做的过程中发现的三件事

1. **`Tool.max_turns` 一直是死字段。** 声明了、8 处设了值、全项目没人读。
   真正的上限是 `agent_page.py` 里写死的 `MAX_TOOL_TURNS = 4`。这类「字段声明了
   但没接上执行」的问题，只能靠通读代码发现 —— 类型检查抓不到。
2. **只按能力包过滤不够。** 一开始 `subagents.usable_tools()` 只筛 `group`，
   结果 `doc` 包里的 `write_xlsx` / `write_pdf` 被发给了「只读」的复核助手。
   正确的判据是 `Tool.is_read_only()`，不是包名。这条写进冒烟测试钉住了。
3. **`smoke_test_agent.py` 里有一个断言一直是错的。** 它断言「某条 system 消息
   等于策略清单」，但那条消息里还拼了工具目录，所以它从来没通过过。全量冒烟
   （`smoke_test_tq.py`）不含这个文件，于是没人发现。教训是：**测试没进 CI 就等于没有**。

### 一个操作失误

验证记忆功能时直接对真实文件 `agent_memory.json` 做了写测试，把用户已有的
记忆覆盖掉了，不可恢复。已经做的补救：`save()` 现在每次覆盖前留一份 `.bak`；
测试改用纯函数 `render()`，不再碰真实文件。这条记在这里，是提醒「测试要碰
真实数据之前先备份」。

---

*2026-10-31*
