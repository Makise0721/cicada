# Cicada

Cicada 是一个用 Python 构建的 coding agent。它通过模型循环调用工具，读取代码、编辑文件、运行 PowerShell 命令，并将工具结果交回模型继续完成任务。

项目采用三层设计：**简洁的 Agent 内核、负责依赖与资源清理的插件运行时、可替换的 Coding 和模型插件**。设计借鉴 Pi 的小型主循环、Cordis 的插件生命周期，以及 Chord 的依赖组合思路。

当前版本为可运行的 MVP，面向 Windows 本地使用，默认连接 Ollama 的 `qwen3.5:9b`。已接通真实模型与真实工具闭环，也支持无需模型服务的剧本模式。

## 已有能力

- **模型循环**：统一消息、工具调用 ID 与结果配对；JSON Schema 参数校验；工具错误反馈；取消与截断输出处理。
- **插件运行时**：声明式 `provide/require`，启动前检查缺失依赖、重复提供者与依赖环；拓扑启动、失败回滚、反向拓扑关闭。
- **资源作用域**：插件拥有自己的清理栈，按 LIFO 执行；清理失败聚合，能力注册随实例关闭撤销。
- **Coding 工具**：UTF-8 文本读取、批量精确编辑、整文件写入、PowerShell 执行与大输出落盘。
- **Ollama 适配**：原生 `/api/chat` NDJSON 流，完整工具调用解析、工具结果回传、连接与协议错误归一、terminal 计量（token/耗时）转交。
- **系统提示**：内置提示（版本标识 `builtin:p3-v1`）描述工作区、工具与工作规则，可用 `--instructions-file` 追加项目指令；提示不进入会话历史。
- **运行摘要**：每次模型调用的 token 计量与本地耗时逐轮记录，运行结束输出已知总和与报告覆盖率；缺失计量如实显示 unknown。
- **非交互 CLI**：接受一次任务，显示模型轮次、工具调用、结果和最终状态。

## 环境要求

| 组件 | 要求 |
|---|---|
| 系统 | Windows；当前在 Windows 11 上验证 |
| Python | 3.12 为开发和验证基线；包声明要求 `>=3.12` |
| 包管理 | `uv` |
| 命令工具 | PowerShell 7；从 PATH 或标准安装位置解析 `pwsh.exe` |
| 真实模型 | Ollama，以及已下载的 `qwen3.5:9b` 或兼容工具调用的模型 |

本地真实模型验证使用 Ollama 0.20.7 和 `qwen3.5:9b`。剧本模式不需要 Ollama。

## 快速开始

以下命令在 PowerShell 7 中执行。

### 1. 获取项目和安装依赖

先安装 Python 3.12 与 `uv`，然后执行：

```powershell
git clone https://github.com/Makise0721/cicada.git
Set-Location cicada

# 将依赖缓存放在项目内
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
uv sync --frozen --python 3.12
```

`uv sync` 使用提交的 `uv.lock` 创建项目虚拟环境，并安装运行和开发依赖。

### 2. 启动 Ollama，下载模型

启动 Ollama 服务（托盘应用或独立窗口均可）：

```powershell
ollama serve
```

在项目所在的终端下载模型：

```powershell
ollama pull qwen3.5:9b
```

**模型标称上下文长度与服务实际分配长度不同。** Ollama 服务端默认上下文为 4K，本项目的源码分析任务在 4K 下会出现输入截断。CLI 默认按 `--num-ctx 32768` 逐请求申请上下文窗口，一般无需再配置服务端环境变量；该值是请求值，不构成实际分配长度的保证，可用 `--num-ctx N` 显式修改。

### 3. 运行一个文件任务

在项目终端中执行：

```powershell
New-Item -ItemType Directory -Force .\sandbox | Out-Null

uv run --frozen python -m cicada --workspace .\sandbox --model qwen3.5:9b '请用 write 创建 hello_cicada.txt，内容为 hello cicada；再用 read 读取确认，最后汇报结果。'
```

终端会显示轮次、工具调用 ID、工具结果及 `finished: stop` 等终结状态。任务结束后，文件位于 `sandbox/hello_cicada.txt`。

也可以让 Cicada 阅读项目自身：

```powershell
uv run --frozen python -m cicada --workspace . '阅读 src/cicada 下的源码，说明三层架构、启动流程和模块依赖关系。请先使用工具核对代码，再给出报告。'
```

这会执行真实模型请求；任务耗时和输出质量取决于模型、硬件、上下文配置和任务内容。

## 无模型服务的剧本模式

剧本提供预设的模型响应，执行同一条 Agent 工具链，适合检查安装与确定性回归。下面的例子使用真实的 `write` 和 `read` 工具：

```powershell
New-Item -ItemType Directory -Force .\sandbox | Out-Null

@'
[
  {
    "tool_calls": [
      {"id": "write-1", "name": "write", "arguments": {"path": "hello_cicada.txt", "content": "hello cicada"}}
    ]
  },
  {
    "tool_calls": [
      {"id": "read-1", "name": "read", "arguments": {"path": "hello_cicada.txt"}}
    ]
  },
  {"text": "文件已写入并读取确认。", "stop": true}
]
'@ | Set-Content -LiteralPath .\demo-script.json -Encoding utf8

uv run --frozen python -m cicada --workspace .\sandbox --script .\demo-script.json '执行文件写入和读取演示'
```

每个数组元素对应一轮模型响应。响应可以含 `text`、`tool_calls`，以及 `stop`、`error` 或 `length` 终结标记。剧本模式与真实模型参数互斥。

## CLI 参数

```powershell
uv run --frozen python -m cicada --help
```

| 参数 | 含义 |
|---|---|
| `prompt` | 任务文本，必填位置参数 |
| `--workspace` | 工作区根目录，默认当前目录 |
| `--model` | Ollama 模型名，默认 `qwen3.5:9b` |
| `--ollama-url` | Ollama 服务地址，默认 `http://127.0.0.1:11434` |
| `--think` | 开启模型 thinking；思考增量不展示，也不写入会话历史 |
| `--num-ctx` | 请求级上下文窗口，默认 `32768`；仅真实模型模式 |
| `--instructions-file` | 追加到系统提示的项目指令文件（≤16 KiB UTF-8），两种模式均可用 |
| `--script` | 使用 JSON 剧本；不能与 `--model`、`--ollama-url`、`--think` 或 `--num-ctx` 同时使用 |

启动时会显示模型、上下文请求值及来源、系统提示版本与哈希（不打印提示全文）；运行结束后输出运行摘要（模型调用次数、token 计量的已知总和与报告覆盖率、本地与 provider 耗时分别标注）。摘要只陈述计量事实，不代表任务验收结论。

退出码：`0` 表示正常 `stop`；`1` 表示 Agent 以 `error`、`aborted` 或 `length` 终止；`2` 表示 CLI 检测到参数、剧本、指令文件、连接预检或启动错误。

## 默认工具

| 工具 | 参数 | 行为 |
|---|---|---|
| `read` | `path`，可选 `offset`、`limit` | 读取 UTF-8 文本，行号从 1 开始；输出带来源路径与行范围头、截断原因与续读 footer；正文最多 2000 行，最终内容（含头/footer）不超过 50 KiB；拒绝识别出的二进制内容 |
| `edit` | `path`、`edits: [{old_text, new_text}]` | 在原始文件上批量精确匹配，每项必须唯一且不重叠；全部验证后才写入；成功结果附有界 unified diff，超限时完整 diff 落盘 `.cicada/outputs/` 并返回可回读路径；保留 BOM 并统一到首个换行风格 |
| `write` | `path`、`content` | UTF-8 整文件写入，自动创建父目录，已有文件覆盖；不继承原文件 BOM 或换行约定 |
| `powershell` | `command`，可选 `timeout` | 以工作区为当前目录执行 `pwsh -NoProfile -NonInteractive`；默认超时 120 秒；合并 stdout/stderr，输出超限后保留尾部，并将完整输出保存到 `.cicada/outputs/` |

`edit` 和 `write` 通过同文件队列串行协调，并检查写入目标在工作区内。`read` 可以访问显式指定的工作区外绝对路径；PowerShell 的工作区设置是执行目录，不是文件系统沙箱。`write` 使用普通文件写入，不承诺原子替换；取消不会回滚已经发生的文件或命令副作用。

`read` 的输出限制不等于输入内存限制：当前实现先读取完整文件，再切片和截断输出。

## 架构与扩展

| 层 | 代码 | 职责 |
|---|---|---|
| Agent 内核 | [core](src/cicada/core/) | 消息与端口协议、模型循环、顺序工具派发、结果配对、取消、会话事件 |
| 插件运行时 | [runtime](src/cicada/runtime/) | `PluginDefinition`、声明式依赖、实例生命周期、能力注册、资源清理 |
| 能力插件 | [plugins](src/cicada/plugins/) | Coding 工具、Ollama 模型、模拟模型与工具 |

[boot.py](src/cicada/boot.py) 负责组装：注册插件、拓扑启动、取得模型和工具实现，再注入内核。内核不导入插件运行时；运行时不导入内核或操作会话消息。CLI 位于 [__main__.py](src/cicada/__main__.py)。

新增模型实现 [ModelPort](src/cicada/core/ports.py)；新增工具实现同文件中的 `Tool` 协议。能力通过 [PluginDefinition](src/cicada/runtime/plugin.py) 声明 `provides/requires`，在 `setup` 中调用 `ctx.provide/require`，用 `ctx.defer` 登记清理，再由启动代码显式装配。

嵌入 Python 程序时，可以直接为 Ollama 插件设置请求级 options：

```python
from cicada.plugins.ollama import OllamaConfig, ollama_plugin

model_plugin = ollama_plugin(
    OllamaConfig(model="qwen3.5:9b", options={"num_ctx": 32768})
)
```

将该插件与 Coding 插件一起传给 `bootstrap`。请求级 options 与 CLI 的服务端环境配置是两种使用方式。

## 测试

```powershell
# 无真实模型调用的测试
uv run --frozen pytest -q -m 'not live'

# 真实 Ollama 冒烟与文件任务闭环
uv run --frozen pytest -q -m live

# 完整套件，包含 live 用例
uv run --frozen pytest -q
```

测试覆盖协议、模型循环、导入边界、插件依赖与清理、Coding 工具、Ollama 录制回放，以及 fake / 真实模型端到端闭环。`live` 用例检查本地服务可达性与模型存在性，条件不满足则 skip；即使排除 live，用例收集仍可能执行有界的服务可达性检查。测试临时目录由配置固定为项目内 `.pytest-tmp`。

## 当前范围

当前 CLI 每次启动执行一个任务，历史保存在该次运行的内存中；工具顺序执行，默认最多 50 轮模型循环。尚未提供交互式会话、历史持久化、上下文压缩或插件热替换。

近期改进方向是搜索类工具、任务验证回执，以及持久会话与上下文预算。
