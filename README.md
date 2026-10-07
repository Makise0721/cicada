# Cicada

Cicada 是一个面向 Windows 的 Python coding agent：通过模型循环调用工具，读写文件、执行 PowerShell 命令并把结果交回模型继续任务。三层设计——简洁 Agent 内核、负责依赖与资源清理的插件运行时、可替换的模型与 Coding 工具插件（设计借鉴 Pi / Cordis / Chord）。当前为可运行 MVP，默认连接本地 Ollama 的 `qwen3.5:9b`，也支持无需模型服务的剧本模式与持久会话。

## 环境要求

Windows 11（验证基线）、Python 3.12、`uv`、PowerShell 7。真实模型需 Ollama（剧本模式不需要）；TUI 的 UI 依赖为可选组，仅 `--tui` 需要。

## 快速开始

```powershell
git clone https://github.com/Makise0721/cicada.git
Set-Location cicada
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
uv sync --frozen --python 3.12
uv sync --extra tui   # 仅 --tui 需要

ollama pull qwen3.5:9b   # Ollama 服务端默认 4K 上下文; CLI 默认按请求申请 32768 (--num-ctx 可改)
```

单次任务（终端显示轮次、工具调用与终结状态）：

```powershell
uv run --frozen python -m cicada --workspace .\sandbox --model qwen3.5:9b '请用 write 创建 hello_cicada.txt，内容为 hello cicada，再用 read 读取确认。'
```

剧本模式（无需 Ollama；JSON 数组每项一轮响应，可含 `text`、`tool_calls` 与 `stop`/`error`/`length` 终结标记，与真实模型参数互斥）：

```powershell
uv run --frozen python -m cicada --workspace .\sandbox --script .\demo-script.json '执行文件写入和读取演示'
```

## 带指定检查的编码任务

`--check-command`（可重复，最多 8 条）固定"必须通过什么检查"：模型只能用 `check` 工具运行启动者声明的命令，不能替换命令或超时。程序独立维护验证事实——基线快照、检查回执（最近一次尝试取代旧 PASS，任何文件变化使旧回执过时）、独立变更证据——并给出交付判定：全部条件满足退出码 `0`；模型正常 stop 但条件不满足为 `3`，不依赖模型自然语言汇报。超时/取消/输出未 EOF 永久锁存 `process_uncertain`，后续新 PASS 不能清除。检查模式要求 `--workspace` 是 Git 工作树根目录、`--num-ctx` ≥ 32768，最多 25 轮；单次请求体另受 65536 字节上限（wire 字节守卫，独立于 token 预算）。

```powershell
uv run --frozen python -m cicada --workspace .\repo --check-command 'python check.py' '修复失败的测试并通过检查'
```

## 交互终端与持久会话

`--tui` 进入同进程连续对话终端（Pi 式主屏：原位流式更新、工具卡与 `Ctrl+O` 展开、忙碌可编辑草稿、斜线命令菜单、忙碌状态行动画、协作取消）：

```powershell
uv run --frozen python -m cicada --tui --workspace .\sandbox
```

持久会话覆盖单次 CLI 与 TUI：

- `--tui` 默认自动保存；`--save-session` 供单次模式落盘。存档位于 `<workspace>/.cicada/sessions/<id>.sqlite3`。
- `--resume <id>` 恢复：完整历史恢复、旧工具不重放、追加保存。省略的 `--model`/`--ollama-url`/`--think`/`--num-ctx` 继承存档，显式参数与存档冲突点名拒绝（先于模型探测）；检查模式存档恢复需显式给出相同 `--check-command`。`--tui` 下省略 ID 启动会话选择器（Esc 取消，退出码 0）。
- `--resume-with-current-config` 显式采用当前配置恢复（差异不拒绝，记入新启动代次）。
- `--list-sessions` 只读分页列表，无需模型探测；标注占用/恢复阻断/坏档。
- 恢复的会话存在未证终结进程活动时为 view-only：新请求零执行，TUI 无新请求关闭后退出码 3。两个进程不能同时续写同一会话（Windows 文件锁独占）。

## 发送预算与上下文压缩

真实模型（Ollama）发送前做预算准入：完整输入按启发式估计（非精确 token），加上输出预留与安全余量后的总估计超过有效 `--num-ctx`（请求值）即拒绝发送并说明原因；`max_request_bytes`（检查模式固定 65536）是独立的请求体字节守卫，不充当 token 预算。持久会话可用 `/compact` 手动压缩较早请求（原始记录保留）；请求超限时先自动压缩至多一次再重估，服务端确认上下文超限且该次无可见输出时压缩后重发至多一次。非持久会话只有预算准入，无压缩。

## 常用参数

| 参数 | 说明 |
|---|---|
| `--workspace` | 工作区根目录，默认当前目录 |
| `--model` / `--ollama-url` / `--think` / `--num-ctx` | 模型与服务配置；`--num-ctx` 默认 32768（请求级，非分配保证），也是真实模型发送前预算准入的上限来源 |
| `--instructions-file` | 追加到系统提示的项目指令文件（≤16 KiB UTF-8） |
| `--script` | JSON 剧本文件；与真实模型参数互斥 |
| `--check-command` / `--check-timeout` | 检查模式；timeout 默认 120 秒，范围 (0, 300] |
| `--save-session` / `--resume` / `--resume-with-current-config` / `--list-sessions` | 持久会话，语义见上节 |
| `--tui` | 交互终端 |

退出码：`0` 正常完成（检查模式=交付条件全部满足；TUI 选择器取消亦为 0）；`1` error/aborted/length 终止或执行期保存失败；`2` 参数/文件/预检/打开错误（含存档缺失、占用、损坏、配置冲突）；`3` 检查或证据条件不满足，或恢复会话 view-only。

## 测试

```powershell
uv run --frozen python -m pytest -q -m 'not live'   # 无真实模型调用
uv run --frozen python -m pytest -q -m live          # 真实 Ollama 用例 (不可达则 skip)
uv run --frozen python -m pytest -q                  # 完整套件
```

须用 `python -m pytest` 而非直接 `pytest`：跨目录的 `tests.` 导入依赖项目根在 sys.path。测试临时目录固定为项目内 `.pytest-tmp`。

## 当前范围

工具顺序执行；普通模式最多 50 轮、检查模式最多 25 轮模型循环。预算估计为启发式（无精确 tokenizer），尚未提供插件热替换。扩展：新模型实现 `ModelPort`、新工具实现 `Tool`（[src/cicada/core/ports.py](src/cicada/core/ports.py)），经 [PluginDefinition](src/cicada/runtime/plugin.py) 声明依赖后由 [boot.py](src/cicada/boot.py) 装配。
