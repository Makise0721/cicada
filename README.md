# Cicada

Cicada 是面向 Windows 的 Python coding agent，支持 DeepSeek Flash/Pro、文件读写、PowerShell、指定检查、会话保存恢复与费用准入。当前预览版本为 **0.1.0a1**。

## 环境与安装

验证环境：Windows 11、PowerShell 7、Python 3.12（包验证为 3.12.10）；需要安装 `uv`。真实任务使用自己的 DeepSeek 凭据，离线演示无需凭据。

在空的下载目录运行以下 PowerShell 命令；文件随 [v0.1.0a1 预览发布](https://github.com/Makise0721/cicada/releases/tag/v0.1.0a1) 提供，发布说明包含变更和已知问题。

```powershell
$release = 'https://github.com/Makise0721/cicada/releases/download/v0.1.0a1'
$wheel = 'cicada-0.1.0a1-py3-none-any.whl'
foreach ($file in @($wheel, 'requirements-windows-py312.txt', 'SHA256SUMS')) {
    Invoke-WebRequest "$release/$file" -OutFile $file
}
foreach ($file in @($wheel, 'requirements-windows-py312.txt')) {
    $expected = ((Get-Content .\SHA256SUMS | Where-Object { $_.EndsWith("  $file") }) -split '\s+')[0]
    if (-not $expected -or (Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash -ne $expected) { throw "文件校验失败：$file" }
}
$wheelUri = ([uri](Resolve-Path -LiteralPath $wheel).Path).AbsoluteUri
uv tool install --python 3.12 --constraints .\requirements-windows-py312.txt "cicada[tui] @ $wheelUri"
uv tool update-shell   # 首次安装后重新打开终端
```

## 先完成一个离线小任务

在新的工作目录运行；脚本创建并读取 `hello.txt`，保存会话并导出报告，不访问模型 API。

```powershell
New-Item -ItemType Directory cicada-demo | Out-Null
Set-Location cicada-demo
@'
[{"tool_calls":[{"id":"write-1","name":"write","arguments":{"path":"hello.txt","content":"hello cicada"}}]},{"tool_calls":[{"id":"read-1","name":"read","arguments":{"path":"hello.txt"}}]},{"text":"离线演示完成","stop":true}]
'@ | Set-Content -LiteralPath demo-script.json -Encoding utf8
cicada --script .\demo-script.json --save-session --export-report demo-report.json '创建并读取 hello.txt'
Get-Content .\hello.txt
cicada --list-sessions
```

## 交互与真实任务

在目标工作区运行 `cicada` 进入 TUI；首次缺凭据时显示遮蔽输入向导。也可预先设置 `$env:DEEPSEEK_API_KEY`。用户配置和独立凭据文件位于 `~/.cicada/`，登录只保存配置，不验证账号。

```powershell
cicada
# 或运行单次真实任务；模型调用会产生费用
cicada --max-model-calls 5 '阅读这个项目，说明入口和运行方法，暂不修改文件'
```

Enter 发送，Shift+Enter 换行，Esc/Ctrl+C 中断，Ctrl+D 退出。TUI 默认保存，空闲 `/resume` 恢复、`/name 名称` 命名、`/export report.json` 导出，输入 `/` 从菜单查看并选择命令。运行和收尾期间不能提交下一问。

## 检查、费用与恢复

- 使用 `--check-command 'python check.py'` 固定验收命令，工作区须为 Git 工作树根目录；退出码 `0` 才表示指定交付条件满足，模型声称完成不构成通过。
- `--max-model-calls N` 限制进程内普通和摘要调用总数；金额上限需同时配置 `--cost-limit-cny` 与 `--cost-profile`，格式见 `cicada --help`。`/cost` 查看本地预留与结算；金额上界不是账户账单。
- 会话保存在工作区 `.cicada/sessions/`。恢复旧对话不恢复旧检查权威；取消或收尾不明可能阻止后续编码。`/rewind` 仅撤回文件工具改动，不回滚对话或 PowerShell 副作用。
- `/export` 把本次会话导出为单个 HTML 并在浏览器打开：文件名按时间与会话 ID 自动生成在工作区根目录，同名自动加序号、不覆盖。**含对话、思考与工具正文**，分享前先核对内容。需要不含正文的元数据报告时用 `--export-report`（单次模式）。

## 升级、回退与卸载

先退出 Cicada，保留当前 wheel 和配套依赖清单。在目标版本下载目录重复安装步骤，将安装命令加上 `--force --reinstall`；回退使用先前保留版本的 wheel 和配套清单。不要只替换 wheel 而沿用另一版本的依赖清单。

这是首个公开预览，尚无历史公开包可回退；已有源码安装请先保留原路径和安装方式。`uv tool uninstall cicada` 只卸载工具环境及命令，工作区会话、用户配置和费用账本保留。

问题反馈入口：[GitHub Issues](https://github.com/Makise0721/cicada/issues)。更多参数和退出码见 `cicada --help`。

## 源码开发

```powershell
git clone https://github.com/Makise0721/cicada.git
Set-Location cicada
uv sync --frozen --extra tui --python 3.12
uv run --frozen python -m cicada --help
uv run --frozen python -m pytest   # fake / MockTransport；使用项目内私有 basetemp
```

内核、插件运行时与能力插件分别位于 `src/cicada/core/`、`runtime/`、`plugins/`，应用侧持有规范历史和交付判定。当前文件修改验证范围为本地 NTFS 普通文件；其他平台与真实缓存收益未验证。
