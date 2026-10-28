# Cicada

Cicada 是跑在 Windows 终端里的 AI 编程助手。你用自然语言说要做什么，它在当前项目里读代码、改文件、执行命令、跑你指定的检查，再把结果告诉你。

它不接管你的编辑器，也不用另开界面：一个终端窗口就是全部。它每一步做了什么都会显示出来，改过的文件可以撤回，对话会保存下来下次接着聊。模型可以用 DeepSeek，也可以用任何 OpenAI Chat 兼容的服务。

当前版本 0.1.0a3，预览阶段，只支持 Windows。每个版本的变更和已知问题见 [Releases](https://github.com/Makise0721/cicada/releases)。

## 装好之后

在你的项目目录里运行：

```powershell
cicada
```

第一次启动会让你填 API Key，填一次就记住。之后直接输入想做的事，Enter 发送，Shift+Enter 换行，Ctrl+D 退出。输入 `/` 能看到全部功能：换模型、换执行档（只读 / 动手前先问你 / 全自动）、找回以前的对话。

需要先装 Python 3.12 和 uv。安装时从 [Releases](https://github.com/Makise0721/cicada/releases) 下载最新 wheel 和同一发布里的 `requirements-windows-py312.txt`，在下载目录执行：

```powershell
uv tool install --python 3.12 --constraints .\requirements-windows-py312.txt "cicada[tui] @ $((Resolve-Path .\cicada-0.1.0a3-py3-none-any.whl).Path)"
```

## 会留下什么

会话、还原点和费用账本都在 `~/.cicada/` 里，项目目录不会多出隐藏文件。想把一次对话发出去，可以导出成单个 HTML，里面包含对话和工具正文，发之前自己先看一眼。

## 真实任务跑得怎么样

2026-10-27 在真实模型下跑了一组任务：从 SWE-bench Pro 里取三个真实仓库的修复任务，每个各跑 3 次，共 9 次。配置固定不变——DeepSeek V4.1 Flash，每次最多 24 次模型调用、20 分钟，任务从零开始，跑完由官方评分脚本判分。

| 任务 | 通过 | 每次调用次数 |
| --- | --- | --- |
| Ansible：密码文件带加密标识时后续读取失败 | **3/3** | 10–12 |
| qutebrowser：版本信息和屏蔽列表下载通知不可靠 | 1/3 | 20–24 |
| Ansible：解析本机可达的 IPv4 / IPv6 路由 | **3/3** | 4–8 |

9 次运行通过 7 次，合计 120 次模型调用，实付约 0.09 美元（按套餐额度折算）。次日修正了失败断言没有正确回传的问题，用同一协议重跑第二批 9 次：ansible 3/3、ansible2 3/3、qutebrowser 0/3，实付约 0.11 美元。

这是自己跑的诚实记录，不是榜单成绩：每题只跑 3 次，样本太小，不能当成功率看；qutebrowser 两批共 6 次只过 1 次，其中 5 次都是 24 次调用额度用满仍未解出——早先写成「模型服务商一侧的传输错误」是我们自己的误标，实际是本地调用额度拒绝，已更正；第三个任务只有一项测试，反馈很粗，它通过不代表没有引入回归。

## 已知限制

只在 Windows 11 + PowerShell 7 上验证过；文件修改只覆盖本地 NTFS 普通文件。设计取舍写在 `docs/adr/`。

问题反馈：[GitHub Issues](https://github.com/Makise0721/cicada/issues)。
