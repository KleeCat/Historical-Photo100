# Global Codex preferences

## Language and style

- 默认使用简体中文。
- 不要输出元评论、调试尾巴或解析错误文本，例如 `Wait`、`INVALID`、`ERROR: Invalid JSON`、`no punctuation`。

## Session titles

- 会话标题只写任务本身，使用简体中文。
- 长度控制在 8-18 个汉字。
- 推荐结构：`动词 + 对象 + 目标`（示例：`修复提交信息校验问题`）。
- 禁止包含：`Wait`、`INVALID`、`ERROR`、`assistant`、`JSON`、`instructions`、`parser`。
- 不要带引号、花括号、代码片段、分支名、日期时间、序号、表情或解释性后缀。
- 若无法准确概括，使用：`任务执行与问题修复`。

## Git commit messages

- 使用 Conventional Commits：`<type>(<scope>): <summary>`。
- `type` 仅限 `feat`、`fix`、`refactor`、`docs`、`chore`、`test`、`perf`、`ci`、`revert`。
- 必须是单行纯文本，且匹配正则：`^(feat|fix|refactor|docs|chore|test|perf|ci|revert)\([^)]+\): .+$`。
- `summary` 要简洁、明确、专业，优先简体中文；若 scope 不明确，统一使用 `repo`。
- 禁止出现提示词残留或调试词：`Wait`、`INVALID`、`ERROR`、`assistant`、`JSON`、`instructions`、`parser`。
- 不要输出 JSON、Markdown、代码块、引号、emoji 或额外说明。

## Git pull request messages

- 仅输出 Markdown，严格使用以下结构：
  - `## 变更目的`
  - `## 主要改动`
  - `## 自测情况`
  - `## 风险与回滚`
- 禁止出现提示词残留或调试词：`Wait`、`INVALID`、`ERROR`、`assistant`、`JSON`、`instructions`、`parser`。
- 不得编造改动/测试/风险；未知项明确写“未运行（未请求）”或“风险较低”。

## Shell command handling on Windows

- 非微小输出的命令优先使用 `rtk` 包装。
- Git 命令优先用 `rtk git ...`。
- 测试、构建、脚本优先用对应的 `rtk` 子命令或 `rtk summary ...`。
- 大输出的 PowerShell 命令优先用 `rtk summary powershell -NoProfile -Command "<cmd>"`。
- 只关心报错时优先用 `rtk err ...`。
- 避免直接输出原始的长 PowerShell 单行命令。
- `rtk proxy` 仅作最后兜底，优先 `rtk summary` / `rtk err` / 专用子命令。
- `rtk proxy` 默认不要用于 `python` / `node` / `powershell`。
- Git 场景优先 `rtk git ...`（或 tiny 探测用原生命令），不要默认走 `rtk proxy git ...`；仅在必须保留原始透传行为（交互流、二进制输出、包装器不兼容）时再用 proxy。
- Git tiny 探测白名单（可原生命令）：`git status --short`、`git branch --show-current`、`git rev-parse --abbrev-ref HEAD`。
- `where.exe`、`Get-Command`、`Test-Path`、`git status --short` 这类探测命令保持原生命令，不用 `rtk proxy`。
- `rtk read` 仅用于中大文件；小片段优先 `rg`、`Select-String`、`Get-Content -TotalCount/-Tail`。
- 对单文件且可预期微小输出的检索（如 `rg -n <pattern> <file>`、`Select-String -Path <file> -Pattern <pattern>`）优先原生命令；递归或大范围检索再用 RTK 包装。
- 对大小不确定的文本/代码文件，在使用 `rtk read` 前先做小探针：优先 `rg -n`、`Get-Content -TotalCount 40` 或 `Get-Content -Tail 40`，除非你已明确需要整文件上下文。
- `rtk read` 仅用于明显更大的文件（约 >300 行或 >40KB），且定点提取仍不够时。
- 小型配置/文档文件（`.md`、`.toml`、`.yaml`、`.yml`、`.ini`、`.env`、小型 `.json`）默认不要用 `rtk read`，除非确实需要整文件上下文。
- 对本轮刚创建、刚编辑、刚补丁过的文件，默认不要立刻用 `rtk read` 全量回读；优先用 `git diff`、`rg` 或小片段核验。
- 同一轮里不要反复对小配置/文档文件使用 `rtk read`；已知结构后改用定点检索命令。
- `Get-Content -TotalCount N` 中，`N <= 120` 可原生命令；`N > 120` 必须走 RTK。
- `Get-Content ... | Select-Object ...` 若没有明确小上限（如 `-First <= 80`），必须走 RTK。
- 对“预计很小但不在 tiny allowlist”的命令，先跑 `rtk rewrite "<raw command>"` 再执行改写结果。
- 含管道或长脚本的 `powershell -Command` 必须改为 `rtk summary powershell ...` 或 `rtk err powershell ...`。
- `powershell -File <script.ps1>` 默认也走 RTK 包装，除非明确是微小探测且输出可控。
- `/fork` 或 `/resume` 出来的线程也必须执行同一套 RTK 规则，不允许沿用旧线程里的裸命令习惯。
- 复杂 PowerShell（嵌套引号/正则/脚本块/多管道）优先写临时 `.ps1`，再用 `rtk summary powershell ... -File` 或 `rtk err ... -File`。
- 不允许为了规避引号解析而删除管道、分号、过滤条件等，命令语义必须等价。
- `powershell -Command` 出现引号/解析失败时，唯一兜底是“临时 `.ps1` + `rtk ... -File`”，不允许改语义重试。
- 定期用 `rtk gain --history` 复盘：若某命令族高频且平均收益 <5%，下一轮规则要收紧该命令族。

## RTK 规则同步

- 当本文件或 `C:\Users\ihggk\.codex\RTK-CODEX.md` 的 RTK 规则发生更新时，必须同步覆盖：
  - `D:\HuaweiMoveData\Users\ihggk\Desktop\thesis-sr-system\AGENTS.md`
  - `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\AGENTS.md`
  - `D:\我的文档\mu_code\五子棋\AGENTS.md`
  - `D:\HuaweiMoveData\Users\ihggk\Desktop\实习工作\软著\实战演练\AGENTS.md`
- 同步脚本：`C:\Users\ihggk\.codex\sync_project_agents.py`（兼容包装：`C:\Users\ihggk\.codex\sync_thesis_rtk_agents.ps1`）
