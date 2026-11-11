# 04 支撑材料 / AI 工具使用报告

本目录放最终交付的支撑材料与必需附件。

## AI 工具使用情况报告（必需附件，若使用 AI）

- 模板：`AI_Tool_Usage_Report_Template.docx`（英文、2 页、可编辑）
- 源/填写指南：`AI_Tool_Usage_Report_Template.md`（含中文填写说明；提交版不要带中文说明）
- 使用 AI 时：填写后导出 PDF/DOCX，按当届要求放在论文 References 后或作为支撑附件；与论文声明、References 工具条目、正文/代码/图表标注保持一致。
- 未使用 AI：论文写 `This team did not use any AI tools.`（中文记录：本参赛队未使用任何AI工具）；报告按当届要求决定是否附。
- 提交前运行：
  `python ~/.codex/skills/mary-modeling-contest-pipeline-shuwei/scripts/audit_ai_tool_report.py --report <report> --paper <paper.pdf> --expect used|none`
  无 FAIL；WARN 逐项人工处理。

## 注意

- 本目录默认被 `.gitignore` 排除大文件；模板用 `git add -f` 单独跟踪。
- 不放入个人信息、密钥、未授权数据或与论文无关的聊天记录。
