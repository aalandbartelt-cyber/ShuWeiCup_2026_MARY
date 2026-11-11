# AI Tool Usage Report

**Competition:** 2026 ShuWei Cup (Autumn)  
**Team Number:** [Team Number]  
**Problem Chosen:** [A/B/C/D]  
**Report Date:** [YYYY-MM-DD]

> **填写说明（中文，提交前删除本框）**
> 1. 本模板对应数维杯 2026-11-11《人工智能工具使用规定》。该规定文本写明适用于 2027 春季赛及以后；2026 秋季赛是否强制以组委会最终通知为准，但建议按本模板留痕。
> 2. 使用 AI 的队伍：在论文正文结尾或参考文献后写声明，并附本报告；未使用 AI 的队伍：在论文中写“本参赛队未使用任何AI工具”，本报告可只保留声明页或按组委会要求不附。
> 3. 只填实际使用过的工具与环节；没有的表格写“None”。只用于翻译/语言润色的，可简化填写工具信息、日期和对应章节。
> 4. 关键提示词写“摘要/目的”，不要粘贴大段无关对话、个人信息、账号、密钥或未授权材料。
> 5. 所有 AI 输出必须由队员逐项核对；虚构、错误、查不到来源的内容一律删除或更正。报告中的“采纳/部分采纳/不采纳”要如实填写。
> 6. 论文正文、代码、图表、公式、结果处的 AI 标注要与本报告一致；参考文献列出工具，格式见第 6 节。
> 7. 本报告是交付附件，不是论文正文；提交前按当届系统要求命名/打包，并删除本说明框。

---

## 1. Tool Inventory

| No. | Tool | Version / Model | Developer / Company | Date(s) Used | Used in |
|---:|---|---|---|---|---|
| 1 | [Tool name] | [version/model] | [developer] | [YYYY-MM-DD] | [stages] |
| 2 |  |  |  |  |  |

## 2. Usage Details

| ID | Tool | Stage | Purpose | Output / Adopted Content | Paper or Material Location |
|---|---|---|---|---|---|
| A1 | [Tool] | [problem analysis / writing / coding / figures / translation] | [what it was asked to do] | [what was used] | [section/table/figure/code file] |
| A2 |  |  |  |  |  |

**Stage vocabulary (use consistently):** problem understanding; literature/data lookup; writing/structure; language editing/translation; coding/debugging; formula/notation check; figure/table design; data preprocessing; result explanation; other.

## 3. Prompting and Process

- **Tools used:** [list]
- **Materials provided to AI:** [problem statement, selected paper sections, table/figure drafts, code snippets, references; do not include private data]
- **Main prompting style:** [text Q&A; provide context and ask for specific checks/rewrites; no broad "write the paper" prompts]
- **Process:** The team first completed [data scope / model assumptions / algorithm design / parameter setting / program runs]. AI was then used for [language/structure/notation/code syntax/figure layout] assistance. The team checked every suggestion against [source code / result CSV / figures / tables / references] before adoption. Core model assumptions, algorithm design, derivations, result interpretation, and conclusions were completed by the team.
- **AI output overview:** [what AI mainly provided; explicitly state what it did NOT do]
- **Corresponding locations:** [sections, tables, figures, code files]

## 4. Key Interaction Records

| No. | Record ID | Prompt (summary) | AI Response (summary) | Human Check |
|---:|---|---|---|---|
| 1 | A1 | [summary of prompt] | [summary of response] | [what was checked and result] |
| 2 |  |  |  |  |

## 5. Adoption, Manual Modification, and Verification

| Record ID | AI Output (summary) | Adoption | Manual Changes | Verification Method and Result |
|---|---|---|---|---|
| A1 | [summary] | Adopted / Partially adopted / Rejected | [what the team changed] | [checked against source code/CSV/table/figure/reference; result] |
| A2 |  |  |  |  |

**Verification rule:** "Adopted" requires a named source/result checked by the team. Do not write "verified" without saying what was compared.

## 6. Declarations and Reference Entries

### 6.1 If AI tools were used (put the declaration after References in the paper)

> **AI Tool Usage Declaration:** This team used [Tool Name(s)] only as auxiliary tools for [stages]. All model assumptions, algorithm design, derivations, result interpretation, and conclusions were completed independently by the team. All AI-assisted text, code, formulas, figures, and results were checked against the team's source data, code, and outputs. AI-generated content is marked in the paper, and the tools are listed in the References. A detailed AI Tool Usage Report is provided after the References and is not counted toward the 25-page limit.

### 6.2 If no AI tools were used

> This team did not use any AI tools.

(Chinese equivalent for team records: 本参赛队未使用任何AI工具。)

### 6.3 Reference entries for AI tools

Use the official reference style and the competition's required order:

```text
[No.] Tool name, version/model, developer/company, date(s) used.
```

Example (replace with actual tools):

```text
[6] DeepSeek, DeepSeek-V4.1-Flash, DeepSeek, 2026-11-12.
[7] ChatGPT, GPT-5.6 Sol, OpenAI, 2026-11-12.
```

Do not list a tool that was not actually used. Do not invent versions, developers, or dates.

## 7. Annotation Checklist for Code, Figures, Formulas, and Results

- [ ] Every AI-assisted code file has a comment with tool, version/model, developer/company, release date, purpose, and "verified by team".
- [ ] Every AI-assisted figure/table/result has a note in the caption or adjacent text: tool/version, what it contributed, and what the team verified.
- [ ] Every AI-assisted formula/derivation is checked; if the full derivation is missing, do not rely on it as a core result.
- [ ] All facts, data, references, formulas, and code from AI are checked; fabricated/unsourced items removed.
- [ ] The paper declaration, References entries, in-text marks, code comments, figure notes, and this report use the same tool names/versions/dates.

Suggested code comment:

```text
# AI-assisted: <Tool> <version/model>, <developer>, <release date>; used for <purpose>; output verified by team against <source/result>.
```

Suggested figure/result note:

```text
Note: AI-assisted <figure/result>; tool/version; team verified against <source data/script>; see AI Tool Usage Report.
```

## 8. Team Confirmation

We confirm that:

1. The core work (model assumptions and construction, algorithm design and solution, derivations, result interpretation, conclusions) was completed by the team.
2. AI was used only as an auxiliary tool as described above.
3. All AI-assisted content was understood, checked, and rewritten where necessary; no raw AI output was copied as the paper's core content.
4. Fabricated or unverifiable facts, data, references, formulas, and code were removed or corrected.
5. This report and the paper's declaration/marks are accurate and consistent.

**Team representative:** ____________________  
**Date:** ____________________

---

## Appendix A. Optional Full Prompt Log

| No. | Date | Tool/version | Full prompt (or summary) | Output used | Verification |
|---:|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |

> Only include prompts/materials that the team is allowed to share. Remove personal information, credentials, and unrelated content.
