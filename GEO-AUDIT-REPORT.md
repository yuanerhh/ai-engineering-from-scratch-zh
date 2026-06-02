# GEO + SEO 审计报告
## aiengineeringfromscratch-zh.com
**审计日期：** 2026-06-02

---

## 综合评分：32 / 100

| 维度 | 得分 | 权重 | 加权得分 |
|------|------|------|---------|
| AI 可引用性与可见度 | 20/100 | 25% | 5.0 |
| 品牌权威信号 | 25/100 | 20% | 5.0 |
| 内容质量与 E-E-A-T | 55/100 | 20% | 11.0 |
| 技术基础 | 45/100 | 15% | 6.75 |
| 结构化数据 | 0/100 | 10% | 0 |
| 平台优化 | 20/100 | 10% | 2.0 |
| **综合** | **32/100** | | |

---

## 🔴 致命问题（立即修复）

### 1. 网站未被 Google 收录
**`site:aiengineeringfromscratch-zh.com` 返回零结果**

中文版网站在 Google 中完全不可见。英文版 `aiengineeringfromscratch.com` 已有索引，但中文版没有。

**根本原因可能包括：**
- 未提交 Google Search Console
- 未请求索引
- 内容主要由 JavaScript 动态渲染，Googlebot 可能未等待 JS 执行

**修复动作（需手动操作）：**
1. 前往 [Google Search Console](https://search.google.com/search-console) → 添加属性 `aiengineeringfromscratch-zh.com`
2. 通过 DNS TXT 记录或 HTML 文件验证所有权
3. 提交 sitemap：`https://aiengineeringfromscratch-zh.com/sitemap.xml`
4. 对首页使用「URL 检查」→「请求编入索引」

---

### 2. 所有页面 og:url 指向英文版（已修复 ✅）
catalog.html、glossary.html、prereqs.html 的 og:url 均指向 `https://aiengineeringfromscratch.com/`，
index.html 指向 GitHub 仓库，而非中文网站本身。这会给搜索引擎发送错误的权威性信号。

**已修复：** 所有页面 og:url 已更新为中文版域名。

---

### 3. 缺少 Canonical 标签（已修复 ✅）
所有 5 个页面均缺少 `<link rel="canonical">`，导致搜索引擎无法明确判断权威 URL。

**已修复：** 已为所有页面添加自引用 canonical 标签。

---

### 4. 缺少 hreflang 标签（已修复 ✅）
中英文两个版本之间没有任何 hreflang 关联，搜索引擎无法理解这两个站点的语言关系。

**已修复：** 已为所有页面添加 `zh-CN`、`en`、`x-default` hreflang。

---

### 5. 无 llms.txt 文件（已修复 ✅）
AI 搜索引擎（ChatGPT、Perplexity、Claude）通过 `/llms.txt` 了解网站内容结构。
该文件缺失会降低 AI 引用概率。

**已修复：** 已创建 `site/llms.txt`，包含完整课程描述、20 个阶段说明、适合人群等。

---

### 6. 缺少 JSON-LD 结构化数据（已修复 ✅）
全站无任何 Schema.org 标记，导致：
- 无法获得 Google 富摘要（Course 卡片）
- AI 搜索无法识别这是一个教育课程
- 搜索结果中缺少课程信息摘要

**已修复：** 已添加以下 Schema：
- `index.html`：`WebSite` + `Course` + `EducationalOrganization`（含 SearchAction）
- `catalog.html`：`CollectionPage`
- `glossary.html`：`DefinedTermSet`
- `prereqs.html`：`WebPage`

---

## 🟠 高优先级问题（本周处理）

### 7. 动态 JS 渲染 — 课程内容对爬虫不可见
`catalog.html` 和 `index.html` 的目录内容完全依赖 JavaScript 渲染（`data.js` + `app.js`）。
Googlebot 虽然能执行 JS，但会延迟处理，且不保证完全渲染。

**建议：** 在 `index.html` 的 `<div id="phasesGrid"></div>` 内添加静态 HTML 初始内容（20 个阶段名称和描述），JS 加载后再覆盖。这不影响现有功能，但给爬虫提供可抓取的文本。

**示例：**
```html
<noscript>
  <ul>
    <li>阶段 00 - 环境配置：Python、TypeScript、Rust、Julia 工具链配置</li>
    <li>阶段 01 - 线性代数：向量、矩阵乘法、特征值分解</li>
    <!-- ... 其余阶段 -->
  </ul>
</noscript>
```

---

### 8. catalog.html、prereqs.html 的 meta description 为英文（已修复 ✅）
catalog.html："Full catalog of 435 AI engineering lessons..." → 已改为中文
prereqs.html："Interactive prerequisite map for 435..." → 已改为中文

---

### 9. Google-Extended 被 robots.txt 屏蔽
`Google-Extended` 是 Google Gemini / AI Overviews 的训练爬虫。当前屏蔽此爬虫会减少内容被 Google AI 功能引用的概率。

**权衡：**
- 允许 Google-Extended：可能被 Google AI Overviews 引用，增加曝光
- 屏蔽 Google-Extended（当前状态）：保护内容不被用于 Google 模型训练

**建议：** 若希望出现在 Google AI 概述中，可考虑允许 Google-Extended：
```
User-agent: Google-Extended
Allow: /
```

---

## 🟡 中优先级（本月处理）

### 10. 中文平台品牌提及为零
搜索结果未发现任何中文平台（知乎、CSDN、V2EX、少数派）专门提及 `aiengineeringfromscratch-zh.com`。

**建议：**
- 在知乎发布文章：「我翻译了一套 435 节课的 AI 工程课程（中文版）」
- 在 CSDN/掘金 发布项目介绍文章
- 在 V2EX、Reddit r/MachineLearning 等社区分享
- 在中文 AI/技术社区 Discord/微信群分享

**目标：** AI 搜索引擎（Perplexity、Claude、ChatGPT）通过抓取这些平台的讨论来发现和引用本站。

---

### 11. sitemap 缺少 `lesson.html` 的具体课程 URL
当前 sitemap 只有 5 个静态页面 URL。435 节课没有独立 URL（通过 `?path=` 查询参数加载），
导致每节课内容无法被单独索引。

**建议：** 为每节课生成独立的可索引 URL，或在 sitemap 中添加关键课程的直接链接。

---

### 12. 缺少 FAQ Schema
词汇表页面有大量问答结构内容，添加 FAQ Schema 可以获得 Google 富摘要展示。

---

## ✅ 已修复清单

| 问题 | 状态 | 文件 |
|------|------|------|
| og:url 指向英文站 | ✅ 已修复 | index.html, catalog.html, glossary.html, prereqs.html |
| 缺少 Canonical 标签 | ✅ 已修复 | 所有 5 个页面 |
| 缺少 hreflang 标签 | ✅ 已修复 | 所有 5 个页面 |
| 无 llms.txt | ✅ 已创建 | site/llms.txt |
| 无 JSON-LD Schema | ✅ 已添加 | index.html, catalog.html, glossary.html, prereqs.html |
| catalog.html meta description 为英文 | ✅ 已修复 | catalog.html |
| prereqs.html meta description/og 为英文 | ✅ 已修复 | prereqs.html |
| 缺少 robots meta tag | ✅ 已添加 | 所有 5 个页面 |

---

## 行动优先级排序

### 立即执行（今天，手动操作）
1. **提交 Google Search Console** — 验证域名所有权，提交 sitemap，请求首页索引
2. **提交 Bing Webmaster Tools** — https://www.bing.com/webmasters
3. **部署以上代码修改** — push 到 GitHub，触发 Vercel 部署

### 本周
4. **添加静态 noscript 内容** — 让 20 个阶段和课程列表对爬虫可见
5. **在知乎/掘金发布介绍文章** — 建立中文平台的品牌提及和外链

### 本月
6. **考虑是否允许 Google-Extended** — 权衡训练数据保护 vs AI 搜索曝光
7. **建立课程独立 URL** — 为每节课创建可索引的静态页面或动态路由
8. **添加 FAQ Schema** 到词汇表页面

---

## 预期效果

完成上述修复后：
- **2-4 周内**：Google 完成首次收录，开始出现在搜索结果
- **1-2 个月**：开始在「AI工程课程中文」「从零学AI」等关键词出现排名
- **2-3 个月**：随着中文平台品牌提及积累，在 Perplexity/ChatGPT 搜索中被引用

---

*报告由 GEO Skill 生成 · 2026-06-02*
