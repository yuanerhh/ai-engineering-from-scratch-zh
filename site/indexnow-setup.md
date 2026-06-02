# Bing IndexNow 快速收录设置指南

IndexNow 可让 Bing（及豆包等基于 Bing 的 AI 搜索）在页面更新后几分钟内完成收录。

## 步骤

1. 前往 https://www.bing.com/indexnow 获取 API Key
2. 将 Key 值（例如 `abc123def456`）保存为文件：  
   在 `site/` 目录下创建文件 `abc123def456.txt`，文件内容只有一行：`abc123def456`
3. 验证文件可访问：`https://aiengineeringfromscratch-zh.com/abc123def456.txt`
4. 提交 URL 通知（每次更新页面后执行）：

```bash
curl -X POST "https://api.indexnow.org/indexnow" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{
    "host": "aiengineeringfromscratch-zh.com",
    "key": "abc123def456",
    "keyLocation": "https://aiengineeringfromscratch-zh.com/abc123def456.txt",
    "urlList": [
      "https://aiengineeringfromscratch-zh.com/",
      "https://aiengineeringfromscratch-zh.com/catalog.html",
      "https://aiengineeringfromscratch-zh.com/glossary.html",
      "https://aiengineeringfromscratch-zh.com/prereqs.html"
    ]
  }'
```

IndexNow 同时支持 Bing、Yandex，豆包（ByteDance）的 Bing 后端会同步获取通知。
