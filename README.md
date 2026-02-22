# NLP 政策情绪分析工具

基于 LangChain v1.0 的结构化输出能力，对政府政策新闻进行每日情绪分析，生成行业信号摘要和宏观政策情绪评分。

## 功能概述

本工具读取包含政策新闻的 Excel 数据，按日期逐日分析，输出：

1. **行业信号摘要** - 每个行业的利好/利空信号提取
2. **每日宏观情绪评分** - 0-100分的政策情绪指数（50=中性，>50=宽松/利好，<50=收紧/利空）
3. **关键政策归因** - 决定当日评分的3个核心政策/事件及原文引用

## 项目结构

```
nlp/
├── .env                    # 环境变量配置（API密钥等）
├── .python-version         # Python 版本要求（3.12+）
├── pyproject.toml          # uv 项目配置（含依赖声明）
├── uv.lock                 # uv 依赖锁定文件
├── govcn_2025.xlsx         # 输入数据：政策新闻Excel
├── govcn_2025_results.json # 输出结果：分析结果JSON
├── README.md               # 本文档
├── 讲解框架.md             # 框架原理详解
└── nlp/
    └── sentiment.py        # 核心分析模块
```

## 环境配置

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

### 1. 安装 uv（如果未安装）

```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用 pip
pip install uv
```

### 2. 安装项目依赖

```bash
# 推荐方式：安装到虚拟环境并保持依赖同步
uv pip install -e .

# 添加新依赖
uv add langchain langchain-openai pandas openpyxl python-dotenv pydantic
```

### 3. 配置 `.env` 文件

在项目根目录创建 `.env` 文件：

```env
MODEL=qwen-max          # 模型名称
BASE_URL=https://xxx    # API端点
API_KEY=sk-xxx          # API密钥
```

## 数据格式要求

输入 Excel 文件需包含以下列：

| 列名 | 说明 |
|------|------|
| `industry_name` | 行业名称（如：金融、新能源、房地产） |
| `title` | 新闻标题 |
| `content` | 新闻正文（含政策细节、具体数据） |
| `date` | 发布日期 |

## 使用方法

### 方式一：命令行批量处理

直接运行脚本，处理所有日期：

```bash
# 使用 uv 运行（推荐，自动使用虚拟环境）
uv run python nlp/sentiment.py

# 使用 ToolAgent (ToolStrategy)
uv run python nlp/sentiment.py tool

# 使用 UnstructuredAgent（无结构化约束）
uv run python nlp/sentiment.py unstructured
```

输出示例：
```
共有 226 天的数据待处理
日期范围: 2025-01-02 到 2025-12-31
============================================================
使用 ProviderAgent (Provider原生)
[1/226] 正在处理 2025-01-02 ... 完成 (情绪分数: 55)
[2/226] 正在处理 2025-01-03 ... 完成 (情绪分数: 52)
...
============================================================
处理完成: 成功 226 天, 失败 0 天
结果已保存到: C:\Users\ta\Desktop\nlp\govcn_2025_results.json
```

### 方式二：代码调用

```python
from nlp.sentiment import ProviderAgent, ToolAgent, load_excel_as_text, batch_process

# 方法1: 批量处理（使用不同Agent）
batch_process("path/to/data.xlsx", agent_type="provider")  # 默认
batch_process("path/to/data.xlsx", agent_type="tool")      # ToolStrategy
batch_process("path/to/data.xlsx", agent_type="unstructured")  # 无约束

# 方法2: 处理单日数据
agent = ProviderAgent()
text = load_excel_as_text("path/to/data.xlsx", date_filter="2025-01-15")
result = agent.analyse(text)

print(f"情绪评分: {result.daily_macro_sentiment_score}")
for industry in result.industry_signal_summaries:
    print(f"{industry.industry_name}: {industry.daily_signal_bullet_points}")
```

### 方式三：指定输出路径

```python
from nlp.sentiment import batch_process

batch_process(
    excel_path="path/to/input.xlsx",
    output_path="path/to/output.json",
    agent_type="tool"  # 可选：provider, tool, unstructured
)
```

## 输出结构

结果保存为 JSON 格式：

```json
{
  "processed_at": "2025-01-22T10:30:00",
  "total_dates": 226,
  "success_count": 226,
  "failed_count": 0,
  "results": [
    {
      "date": "2025-01-02",
      "daily_macro_sentiment_score": 55,
      "industry_signal_summaries": [
        {
          "industry_name": "金融",
          "positive_signals": ["央行降准0.5个百分点"],
          "negative_signals": [],
          "daily_signal_bullet_points": [
            "央行宣布全面降准，释放流动性约1万亿元",
            "银行间市场利率下行"
          ]
        }
      ],
      "key_policy_attributions": [
        {
          "key_event": "央行宣布全面降准",
          "quote_text": "中国人民银行决定于1月5日下调金融机构存款准备金率0.5个百分点"
        }
      ]
    }
  ],
  "failed_dates": []
}
```

## 核心类说明

### 三种 Agent 策略对比

| Agent 类型 | response_format | 输出约束 | 返回类型 | 适用场景 |
|-----------|-----------------|---------|---------|---------|
| `ProviderAgent` | Pydantic 模型 | 强制 JSON Schema + 字段验证 | `ReportRiskAnalysis` 对象 | 生产环境（推荐） |
| `ToolAgent` | `ToolStrategy(Pydantic模型)` | 通过 Tool Calling 强制结构 | `ReportRiskAnalysis` 对象 | 备选方案 |
| `UnstructuredAgent` | `None` | 无约束，自由文本 | 原始响应字典 | 调试/测试 |

### `ProviderAgent`

使用 LLM Provider 原生结构化输出能力的代理类。

```python
agent = ProviderAgent(temperature=0.6)
result = agent.analyse(text)  # 返回 ReportRiskAnalysis 对象
```

### `ToolAgent`

使用 Tool Calling 策略的代理类（备选方案）。

```python
agent = ToolAgent(temperature=0.6)
result = agent.analyse(text)  # 返回 ReportRiskAnalysis 对象
```

### `UnstructuredAgent`

不限制输出格式的代理类（用于调试）。无任何 JSON Schema 约束，模型自由输出。

```python
agent = UnstructuredAgent(temperature=0.6)
result = agent.analyse(text)  # 返回原始响应字典
```

**注意**：`UnstructuredAgent` 的输出结构与结构化 Agent 不同，`batch_process` 会将其保存为 `raw_response` 字段。

## 工具函数

### `load_excel_as_text(excel_path, date_filter=None)`

从 Excel 加载数据并转换为 CSV 文本。

| 参数 | 类型 | 说明 |
|------|------|------|
| `excel_path` | str | Excel 文件路径 |
| `date_filter` | str | 可选，日期过滤器（格式：`YYYY-MM-DD`） |

### `get_all_dates(excel_path)`

获取 Excel 中所有唯一日期列表（升序排列）。

### `batch_process(excel_path, output_path=None, agent_type="provider")`

批量处理 Excel 数据，按日期逐日分析。

| 参数 | 类型 | 说明 |
|------|------|------|
| `excel_path` | str | 输入 Excel 路径 |
| `output_path` | str | 可选，输出 JSON 路径（默认为 `{excel_path}_results.json`） |
| `agent_type` | str | 可选，Agent 类型：`"provider"`（默认）、`"tool"`、`"unstructured"` |

## 注意事项

1. **上下文长度限制**：单日数据量过大（超过约50条新闻）可能导致 API 报错 `Range of input length should be [1, 129024]`
2. **API 兼容性**：部分 API 端点可能不完全支持 LangChain 的 agent 框架，导致 `tool_calls` 相关错误
3. **失败重试**：失败的日期会记录在 `failed_dates` 中，可提取后单独处理

## 情绪评分标准

| 分数区间 | 含义 | 典型事件 |
|---------|------|---------|
| 0-20 | 恐慌/极度悲观 | 破产、崩盘、战争、恶性通胀 |
| 20-40 | 收紧/利空 | 加强监管、罚款、行业整顿、加税 |
| 40-60 | 中性/观望 | 信号混合、政策延续、等待观察 |
| 60-80 | 宽松/利好 | 降息、降准、财政补贴、消费刺激 |
| 80-100 | 狂热/极度乐观 | 历史新高、超预期增长、重大突破 |
