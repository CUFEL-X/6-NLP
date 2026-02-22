"""
Sentiment Analysis Module with LangChain v1.0 Structured Output

提供三种不同的response_format策略实现结构化输出:
1. ProviderAgent - 使用Provider原生结构化输出能力 (create_agent + Pydantic模型)
2. ToolAgent - 使用Tool Calling策略 (create_agent + ToolStrategy)
3. UnstructuredAgent - 不限制输出方式 (create_agent + None)
"""

import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
import pandas as pd

load_dotenv()

# --- Excel数据路径 ---
EXCEL_PATH = r"D:\4-nlp\govcn_2025.xlsx"


# --- Pydantic 结构化输出模型 ---
class IndustrySignalSummary(BaseModel):
    """Daily core signal summary for a single industry"""
    industry_name: str = Field(description="Industry name (e.g., 'New Energy', 'Finance', 'Real Estate', etc.)")
    positive_signals: List[str] = Field(
        description="List of substantive bullish signals extracted from news related to the industry (e.g., subsidies, tax cuts, growth data, etc.), empty list if none",
        default=[]
    )
    negative_signals: List[str] = Field(
        description="List of substantive bearish signals extracted from news related to the industry (e.g., penalties, restrictions, declining data, etc.), empty list if none",
        default=[]
    )
    daily_signal_bullet_points: List[str] = Field(
        description="Daily signal summary for the industry, a bulleted list of 3-5 key points containing only substantive information (excluding meaningless jargon)"
    )

class PolicyAttribution(BaseModel):
    """Core policies/events determining the overall sentiment score and their original text citations"""
    key_event: str = Field(description="Brief description of one of the 3-5 tone-setting core policies/events")
    quote_text: str = Field(description="Citation of specific wording from the original text to support the event (e.g., 'issued a notice on XX', 'granted subsidies of XX billion yuan')")

class ReportRiskAnalysis(BaseModel):
    """Complete output structure for report risk analysis (adapted for narrative economics sentiment analysis)"""
    # Task 1: Signal summary for all industries
    industry_signal_summaries: List[IndustrySignalSummary] = Field(
        description="List of signal summaries for all unique industries, with each element corresponding to one industry"
    )
    # Task 2: Daily macro policy sentiment scoring & attribution
    daily_macro_sentiment_score: int = Field(
        description="Daily macro policy sentiment score on a scale of 0 to 100 (50 = neutral, >50 = positive/bullish, <50 = negative/bearish)",
        ge=0, le=100
    )
    key_policy_attributions: List[PolicyAttribution] = Field(
        description="The 3 most critical policies/events determining the overall score and their original text citations (fixed list length of 3)",
        min_length=3, max_length=3
    )

# --- Prompt 模板 ---
SYSTEM_PROMPT = (
    "You are an expert Quantitative Financial Analyst specializing in 'Narrative Economics'. "
    "Your task is to process a daily feed of raw news, filter out noise, and construct a sentiment index. "
)

USER_PROMPT_TEMPLATE = """
### Instructions:
I will provide a dataset containing daily financial news and policy texts (in CSV format).
Column Definitions:
- `industry_name`: The industry sector.
- `title`: The news headline.
- `content`: The body text (containing detailed policy wording, specific data, or official announcements).
- `date`: The publication date.
### Task 1: Industry-Specific Signal Summary
Iterate through every unique `industry_name` in the data. For each industry:
1. **Extract Core Signals**: Read all `content` associated with that industry. Identify substantive **Positive/Bullish signals** (e.g., subsidies, tax cuts, supportive documents, growth data) or **Negative/Bearish signals** (e.g., penalties, restrictions, strict regulations, declining data). *Ignore generic slogans or fluff with no substantive information.*
2. **Generate Brief**: Create a "Daily Signal Summary" containing 3-5 key bullet points for that industry.
### Task 2: Daily Overall Policy Sentiment Scoring & Attribution
Comprehensively analyze all texts in the dataset to provide a "Daily Macro Policy Sentiment Score".
1. **Scoring Standard**: Use a scale of 0 to 100.
   - **50**: Neutral / No major structural changes.
   - **>50**: Positive / Loose / Expansionary (e.g., interest rate cuts, RRR cuts, fiscal subsidies, consumption stimulus). Higher scores indicate greater intensity.
   - **<50**: Negative / Tightening / Restrictive (e.g., increased regulation, fines, industry rectifications, tax increases). Lower scores indicate greater intensity.
2. **Provide Score**: Give a specific integer score.
3. **Reasoning**:
   - List the 3 most critical "Tone-Setting" policies or events that determined this score.
   - **Quote specific wording** from the source text (e.g., "issued notice on...", "granting subsidies of...", "strengthening supervision") to validate your score.

### Input Text:
{report_content}

"""


# --- Provider策略代理 ---
class ProviderAgent:
    """
    Provider策略代理 - 使用Provider原生结构化输出能力

    response_format传递方式：直接传递Pydantic模型
    create_agent(model, response_format=Pydantic模型)
    """

    def __init__(self, temperature: float = 0.6):
        self.model = ChatOpenAI(
            model=os.getenv("MODEL"),
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
            temperature=temperature,
        )
        self.agent = create_agent(
            model=self.model,
            response_format=ReportRiskAnalysis,  # 直接传Pydantic模型
        )

    def analyse(self, text: str) -> ReportRiskAnalysis:
        user_message = USER_PROMPT_TEMPLATE.format(report_content=text)
        result = self.agent.invoke({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        })
        return result["structured_response"]


# --- Tool策略代理 ---
class ToolAgent:
    """
    Tool Calling策略代理 - 使用Tool Calling方式提取结构化数据

    response_format传递方式：传递ToolStrategy包装的schema
    create_agent(model, response_format=ToolStrategy(schema))
    """

    def __init__(self, temperature: float = 0.6):
        self.model = ChatOpenAI(
            model=os.getenv("MODEL"),
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
            temperature=temperature,
        )
        self.agent = create_agent(
            model=self.model,
            response_format=ToolStrategy(ReportRiskAnalysis),  # 传ToolStrategy
        )

    def analyse(self, text: str) -> ReportRiskAnalysis:
        user_message = USER_PROMPT_TEMPLATE.format(report_content=text)
        result = self.agent.invoke({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        })
        return result["structured_response"]


# --- 非结构化代理 ---
class UnstructuredAgent:
    """
    非结构化代理 - 不限制输出方式

    response_format传递方式：传递None
    create_agent(model, response_format=None)
    """

    def __init__(self, temperature: float = 0.6):
        self.model = ChatOpenAI(
            model=os.getenv("MODEL"),
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
            temperature=temperature,
        )
        self.agent = create_agent(
            model=self.model,
            response_format=None,  # 传None
        )

    def analyse(self, text: str):
        user_message = USER_PROMPT_TEMPLATE.format(report_content=text)
        result = self.agent.invoke({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        })
        return result


# --- 数据加载函数 ---
def load_excel_as_text(excel_path: str, date_filter: str = None) -> str:
    """
    从Excel文件加载数据并转换为CSV格式的文本

    Args:
        excel_path: Excel文件路径
        date_filter: 可选的日期过滤器，格式如 "2025-01-15"

    Returns:
        CSV格式的文本字符串
    """
    df = pd.read_excel(excel_path)

    # 选择需要的列
    columns_to_use = ['industry_name', 'title', 'content', 'date']
    df = df[columns_to_use]

    # 如果指定了日期过滤
    if date_filter:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df[df['date'] == date_filter]

    # 转换为CSV格式文本
    return df.to_csv(index=False)


# --- 获取所有日期 ---
def get_all_dates(excel_path: str) -> List[str]:
    """获取Excel中所有唯一日期，按升序排列"""
    df = pd.read_excel(excel_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    dates = sorted(df['date'].dropna().unique().tolist())
    return dates


# --- 批量处理 ---
def batch_process(excel_path: str, output_path: str = None, agent_type: str = "provider"):
    """
    按日期逐日处理Excel数据

    Args:
        excel_path: 输入Excel路径
        output_path: 输出结果JSON路径（可选）
        agent_type: Agent类型，可选 "provider"（默认）、"tool"、"unstructured"
    """
    import json
    from datetime import datetime

    # 获取所有日期
    dates = get_all_dates(excel_path)
    print(f"共有 {len(dates)} 天的数据待处理")
    print(f"日期范围: {dates[0]} 到 {dates[-1]}")
    print("=" * 60)

    # 根据类型初始化Agent
    if agent_type == "tool":
        print("使用 ToolAgent (ToolStrategy)")
        agent = ToolAgent()
    elif agent_type == "unstructured":
        print("使用 UnstructuredAgent (无结构化)")
        agent = UnstructuredAgent()
    else:
        print("使用 ProviderAgent (Provider原生)")
        agent = ProviderAgent()

    # 存储结果
    results = []
    failed_dates = []

    for i, date in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] 正在处理 {date} ...", end=" ")

        try:
            # 加载当天数据
            text = load_excel_as_text(excel_path, date_filter=date)
            # 分析
            result = agent.analyse(text)

            # 根据agent类型处理结果
            if agent_type == "unstructured":
                # 非结构化输出，直接保存原始响应
                results.append({
                    "date": date,
                    "raw_response": str(result),
                })
                print("完成 (非结构化输出)")
            else:
                # 结构化输出
                results.append({
                    "date": date,
                    "daily_macro_sentiment_score": result.daily_macro_sentiment_score,
                    "industry_signal_summaries": [s.model_dump() for s in result.industry_signal_summaries],
                    "key_policy_attributions": [p.model_dump() for p in result.key_policy_attributions],
                })
                print(f"完成 (情绪分数: {result.daily_macro_sentiment_score})")

        except Exception as e:
            failed_dates.append({"date": date, "error": str(e)})
            print(f"失败: {e}")

    # 输出统计
    print("=" * 60)
    print(f"处理完成: 成功 {len(results)} 天, 失败 {len(failed_dates)} 天")

    # 保存结果到JSON
    if output_path is None:
        output_path = excel_path.replace('.xlsx', '_results.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "processed_at": datetime.now().isoformat(),
            "total_dates": len(dates),
            "success_count": len(results),
            "failed_count": len(failed_dates),
            "results": results,
            "failed_dates": failed_dates,
        }, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {output_path}")
    return results


# --- 使用示例 ---
if __name__ == "__main__":
    import sys

    # 从命令行参数获取agent类型，默认为provider
    agent_type = sys.argv[1] if len(sys.argv) > 1 else "provider"

    # 批量处理所有日期
    batch_process(EXCEL_PATH, agent_type=agent_type)