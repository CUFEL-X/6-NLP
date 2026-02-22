from nlp.sentiment import ProviderAgent, ToolAgent, load_excel_as_text, batch_process

# 方法1: 批量处理（使用不同Agent）
batch_process(r"D:\4-nlp\govcn_2025.xlsx", agent_type="provider")  # 默认
# batch_process(r"D:\4-nlp\govcn_2025.xlsx", agent_type="tool")      # ToolStrategy
# batch_process(r"D:\4-nlp\govcn_2025.xlsx", agent_type="unstructured")  # 无约束

# 方法2: 处理单日数据
# agent = ProviderAgent()
# text = load_excel_as_text("path/to/data.xlsx", date_filter="2025-01-15")
# result = agent.analyse(text)

# print(f"情绪评分: {result.daily_macro_sentiment_score}")
# for industry in result.industry_signal_summaries:
#     print(f"{industry.industry_name}: {industry.daily_signal_bullet_points}")