"""
Prompt templates for the LLM judge reward function.
"""

JUDGE_PROMPT_TEMPLATE = """你是一个题目评判专家。请根据下面的【题目】和【标准答案】，判断【模型答案】是否正确。

请严格按照以下规则判断：
- 提取模型答案中的选项字母（如 (A)、A、选A 等），与标准答案比较。
- 如果模型明确选择了正确答案，判定为 **正确（yes）**。
- 如果模型选择了错误答案，或没有给出明确答案，判定为 **错误（no）**。
- 请只输出一个词：yes 或 no，不要输出其他内容。

【题目】
{question}

【标准答案】
{result}

【模型答案】
{answer}
"""


def build_judge_messages(question: str, standard_answer: str, model_answer: str) -> list[dict]:
    """
    Build the messages list for the LLM judge API call.
    """
    user_content = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        result=standard_answer,
        answer=model_answer,
    )
    return [{"role": "user", "content": user_content}]


def parse_judge_response(content: str) -> bool:
    """
    Parse the response from the LLM judge.

    Returns:
        True if response contains "yes", False otherwise.
    """
    return "yes" in content.strip().lower()
