from langchain_openai import ChatOpenAI

from scientific_research_assistant.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    LLM_MODEL,
)


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=temperature,
        max_retries=2,
        timeout=60,
    )