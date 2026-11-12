from pydantic import BaseModel, Field
from typing import Literal

from scientific_research_assistant.generation.llm import get_llm


Domain = Literal[
    "llm",
    "machine_learning",
    "computer_vision",
]


class RoutingDecision(BaseModel):
    domains: list[Domain] = Field(
        description=(
            "All research domains relevant to the user's question."
        )
    )

    reasoning: str = Field(
        description=(
            "A brief explanation of why these domains were selected."
        )
    )


class DomainRouter:

    def __init__(self):
        self.llm = get_llm(temperature=0.0)

        self.router = self.llm.with_structured_output(
            RoutingDecision
        )

    def route(self, question: str) -> RoutingDecision:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        prompt = f"""
You are a domain classification system for a scientific
research assistant.

Classify the user's question into one or more of these domains:

1. llm:
   Large Language Models, Transformers, NLP, attention,
   prompting, fine-tuning, RAG, BERT, and related topics.

2. machine_learning:
   General machine learning, optimization, neural network
   training, regularization, model interpretability,
   classification, and related topics.

3. computer_vision:
   Image processing, image classification, object detection,
   image segmentation, CNNs, ResNet, ViT, GANs, and related topics.

Rules:
- Select every domain that is genuinely relevant.
- A question can belong to multiple domains.
- Do not select a domain merely because it is indirectly related.
- If the question is unrelated to all three domains,
  return an empty domains list.
- Return a short explanation of your decision.

User question:
{question}
"""

        return self.router.invoke(prompt)